"""Turn-boundary merge and eviction caches used by ``eval_ppl_final.py``."""

import torch

from kv_cache_nova_similarity_eviction import ElasticCache, H2OCache, LocalCache
from tgv_kv_legacy_nova import (
    TGVKVCache,
    enable_legacy_tgv_attention,
    set_legacy_logical_position_limit,
)


class TurnConditionedLayerwiseKVCache:
    """Boundary-only KV compression with selectable layer-budget evidence.

    Allocation signal for layer l at the end of turn t:

        grounding_l + cross_turn_weight * history_l

    where grounding is attention from current-turn text queries to image keys,
    and history is attention from those queries to keys tagged with an earlier
    turn id. ``allocation_signal="uniform"`` gives every layer equal weight.
    ``allocation_signal="sensitivity"`` selects a frozen joint, image, text,
    or split profile. Split mode uses image sensitivity at the first boundary,
    then averages separately powered image and prior-text sensitivities later.
    Token retention itself remains unchanged in every allocation ablation.
    """

    IMAGE = -1

    def __init__(
        self,
        start_size=4,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        cross_turn_weight=1.0,
        image_start=None,
        image_token_len=0,
        allocation_signal="turn",
        sensitivity_mode="split",
        sensitivity_joint=None,
        sensitivity_image=None,
        sensitivity_text=None,
        sensitivity_power=1.0,
    ):
        self.start_size = int(start_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.ratio = float(ratio)
        self.layer_num = int(layer_num)
        self.attn_weight = float(attn_weight)
        self.conn_weight = float(conn_weight)
        self.cross_turn_weight = float(cross_turn_weight)
        self.image_start = image_start
        self.image_token_len = int(image_token_len)
        self.allocation_signal = str(allocation_signal)
        self.sensitivity_mode = str(sensitivity_mode)
        self.sensitivity_power = float(sensitivity_power)

        if self.allocation_signal not in {"uniform", "turn", "sensitivity"}:
            raise ValueError(
                "allocation_signal must be 'uniform', 'turn', or 'sensitivity'"
            )
        if self.sensitivity_mode not in {"joint", "image", "text", "split"}:
            raise ValueError(
                "sensitivity_mode must be 'joint', 'image', 'text', or 'split'"
            )
        if self.sensitivity_power <= 0:
            raise ValueError("sensitivity_power must be positive")

        self.sensitivity_joint = self._validate_sensitivity_profile(
            sensitivity_joint, "joint"
        )
        self.sensitivity_image = self._validate_sensitivity_profile(
            sensitivity_image, "image"
        )
        self.sensitivity_text = self._validate_sensitivity_profile(
            sensitivity_text, "text"
        )
        if self.allocation_signal == "sensitivity":
            required = {
                "joint": (("joint", self.sensitivity_joint),),
                "image": (("image", self.sensitivity_image),),
                "text": (("text", self.sensitivity_text),),
                "split": (
                    ("image", self.sensitivity_image),
                    ("text", self.sensitivity_text),
                ),
            }[self.sensitivity_mode]
            missing = [name for name, values in required if values is None]
            if missing:
                raise ValueError(
                    "Sensitivity mode '{}' requires profile field(s): {}".format(
                        self.sensitivity_mode, ", ".join(missing)
                    )
                )
        self.protect_size = 1
        self.reset()

    def _validate_sensitivity_profile(self, values, name):
        """Validate one frozen scalar-per-layer calibration profile."""
        if values is None:
            return None
        tensor = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
        if tensor.numel() != self.layer_num:
            raise ValueError(
                f"{name} sensitivity has {tensor.numel()} values; "
                f"expected {self.layer_num}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} sensitivity contains NaN or infinity")
        if (tensor < 0).any():
            raise ValueError(f"{name} sensitivity must be non-negative")
        return tuple(float(x) for x in tensor.tolist())

    def reset(self):
        self.score_sum = None
        self.conn_score_sum = None
        self.provenance = None
        self.current_turn = 0
        self.latest_logical_len = None
        self.grounding_sum = None
        self.history_sum = None
        self.query_count = None
        self.last_layer_scores = None
        self.last_layer_budgets = None
        self.last_finalize_stats = None

    def begin_turn(self, turn_id):
        """Mark the provenance id used for text appended in this turn."""
        self.current_turn = int(turn_id)
        if self.grounding_sum is not None:
            for tensor in (self.grounding_sum, self.history_sum, self.query_count):
                tensor.zero_()

    def _initial_provenance(self, length, device):
        p = torch.full((length,), self.current_turn, dtype=torch.long, device=device)
        if self.image_start is not None and self.image_token_len > 0:
            lo = max(0, int(self.image_start))
            hi = min(length, lo + self.image_token_len)
            p[lo:hi] = self.IMAGE
        return p

    def _initialize(self, past_key_values, device):
        if len(past_key_values) != self.layer_num:
            raise RuntimeError(f"Expected {self.layer_num} layers, got {len(past_key_values)}")
        self.score_sum, self.conn_score_sum, self.provenance = [], [], []
        for k, _ in past_key_values:
            length = k.size(self.k_seq_dim)
            self.score_sum.append(torch.zeros(length, dtype=torch.float32, device=device))
            self.conn_score_sum.append(torch.zeros(length, dtype=torch.float32, device=device))
            self.provenance.append(self._initial_provenance(length, device))
        self.grounding_sum = torch.zeros(self.layer_num, dtype=torch.float32, device=device)
        self.history_sum = torch.zeros_like(self.grounding_sum)
        self.query_count = torch.zeros_like(self.grounding_sum)

    def _grow_layer(self, layer_idx, length):
        old = self.score_sum[layer_idx].numel()
        if length < old:
            raise RuntimeError(f"Layer {layer_idx}: metadata length {old} exceeds KV length {length}")
        add = length - old
        if add == 0:
            return
        device = self.score_sum[layer_idx].device
        zeros = torch.zeros(add, dtype=torch.float32, device=device)
        turns = torch.full((add,), self.current_turn, dtype=torch.long, device=device)
        self.score_sum[layer_idx] = torch.cat((self.score_sum[layer_idx], zeros))
        self.conn_score_sum[layer_idx] = torch.cat((self.conn_score_sum[layer_idx], zeros.clone()))
        self.provenance[layer_idx] = torch.cat((self.provenance[layer_idx], turns))

    @staticmethod
    def _layer_attention(attention):
        if attention.dim() != 4 or attention.shape[0] != 1:
            raise RuntimeError(f"Expected [1, heads, queries, keys], got {tuple(attention.shape)}")
        return attention[0].mean(dim=0)

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("num_of_token is required for ratio-based pruning")
        if attentions is None or len(attentions) != self.layer_num:
            raise ValueError("One attention tensor per transformer layer is required")

        device = attentions[0].device
        if self.score_sum is None:
            self._initialize(past_key_values, device)

        for layer_idx, ((k, _), attention) in enumerate(zip(past_key_values, attentions)):
            length = k.size(self.k_seq_dim)
            self._grow_layer(layer_idx, length)
            attn = self._layer_attention(attention)
            if attn.shape[-1] != length:
                raise RuntimeError(
                    f"Layer {layer_idx}: attention key length {attn.shape[-1]} != KV length {length}"
                )
            query_len = attn.shape[0]
            query_start = length - query_len
            provenance = self.provenance[layer_idx]
            query_provenance = provenance[query_start:]
            current_text = query_provenance == self.current_turn

            self.score_sum[layer_idx].add_(attn.sum(dim=0).float())
            image_keys = provenance == self.IMAGE
            if image_keys.any():
                grounded = attn[:, image_keys].sum(dim=1).float()
                self.conn_score_sum[layer_idx][query_start:].add_(grounded * current_text)
                self.grounding_sum[layer_idx].add_(grounded[current_text].sum())

            prior_keys = (provenance >= 0) & (provenance < self.current_turn)
            if prior_keys.any():
                history = attn[:, prior_keys].sum(dim=1).float()
                self.history_sum[layer_idx].add_(history[current_text].sum())
            self.query_count[layer_idx].add_(current_text.sum().float())

        self.latest_logical_len = int(num_of_token)
        return past_key_values

    @staticmethod
    def _normalize(values):
        if values.numel() == 0:
            return values
        maximum = values.max()
        return torch.zeros_like(values) if maximum <= 0 else values / (maximum + 1e-6)

    def _retention_scores(self, layer_idx, start, end):
        raw = self.score_sum[layer_idx][start:end]
        conn = self.conn_score_sum[layer_idx][start:end]
        text = self.provenance[layer_idx][start:end] != self.IMAGE
        result = self.attn_weight * self._normalize(raw)
        if text.any():
            normalized_conn = torch.zeros_like(conn)
            normalized_conn[text] = self._normalize(conn[text])
            result[text] += self.conn_weight * normalized_conn[text]
        return result

    def _allocation_scores(self):
        if self.allocation_signal == "uniform":
            return torch.ones_like(self.grounding_sum)

        if self.allocation_signal == "sensitivity":
            device = self.grounding_sum.device

            def powered(values):
                return torch.tensor(
                    values, dtype=torch.float32, device=device
                ).clamp_min(0.0).pow(self.sensitivity_power)

            if self.sensitivity_mode == "joint":
                weights = powered(self.sensitivity_joint)
            elif self.sensitivity_mode == "image":
                weights = powered(self.sensitivity_image)
            elif self.sensitivity_mode == "text":
                weights = powered(self.sensitivity_text)
            else:
                image_weights = powered(self.sensitivity_image)
                if self.current_turn == 0:
                    weights = image_weights
                else:
                    text_weights = powered(self.sensitivity_text)
                    weights = 0.5 * (image_weights + text_weights)

            # A completely flat-zero profile contains no allocation evidence.
            return torch.ones_like(weights) if weights.sum() <= 0 else weights

        denom = self.query_count.clamp_min(1.0)
        grounding = self.grounding_sum / denom
        history = self.history_sum / denom
        signal = grounding + self.cross_turn_weight * history
        # A tiny uniform prior gives a defined, conservative allocation when a
        # turn has no usable attention mass (especially turn 1 history).
        return signal.clamp_min(0.0) + 1e-6

    @staticmethod
    def _bounded_largest_remainder(total, weights, lower, upper):
        """Integer allocation summing to total while respecting layer bounds."""
        lower = lower.to(torch.long)
        upper = upper.to(torch.long)
        total = max(int(lower.sum()), min(int(total), int(upper.sum())))
        allocation = lower.clone()
        remaining = total - int(allocation.sum())
        capacity = upper - allocation
        weights = weights.double().clamp_min(0)
        while remaining > 0 and capacity.sum() > 0:
            active = capacity > 0
            active_weights = weights * active
            if active_weights.sum() <= 0:
                active_weights = active.double()
            quota = remaining * active_weights / active_weights.sum()
            add = torch.minimum(torch.floor(quota).long(), capacity)
            if add.sum() == 0:
                order = torch.argsort(quota - torch.floor(quota), descending=True)
                for idx in order.tolist():
                    if capacity[idx] > 0 and remaining > 0:
                        add[idx] += 1
                        remaining -= 1
                allocation += add
                capacity -= add
                continue
            allocation += add
            capacity -= add
            remaining -= int(add.sum())
        return allocation

    def _select(self, layer_idx, budget):
        length = self.score_sum[layer_idx].numel()
        start = min(self.start_size, length)
        end = max(start, length - self.protect_size)
        prefix = torch.arange(start, device=self.score_sum[layer_idx].device)
        suffix = torch.arange(end, length, device=prefix.device)
        candidate_count = end - start
        keep_candidates = max(0, min(candidate_count, int(budget) - prefix.numel() - suffix.numel()))
        scores = self._retention_scores(layer_idx, start, end)
        if keep_candidates == candidate_count:
            selected = torch.arange(start, end, device=prefix.device)
        elif keep_candidates == 0:
            selected = prefix[:0]
        else:
            selected = torch.topk(scores, keep_candidates, sorted=False).indices + start

        retained = torch.sort(torch.cat((prefix, selected, suffix))).values
        all_candidates = torch.arange(start, end, device=prefix.device)
        keep_mask = torch.zeros(candidate_count, dtype=torch.bool, device=prefix.device)
        if selected.numel():
            keep_mask[selected - start] = True
        thrown = all_candidates[~keep_mask]

        # Preserve at least one image slot when image candidates exist.
        provenance = self.provenance[layer_idx]
        if (provenance[start:end] == self.IMAGE).any() and not (provenance[retained] == self.IMAGE).any():
            image_thrown = thrown[provenance[thrown] == self.IMAGE]
            text_selected = selected[provenance[selected] != self.IMAGE]
            if image_thrown.numel() and text_selected.numel():
                rescue = image_thrown[torch.argmax(scores[image_thrown - start])]
                evict = text_selected[torch.argmin(scores[text_selected - start])]
                retained = torch.sort(torch.cat((retained[retained != evict], rescue.view(1)))).values
                thrown = torch.cat((thrown[thrown != rescue], evict.view(1)))
        return retained, thrown

    def _similarity_targets(self, k, layer_idx, thrown, retained):
        if thrown.numel() == 0:
            return thrown
        compact = k.float().movedim(self.k_seq_dim, 0).reshape(k.size(self.k_seq_dim), -1)
        compact = torch.nn.functional.normalize(compact, dim=1)
        provenance = self.provenance[layer_idx]
        targets = []
        for idx in thrown:
            same = retained[(provenance[retained] == provenance[idx])]
            if same.numel() == 0:
                # Do not merge across provenance groups. If a matching
                # destination does not exist, evict this dropped entry.
                targets.append(torch.tensor(-1, device=retained.device))
            else:
                targets.append(same[torch.argmax(compact[same] @ compact[idx])])
        return torch.stack(targets)

    def _merge_gather(self, tensor, seq_dim, thrown, retained, targets):
        if thrown.numel() == 0:
            return tensor.index_select(seq_dim, retained.to(tensor.device))
        valid = targets >= 0
        if not valid.any():
            return tensor.index_select(seq_dim, retained.to(tensor.device))
        forgotten = tensor.index_select(seq_dim, thrown[valid].to(tensor.device))
        shape = [1] * tensor.dim()
        shape[seq_dim] = -1
        shape[-1] = 1
        index = targets[valid].to(tensor.device).view(*shape).expand_as(forgotten)
        merged = tensor.scatter_reduce(seq_dim, index, forgotten, reduce="mean")
        return merged.index_select(seq_dim, retained.to(tensor.device))

    def _turn_budgets(self):
        scores = self._allocation_scores()
        lengths = torch.tensor(
            [x.numel() for x in self.score_sum], device=scores.device
        )
        uniform_target = max(
            1, int(self.latest_logical_len * (1.0 - self.ratio))
        )
        lower = torch.tensor(
            [
                min(int(length), self.start_size + self.protect_size)
                for length in lengths.tolist()
            ],
            device=scores.device,
        )
        budgets = self._bounded_largest_remainder(
            uniform_target * self.layer_num,
            scores,
            lower,
            lengths,
        )
        return scores, budgets, uniform_target

    def _finish_finalize(self, scores, budgets, uniform_target, policy):
        self.last_layer_scores = scores.detach().cpu().tolist()
        self.last_layer_budgets = budgets.detach().cpu().tolist()
        self.last_finalize_stats = {
            "turn": self.current_turn + 1,
            "allocation_signal": self.allocation_signal,
            "sensitivity_mode": (
                self.sensitivity_mode
                if self.allocation_signal == "sensitivity"
                else None
            ),
            "sensitivity_power": (
                self.sensitivity_power
                if self.allocation_signal == "sensitivity"
                else None
            ),
            "global_budget": int(sum(self.last_layer_budgets)),
            "uniform_target_per_layer": uniform_target,
            "layer_scores": self.last_layer_scores,
            "layer_budgets": self.last_layer_budgets,
            "dropped_token_policy": policy,
        }

    def finalize_turn(self, past_key_values):
        if past_key_values is None or self.score_sum is None:
            return past_key_values
        scores, budgets, uniform_target = self._turn_budgets()

        new_past, new_score, new_conn, new_provenance = [], [], [], []
        for layer_idx, (k, v) in enumerate(past_key_values):
            retained, thrown = self._select(layer_idx, int(budgets[layer_idx]))
            targets = self._similarity_targets(k, layer_idx, thrown, retained)
            k_new = self._merge_gather(k, self.k_seq_dim, thrown, retained, targets)
            v_new = self._merge_gather(v, self.v_seq_dim, thrown, retained, targets)

            layer_score = self.score_sum[layer_idx].clone()
            layer_conn = self.conn_score_sum[layer_idx].clone()
            valid = targets >= 0
            if valid.any():
                layer_score.scatter_add_(0, targets[valid], layer_score[thrown[valid]])
                layer_conn.scatter_add_(0, targets[valid], layer_conn[thrown[valid]])
            new_past.append([k_new, v_new])
            new_score.append(layer_score[retained])
            new_conn.append(layer_conn[retained])
            new_provenance.append(self.provenance[layer_idx][retained])

        self.score_sum, self.conn_score_sum = new_score, new_conn
        self.provenance = new_provenance
        self._finish_finalize(scores, budgets, uniform_target, "merge")
        return new_past


class TurnConditionedLayerwiseEvictionKVCache(TurnConditionedLayerwiseKVCache):
    """Pure-eviction ablation of the turn-conditioned layer-wise cache.

    Layer scoring, global-to-layer budget allocation, token importance,
    protected positions, and retained-token selection are inherited unchanged.
    Only the final treatment of dropped entries differs: their K/V tensors and
    score state are discarded instead of being merged into retained entries.
    """

    def finalize_turn(self, past_key_values):
        if past_key_values is None or self.score_sum is None:
            return past_key_values

        scores, budgets, uniform_target = self._turn_budgets()

        new_past, new_score, new_conn, new_provenance = [], [], [], []
        for layer_idx, (k, v) in enumerate(past_key_values):
            retained, _ = self._select(layer_idx, int(budgets[layer_idx]))

            # Pure eviction: gather the selected entries without modifying K/V.
            k_new = k.index_select(self.k_seq_dim, retained.to(k.device))
            v_new = v.index_select(self.v_seq_dim, retained.to(v.device))
            new_past.append([k_new, v_new])

            # Dropped score mass is discarded rather than inherited by a survivor.
            new_score.append(self.score_sum[layer_idx][retained])
            new_conn.append(self.conn_score_sum[layer_idx][retained])
            new_provenance.append(self.provenance[layer_idx][retained])

        self.score_sum, self.conn_score_sum = new_score, new_conn
        self.provenance = new_provenance
        self._finish_finalize(scores, budgets, uniform_target, "evict")
        return new_past
