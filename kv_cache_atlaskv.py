"""AtlasKV: question-independent spatial visual witnesses + turn-aware text memory.

Research prototype for the MultiTurn LLaVA continuous-cache evaluators.

Rather than rating images by attention from the present question, keep a spatial
atlas of visual tokens. Each grid cell gets a representative whose value vector
has a large *local innovation* (difference from its spatial neighbours). Any
remaining slots are awarded to the most innovative unselected patches.

Text entries share the remaining budget. The recent suffix is protected and
older dialogue gets coverage across completed turns, then high-innovation text
entries. Selection requires NO attention matrices, extra model pass, learned
parameters, or tuning other than the repo's existing eviction --ratio.

Note: This is a falsifiable research hypothesis, NOT a validated improvement
on PyramidKV. Pure token eviction cannot recover an earlier evicted patch.
"""

import math

import torch
import torch.nn.functional as F


class AtlasKVCache:
    """Drop-in LLaVA legacy tuple KV cache policy, batch-size one."""

    needs_attention = False

    def __init__(self, ratio=0.0, layer_num=32, image_start=None,
                 image_token_len=0, start_size=1, k_seq_dim=2, v_seq_dim=2):
        self.ratio = float(ratio)
        self.layer_num = int(layer_num)
        self.image_start = image_start
        self.image_token_len = int(image_token_len)
        self.start_size = int(start_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        if not 0 <= self.ratio < 1:
            raise ValueError("ratio must be in [0, 1)")
        if self.image_start is None or self.image_token_len < 1:
            raise ValueError("AtlasKV requires an actual visual KV span")
        if self.k_seq_dim != 2 or self.v_seq_dim != 2:
            raise ValueError("Expected [batch, heads, tokens, dimension] KV")
        self.reset()

    def reset(self):
        self._image_ids = None      # Per-layer: original 0..patch_count-1, or -1
        self._turn_ids = None       # Per-layer: integer arrival turn; -1 for image
        self._logical_len = None
        self.turn_index = 0
        self.last_layer_budgets = None
        self.last_finalize_stats = None

    @staticmethod
    def _innovation(values):
        """Normalized value residual to adjacent tokens [N,D] -> [N]."""
        n = values.shape[0]
        if n < 2:
            return torch.ones(n, dtype=torch.float32, device=values.device)
        x = F.normalize(values.float(), dim=-1)
        left = torch.cat((x[:1], x[:-1]), 0)
        right = torch.cat((x[1:], x[-1:]), 0)
        return (x - (left + right) * 0.5).square().sum(dim=-1)

    @staticmethod
    def _spatial_witnesses(features, original_ids, number, original_count):
        """One maximally informative witness per spatial cell, then fill by detail.

        This is coverage of *physical image cells*, not facility-location/key
        similarity, and remains meaningful when no later question is known.
        """
        n = int(features.shape[0])
        number = min(max(int(number), 0), n)
        device = features.device
        if number == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if number == n:
            return torch.arange(n, device=device)
        side = math.ceil(math.sqrt(original_count))
        normalized = F.normalize(features.float(), dim=-1)
        # Spatial neighbour residual: fast 3x3 pooling on the FULL initial
        # image. After compression original_ids are sparse: a stable atlas is
        # reused unchanged while it still fits, so a sparse score is rarely
        # needed (the simple 1D innovation fallback covers that case).
        if n == original_count and torch.equal(original_ids, torch.arange(n, device=device)):
            grid = F.pad(normalized.T.reshape(1, normalized.shape[-1], -1), (0, side * side - n))
            grid = grid.reshape(1, normalized.shape[-1], side, side)
            local = F.avg_pool2d(grid, 3, stride=1, padding=1, count_include_pad=False)
            neighbours = local.flatten(2)[0].T[:n]
            scores = (normalized - neighbours).square().sum(dim=-1)
        else:
            scores = AtlasKVCache._innovation(features)

        cells = max(1, math.isqrt(number))
        row = original_ids // side
        col = original_ids % side
        region = (row * cells // side) * cells + (col * cells // side)
        # One winning index from each populated cell. Ties resolve in
        # ascending original position, making runs reproducible.
        max_scores = torch.full((cells * cells,), float("-inf"), device=device)
        max_scores.scatter_reduce_(0, region, scores, reduce="amax", include_self=True)
        contenders = torch.where(scores == max_scores[region],
                                 torch.arange(n, device=device), n)
        representatives = torch.full((cells * cells,), n, dtype=torch.long, device=device)
        representatives.scatter_reduce_(0, region, contenders, reduce="amin", include_self=True)
        selected = representatives[representatives < n]
        remaining = number - selected.numel()
        if remaining > 0:
            rest_score = scores.clone()
            rest_score[selected] = float("-inf")
            extra = torch.topk(rest_score, remaining, sorted=False).indices
            selected = torch.cat((selected, extra))
        return selected.sort().values

    @staticmethod
    def _text_witnesses(features, turns, num_to_keep, latest_text_count, start_size=1):
        """Recent context + one witness per older turn + value innovations."""
        n = int(features.shape[0])
        k = min(max(int(num_to_keep), 0), n)
        dev = features.device
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=dev)
        if k == n:
            return torch.arange(n, device=dev)

        score = AtlasKVCache._innovation(features)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # Recent text survives; grows sublinearly with the actual memory budget.
        prefix = min(n, k, max(1, int(start_size)))
        mask[:prefix] = True  # BOS/system prefix survives across turns
        recent = min(n-prefix, k-prefix, max(1, math.isqrt(max(1, latest_text_count))))
        if recent:
            mask[n-recent:] = True
        left = k - int(mask.sum().item())
        if left:
            # An older turn receives at least one witness where feasible.
            older_turns = torch.unique(turns[~mask], sorted=True)
            for turn in older_turns.tolist():
                if left <= 0:
                    break
                eligible = (turns == turn) & ~mask
                ranked = torch.where(eligible, score, float('-inf'))
                winner = int(ranked.argmax().item())
                if eligible[winner]:
                    mask[winner] = True
                    left -= 1
        if left:
            ranked = score.masked_fill(mask, float('-inf'))
            mask[torch.topk(ranked, left, sorted=False).indices] = True
        return torch.where(mask)[0]

    def _grow(self, index, seq_len, device):
        img = self._image_ids[index]
        old = img.numel()
        if seq_len < old:
            raise RuntimeError("AtlasKV metadata longer than the physical cache")
        if seq_len == old:
            return
        delta = seq_len-old
        self._image_ids[index] = torch.cat((img, torch.full((delta,), -1, dtype=torch.long, device=device)))
        self._turn_ids[index] = torch.cat((self._turn_ids[index],
                                         torch.full((delta,), self.turn_index, dtype=torch.long, device=device)))

    def _init(self, past):
        if len(past) != self.layer_num:
            raise ValueError(f"Expected {self.layer_num} layers, received {len(past)}")
        length = past[0][0].shape[2]
        if length < int(self.image_start) + self.image_token_len:
            raise ValueError("The visual token span exceeds the prefill cache")
        self._image_ids, self._turn_ids = [], []
        for k, _ in past:
            if k.shape[0] != 1:
                raise ValueError("This experimental cache supports batch_size=1 only")
            if k.shape[2] != length:
                raise ValueError("All layers must have the same prefill sequence length")
            dev = k.device
            ids = torch.full((length,), -1, dtype=torch.long, device=dev)
            lo = int(self.image_start)
            ids[lo:lo+self.image_token_len] = torch.arange(self.image_token_len, device=dev)
            turns = torch.zeros(length, dtype=torch.long, device=dev)
            turns[lo:lo+self.image_token_len] = -1
            self._image_ids.append(ids)
            self._turn_ids.append(turns)

    def _compress(self, past, stage):
        capacity = max(1, int(self._logical_len * (1 - self.ratio)))
        compressed, new_img, new_turn, counts = [], [], [], []
        for layer, (key, value) in enumerate(past):
            length = key.shape[2]
            ids = self._image_ids[layer]
            turns = self._turn_ids[layer]
            if length <= capacity:
                compressed.append([key, value])
                new_img.append(ids)
                new_turn.append(turns)
                counts.append(length)
                continue

            # Shared selected positions across heads prevent per-head provenance
            # mismatches and make the selection far cheaper than a headwise loop.
            image_candidates = torch.where(ids >= 0)[0]
            text_candidates = torch.where(ids < 0)[0]
            n_img, n_text = image_candidates.numel(), text_candidates.numel()
            # Reserve enough text for dialogue while retaining the visual atlas.
            text_quota = min(n_text, max(math.ceil(capacity/3), math.isqrt(capacity)))
            if n_img < capacity - text_quota:
                text_quota = min(n_text, capacity-n_img)
            image_quota = min(n_img, capacity-text_quota)
            text_quota = min(n_text, capacity-image_quota)
            if image_quota + text_quota < capacity:
                image_quota = min(n_img, capacity-text_quota)

            value_means = value[0].float().mean(dim=0)
            if image_quota == n_img:
                img_sel = image_candidates
            else:
                pos = self._spatial_witnesses(value_means[image_candidates],
                                              ids[image_candidates], image_quota,
                                              self.image_token_len)
                img_sel = image_candidates[pos]
            if text_quota == n_text:
                text_sel = text_candidates
            else:
                pos = self._text_witnesses(value_means[text_candidates],
                                           turns[text_candidates], text_quota,
                                           capacity, self.start_size)
                text_sel = text_candidates[pos]
            selection = torch.cat((img_sel, text_sel)).sort().values
            if selection.numel() != min(capacity, length):
                raise RuntimeError("AtlasKV did not select the exact budget")
            compressed.append([key.index_select(2, selection), value.index_select(2, selection)])
            new_img.append(ids.index_select(0, selection))
            new_turn.append(turns.index_select(0, selection))
            counts.append(selection.numel())
        self._image_ids, self._turn_ids = new_img, new_turn
        self.last_layer_budgets = counts
        self.last_finalize_stats = {
            "stage": stage,
            "turn": self.turn_index + 1,
            "logical_len": self._logical_len,
            "target_per_layer": capacity,
            "kv_after_mean": float(sum(counts) / len(counts)),
            "image_witnesses_layer0": int((new_img[0] >= 0).sum().item()),
            "layer_budgets": counts,
        }
        return compressed

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("AtlasKV needs the logical token count")
        if self._image_ids is None:
            self._init(past_key_values)
            self._logical_len = int(num_of_token)
            return self._compress(past_key_values, stage="prefill")
        self._logical_len = int(num_of_token)
        for i,(k,_) in enumerate(past_key_values):
            self._grow(i, k.shape[2], k.device)
        return past_key_values

    def finalize_turn(self, past_key_values):
        if past_key_values is None or self._image_ids is None:
            return past_key_values
        out = self._compress(past_key_values, stage="turn_boundary")
        self.turn_index += 1
        return out
