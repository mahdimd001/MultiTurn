"""
MEDA adapted to the continuous multi-turn evaluators.

Paper: Wan et al., "MEDA: Dynamic KV Cache Allocation for Efficient Multimodal
Long-Context Inference", NAACL 2025 (arXiv:2502.17599).
Code:  https://github.com/AIoT-MLSys-Lab/MEDA
       (Dynamic-MLLMs/LLaVA-mix_merge_v1/llava/model/kv_token_merge/)

MEDA has three parts.  All three are kept here:

1. Layer-wise budget from cross-modal attention entropy (paper Eqs. 4-7):

       E_TV^l = mean entropy of text-query rows over visual keys
       E_VT^l = mean entropy of visual-query rows over text keys
       E_CM^l = E_TV^l + E_VT^l
       alpha_l = softmax_l(E_CM) * L          (mean over layers is 1)
       budget_l = alpha_l * target_per_layer

   Layers whose cross-modal attention is MORE diffuse receive MORE cache.

2. Token selection inside each layer (paper Eqs. 8-10):
       score   = cumulative attention received by each KV entry
       text    : score += max(score)          (text outranks visual entries)
       keep    = most recent M entries  +  top-N of the rest by score
       M : N   = recent_fraction : (1 - recent_fraction), paper uses 3:1

3. Merging (paper Eqs. 11-12): each dropped entry is matched to the kept
   entry with the most cosine-similar KEY, and the kept K/V becomes the mean
   of itself and every entry matched to it.  Matching is done per head.

What changes for multi-turn
---------------------------
The original MEDA compresses ONCE, at the end of prompt encoding, and decoding
only appends.  In a continuous conversation that would let the cache grow
without bound after Turn 1, so it would not be comparable with the other
baselines at the same ``--ratio``.  This adaptation therefore:

  * Turn-1 prefill: measures E_CM per layer, freezes the layer weights, and
    compresses immediately (MEDA's prompt-stage compression).
  * Inside a turn: new tokens are appended and their attention keeps
    accumulating into the scores (no compression, as in MEDA decoding).
  * Every turn boundary (``finalize_turn``): compresses again to
    ``(1 - ratio) * logical_length`` per layer on average, using the same
    frozen entropy weights.

The layer weights are measured once because visual queries exist only in the
Turn-1 multimodal prefill; later turns contain text queries only.

Bookkeeping follows the multi-turn H2O cache:
  * state is created lazily on the attention device (no ``.cuda()``);
  * per-layer score/modality state grows when later turns append text;
  * after compression the state is gathered with exactly the KV indices;
  * ``num_of_token`` (the logical, uncompressed length) sets the budget.

Deviations from the released code, all deliberate:
  * Cross-modal entropy is the paper's definition.  The released code uses the
    entropy of the whole attention map; use ``entropy_mode="full"`` for that.
  * Under the causal mask, text queries BEFORE the image cannot see it and
    visual queries can see only the text BEFORE the image.  E_TV therefore
    uses post-image text queries, E_VT uses pre-image text keys, and each
    cross-modal block is renormalized (equal to a softmax over that block).
  * Budgets are bounded to [min_layer_ratio * logical_len, layer length] like
    the code's clamp, then redistributed so the total equals the ratio target.
  * The released code keeps every text token regardless of budget.  This
    version uses the paper's bounded rule (text boost + recent window).
"""

import math

import torch
import torch.nn.functional as F


class MEDACache:
    """Multi-turn MEDA: entropy-guided layer budgets + text-prior + K-similarity merge."""

    def __init__(
        self,
        start_size=1,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        layer_num=40,
        image_start=None,
        image_token_len=0,
        recent_fraction=0.75,
        entropy_mode="crossmodal",
        merge=True,
        min_layer_ratio=0.01,
    ):
        self.start_size = int(start_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.ratio = float(ratio)
        self.layer_num = int(layer_num)
        self.image_start = image_start
        self.image_token_len = int(image_token_len)
        self.recent_fraction = float(recent_fraction)
        self.entropy_mode = str(entropy_mode)
        self.merge = bool(merge)
        self.min_layer_ratio = float(min_layer_ratio)

        if self.k_seq_dim != 2 or self.v_seq_dim != 2:
            raise ValueError("MEDACache expects LLaMA KV layout [batch, heads, seq, dim]")
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError("ratio must satisfy 0 <= ratio < 1")
        if not 0.0 <= self.recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        if self.entropy_mode not in {"crossmodal", "full"}:
            raise ValueError("entropy_mode must be 'crossmodal' or 'full'")

        # Protect the final KV entry, like the other multi-turn caches.
        self.protect_size = 1
        self.reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state at the beginning of a NEW dataset conversation."""
        self.score_sum = None      # per layer: [S_l] cumulative attention received
        self.image_mask = None     # per layer: [S_l] True for visual entries
        self.layer_entropy = None  # [L] E_CM measured at the multimodal prefill
        self.layer_weights = None  # [L] alpha_l, mean 1, frozen after Turn 1
        self.latest_logical_len = None
        self.turn_index = 0
        self._pending_prefill = False
        self.last_layer_budgets = None
        self.last_finalize_stats = None

    def _initial_image_mask(self, length, device):
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        if self.image_start is not None and self.image_token_len > 0:
            lo = max(0, int(self.image_start))
            hi = min(length, lo + self.image_token_len)
            mask[lo:hi] = True
        return mask

    def _initialize(self, length, device):
        self.score_sum = [
            torch.zeros(length, dtype=torch.float32, device=device)
            for _ in range(self.layer_num)
        ]
        self.image_mask = [
            self._initial_image_mask(length, device) for _ in range(self.layer_num)
        ]
        self.layer_entropy = torch.full(
            (self.layer_num,), float("nan"), dtype=torch.float32, device=device
        )

    def _grow_layer(self, layer_idx, length):
        """Extend one layer's state when later turns append TEXT tokens."""
        current = self.score_sum[layer_idx].numel()
        if length < current:
            raise RuntimeError(
                f"MEDA layer {layer_idx}: state length {current} exceeds KV length "
                f"{length}. Score/KV bookkeeping is inconsistent."
            )
        if length == current:
            return
        add = length - current
        device = self.score_sum[layer_idx].device
        self.score_sum[layer_idx] = torch.cat(
            [self.score_sum[layer_idx], torch.zeros(add, dtype=torch.float32, device=device)]
        )
        self.image_mask[layer_idx] = torch.cat(
            [self.image_mask[layer_idx], torch.zeros(add, dtype=torch.bool, device=device)]
        )

    # ------------------------------------------------------------------
    # Entropy (paper Eqs. 4-6)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_entropy(probabilities, eps=1e-12):
        p = probabilities.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def _renormalized_block_entropy(self, block):
        """Mean row entropy of an attention block renormalized over its columns."""
        if block.numel() == 0:
            return block.new_zeros(())
        block = block / block.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return self._row_entropy(block).mean()

    def _layer_entropy(self, heads, head_mean):
        """E_CM for one layer from its prefill attention.

        heads:     [heads, S, S] attention probabilities
        head_mean: [S, S] float32, attention averaged across heads
        """
        if self.entropy_mode == "full":
            # Released-code definition: per-head entropy of the whole map,
            # averaged over heads and query rows.
            return self._row_entropy(heads.float()).mean()

        length = head_mean.shape[-1]
        if self.image_start is None or self.image_token_len <= 0:
            return head_mean.new_zeros(())
        lo = max(0, int(self.image_start))
        hi = min(length, lo + self.image_token_len)
        # Text queries after the image -> visual keys.
        e_tv = self._renormalized_block_entropy(head_mean[hi:, lo:hi])
        # Visual queries -> the text keys they can see (before the image).
        e_vt = self._renormalized_block_entropy(head_mean[lo:hi, :lo])
        return e_tv + e_vt

    def _frozen_layer_weights(self):
        """alpha_l = softmax(E_CM) * L, measured once (paper Eq. 7)."""
        if self.layer_weights is None:
            if torch.isnan(self.layer_entropy).any():
                missing = torch.isnan(self.layer_entropy).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"MEDA did not observe prefill attention for layers {missing}."
                )
            self.layer_weights = torch.softmax(self.layer_entropy, dim=0) * self.layer_num
        return self.layer_weights

    # ------------------------------------------------------------------
    # Attention observation
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        """Accumulate one layer's attention; measure entropy at the prefill."""
        if attention.dim() != 4 or attention.shape[0] != 1:
            raise RuntimeError(
                f"Expected [1, heads, queries, keys], got {tuple(attention.shape)}"
            )
        query_len, key_len = attention.shape[-2], attention.shape[-1]

        if self.score_sum is None:
            if query_len != key_len:
                raise RuntimeError(
                    "MEDACache must observe the multimodal prefill before decoding."
                )
            self._initialize(key_len, attention.device)

        self._grow_layer(layer_idx, key_len)

        heads = attention[0]
        head_mean = heads.mean(dim=0, dtype=torch.float32)  # [queries, keys]
        # Cumulative attention received by every key (paper Eq. 8).
        self.score_sum[layer_idx].add_(head_mean.sum(dim=0))

        if query_len > 1 and torch.isnan(self.layer_entropy[layer_idx]):
            self.layer_entropy[layer_idx] = self._layer_entropy(heads, head_mean)
            self._pending_prefill = True

    # ------------------------------------------------------------------
    # Budgets (paper Eq. 7, with exact global total)
    # ------------------------------------------------------------------
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

    def _layer_budgets(self):
        weights = self._frozen_layer_weights()
        device = weights.device
        lengths = torch.tensor([x.numel() for x in self.score_sum], device=device)
        target_per_layer = max(1, int(self.latest_logical_len * (1.0 - self.ratio)))
        floor = max(
            self.start_size + self.protect_size,
            int(math.ceil(self.min_layer_ratio * self.latest_logical_len)),
        )
        lower = torch.clamp(lengths, max=floor)
        budgets = self._bounded_largest_remainder(
            target_per_layer * self.layer_num, weights, lower, lengths
        )
        return budgets, target_per_layer

    # ------------------------------------------------------------------
    # Selection (paper Eqs. 8-10) and merge (Eqs. 11-12)
    # ------------------------------------------------------------------
    def _select(self, layer_idx, budget):
        scores = self.score_sum[layer_idx]
        length = scores.numel()
        device = scores.device
        if budget >= length:
            return torch.arange(length, device=device), torch.arange(0, device=device)

        start = min(self.start_size, length)
        prefix = torch.arange(start, device=device)
        remaining = max(self.protect_size, int(budget) - start)

        # Recent window M (always kept) and important top-N (paper 3:1 split).
        recent_count = int(round(self.recent_fraction * remaining))
        recent_count = max(self.protect_size, min(remaining, recent_count))
        recent_start = max(start, length - recent_count)
        recent = torch.arange(recent_start, length, device=device)

        important = remaining - recent.numel()
        candidate_count = recent_start - start
        if important > 0 and candidate_count > 0:
            candidate_scores = scores[start:recent_start].clone()
            text = ~self.image_mask[layer_idx][start:recent_start]
            # Eq. 9: text entries get the max score added, so they outrank
            # every visual entry when competing for the top-N slots.
            candidate_scores[text] += scores.max()
            top = torch.topk(
                candidate_scores, min(important, candidate_count), sorted=False
            ).indices + start
        else:
            top = prefix[:0]

        retained = torch.sort(torch.cat([prefix, top, recent])).values
        keep = torch.zeros(length, dtype=torch.bool, device=device)
        keep[retained] = True
        thrown = torch.arange(length, device=device)[~keep]
        return retained, thrown

    def _merge_layer(self, k, v, retained, thrown):
        """Gather retained entries; average each dropped entry into its nearest key."""
        retained = retained.to(k.device)
        k_keep = k.index_select(self.k_seq_dim, retained)
        v_keep = v.index_select(self.v_seq_dim, retained)
        if not self.merge or thrown.numel() == 0:
            return k_keep, v_keep

        thrown = thrown.to(k.device)
        k_drop = k.index_select(self.k_seq_dim, thrown)
        v_drop = v.index_select(self.v_seq_dim, thrown)

        # Eq. 11: per-head cosine similarity between dropped and kept keys.
        similarity = F.normalize(k_drop.float(), dim=-1) @ F.normalize(
            k_keep.float(), dim=-1
        ).transpose(-1, -2)                                   # [B, H, dropped, kept]
        nearest = similarity.argmax(dim=-1)                   # [B, H, dropped]
        index = nearest.unsqueeze(-1).expand_as(k_drop)

        # Eq. 12: kept <- (kept + sum of matched dropped) / (count + 1).
        k_new = k_keep.float().scatter_reduce(
            self.k_seq_dim, index, k_drop.float(), reduce="mean", include_self=True
        ).to(k.dtype)
        v_new = v_keep.float().scatter_reduce(
            self.v_seq_dim, index, v_drop.float(), reduce="mean", include_self=True
        ).to(v.dtype)
        return k_new, v_new

    def _compress(self, past_key_values, stage):
        budgets, target_per_layer = self._layer_budgets()
        kv_before = past_key_values[0][0].size(self.k_seq_dim)

        new_past = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            retained, thrown = self._select(layer_idx, int(budgets[layer_idx]))
            k_new, v_new = self._merge_layer(k, v, retained, thrown)
            new_past.append([k_new, v_new])
            # State now describes the NEW physical cache of this layer.
            self.score_sum[layer_idx] = self.score_sum[layer_idx][retained]
            self.image_mask[layer_idx] = self.image_mask[layer_idx][retained]

        self.last_layer_budgets = budgets.detach().cpu().tolist()
        lengths_after = [k.size(self.k_seq_dim) for k, _ in new_past]
        self.last_finalize_stats = {
            "stage": stage,
            "turn": self.turn_index + 1,
            "logical_len": int(self.latest_logical_len),
            "target_per_layer": int(target_per_layer),
            "kv_before_layer0": int(kv_before),
            "kv_after_min": int(min(lengths_after)),
            "kv_after_max": int(max(lengths_after)),
            "kv_after_total": int(sum(lengths_after)),
            "image_kept_layer0": int(self.image_mask[0].sum().item()),
            "layer_budgets": self.last_layer_budgets,
        }
        return new_past

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("MEDACache requires num_of_token for ratio-based pruning.")

        # Fallback when attentions were returned normally instead of the hook.
        if attentions is not None and attentions[0] is not None:
            if len(attentions) != self.layer_num:
                raise ValueError("One attention tensor per transformer layer is required")
            for layer_idx, attention in enumerate(attentions):
                self.observe_layer(layer_idx, attention)

        if self.score_sum is None:
            raise ValueError(
                "MEDACache requires output_attentions=True (or the online attention hook)."
            )

        for layer_idx, (k, _) in enumerate(past_key_values):
            length = k.size(self.k_seq_dim)
            if self.score_sum[layer_idx].numel() != length:
                raise RuntimeError(
                    f"MEDA layer {layer_idx}: observed attention covers "
                    f"{self.score_sum[layer_idx].numel()} keys, KV has {length}."
                )

        self.latest_logical_len = int(num_of_token)

        # MEDA compresses once the prompt has been encoded (Turn-1 prefill).
        if self._pending_prefill:
            self._pending_prefill = False
            return self._compress(past_key_values, stage="prefill")

        # Within a turn: append only, as in MEDA decoding.
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Re-compress the completed turn with the frozen entropy weights."""
        if past_key_values is None or self.score_sum is None:
            return past_key_values
        result = self._compress(past_key_values, stage="turn_boundary")
        self.turn_index += 1
        return result