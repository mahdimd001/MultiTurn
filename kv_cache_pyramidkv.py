"""
PyramidKV adapted to the continuous multi-turn evaluators.

Paper: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funneling" (arXiv:2406.02069).
Code:  https://github.com/Zefan-Cai/KVCache-Factory
       (pyramidkv/pyramidkv_utils.py, class PyramidKVCluster)

PyramidKV, as implemented in KVCache-Factory:

1. Per-layer budget shrinks linearly with depth (the "pyramid").  With
   C = average per-layer capacity, W = observation window, M = C - W:

       min_num = M // beta
       max_num = 2 * M - min_num
       if max_num >= q_len - W:            # cannot keep more than exists
           max_num = q_len - W
           min_num = 2 * M - max_num
       steps   = (max_num - min_num) // (L - 1)
       past_l  = max_num - l * steps       # layer l keeps past_l + W entries

   Two regimes come first: q_len < C keeps everything, and q_len < 2M gives
   every layer the same M (SnapKV behavior).

2. Scores: attention from the last W queries (the observation window) to the
   earlier keys, summed over those queries, then smoothed with a 1-D pooling
   (KVCache-Factory's PyramidKV runs use maxpool, kernel 7, window 8).

3. Selection: per-head top-k of the pooled scores, plus the window itself.
   Heads in a layer keep the same COUNT of entries but may keep different
   positions.  Dropped entries are evicted (KVCache-Factory default merge=None).

What changes for multi-turn
---------------------------
KVCache-Factory compresses once, during prefill, and decoding only appends.
In a continuous conversation that would let the cache grow without bound after
Turn 1, so this adaptation compresses at the same points as the multi-turn
MEDA cache:

  * Turn-1 prefill: the observation window is the last W prompt tokens, exactly
    as in the original.
  * Inside a turn: tokens are appended; the attention rows of the most recent
    W tokens are kept as the next observation window.
  * Every turn boundary (``finalize_turn``): recompress with the last W tokens
    of the finished turn as the observation window.

The average capacity follows the other baselines: C = int(logical_len *
(1 - ratio)), where the logical length is the uncompressed conversation
length (``num_of_token``).  The pyramid schedule and its integer rounding are
unchanged from KVCache-Factory, so the realized average can exceed C by at
most (L - 1) / 2 entries per layer.

Bookkeeping follows the multi-turn H2O cache: state is created lazily on the
attention device, only W attention rows per layer are retained, and every
compression starts a fresh window over the new physical layout.
"""

import torch
import torch.nn.functional as F


class PyramidKVCache:
    """Multi-turn PyramidKV: depth-decreasing budgets + windowed per-head top-k."""

    def __init__(
        self,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        layer_num=40,
        window_size=8,
        kernel_size=7,
        pooling="maxpool",
        beta=20,
    ):
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.ratio = float(ratio)
        self.layer_num = int(layer_num)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = str(pooling)
        self.beta = float(beta)

        if self.k_seq_dim != 2 or self.v_seq_dim != 2:
            raise ValueError("PyramidKVCache expects KV layout [batch, heads, seq, dim]")
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError("ratio must satisfy 0 <= ratio < 1")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.pooling not in {"avgpool", "maxpool"}:
            raise ValueError("pooling must be 'avgpool' or 'maxpool'")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.layer_num < 2:
            raise ValueError("PyramidKV needs at least two layers")
        self.reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state at the beginning of a NEW dataset conversation."""
        # Per layer: list of [heads, rows, keys_at_that_time] attention blocks,
        # holding at most window_size rows in total.
        self._window = None
        self._pending_prefill = False
        self.latest_logical_len = None
        self.turn_index = 0
        self.last_layer_budgets = None
        self.last_finalize_stats = None

    def _row_count(self, layer_idx):
        return sum(block.shape[1] for block in self._window[layer_idx])

    def _trim_window(self, layer_idx):
        blocks = self._window[layer_idx]
        while blocks and self._row_count(layer_idx) - blocks[0].shape[1] >= self.window_size:
            blocks.pop(0)

    def _window_rows(self, layer_idx, length):
        """Last <= W attention rows of this layer, right-padded to ``length`` keys.

        A row recorded earlier covers fewer keys; padding those missing keys
        with zero attention is exactly the causal mask.
        """
        blocks = self._window[layer_idx]
        if not blocks:
            return None
        padded = [F.pad(block, (0, length - block.shape[-1])) for block in blocks]
        rows = torch.cat(padded, dim=1)
        return rows[:, -self.window_size:, :]

    # ------------------------------------------------------------------
    # Attention observation
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        """Keep the attention rows of the most recent W queries of this layer."""
        if attention.dim() != 4 or attention.shape[0] != 1:
            raise RuntimeError(
                f"Expected [1, heads, queries, keys], got {tuple(attention.shape)}"
            )
        query_len, key_len = attention.shape[-2], attention.shape[-1]

        if self._window is None:
            if query_len != key_len:
                raise RuntimeError(
                    "PyramidKVCache must observe the multimodal prefill before decoding."
                )
            self._window = [[] for _ in range(self.layer_num)]

        # Copy only the window rows, so the full prefill attention can be freed.
        rows = attention[0, :, -self.window_size:, :].float()
        if query_len > 1:
            self._window[layer_idx] = [rows]
            self._pending_prefill = True
        else:
            self._window[layer_idx].append(rows)
            self._trim_window(layer_idx)

    # ------------------------------------------------------------------
    # Pyramid budget (unchanged KVCache-Factory schedule)
    # ------------------------------------------------------------------
    def pyramid_capacities(self, capacity, window, q_len):
        """Per-layer number of PAST entries kept (window excluded).

        Returns None when q_len < capacity (no compression in that regime).
        """
        past_average = capacity - window
        if q_len < capacity:
            return None
        if q_len < past_average * 2:
            return [past_average] * self.layer_num

        min_num = int(past_average // self.beta)
        max_num = past_average * 2 - min_num
        if max_num >= q_len - window:
            max_num = q_len - window
            min_num = past_average * 2 - max_num
        steps = (max_num - min_num) // (self.layer_num - 1)
        return [max_num - layer_idx * steps for layer_idx in range(self.layer_num)]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def _pool(self, scores):
        """1-D pooling over keys, per head (scores: [heads, keys])."""
        pool = F.max_pool1d if self.pooling == "maxpool" else F.avg_pool1d
        return pool(
            scores.unsqueeze(0),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).squeeze(0)

    def _compress(self, past_key_values, stage):
        lengths = [k.size(self.k_seq_dim) for k, _ in past_key_values]
        capacity = max(1, int(self.latest_logical_len * (1.0 - self.ratio)))
        # The window must leave at least one past slot (KVCache-Factory asserts
        # max_capacity_prompt - window_size > 0).
        window = max(1, min(self.window_size, capacity - 1)) if capacity > 1 else 1
        past_caps = self.pyramid_capacities(capacity, window, max(lengths))

        new_past, kept = [], []
        for layer_idx, (k, v) in enumerate(past_key_values):
            length = lengths[layer_idx]
            rows = self._window_rows(layer_idx, length)
            layer_window = min(window, rows.shape[1]) if rows is not None else 0
            past_len = length - layer_window
            keep_past = None if past_caps is None else min(past_caps[layer_idx], past_len)

            if keep_past is None or keep_past >= past_len or layer_window == 0:
                new_past.append([k, v])
                kept.append(length)
                continue

            # Observation-window attention to the past keys, summed over queries.
            scores = rows[:, -layer_window:, :past_len].sum(dim=1)     # [heads, past]
            kv_heads = k.shape[1]
            if scores.shape[0] != kv_heads:
                # GQA: average the query heads that share a KV head.
                scores = scores.view(kv_heads, -1, past_len).mean(dim=1)
            scores = self._pool(scores)

            top = scores.topk(max(0, keep_past), dim=-1).indices.sort(dim=-1).values
            window_idx = torch.arange(past_len, length, device=top.device)
            index = torch.cat([top, window_idx.expand(kv_heads, -1)], dim=-1)  # [heads, keep]
            index = index.to(k.device)[None, :, :, None]

            k_new = k.gather(self.k_seq_dim, index.expand(k.shape[0], -1, -1, k.shape[-1]))
            v_new = v.gather(self.v_seq_dim, index.expand(v.shape[0], -1, -1, v.shape[-1]))
            new_past.append([k_new, v_new])
            kept.append(k_new.size(self.k_seq_dim))

        # Rows recorded so far refer to the old layout; start a fresh window.
        self._window = [[] for _ in range(self.layer_num)]
        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": stage,
            "turn": self.turn_index + 1,
            "logical_len": int(self.latest_logical_len),
            "target_per_layer": int(capacity),
            "window": int(window),
            "regime": (
                "keep_all" if past_caps is None
                else "uniform" if len(set(past_caps)) == 1
                else "pyramid"
            ),
            "kv_before_layer0": int(lengths[0]),
            "kv_after_first": int(kept[0]),
            "kv_after_last": int(kept[-1]),
            "kv_after_mean": float(sum(kept) / len(kept)),
            "layer_budgets": kept,
        }
        return new_past

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("PyramidKVCache requires num_of_token for ratio-based pruning.")

        # Fallback when attentions were returned normally instead of the hook.
        if attentions is not None and attentions[0] is not None:
            if len(attentions) != self.layer_num:
                raise ValueError("One attention tensor per transformer layer is required")
            for layer_idx, attention in enumerate(attentions):
                self.observe_layer(layer_idx, attention)

        if self._window is None:
            raise ValueError(
                "PyramidKVCache requires output_attentions=True (or the online attention hook)."
            )

        for layer_idx, (k, _) in enumerate(past_key_values):
            blocks = self._window[layer_idx]
            length = k.size(self.k_seq_dim)
            if blocks and blocks[-1].shape[-1] != length:
                raise RuntimeError(
                    f"PyramidKV layer {layer_idx}: latest attention covers "
                    f"{blocks[-1].shape[-1]} keys, KV has {length}."
                )

        self.latest_logical_len = int(num_of_token)

        # PyramidKV compresses once the prompt has been encoded (Turn-1 prefill).
        if self._pending_prefill:
            self._pending_prefill = False
            return self._compress(past_key_values, stage="prefill")

        # Within a turn: append only, as in PyramidKV decoding.
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Recompress the finished turn; its last W tokens are the window."""
        if past_key_values is None or self._window is None:
            return past_key_values
        result = self._compress(past_key_values, stage="turn_boundary")
        self.turn_index += 1
        return result