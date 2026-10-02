"""
Corrected multimodal Elastic / H-Elastic KV-cache implementations.

This module is intentionally a DROP-IN companion to the original kv_cache.py:

    - ElasticCache, LocalCache, H2OCache, HelasticCache are imported unchanged.
    - MultimodalElasticCache and MultimodalHelasticCache are corrected here.

The multimodal fixes are designed for a conversation of the form:

    Turn 1: image + text
    Turn 2+: text only

Important design choices
------------------------
1. The image/text modality mask is tracked PER LAYER and is gathered whenever
   each layer selects a different set of KV entries.
2. A forgotten text KV entry may merge only into a retained text KV entry.
   A forgotten image KV entry may merge only into a retained image KV entry.
   Therefore a retained slot never becomes an ambiguous text/image mixture.
3. The multimodal connection score is assigned to a TEXT QUERY TOKEN itself:

       connection(text_token) = sum attention(text_query -> image_keys)

   This captures how visually grounded a text token is.
4. Raw attention still scores KEY tokens.  Therefore if a text query strongly
   attends to an image key, that image key already receives raw-attention mass;
   we do not need to add the text-token connection bonus to image tokens too.
5. After pruning, score_sum, conn_score_sum, and modality masks are gathered
   with exactly the same indices as the KV cache.
6. MultimodalElasticCache uses cumulative (non-decayed) importance.
   MultimodalHelasticCache uses exponentially decayed importance so recent
   turns can change which tokens are important.

This makes both classes usable in true multi-turn evaluation while preserving
one clear distinction between Elastic and H-Elastic: cumulative vs. recency-
weighted/dynamic importance.
"""

import torch

# Keep the original non-multimodal baselines unchanged.
from kv_cache import ElasticCache, LocalCache, HelasticCache


class H2OCache:
    """Multi-turn-safe H2O-style cumulative heavy-hitter cache.

    This preserves the intent of the original baseline (retain high cumulative
    attention keys) while fixing four implementation problems that break after
    pruning and across later turns:

    1. No hard-coded ``.cuda()`` allocation; state follows the attention device.
    2. The score vector grows when new text is appended in later turns.
    3. Heavy hitters are selected by score rank correctly.
    4. After physical KV pruning, the score vector is gathered with the exact
       same retained indices, so score position i continues to describe KV i.

    Scores are shared across layers by averaging attention across layers and
    heads.  Therefore all transformer layers retain the same token positions,
    matching the spirit of the original implementation.
    """

    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
    ):
        self.start_size = int(start_size)
        self.recent_size = int(recent_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.ratio = float(ratio)
        self.protect_size = 1
        self.score_sum = None

    def reset(self):
        self.score_sum = None

    @staticmethod
    def _mean_attention(attentions):
        if attentions is None:
            raise ValueError(
                "H2OCache requires output_attentions=True, but attentions is None."
            )
        # Each layer: [batch=1, heads, query, key].
        # Stack -> [layers, batch, heads, query, key], then average layers,
        # batch and heads -> [query, key].
        stacked = torch.stack(tuple(attentions), dim=0)
        if stacked.dim() != 5:
            raise RuntimeError(
                f"Expected attention rank 5 after stacking, got {tuple(stacked.shape)}"
            )
        return stacked.float().mean(dim=(0, 1, 2))

    def _grow_scores(self, seq_len, device):
        if self.score_sum is None:
            self.score_sum = torch.zeros(seq_len, device=device, dtype=torch.float32)
            return
        current = self.score_sum.numel()
        if seq_len < current:
            raise RuntimeError(
                f"H2O score length ({current}) exceeds physical KV length ({seq_len}). "
                "Score/KV bookkeeping is inconsistent."
            )
        if seq_len > current:
            self.score_sum = torch.cat(
                [
                    self.score_sum,
                    torch.zeros(
                        seq_len - current,
                        device=self.score_sum.device,
                        dtype=self.score_sum.dtype,
                    ),
                ]
            )

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("H2OCache requires num_of_token for ratio-based pruning.")

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        attn = self._mean_attention(attentions)  # [query_len, key_len]
        if attn.shape[-1] != seq_len:
            raise RuntimeError(
                f"H2O attention key length {attn.shape[-1]} != KV length {seq_len}."
            )

        self._grow_scores(seq_len, attn.device)

        # Incoming attention received by each physical KV key from the current
        # query block. Works for both initial prefill and one-token later turns.
        self.score_sum += attn.sum(dim=0)

        # num_of_token is the logical, uncompressed stream length.
        target_size = max(1, int(num_of_token * (1.0 - self.ratio)))
        forget_num = seq_len - target_size
        if forget_num <= 0:
            return past_key_values

        start = min(self.start_size, seq_len)
        end = max(start, seq_len - self.protect_size)
        candidate_count = end - start
        if candidate_count <= 0:
            return past_key_values

        drop_count = min(int(forget_num), candidate_count)
        candidate_scores = self.score_sum[start:end]
        order = torch.argsort(candidate_scores)  # low importance -> high

        # Drop exactly the lowest-scoring candidates; retain all other
        # candidates plus protected prefix and newest suffix token(s).
        selected_candidate = order[drop_count:] + start
        prefix = torch.arange(start, device=self.score_sum.device, dtype=torch.long)
        suffix = torch.arange(end, seq_len, device=self.score_sum.device, dtype=torch.long)
        selected_idx = torch.cat([prefix, selected_candidate, suffix])
        selected_idx = torch.sort(selected_idx).values

        new_past = []
        for k, v in past_key_values:
            idx_k = selected_idx.to(k.device)
            idx_v = selected_idx.to(v.device)
            k_new = k.index_select(self.k_seq_dim, idx_k)
            v_new = v.index_select(self.v_seq_dim, idx_v)
            new_past.append([k_new, v_new])

        # Critical multi-turn fix: scores now describe the NEW physical cache.
        self.score_sum = self.score_sum[selected_idx]
        return new_past


class _MultimodalCacheBase:
    """Shared, corrected bookkeeping for multimodal Elastic-style caches."""

    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        distance=-25,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=0,
        decay=None,
    ):
        # Number of tokens at the beginning that are never considered for
        # score-based eviction.  This follows the original implementation.
        self.start_size = start_size

        # Kept for compatibility with the original constructor / experiments.
        self.recent_size = recent_size

        # KV sequence dimensions used by LLaVA/LLaMA-style caches.
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        # Fraction of the logical uncompressed stream that should be removed.
        self.ratio = ratio

        # Protect the final KV entry from score-based eviction.
        self.protect_size = 1

        # Kept for compatibility with the original implementation.
        self.distance = distance

        # Number of transformer layers.  Each layer may select different KV
        # entries, so scores and modality masks MUST be layer-specific.
        self.layer_num = layer_num

        # Weight of ordinary incoming-attention importance.
        self.attn_weight = attn_weight

        # Weight of the visual-grounding bonus for TEXT tokens.
        self.conn_weight = conn_weight

        # Location of the ACTUAL expanded visual-token block in the first
        # model prefill.  The fixed evaluator infers this from the expanded KV.
        self.image_start = image_start
        self.image_token_len = image_token_len

        # decay=None  -> cumulative Elastic score
        # decay=0.95  -> recency-weighted H-Elastic score
        self.decay = decay

        # State tensors are allocated lazily on the attention tensor's device.
        # This avoids hard-coded .cuda() calls and keeps device handling sane.
        self.score_sum = None
        self.conn_score_sum = None

        # True at a cache position means that KV slot is an IMAGE slot.
        # Shape: [layer_num, current_physical_cache_length].
        self.image_mask = None

        # The first call receives the full multimodal prefill.  Every later
        # token in the intended workload is text-only.
        self.initialized = False

    # ------------------------------------------------------------------
    # State initialization / growth
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state at the beginning of a NEW dataset conversation."""
        self.score_sum = None
        self.conn_score_sum = None
        self.image_mask = None
        self.initialized = False

    def _initial_image_mask(self, seq_len, device):
        """Build the image mask for the FIRST, unpruned multimodal prefill."""
        mask = torch.zeros(
            self.layer_num, seq_len, dtype=torch.bool, device=device
        )

        if self.image_start is None or self.image_token_len <= 0:
            return mask

        start = max(0, int(self.image_start))
        end = min(seq_len, start + int(self.image_token_len))

        if start < end:
            mask[:, start:end] = True

        return mask

    def _initialize_state(self, seq_len, device):
        """Allocate scores and modality masks for the first model call."""
        self.score_sum = torch.zeros(
            self.layer_num, seq_len, device=device, dtype=torch.float32
        )
        self.conn_score_sum = torch.zeros_like(self.score_sum)
        self.image_mask = self._initial_image_mask(seq_len, device)
        self.initialized = True

    def _grow_state_for_new_text(self, seq_len):
        """
        Extend score/mask state when Q2/A2/Q3/... append new TEXT tokens.

        After pruning, the physical KV cache is shorter than the logical
        conversation.  When the model appends q new tokens, seq_len grows by q.
        These new entries are text, so their image-mask values are False.
        """
        current_len = self.score_sum.shape[1]

        if seq_len < current_len:
            raise RuntimeError(
                f"Cache state length ({current_len}) is larger than model KV "
                f"length ({seq_len}). Score/KV bookkeeping is inconsistent."
            )

        if seq_len == current_len:
            return

        add_len = seq_len - current_len
        device = self.score_sum.device

        self.score_sum = torch.cat(
            [
                self.score_sum,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=self.score_sum.dtype,
                ),
            ],
            dim=1,
        )

        self.conn_score_sum = torch.cat(
            [
                self.conn_score_sum,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=self.conn_score_sum.dtype,
                ),
            ],
            dim=1,
        )

        # All tokens appended after the first multimodal prefill are text.
        self.image_mask = torch.cat(
            [
                self.image_mask,
                torch.zeros(
                    self.layer_num,
                    add_len,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Attention / multimodal score updates
    # ------------------------------------------------------------------
    def _prepare_attention(self, attentions):
        """
        Convert Hugging Face attention output to [layers, queries, keys].

        Each element is normally [batch=1, heads, query_len, key_len].
        Concatenating over the list dimension gives layers in dimension 0;
        averaging over heads leaves one attention matrix per layer.
        """
        if attentions is None:
            raise ValueError(
                "attentions is None. Run the model with output_attentions=True."
            )

        attn = torch.cat([a for a in attentions], dim=0)
        # Keep the model dtype here. Casting the complete [L,Q,K] tensor to
        # float32 is expensive on MPS and unnecessary; only the much smaller
        # reduced score tensors need float32 accumulation.
        return attn.mean(dim=1)  # [layers, query_len, key_len]

    def _update_scores(self, attn):
        """
        Update ordinary attention importance and text->image connection score.

        RAW SCORE (key-side importance)
        --------------------------------
        For each KV key, sum attention received from the current query tokens.
        This naturally rewards image keys when text queries attend to them.

        CONNECTION SCORE (query-side visual grounding)
        ------------------------------------------------
        For each TEXT query token q:

            conn(q) = sum_{k is image} Attention(q -> k)

        That scalar is written at q's OWN KV position.  Therefore a text token
        that strongly uses the image gets an extra retention bonus later.

        This is intentionally NOT "give image keys another connection bonus".
        Their text->image importance is already represented in raw_score.
        """
        num_layers, query_len, key_len = attn.shape

        if num_layers != self.layer_num:
            raise RuntimeError(
                f"Expected {self.layer_num} attention layers, got {num_layers}."
            )

        if key_len != self.score_sum.shape[1]:
            raise RuntimeError(
                f"Attention key length ({key_len}) != score length "
                f"({self.score_sum.shape[1]})."
            )

        # Sum the attention received by every key from the CURRENT query block.
        incoming = attn.sum(dim=1).float()  # [layers, key_len]

        if self.decay is None:
            # MMElastic: cumulative importance; old evidence is never decayed.
            self.score_sum += incoming
        else:
            # MMHelastic: recent attention gradually replaces stale history.
            self.score_sum.mul_(self.decay).add_(incoming)
            self.conn_score_sum.mul_(self.decay)

        # Query rows correspond to the LAST query_len positions in key space.
        # - first prefill: query_len == key_len -> positions 0..key_len-1
        # - later turn/decode: query positions are newly appended tail entries
        query_start = key_len - query_len

        # Vectorized equivalent of the former layer/query Python loops:
        # sum each query's attention to image keys, then write that grounding
        # score only at text-query positions in the KV tail. This avoids device
        # boolean checks and synchronizations for every layer and token.
        image_keys = self.image_mask.to(attn.dtype).unsqueeze(-1)
        image_attention = torch.bmm(attn, image_keys).squeeze(-1).float()
        query_is_text = ~self.image_mask[:, query_start:key_len]
        self.conn_score_sum[:, query_start:key_len].add_(
            image_attention * query_is_text
        )

    # ------------------------------------------------------------------
    # Score normalization / selection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_nonnegative(values):
        """Normalize non-negative attention scores without NaNs."""
        if values.numel() == 0:
            return values
        max_value = values.max()
        if max_value <= 0:
            return torch.zeros_like(values)
        return values / (max_value + 1e-6)

    def _combined_scores(self, layer_idx, start, end):
        """Return eviction scores for the candidate interval [start, end)."""
        raw = self.score_sum[layer_idx, start:end]
        conn = self.conn_score_sum[layer_idx, start:end]
        image = self.image_mask[layer_idx, start:end]
        text = ~image

        raw_norm = self._normalize_nonnegative(raw)
        conn_norm = torch.zeros_like(conn)

        # Connection score is meaningful only for text tokens.
        if text.any():
            conn_norm[text] = self._normalize_nonnegative(conn[text])

        combined = self.attn_weight * raw_norm
        combined[text] += self.conn_weight * conn_norm[text]
        return combined

    def _ensure_image_survives(
        self,
        layer_idx,
        start,
        candidate_scores,
        selected_rel,
        throw_rel,
    ):
        """
        Ensure same-modality image merging remains possible.

        If image candidates exist and an eviction would remove ALL of them,
        retain the highest-scoring image candidate and instead evict the
        lowest-scoring selected text candidate.  The number of retained entries
        stays unchanged.
        """
        candidate_image = self.image_mask[layer_idx, start:start + len(candidate_scores)]

        # Nothing special is needed if there are no image candidates.
        if not candidate_image.any():
            return selected_rel, throw_rel

        # If at least one image token already survives, we are done.
        if selected_rel.numel() > 0 and candidate_image[selected_rel].any():
            return selected_rel, throw_rel

        # Choose the best image token from the throw set to rescue.
        thrown_images = throw_rel[candidate_image[throw_rel]]
        if thrown_images.numel() == 0:
            return selected_rel, throw_rel

        rescue_rel = thrown_images[
            torch.argmax(candidate_scores[thrown_images])
        ]

        # To keep cache size unchanged, swap out the lowest-scoring selected
        # TEXT candidate.  Start/protected tokens are outside this candidate
        # set and therefore remain untouched.
        selected_text = selected_rel[~candidate_image[selected_rel]]
        if selected_text.numel() == 0:
            # Extremely small/degenerate cache: cannot perform a size-neutral
            # swap.  Keep selection unchanged rather than corrupt modalities.
            return selected_rel, throw_rel

        evict_rel = selected_text[
            torch.argmin(candidate_scores[selected_text])
        ]

        selected_rel = selected_rel[selected_rel != evict_rel]
        selected_rel = torch.cat([selected_rel, rescue_rel.view(1)])

        throw_rel = throw_rel[throw_rel != rescue_rel]
        throw_rel = torch.cat([throw_rel, evict_rel.view(1)])

        return selected_rel, throw_rel

    def _same_modality_merge_targets(
        self, layer_idx, throw_idx, retained_idx
    ):
        """
        Map every forgotten token to the nearest retained token of SAME modality.

        Because image/text masks are hard labels, cross-modality merging would
        create a mixed KV vector that cannot truthfully be called either text or
        image.  We therefore forbid such merges.
        """
        if throw_idx.numel() == 0:
            return throw_idx.clone()

        layer_image = self.image_mask[layer_idx]
        targets = []

        for forgotten in throw_idx:
            forgotten_is_image = layer_image[forgotten]
            same_modality = retained_idx[
                layer_image[retained_idx] == forgotten_is_image
            ]

            if same_modality.numel() == 0:
                raise RuntimeError(
                    "No retained KV slot of the same modality is available for "
                    "merging. Reduce the pruning ratio or retain at least one "
                    "image/text token."
                )

            nearest = torch.argmin(torch.abs(same_modality - forgotten))
            targets.append(same_modality[nearest])

        return torch.stack(targets)

    # ------------------------------------------------------------------
    # Pruning / merging / state realignment
    # ------------------------------------------------------------------
    def _merge_and_gather_layer(
        self,
        k,
        v,
        layer_idx,
        retained_idx,
        throw_idx,
        merge_idx,
    ):
        """Merge forgotten KV entries, then gather retained entries."""
        device = k.device
        retained_idx = retained_idx.to(device)
        throw_idx = throw_idx.to(device)
        merge_idx = merge_idx.to(device)

        # If nothing is being dropped, only gather (normally identity).
        if throw_idx.numel() == 0:
            k_new = k.index_select(self.k_seq_dim, retained_idx)
            v_new = v.index_select(self.v_seq_dim, retained_idx)
            return k_new, v_new

        # Gather forgotten K/V vectors.
        k_forget = k.index_select(self.k_seq_dim, throw_idx)
        v_forget = v.index_select(self.v_seq_dim, throw_idx)

        # scatter_reduce expects an index tensor with the same rank as K/V.
        k_index_shape = [1] * k.dim()
        k_index_shape[self.k_seq_dim] = -1
        k_index_shape[-1] = 1
        k_index = merge_idx.view(*k_index_shape).expand_as(k_forget)

        v_index_shape = [1] * v.dim()
        v_index_shape[self.v_seq_dim] = -1
        v_index_shape[-1] = 1
        v_index = merge_idx.view(*v_index_shape).expand_as(v_forget)

        # include_self=True (PyTorch default) averages the retained vector with
        # all forgotten vectors mapped to that slot.
        k_merged = k.scatter_reduce(
            self.k_seq_dim, k_index, k_forget, reduce="mean"
        )
        v_merged = v.scatter_reduce(
            self.v_seq_dim, v_index, v_forget, reduce="mean"
        )

        # Finally physically remove forgotten slots.
        k_new = k_merged.index_select(self.k_seq_dim, retained_idx)
        v_new = v_merged.index_select(self.v_seq_dim, retained_idx)
        return k_new, v_new

    def _realign_scores_and_masks(
        self,
        retained_idx_all,
        throw_idx_all,
        merge_idx_all,
    ):
        """
        Realign score_sum, conn_score_sum, and image_mask after KV pruning.

        Before gathering, score mass from a forgotten token is transferred to
        the same retained slot that receives its merged KV representation.
        Then all state tensors are gathered with each layer's retained indices.
        """
        new_scores = []
        new_conn_scores = []
        new_masks = []

        for layer_idx in range(self.layer_num):
            retained = retained_idx_all[layer_idx]
            thrown = throw_idx_all[layer_idx]
            merge = merge_idx_all[layer_idx]

            layer_score = self.score_sum[layer_idx].clone()
            layer_conn = self.conn_score_sum[layer_idx].clone()

            # Transfer the importance carried by forgotten entries into the
            # retained representation that absorbs them.
            if thrown.numel() > 0:
                layer_score.scatter_add_(0, merge, layer_score[thrown])
                layer_conn.scatter_add_(0, merge, layer_conn[thrown])

            new_scores.append(layer_score[retained])
            new_conn_scores.append(layer_conn[retained])
            new_masks.append(self.image_mask[layer_idx, retained])

        self.score_sum = torch.stack(new_scores, dim=0)
        self.conn_score_sum = torch.stack(new_conn_scores, dim=0)
        self.image_mask = torch.stack(new_masks, dim=0)

    def _prune(self, past_key_values, forget_num):
        """Score, same-modality merge, gather, and realign all cache state."""
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # Candidate interval excludes protected beginning and final token.
        start = min(self.start_size, seq_len)
        end = max(start, seq_len - self.protect_size)
        candidate_count = max(0, end - start)

        if candidate_count == 0 or forget_num <= 0:
            return past_key_values

        # We cannot drop more than the score-based candidate interval.
        drop_count = min(int(forget_num), candidate_count)

        retained_idx_all = []
        throw_idx_all = []
        merge_idx_all = []

        device = self.score_sum.device

        for layer_idx in range(self.layer_num):
            combined = self._combined_scores(layer_idx, start, end)
            order = torch.argsort(combined)  # low score -> high score

            throw_rel = order[:drop_count]
            selected_rel = order[drop_count:]

            # If pruning would eliminate every image candidate, swap one back
            # in so image->image merging remains valid.
            selected_rel, throw_rel = self._ensure_image_survives(
                layer_idx,
                start,
                combined,
                selected_rel,
                throw_rel,
            )

            selected_candidate = selected_rel + start
            thrown = throw_rel + start

            # Protected entries are retained regardless of their scores.
            prefix = torch.arange(0, start, device=device, dtype=torch.long)
            suffix = torch.arange(end, seq_len, device=device, dtype=torch.long)

            retained = torch.cat(
                [prefix, selected_candidate, suffix], dim=0
            )

            # Restore physical sequence order.  The score ranking must NOT
            # reorder the cache, because positional order still matters.
            retained = torch.sort(retained).values

            merge = self._same_modality_merge_targets(
                layer_idx, thrown, retained
            )

            retained_idx_all.append(retained)
            throw_idx_all.append(thrown)
            merge_idx_all.append(merge)

        # Apply the per-layer selections to K/V.
        new_past = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_new, v_new = self._merge_and_gather_layer(
                k,
                v,
                layer_idx,
                retained_idx_all[layer_idx],
                throw_idx_all[layer_idx],
                merge_idx_all[layer_idx],
            )
            new_past.append([k_new, v_new])

        # Crucial: state arrays must now describe the NEW physical KV layout.
        self._realign_scores_and_masks(
            retained_idx_all, throw_idx_all, merge_idx_all
        )

        return new_past

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        """Update importance from this model call and prune to target size."""
        if past_key_values is None:
            return None

        if num_of_token is None:
            raise ValueError("num_of_token is required for ratio-based pruning.")

        attn = self._prepare_attention(attentions)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # Initialize the original image/text layout once, then append text-only
        # modality entries on all later calls.
        if not self.initialized:
            self._initialize_state(seq_len, attn.device)
        else:
            self._grow_state_for_new_text(seq_len)

        # Scores and masks now have exactly the same physical key dimension as
        # the model attention and KV cache.
        self._update_scores(attn)

        # num_of_token is the logical UNCOMPRESSED stream length maintained by
        # the evaluator.  The target grows as the conversation grows.
        target_cache_size = max(1, int(num_of_token * (1.0 - self.ratio)))
        forget_num = seq_len - target_cache_size

        if forget_num <= 0:
            return past_key_values

        return self._prune(past_key_values, forget_num)


class MultimodalElasticCache(_MultimodalCacheBase):
    """
    Corrected multi-turn MMElastic.

    Difference from MMHelastic:
      - MMElastic accumulates attention evidence over the entire conversation.
      - Old importance is NOT decayed.

    This is more stable/history-preserving, but less responsive to a shift in
    what later turns are discussing.
    """

    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        distance=-25,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=0,
    ):
        super().__init__(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=ratio,
            distance=distance,
            layer_num=layer_num,
            attn_weight=attn_weight,
            conn_weight=conn_weight,
            image_start=image_start,
            image_token_len=image_token_len,
            decay=None,
        )


class MultimodalHelasticCache(_MultimodalCacheBase):
    """
    Corrected multi-turn MMHelastic.

    Difference from MMElastic:
      - MMHelastic multiplies old importance by `decay` before adding the
        current attention evidence.
      - With decay=0.95, recent turns gradually outweigh stale early evidence.

    This is the dynamic/adaptive version for evolving multi-turn context.
    """

    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        distance=-25,
        layer_num=40,
        attn_weight=1.0,
        conn_weight=0.8,
        image_start=None,
        image_token_len=0,
        decay=0.95,
    ):
        super().__init__(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=ratio,
            distance=distance,
            layer_num=layer_num,
            attn_weight=attn_weight,
            conn_weight=conn_weight,
            image_start=image_start,
            image_token_len=image_token_len,
            decay=decay,
        )


class TurnIsolatedMultimodalCache(_MultimodalCacheBase):
    """MMElastic variant that compresses once per turn and freezes survivors.

    Calls to ``__call__`` only collect attention evidence. ``finalize_turn``
    independently compresses the active image/text segments (image exists only
    in turn 1), while entries retained from completed turns never compete or
    merge with entries from a later turn.
    """

    def __init__(self, *args, initial_protected_text_len=0, **kwargs):
        kwargs["decay"] = None
        super().__init__(*args, **kwargs)
        self.initial_protected_text_len = max(0, int(initial_protected_text_len))
        self.frozen_mask = None
        self.turn_index = 0
        self.last_finalize_stats = None

    def reset(self):
        super().reset()
        self.frozen_mask = None
        self.turn_index = 0
        self.last_finalize_stats = None

    def _initialize_state(self, seq_len, device):
        super()._initialize_state(seq_len, device)
        self.frozen_mask = torch.zeros(
            self.layer_num, seq_len, dtype=torch.bool, device=device
        )

    def _grow_state_for_new_text(self, seq_len):
        old_len = self.score_sum.shape[1]
        super()._grow_state_for_new_text(seq_len)
        add_len = seq_len - old_len
        if add_len > 0:
            self.frozen_mask = torch.cat(
                [
                    self.frozen_mask,
                    torch.zeros(
                        self.layer_num,
                        add_len,
                        dtype=torch.bool,
                        device=self.frozen_mask.device,
                    ),
                ],
                dim=1,
            )

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn = self._prepare_attention(attentions)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if not self.initialized:
            self._initialize_state(seq_len, attn.device)
        else:
            self._grow_state_for_new_text(seq_len)
        self._update_scores(attn)
        return past_key_values

    def _keep_count(self, count):
        if count == 0:
            return 0
        return max(1, int(count * (1.0 - self.ratio)))

    @staticmethod
    def _top_keep(indices, scores, keep_count):
        if keep_count == 0:
            return indices[:0]
        relative = torch.topk(scores, k=min(keep_count, indices.numel())).indices
        return torch.sort(indices[relative]).values

    @staticmethod
    def _nearest_targets(thrown, kept):
        if thrown.numel() == 0:
            return thrown.clone()
        if kept.numel() == 0:
            raise RuntimeError("Turn isolation cannot merge into an empty segment.")
        distances = torch.abs(thrown[:, None] - kept[None, :])
        return kept[torch.argmin(distances, dim=1)]

    def _select_image_keep(self, image_candidates, image_scores, keep_count):
        """Select active visual entries; base v2 uses attention importance."""
        return self._top_keep(image_candidates, image_scores, keep_count)

    def _visual_policy_name(self):
        return "turn1_attention_ranked" if self.turn_index == 0 else "already_frozen"

    def finalize_turn(self, past_key_values):
        """Compress the active turn once, then mark every survivor frozen."""
        if past_key_values is None or not self.initialized:
            return past_key_values
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if self.score_sum.shape[1] != seq_len:
            raise RuntimeError(
                f"Turn-finalization state length {self.score_sum.shape[1]} "
                f"does not match KV length {seq_len}."
            )

        retained_all, thrown_all, merge_all = [], [], []
        stats = None
        for layer_idx in range(self.layer_num):
            device = self.score_sum.device
            indices = torch.arange(seq_len, device=device)
            frozen = self.frozen_mask[layer_idx]
            image = self.image_mask[layer_idx]
            protected = torch.zeros(seq_len, dtype=torch.bool, device=device)
            if self.turn_index == 0:
                prefix_keep = max(
                    self.start_size, self.initial_protected_text_len
                )
            else:
                prefix_keep = self.start_size
            protected[: min(prefix_keep, seq_len)] = True
            active = ~frozen
            candidates = active & ~protected
            image_candidates = indices[candidates & image]
            text_candidates = indices[candidates & ~image]

            image_scores = self._normalize_nonnegative(
                self.score_sum[layer_idx, image_candidates]
            )
            text_scores = (
                self.attn_weight
                * self._normalize_nonnegative(
                    self.score_sum[layer_idx, text_candidates]
                )
                + self.conn_weight
                * self._normalize_nonnegative(
                    self.conn_score_sum[layer_idx, text_candidates]
                )
            )
            image_keep = self._select_image_keep(
                image_candidates, image_scores, self._keep_count(image_candidates.numel())
            )
            text_keep = self._top_keep(
                text_candidates, text_scores, self._keep_count(text_candidates.numel())
            )
            fixed_keep = indices[frozen | (active & protected)]
            retained = torch.unique(
                torch.cat([fixed_keep, image_keep, text_keep]), sorted=True
            )
            image_throw = image_candidates[~torch.isin(image_candidates, image_keep)]
            text_throw = text_candidates[~torch.isin(text_candidates, text_keep)]
            thrown = torch.cat([image_throw, text_throw])
            merge = torch.cat(
                [
                    self._nearest_targets(image_throw, image_keep),
                    self._nearest_targets(text_throw, text_keep),
                ]
            )
            if thrown.numel():
                order = torch.argsort(thrown)
                thrown, merge = thrown[order], merge[order]
            retained_all.append(retained)
            thrown_all.append(thrown)
            merge_all.append(merge)
            if stats is None:
                stats = {
                    "turn": self.turn_index + 1,
                    "kv_before": seq_len,
                    "frozen_before": int(frozen.sum().item()),
                    "protected_prefix": int(min(prefix_keep, seq_len)),
                    "visual_policy": self._visual_policy_name(),
                    "image_candidates": int(image_candidates.numel()),
                    "image_kept": int(image_keep.numel()),
                    "text_candidates": int(text_candidates.numel()),
                    "text_kept": int(text_keep.numel()),
                    "kv_after": int(retained.numel()),
                }

        new_past = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            key, value = self._merge_and_gather_layer(
                key,
                value,
                layer_idx,
                retained_all[layer_idx],
                thrown_all[layer_idx],
                merge_all[layer_idx],
            )
            new_past.append([key, value])
        self._realign_scores_and_masks(retained_all, thrown_all, merge_all)
        self.frozen_mask = torch.ones_like(self.image_mask)
        self.turn_index += 1
        self.last_finalize_stats = stats
        return new_past


class PersistentVisualTurnIsolatedCache(TurnIsolatedMultimodalCache):
    """Turn-isolated MMElastic with question-independent visual anchors.

    Turn 1 keeps uniformly spaced entries over the expanded visual-token block,
    merges discarded visual entries into their nearest retained visual anchor,
    and freezes those anchors for all later turns. Text selection remains the
    same attention-plus-grounding rule used by turn-isolated MMElastic.
    """

    @staticmethod
    def _coverage_keep(indices, keep_count):
        count = int(indices.numel())
        keep_count = int(keep_count)
        if count == 0 or keep_count <= 0:
            return indices[:0]
        if keep_count >= count:
            return indices.clone()
        if keep_count == 1:
            return indices[count // 2 : count // 2 + 1]

        positions = torch.linspace(
            0,
            count - 1,
            steps=keep_count,
            device=indices.device,
            dtype=torch.float32,
        ).round().long()
        positions = torch.unique(positions, sorted=True)
        if positions.numel() < keep_count:
            all_positions = torch.arange(count, device=indices.device)
            unused = all_positions[~torch.isin(all_positions, positions)]
            positions = torch.sort(
                torch.cat([positions, unused[: keep_count - positions.numel()]])
            ).values
        return indices[positions]

    def _select_image_keep(self, image_candidates, image_scores, keep_count):
        del image_scores
        return self._coverage_keep(image_candidates, keep_count)

    def _visual_policy_name(self):
        return (
            "uniform_persistent_coverage"
            if self.turn_index == 0
            else "already_frozen"
        )


class MultimodalElasticSimilarityCache(MultimodalElasticCache):
    """MMElastic with similarity-aware merging for BOTH image and text.

    This class intentionally keeps MMElastic's token-importance mechanism,
    retention budget, and retained/dropped-token selection unchanged.

    The only algorithmic change is the merge destination for dropped tokens:

        dropped image token -> most K-similar retained IMAGE token
        dropped text token  -> most K-similar retained TEXT token

    Similarity is computed within the same modality only; image/text
    cross-modality merging is never allowed.

    Similarity representation
    -------------------------
    For each token position, each layer's K tensor contains one vector per
    attention head. To form a compact representation we:
      1. L2-normalize each head's K vector;
      2. average the normalized vectors across heads;
      3. L2-normalize that average.

    Since the resulting vectors are normalized, matrix multiplication computes
    cosine similarity.
    """

    @staticmethod
    def _compact_key_representation(k):
        """Return one normalized compact K vector per sequence position.

        Expected LLaMA/LLaVA K shape: [batch=1, heads, seq, head_dim].
        """
        if k.dim() != 4:
            raise RuntimeError(
                f"Similarity merge expects 4-D K tensor, got shape {tuple(k.shape)}"
            )
        if k.shape[0] != 1:
            raise RuntimeError(
                "Similarity merge currently expects batch size 1, matching the "
                "per-conversation evaluator."
            )

        per_head = k[0].float()
        per_head = torch.nn.functional.normalize(per_head, p=2, dim=-1, eps=1e-6)
        compact = per_head.mean(dim=0)
        compact = torch.nn.functional.normalize(compact, p=2, dim=-1, eps=1e-6)
        return compact

    def _same_modality_similarity_targets(self, k, layer_idx, throw_idx, retained_idx):
        """Return cosine-similarity merge targets for both modalities.

        Image:
            dropped image -> retained image with maximum K cosine similarity.

        Text:
            dropped text -> retained text with maximum K cosine similarity.

        The returned tensor follows throw_idx order exactly.
        """
        if throw_idx.numel() == 0:
            return throw_idx.clone()

        layer_image = self.image_mask[layer_idx]
        merge_targets = torch.empty_like(throw_idx)
        compact_k = self._compact_key_representation(k)

        # TEXT similarity within retained text only.
        text_mask = ~layer_image[throw_idx]
        text_throw = throw_idx[text_mask]
        if text_throw.numel() > 0:
            retained_text = retained_idx[~layer_image[retained_idx]]
            if retained_text.numel() == 0:
                raise RuntimeError(
                    "No retained text KV entry is available for similarity merging."
                )
            thrown_repr = compact_k[text_throw]
            retained_repr = compact_k[retained_text]
            similarity = thrown_repr @ retained_repr.transpose(0, 1)
            best_relative = torch.argmax(similarity, dim=1)
            text_targets = retained_text[best_relative]
            merge_targets[text_mask] = text_targets

        # IMAGE similarity within retained image only.
        image_mask = layer_image[throw_idx]
        image_throw = throw_idx[image_mask]
        if image_throw.numel() > 0:
            retained_image = retained_idx[layer_image[retained_idx]]
            if retained_image.numel() == 0:
                raise RuntimeError(
                    "No retained image KV entry is available for similarity merging."
                )
            thrown_repr = compact_k[image_throw]
            retained_repr = compact_k[retained_image]
            similarity = thrown_repr @ retained_repr.transpose(0, 1)
            best_relative = torch.argmax(similarity, dim=1)
            image_targets = retained_image[best_relative]
            merge_targets[image_mask] = image_targets

        return merge_targets

    def _prune(self, past_key_values, forget_num):
        """MMElastic pruning with same-modality cosine merge destinations.

        Token scoring and retained/dropped selection are identical to
        MMElastic. Only the merge target changes: both text and image tokens
        merge into the most cosine-similar retained token of the same modality.
        """
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        start = min(self.start_size, seq_len)
        end = max(start, seq_len - self.protect_size)
        candidate_count = max(0, end - start)

        if candidate_count == 0 or forget_num <= 0:
            return past_key_values

        drop_count = min(int(forget_num), candidate_count)
        retained_idx_all = []
        throw_idx_all = []
        merge_idx_all = []
        device = self.score_sum.device

        # Retained/thrown entries are selected exactly as in MMElastic.
        for layer_idx in range(self.layer_num):
            combined = self._combined_scores(layer_idx, start, end)
            order = torch.argsort(combined)  # low score -> high score

            throw_rel = order[:drop_count]
            selected_rel = order[drop_count:]

            selected_rel, throw_rel = self._ensure_image_survives(
                layer_idx,
                start,
                combined,
                selected_rel,
                throw_rel,
            )

            selected_candidate = selected_rel + start
            thrown = throw_rel + start

            prefix = torch.arange(0, start, device=device, dtype=torch.long)
            suffix = torch.arange(end, seq_len, device=device, dtype=torch.long)
            retained = torch.cat([prefix, selected_candidate, suffix], dim=0)

            # Preserve physical sequence order after score-based selection.
            retained = torch.sort(retained).values

            k = past_key_values[layer_idx][0]
            merge = self._same_modality_similarity_targets(
                k,
                layer_idx,
                thrown,
                retained,
            )

            retained_idx_all.append(retained)
            throw_idx_all.append(thrown)
            merge_idx_all.append(merge)

        # Use the unchanged MMElastic merge/gather and state realignment.
        new_past = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_new, v_new = self._merge_and_gather_layer(
                k,
                v,
                layer_idx,
                retained_idx_all[layer_idx],
                throw_idx_all[layer_idx],
                merge_idx_all[layer_idx],
            )
            new_past.append([k_new, v_new])

        self._realign_scores_and_masks(
            retained_idx_all,
            throw_idx_all,
            merge_idx_all,
        )
        return new_past


class DelayedFirstCompressionSimilarityCache(MultimodalElasticSimilarityCache):
    """MMElastic similarity with the FIRST compression delayed until Turn 1 ends.

    During image+Q1 prefill and teacher-forced A1, the full KV cache is kept.
    Attention and text-image grounding metadata are still accumulated normally.
    At the end of Turn 1, ``finalize_turn`` performs the first compression using
    all evidence collected from Q1 and A1. After that, the cache behaves exactly
    like ``MultimodalElasticSimilarityCache`` on later turns.

    Similarity behavior is unchanged from the current similarity baseline:
        dropped image -> most K-cosine-similar retained image
        dropped text  -> most K-cosine-similar retained text
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_enabled = False
        self._latest_num_of_token = None

    def reset(self):
        super().reset()
        self.compression_enabled = False
        self._latest_num_of_token = None

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("num_of_token is required for ratio-based pruning.")

        attn = self._prepare_attention(attentions)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        if not self.initialized:
            self._initialize_state(seq_len, attn.device)
        else:
            self._grow_state_for_new_text(seq_len)

        # Always accumulate evidence, including during the uncompressed first turn.
        self._update_scores(attn)
        self._latest_num_of_token = int(num_of_token)

        # Turn 1: collect evidence only; do not prune yet.
        if not self.compression_enabled:
            return past_key_values

        # Turn 2+: behave exactly like the normal MMElastic similarity cache.
        target_cache_size = max(1, int(num_of_token * (1.0 - self.ratio)))
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values
        return self._prune(past_key_values, forget_num)

    def finalize_turn(self, past_key_values):
        """Perform the first compression once, at the end of Turn 1."""
        if past_key_values is None or not self.initialized:
            return past_key_values

        # Only the first turn boundary triggers delayed compression.
        if self.compression_enabled:
            return past_key_values

        if self._latest_num_of_token is None:
            raise RuntimeError(
                "Delayed compression has no logical token count from Turn 1."
            )

        self.compression_enabled = True
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        target_cache_size = max(
            1,
            int(self._latest_num_of_token * (1.0 - self.ratio)),
        )
        forget_num = seq_len - target_cache_size
        if forget_num <= 0:
            return past_key_values

        return self._prune(past_key_values, forget_num)


class DelayedFirstCompressionEvictionCache(
    DelayedFirstCompressionSimilarityCache
):
    """Delayed-first-compression ablation that evicts instead of merging.

    This preserves the delayed baseline's importance scoring, alpha/beta
    weighting, target ratio, protected positions, and retained-token selection.
    The only change is what happens after selection: dropped K/V entries and
    their score mass are discarded rather than merged into retained entries.
    """

    def _prune(self, past_key_values, forget_num):
        """Select exactly as the similarity baseline, then gather survivors."""
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        start = min(self.start_size, seq_len)
        end = max(start, seq_len - self.protect_size)
        candidate_count = max(0, end - start)

        if candidate_count == 0 or forget_num <= 0:
            return past_key_values

        drop_count = min(int(forget_num), candidate_count)
        retained_idx_all = []
        device = self.score_sum.device

        for layer_idx in range(self.layer_num):
            combined = self._combined_scores(layer_idx, start, end)
            order = torch.argsort(combined)  # low score -> high score

            throw_rel = order[:drop_count]
            selected_rel = order[drop_count:]

            # Preserve the baseline's retained/dropped-token selection exactly.
            selected_rel, throw_rel = self._ensure_image_survives(
                layer_idx,
                start,
                combined,
                selected_rel,
                throw_rel,
            )

            selected_candidate = selected_rel + start
            prefix = torch.arange(0, start, device=device, dtype=torch.long)
            suffix = torch.arange(end, seq_len, device=device, dtype=torch.long)
            retained = torch.cat([prefix, selected_candidate, suffix], dim=0)

            # Preserve physical sequence order after score-based selection.
            retained_idx_all.append(torch.sort(retained).values)

        # Pure eviction: gather retained K/V entries without altering them.
        new_past = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            retained = retained_idx_all[layer_idx]
            k_new = k.index_select(self.k_seq_dim, retained.to(k.device))
            v_new = v.index_select(self.v_seq_dim, retained.to(v.device))
            new_past.append([k_new, v_new])

        # Discard dropped score state too; do not transfer it to survivors.
        self.score_sum = torch.stack(
            [
                self.score_sum[layer_idx, retained_idx_all[layer_idx]]
                for layer_idx in range(self.layer_num)
            ],
            dim=0,
        )
        self.conn_score_sum = torch.stack(
            [
                self.conn_score_sum[layer_idx, retained_idx_all[layer_idx]]
                for layer_idx in range(self.layer_num)
            ],
            dim=0,
        )
        self.image_mask = torch.stack(
            [
                self.image_mask[layer_idx, retained_idx_all[layer_idx]]
                for layer_idx in range(self.layer_num)
            ],
            dim=0,
        )

        return new_past
