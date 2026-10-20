# """
# Turn-conditioned coverage cache: anticipatory KV retention for multi-turn VLMs.

# Built on ``TurnConditionedLayerwiseKVCache`` (kv_cache_final.py).  Layer-budget
# allocation (uniform / turn / sensitivity) and provenance tracking are reused
# unchanged; what changes is WHEN the cache compresses and WHICH entries each
# layer keeps.

# Why
# ---
# In a continuous conversation the cache is compressed before the next question
# is known.  Keeping only what past queries attended to (H2O / SnapKV / PyramidKV
# / TGV style) is myopic: Turn 3 may ask about a region of the image that Turns 1
# and 2 never looked at.  The base method has three further issues:

#   * it does not compress during Turn 1, so it exceeds the budget there;
#   * its score is cumulative attention from ALL queries, dominated by the
#     image tokens attending to each other during prefill, and biased toward
#     old tokens (more observers) against the turn that just ended;
#   * only the very last token is protected.

# And every eviction method shares one more: dropping a visual token also drops
# its share of the softmax mass, so the attention that remains shifts toward
# text and sink tokens.  The model then sees a "smaller" image than it was
# trained on; this hurts counting most, and colour/position reads as well.

# What
# ----
# (A) Compress right after the Turn-1 multimodal prefill (respects the budget
#     for the whole conversation), and again at every turn boundary.
# (B) Debiased relevance: for every KV entry, the AVERAGE attention lift it
#     received per TEXT query that could see it.  Lift = attention probability
#     x number of keys that query could see (uniform attention gives 1), so
#     early queries, which see few keys and hand out large probabilities, do
#     not inflate the tokens before the image.  Image queries are excluded and
#     heads are averaged.
# (C) Anticipatory coverage selection.  Each candidate i gets an expected future
#     attention weight

#         w_i = (1 - alpha) * relevance_i / sum(relevance) + alpha / N

#     (alpha = probability mass reserved for questions not seen yet).  Each layer
#     keeps the set S (|S| = its budget) that maximises the relevance-weighted
#     facility-location objective

#         F(S) = sum_i  w_i * max_{j in S} s(i, j)

#     where s(i, i) = 1, s(image, image) = exp(-(1 - cos(k_i, k_j)) / h_l) with a
#     per-layer adaptive bandwidth h_l (kappa x median nearest-neighbour key
#     distance), and s = 0 otherwise (text entries only stand for themselves).
#     F is monotone submodular, so the greedy solution is within (1 - 1/e) of
#     optimal.  The same greedy decides how a layer's budget is split between
#     visual and text entries: a visual anchor wins when it newly "covers" more
#     expected attention mass than the best remaining text entry.
# (D) Protected entries: the attention-sink prefix, the ``recent_window`` most
#     recent TEXT entries (never the image tail; shrinks to budget // 4 at very
#     small budgets), and (at the Turn-1 prefill only) the current turn's text
#     up to ``text_prior_frac`` of the layer budget.
# (E) Layer floor: every layer is guaranteed ``layer_floor`` x the uniform
#     per-layer budget, and only the rest follows the configured allocation
#     (turn / sensitivity).  A sensitivity profile is probed at one ratio; a
#     layer it rates as insensitive (layer 0 in the LLaVA profiles) would
#     otherwise be cut to a handful of entries, and with (A) that already
#     happens before the Turn-1 answer.
# (F) Mass-preserving eviction (``mass_preserving``, default on).  Each evicted
#     visual entry hands its mass to the kept visual anchor most similar to it,
#     and an anchor standing for m original entries gets +log(m) on its
#     attention logit (proportional attention, as in ToMe):

#         softmax(q.k_j + log m_j) == attention over m identical copies of k_j

#     Mass moves only to an anchor with kernel similarity >= ``mass_min_sim``
#     (default 0.25); 0 conserves the image's total mass exactly, which was
#     best at ratio 0.5 but over-weights dissimilar anchors at ratio 0.8 on the
#     synthetic test.  Memory cost: one float per kept entry.  Needs ``attach_model(model)`` and an attention
#     forward that adds ``self_attn._kv_logit_bias`` to the key logits (the
#     tgv_kv_legacy_nova.py edit).  Masses also enter the uniform prior in (C).
# (G) Optional: each evicted visual entry is also averaged into that anchor's
#     key/value (``merge_visual``; mass-weighted when (F) is on).

# alpha = 0 and kappa -> 0 reduces (C) to a relevance top-k; the ablation flags
# below switch every component off independently.
# """

# import math

# import torch
# import torch.nn.functional as F

# from kv_cache_final import TurnConditionedLayerwiseKVCache


# class TurnConditionedCoverageKVCache(TurnConditionedLayerwiseKVCache):
#     """Turn-conditioned layer budgets + anticipatory coverage token selection."""

#     def __init__(
#         self,
#         *args,
#         coverage_alpha=0.5,
#         coverage_kappa=1.0,
#         similarity="key",
#         recent_window=8,
#         text_prior_frac=0.5,
#         prefill_compress=True,
#         relevance="text_mean",
#         merge_visual=False,
#         layer_floor=0.5,
#         floor_stage="all",
#         mass_preserving=True,
#         mass_min_sim=0.25,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.mass_preserving = bool(mass_preserving)
#         self.mass_min_sim = float(mass_min_sim)
#         if not 0.0 <= self.mass_min_sim <= 1.0:
#             raise ValueError("mass_min_sim must be in [0, 1]")
#         self._attn_modules = None
#         self.layer_floor = float(layer_floor)
#         self.floor_stage = str(floor_stage)
#         if not 0.0 <= self.layer_floor <= 1.0:
#             raise ValueError("layer_floor must be in [0, 1]")
#         if self.floor_stage not in {"all", "prefill"}:
#             raise ValueError("floor_stage must be 'all' or 'prefill'")
#         self.coverage_alpha = float(coverage_alpha)
#         self.coverage_kappa = float(coverage_kappa)
#         self.similarity = str(similarity)
#         self.recent_window = max(1, int(recent_window))
#         self.text_prior_frac = float(text_prior_frac)
#         self.prefill_compress = bool(prefill_compress)
#         self.relevance = str(relevance)
#         self.merge_visual = bool(merge_visual)
#         if not 0.0 <= self.coverage_alpha <= 1.0:
#             raise ValueError("coverage_alpha must be in [0, 1]")
#         if self.coverage_kappa < 0:
#             raise ValueError("coverage_kappa must be non-negative")
#         if self.similarity not in {"key", "value", "key_value"}:
#             raise ValueError("similarity must be 'key', 'value' or 'key_value'")
#         if self.relevance not in {"text_mean", "cumulative"}:
#             raise ValueError("relevance must be 'text_mean' or 'cumulative'")
#         if not 0.0 <= self.text_prior_frac <= 1.0:
#             raise ValueError("text_prior_frac must be in [0, 1]")
#         # The base allocation's per-layer lower bound is start_size + protect_size.
#         self.protect_size = self.recent_window

#     # ------------------------------------------------------------------
#     # Extra per-entry state, kept in the same spare-slot buffers as the base
#     # ------------------------------------------------------------------
#     # Extra buffers, all zero for a new entry:
#     #   0 text_attn  summed attention lift from text queries
#     #   1 text_obs   number of text queries that could see the entry
#     #   2 log_mass   log of how many original entries this entry stands for
#     TEXT_ATTN, TEXT_OBS, LOG_MASS = 0, 1, 2

#     def reset(self):
#         super().reset()
#         self._extra_buffers = None
#         self._extra_views = None
#         self._pending_prefill = False
#         self._prefill_done = False
#         self._set_attention_bias(None)

#     def attach_model(self, model):
#         """Give the cache the attention modules that read ``_kv_logit_bias``.

#         Only needed with ``mass_preserving``; the patched legacy LLaMA
#         attention adds the bias to the logits of the first cached keys.
#         """
#         inner = model.get_model() if hasattr(model, "get_model") else model.model
#         modules = [layer.self_attn for layer in inner.layers]
#         if self.mass_preserving and not getattr(type(modules[0]).forward, "supports_kv_logit_bias", False):
#             raise RuntimeError(
#                 "mass_preserving needs an attention forward that adds self_attn._kv_logit_bias "
#                 "to the key logits (see the tgv_kv_legacy_nova.py edit), or pass "
#                 "--coverage-no-mass"
#             )
#         self._attn_modules = modules
#         self._set_attention_bias(None)

#     def _set_attention_bias(self, per_layer):
#         if not getattr(self, "_attn_modules", None):
#             return
#         for layer_idx, module in enumerate(self._attn_modules):
#             module._kv_logit_bias = None if per_layer is None else per_layer[layer_idx]

#     def _initialize(self, lengths, device):
#         self._extra_buffers = [None] * self.layer_num
#         self._extra_views = [None] * self.layer_num
#         super()._initialize(lengths, device)

#     def _store_layer(self, layer_idx, score, conn, provenance, min_capacity=0, extras=None):
#         if extras is None:
#             if self._extra_views is not None and self._extra_views[layer_idx] is not None:
#                 extras = self._extra_views[layer_idx]
#             else:
#                 zeros = torch.zeros(score.numel(), dtype=torch.float32, device=score.device)
#                 extras = (zeros, zeros, zeros)
#         super()._store_layer(layer_idx, score, conn, provenance, min_capacity)
#         capacity = self._buffers[layer_idx][0].numel()
#         length = score.numel()
#         buffers = []
#         for values in extras:
#             buf = torch.zeros(capacity, dtype=torch.float32, device=score.device)
#             buf[: values.numel()] = values
#             buffers.append(buf)
#         self._extra_buffers[layer_idx] = tuple(buffers)
#         self._extra_views[layer_idx] = tuple(b[:length] for b in buffers)

#     def _grow_layer(self, layer_idx, length):
#         super()._grow_layer(layer_idx, length)
#         self._extra_views[layer_idx] = tuple(b[:length] for b in self._extra_buffers[layer_idx])

#     # ------------------------------------------------------------------
#     # Observation
#     # ------------------------------------------------------------------
#     def observe_layer(self, layer_idx, attention):
#         first_prefill = self.score_sum is None
#         super().observe_layer(layer_idx, attention)  # base stats + growth

#         attn = attention[0].mean(dim=0, dtype=torch.float32)  # [queries, keys]
#         query_len, key_len = attn.shape
#         text_attn = self._extra_views[layer_idx][self.TEXT_ATTN]
#         text_obs = self._extra_views[layer_idx][self.TEXT_OBS]
#         # Attention is measured as LIFT over uniform (probability x number of
#         # visible keys): a query that sees few keys hands out large
#         # probabilities, so raw probabilities would favour tokens that are
#         # mostly observed by early queries (e.g. text before the image).
#         if query_len == 1:
#             # Decode step: the only query is text appended in the current turn.
#             text_attn.add_(attn[0] * key_len)
#             text_obs.add_(1.0)
#             return

#         query_start = key_len - query_len
#         is_text = (self.provenance[layer_idx][query_start:] != self.IMAGE).float()
#         visible = torch.arange(query_start + 1, key_len + 1, device=attn.device, dtype=torch.float32)
#         text_attn.add_((is_text * visible) @ attn)
#         # Observers of key j = text queries at positions >= j (causal mask).
#         later = torch.flip(torch.cumsum(torch.flip(is_text, [0]), 0), [0])
#         if query_start > 0:
#             text_obs[:query_start].add_(is_text.sum())
#         text_obs[query_start:].add_(later)
#         if first_prefill or not self._prefill_done:
#             self._pending_prefill = True

#     # ------------------------------------------------------------------
#     # Selection
#     # ------------------------------------------------------------------
#     def _relevance(self, layer_idx):
#         if self.relevance == "cumulative":
#             return self.score_sum[layer_idx].float().clamp_min(0)
#         text_attn = self._extra_views[layer_idx][self.TEXT_ATTN]
#         text_obs = self._extra_views[layer_idx][self.TEXT_OBS]
#         return torch.where(text_obs > 0, text_attn / text_obs.clamp_min(1.0), torch.zeros_like(text_attn))

#     def _features(self, k, v, image_idx):
#         """Per-head centred, normalised features flattened so that f f^T = mean-head cosine."""
#         parts = []
#         sources = {"key": (k,), "value": (v,), "key_value": (k, v)}[self.similarity]
#         for t in sources:
#             x = t[0].index_select(1, image_idx).float()          # [H, n, D]
#             x = x - x.mean(dim=1, keepdim=True)
#             x = F.normalize(x, dim=-1, eps=1e-6)
#             parts.append(x.permute(1, 0, 2).reshape(image_idx.numel(), -1) / math.sqrt(x.shape[0]))
#         feats = torch.cat(parts, dim=-1) / math.sqrt(len(parts))
#         return feats                                              # [n, H*D*len(parts)]

#     def _similarity(self, feats):
#         cos = (feats @ feats.T).clamp(-1.0, 1.0)
#         n = cos.shape[0]
#         if self.coverage_kappa == 0 or n < 2:
#             return torch.eye(n, device=cos.device)
#         dist = 1.0 - cos
#         nn_dist = dist.masked_fill(torch.eye(n, dtype=torch.bool, device=dist.device), float("inf")).min(dim=1).values
#         bandwidth = (self.coverage_kappa * nn_dist.median()).clamp_min(1e-4)
#         sim = torch.exp(-dist.clamp_min(0) / bandwidth)
#         sim.fill_diagonal_(1.0)
#         return sim

#     def _protected(self, layer_idx, budget, stage, relevance):
#         length = self.score_sum[layer_idx].numel()
#         device = relevance.device
#         keep = torch.zeros(length, dtype=torch.bool, device=device)
#         keep[: min(self.start_size, length)] = True
#         # Recency is a property of the conversation, so the window covers the
#         # most recent TEXT entries (never the tail of the image), and shrinks
#         # at very small budgets.
#         window = min(self.recent_window, max(1, int(budget) // 4))
#         text_idx = (self.provenance[layer_idx] != self.IMAGE).nonzero(as_tuple=True)[0]
#         keep[text_idx[-window:]] = True
#         keep[length - 1] = True
#         if stage == "prefill" and self.text_prior_frac > 0:
#             text = (self.provenance[layer_idx] == self.current_turn) & ~keep
#             cap = int(self.text_prior_frac * budget) - int(keep.sum())
#             idx = text.nonzero(as_tuple=True)[0]
#             if cap > 0 and idx.numel() > 0:
#                 if idx.numel() > cap:
#                     idx = idx[torch.topk(relevance[idx], cap).indices]
#                 keep[idx] = True
#         return keep

#     def _layer_problem(self, layer_idx, budget, k, v, stage):
#         """Greedy inputs for one layer: protected mask, weights, visual similarity."""
#         length = self.score_sum[layer_idx].numel()
#         device = self.score_sum[layer_idx].device
#         relevance = self._relevance(layer_idx)
#         keep = self._protected(layer_idx, budget, stage, relevance)
#         candidate = ~keep
#         weights = torch.zeros(length, dtype=torch.float32, device=device)
#         n_cand = int(candidate.sum())
#         if n_cand > 0:
#             rel = relevance[candidate].clamp_min(0)
#             total = rel.sum()
#             # Uniform prior over ORIGINAL entries: an entry that stands for m
#             # merged entries carries m shares (m = 1 without mass_preserving).
#             mass = self._extra_views[layer_idx][self.LOG_MASS][candidate].exp()
#             prior = mass / mass.sum()
#             rel_part = torch.where(total > 0, rel / total.clamp_min(1e-30), prior)
#             weights[candidate] = (1.0 - self.coverage_alpha) * rel_part + self.coverage_alpha * prior
#         image_idx = (self.provenance[layer_idx] == self.IMAGE).nonzero(as_tuple=True)[0]
#         sim = None
#         if image_idx.numel() > 0:
#             sim = self._similarity(self._features(k, v, image_idx))
#         keep_count = int(keep.sum())
#         quota = (length - keep_count) if budget >= length else max(0, int(budget) - keep_count)
#         return {"length": length, "keep": keep, "weights": weights, "image_idx": image_idx,
#                 "sim": sim, "quota": quota}

#     def _select_all(self, past_key_values, budgets, stage):
#         """Batched greedy facility location over all layers at once.

#         One fixed-length loop; each step is a handful of [L, N, N] tensor ops
#         and there is no host synchronisation inside the loop.
#         """
#         problems = [
#             self._layer_problem(l, int(budgets[l]), k, v, stage)
#             for l, (k, v) in enumerate(past_key_values)
#         ]
#         layers = len(problems)
#         n_max = max(p["length"] for p in problems)
#         ni_max = max(1, max(p["image_idx"].numel() for p in problems))
#         device = problems[0]["weights"].device
#         # Visual block: only visual entries can stand in for one another.
#         sim = torch.zeros(layers, ni_max, ni_max, dtype=torch.float32, device=device)
#         w_img = torch.zeros(layers, ni_max, dtype=torch.float32, device=device)
#         img_sel = torch.zeros(layers, ni_max, dtype=torch.bool, device=device)
#         img_free = torch.zeros(layers, ni_max, dtype=torch.bool, device=device)
#         # Text entries only stand for themselves: their gain is their weight.
#         txt_gain = torch.full((layers, n_max), -1.0, dtype=torch.float32, device=device)
#         selected = torch.zeros(layers, n_max, dtype=torch.bool, device=device)
#         img_pos = torch.zeros(layers, ni_max, dtype=torch.long, device=device)
#         quota = torch.zeros(layers, dtype=torch.long, device=device)
#         for l, p in enumerate(problems):
#             n = p["length"]
#             keep = p["keep"]
#             selected[l, :n] = keep
#             image = torch.zeros(n, dtype=torch.bool, device=device)
#             ii = p["image_idx"]
#             image[ii] = True
#             text_free = ~keep & ~image
#             txt_gain[l, :n] = torch.where(text_free, p["weights"], torch.full_like(p["weights"], -1.0))
#             ni = ii.numel()
#             if ni:
#                 sim[l, :ni, :ni] = p["sim"]
#                 w_img[l, :ni] = p["weights"][ii]
#                 img_sel[l, :ni] = keep[ii]
#                 img_free[l, :ni] = ~keep[ii]
#                 img_pos[l, :ni] = ii
#             quota[l] = p["quota"]

#         # cover[l, i] = max over selected visual j of sim[l, i, j]
#         cover = (sim * img_sel.unsqueeze(1)).amax(dim=2)
#         rows = torch.arange(layers, device=device)
#         # Text gains never change, so the greedy takes text in descending weight
#         # order; every text entry that beats the best visual gain is taken in
#         # one step (identical result to one-at-a-time greedy, fewer steps).
#         txt_sorted, txt_order = txt_gain.sort(dim=1, descending=True)
#         neg_sorted = (-txt_sorted).contiguous()
#         txt_count = (txt_sorted > -1.0).sum(dim=1)
#         ptr = torch.zeros(layers, dtype=torch.long, device=device)
#         span = torch.arange(n_max, device=device).unsqueeze(0)
#         max_steps = max(p["quota"] for p in problems)
#         for step in range(max_steps):
#             if step % 16 == 15 and not bool((quota > 0).any()):
#                 break
#             gain_img = (w_img.unsqueeze(2) * (sim - cover.unsqueeze(2)).clamp_min(0)).sum(dim=1)
#             gain_img = gain_img.masked_fill(~img_free, -1.0)
#             gi_val, gi = gain_img.max(dim=1)
#             next_txt = torch.where(ptr < txt_count, txt_sorted[rows, ptr.clamp_max(n_max - 1)],
#                                    torch.full_like(gi_val, -1.0))
#             active = quota > 0
#             take_img = active & (gi_val > -1.0) & (gi_val >= next_txt)
#             take_txt = active & ~take_img & (next_txt > -1.0)
#             # Text block: all remaining text with weight strictly above the best visual gain.
#             stop = torch.searchsorted(neg_sorted, (-gi_val).unsqueeze(1), side="left").squeeze(1)
#             n_txt = torch.minimum((stop - ptr).clamp_min(0), quota)
#             n_txt = torch.where(take_txt, n_txt.clamp_min(1), torch.zeros_like(n_txt))
#             block = (span >= ptr.unsqueeze(1)) & (span < (ptr + n_txt).unsqueeze(1))
#             selected.scatter_(1, txt_order, block | selected.gather(1, txt_order))
#             ptr = ptr + n_txt
#             # Visual pick.
#             img_free[rows, gi] &= ~take_img
#             img_sel[rows, gi] |= take_img
#             selected[rows, img_pos[rows, gi]] |= take_img
#             cover = torch.where(take_img.unsqueeze(1), torch.maximum(cover, sim[rows, :, gi]), cover)
#             quota = quota - n_txt - take_img.long()

#         out = []
#         for l, p in enumerate(problems):
#             n = p["length"]
#             sel = selected[l, :n]
#             retained = sel.nonzero(as_tuple=True)[0]
#             src = dst = retained[:0]
#             if (self.merge_visual or self.mass_preserving) and p["sim"] is not None:
#                 kept_img = sel[p["image_idx"]]
#                 drop_img = ~kept_img
#                 if bool(kept_img.any()) and bool(drop_img.any()):
#                     best, assign = p["sim"][drop_img][:, kept_img].max(dim=1)
#                     src = p["image_idx"][drop_img]
#                     dst = p["image_idx"][kept_img][assign]
#                     if self.mass_preserving:
#                         # Only an anchor similar enough to an entry absorbs it
#                         # (mass_min_sim = 0: all image mass is conserved).  Zero
#                         # similarity (kappa = 0) never transfers.
#                         covered = (best >= self.mass_min_sim) & (best > 0)
#                         src, dst = src[covered], dst[covered]
#             out.append((retained, src, dst))
#         return out

#     # ------------------------------------------------------------------
#     # Compression
#     # ------------------------------------------------------------------
#     def _stage_budgets(self, stage):
#         """Base layer allocation, mixed with a uniform share of the budget.

#         Every layer is guaranteed about ``layer_floor`` x the uniform per-layer
#         budget; the rest follows the configured allocation signal.  The same
#         hedge as ``coverage_alpha``, applied to layers: a sensitivity profile
#         probed at one ratio says little about a layer cut far below it.
#         """
#         scores, budgets, uniform_target = self._turn_budgets()
#         if self.layer_floor <= 0 or (self.floor_stage == "prefill" and stage != "prefill"):
#             return scores, budgets, uniform_target
#         lengths = torch.tensor([x.numel() for x in self.score_sum], device=scores.device)
#         lower = torch.tensor(
#             [min(int(n), self.start_size + self.protect_size) for n in lengths.tolist()],
#             device=scores.device,
#         )
#         weights = scores.double().clamp_min(0)
#         if weights.sum() > 0:
#             weights = weights / weights.sum()
#         else:
#             weights = torch.full_like(weights, 1.0 / weights.numel())
#         mixed = (1.0 - self.layer_floor) * weights + self.layer_floor / weights.numel()
#         budgets = self._bounded_largest_remainder(
#             uniform_target * self.layer_num, mixed, lower, lengths
#         )
#         return mixed.to(scores.dtype), budgets, uniform_target

#     def _compress(self, past_key_values, stage):
#         if past_key_values is None or self.score_sum is None:
#             return past_key_values
#         if self.mass_preserving and not self._attn_modules:
#             raise RuntimeError(
#                 "mass_preserving needs kv_cache.attach_model(model) and an attention "
#                 "forward that adds self_attn._kv_logit_bias to the key logits"
#             )
#         scores, budgets, uniform_target = self._stage_budgets(stage)
#         policy = "coverage_merge" if self.merge_visual else "coverage"
#         policy += "_mass" if self.mass_preserving else ""
#         lengths = [x.numel() for x in self.score_sum]
#         if all(int(budgets[l]) >= lengths[l] for l in range(len(lengths))):
#             # Nothing to evict (e.g. ratio 0): same result as the greedy keeping all.
#             self._finish_finalize(scores, budgets, uniform_target, policy)
#             self.last_finalize_stats["stage"] = stage
#             self.last_finalize_stats["image_kept_per_layer"] = [
#                 int((p == self.IMAGE).sum()) for p in self.provenance
#             ]
#             return past_key_values
#         selections = self._select_all(past_key_values, budgets, stage)
#         new_past = []
#         new_state = []
#         image_kept = []
#         for layer_idx, (k, v) in enumerate(past_key_values):
#             retained, src, dst = selections[layer_idx]
#             dev = k.device
#             retained_d = retained.to(dev)
#             k_new = k.index_select(2, retained_d)
#             v_new = v.index_select(2, retained_d)
#             views = self._extra_views[layer_idx]
#             extras = [x[retained] for x in views]
#             if src.numel() > 0:
#                 pos = torch.full((k.shape[2],), -1, dtype=torch.long, device=dev)
#                 pos[retained_d] = torch.arange(retained.numel(), device=dev)
#                 target = pos[dst.to(dev)]
#                 src_d = src.to(dev)
#                 index = target.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
#                 mass_all = views[self.LOG_MASS].exp().to(dev)
#                 if self.merge_visual and self.mass_preserving:
#                     # Mass-weighted mean of each anchor and the entries it absorbs.
#                     shape = (1, 1, -1, 1)
#                     m_keep, m_src = mass_all[retained_d], mass_all[src_d]
#                     den = m_keep.scatter_add(0, target, m_src)
#                     k_sum = (k_new.float() * m_keep.view(shape)).scatter_add(
#                         2, index, k.index_select(2, src_d).float() * m_src.view(shape))
#                     v_sum = (v_new.float() * m_keep.view(shape)).scatter_add(
#                         2, index, v.index_select(2, src_d).float() * m_src.view(shape))
#                     k_new = (k_sum / den.view(shape)).to(k.dtype)
#                     v_new = (v_sum / den.view(shape)).to(v.dtype)
#                 elif self.merge_visual:
#                     # Average each dropped visual entry into its covering anchor.
#                     k_new = k_new.scatter_reduce(2, index, k.index_select(2, src_d), reduce="mean", include_self=True)
#                     v_new = v_new.scatter_reduce(2, index, v.index_select(2, src_d), reduce="mean", include_self=True)
#                 if self.mass_preserving:
#                     # The anchor now stands for its own mass plus the absorbed mass.
#                     mass = mass_all[retained_d].scatter_add(0, target, mass_all[src_d])
#                     extras[self.LOG_MASS] = mass.log().to(views[self.LOG_MASS].device)
#             new_past.append([k_new, v_new])
#             new_state.append((
#                 self.score_sum[layer_idx][retained],
#                 self.conn_score_sum[layer_idx][retained],
#                 self.provenance[layer_idx][retained],
#                 tuple(extras),
#             ))
#             image_kept.append(int((self.provenance[layer_idx][retained] == self.IMAGE).sum()))
#         for layer_idx, (score, conn, prov, extras) in enumerate(new_state):
#             self._store_layer(layer_idx, score, conn, prov, extras=extras)
#         if self.mass_preserving:
#             # Proportional attention: logit + log(mass) on the first cached keys.
#             self._set_attention_bias([
#                 views[self.LOG_MASS].clone() if bool((views[self.LOG_MASS] != 0).any()) else None
#                 for views in self._extra_views
#             ])
#         self._finish_finalize(scores, budgets, uniform_target, policy)
#         self.last_finalize_stats["stage"] = stage
#         self.last_finalize_stats["image_kept_per_layer"] = image_kept
#         return new_past

#     # ------------------------------------------------------------------
#     # Evaluator interface
#     # ------------------------------------------------------------------
#     def __call__(self, past_key_values, num_of_token=None, attentions=None):
#         past_key_values = super().__call__(past_key_values, num_of_token, attentions)
#         if self._pending_prefill:
#             self._pending_prefill = False
#             self._prefill_done = True
#             if self.prefill_compress:
#                 return self._compress(past_key_values, stage="prefill")
#         return past_key_values

#     def finalize_turn(self, past_key_values):
#         return self._compress(past_key_values, stage="boundary")


"""
Turn-conditioned coverage cache: anticipatory KV retention for multi-turn VLMs.

Built on ``TurnConditionedLayerwiseKVCache`` (kv_cache_final.py).  Layer-budget
allocation (uniform / turn / sensitivity) and provenance tracking are reused
unchanged; what changes is WHEN the cache compresses and WHICH entries each
layer keeps.

Why
---
In a continuous conversation the cache is compressed before the next question
is known.  Keeping only what past queries attended to (H2O / SnapKV / PyramidKV
/ TGV style) is myopic: Turn 3 may ask about a region of the image that Turns 1
and 2 never looked at.  The base method has three further issues:

  * it does not compress during Turn 1, so it exceeds the budget there;
  * its score is cumulative attention from ALL queries, dominated by the
    image tokens attending to each other during prefill, and biased toward
    old tokens (more observers) against the turn that just ended;
  * only the very last token is protected.

And every eviction method shares one more: dropping a visual token also drops
its share of the softmax mass, so the attention that remains shifts toward
text and sink tokens.  The model then sees a "smaller" image than it was
trained on; this hurts counting most, and colour/position reads as well.

What
----
(A) Compress right after the Turn-1 multimodal prefill (respects the budget
    for the whole conversation), and again at every turn boundary.
(B) Debiased relevance: for every KV entry, the AVERAGE attention lift it
    received per TEXT query that could see it.  Lift = attention probability
    x number of keys that query could see (uniform attention gives 1), so
    early queries, which see few keys and hand out large probabilities, do
    not inflate the tokens before the image.  Image queries are excluded and
    heads are averaged.
(C) Anticipatory coverage selection.  Each candidate i gets an expected future
    attention weight

        w_i = (1 - alpha) * relevance_i / sum(relevance) + alpha / N

    (alpha = probability mass reserved for questions not seen yet).  Each layer
    keeps the set S (|S| = its budget) that maximises the relevance-weighted
    facility-location objective

        F(S) = sum_i  w_i * max_{j in S} s(i, j)

    where s(i, i) = 1, s(image, image) = exp(-(1 - cos(k_i, k_j)) / h_l) with a
    per-layer adaptive bandwidth h_l (kappa x median nearest-neighbour key
    distance), and s = 0 otherwise (text entries only stand for themselves).
    F is monotone submodular, so the greedy solution is within (1 - 1/e) of
    optimal.  The same greedy decides how a layer's budget is split between
    visual and text entries: a visual anchor wins when it newly "covers" more
    expected attention mass than the best remaining text entry.
(D) Protected entries: the attention-sink prefix, the ``recent_window`` most
    recent TEXT entries (never the image tail; shrinks to budget // 4 at very
    small budgets), and (at the Turn-1 prefill only) the current turn's text
    up to ``text_prior_frac`` of the layer budget.
(E) Layer floor: every layer is guaranteed ``layer_floor`` x the uniform
    per-layer budget, and only the rest follows the configured allocation
    (turn / sensitivity).  A sensitivity profile is probed at one ratio; a
    layer it rates as insensitive (layer 0 in the LLaVA profiles) would
    otherwise be cut to a handful of entries, and with (A) that already
    happens before the Turn-1 answer.
(F) Mass-preserving eviction (``mass_preserving``, default on).  Each evicted
    visual entry hands its mass to the kept visual anchor most similar to it,
    and an anchor standing for m original entries gets +log(m) on its
    attention logit (proportional attention, as in ToMe):

        softmax(q.k_j + log m_j) == attention over m identical copies of k_j

    Mass moves only to an anchor with kernel similarity >= ``mass_min_sim``
    (default 0.25); 0 conserves the image's total mass exactly, which was
    best at ratio 0.5 but over-weights dissimilar anchors at ratio 0.8 on the
    synthetic test.  Memory cost: one float per kept entry.  Needs ``attach_model(model)`` and an attention
    forward that adds ``self_attn._kv_logit_bias`` to the key logits (the
    tgv_kv_legacy_nova.py edit).  Masses also enter the uniform prior in (C).
(G) Optional: each evicted visual entry is also averaged into that anchor's
    key/value (``merge_visual``; mass-weighted when (F) is on).
(H) Automatic alpha (``alpha_auto``; otherwise ``coverage_alpha`` is used).
    alpha says how far to trust the relevance ranking for the next turn.  It
    is measured as the conversation goes: after every turn, over the entries
    that were evictable at the previous compression (image and text), with
    K = the number of entries a compression keeps,

        overlap = |top-K by this turn's attention  &  top-K by earlier relevance| / K
        novelty = (1 - overlap) / (1 - K / n)          (0 = as predicted, 1 = chance)

    Rank-based on purpose: LLaVA's attention is dominated by a few heavy
    hitters and long answers make it more peaked than the accumulated
    relevance, and a mass-based fit then reports zero novelty even when the
    turn moved to new entries.  The next compression uses the mean of this
    conversation's novelties plus one prior value, the mean novelty of the
    run's earlier conversations (0.5 before there are any).  Cost: two top-k
    calls per layer per turn.

alpha = 0 and kappa -> 0 reduces (C) to a relevance top-k; the ablation flags
below switch every component off independently.
"""

import math

import torch
import torch.nn.functional as F

from kv_cache_final import TurnConditionedLayerwiseKVCache


class TurnConditionedCoverageKVCache(TurnConditionedLayerwiseKVCache):
    """Turn-conditioned layer budgets + anticipatory coverage token selection."""

    def __init__(
        self,
        *args,
        coverage_alpha=0.5,
        coverage_kappa=1.0,
        similarity="key",
        recent_window=8,
        text_prior_frac=0.5,
        prefill_compress=True,
        relevance="text_mean",
        merge_visual=False,
        layer_floor=0.5,
        floor_stage="all",
        mass_preserving=True,
        mass_min_sim=0.25,
        alpha_auto=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mass_preserving = bool(mass_preserving)
        self.mass_min_sim = float(mass_min_sim)
        if not 0.0 <= self.mass_min_sim <= 1.0:
            raise ValueError("mass_min_sim must be in [0, 1]")
        self._attn_modules = None
        self.layer_floor = float(layer_floor)
        self.floor_stage = str(floor_stage)
        if not 0.0 <= self.layer_floor <= 1.0:
            raise ValueError("layer_floor must be in [0, 1]")
        if self.floor_stage not in {"all", "prefill"}:
            raise ValueError("floor_stage must be 'all' or 'prefill'")
        self.coverage_alpha = float(coverage_alpha)
        self.alpha_auto = bool(alpha_auto)
        self._alpha_now = self.coverage_alpha
        self.coverage_kappa = float(coverage_kappa)
        self.similarity = str(similarity)
        self.recent_window = max(1, int(recent_window))
        self.text_prior_frac = float(text_prior_frac)
        self.prefill_compress = bool(prefill_compress)
        self.relevance = str(relevance)
        self.merge_visual = bool(merge_visual)
        if not 0.0 <= self.coverage_alpha <= 1.0:
            raise ValueError("coverage_alpha must be in [0, 1]")
        if self.coverage_kappa < 0:
            raise ValueError("coverage_kappa must be non-negative")
        if self.similarity not in {"key", "value", "key_value"}:
            raise ValueError("similarity must be 'key', 'value' or 'key_value'")
        if self.relevance not in {"text_mean", "cumulative"}:
            raise ValueError("relevance must be 'text_mean' or 'cumulative'")
        if not 0.0 <= self.text_prior_frac <= 1.0:
            raise ValueError("text_prior_frac must be in [0, 1]")
        # The base allocation's per-layer lower bound is start_size + protect_size.
        self.protect_size = self.recent_window

    # ------------------------------------------------------------------
    # Extra per-entry state, kept in the same spare-slot buffers as the base
    # ------------------------------------------------------------------
    # Extra buffers, all zero for a new entry:
    #   0 text_attn  summed attention lift from text queries
    #   1 text_obs   number of text queries that could see the entry
    #   2 log_mass   log of how many original entries this entry stands for
    TEXT_ATTN, TEXT_OBS, LOG_MASS = 0, 1, 2

    def reset(self):
        super().reset()
        self._extra_buffers = None
        self._extra_views = None
        self._pending_prefill = False
        self._prefill_done = False
        self._set_attention_bias(None)
        # alpha_auto: state saved at the last compression, and the backtest
        # curve of each finished turn of this conversation.
        self._alpha_snapshot = None
        self._curves_own = []   # novelty of each finished turn
        self.last_turn_stats = None
        self.alpha_history = []

    def attach_model(self, model):
        """Give the cache the attention modules that read ``_kv_logit_bias``.

        Only needed with ``mass_preserving``; the patched legacy LLaMA
        attention adds the bias to the logits of the first cached keys.
        """
        inner = model.get_model() if hasattr(model, "get_model") else model.model
        modules = [layer.self_attn for layer in inner.layers]
        if self.mass_preserving and not getattr(type(modules[0]).forward, "supports_kv_logit_bias", False):
            raise RuntimeError(
                "mass_preserving needs an attention forward that adds self_attn._kv_logit_bias "
                "to the key logits (see the tgv_kv_legacy_nova.py edit), or pass "
                "--coverage-no-mass"
            )
        self._attn_modules = modules
        self._set_attention_bias(None)

    def _set_attention_bias(self, per_layer):
        if not getattr(self, "_attn_modules", None):
            return
        for layer_idx, module in enumerate(self._attn_modules):
            module._kv_logit_bias = None if per_layer is None else per_layer[layer_idx]

    def _initialize(self, lengths, device):
        self._extra_buffers = [None] * self.layer_num
        self._extra_views = [None] * self.layer_num
        super()._initialize(lengths, device)

    def _store_layer(self, layer_idx, score, conn, provenance, min_capacity=0, extras=None):
        if extras is None:
            if self._extra_views is not None and self._extra_views[layer_idx] is not None:
                extras = self._extra_views[layer_idx]
            else:
                zeros = torch.zeros(score.numel(), dtype=torch.float32, device=score.device)
                extras = (zeros, zeros, zeros)
        super()._store_layer(layer_idx, score, conn, provenance, min_capacity)
        capacity = self._buffers[layer_idx][0].numel()
        length = score.numel()
        buffers = []
        for values in extras:
            buf = torch.zeros(capacity, dtype=torch.float32, device=score.device)
            buf[: values.numel()] = values
            buffers.append(buf)
        self._extra_buffers[layer_idx] = tuple(buffers)
        self._extra_views[layer_idx] = tuple(b[:length] for b in buffers)

    def _grow_layer(self, layer_idx, length):
        super()._grow_layer(layer_idx, length)
        self._extra_views[layer_idx] = tuple(b[:length] for b in self._extra_buffers[layer_idx])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        first_prefill = self.score_sum is None
        super().observe_layer(layer_idx, attention)  # base stats + growth

        attn = attention[0].mean(dim=0, dtype=torch.float32)  # [queries, keys]
        query_len, key_len = attn.shape
        text_attn = self._extra_views[layer_idx][self.TEXT_ATTN]
        text_obs = self._extra_views[layer_idx][self.TEXT_OBS]
        # Attention is measured as LIFT over uniform (probability x number of
        # visible keys): a query that sees few keys hands out large
        # probabilities, so raw probabilities would favour tokens that are
        # mostly observed by early queries (e.g. text before the image).
        if query_len == 1:
            # Decode step: the only query is text appended in the current turn.
            text_attn.add_(attn[0] * key_len)
            text_obs.add_(1.0)
            return

        query_start = key_len - query_len
        is_text = (self.provenance[layer_idx][query_start:] != self.IMAGE).float()
        visible = torch.arange(query_start + 1, key_len + 1, device=attn.device, dtype=torch.float32)
        text_attn.add_((is_text * visible) @ attn)
        # Observers of key j = text queries at positions >= j (causal mask).
        later = torch.flip(torch.cumsum(torch.flip(is_text, [0]), 0), [0])
        if query_start > 0:
            text_obs[:query_start].add_(is_text.sum())
        text_obs[query_start:].add_(later)
        if first_prefill or not self._prefill_done:
            self._pending_prefill = True

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _relevance(self, layer_idx):
        if self.relevance == "cumulative":
            return self.score_sum[layer_idx].float().clamp_min(0)
        text_attn = self._extra_views[layer_idx][self.TEXT_ATTN]
        text_obs = self._extra_views[layer_idx][self.TEXT_OBS]
        return torch.where(text_obs > 0, text_attn / text_obs.clamp_min(1.0), torch.zeros_like(text_attn))

    def _features(self, k, v, image_idx):
        """Per-head centred, normalised features flattened so that f f^T = mean-head cosine."""
        parts = []
        sources = {"key": (k,), "value": (v,), "key_value": (k, v)}[self.similarity]
        for t in sources:
            x = t[0].index_select(1, image_idx).float()          # [H, n, D]
            x = x - x.mean(dim=1, keepdim=True)
            x = F.normalize(x, dim=-1, eps=1e-6)
            parts.append(x.permute(1, 0, 2).reshape(image_idx.numel(), -1) / math.sqrt(x.shape[0]))
        feats = torch.cat(parts, dim=-1) / math.sqrt(len(parts))
        return feats                                              # [n, H*D*len(parts)]

    def _similarity(self, feats):
        cos = (feats @ feats.T).clamp(-1.0, 1.0)
        n = cos.shape[0]
        if self.coverage_kappa == 0 or n < 2:
            return torch.eye(n, device=cos.device)
        dist = 1.0 - cos
        nn_dist = dist.masked_fill(torch.eye(n, dtype=torch.bool, device=dist.device), float("inf")).min(dim=1).values
        bandwidth = (self.coverage_kappa * nn_dist.median()).clamp_min(1e-4)
        sim = torch.exp(-dist.clamp_min(0) / bandwidth)
        sim.fill_diagonal_(1.0)
        return sim

    def _protected(self, layer_idx, budget, stage, relevance):
        length = self.score_sum[layer_idx].numel()
        device = relevance.device
        keep = torch.zeros(length, dtype=torch.bool, device=device)
        keep[: min(self.start_size, length)] = True
        # Recency is a property of the conversation, so the window covers the
        # most recent TEXT entries (never the tail of the image), and shrinks
        # at very small budgets.
        window = min(self.recent_window, max(1, int(budget) // 4))
        text_idx = (self.provenance[layer_idx] != self.IMAGE).nonzero(as_tuple=True)[0]
        keep[text_idx[-window:]] = True
        keep[length - 1] = True
        if stage == "prefill" and self.text_prior_frac > 0:
            text = (self.provenance[layer_idx] == self.current_turn) & ~keep
            cap = int(self.text_prior_frac * budget) - int(keep.sum())
            idx = text.nonzero(as_tuple=True)[0]
            if cap > 0 and idx.numel() > 0:
                if idx.numel() > cap:
                    idx = idx[torch.topk(relevance[idx], cap).indices]
                keep[idx] = True
        return keep

    def _layer_problem(self, layer_idx, budget, k, v, stage):
        """Greedy inputs for one layer: protected mask, weights, visual similarity."""
        length = self.score_sum[layer_idx].numel()
        device = self.score_sum[layer_idx].device
        relevance = self._relevance(layer_idx)
        keep = self._protected(layer_idx, budget, stage, relevance)
        candidate = ~keep
        weights = torch.zeros(length, dtype=torch.float32, device=device)
        n_cand = int(candidate.sum())
        if n_cand > 0:
            rel = relevance[candidate].clamp_min(0)
            total = rel.sum()
            # Uniform prior over ORIGINAL entries: an entry that stands for m
            # merged entries carries m shares (m = 1 without mass_preserving).
            mass = self._extra_views[layer_idx][self.LOG_MASS][candidate].exp()
            prior = mass / mass.sum()
            rel_part = torch.where(total > 0, rel / total.clamp_min(1e-30), prior)
            alpha = getattr(self, "_alpha_now", self.coverage_alpha)
            weights[candidate] = (1.0 - alpha) * rel_part + alpha * prior
        image_idx = (self.provenance[layer_idx] == self.IMAGE).nonzero(as_tuple=True)[0]
        sim = None
        if image_idx.numel() > 0:
            sim = self._similarity(self._features(k, v, image_idx))
        keep_count = int(keep.sum())
        quota = (length - keep_count) if budget >= length else max(0, int(budget) - keep_count)
        return {"length": length, "keep": keep, "weights": weights, "image_idx": image_idx,
                "sim": sim, "quota": quota}

    def _select_all(self, past_key_values, budgets, stage):
        """Batched greedy facility location over all layers at once.

        One fixed-length loop; each step is a handful of [L, N, N] tensor ops
        and there is no host synchronisation inside the loop.
        """
        problems = [
            self._layer_problem(l, int(budgets[l]), k, v, stage)
            for l, (k, v) in enumerate(past_key_values)
        ]
        layers = len(problems)
        n_max = max(p["length"] for p in problems)
        ni_max = max(1, max(p["image_idx"].numel() for p in problems))
        device = problems[0]["weights"].device
        # Visual block: only visual entries can stand in for one another.
        sim = torch.zeros(layers, ni_max, ni_max, dtype=torch.float32, device=device)
        w_img = torch.zeros(layers, ni_max, dtype=torch.float32, device=device)
        img_sel = torch.zeros(layers, ni_max, dtype=torch.bool, device=device)
        img_free = torch.zeros(layers, ni_max, dtype=torch.bool, device=device)
        # Text entries only stand for themselves: their gain is their weight.
        txt_gain = torch.full((layers, n_max), -1.0, dtype=torch.float32, device=device)
        selected = torch.zeros(layers, n_max, dtype=torch.bool, device=device)
        img_pos = torch.zeros(layers, ni_max, dtype=torch.long, device=device)
        quota = torch.zeros(layers, dtype=torch.long, device=device)
        for l, p in enumerate(problems):
            n = p["length"]
            keep = p["keep"]
            selected[l, :n] = keep
            image = torch.zeros(n, dtype=torch.bool, device=device)
            ii = p["image_idx"]
            image[ii] = True
            text_free = ~keep & ~image
            txt_gain[l, :n] = torch.where(text_free, p["weights"], torch.full_like(p["weights"], -1.0))
            ni = ii.numel()
            if ni:
                sim[l, :ni, :ni] = p["sim"]
                w_img[l, :ni] = p["weights"][ii]
                img_sel[l, :ni] = keep[ii]
                img_free[l, :ni] = ~keep[ii]
                img_pos[l, :ni] = ii
            quota[l] = p["quota"]

        # cover[l, i] = max over selected visual j of sim[l, i, j]
        cover = (sim * img_sel.unsqueeze(1)).amax(dim=2)
        rows = torch.arange(layers, device=device)
        # Text gains never change, so the greedy takes text in descending weight
        # order; every text entry that beats the best visual gain is taken in
        # one step (identical result to one-at-a-time greedy, fewer steps).
        txt_sorted, txt_order = txt_gain.sort(dim=1, descending=True)
        neg_sorted = (-txt_sorted).contiguous()
        txt_count = (txt_sorted > -1.0).sum(dim=1)
        ptr = torch.zeros(layers, dtype=torch.long, device=device)
        span = torch.arange(n_max, device=device).unsqueeze(0)
        max_steps = max(p["quota"] for p in problems)
        for step in range(max_steps):
            if step % 16 == 15 and not bool((quota > 0).any()):
                break
            gain_img = (w_img.unsqueeze(2) * (sim - cover.unsqueeze(2)).clamp_min(0)).sum(dim=1)
            gain_img = gain_img.masked_fill(~img_free, -1.0)
            gi_val, gi = gain_img.max(dim=1)
            next_txt = torch.where(ptr < txt_count, txt_sorted[rows, ptr.clamp_max(n_max - 1)],
                                   torch.full_like(gi_val, -1.0))
            active = quota > 0
            take_img = active & (gi_val > -1.0) & (gi_val >= next_txt)
            take_txt = active & ~take_img & (next_txt > -1.0)
            # Text block: all remaining text with weight strictly above the best visual gain.
            stop = torch.searchsorted(neg_sorted, (-gi_val).unsqueeze(1), side="left").squeeze(1)
            n_txt = torch.minimum((stop - ptr).clamp_min(0), quota)
            n_txt = torch.where(take_txt, n_txt.clamp_min(1), torch.zeros_like(n_txt))
            block = (span >= ptr.unsqueeze(1)) & (span < (ptr + n_txt).unsqueeze(1))
            selected.scatter_(1, txt_order, block | selected.gather(1, txt_order))
            ptr = ptr + n_txt
            # Visual pick.
            img_free[rows, gi] &= ~take_img
            img_sel[rows, gi] |= take_img
            selected[rows, img_pos[rows, gi]] |= take_img
            cover = torch.where(take_img.unsqueeze(1), torch.maximum(cover, sim[rows, :, gi]), cover)
            quota = quota - n_txt - take_img.long()

        out = []
        for l, p in enumerate(problems):
            n = p["length"]
            sel = selected[l, :n]
            retained = sel.nonzero(as_tuple=True)[0]
            src = dst = retained[:0]
            if (self.merge_visual or self.mass_preserving) and p["sim"] is not None:
                kept_img = sel[p["image_idx"]]
                drop_img = ~kept_img
                if bool(kept_img.any()) and bool(drop_img.any()):
                    best, assign = p["sim"][drop_img][:, kept_img].max(dim=1)
                    src = p["image_idx"][drop_img]
                    dst = p["image_idx"][kept_img][assign]
                    if self.mass_preserving:
                        # Only an anchor similar enough to an entry absorbs it
                        # (mass_min_sim = 0: all image mass is conserved).  Zero
                        # similarity (kappa = 0) never transfers.
                        covered = (best >= self.mass_min_sim) & (best > 0)
                        src, dst = src[covered], dst[covered]
            out.append((retained, src, dst))
        return out

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def _stage_budgets(self, stage):
        """Base layer allocation, mixed with a uniform share of the budget.

        Every layer is guaranteed about ``layer_floor`` x the uniform per-layer
        budget; the rest follows the configured allocation signal.  The same
        hedge as ``coverage_alpha``, applied to layers: a sensitivity profile
        probed at one ratio says little about a layer cut far below it.
        """
        scores, budgets, uniform_target = self._turn_budgets()
        if self.layer_floor <= 0 or (self.floor_stage == "prefill" and stage != "prefill"):
            return scores, budgets, uniform_target
        lengths = torch.tensor([x.numel() for x in self.score_sum], device=scores.device)
        lower = torch.tensor(
            [min(int(n), self.start_size + self.protect_size) for n in lengths.tolist()],
            device=scores.device,
        )
        weights = scores.double().clamp_min(0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.full_like(weights, 1.0 / weights.numel())
        mixed = (1.0 - self.layer_floor) * weights + self.layer_floor / weights.numel()
        budgets = self._bounded_largest_remainder(
            uniform_target * self.layer_num, mixed, lower, lengths
        )
        return mixed.to(scores.dtype), budgets, uniform_target

    def _compress(self, past_key_values, stage):
        if past_key_values is None or self.score_sum is None:
            return past_key_values
        if self.mass_preserving and not self._attn_modules:
            raise RuntimeError(
                "mass_preserving needs kv_cache.attach_model(model) and an attention "
                "forward that adds self_attn._kv_logit_bias to the key logits"
            )
        alpha, turn_alpha = self._choose_alpha(past_key_values)
        self._alpha_now = alpha
        self.alpha_history.append(
            {"stage": stage, "turn": self.current_turn + 1, "alpha": alpha, "turn_novelty": turn_alpha,
             "overlap": None if self.last_turn_stats is None else round(self.last_turn_stats["overlap"], 4),
             "chance": None if self.last_turn_stats is None else round(self.last_turn_stats["chance"], 4)}
        )
        scores, budgets, uniform_target = self._stage_budgets(stage)
        policy = "coverage_merge" if self.merge_visual else "coverage"
        policy += "_mass" if self.mass_preserving else ""
        lengths = [x.numel() for x in self.score_sum]
        if all(int(budgets[l]) >= lengths[l] for l in range(len(lengths))):
            # Nothing to evict (e.g. ratio 0): same result as the greedy keeping all.
            self._finish_finalize(scores, budgets, uniform_target, policy)
            self.last_finalize_stats["stage"] = stage
            self.last_finalize_stats["image_kept_per_layer"] = [
                int((p == self.IMAGE).sum()) for p in self.provenance
            ]
            self._after_compress(alpha, turn_alpha)
            return past_key_values
        selections = self._select_all(past_key_values, budgets, stage)
        new_past = []
        new_state = []
        image_kept = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            retained, src, dst = selections[layer_idx]
            dev = k.device
            retained_d = retained.to(dev)
            k_new = k.index_select(2, retained_d)
            v_new = v.index_select(2, retained_d)
            views = self._extra_views[layer_idx]
            extras = [x[retained] for x in views]
            if src.numel() > 0:
                pos = torch.full((k.shape[2],), -1, dtype=torch.long, device=dev)
                pos[retained_d] = torch.arange(retained.numel(), device=dev)
                target = pos[dst.to(dev)]
                src_d = src.to(dev)
                index = target.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
                mass_all = views[self.LOG_MASS].exp().to(dev)
                if self.merge_visual and self.mass_preserving:
                    # Mass-weighted mean of each anchor and the entries it absorbs.
                    shape = (1, 1, -1, 1)
                    m_keep, m_src = mass_all[retained_d], mass_all[src_d]
                    den = m_keep.scatter_add(0, target, m_src)
                    k_sum = (k_new.float() * m_keep.view(shape)).scatter_add(
                        2, index, k.index_select(2, src_d).float() * m_src.view(shape))
                    v_sum = (v_new.float() * m_keep.view(shape)).scatter_add(
                        2, index, v.index_select(2, src_d).float() * m_src.view(shape))
                    k_new = (k_sum / den.view(shape)).to(k.dtype)
                    v_new = (v_sum / den.view(shape)).to(v.dtype)
                elif self.merge_visual:
                    # Average each dropped visual entry into its covering anchor.
                    k_new = k_new.scatter_reduce(2, index, k.index_select(2, src_d), reduce="mean", include_self=True)
                    v_new = v_new.scatter_reduce(2, index, v.index_select(2, src_d), reduce="mean", include_self=True)
                if self.mass_preserving:
                    # The anchor now stands for its own mass plus the absorbed mass.
                    mass = mass_all[retained_d].scatter_add(0, target, mass_all[src_d])
                    extras[self.LOG_MASS] = mass.log().to(views[self.LOG_MASS].device)
            new_past.append([k_new, v_new])
            new_state.append((
                self.score_sum[layer_idx][retained],
                self.conn_score_sum[layer_idx][retained],
                self.provenance[layer_idx][retained],
                tuple(extras),
            ))
            image_kept.append(int((self.provenance[layer_idx][retained] == self.IMAGE).sum()))
        for layer_idx, (score, conn, prov, extras) in enumerate(new_state):
            self._store_layer(layer_idx, score, conn, prov, extras=extras)
        if self.mass_preserving:
            # Proportional attention: logit + log(mass) on the first cached keys.
            self._set_attention_bias([
                views[self.LOG_MASS].clone() if bool((views[self.LOG_MASS] != 0).any()) else None
                for views in self._extra_views
            ])
        self._finish_finalize(scores, budgets, uniform_target, policy)
        self.last_finalize_stats["stage"] = stage
        self.last_finalize_stats["image_kept_per_layer"] = image_kept
        self._after_compress(alpha, turn_alpha)
        return new_past

    # ------------------------------------------------------------------
    # Automatic coverage_alpha (alpha_auto=True)
    # ------------------------------------------------------------------
    # Turn novelties measured in this run, per (ratio, layer count): [sum, count].
    # Class-level, so it is shared by the cache objects of all conversations.
    _ALPHA_POOL = {}

    def _after_compress(self, alpha, turn_alpha):
        self.last_finalize_stats["coverage_alpha"] = alpha
        self.last_finalize_stats["turn_novelty"] = turn_alpha
        self.last_finalize_stats["turn_overlap"] = None if self.last_turn_stats is None else self.last_turn_stats["overlap"]
        if self.alpha_auto:
            self._take_snapshot()

    def _choose_alpha(self, past_key_values):
        """alpha for this compression, and the novelty of the turn that just ended.

        Manual mode: coverage_alpha.  Auto mode: the mean of this conversation's
        turn novelties plus one prior value, the mean novelty of the run's
        earlier conversations (0.5 before there are any).
        """
        if not self.alpha_auto:
            self.last_turn_stats = None
            return self.coverage_alpha, None
        novelty, self.last_turn_stats = self._turn_novelty()
        pool = self._ALPHA_POOL.setdefault((round(float(self.ratio), 6), int(self.layer_num)), [0.0, 0])
        if novelty is not None:
            self._curves_own.append(novelty)
            pool[0] += novelty
            pool[1] += 1
        own_sum, own_count = sum(self._curves_own), len(self._curves_own)
        others = pool[1] - own_count
        if others <= 0:
            return float(self.coverage_alpha), novelty
        run_novelty = (pool[0] - own_sum) / others
        if run_novelty <= 0:
            return float(self.coverage_alpha), novelty
        conversation_novelty = (run_novelty + own_sum) / (1 + own_count)
        alpha = self.coverage_alpha * conversation_novelty / run_novelty
        return float(min(1.0, max(0.0, alpha))), novelty

    def _take_snapshot(self):
        """Remember, for every evictable entry, its relevance and attention so far.

        Evictable = everything except the attention-sink prefix and the most
        recent text window.  O(entries) per layer.
        """
        snapshot = []
        for layer_idx in range(self.layer_num):
            length = self.score_sum[layer_idx].numel()
            relevance = self._relevance(layer_idx)
            protected = self._protected(layer_idx, length, "boundary", relevance)
            cand = (~protected).nonzero(as_tuple=True)[0]
            views = self._extra_views[layer_idx]
            mass = views[self.LOG_MASS][cand].double().exp()
            expected = relevance[cand].double().clamp_min(0) * mass
            seen = views[self.TEXT_ATTN][cand].double().clone()
            snapshot.append((cand, seen, expected, mass))
        self._alpha_snapshot = snapshot

    def _turn_novelty(self):
        """How much the last turn focused on entries the earlier relevance ranked low.

        Per layer, over the entries that were evictable at the previous
        compression (still the first entries of the cache): a = attention they
        received during the turn, r = relevance known at that compression.
        With K = the number of entries a compression keeps (keep fraction 1 - ratio),

            overlap = |top-K(a) & top-K(r)| / K          (chance level K / n)
            novelty = (1 - overlap) / (1 - K / n), clipped to [0, 1]

        0 = the turn looked exactly where the relevance ranking expected,
        1 = no better than chance.  Being rank-based, it does not care how
        peaked the attention is or how much mass a few heavy-hitter tokens soak
        up.  Two top-k calls per layer.  Returns (novelty, stats) or (None, None).
        """
        if self._alpha_snapshot is None:
            return None, None
        keep = 1.0 - float(self.ratio)
        values, overlaps, chances = [], [], []
        for layer_idx, (cand, seen, expected, mass) in enumerate(self._alpha_snapshot):
            n = cand.numel()
            k = int(round(keep * n))
            if n < 2 or k < 1 or k >= n:
                continue
            gain = (self._extra_views[layer_idx][self.TEXT_ATTN][cand].double() - seen).clamp_min(0)
            if gain.sum() <= 0 or expected.sum() <= 0:
                continue
            top_a = torch.zeros(n, dtype=torch.bool, device=gain.device)
            top_a[gain.topk(k).indices] = True
            overlap = float(top_a[expected.topk(k).indices].sum()) / k
            chance = k / n
            values.append(min(1.0, max(0.0, (1.0 - overlap) / (1.0 - chance))))
            overlaps.append(overlap)
            chances.append(chance)
        if not values:
            return None, None
        novelty = sum(values) / len(values)
        stats = {"overlap": sum(overlaps) / len(overlaps), "chance": sum(chances) / len(chances), "layers": len(values)}
        return novelty, stats

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        past_key_values = super().__call__(past_key_values, num_of_token, attentions)
        if self._pending_prefill:
            self._pending_prefill = False
            self._prefill_done = True
            if self.prefill_compress:
                return self._compress(past_key_values, stage="prefill")
        return past_key_values

    def finalize_turn(self, past_key_values):
        return self._compress(past_key_values, stage="boundary")
