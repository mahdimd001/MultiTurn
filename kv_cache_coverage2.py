"""
Turn-conditioned coverage 2 (TC-Coverage2): turn-flow scores and part-wise budgets.

Built on ``TurnConditionedCoverageKVCache`` (kv_cache_coverage.py).  Reused from
TC-Coverage: the compression points (after the Turn-1 prefill and at every turn
boundary), the layer budgets (so the total budget is identical), the visual
coverage selection with ``coverage_alpha`` / ``coverage_kappa``, and the
mass-preserving eviction.  New: how entries are scored and how each layer's
budget is divided.  Removed: the recent window and the Turn-1 text prior.

Three parts at every compression
--------------------------------
  V  image entries
  H  history text: text kept by earlier compressions (it has an old score)
  N  new text, the READERS: text that has not been scored yet.  At the Turn-1
     prefill this is the whole prompt text; at a turn boundary it is every
     token appended during the turn (the answer, and from Turn 2 on also the
     question).

(1) Turn flow (who mattered in this turn).  While the turn runs, each reader's
    attention row is stored (head mean; the attention sink and the reader
    itself are left out and the row is renormalised).  At the compression a
    reader's importance is the reading that flows through it: one unit seeded
    uniformly on the readers and passed back along their attention to earlier
    readers,

        pi_q = 1/|N| + sum_{r > q} pi_r a_{rq}            (one triangular solve)

    so an answer token that later answer tokens rely on is important, and what
    it read is credited through it.  Each entry receives the flow
    F_j = sum_q pi_q a_{qj}; its evidence is the flow LIFT

        e_j = F_j / sum_{q sees j} pi_q / n_q              (1 = uniform reading)

    where n_q is the number of entries q could read.  Dividing by the flow under
    uniform reading removes the bias toward early tokens (their readers see few
    entries and hand out large probabilities) and toward old tokens (more
    readers).
(2) Score update.  Old entries (V, H): running mean of e over the
    compressions that measured them.  New entries: their first evidence.  A
    text entry is ranked by its score averaged over the layers that hold it
    (a word carries the same content in every layer; one layer's attention is a
    noisy probe of it).  Image entries keep per-layer scores.
(3) Part budgets.  Each part is compressed by its own rule into a ranked list:
    V by facility-location coverage (marginal gains, with the TC-Coverage
    kernel and coverage_alpha), H and N by score.  The layer's free budget
    (budget minus the attention sink and the newest entry) is divided among
    the parts from these scores, with no per-part hyperparameter:
      part_split="fill"          (default) water-filling: every slot goes to
                                 the part whose next entry adds the most, so
                                 the last entry kept in every part is worth
                                 about the same
      part_split="proportional"  ablation: quota of a part ∝ its total
                                 importance (under-serves small parts such as
                                 a short answer that the next turn refers to)
    At the Turn-1 prefill nothing has read the prompt yet as an answer, so
    the text quota is filled newest first (the question is the newest text).
(4) An evicted visual entry moves its mass AND its score to its anchor.

Extra memory: three scalars per entry (score, count, token id) and, during a
turn, one attention row per reader per layer.  Extra compute: one batched
triangular solve per compression.
"""

import torch

from kv_cache_coverage import TurnConditionedCoverageKVCache


class TurnConditionedCoverage2KVCache(TurnConditionedCoverageKVCache):
    """TC-Coverage with turn-flow scoring and part-wise budgets."""

    # Extra per-entry buffers after TC-Coverage's (text_attn, text_obs, log_mass).
    SCORE, COUNT, TOKEN = 3, 4, 5      # TOKEN: conversation-wide id, the same in every layer
    PARTS = ("image", "history", "new")

    def __init__(self, *args, part_split="fill", **kwargs):
        if part_split not in {"fill", "proportional"}:
            raise ValueError("part_split must be 'fill' or 'proportional'")
        if kwargs.get("alpha_auto"):
            raise ValueError("TC-Coverage2 uses the fixed coverage_alpha")
        kwargs["text_prior_frac"] = 0.0     # replaced by the part budgets
        self.part_split = part_split
        super().__init__(*args, **kwargs)
        # Lower bound of a layer budget in the base allocation: sink + newest entry.
        self.protect_size = 1

    def reset(self):
        super().reset()
        self._seg_start = None   # per layer: first entry appended after the last compression
        self._first = None       # per layer: reader positions inside the Turn-1 prefill
        self._rows = None        # per layer: [readers, keys] stored attention rows
        self._readers = None     # per layer: reader mask of the last score update
        self._known = None       # per layer: number of entries that have a token id
        self._next_token = 0
        self.last_part_stats = None
        self.part_history = []

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _store_layer(self, layer_idx, score, conn, provenance, min_capacity=0, extras=None):
        if extras is None and (self._extra_views is None or self._extra_views[layer_idx] is None):
            zeros = torch.zeros(score.numel(), dtype=torch.float32, device=score.device)
            extras = (zeros,) * 6
        super()._store_layer(layer_idx, score, conn, provenance, min_capacity, extras)

    def _rows_for(self, layer_idx, rows, cols, device):
        buf = self._rows[layer_idx]
        if buf is None or buf.shape[0] < rows or buf.shape[1] < cols:
            old_r, old_c = (0, 0) if buf is None else buf.shape
            grown = torch.zeros(max(rows, old_r + max(32, old_r // 2)),
                                max(cols, old_c + max(32, old_c // 2)),
                                dtype=torch.float32, device=device)
            if buf is not None:
                grown[:old_r, :old_c] = buf
            self._rows[layer_idx] = buf = grown
        return buf

    def _reader_positions(self, layer_idx):
        length = self.score_sum[layer_idx].numel()
        seg = self._seg_start[layer_idx]
        first = self._first[layer_idx]
        return torch.cat([first, torch.arange(seg, length, device=first.device)])

    # ------------------------------------------------------------------
    # Observation: token ids and the readers' attention rows
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        super().observe_layer(layer_idx, attention)
        key_len, query_len = attention.shape[-1], attention.shape[-2]
        device = attention.device
        if self._known is None:
            self._known = [0] * self.layer_num
        known = self._known[layer_idx]
        if key_len > known:
            # Every layer appends the same tokens in the same order (layer 0 first).
            added = key_len - known
            if layer_idx == 0:
                self._next_token += added
            ids = self._extra_views[layer_idx][self.TOKEN]
            ids[known:key_len] = torch.arange(self._next_token - added, self._next_token,
                                              dtype=torch.float32, device=ids.device)
            self._known[layer_idx] = key_len
        if self._seg_start is None:
            # Turn-1 prefill: every prompt text entry except the sink is a reader.
            prov = self.provenance[layer_idx]
            pos = torch.arange(key_len, device=prov.device)
            first = ((prov != self.IMAGE) & (pos >= min(self.start_size, key_len))).nonzero(as_tuple=True)[0]
            self._first = [first] * self.layer_num
            self._seg_start = [key_len] * self.layer_num
            self._rows = [None] * self.layer_num
        first = self._first[layer_idx]
        seg = self._seg_start[layer_idx]
        query_start = key_len - query_len
        if first.numel() and int(first[-1]) >= query_start and self._rows[layer_idx] is None:
            idx = first[first >= query_start]
            rows = attention[0][:, (idx - query_start).to(device), :].mean(dim=0, dtype=torch.float32)
            buf = self._rows_for(layer_idx, first.numel(), key_len, device)
            buf[first.numel() - idx.numel(): first.numel(), :key_len] = rows
        lo = max(seg, query_start)
        if lo < key_len:
            rows = attention[0, :, lo - query_start:, :].mean(dim=0, dtype=torch.float32)
            n_first = first.numel()
            buf = self._rows_for(layer_idx, n_first + key_len - seg, key_len, device)
            buf[n_first + lo - seg: n_first + key_len - seg, :key_len] = rows

    # ------------------------------------------------------------------
    # (1) + (2): turn flow and score update, before every compression
    # ------------------------------------------------------------------
    def _turn_flow(self):
        """Per layer: (reader mask, flow lift e of every entry), or None without readers."""
        if self._seg_start is None or self._rows is None:
            return None
        positions = [self._reader_positions(l) for l in range(self.layer_num)]
        n = positions[0].numel()
        if n == 0:
            return None
        sink = self.start_size
        rows_all, m_all = [], []
        for layer_idx, pos in enumerate(positions):
            length = self.score_sum[layer_idx].numel()
            buf = self._rows[layer_idx]
            if buf is None:
                rows = torch.zeros(n, length, dtype=torch.float32, device=self.score_sum[layer_idx].device)
            else:
                rows = buf[:n, :length].clone()
            pos = pos.to(rows.device)
            rows[:, : min(sink, length)] = 0.0                       # attention sink
            rows[torch.arange(n, device=rows.device), pos] = 0.0      # the reader itself
            rows = rows / rows.sum(dim=1, keepdim=True).clamp_min(1e-12)
            rows_all.append(rows)
            m_all.append(rows[:, pos])
        m = torch.stack(m_all).double()                              # [layers, n, n], strictly lower
        eye = torch.eye(n, dtype=m.dtype, device=m.device)
        seed = torch.full((self.layer_num, n, 1), 1.0 / n, dtype=m.dtype, device=m.device)
        # pi = seed + M^T pi  <=>  (I - M^T) pi = seed; I - M^T is unit upper triangular.
        pi_all = torch.linalg.solve_triangular(eye - m.transpose(1, 2), seed, upper=True).squeeze(-1)
        out = []
        for layer_idx, (rows, pos) in enumerate(zip(rows_all, positions)):
            length = rows.shape[1]
            pos = pos.to(rows.device)
            pi = pi_all[layer_idx].to(rows.device, torch.float32)
            flow = pi @ rows
            # Flow under uniform reading: sum over the readers after j of pi_q / n_q.
            per_key = torch.zeros(length, dtype=torch.float32, device=rows.device)
            per_key[pos] = pi / (pos - min(sink, length)).clamp_min(1).float()
            uniform = (per_key.flip(0).cumsum(0).flip(0) - per_key).clamp_min(0)
            lift = torch.where(uniform > 1e-12, flow / uniform.clamp_min(1e-12), torch.zeros_like(flow))
            lift[: min(sink, length)] = 0.0
            readers = torch.zeros(length, dtype=torch.bool, device=rows.device)
            readers[pos] = True
            out.append((readers, lift))
        return out

    def _update_scores(self):
        flows = self._turn_flow()
        self._readers = [None] * self.layer_num
        if flows is None:
            return
        for layer_idx, (readers, evidence) in enumerate(flows):
            views = self._extra_views[layer_idx]
            score, count = views[self.SCORE], views[self.COUNT]
            old = ~readers
            score[old] = (score[old] * count[old] + evidence[old]) / (count[old] + 1.0)
            count[old] += 1.0
            score[readers] = evidence[readers]
            count[readers] = 1.0
            self._readers[layer_idx] = readers

    def _pooled_text_scores(self):
        """Score of every token id averaged over the layers whose cache holds it as text."""
        n = max(1, self._next_token)
        device = self.score_sum[0].device
        total = torch.zeros(n, dtype=torch.float32, device=device)
        count = torch.zeros(n, dtype=torch.float32, device=device)
        for layer_idx in range(self.layer_num):
            views = self._extra_views[layer_idx]
            text = self.provenance[layer_idx] != self.IMAGE
            ids = views[self.TOKEN][text].long()
            total.index_add_(0, ids, views[self.SCORE][text].clamp_min(0))
            count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        return total / count.clamp_min(1.0)

    def _compress(self, past_key_values, stage):
        if past_key_values is not None and self.score_sum is not None:
            self._update_scores()
        return super()._compress(past_key_values, stage)

    def _after_compress(self, alpha, turn_alpha):
        super()._after_compress(alpha, turn_alpha)
        self._seg_start = [x.numel() for x in self.score_sum]
        self._first = [torch.zeros(0, dtype=torch.long, device=self.score_sum[0].device)] * self.layer_num
        self._rows = [None] * self.layer_num
        self._readers = None
        self._known = [x.numel() for x in self.score_sum]
        self.last_finalize_stats["parts"] = self.last_part_stats

    # ------------------------------------------------------------------
    # (3) + (4): part budgets and per-part selection
    # ------------------------------------------------------------------
    def _select_all(self, past_key_values, budgets, stage):
        alpha = getattr(self, "_alpha_now", self.coverage_alpha)
        device = self.score_sum[0].device
        pooled = self._pooled_text_scores()
        plans = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            length = self.score_sum[layer_idx].numel()
            views = self._extra_views[layer_idx]
            prov = self.provenance[layer_idx]
            image = prov == self.IMAGE
            score = torch.where(image, views[self.SCORE].clamp_min(0), pooled[views[self.TOKEN].long()])
            positions = torch.arange(length, device=device)
            keep = positions < min(self.start_size, length)
            keep[length - 1] = True
            readers = self._readers[layer_idx] if self._readers else None
            new = (readers if readers is not None else torch.zeros_like(image)) & ~image
            parts = [image & ~keep, ~image & ~new & ~keep, new & ~keep]
            free = max(0, min(int(budgets[layer_idx]), length) - int(keep.sum()))
            image_idx = image.nonzero(as_tuple=True)[0]
            plan = {"keep": keep, "parts": parts, "free": free, "score": score,
                    "image_idx": image_idx, "sim": None, "weights": None}
            if image_idx.numel() > 0:
                plan["sim"] = self._similarity(self._features(k, v, image_idx))
                cand = parts[0][image_idx]
                w = torch.zeros(image_idx.numel(), dtype=torch.float32, device=device)
                if bool(cand.any()):
                    s = score[image_idx][cand]
                    m = views[self.LOG_MASS][image_idx][cand].exp()
                    total = s.sum()
                    # The hedge spreads alpha of the image part's own score over its original entries.
                    w[cand] = (1.0 - alpha) * s + alpha * (total if total > 0 else 1.0) * m / m.sum()
                plan["weights"] = w
            plans.append(plan)

        gains, order = self._greedy_images(plans, device)

        out, stats = [], []
        for layer_idx, plan in enumerate(plans):
            parts, free, score = plan["parts"], plan["free"], plan["score"]
            ii = plan["image_idx"]
            text_idx = [p.nonzero(as_tuple=True)[0] for p in parts[1:]]
            text_rank = [score[i].argsort(descending=True) for i in text_idx]
            if stage == "prefill":
                # No answer has read the prompt yet: text is kept newest first.
                text_rank = [torch.arange(i.numel() - 1, -1, -1, device=device) for i in text_idx]
            img_gain = gains[layer_idx, : min(free, ii.numel())] if ii.numel() else gains.new_zeros(0)
            img_gain = img_gain[img_gain > -1.0]
            sizes = [img_gain.numel()] + [i.numel() for i in text_idx]
            if self.part_split == "fill":
                pool = torch.cat([img_gain] + [score[i].sort(descending=True).values for i in text_idx])
                label = torch.cat([torch.full((s,), j, device=device) for j, s in enumerate(sizes)])
                top = torch.topk(pool, min(free, pool.numel())).indices if pool.numel() else label[:0]
                quota = [int((label[top] == j).sum()) for j in range(3)]
            else:
                mass = torch.stack([plan["weights"].sum() if ii.numel() else score.new_zeros(())] +
                                   [score[i].sum() for i in text_idx]).double()
                quota = self._bounded_largest_remainder(
                    free, mass, torch.zeros(3, dtype=torch.long, device=device),
                    torch.tensor(sizes, device=device)).tolist()
            selected = plan["keep"].clone()
            if ii.numel() and quota[0] > 0:
                selected[ii[order[layer_idx, : quota[0]]]] = True
            for idx, rank, q in zip(text_idx, text_rank, quota[1:]):
                if q > 0:
                    selected[idx[rank[:q]]] = True
            retained = selected.nonzero(as_tuple=True)[0]
            src = dst = retained[:0]
            if self.mass_preserving and plan["sim"] is not None:
                kept_img = selected[ii]
                drop_img = ~kept_img
                if bool(kept_img.any()) and bool(drop_img.any()):
                    best, assign = plan["sim"][drop_img][:, kept_img].max(dim=1)
                    covered = (best >= self.mass_min_sim) & (best > 0)
                    src = ii[drop_img][covered]
                    dst = ii[kept_img][assign][covered]
                    if src.numel():
                        # The anchor serves the evicted entry's share from now on.
                        layer_score = self._extra_views[layer_idx][self.SCORE]
                        layer_score.index_add_(0, dst, layer_score[src])
            out.append((retained, src, dst))
            stats.append({"mass": [float(score[p].sum()) for p in parts], "quota": quota})
        self._record_parts(stats, stage)
        return out

    def _greedy_images(self, plans, device):
        """Batched facility-location greedy on the image part of every layer.

        Returns the marginal gain and image index of every greedy pick,
        [layers, steps]; gains are non-increasing and -1 marks no pick.
        """
        layers = len(plans)
        ni_max = max(1, max(p["image_idx"].numel() for p in plans))
        sim = torch.zeros(layers, ni_max, ni_max, dtype=torch.float32, device=device)
        w = torch.zeros(layers, ni_max, dtype=torch.float32, device=device)
        free = torch.zeros(layers, ni_max, dtype=torch.bool, device=device)
        steps = torch.zeros(layers, dtype=torch.long, device=device)
        for l, p in enumerate(plans):
            ni = p["image_idx"].numel()
            if ni:
                sim[l, :ni, :ni] = p["sim"]
                w[l, :ni] = p["weights"]
                free[l, :ni] = p["parts"][0][p["image_idx"]]
                steps[l] = min(p["free"], ni)
        n_steps = int(steps.max()) if layers else 0
        gains = torch.full((layers, max(1, n_steps)), -1.0, device=device)
        order = torch.zeros(layers, max(1, n_steps), dtype=torch.long, device=device)
        cover = torch.zeros(layers, ni_max, dtype=torch.float32, device=device)
        rows = torch.arange(layers, device=device)
        for step in range(n_steps):
            gain = (w.unsqueeze(2) * (sim - cover.unsqueeze(2)).clamp_min(0)).sum(dim=1)
            gain = gain.masked_fill(~free, -1.0)
            g_val, g_idx = gain.max(dim=1)
            take = (step < steps) & (g_val > -1.0)
            gains[:, step] = torch.where(take, g_val, torch.full_like(g_val, -1.0))
            order[:, step] = g_idx
            free[rows, g_idx] &= ~take
            cover = torch.where(take.unsqueeze(1), torch.maximum(cover, sim[rows, :, g_idx]), cover)
        return gains, order

    def _record_parts(self, stats, stage):
        layers = max(1, len(stats))
        mass = [sum(s["mass"][i] for s in stats) / layers for i in range(3)]
        total = sum(mass) or 1.0
        quota = [sum(s["quota"][i] for s in stats) / layers for i in range(3)]
        self.last_part_stats = {
            "stage": stage,
            "turn": self.current_turn + 1,
            "score_share": {name: round(m / total, 4) for name, m in zip(self.PARTS, mass)},
            "kept_per_layer": {name: round(q, 2) for name, q in zip(self.PARTS, quota)},
        }
        self.part_history.append(self.last_part_stats)
