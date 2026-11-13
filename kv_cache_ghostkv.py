"""
GhostKV: demand paging for the image part of a multi-turn KV cache.

Idea
----
SeedKV (kv_cache_seedkv.py) regenerates the WHOLE image for every turn.
GhostKV regenerates only the patches the model is about to read and does not
have, at the moment it reaches for them:

  1. Ghost keys (which part of the image is relevant).  A few attention heads
     of a LVLM localise: when the model needs an object, they put their
     attention on its patches.  For those "locator" heads GhostKV keeps the
     KEY of every image patch (no values, no other heads, no other layers).
     At every decoding step it computes, for each locator head, the attention
     the current token WOULD pay to every image patch if the whole image were
     cached ("virtual attention").  Summed over the answer tokens this is the
     relevance map of the turn (``relevance_map()``, ``save_maps``).

  2. Page fault (is it in the cache).  If a locator head puts more than its
     threshold of virtual attention on patches that are not resident, the
     step faults.

  3. Partial regeneration (only the missing part).  The faulting patches are
     regenerated from the seed with one small forward: the text before the
     image plus those patches, with the resident image entries as context.
     Their KV is spliced into the cache INSIDE the running forward pass, from
     the faulting layer upward, so the token that asked for them already reads
     them.  Shallower layers get them after the step.

Nothing is trained.  The locator heads are found once per model by a
label-free calibration run with the full cache (``calibrate=PATH``, about 20
conversations): a head scores high if hiding the patches it looks at changes
the model's next-token distribution.  The same run records how much attention
each head puts on its top patches (the fault threshold) and how many patches
it reads at a time (the size of the page pool).  Without that file the heads
are picked from the Turn-1 prefill attention, which works much worse.

Budget
------
C(n) = int(n * (1 - ratio)) entries per layer on average, like every method
here, minus the seed and the ghost keys (both charged in KV entries).  The
cache is compressed after the Turn-1 prefill and at every turn boundary with
PyramidKV's rule (every head keeps its own entries).  A page fault does not
grow the cache: each head overwrites its coldest entries with the patches it
is missing (old recent-window entries only when nothing colder is left; never
the attention sink or a token of the current turn).  At most ``page_slots``
paged patches are protected at a time, the ones in highest recent demand; the
others are overwritten first.  KV + seed + ghost keys therefore never exceeds
L * (C(length at last boundary) + tokens appended since): no transient peak.

Approximation
-------------
A regenerated patch is computed without the image patches that are neither
resident nor regenerated with it, so its KV is close to, not equal to, the
original.  ``context`` chooses what it does see.

Interface: the same as the other caches (``attach_model``, ``__call__``,
``begin_answer``, ``finalize_turn``); no evaluator change is needed.  Page
faults happen on single-token forwards (decoding and, in this repo's
evaluators, later-turn questions).
"""

import functools
import math
import os
import time

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from kv_cache_seedkv import SeedKVCache, _warn_once

_LIVE_HOOKS = []          # hooks of the conversation that is running
_CALIBRATION = {}         # path -> running calibration statistics
_MAPS = {}                # path -> {"conversations": int, "turns": [dict]}


def _clear_live_hooks():
    while _LIVE_HOOKS:
        _LIVE_HOOKS.pop().remove()


def _new_calibration():
    return {"prior": None, "steps": 0, "conversations": 0, "candidates": None, "kl": None, "kl_n": None,
            "top_mass": [], "spread": []}


QUANTILES = (0.5, 0.75, 0.9, 0.95)


def concentration_score(attention):
    """[heads, patches] attention -> [heads] mass on the image x how peaked it is."""
    mass = attention.sum(dim=-1)
    prob = attention / mass.clamp_min(1e-9).unsqueeze(-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1)
    peaked = (1.0 - entropy / math.log(max(2, attention.shape[-1]))).clamp_min(0.0)
    return mass * peaked


def load_locators(path, count):
    """The ``count`` best heads of a calibration file.

    Returns [(layer, head)], their weights (score relative to the best head),
    per head the quantiles ``QUANTILES`` of the attention it puts on its top
    image patches while answering with the full cache (or None), and per head
    its footprint: how many patches it reads when it looks hard (or None).
    """
    blob = torch.load(path, map_location="cpu")
    score = blob["score"]
    order = score.flatten().argsort(descending=True)[:count].tolist()
    heads = score.shape[1]
    pairs = [(index // heads, index % heads) for index in order]
    best = float(score.flatten()[order[0]])
    weights = [float(score[layer, head]) / best if best > 0 else 1.0 for layer, head in pairs]
    quantiles = blob.get("top_mass_quantiles")
    levels = None if quantiles is None else [quantiles[:, layer, head].tolist() for layer, head in pairs]
    footprint = blob.get("footprint")
    footprints = None if footprint is None else [int(footprint[layer, head]) for layer, head in pairs]
    return pairs, weights, levels, footprints


class GhostKVCache(SeedKVCache):
    """PyramidKV-style retention plus on-demand regeneration of missing image patches."""

    def __init__(
        self,
        layer_num,
        ratio=0.0,
        image_start=None,
        image_token_len=0,
        window_size=8,
        kernel_size=7,
        pooling="maxpool",
        allocation="pyramid",
        beta=20.0,
        seed="auto",
        charge=True,
        locators=None,
        n_locators=4,
        ghost_share=0.02,
        threshold="auto",
        cover=0.9,
        page_slots=None,
        page_frac=0.3,
        demand_decay=0.9,
        image_weight=1.0,
        fault_on="all",
        max_faults=0,
        context="prefix+resident",
        paging=True,
        calibrate=None,
        save_maps=None,
        calib_heads=32,
        calib_warmup=4,
        calib_dense=8,
        calib_stride=8,
        calib_hide=0.03,
    ):
        if context not in {"prefix+resident", "prefix", "resident", "none"}:
            raise ValueError("context must be 'prefix+resident', 'prefix', 'resident' or 'none'")
        self.paging = bool(paging) and calibrate is None
        self.calibrate = calibrate
        self.save_maps = save_maps                # file that collects the relevance map of every turn
        self.calib_heads = int(calib_heads)       # heads that get the causal test
        self.calib_warmup = int(calib_warmup)     # conversations used to pick those heads
        self.calib_dense = int(calib_dense)       # test the first answer tokens of a turn ...
        self.calib_stride = int(calib_stride)     # ... and every n-th after that
        self.calib_hide = float(calib_hide)       # share of the image hidden per test
        self.locator_source = locators
        self.n_locators = int(n_locators)
        self.ghost_share = float(ghost_share)     # ghost keys may use this share of the KV budget
        self.threshold = None if str(threshold) == "auto" else float(threshold)
        self.cover = float(cover)
        self.page_slots_arg = None if page_slots is None else int(page_slots)
        self.page_frac = float(page_frac)
        self.max_faults = int(max_faults)
        self.demand_decay = float(demand_decay)
        self.image_weight = float(image_weight)   # < 1: prefer text at compression, the image can be paged back
        if fault_on not in {"all", "answer"}:
            raise ValueError("fault_on must be 'all' or 'answer'")
        self.fault_on = fault_on
        self.use_prefix = "prefix" in context
        self.use_resident = "resident" in context
        self._page_hooks = []
        super().__init__(
            layer_num, ratio, image_start, image_token_len,
            window_size=window_size, kernel_size=kernel_size, pooling=pooling,
            allocation=allocation, beta=beta, seed=seed, charge_seed=charge,
            regenerate=self.paging,
        )
        self.image_end = None if self.image_start is None else self.image_start + self.image_len

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        super().reset()
        for handle in getattr(self, "_page_hooks", []):
            handle.remove()
        self._page_hooks = []
        self.locators = []             # [(layer, head)]
        self._weight = {}              # (layer, head) -> weight of the head (best head = 1)
        self._threshold = {}           # (layer, head) -> missing attention that makes a fault
        self.footprint = 0             # patches the best locator head reads when it looks hard (0: unknown)
        self._loc_by_layer = {}        # layer -> [head]
        self._ghost = {}               # (layer, head) -> [patches, head_dim] keys
        self.ghost_bytes = 0
        self.page_slots = 0            # paged-in patches a head may hold per layer
        self._heat = None              # per layer: [kv_heads, entries] score that kept the entry
        self._paged = set()            # patches paged in this turn and still protected
        self._demand = None            # [patches] recent virtual attention of the locator heads
        self._model = None
        self._calib_kwargs = None
        self._calib_step = 0
        self._step_past = None
        self._faulted = False
        self._inflight = {}            # layer -> fix, applied inside the running forward
        self._late = {}                # layer -> fix, applied after the forward
        self.faults = 0
        self.regen_tokens = 0          # tokens run through the model to regenerate
        self.regen_patches = 0
        self.regen_seconds = 0.0
        self.turn_log = []             # one dict per finished turn (faults, relevance maps, ...)
        self.fault_log = []            # one dict per page fault
        self._new_turn()

    def _new_turn(self):
        self._turn = {"faults": 0, "regen_tokens": 0, "paged": set(), "relevance": None,
                      "relevance_answer": None, "peak": None, "peak_sharpness": 0.0, "steps": 0}
        self._paged = set()
        self._demand = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def attach_model(self, model):
        super().attach_model(model)          # the seed capture (only if paging)
        _clear_live_hooks()
        self._model = model
        if self.calibrate is not None:
            stats = _CALIBRATION.setdefault(self.calibrate, _new_calibration())
            stats["conversations"] += 1
            handle = self._inner.register_forward_pre_hook(self._calibrate_capture, with_kwargs=True)
            self._page_hooks.append(handle)
            _LIVE_HOOKS.append(handle)
            return
        if self.ratio <= 0 or not self.paging:
            return
        if self.save_maps:
            _MAPS.setdefault(self.save_maps, {"conversations": 0, "turns": []})["conversations"] += 1
        self._layers = list(self._inner.layers)
        handle = self._inner.register_forward_pre_hook(self._step_begin, with_kwargs=True)
        self._page_hooks.append(handle)
        for index, layer in enumerate(self._layers):
            handle = layer.self_attn.register_forward_pre_hook(
                functools.partial(self._attention_pre, index), with_kwargs=True)
            self._page_hooks.append(handle)
        _LIVE_HOOKS.extend(self._page_hooks)

    def _step_begin(self, module, args, kwargs):
        if self._busy:
            return None
        self._step_past = kwargs.get("past_key_values")
        self._faulted = False
        self._inflight = {}
        if self._demand is not None:
            self._demand = self._demand * self.demand_decay
        return None

    def _attention_pre(self, index, module, args, kwargs):
        if self._busy or self.state != "compressed" or self._step_past is None:
            return None
        hidden = kwargs.get("hidden_states")
        past = kwargs.get("past_key_value")
        if hidden is None or past is None or hidden.shape[1] != 1 or kwargs.get("position_ids") is None:
            return None
        if index in self._loc_by_layer and not self._faulted:
            self._detect(index, module, hidden, past, kwargs.get("position_ids"))
        fix = self._inflight.get(index)
        if fix is None:
            return None
        kwargs = dict(kwargs)
        kwargs["past_key_value"] = self._apply_fix(past, fix)
        return args, kwargs

    # ------------------------------------------------------------------
    # Ghost keys and locator heads
    # ------------------------------------------------------------------
    AUTO_QUANTILE = 2              # index into QUANTILES: the head's 90th percentile
    AUTO_SHARE = 0.5
    FALLBACK_THRESHOLD = 0.3

    def _choose_locators(self):
        """[(layer, head)], weights, fault thresholds."""
        source = self.locator_source
        levels, footprints = None, None
        if isinstance(source, (list, tuple)):
            pairs = [(int(layer), int(head)) for layer, head in source][: self.n_locators]
            weights = [1.0] * len(pairs)
        elif isinstance(source, str) and os.path.exists(source):
            pairs, weights, levels, footprints = load_locators(source, self.n_locators)
            if footprints:
                self.footprint = footprints[0]
        else:
            _warn_once(
                "ghost_auto",
                "GhostKV: no calibration file, locator heads are picked from the Turn-1 prefill "
                "attention. Run once with calibrate=PATH for better heads.")
            rows = []
            for layer in range(self.layer_num):
                window = self._window_rows(layer, self.n0)
                if window is None:
                    continue
                image = window[:, :, self.image_start: self.image_end]          # [heads, rows, patches]
                score = concentration_score(image.reshape(-1, image.shape[-1])).view(image.shape[:2]).mean(1)
                rows += [(float(value), layer, head) for head, value in enumerate(score.tolist())]
            rows.sort(reverse=True)
            rows = rows[: self.n_locators]
            pairs = [(layer, head) for _, layer, head in rows]
            weights = [value / rows[0][0] if rows[0][0] > 0 else 1.0 for value, _, _ in rows]
        if self.threshold is not None:
            thresholds = [self.threshold] * len(pairs)
        elif levels is not None:
            # "Looking hard": half of what the head puts on its top patches in its sharpest 10% of steps.
            thresholds = [min(0.9, max(0.02, self.AUTO_SHARE * level[self.AUTO_QUANTILE])) for level in levels]
        else:
            thresholds = [self.FALLBACK_THRESHOLD] * len(pairs)
        return pairs, weights, thresholds

    def _build_ghost(self, past_key_values):
        """Keep the keys of every image patch for the locator heads (fp16)."""
        pairs, weights, thresholds = self._choose_locators()
        key = past_key_values[0][0]
        entry_bytes = 2 * key.shape[1] * key.shape[3] * key.element_size()      # one KV entry of one layer
        head_entries = self.image_len * key.shape[3] * 2 / float(entry_bytes)   # ghost keys of one head
        budget = self.ghost_share * self.layer_num * int(self.n0 * (1.0 - self.ratio))
        count = max(1, min(len(pairs), int(budget / head_entries)))
        self.locators = pairs[:count]
        self._weight = dict(zip(self.locators, weights))
        self._threshold = dict(zip(self.locators, thresholds))
        self._loc_by_layer = {}
        for layer, head in self.locators:
            self._loc_by_layer.setdefault(layer, []).append(head)
            groups = self._layers[layer].self_attn.num_heads // past_key_values[layer][0].shape[1]
            ghost = past_key_values[layer][0][0, head // groups, self.image_start: self.image_end]
            self._ghost[(layer, head)] = ghost.to(torch.float16).clone()
        self.ghost_bytes = sum(t.numel() * t.element_size() for t in self._ghost.values())
        if self.charge_seed:
            self.charge += int(math.ceil(self.ghost_bytes / float(entry_bytes)))

    def _image_columns(self, pos_row):
        return (pos_row >= self.image_start) & (pos_row < self.image_end)

    @torch.no_grad()
    def _detect(self, layer, attention, hidden, past, position_ids):
        """Virtual attention of this layer's locator heads over ALL image patches."""
        key_cache = past[0]
        kv_heads, length, dim = key_cache.shape[1], key_cache.shape[2], key_cache.shape[3]
        heads = attention.num_heads
        groups = heads // kv_heads
        query = attention.q_proj(hidden).view(1, 1, heads, dim).transpose(1, 2)        # [1, H, 1, d]
        key = attention.k_proj(hidden).view(1, 1, kv_heads, dim).transpose(1, 2)
        limit = max(int(getattr(attention, "_logical_position_limit", 0)), int(position_ids.max()) + 1)
        cos, sin = attention.rotary_emb(query, seq_len=limit)
        if groups == 1:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        else:
            query, _ = apply_rotary_pos_emb(query, query, cos, sin, position_ids)
            key, _ = apply_rotary_pos_emb(key, key, cos, sin, position_ids)

        pos = self._full_pos(layer, length, kv_heads)
        scale = 1.0 / math.sqrt(dim)
        slots = max(1, self.page_slots)

        missing_score, relevance, floor = None, None, None
        for head in self._loc_by_layer[layer]:
            kv_head = head // groups
            q = query[0, head, 0].float()
            is_image = self._image_columns(pos[kv_head])
            resident = torch.zeros(self.image_len, dtype=torch.bool, device=key_cache.device)
            resident[pos[kv_head][is_image] - self.image_start] = True
            real = (key_cache[0, kv_head].float() @ q) * scale
            real = real.masked_fill(is_image, float("-inf"))                 # image comes from the ghosts
            own = (key[0, kv_head, 0].float() @ q).view(1) * scale
            ghost = (self._ghost[(layer, head)].to(q.device).float() @ q) * scale
            virtual = torch.softmax(torch.cat([real, own, ghost]), dim=0)[-self.image_len:]
            weight, limit = self._weight[(layer, head)], self._threshold[(layer, head)]
            virtual = virtual * weight
            relevance = virtual if relevance is None else torch.maximum(relevance, virtual)
            missing = virtual.masked_fill(resident, 0.0)
            # A fault must be worth it: the head looks hard at a few missing patches.
            if float(missing.topk(min(slots, missing.numel())).values.sum()) >= weight * limit:
                missing_score = missing if missing_score is None else torch.maximum(missing_score, missing)
                floor = weight * limit if floor is None else min(floor, weight * limit)

        self._note_relevance(relevance, slots)
        demand = relevance if self._demand is None else torch.maximum(self._demand.to(relevance.device), relevance)
        self._demand = demand
        if missing_score is None or self.page_slots <= 0:
            return
        if self.fault_on == "answer" and not self._answer_begun:
            return
        if self.max_faults and self._turn["faults"] >= self.max_faults:
            return

        # Candidates: the missing patches the locator heads look at most.
        top = missing_score.topk(min(slots, missing_score.numel()))
        cumulative = top.values.cumsum(0)
        count = int((cumulative < self.cover * cumulative[-1]).sum().item()) + 1
        candidates = [p for p, v in zip(top.indices[:count].tolist(), top.values[:count].tolist()) if v > 0]
        # Admission: at most ``page_slots`` paged patches at a time, the ones in
        # highest demand.  A weak request cannot push out a strong recent one.
        pool = sorted(self._paged | set(candidates))
        order = demand[torch.as_tensor(pool, device=demand.device)].argsort(descending=True)[:slots].tolist()
        keep = {pool[i] for i in order}
        admitted = [p for p in candidates if p in keep]
        if not admitted or float(missing_score[torch.as_tensor(admitted, device=missing_score.device)].sum()) < floor:
            return
        before = self.faults
        self._fault(layer, admitted, self._paged - keep)
        if self.faults > before:
            self.fault_log.append({
                "turn": self.turn_index + 1, "token": int(self.n_logical), "answer": bool(self._answer_begun),
                "layer": layer, "patches": sorted(admitted),
                "predicted": float(missing_score[torch.as_tensor(admitted, device=missing_score.device)].sum()),
                "regenerated": int(self.regen_patches),
            })

    SHARPNESS_POWER = 4

    def _note_relevance(self, virtual, slots):
        """Accumulate the relevance map of the turn (virtual: [patches], this step)."""
        turn = self._turn
        sharp = float(virtual.topk(min(slots, virtual.numel())).values.sum())
        weighted = virtual * sharp ** self.SHARPNESS_POWER    # tokens that look hard at few patches count more
        turn["steps"] += 1
        turn["relevance"] = weighted if turn["relevance"] is None else turn["relevance"] + weighted
        if self._answer_begun:
            seen = turn["relevance_answer"]
            turn["relevance_answer"] = weighted if seen is None else seen + weighted
            if sharp > turn["peak_sharpness"]:
                turn["peak_sharpness"], turn["peak"] = sharp, virtual

    # ------------------------------------------------------------------
    # Partial regeneration
    # ------------------------------------------------------------------
    def _regen_embeds(self, patches):
        index = torch.as_tensor(patches, dtype=torch.long)
        if len(self._seed) == 1:
            embeds = self._seed[0]
            image = embeds[:, (index + self.image_start).to(embeds.device)]
            prefix = embeds[:, : self.image_start]
        else:
            prefix, features, _ = self._seed
            image = self._projector(features[:, index.to(features.device)])
            if self.use_prefix and not prefix.is_floating_point():
                prefix = self._inner.get_input_embeddings()(prefix)[None]
        if not self.use_prefix:
            return image, 0
        return torch.cat([prefix.to(image.dtype), image], dim=1), prefix.shape[1]

    @torch.no_grad()
    def _regenerate_patches(self, patches, context):
        """KV of ``patches`` (ascending) for every layer: [(K, V)] with len(patches) entries.

        context: per layer (K, V, positions, usable) of resident image entries, or None.
        A regenerated token sees the text before the image (if ``use_prefix``),
        the context entries at earlier positions, and the earlier regenerated
        patches.  Exactly the original computation when nothing is missing.
        """
        start = time.perf_counter()
        hidden, n_prefix = self._regen_embeds(patches)
        device, dtype = hidden.device, hidden.dtype
        positions = torch.cat([
            torch.arange(n_prefix, device=device),
            torch.as_tensor(patches, device=device) + self.image_start,
        ])
        tokens = positions.numel()
        blocked = torch.finfo(dtype).min
        causal = torch.full((tokens, tokens), blocked, dtype=dtype, device=device).triu(1)
        out = []
        for index, layer in enumerate(self._layers):
            attention = layer.self_attn
            attention._logical_position_limit = max(
                int(getattr(attention, "_logical_position_limit", 0)), self.n0)
            past, mask = None, causal[None, None]
            if context is not None and context[index][2].shape[1] > 0:
                key, value, ctx_pos, usable = context[index]            # per KV head: [heads, ctx]
                past = (key, value)
                visible = usable.to(device)[:, None, :] & (ctx_pos.to(device)[:, None, :] < positions[None, :, None])
                visible = visible.repeat_interleave(attention.num_heads // visible.shape[0], dim=0)
                ctx_mask = torch.zeros(visible.shape, dtype=dtype, device=device).masked_fill(~visible, blocked)
                mask = torch.cat([ctx_mask, causal[None].expand(ctx_mask.shape[0], -1, -1)], dim=2)[None]
            result = layer(
                hidden, attention_mask=mask, position_ids=positions[None],
                past_key_value=past, use_cache=True)
            hidden = result[0]
            key, value = result[-1]
            out.append((key[:, :, -len(patches):], value[:, :, -len(patches):]))
        self.regen_tokens += tokens
        self.regen_patches += len(patches)
        self._turn["regen_tokens"] += tokens
        self.regen_seconds += time.perf_counter() - start
        return out

    def _fault(self, fault_layer, wanted, evicted=()):
        """Regenerate the wanted patches that some head lacks.  Every head writes
        the ones it is missing over the paged patches in ``evicted`` and then
        over its coldest entries."""
        past = self._step_past
        wanted_pos = torch.as_tensor(sorted(wanted)) + self.image_start
        wanted = sorted(wanted)
        have = []
        for layer, (key, _) in enumerate(past):
            pos = self._full_pos(layer, key.shape[2], key.shape[1]).clone()
            self._pos[layer] = pos
            # have[h, i]: head h holds wanted patch i
            have.append((pos[:, :, None] == wanted_pos.to(pos.device)[None, None, :]).any(dim=1))
        missing_any = torch.stack([~h.all(dim=0) for h in have]).any(dim=0)
        patches = [p for i, p in enumerate(wanted) if bool(missing_any[i])]
        if not patches:
            return

        context = None
        if self.use_resident:
            context = []
            regenerated = torch.as_tensor(patches) + self.image_start
            first = self.image_start if self.use_prefix else 0      # the prefix is recomputed or read from the cache
            for layer, (key, value) in enumerate(past):
                pos = self._pos[layer]
                usable = (pos >= first) & (pos < self.image_end) & ~torch.isin(pos, regenerated.to(pos.device))
                cols = usable.any(dim=0).nonzero().flatten()
                cols_kv = cols.to(key.device)
                context.append((key[:, :, cols_kv], value[:, :, cols_kv], pos[:, cols], usable[:, cols]))

        self._busy = True
        try:
            fresh = self._regenerate_patches(patches, context)
        finally:
            self._busy = False
        where = {patch: i for i, patch in enumerate(patches)}
        self.faults += 1
        self._turn["faults"] += 1
        self._turn["paged"].update(patches)
        self._paged = (self._paged - set(evicted)) | set(patches)
        protected = torch.as_tensor(sorted(self._paged | set(wanted))) + self.image_start
        released = torch.as_tensor(sorted(evicted), dtype=torch.long) + self.image_start

        for layer, (key, _) in enumerate(past):
            pos = self._pos[layer]
            heads, length = pos.shape
            heat = torch.nn.functional.pad(self._heat[layer], (0, length - self._heat[layer].shape[1]),
                                           value=float("inf"))          # tokens of this turn are not overwritten
            heat[torch.isin(pos, released.to(pos.device)) & (heat == float("inf")) & self._image_columns(pos)] = -1.0
            priority = heat
            rows = self._window_rows(layer, length)
            if rows is not None:
                # Among the old recent-window entries, the ones this turn has read least go first.
                used = rows.sum(dim=1)
                if used.shape[0] != heads:
                    used = used.view(heads, -1, length).mean(dim=1)
                old_window = torch.isfinite(heat) & (heat >= self.WINDOW_HEAT)
                priority = torch.where(old_window, self.WINDOW_HEAT + used.to(heat.device, heat.dtype), heat)
            priority = priority.masked_fill(torch.isin(pos, protected.to(pos.device)), float("inf"))
            head_idx, col_idx, src_idx = [], [], []
            for head in range(heads):
                todo = [p for i, p in enumerate(wanted) if not bool(have[layer][head, i])]
                if not todo:
                    continue
                order = priority[head].argsort()[: len(todo)]
                order = order[torch.isfinite(priority[head][order])].tolist()
                for col, patch in zip(order, todo):
                    head_idx.append(head)
                    col_idx.append(col)
                    src_idx.append(where[patch])
                    pos[head, col] = patch + self.image_start
                    heat[head, col] = float("inf")
            self._heat[layer] = heat
            if not head_idx:
                continue
            device = key.device
            hi = torch.as_tensor(head_idx, device=device)
            ci = torch.as_tensor(col_idx, device=device)
            si = torch.as_tensor(src_idx, device=device)
            fix = (hi, ci, fresh[layer][0][0][hi, si], fresh[layer][1][0][hi, si])
            groups = self._layers[layer].self_attn.num_heads // heads
            for block in self._window[layer]:            # the attention history of these entries is void
                ok = ci < block.shape[-1]
                for g in range(groups):
                    block[(hi[ok] * groups + g).to(block.device), :, ci[ok].to(block.device)] = 0
            (self._late if layer < fault_layer else self._inflight)[layer] = fix
        self._faulted = True

    @staticmethod
    def _apply_fix(past, fix):
        """Overwrite (head, column) entries of one layer's (K, V) with paged-in patches."""
        key, value = past
        head_idx, col_idx, new_key, new_value = fix
        key, value = key.clone(), value.clone()
        key[0, head_idx, col_idx] = new_key.to(key.dtype)
        value[0, head_idx, col_idx] = new_value.to(value.dtype)
        return key, value

    def _apply_late(self, past_key_values, attentions):
        """Layers below the faulting layer ran before the fault: patch their outputs."""
        past = list(past_key_values)
        attentions = None if attentions is None else list(attentions)
        for layer, fix in self._late.items():
            past[layer] = self._apply_fix(past[layer], fix)
            if attentions is None or attentions[layer] is None:
                continue
            row = attentions[layer].clone()
            groups = row.shape[1] // past[layer][0].shape[1]
            for g in range(groups):
                row[0, (fix[0] * groups + g).to(row.device), :, fix[1].to(row.device)] = 0
            attentions[layer] = row
        self._late = {}
        return tuple(past), attentions

    # ------------------------------------------------------------------
    # Budget and compression
    # ------------------------------------------------------------------
    WINDOW_HEAT = 1e6
    DEFAULT_FOOTPRINT = 4

    def _compress(self, past_key_values, capacity, stage):
        """PyramidKV's rule: every head keeps its top entries plus the recent window."""
        lengths = [k.shape[2] for k, _ in past_key_values]
        budgets = self._budgets(capacity, lengths)
        new_past, kept, image_kept = [], [], []
        for layer, (key, value) in enumerate(past_key_values):
            length, heads, budget = lengths[layer], key.shape[1], budgets[layer]
            pos = self._full_pos(layer, length, heads)
            rows = self._window_rows(layer, length)
            heat = torch.zeros((heads, length), device=pos.device)
            index = None
            if rows is not None:
                window = max(1, min(rows.shape[1], budget - 1))
                past_len = length - window
                scores = self._scores(rows[:, -window:, :past_len], heads).to(pos.device)
                if self.paging and self.image_weight != 1.0 and self.image_start is not None:
                    scores = torch.where(self._image_columns(pos[:, :past_len]), scores * self.image_weight, scores)
                scores = scores.masked_fill(pos[:, :past_len] == 0, float("inf"))          # sink
                # The recent window is kept; a page fault may overwrite it, oldest
                # token first, only when a head has nothing colder left.
                recent = self.WINDOW_HEAT + torch.arange(window, device=scores.device, dtype=scores.dtype)
                heat = torch.cat([scores, recent.expand(heads, -1)], dim=1)
                if budget < length:
                    top = scores.topk(budget - window, dim=-1).indices.sort(dim=-1).values
                    tail = torch.arange(past_len, length, device=top.device).expand(heads, -1)
                    index = torch.cat([top, tail], dim=-1)
            elif budget < length:
                index = torch.arange(length - budget, length, device=pos.device).expand(heads, -1)
            if index is None:
                new_past.append((key, value))
            else:
                new_past.append((self._gather(key, index), self._gather(value, index)))
                pos, heat = pos.gather(1, index), heat.gather(1, index)
            self._pos[layer], self._heat[layer] = pos, heat
            kept.append(pos.shape[1])
            image_kept.append(
                float(self._image_columns(pos).float().sum(1).mean()) if self.image_start is not None else 0.0)

        self._reset_window()
        self.state = "compressed"
        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": stage,
            "turn": self.turn_index + 1,
            "logical_len": int(self.n_logical),
            "target_total": int(capacity),
            "overhead_entries_total": int(self.charge),
            "page_slots_per_head": int(self.page_slots),
            "locators": list(self.locators),
            "kv_before_mean": float(sum(lengths) / len(lengths)),
            "kv_after_mean": float(sum(kept) / len(kept)),
            "image_entries_mean": float(sum(image_kept) / len(image_kept)),
            "faults": int(self.faults),
            "regen_tokens": int(self.regen_tokens),
            "layer_budgets": kept,
        }
        return tuple(new_past)

    # ------------------------------------------------------------------
    # Calibration: which heads localise
    # ------------------------------------------------------------------
    def _calibrate_capture(self, module, args, kwargs):
        if not self._busy and kwargs.get("past_key_values") is not None:
            self._calib_kwargs = dict(kwargs)
        return None

    @torch.no_grad()
    def _calibrate_logp(self, mask):
        kwargs = dict(self._calib_kwargs)
        kwargs.update(attention_mask=mask, use_cache=False, output_attentions=False,
                      output_hidden_states=False, return_dict=True)
        self._busy = True
        try:
            hidden = self._inner(**kwargs).last_hidden_state[:, -1]
        finally:
            self._busy = False
        return torch.log_softmax(self._model.lm_head(hidden).float(), dim=-1)[0]

    def _calibrate_observe(self, attentions):
        """Score heads on one answer step of a full-cache run.

        prior : image attention mass x how peaked it is (cheap, every step).
        causal: KL between the next-token distribution with the full image and
                with the head's top patches hidden (the candidate heads only).
        """
        if attentions is None or attentions[0] is None or not self._answer_begun:
            return
        if attentions[0].shape[-2] != 1 or attentions[0].shape[-1] < self.image_end:
            return
        stats = _CALIBRATION[self.calibrate]
        image = [a[0, :, 0, self.image_start: self.image_end].float() for a in attentions]
        prior = torch.stack([concentration_score(a).cpu() for a in image])
        stats["prior"] = prior if stats["prior"] is None else stats["prior"] + prior
        stats["steps"] += 1
        hide = max(1, int(round(self.calib_hide * self.image_len)))
        if len(stats["top_mass"]) < 4096:
            stats["top_mass"].append(torch.stack([a.topk(hide, dim=-1).values.sum(-1).cpu() for a in image]))
            spread = []
            for a in image:                               # patches that hold ``cover`` of the head's image attention
                ordered = a.sort(dim=-1, descending=True).values.cumsum(-1)
                spread.append((ordered < self.cover * ordered[:, -1:]).sum(-1).add(1).cpu())
            stats["spread"].append(torch.stack(spread))
        if stats["conversations"] <= self.calib_warmup or self._calib_kwargs is None:
            return
        if stats["candidates"] is None:
            order = stats["prior"].flatten().argsort(descending=True)[: self.calib_heads].tolist()
            heads = stats["prior"].shape[1]
            stats["candidates"] = [(i // heads, i % heads) for i in order]
            stats["kl"] = torch.zeros_like(stats["prior"])
            stats["kl_n"] = torch.zeros_like(stats["prior"])
        self._calib_step += 1
        if self._calib_step > self.calib_dense and self._calib_step % self.calib_stride:
            return
        length = attentions[0].shape[-1]
        device = attentions[0].device
        full = torch.ones((1, length), dtype=torch.long, device=device)
        base = self._calibrate_logp(full)
        for layer, head in stats["candidates"]:
            mask = full.clone()
            mask[0, image[layer][head].topk(hide).indices + self.image_start] = 0
            logp = self._calibrate_logp(mask)
            stats["kl"][layer, head] += float((base.exp() * (base - logp)).sum())
            stats["kl_n"][layer, head] += 1

    def _calibrate_save(self):
        stats = _CALIBRATION.get(self.calibrate)
        if not stats or not stats["steps"]:
            return
        prior = stats["prior"] / stats["steps"]
        if stats["kl"] is not None and float(stats["kl_n"].sum()) > 0:
            score = stats["kl"] / stats["kl_n"].clamp_min(1)
        else:
            score = prior
        os.makedirs(os.path.dirname(os.path.abspath(self.calibrate)), exist_ok=True)
        top_mass = torch.stack(stats["top_mass"]).float()                  # [steps, layers, heads]
        quantiles = torch.quantile(top_mass, torch.tensor(QUANTILES), dim=0)   # [len(QUANTILES), layers, heads]
        # Footprint: the median number of patches a head reads in its sharpest 10% of steps.
        spread = torch.stack(stats["spread"]).float()
        sharp = top_mass >= quantiles[2][None]
        footprint = torch.where(sharp, spread, torch.full_like(spread, float("nan"))).nanmedian(dim=0).values
        footprint = torch.nan_to_num(footprint, nan=0.0).round().long()
        torch.save({"score": score, "prior": prior, "top_mass_quantiles": quantiles, "footprint": footprint,
                    "steps": stats["steps"], "conversations": stats["conversations"]}, self.calibrate)

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        if self.calibrate is None:
            super().observe_layer(layer_idx, attention)

    def prepare_turn(self, past_key_values, num_of_token=None):
        return past_key_values

    def relevance_map(self, which="relevance_answer"):
        """[patches] relevance of every image patch for the current turn.

        "relevance_answer": virtual attention of the locator heads summed over
        the answer tokens, each token weighted by how concentrated its look at
        the image is (default); "relevance": the same over all tokens of the
        turn; "peak": the single answer token that looked hardest at the image.
        """
        return self._turn[which]

    def _save_turn_maps(self):
        store = _MAPS.get(self.save_maps) if self.save_maps else None
        if store is None:
            return
        turn = self._turn

        def cpu(value):
            return None if value is None else value.detach().float().cpu()

        store["turns"].append({
            "conversation": store["conversations"] - 1, "turn": self.turn_index + 1,
            "relevance_answer": cpu(turn["relevance_answer"]), "relevance": cpu(turn["relevance"]),
            "peak": cpu(turn["peak"]), "faults": turn["faults"], "regenerated": sorted(turn["paged"]),
            "regen_tokens": turn["regen_tokens"], "locators": list(self.locators),
        })
        os.makedirs(os.path.dirname(os.path.abspath(self.save_maps)), exist_ok=True)
        torch.save(store, self.save_maps)

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("GhostKVCache requires num_of_token for ratio-based pruning.")
        if self.calibrate is not None:
            self._calibrate_observe(attentions)
            return past_key_values
        if self.ratio <= 0:
            return past_key_values
        if len(past_key_values) != self.layer_num:
            raise ValueError(f"Expected {self.layer_num} layers, got {len(past_key_values)}")

        if self._late:
            past_key_values, attentions = self._apply_late(past_key_values, attentions)
        self._inflight = {}
        self._step_past = None
        if attentions is not None and attentions[0] is not None:
            for layer, attention in enumerate(attentions):
                self.observe_layer(layer, attention)
        self.n_logical = int(num_of_token)

        if self._pending_prefill:
            self._pending_prefill = False
            first = past_key_values[0][0]
            if first.shape[0] != 1:
                raise ValueError("GhostKVCache supports batch size 1 only")
            self.n0 = int(first.shape[2])
            if self.paging and self.image_start is not None and self.image_len > 0:
                self._build_seed(past_key_values)
                self._build_ghost(past_key_values)
                base = self._capacity(self.n0) // self.layer_num
                if self.page_slots_arg is not None:
                    self.page_slots = self.page_slots_arg
                else:
                    # A share of the layer budget, but room for one whole "look" of the best head.
                    look = int(math.ceil((self.footprint or self.DEFAULT_FOOTPRINT) / max(self.cover, 1e-6)))
                    wanted = max(int(self.page_frac * base), look)
                    self.page_slots = max(1, min(wanted, base - 1))
            self._pos = [
                torch.arange(self.n0, device=key.device).view(1, -1).expand(key.shape[1], -1)
                for key, _ in past_key_values]
            self._heat = [None] * self.layer_num
            self.boundary_logical = self.n0
            return self._compress(past_key_values, self._capacity(self.n_logical), "prefill")

        if self.state == "init":
            raise ValueError(
                "GhostKVCache must observe the multimodal prefill first "
                "(output_attentions=True or the online attention hook).")
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Turn boundary: compress image and dialogue text back to the budget."""
        if self.calibrate is not None:
            self._calibrate_save()
            self._calib_step = 0
            self._answer_begun = False
            return past_key_values
        if past_key_values is None or self.ratio <= 0 or self.state == "init":
            return past_key_values
        self.boundary_logical = int(self.n_logical)
        self.turn_log.append(self._turn)
        self._save_turn_maps()
        result = self._compress(past_key_values, self._capacity(self.n_logical), "turn_boundary")
        self._new_turn()
        self._answer_begun = False
        self.turn_index += 1
        return result
