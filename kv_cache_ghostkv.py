"""
GhostKV: regenerate only the image patches the answer needs and the cache lacks.

How it works
------------
1. Retention.  After the Turn-1 prefill and at every turn boundary the cache is
   cut to the budget with PyramidKV's rule (every head keeps its top entries
   and the recent window).

2. Ghost keys.  For a few "locator" heads (found once per model by
   kv_cache_ghostkv_calibrate.py) the KEY of every image patch is kept.  While
   the model answers, each locator head's query is scored against those keys:
   the attention it WOULD pay to every patch if the whole image were cached.
   Summed over the answer tokens this is the relevance map of the turn.

3. Page fault.  If a locator head puts more than its threshold of that
   attention on patches that are not in the cache, those patches are
   regenerated from the seed (the text before the image plus just those
   patches, with the image entries still in the cache as context) and written
   over the coldest image entries of every head.  Layers from the faulting one
   upward get them inside the running forward pass, the shallower ones after
   the step.  The cache never grows, and text is never overwritten.

4. Budget per turn.  Detection runs on the first 16 answer
   tokens and stops after ``max_faults`` regenerations; after that a decoding
   step costs what it costs any other cache.

Memory: C(n) = int(n * (1 - ratio)) entries per layer on average, minus the
seed and the ghost keys (both charged in KV entries).

Interface: the same as SeedKVCache (attach_model, __call__, begin_answer,
finalize_turn).  Batch size 1, one contiguous image block.
"""
import functools
import math
import os

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from kv_cache_seedkv import SeedKVCache

COVER = 0.9            # a fault pages the missing patches that hold this share of the missing attention
GHOST_SHARE = 0.02     # the ghost keys may use this share of the KV budget (at least one head)
PAGE_FRAC = 0.3        # patches per fault: this share of a layer's budget, or one "look" of the best head
N_LOCATORS = 4         # at most this many locator heads
DETECT_TOKENS = 32     # faults are looked for on the first answer tokens of a turn only
SHARPNESS_POWER = 4    # relevance map: tokens that look hard at few patches count more

_LIVE_HOOKS = []       # hooks of the conversation that is running
_MAPS = {}             # path -> {"conversations": int, "turns": [dict]}


def load_locators(path, count):
    """The ``count`` best heads of a calibration file, best first.

    Each is {"layer", "head", "weight" (score relative to the best head),
    "level" (attention on its top patches in its sharpest 10% of answer tokens),
    "footprint" (patches it reads at such a token)}.
    """
    blob = torch.load(path, map_location="cpu")
    score = blob["score"]
    heads = score.shape[1]
    order = score.flatten().argsort(descending=True)[:count].tolist()
    best = max(float(score.flatten()[order[0]]), 1e-9)
    return [{
        "layer": index // heads, "head": index % heads,
        "weight": float(score.flatten()[index]) / best,
        "level": float(blob["top_mass_quantiles"][2, index // heads, index % heads]),
        "footprint": int(blob["footprint"][index // heads, index % heads]),
    } for index in order]


class GhostKVCache(SeedKVCache):

    def __init__(self, layer_num, ratio=0.0, image_start=None, image_token_len=0,
                 locators=None, max_faults=2, paging=True, save_maps=None):
        self.locator_file = locators              # made once per model by kv_cache_ghostkv_calibrate.py
        self.max_faults = int(max_faults)         # regenerations per turn (0: no limit)
        self.paging = bool(paging)                # False: retention only (ablation)
        self.save_maps = save_maps                # file that collects the relevance map of every turn
        self._page_hooks = []
        super().__init__(layer_num, ratio, image_start, image_token_len, regenerate=self.paging)
        self.image_end = None if self.image_start is None else self.image_start + self.image_len

    # ------------------------------------------------------------------
    # State and hooks
    # ------------------------------------------------------------------
    def reset(self):
        super().reset()
        for handle in getattr(self, "_page_hooks", []):
            handle.remove()
        self._page_hooks = []
        self.locators = []         # [(layer, head)]
        self._ghost = {}           # layer -> {"heads", "keys" [h, patches, d], "weight" [h], "floor" [h]}
        self.ghost_bytes = 0
        self.page_slots = 0        # patches one fault may bring in
        self._heat = None          # per layer [heads, entries]: the score that kept an entry (inf: protected)
        self._step_past = None     # the KV the running forward pass reads
        self._detecting = False
        self._answer_steps = 0
        self._inflight = {}        # layer -> fix for the running forward pass
        self._late = {}            # layer -> fix for layers that ran before the fault
        self.faults = 0
        self.regen_tokens = 0      # tokens run through the model to regenerate
        self.turn_log = []         # one dict per finished turn
        self.fault_log = []        # one dict per page fault
        self._new_turn()

    def _new_turn(self):
        self._turn = {"faults": 0, "regen_tokens": 0, "paged": set(),
                      "relevance_answer": None, "peak": None, "peak_sharpness": 0.0}
        self._answer_steps = 0

    def attach_model(self, model):
        super().attach_model(model)                      # captures the seed at the prefill
        while _LIVE_HOOKS:
            _LIVE_HOOKS.pop().remove()
        if self.ratio <= 0 or not self.paging:
            return
        if self.save_maps:
            _MAPS.setdefault(self.save_maps, {"conversations": 0, "turns": []})["conversations"] += 1
        self._layers = list(self._inner.layers)
        self._page_hooks = [self._inner.register_forward_pre_hook(self._step_begin, with_kwargs=True)]
        for index, layer in enumerate(self._layers):
            self._page_hooks.append(layer.self_attn.register_forward_pre_hook(
                functools.partial(self._attention_pre, index), with_kwargs=True))
        _LIVE_HOOKS.extend(self._page_hooks)

    def _step_begin(self, module, args, kwargs):
        """Before every forward pass: is this a token on which faults are looked for?"""
        if self._busy:
            return None
        self._step_past = kwargs.get("past_key_values")
        self._inflight = {}
        self._detecting = False
        if self._answer_begun and self.state == "compressed" and self._ghost:
            self._answer_steps += 1
            spent = self.max_faults and self._turn["faults"] >= self.max_faults
            late = self._answer_steps > DETECT_TOKENS
            self._detecting = not (spent or late)
        return None

    def _attention_pre(self, index, module, args, kwargs):
        """Before a layer's attention: look for a fault, hand it the paged-in patches."""
        if self._busy or not (self._detecting or self._inflight):
            return None
        hidden, past = kwargs.get("hidden_states"), kwargs.get("past_key_value")
        if hidden is None or past is None or hidden.shape[1] != 1 or kwargs.get("position_ids") is None:
            return None
        if self._detecting and index in self._ghost:
            self._detect(index, module, hidden, past, kwargs["position_ids"])
        fix = self._inflight.get(index)
        if fix is None:
            return None
        kwargs = dict(kwargs)
        kwargs["past_key_value"] = self._apply_fix(past, fix)
        return args, kwargs

    # ------------------------------------------------------------------
    # Ghost keys
    # ------------------------------------------------------------------
    def _build_ghost(self, past_key_values):
        """At the prefill: keep the image keys of the locator heads, size the page pool."""
        if not self.locator_file or not os.path.exists(str(self.locator_file)):
            raise FileNotFoundError(
                f"GhostKV needs a locator file (got {self.locator_file!r}). Make it once per model with "
                "--ghostkv-calibrate FILE, or run with --ghostkv-no-paging.")
        key = past_key_values[0][0]
        if self._layers[0].self_attn.num_heads != key.shape[1]:
            raise ValueError("GhostKVCache expects one KV head per attention head")
        rows = load_locators(self.locator_file, N_LOCATORS)
        entry_bytes = 2 * key.shape[1] * key.shape[3] * key.element_size()       # one KV entry of one layer
        head_entries = self.image_len * key.shape[3] * 2 / float(entry_bytes)    # fp16 ghost keys of one head
        budget = GHOST_SHARE * self.layer_num * int(self.n0 * (1.0 - self.ratio))
        rows = rows[: max(1, int(budget / head_entries))]
        self.locators = [(row["layer"], row["head"]) for row in rows]
        for layer in sorted({row["layer"] for row in rows}):
            mine = [row for row in rows if row["layer"] == layer]
            keys = past_key_values[layer][0]
            heads = torch.tensor([row["head"] for row in mine], device=keys.device)
            weight = torch.tensor([row["weight"] for row in mine], device=keys.device)
            # "Looking hard": half of what the head puts on its top patches in its sharpest tokens.
            limit = torch.tensor([
                min(0.9, max(0.02, 0.5 * row["level"]))
                for row in mine], device=keys.device)
            self._ghost[layer] = {
                "heads": heads, "weight": weight, "floor": weight * limit,
                "keys": keys[0, heads, self.image_start: self.image_end].clone(),
            }
        self.ghost_bytes = len(rows) * self.image_len * key.shape[3] * 2
        if self.charge_seed:
            self.charge += int(math.ceil(self.ghost_bytes / float(entry_bytes)))
        base = self._capacity(self.n0) // self.layer_num
        look = int(math.ceil(max(1, rows[0]["footprint"]) / COVER))
        self.page_slots = max(1, min(max(int(PAGE_FRAC * base), look), base - 1, self.image_len))

    def _is_image(self, pos):
        return (pos >= self.image_start) & (pos < self.image_end)

    @torch.no_grad()
    def _detect(self, layer, attention, hidden, past, position_ids):
        """Virtual attention of this layer's locator heads over ALL image patches."""
        key_cache = past[0]
        heads, dim = key_cache.shape[1], key_cache.shape[3]
        query = attention.q_proj(hidden).view(1, 1, heads, dim).transpose(1, 2)        # [1, H, 1, d]
        key = attention.k_proj(hidden).view(1, 1, heads, dim).transpose(1, 2)
        limit = max(int(getattr(attention, "_logical_position_limit", 0)), int(position_ids.max()) + 1)
        cos, sin = attention.rotary_emb(query, seq_len=limit)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        ghost = self._ghost[layer]
        index = ghost["heads"]
        q = query[0, index, 0]                                                          # [h, d]
        pos = self._full_pos(layer, key_cache.shape[2], heads).to(key_cache.device)[index]
        is_image = self._is_image(pos)
        scale = 1.0 / math.sqrt(dim)
        real = torch.bmm(key_cache[0, index], q[:, :, None])[:, :, 0].float() * scale
        real = real.masked_fill(is_image, float("-inf"))                 # image: from the ghost keys
        own = (key[0, index, 0] * q).sum(-1, keepdim=True).float() * scale
        image = torch.bmm(ghost["keys"], q[:, :, None])[:, :, 0].float() * scale
        virtual = torch.softmax(torch.cat([real, own, image], dim=1), dim=1)[:, -self.image_len:]
        virtual = virtual * ghost["weight"][:, None]                                    # [h, patches]

        self._note_relevance(virtual.max(dim=0).values)
        resident = torch.zeros((pos.shape[0], self.image_len + 1), dtype=torch.bool, device=pos.device)
        resident.scatter_(1, torch.where(is_image, pos - self.image_start, self.image_len), True)
        missing = virtual.masked_fill(resident[:, : self.image_len], 0.0)
        # A fault must be worth it: a head looks hard at a few missing patches.
        triggered = missing.topk(self.page_slots, dim=1).values.sum(1) >= ghost["floor"]
        if not bool(triggered.any()):                    # the only GPU->CPU sync of a step without a fault
            return
        top = (missing * triggered[:, None]).max(dim=0).values.topk(self.page_slots)
        cumulative = top.values.cumsum(0)
        count = int((cumulative < COVER * cumulative[-1]).sum()) + 1
        wanted = sorted(p for p, v in zip(top.indices[:count].tolist(), top.values[:count].tolist()) if v > 0)
        if wanted and self._fault(layer, wanted):
            self.fault_log.append({"turn": self.turn_index + 1, "token": int(self.n_logical), "layer": layer,
                                   "patches": wanted, "predicted": float(top.values[:count].sum())})

    def _note_relevance(self, virtual):
        """Add one answer token to the relevance map of the turn (no sync)."""
        turn = self._turn
        sharp = virtual.topk(self.page_slots).values.sum()
        weighted = virtual * sharp ** SHARPNESS_POWER
        if turn["peak"] is None:
            turn["relevance_answer"], turn["peak"], turn["peak_sharpness"] = weighted, virtual, sharp
        else:
            turn["relevance_answer"] = turn["relevance_answer"] + weighted
            turn["peak"] = torch.where(sharp > turn["peak_sharpness"], virtual, turn["peak"])
            turn["peak_sharpness"] = torch.maximum(sharp, turn["peak_sharpness"])

    # ------------------------------------------------------------------
    # Page fault: regenerate, then overwrite the coldest image entries
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _regenerate_patches(self, patches, context):
        """KV of ``patches`` (ascending) in every layer: [(K, V)] with len(patches) entries.

        The patches see the text before the image, the context entries at earlier
        positions (per layer: K, V, positions, usable; or None) and each other.
        Exactly the original computation when no earlier patch is missing.
        """
        index = torch.as_tensor(patches, dtype=torch.long)
        if len(self._seed) == 1:                                    # seed stored as input embeddings
            embeds = self._seed[0]
            prefix = embeds[:, : self.image_start]
            image = embeds[:, (index + self.image_start).to(embeds.device)]
        else:                                                       # token ids + vision features
            prefix, features, _ = self._seed
            image = self._projector(features[:, index.to(features.device)])
            if not prefix.is_floating_point():
                prefix = self._inner.get_input_embeddings()(prefix)[None]
        hidden = torch.cat([prefix.to(image.dtype), image], dim=1)
        device, dtype = hidden.device, hidden.dtype
        positions = torch.cat([torch.arange(prefix.shape[1], device=device),
                               index.to(device) + self.image_start])
        tokens = positions.numel()
        blocked = torch.finfo(dtype).min
        causal = torch.full((tokens, tokens), blocked, dtype=dtype, device=device).triu(1)
        out = []
        for number, layer in enumerate(self._layers):
            attention = layer.self_attn
            attention._logical_position_limit = max(int(getattr(attention, "_logical_position_limit", 0)), self.n0)
            past, mask = None, causal[None, None]
            if context is not None and context[number][2].shape[1] > 0:
                key, value, ctx_pos, usable = context[number]
                past = (key, value)
                visible = usable.to(device)[:, None, :] & (ctx_pos.to(device)[:, None, :] < positions[None, :, None])
                ctx_mask = torch.zeros(visible.shape, dtype=dtype, device=device).masked_fill(~visible, blocked)
                mask = torch.cat([ctx_mask, causal[None].expand(ctx_mask.shape[0], -1, -1)], dim=2)[None]
            result = layer(hidden, attention_mask=mask, position_ids=positions[None],
                           past_key_value=past, use_cache=True)
            hidden = result[0]
            key, value = result[-1]
            out.append((key[:, :, -len(patches):], value[:, :, -len(patches):]))
        self.regen_tokens += tokens
        self._turn["regen_tokens"] += tokens
        return out

    def _fault(self, fault_layer, wanted):
        """Regenerate the wanted patches that some head lacks; every head writes the
        ones it is missing over its coldest image entries.  Returns True if it did."""
        past = self._step_past
        wanted_pos = torch.as_tensor(wanted) + self.image_start
        lacks = []                                       # per layer [heads, wanted]: head lacks the patch
        for layer, (key, _) in enumerate(past):
            pos = self._full_pos(layer, key.shape[2], key.shape[1]).clone()
            self._pos[layer] = pos
            lacks.append(~(pos[:, :, None] == wanted_pos.to(pos.device)[None, None, :]).any(dim=1))
        lacking = torch.stack([l.any(dim=0) for l in lacks]).any(dim=0)
        patches = [p for p, missing in zip(wanted, lacking.tolist()) if missing]
        if not patches:
            return False

        # Context: the image entries that are in the cache and are not being regenerated.
        new_pos = torch.as_tensor(patches) + self.image_start
        context = []
        for layer, (key, value) in enumerate(past):
            pos = self._pos[layer]
            usable = self._is_image(pos) & ~torch.isin(pos, new_pos.to(pos.device))
            cols = usable.any(dim=0).nonzero().flatten()
            context.append((key[:, :, cols.to(key.device)], value[:, :, cols.to(key.device)],
                            pos[:, cols], usable[:, cols]))
        self._busy = True
        try:
            fresh = self._regenerate_patches(patches, context)
        finally:
            self._busy = False
        source = lacking.long().cumsum(0) - 1            # index of a wanted patch among the regenerated
        self.faults += 1
        self._turn["faults"] += 1
        self._turn["paged"].update(patches)

        for layer, (key, _) in enumerate(past):
            pos = self._pos[layer]
            heat = torch.nn.functional.pad(self._heat[layer], (0, pos.shape[1] - self._heat[layer].shape[1]),
                                           value=float("inf"))
            self._heat[layer] = heat
            if not bool(lacks[layer].any()):
                continue
            # Victims: image entries only, coldest first; never a wanted or an already paged-in patch.
            priority = heat.masked_fill(~self._is_image(pos) | torch.isin(pos, wanted_pos.to(pos.device)),
                                        float("inf"))
            order = priority.argsort(dim=1)[:, : len(wanted)]
            if order.shape[1] < len(wanted):
                order = torch.nn.functional.pad(order, (0, len(wanted) - order.shape[1]))
            rank = (lacks[layer].long().cumsum(dim=1) - 1).clamp(min=0)      # its j-th missing patch ...
            column = order.gather(1, rank)                                   # ... goes over its j-th coldest entry
            valid = lacks[layer] & torch.isfinite(priority.gather(1, column))
            head_sel, want_sel = valid.nonzero(as_tuple=True)
            if head_sel.numel() == 0:
                continue
            col_sel = column[head_sel, want_sel]
            pos[head_sel, col_sel] = wanted_pos.to(pos.device)[want_sel]
            heat[head_sel, col_sel] = float("inf")
            device = key.device
            hi, ci, si = head_sel.to(device), col_sel.to(device), source.to(pos.device)[want_sel].to(device)
            fix = (hi, ci, fresh[layer][0][0][hi, si], fresh[layer][1][0][hi, si])
            for block in self._window[layer]:            # the attention history of these entries is void
                ok = ci < block.shape[-1]
                block[hi[ok].to(block.device), :, ci[ok].to(block.device)] = 0
            (self._late if layer < fault_layer else self._inflight)[layer] = fix
        return True

    @staticmethod
    def _apply_fix(past, fix):
        """Overwrite (head, column) entries of one layer's (K, V) with paged-in patches."""
        key, value = past[0].clone(), past[1].clone()
        head_idx, col_idx, new_key, new_value = fix
        key[0, head_idx, col_idx] = new_key.to(key.dtype)
        value[0, head_idx, col_idx] = new_value.to(value.dtype)
        return key, value

    def _apply_late(self, past_key_values, attentions):
        """Layers below the faulting layer ran before the fault: patch their outputs."""
        past = list(past_key_values)
        attentions = None if attentions is None else list(attentions)
        for layer, fix in self._late.items():
            past[layer] = self._apply_fix(past[layer], fix)
            if attentions is not None and attentions[layer] is not None:
                row = attentions[layer].clone()
                row[0, fix[0].to(row.device), :, fix[1].to(row.device)] = 0
                attentions[layer] = row
        self._late = {}
        return tuple(past), attentions

    # ------------------------------------------------------------------
    # Retention (PyramidKV's rule, per head)
    # ------------------------------------------------------------------
    def _compress(self, past_key_values, stage):
        capacity = self._capacity(self.n_logical)
        lengths = [k.shape[2] for k, _ in past_key_values]
        budgets = self._budgets(capacity, lengths)
        new_past, kept = [], []
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
                scores = scores.masked_fill(pos[:, :past_len] == 0, float("inf"))          # attention sink
                heat = torch.cat([scores, scores.new_full((heads, window), float("inf"))], dim=1)
                if budget < length:
                    top = scores.topk(budget - window, dim=-1).indices.sort(dim=-1).values
                    tail = torch.arange(past_len, length, device=top.device).expand(heads, -1)
                    index = torch.cat([top, tail], dim=-1)
            elif budget < length:
                index = torch.arange(length - budget, length, device=pos.device).expand(heads, -1)
            if index is not None:
                key, value = self._gather(key, index), self._gather(value, index)
                pos, heat = pos.gather(1, index), heat.gather(1, index)
            new_past.append((key, value))
            self._pos[layer], self._heat[layer] = pos, heat
            kept.append(pos.shape[1])
        self._reset_window()
        self.state = "compressed"
        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": stage, "turn": self.turn_index + 1, "logical_len": int(self.n_logical),
            "target_total": int(capacity), "overhead_entries_total": int(self.charge),
            "kv_before_mean": float(sum(lengths) / len(lengths)), "kv_after_mean": float(sum(kept) / len(kept)),
            "locators": list(self.locators), "page_slots": int(self.page_slots),
            "faults": int(self.faults), "regen_tokens": int(self.regen_tokens), "layer_budgets": kept,
        }
        return tuple(new_past)

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def prepare_turn(self, past_key_values, num_of_token=None):
        return past_key_values                           # nothing is regenerated up front

    def relevance_map(self, which="relevance_answer"):
        """[patches] relevance of every image patch for the current turn.

        "relevance_answer": the locator heads' virtual attention summed over the
        checked answer tokens; "peak": the single token that looked hardest.
        """
        return self._turn[which]

    def _save_turn_maps(self):
        store = _MAPS.get(self.save_maps) if self.save_maps else None
        if store is None:
            return
        turn = self._turn
        store["turns"].append({
            "conversation": store["conversations"] - 1, "turn": self.turn_index + 1,
            "relevance_answer": None if turn["peak"] is None else turn["relevance_answer"].float().cpu(),
            "peak": None if turn["peak"] is None else turn["peak"].float().cpu(),
            "faults": turn["faults"], "regenerated": sorted(turn["paged"]),
            "regen_tokens": turn["regen_tokens"], "locators": list(self.locators),
        })
        os.makedirs(os.path.dirname(os.path.abspath(self.save_maps)), exist_ok=True)
        torch.save(store, self.save_maps)

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None or self.ratio <= 0:
            return past_key_values
        if num_of_token is None:
            raise ValueError("GhostKVCache requires num_of_token for ratio-based pruning.")
        if self._late:
            past_key_values, attentions = self._apply_late(past_key_values, attentions)
        self._inflight, self._step_past = {}, None
        if attentions is not None and attentions[0] is not None:
            for layer, attention in enumerate(attentions):
                self.observe_layer(layer, attention)
        self.n_logical = int(num_of_token)

        if self._pending_prefill:                        # the Turn-1 prefill
            self._pending_prefill = False
            first = past_key_values[0][0]
            if first.shape[0] != 1:
                raise ValueError("GhostKVCache supports batch size 1 only")
            self.n0 = int(first.shape[2])
            if self.paging and self.image_start is not None and self.image_len > 0:
                self._build_seed(past_key_values)
                self._build_ghost(past_key_values)
            self._pos = [torch.arange(self.n0, device=key.device).view(1, -1).expand(key.shape[1], -1)
                         for key, _ in past_key_values]
            self._heat = [None] * self.layer_num
            self.boundary_logical = self.n0
            return self._compress(past_key_values, "prefill")
        if self.state == "init":
            raise ValueError("GhostKVCache must observe the multimodal prefill first "
                             "(output_attentions=True or the online attention hook).")
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Turn boundary: cut image and dialogue back to the budget."""
        if past_key_values is None or self.ratio <= 0 or self.state == "init":
            return past_key_values
        self.boundary_logical = int(self.n_logical)
        self._turn["peak_sharpness"] = float(self._turn["peak_sharpness"])
        self.turn_log.append(self._turn)
        self._save_turn_maps()
        result = self._compress(past_key_values, "turn_boundary")
        self._new_turn()
        self._answer_begun = False
        self.turn_index += 1
        return result
