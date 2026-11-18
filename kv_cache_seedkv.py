"""
SeedKV: evict the image to a seed, regrow it when the next question arrives.

Problem
-------
Every cache policy in this repo is retain-only.  It decides which visual
entries survive right after the Turn-1 prefill and at each turn boundary,
always BEFORE the next question exists, and an evicted entry can never come
back.  Later turns therefore read whatever an earlier question happened to
keep.

Observation
-----------
In a multimodal conversation the prefill prompt ("<system> USER: Q1 <image>
ASSISTANT:") sits at the very start of the context, so its KV entries depend
on nothing that comes later.  They can be regenerated EXACTLY, at any turn,
by running the frozen LLM once over the prompt's input embeddings.  What is
needed to rebuild those embeddings (the "seed") is tiny next to the cache:
for LLaVA-1.5-7B the 576 CLIP features and the prompt's token ids are about
1.2 MB in 16-bit floats, against about 300 MB for the image KV over 32 layers.

Method
------
  1. Turn-1 prefill.  Keep the seed (vision features + prompt token ids).
     Compress the prefill to the budget with the question in view.
  2. Turn boundary (``finalize_turn``).  Drop EVERY prefill entry.  The whole
     budget goes to dialogue text, which cannot be regenerated cheaply.
  3. Next turn (``prepare_turn``).  Regenerate the full prefill KV from the
     seed, with one batched forward of the prefill length.
  4. Commit (first call after ``begin_answer``).  The new question has now
     been read with the whole image visible.  Select image and text entries
     together from its attention, down to the same number of entries a
     boundary-only method holds at that moment.

So the image is never irreversibly evicted, every turn selects visual entries
with its own question, and selection errors do not carry over to later turns.

Budget
------
C(n) = int(n * (1 - ratio)) entries per layer on average, like every other
method here, MINUS the seed's own size (in KV entries), so that KV + seed
never exceeds L * C:

  * after the Turn-1 prefill   : L * C(prefill length)
  * at rest between turns      : at most L * C(logical length), text only
  * after the commit of a turn : L * (C(length at last boundary) + tokens
                                 appended since), i.e. what PyramidKV holds

Per-layer budgets follow PyramidKV's linear pyramid with an exact total.
``allocation="pyramidkv"`` reproduces this repo's PyramidKVCache schedule
instead (it rounds the slope down and so holds more than L * C);
``allocation="uniform"`` gives every layer the same budget.  Inside a layer
every KV head keeps its own entries; all heads keep the same NUMBER of image
and text entries, so the tensors stay rectangular.

The one moment above budget is the question of a later turn: the full prefill
KV is resident while the question tokens are read (the same size as the
Turn-1 prefill that every method holds before its first compression).

Cost
----
One extra forward over the prefill (about TTFT of Turn 1) per later turn.
No training, no extra generated tokens, no change to the attention function.
Scoring is PyramidKV's rule (attention of the last ``window_size`` queries,
1-D max-pool); what is new is where and when it is applied.

Interface (eval_ppl_final.py / eval_rouge_final2.py / eval_efficiency.py)
------------------------------------------------------------------------
    cache = SeedKVCache(layer_num, ratio, image_start=..., image_token_len=...)
    cache.attach_model(model)                      # before the prefill forward
    past = cache(past, num_of_token, attentions)   # after every forward
    cache.begin_answer()                           # once per turn, before the answer
    past = cache.finalize_turn(past)               # at every turn boundary
    past = cache.prepare_turn(past, num_of_token)  # before a later turn's tokens

``prepare_turn`` is optional: without it the cache regenerates the prefill
inside the first ``__call__`` of the turn, so that one token is computed
without the image.  ``begin_answer`` tells the cache that the question has
been read; a harness that cannot call it should set ``max_hydrated_tokens``
so the commit happens after that many tokens instead.
"""

import math
import time
import warnings

import torch
import torch.nn.functional as F

_WARNED = set()


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=3)


# ----------------------------------------------------------------------
# Per-layer budgets
# ----------------------------------------------------------------------
def layer_budgets(n_layers, total, window, lengths, allocation="pyramid", beta=20.0):
    """Entries kept per layer (window included) for ``total`` entries over all layers.

    ``pyramid`` is PyramidKV's linear schedule (KVCache-Factory) with a float
    slope and largest-remainder rounding, so the sum is exactly ``total``
    whenever every layer is long enough.  A layer never gets more than it has.
    """
    q_len = max(lengths)
    if allocation == "pyramidkv" and n_layers > 1:
        # The schedule of this repo's PyramidKVCache, integer step included.  Its
        # sum is above ``total`` (about +12% for LLaVA-1.5-7B at ratio 0.9); use it
        # to hold exactly as many entries per layer as that baseline does.
        past_average = max(0, int(total) // n_layers - window)
        if q_len < past_average * 2:
            caps = [past_average] * n_layers
        else:
            min_num = int(past_average // beta)
            max_num = past_average * 2 - min_num
            if max_num >= q_len - window:
                max_num = q_len - window
                min_num = past_average * 2 - max_num
            steps = (max_num - min_num) // (n_layers - 1)
            caps = [max_num - layer * steps for layer in range(n_layers)]
        return [min(max(0, cap) + window, length) for cap, length in zip(caps, lengths)]
    past_total = max(0, int(total) - n_layers * window)
    average = past_total / float(n_layers)
    if allocation == "uniform" or n_layers < 2 or q_len < average * 2:
        values = [average] * n_layers
    else:
        min_num = float(int(average // beta))
        max_num = average * 2 - min_num
        if max_num >= q_len - window:
            max_num = max(float(q_len - window), average)
            min_num = average * 2 - max_num
        slope = (max_num - min_num) / (n_layers - 1)
        values = [max_num - layer * slope for layer in range(n_layers)]
    caps = [int(x) for x in values]
    order = sorted(range(n_layers), key=lambda i: values[i] - caps[i], reverse=True)
    for layer in order[:max(0, past_total - sum(caps))]:
        caps[layer] += 1
    return [min(cap + window, length) for cap, length in zip(caps, lengths)]


class SeedKVCache:
    """Multi-turn KV eviction that regenerates the prefill instead of retaining it."""

    def __init__(
        self,
        layer_num,
        ratio=0.0,
        image_start=None,
        image_token_len=0,
        k_seq_dim=2,
        v_seq_dim=2,
        window_size=8,
        kernel_size=7,
        pooling="maxpool",
        allocation="pyramid",
        beta=20.0,
        seed="auto",
        charge_seed=True,
        max_hydrated_tokens=0,
        refresh_every=0,
        regenerate=True,
    ):
        if k_seq_dim != 2 or v_seq_dim != 2:
            raise ValueError("SeedKVCache expects KV layout [batch, heads, seq, dim]")
        if not 0.0 <= float(ratio) < 1.0:
            raise ValueError("ratio must satisfy 0 <= ratio < 1")
        if int(window_size) < 1:
            raise ValueError("window_size must be at least 1")
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if pooling not in {"maxpool", "avgpool"}:
            raise ValueError("pooling must be 'maxpool' or 'avgpool'")
        if allocation not in {"pyramid", "pyramidkv", "uniform"}:
            raise ValueError("allocation must be 'pyramid', 'pyramidkv' or 'uniform'")
        if seed not in {"auto", "features", "embeds"}:
            raise ValueError("seed must be 'auto', 'features' or 'embeds'")
        self.layer_num = int(layer_num)
        self.ratio = float(ratio)
        self.image_start = None if image_start is None else int(image_start)
        self.image_len = int(image_token_len)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.allocation = allocation
        self.beta = float(beta)
        self.seed_mode = seed
        self.charge_seed = bool(charge_seed)
        self.max_hydrated_tokens = int(max_hydrated_tokens)
        self.refresh_every = int(refresh_every)
        # regenerate=False is the ablation: same scoring, budgets and code, but
        # retain-only (the prefill entries that survive are kept across turns).
        self.regenerate = bool(regenerate)
        self._inner = None
        self._projector = None
        self._hooks = []
        self.reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state at the beginning of a NEW conversation."""
        self._remove_hooks()
        self.state = "init"            # init -> compressed -> rest -> hydrated -> compressed ...
        self._window = None            # per layer: list of [heads, rows, keys] attention blocks
        self._pos = None               # per layer: [kv_heads, entries] logical positions
        self._a_count = None           # per layer: prefill entries at the front of the layout
        self._pending_prefill = False
        self._raw_embeds = None
        self._raw_features = None
        self._raw_ids = []
        self._seed = None
        self._busy = False
        self._answer_begun = False
        self._refreshing = False
        self.n0 = 0                    # prefill length = the regenerable region
        self.n_logical = 0
        self.boundary_logical = 0
        self.hydrated_tokens = 0
        self.tokens_since_commit = 0
        self.turn_index = 0
        self.seed_bytes = 0
        self.seed_kind = None
        self.charge = 0                # seed size in KV entries (one K and one V of one layer)
        self.rehydrations = 0
        self.rehydrate_seconds = 0.0
        self.last_layer_budgets = None
        self.last_finalize_stats = None

    def _remove_hooks(self):
        for handle in getattr(self, "_hooks", []):
            handle.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # The seed
    # ------------------------------------------------------------------
    def attach_model(self, model):
        """Give the cache the model (to regenerate) and capture the seed at prefill."""
        self._inner = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", model)
        self._remove_hooks()
        if self.ratio <= 0 or not self.regenerate:
            return
        self._hooks.append(
            self._inner.register_forward_pre_hook(self._capture_embeds, with_kwargs=True))
        self._projector = getattr(self._inner, "mm_projector", None)
        if self._projector is not None and self.seed_mode != "embeds":
            self._hooks.append(self._projector.register_forward_hook(self._capture_features))
            self._hooks.append(
                self._inner.get_input_embeddings().register_forward_hook(self._capture_ids))

    def _capture_embeds(self, module, args, kwargs):
        if self._busy or self._raw_embeds is not None or kwargs.get("past_key_values") is not None:
            return None
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is None or input_ids.shape[1] < 2:
                return None
            embeds = module.get_input_embeddings()(input_ids)
        if embeds.shape[1] < 2:
            return None
        self._raw_embeds = embeds.detach()
        return None

    def _capture_features(self, module, args, output):
        if self._busy or self._raw_features is not None:
            return
        self._raw_features = args[0].detach()

    def _capture_ids(self, module, args, output):
        # LLaVA embeds the text before and after the image in separate calls.
        if self._busy or self._raw_embeds is not None:
            return
        self._raw_ids.append(args[0].detach())

    def _build_seed(self, past_key_values):
        embeds = self._raw_embeds
        if embeds is None:
            raise RuntimeError(
                "SeedKV did not see the prefill inputs. Call cache.attach_model(model) "
                "before the multimodal prefill forward.")
        if embeds.shape[0] != 1 or embeds.shape[1] != self.n0:
            raise RuntimeError(
                f"SeedKV: prefill embeddings have shape {tuple(embeds.shape)}, "
                f"expected [1, {self.n0}, hidden]")
        features = self._raw_features
        if (features is not None and features.dim() == 3 and features.shape[0] > 1
                and features.shape[0] * features.shape[1] == self.image_len):
            # Several images (one contiguous visual block): one row of features.
            features = features.reshape(1, self.image_len, features.shape[-1])
        image_end = None if self.image_start is None else self.image_start + self.image_len
        use_features = (
            self.seed_mode != "embeds"
            and features is not None
            and self.image_start is not None
            and features.dim() == 3
            and features.shape[0] == 1
            and features.shape[1] == self.image_len
            and image_end <= self.n0
        )
        if use_features:
            # The features must rebuild this prefill's image embeddings.
            reference = embeds[:, self.image_start: image_end]
            rebuilt = self._projector(features)
            use_features = rebuilt.shape == reference.shape and torch.allclose(
                rebuilt.float(), reference.float(), rtol=1e-2, atol=1e-3)
        if self.seed_mode == "features" and not use_features:
            raise RuntimeError("SeedKV: seed='features' needs one image and an mm_projector")
        if use_features:
            text = self._text_ids_seed(embeds, image_end)
            if text is None:
                # .clone() so that the full prefill embeddings can be freed.
                text = (embeds[:, : self.image_start].clone(), embeds[:, image_end:].clone())
                self.seed_kind = "features+embeds"
            else:
                self.seed_kind = "features+ids"
            self._seed = (text[0], features.clone(), text[1])
        else:
            self._seed = (embeds.clone(),)
            self.seed_kind = "embeds"
        self._raw_embeds = None
        self._raw_features = None
        self._raw_ids = []
        self._remove_hooks()

        self.seed_bytes = sum(t.numel() * t.element_size() for t in self._seed)
        key = past_key_values[0][0]
        entry_bytes = 2 * key.shape[1] * key.shape[3] * key.element_size()   # one K and one V
        self.charge = int(math.ceil(self.seed_bytes / float(entry_bytes))) if self.charge_seed else 0

    def _text_ids_seed(self, embeds, image_end):
        """Token ids of the text before/after the image, if they give back its embeddings."""
        ids = self._raw_ids
        if any(x.dim() != 1 or x.is_floating_point() for x in ids):
            return None
        ids = [x for x in ids if x.numel() > 0]          # nothing between two adjacent images
        suffix_len = self.n0 - image_end
        if len(ids) == 1 and ids[0].numel() == self.image_start + suffix_len:
            ids = [ids[0][: self.image_start], ids[0][self.image_start:]]     # embedded in one call
        elif len(ids) == 1 and suffix_len == 0:
            ids = [ids[0], ids[0][:0]]
        if len(ids) != 2:
            return None
        if ids[0].numel() != self.image_start or ids[1].numel() != self.n0 - image_end:
            return None
        prefix, suffix = self._embed_text(ids[0], ids[1])
        if not torch.equal(prefix.to(embeds.dtype), embeds[:, : self.image_start]):
            return None
        if not torch.equal(suffix.to(embeds.dtype), embeds[:, image_end:]):
            return None
        return ids[0].clone(), ids[1].clone()

    def _embed_text(self, prefix, suffix):
        embed = self._inner.get_input_embeddings()
        was_busy, self._busy = self._busy, True      # keep the capture hooks quiet
        try:
            return embed(prefix)[None], embed(suffix)[None]
        finally:
            self._busy = was_busy

    def _assemble(self, prefix, features, suffix):
        image = self._projector(features)
        if not prefix.is_floating_point():
            prefix, suffix = self._embed_text(prefix, suffix)
        return torch.cat([prefix.to(image.dtype), image, suffix.to(image.dtype)], dim=1)

    def _seed_embeds(self):
        if len(self._seed) == 1:
            return self._seed[0]
        return self._assemble(*self._seed)

    @torch.no_grad()
    def _regenerate(self):
        """One forward over the prefill embeddings -> its KV for every layer."""
        if self._inner is None or self._seed is None:
            raise RuntimeError("SeedKV cannot regenerate: attach_model() or the prefill is missing")
        start = time.perf_counter()
        self._busy = True
        try:
            out = self._inner(
                inputs_embeds=self._seed_embeds(),
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            self._busy = False
        self.rehydrations += 1
        self.rehydrate_seconds += time.perf_counter() - start
        return out.past_key_values

    # ------------------------------------------------------------------
    # Attention window (PyramidKV's observation window)
    # ------------------------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        """Keep the attention rows of the most recent ``window_size`` queries."""
        if self.ratio <= 0:
            return
        if attention.dim() != 4 or attention.shape[0] != 1:
            raise RuntimeError(f"Expected [1, heads, queries, keys], got {tuple(attention.shape)}")
        query_len = attention.shape[-2]
        if self._window is None:
            self._window = [[] for _ in range(self.layer_num)]
        if query_len > 1:
            # Copy only the window rows so the full prefill map can be freed.
            self._window[layer_idx] = [attention[0, :, -self.window_size:, :].clone()]
            if self.state == "init":
                self._pending_prefill = True
        else:
            blocks = self._window[layer_idx]
            blocks.append(attention[0])
            if len(blocks) > self.window_size:
                del blocks[: len(blocks) - self.window_size]

    def _reset_window(self):
        self._window = [[] for _ in range(self.layer_num)]

    def _window_rows(self, layer_idx, length):
        """Last <= W rows of this layer as float32 [heads, rows, length], or None."""
        blocks = self._window[layer_idx] if self._window is not None else None
        if not blocks:
            return None
        padded = [F.pad(block.float(), (0, length - block.shape[-1])) for block in blocks]
        return torch.cat(padded, dim=1)[:, -self.window_size:, :]

    def _scores(self, rows, kv_heads):
        """PyramidKV's score: window attention summed over queries, pooled over keys."""
        scores = rows.sum(dim=1)                                  # [heads, keys]
        if scores.shape[0] != kv_heads:                           # GQA: average the group
            scores = scores.view(kv_heads, -1, scores.shape[-1]).mean(dim=1)
        if scores.shape[-1] == 0 or self.kernel_size == 1:
            return scores
        pool = F.max_pool1d if self.pooling == "maxpool" else F.avg_pool1d
        return pool(scores.unsqueeze(0), kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2, stride=1).squeeze(0)

    # ------------------------------------------------------------------
    # Budgets
    # ------------------------------------------------------------------
    def _capacity(self, logical_len):
        """Entries over all layers for this logical length: L * C minus the seed."""
        nominal = self.layer_num * max(1, int(logical_len * (1.0 - self.ratio)))
        return max(2 * self.layer_num, nominal - self.charge)

    def _commit_capacity(self):
        """What a boundary-only method holds now: its last budget plus the appended tokens."""
        appended = self.n_logical - self.boundary_logical
        return self._capacity(self.boundary_logical) + self.layer_num * appended

    def _budgets(self, capacity, lengths):
        window = max(1, min(self.window_size, capacity // self.layer_num - 1))
        return layer_budgets(self.layer_num, capacity, window, lengths, self.allocation, self.beta)

    def _full_pos(self, layer_idx, length, heads):
        """Logical position of every physical entry, including appended tokens."""
        pos = self._pos[layer_idx]
        extra = length - pos.shape[1]
        if extra < 0:
            raise RuntimeError(f"SeedKV layer {layer_idx}: bookkeeping longer than the KV")
        if extra == 0:
            return pos
        new = torch.arange(self.n_logical - extra, self.n_logical, device=pos.device)
        return torch.cat([pos, new.view(1, -1).expand(heads, -1)], dim=1)

    @staticmethod
    def _gather(tensor, index):
        gather = index.to(tensor.device)[None, :, :, None]
        return tensor.gather(2, gather.expand(tensor.shape[0], -1, -1, tensor.shape[-1]))

    # ------------------------------------------------------------------
    # Commit: image and text together, with the current question in view
    # ------------------------------------------------------------------
    def _commit(self, past_key_values, capacity, stage):
        lengths = [k.shape[2] for k, _ in past_key_values]
        budgets = self._budgets(capacity, lengths)
        new_past, kept, image_kept = [], [], []
        for layer_idx, (key, value) in enumerate(past_key_values):
            length, heads = lengths[layer_idx], key.shape[1]
            pos = self._full_pos(layer_idx, length, heads)
            a_count = self._a_count[layer_idx]
            budget = budgets[layer_idx]
            rows = self._window_rows(layer_idx, length)
            if rows is None or budget >= length:
                self._pos[layer_idx] = pos
                new_past.append((key, value))
                kept.append(length)
                image_kept.append(a_count)
                continue

            window = max(1, min(rows.shape[1], budget - 1))
            past_len = length - window
            keep_past = max(0, budget - window)
            scores = self._scores(rows[:, -window:, :past_len], heads)
            a_past = min(a_count, past_len)          # prefill entries sit at the front
            b_past = past_len - a_past
            if a_past > 0:
                scores[:, 0] = float("inf")           # the attention sink (logical position 0)

            # Split the layer budget between prefill and dialogue entries the way
            # an unconstrained per-head top-k would on average, then let every
            # head choose its own entries inside each part.
            if b_past == 0 or a_past == 0:
                n_a = min(a_past, keep_past)
            else:
                joint = scores.topk(keep_past, dim=-1).indices
                n_a = int(torch.round((joint < a_past).sum(dim=-1).float().mean()).item())
                n_a = max(min(1, keep_past), min(n_a, a_past, keep_past))
            n_b = min(keep_past - n_a, b_past)
            n_a = min(a_past, keep_past - n_b)

            parts = []
            if n_a > 0:
                parts.append(scores[:, :a_past].topk(n_a, dim=-1).indices.sort(dim=-1).values)
            if n_b > 0:
                parts.append(scores[:, a_past:].topk(n_b, dim=-1).indices.sort(dim=-1).values + a_past)
            parts.append(torch.arange(past_len, length, device=scores.device).expand(heads, -1))
            index = torch.cat(parts, dim=-1)

            new_past.append((self._gather(key, index), self._gather(value, index)))
            self._pos[layer_idx] = pos.gather(1, index.to(pos.device))
            self._a_count[layer_idx] = n_a + max(0, a_count - past_len)
            kept.append(index.shape[1])
            image_kept.append(self._a_count[layer_idx])

        self._reset_window()
        self.state = "compressed"
        self._refreshing = False
        self.tokens_since_commit = 0
        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": stage,
            "turn": self.turn_index + 1,
            "logical_len": int(self.n_logical),
            "target_total": int(capacity),
            "seed": self.seed_kind,
            "seed_charge_total": int(self.charge),
            "kv_before_mean": float(sum(lengths) / len(lengths)),
            "kv_after_mean": float(sum(kept) / len(kept)),
            "prefill_entries_mean": float(sum(image_kept) / len(image_kept)),
            "rehydrations": int(self.rehydrations),
            "layer_budgets": kept,
        }
        return tuple(new_past)

    # ------------------------------------------------------------------
    # Rest: drop the regenerable prefill, keep dialogue text
    # ------------------------------------------------------------------
    def _rest(self, past_key_values):
        capacity = self._capacity(self.n_logical)
        a_counts = list(self._a_count)
        lengths = [k.shape[2] - a for (k, _), a in zip(past_key_values, a_counts)]
        budgets = self._budgets(capacity, lengths)
        new_past, kept = [], []
        for layer_idx, (key, value) in enumerate(past_key_values):
            full, heads = key.shape[2], key.shape[1]
            a_count, length, budget = a_counts[layer_idx], lengths[layer_idx], budgets[layer_idx]
            pos = self._full_pos(layer_idx, full, heads)[:, a_count:]
            key, value = key[:, :, a_count:], value[:, :, a_count:]
            rows = self._window_rows(layer_idx, full)
            self._a_count[layer_idx] = 0
            if budget >= length:
                # Nothing to evict.  .contiguous() releases the dropped prefill.
                self._pos[layer_idx] = pos
                new_past.append((key.contiguous(), value.contiguous()))
                kept.append(length)
                continue
            if rows is None:
                # No attention since the last layout change: keep the newest text.
                index = torch.arange(length - budget, length, device=pos.device).expand(heads, -1)
            else:
                rows = rows[:, :, a_count:]
                window = max(1, min(rows.shape[1], budget - 1))
                past_len = length - window
                scores = self._scores(rows[:, -window:, :past_len], heads)
                top = scores.topk(max(0, budget - window), dim=-1).indices.sort(dim=-1).values
                tail = torch.arange(past_len, length, device=top.device).expand(heads, -1)
                index = torch.cat([top, tail], dim=-1)
            new_past.append((self._gather(key, index), self._gather(value, index)))
            self._pos[layer_idx] = pos.gather(1, index.to(pos.device))
            kept.append(index.shape[1])

        self._reset_window()
        self.state = "rest"
        self.boundary_logical = int(self.n_logical)
        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": "turn_boundary",
            "turn": self.turn_index + 1,
            "logical_len": int(self.n_logical),
            "target_total": int(capacity),
            "seed": self.seed_kind,
            "seed_charge_total": int(self.charge),
            "kv_before_mean": float(sum(l + a for l, a in zip(lengths, a_counts)) / len(lengths)),
            "kv_after_mean": float(sum(kept) / len(kept)),
            "prefill_entries_mean": 0.0,
            "rehydrations": int(self.rehydrations),
            "layer_budgets": kept,
        }
        return tuple(new_past)

    # ------------------------------------------------------------------
    # Hydrate: put the full prefill back in front of the dialogue entries
    # ------------------------------------------------------------------
    def _hydrate(self, past_key_values):
        region = self._regenerate()
        new_past = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            a_count, heads = self._a_count[layer_idx], key.shape[1]
            key_a, value_a = region[layer_idx]
            new_past.append((
                torch.cat([key_a.to(key.device, key.dtype), key[:, :, a_count:]], dim=2),
                torch.cat([value_a.to(value.device, value.dtype), value[:, :, a_count:]], dim=2),
            ))
            pos = self._pos[layer_idx]
            front = torch.arange(self.n0, device=pos.device).view(1, -1).expand(heads, -1)
            self._pos[layer_idx] = torch.cat([front, pos[:, a_count:]], dim=1)
            self._a_count[layer_idx] = self.n0
        self._reset_window()
        self.state = "hydrated"
        self.hydrated_tokens = 0
        return tuple(new_past)

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------
    def begin_answer(self):
        """The question of this turn has been read; commit at the next call."""
        self._answer_begun = True

    def prepare_turn(self, past_key_values, num_of_token=None):
        """Call before feeding a later turn's tokens: regenerate the prefill KV."""
        if past_key_values is None or self.ratio <= 0 or self.state != "rest" or not self.regenerate:
            return past_key_values
        return self._hydrate(past_key_values)

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("SeedKVCache requires num_of_token for ratio-based pruning.")
        if self.ratio <= 0:
            return past_key_values
        if len(past_key_values) != self.layer_num:
            raise ValueError(f"Expected {self.layer_num} layers, got {len(past_key_values)}")

        # Fallback when attentions are returned normally instead of the online hook.
        has_attention = attentions is not None and attentions[0] is not None
        lazy = self.state == "rest"        # prepare_turn() was not called for this turn
        if has_attention and not lazy:
            if len(attentions) != self.layer_num:
                raise ValueError("One attention tensor per transformer layer is required")
            for layer_idx, attention in enumerate(attentions):
                self.observe_layer(layer_idx, attention)
        self.n_logical = int(num_of_token)

        if self._pending_prefill:
            self._pending_prefill = False
            first = past_key_values[0][0]
            if first.shape[0] != 1:
                raise ValueError("SeedKVCache supports batch size 1 only")
            self.n0 = int(first.shape[2])
            if self.regenerate:
                self._build_seed(past_key_values)
            self._pos = []
            for key, _ in past_key_values:
                self._pos.append(
                    torch.arange(self.n0, device=key.device).view(1, -1).expand(key.shape[1], -1))
            self._a_count = [self.n0] * self.layer_num
            self.boundary_logical = self.n0
            return self._commit(past_key_values, self._commit_capacity(), stage="prefill")

        if self.state == "init":
            raise ValueError(
                "SeedKVCache must observe the multimodal prefill first "
                "(output_attentions=True or the online attention hook).")

        if lazy:
            _warn_once(
                "lazy",
                "SeedKV: prepare_turn() was not called, so the first token of each later "
                "turn is computed before the image is regenerated.")
            return self._hydrate(past_key_values)

        if not has_attention:
            raise ValueError("SeedKVCache requires decoding attention (output_attentions=True).")

        if self.state == "hydrated":
            self.hydrated_tokens += 1
            if self._refreshing:
                ready = self.hydrated_tokens >= self.window_size
            else:
                ready = self._answer_begun or 0 < self.max_hydrated_tokens <= self.hydrated_tokens
            if ready:
                return self._commit(past_key_values, self._commit_capacity(), stage="commit")
            return past_key_values

        # Compressed: append only.  Optionally refresh the image during a long answer.
        self.tokens_since_commit += 1
        if (self.refresh_every > 0 and self.regenerate and self._answer_begun
                and self.tokens_since_commit >= self.refresh_every):
            self._refreshing = True
            return self._hydrate(past_key_values)
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Turn boundary: drop the prefill entries and compress the dialogue text."""
        if past_key_values is None or self.ratio <= 0 or self.state == "init":
            return past_key_values
        if self.state == "hydrated" and not self._refreshing:
            _warn_once(
                "no_commit",
                "SeedKV: a turn ended before the commit. Call cache.begin_answer() once the "
                "question has been fed, or set max_hydrated_tokens.")
        if self.regenerate:
            result = self._rest(past_key_values)
        else:
            # Ablation: compress prefill and dialogue together and keep both.
            self.boundary_logical = int(self.n_logical)
            result = self._commit(past_key_values, self._commit_capacity(), stage="turn_boundary")
        self._answer_begun = False
        self._refreshing = False
        self.turn_index += 1
        return result
