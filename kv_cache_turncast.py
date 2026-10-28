"""
TurnCast-KV: a learned forecast of what the whole conversation will read,
used where multi-turn KV eviction is decided.

Problem
-------
In LLaVA's multi-turn protocol the cache is compressed right after the Turn-1
prefill ("USER: Q1\n<image> ASSISTANT:") and at every turn boundary, always
BEFORE the next question exists.  The Turn-1 compression decides which of the
576 visual tokens survive for the WHOLE conversation.  PyramidKV decides it with
the attention of the last 8 prompt tokens; because Q1 comes before the image,
those 8 rows are 3 image patches and " ASSISTANT:", and none of them knows what
the later turns will ask.

Idea
----
Train a tiny model to FORECAST the attention that the rest of the conversation
(the Turn-1 answer and every later turn) will pay to each cached entry, from
signals that are free at the compression point:

  * the attention each entry already received in this turn (question rows and
    answer rows separately) and from the last 8 rows (PyramidKV's window);
  * the strongest attention it received in any earlier turn;
  * the same signals averaged over the layer's heads and over all layers;
  * the same signals max-pooled over neighbours (3x3 on the image grid, +-3 text tokens);
  * key and value norms;
  * structure: segment (sink / prompt text / image / user / assistant), turn
    age, recency, image-grid position.

The forecaster is a 2-layer MLP (64 hidden units; ~15k parameters for LLaVA-7B,
8k of them a per-(layer, head) embedding).  It is trained by distillation from the
frozen LLM: one teacher-forced pass over 200 conversations per dataset gives
the inputs above and the target, i.e. the attention the REAL later turns paid
(turncast/collect.py, turncast/train.py).  Nothing extra is generated or run at
inference.

Where it is used (``boundary``)
-------------------------------
  * Turn-1 compression: always the forecaster.  Every KV head keeps its top-B_l
    entries; B_l follows PyramidKV's linear pyramid with an exact total of
    L * C, C = int(logical_len * (1 - ratio)); the attention sink is kept.
  * Turn boundaries: ``boundary="window"`` (default) applies PyramidKV's rule
    (maxpool-7 of the last-8-row attention, newest 8 kept) with the same
    budgets; ``boundary="forecast"`` uses the forecaster there too.
    In the CPU test the forecast won by a wide margin at the Turn-1 compression
    but lost at turn boundaries: even the TRUE future attention (an oracle)
    under-protected the just-finished answer, which later turns copy from.
    Run both on LLaVA (run-job-turncast) before fixing the default.

Cost
----
Same model passes and attention reads as PyramidKV.  Decoding attention rows are
appended to a list and folded every 32 tokens.  By default the forecaster runs
once per conversation (Turn-1 compression: L * H * S ~ 0.67M entries for
LLaVA-7B, ~9 GFLOPs, less than one decoding step).

Interface (used by eval_rouge_final2.py / eval_ppl_final.py / eval_efficiency.py)
------------------------------------------------------------------------------
    cache = TurnCastKVCache(layer_num, ratio, scorer_path=..., image_start=..., image_token_len=...)
    cache.register_question(q_text)   # once per turn, before its tokens
    cache.begin_answer()              # once per turn, before the answer tokens
    past = cache(past, num_of_token, attentions)   # after every forward
    past = cache.finalize_turn(past)               # at every turn boundary
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



EPS = 1e-4
LOG_EPS = math.log(EPS)
WINDOW = 8          # PyramidKV's observation window; used here as one input signal
FOLD_EVERY = 32     # decoding rows are folded into the running sums in chunks
SEGMENTS = ("sink", "prompt_text", "image", "user", "assistant")
SEG_ID = {name: index for index, name in enumerate(SEGMENTS)}
FEATURE_NAMES = (
    "q", "a", "w", "h",
    "q_has", "a_has", "h_has",
    "pool_q", "pool_a", "pool_w",
    "layer_q", "layer_a", "layer_w", "layer_h",
    "global_q", "global_a", "global_w", "global_h",
    "key_norm", "value_norm",
    "seg_sink", "seg_prompt_text", "seg_image", "seg_user", "seg_assistant",
    "age", "current_turn", "recency",
    "img_row", "img_col", "img_radius",
)
FEATURE_VERSION = 1


# ----------------------------------------------------------------------
# Running attention statistics of one layer (shared by the online cache and
# by the offline collector, so training and inference see identical inputs)
# ----------------------------------------------------------------------
class LayerStats:
    """Attention received by every cached entry of one layer, per KV head.

    q_sum / q_cnt: attention from the current turn's question rows (for Turn 1:
                   the whole multimodal prompt), counted only over rows that
                   come AFTER the entry (an entry's attention to itself is not
                   evidence of importance).
    a_sum / a_cnt: the same for the current turn's answer rows.
    hist:          max over finished turns of log(S * turn-mean attention).
    window:        the last <= WINDOW rows since the previous compression.
    pos:           logical (RoPE) position of each physical entry.
    """

    def __init__(self, heads, device, capacity=1024):
        self.heads = int(heads)
        self.device = device
        self.n = 0
        self.phase = "q"
        self.track_turns = True     # False: only the window and positions are kept (cheaper)
        self.window = []
        self._pending = []
        self._pending_pos = []
        self._alloc(max(16, int(capacity)))

    # -- storage -------------------------------------------------------
    def _new(self, capacity, fill, dtype=torch.float32):
        return torch.full((self.heads, capacity), fill, dtype=dtype, device=self.device)

    def _alloc(self, capacity):
        self.cap = capacity
        self.q_sum = self._new(capacity, 0.0)
        self.q_cnt = self._new(capacity, 0.0)
        self.a_sum = self._new(capacity, 0.0)
        self.a_cnt = self._new(capacity, 0.0)
        self.hist = self._new(capacity, float("-inf"))
        self.pos = self._new(capacity, 0, torch.long)

    _FIELDS = (("q_sum", 0.0), ("q_cnt", 0.0), ("a_sum", 0.0), ("a_cnt", 0.0),
               ("hist", float("-inf")), ("pos", 0))

    def _ensure(self, n):
        if n <= self.cap:
            return
        capacity = max(n, 2 * self.cap)
        for name, fill in self._FIELDS:
            old = getattr(self, name)
            new = self._new(capacity, fill, old.dtype)
            new[:, : self.n] = old[:, : self.n]
            setattr(self, name, new)
        self.cap = capacity

    # -- adding attention rows --------------------------------------------
    def add_block(self, rows, positions):
        """rows: [H, R, K] attention of the R newest entries (as queries) to all K
        entries (causal, so row r sees entries 0 .. K-R+r).  positions: [R]."""
        heads, count, keys = rows.shape
        start = keys - count
        if start != self.n:
            raise RuntimeError(f"LayerStats: expected {self.n} existing entries, got {start}")
        self._ensure(keys)
        self.pos[:, start:keys] = positions.to(self.device).view(1, -1)
        rows = rows.float()
        received = rows.sum(dim=1)
        received[:, start:] -= rows[:, :, start:].diagonal(dim1=1, dim2=2)
        seen = (keys - 1 - torch.arange(keys, device=self.device)).clamp(min=0, max=count)
        if self.phase == "q":
            self.q_sum[:, :keys] += received
            self.q_cnt[:, :keys] += seen.float()
        else:
            self.a_sum[:, :keys] += received
            self.a_cnt[:, :keys] += seen.float()
        for r in range(max(0, count - WINDOW), count):
            self.window.append(rows[:, r, : start + r + 1].clone())
        del self.window[:-WINDOW]
        self.n = keys

    def add_row(self, row, position):
        """One decoding row [H, K] (K = entries after this token).  Folded lazily."""
        if row.shape[-1] != self.n + len(self._pending_pos) + 1:
            raise RuntimeError(
                f"LayerStats: decoding row covers {row.shape[-1]} keys, expected "
                f"{self.n + len(self._pending_pos) + 1}")
        if self.track_turns:
            self._pending.append(row)
        self._pending_pos.append(int(position))
        self.window.append(row)
        if len(self.window) > WINDOW:
            self.window.pop(0)
        if len(self._pending_pos) >= FOLD_EVERY:
            self.fold()

    def fold(self):
        if not self._pending_pos:
            return
        if not self.track_turns:
            start, count = self.n, len(self._pending_pos)
            self._ensure(start + count)
            self.pos[:, start:start + count] = torch.tensor(self._pending_pos, device=self.device).view(1, -1)
            self.n += count
            self._pending_pos = []
            return
        keys = self._pending[-1].shape[-1]
        block = torch.stack([F.pad(r, (0, keys - r.shape[-1])) for r in self._pending], dim=1)
        positions = torch.tensor(self._pending_pos, dtype=torch.long)
        self._pending, self._pending_pos = [], []
        window, self.window = self.window, []
        self.add_block(block, positions)
        self.window = window          # these rows are already in the window, in order

    def set_phase(self, phase):
        self.fold()
        self.phase = phase

    # -- boundary operations ----------------------------------------------
    def snapshot(self, size):
        """Per-head log features of the current entries (size = entries in this layer)."""
        self.fold()
        n = self.n
        scale = float(size)
        q_has = self.q_cnt[:, :n] > 0
        a_has = self.a_cnt[:, :n] > 0
        q = torch.log(scale * self.q_sum[:, :n] / self.q_cnt[:, :n].clamp(min=1.0) + EPS)
        a = torch.log(scale * self.a_sum[:, :n] / self.a_cnt[:, :n].clamp(min=1.0) + EPS)
        w_sum = torch.zeros(self.heads, n, device=self.device)
        w_cnt = torch.zeros(n, device=self.device)
        for row in self.window:
            visible = row.shape[-1] - 1          # entries strictly before this row's token
            w_sum[:, :visible] += row[:, :visible].float()
            w_cnt[:visible] += 1
        w = torch.log(scale * w_sum / w_cnt.clamp(min=1.0) + EPS)
        h_has = torch.isfinite(self.hist[:, :n])
        h = torch.where(h_has, self.hist[:, :n], torch.full_like(self.hist[:, :n], LOG_EPS))
        return {"q": q, "a": a, "w": w, "h": h, "q_has": q_has, "a_has": a_has,
                "h_has": h_has, "pos": self.pos[:, :n].clone()}

    def end_turn(self, size):
        """Fold the finished turn into hist and clear the per-turn sums."""
        self.fold()
        n = self.n
        cnt = self.q_cnt[:, :n] + self.a_cnt[:, :n]
        mean = (self.q_sum[:, :n] + self.a_sum[:, :n]) / cnt.clamp(min=1.0)
        value = torch.log(float(size) * mean + EPS)
        self.hist[:, :n] = torch.where(cnt > 0, torch.maximum(self.hist[:, :n], value), self.hist[:, :n])
        for name in ("q_sum", "q_cnt", "a_sum", "a_cnt"):
            getattr(self, name)[:, :n] = 0.0
        self.phase = "q"

    def reset_window(self):
        self.fold()
        self.window = []

    def gather(self, index):
        """Keep the entries index [H, keep] (per head)."""
        self.fold()
        keep = index.shape[1]
        index = index.to(self.device)
        capacity = keep + 256
        for name, fill in self._FIELDS:
            old = getattr(self, name)
            new = self._new(capacity, fill, old.dtype)
            new[:, :keep] = old[:, : self.n].gather(1, index)
            setattr(self, name, new)
        self.cap = capacity
        self.n = keep
        self.window = []


# ----------------------------------------------------------------------
# Features (shared by online inference and offline training)
# ----------------------------------------------------------------------
def layer_consensus(snap, n_logical):
    """Mean over this layer's heads of q/a/w/h at each logical position.

    Returns ([4, N] values, [N] presence)."""
    pos = snap["pos"]
    values = torch.stack([snap["q"], snap["a"], snap["w"], snap["h"]])       # [4, H, S]
    total = torch.zeros(4, n_logical, device=pos.device)
    total.scatter_add_(1, pos.reshape(1, -1).expand(4, -1), values.reshape(4, -1))
    count = torch.zeros(n_logical, device=pos.device)
    count.scatter_add_(0, pos.reshape(-1), torch.ones(pos.numel(), device=pos.device))
    present = count > 0
    mean = torch.where(present, total / count.clamp(min=1.0), torch.full_like(total, LOG_EPS))
    return mean, present


def global_consensus(layer_values, layer_present):
    """Mean over layers of the layer consensus, where present."""
    values = torch.stack(layer_values)                                      # [L, 4, N]
    present = torch.stack(layer_present).float()                            # [L, N]
    count = present.sum(0)
    mean = (values * present[:, None, :]).sum(0) / count.clamp(min=1.0)
    return torch.where(count > 0, mean, torch.full_like(mean, LOG_EPS))


def _pool(x, pos, n_logical, image_start, image_len, grid):
    """Max of x over neighbours present in the same head: +-3 text tokens, 3x3 image."""
    heads = x.shape[0]
    buf = torch.full((heads, n_logical), LOG_EPS, device=x.device)
    buf.scatter_(1, pos, x)
    pooled = F.max_pool1d(buf.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
    if grid > 0 and image_start + image_len <= n_logical:
        image = buf[:, image_start: image_start + image_len].reshape(heads, 1, grid, grid)
        image = F.max_pool2d(image, kernel_size=3, stride=1, padding=1)
        pooled[:, image_start: image_start + image_len] = image.reshape(heads, image_len)
    return pooled.gather(1, pos)


def _zscore(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


def layer_features(snap, log_key_norm, log_value_norm, consensus, global_values, meta):
    """[H, S, len(FEATURE_NAMES)] features of one layer.

    snap: LayerStats.snapshot() (possibly restricted to some heads);
    consensus: [4, N] from layer_consensus over ALL heads of the layer;
    global_values: [4, N] from global_consensus; meta: see conversation_meta()."""
    pos = snap["pos"]
    n_logical = meta["n_logical"]
    image_start, image_len, grid = meta["image_start"], meta["image_len"], meta["grid"]
    own = torch.stack([snap["q"], snap["a"], snap["w"], snap["h"]], dim=-1)
    flags = torch.stack([snap["q_has"], snap["a_has"], snap["h_has"]], dim=-1).float()
    pooled = torch.stack(
        [_pool(snap[name], pos, n_logical, image_start, image_len, grid) for name in ("q", "a", "w")],
        dim=-1,
    )
    layer = consensus[:, pos].permute(1, 2, 0)
    glob = global_values[:, pos].permute(1, 2, 0)
    norms = torch.stack([_zscore(log_key_norm), _zscore(log_value_norm)], dim=-1)
    seg = F.one_hot(meta["seg"][pos], len(SEGMENTS)).float()
    turn = meta["turn"][pos]
    age = (meta["turn_now"] - turn).clamp(min=0, max=4).float() / 4.0
    current = (turn == meta["turn_now"]).float()
    recency = torch.log1p((n_logical - 1 - pos).clamp(min=0).float()) / 8.0
    offset = pos - image_start
    in_image = (offset >= 0) & (offset < image_len)
    if grid == 0:
        in_image = torch.zeros_like(in_image)
    half = max(1.0, (grid - 1) / 2.0)
    row = torch.where(in_image, (offset // max(grid, 1)).float() / half - 1.0, torch.zeros_like(recency))
    col = torch.where(in_image, (offset % max(grid, 1)).float() / half - 1.0, torch.zeros_like(recency))
    radius = torch.sqrt(row * row + col * col) / math.sqrt(2.0)
    structure = torch.stack([age, current, recency, row, col, radius], dim=-1)
    return torch.cat([own, flags, pooled, layer, glob, norms, seg, structure], dim=-1)


def conversation_meta(seg, turn, turn_now, image_start, image_len, device):
    grid = int(round(math.sqrt(image_len))) if image_len else 0
    if grid * grid != image_len:
        grid = 0
    return {
        "seg": torch.as_tensor(seg, dtype=torch.long, device=device),
        "turn": torch.as_tensor(turn, dtype=torch.long, device=device),
        "turn_now": int(turn_now),
        "n_logical": len(seg),
        "image_start": int(image_start),
        "image_len": int(image_len),
        "grid": grid,
    }


def prompt_segments(prompt_len, image_start, image_len):
    """Segment ids of the Turn-1 prompt "<system> USER: Q1\\n<image> ASSISTANT:"."""
    seg = []
    for p in range(prompt_len):
        if p == 0:
            seg.append(SEG_ID["sink"])
        elif p < image_start:
            seg.append(SEG_ID["prompt_text"])
        elif p < image_start + image_len:
            seg.append(SEG_ID["image"])
        else:
            seg.append(SEG_ID["user"])
    return seg


def window_scores(snaps):
    """PyramidKV's rule from the same statistics: maxpool-7 of the last-8-row
    attention, the newest WINDOW entries always kept."""
    out = []
    for snap in snaps:
        w = snap["w"]
        past = max(0, w.shape[1] - WINDOW)
        pooled = torch.full_like(w, float("inf"))
        if past:
            pooled[:, :past] = F.max_pool1d(w[:, :past].unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
        out.append(pooled)
    return out


def pyramid_capacities(n_layers, capacity, window, q_len, exact=True, beta=20):
    """PyramidKV's per-layer PAST capacities (KVCache-Factory schedule, window excluded).

    Same as PyramidKVCache.pyramid_capacities; kept here so TurnCast does not
    depend on the version of kv_cache_pyramidkv.py.  exact=True: float slope +
    largest-remainder rounding so the total is exactly L * (capacity - window);
    exact=False: the original floored slope.  None = keep everything."""
    past_average = capacity - window
    if q_len < capacity:
        return None
    if q_len < past_average * 2:
        return [past_average] * n_layers
    min_num = int(past_average // beta)
    max_num = past_average * 2 - min_num
    if max_num >= q_len - window:
        max_num = q_len - window
        min_num = past_average * 2 - max_num
    if exact:
        slope = (max_num - min_num) / (n_layers - 1)
        values = [max_num - layer * slope for layer in range(n_layers)]
        caps = [int(x) for x in values]
        missing = past_average * n_layers - sum(caps)
        order = sorted(range(n_layers), key=lambda i: values[i] - caps[i], reverse=True)
        for layer in order[:max(0, missing)]:
            caps[layer] += 1
        return caps
    steps = (max_num - min_num) // (n_layers - 1)
    return [max_num - layer * steps for layer in range(n_layers)]


def pyramid_budgets(n_layers, ratio, n_logical, lengths):
    """Entries kept per layer: PyramidKV's linear pyramid, exact total L * C.

    None when nothing needs to be evicted (logical length below the budget)."""
    capacity = max(1, int(n_logical * (1.0 - ratio)))
    window = max(1, min(WINDOW, capacity - 1)) if capacity > 1 else 1
    caps = pyramid_capacities(n_layers, capacity, window, max(lengths), exact=True)
    if caps is None:
        return None
    return [min(cap + window, length) for cap, length in zip(caps, lengths)]


# ----------------------------------------------------------------------
# The forecaster
# ----------------------------------------------------------------------
class TurnCastScorer(nn.Module):
    """Tiny MLP: features of one cached entry -> logit of its future attention share."""

    def __init__(self, n_layers, n_heads, n_features=len(FEATURE_NAMES), embed_dim=8, hidden=64):
        super().__init__()
        self.config = {
            "n_layers": int(n_layers), "n_heads": int(n_heads), "n_features": int(n_features),
            "embed_dim": int(embed_dim), "hidden": int(hidden),
            "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION,
        }
        self.embed = nn.Embedding(n_layers * n_heads, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(n_features + embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def forward(self, features, ids):
        x = (features - self.feat_mean) / self.feat_std
        return self.mlp(torch.cat([x, self.embed(ids)], dim=-1)).squeeze(-1)


def save_scorer(scorer, path, extra=None):
    torch.save({"config": scorer.config, "state_dict": scorer.state_dict(), "extra": extra or {}}, path)


# def load_scorer(path, device="cpu"):
#     blob = torch.load(path, map_location="cpu")
#     config = blob["config"]
#     if list(config["feature_names"]) != list(FEATURE_NAMES):
#         raise ValueError(f"{path} was trained with different features: {config['feature_names']}")
#     scorer = TurnCastScorer(config["n_layers"], config["n_heads"], config["n_features"],
#                             config["embed_dim"], config["hidden"])
#     scorer.load_state_dict(blob["state_dict"])
#     return scorer.to(device).eval()



_SCORER_CACHE = {}

def load_scorer(path, device="cuda"):
    key = (str(path), str(device))
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    blob = torch.load(path, map_location=device)
    config = blob["config"]
    if list(config["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError(f"{path} was trained with different features: {config['feature_names']}")
    scorer = TurnCastScorer(config["n_layers"], config["n_heads"], config["n_features"],
                            config["embed_dim"], config["hidden"])
    scorer.load_state_dict(blob["state_dict"])
    scorer = scorer.to(device).eval().requires_grad_(False)
    _SCORER_CACHE[key] = scorer
    return scorer


# ----------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------
class TurnCastKVCache:
    """Multi-turn KV eviction with a learned forecast of what later turns will read."""

    def __init__(
        self,
        layer_num,
        ratio=0.0,
        scorer_path=None,
        boundary="window",
        image_start=None,
        image_token_len=576,
        k_seq_dim=2,
        v_seq_dim=2,
        scorer=None,
        record=False,
    ):
        if k_seq_dim != 2 or v_seq_dim != 2:
            raise ValueError("TurnCastKVCache expects KV layout [batch, heads, seq, dim]")
        if not 0.0 <= float(ratio) < 1.0:
            raise ValueError("ratio must satisfy 0 <= ratio < 1")
        if boundary not in {"window", "forecast"}:
            raise ValueError("boundary must be 'window' or 'forecast'")
        if image_start is None:
            raise ValueError("TurnCastKVCache needs image_start (passed by build_cache)")
        self.layer_num = int(layer_num)
        self.ratio = float(ratio)
        self.boundary = boundary
        self.image_start = int(image_start)
        self.image_len = int(image_token_len)
        self.record = bool(record)
        self.score_override = None      # diagnostics: fn(cache, stage, feats, snaps) -> scores
        self.scorer = scorer if scorer is not None else (load_scorer(scorer_path) if scorer_path else None)
        if self.scorer is None and self.ratio > 0:
            raise ValueError("TurnCastKVCache needs a trained scorer (--turncast-scorer)")
        if self.scorer is not None and self.scorer.config["n_layers"] != self.layer_num:
            raise ValueError(
                f"scorer was trained for {self.scorer.config['n_layers']} layers, model has {self.layer_num}")
        self.reset()

    def reset(self):
        self.stats = None
        self.turn_now = -1
        self.phase = "q"
        self.seg, self.turn = [], []
        self.n_logical = 0
        self._pending_prefill = False
        self.records = []
        self.last_layer_budgets = None
        self.last_finalize_stats = None
        self.scored_entries = 0

    def attach_model(self, model):
        device = next(model.parameters()).device
        if self.scorer is not None:
            self.scorer = self.scorer.to(device)

    # -- conversation structure ------------------------------------------
    def register_question(self, question=None):
        self.turn_now += 1
        self._set_phase("q")

    def begin_answer(self):
        self._set_phase("a")

    def _set_phase(self, phase):
        self.phase = phase
        if self.stats is not None:
            for stats in self.stats:
                stats.set_phase(phase)

    # -- attention intake --------------------------------------------------
    def observe_layer(self, layer_idx, attention):
        """Prefill attention [1, H, N, N], delivered layer by layer by the hook."""
        if attention.dim() != 4 or attention.shape[0] != 1:
            raise RuntimeError(f"Expected [1, heads, queries, keys], got {tuple(attention.shape)}")
        if attention.shape[2] == 1:
            raise RuntimeError("TurnCastKVCache receives decoding attention through __call__")
        if self.stats is None:
            heads, device = attention.shape[1], attention.device
            self.stats = [LayerStats(heads, device, attention.shape[-1] + 512) for _ in range(self.layer_num)]
            self._set_phase(self.phase)
        keys = attention.shape[-1]
        self.stats[layer_idx].add_block(attention[0], torch.arange(keys))
        self._pending_prefill = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if num_of_token is None:
            raise ValueError("TurnCastKVCache requires num_of_token")
        if self.turn_now < 0:
            self.turn_now = 0          # caller did not register the first question
        if self._pending_prefill:
            self.seg = prompt_segments(num_of_token, self.image_start, self.image_len)
            self.turn = [0] * num_of_token
            self.n_logical = int(num_of_token)
            self._pending_prefill = False
            return self._compress(past_key_values, stage="prefill")

        if attentions is None or attentions[0] is None:
            raise ValueError("TurnCastKVCache requires decoding attention (output_attentions=True)")
        if len(attentions) != self.layer_num or self.stats is None:
            raise ValueError("TurnCastKVCache must observe the prefill first, one attention per layer")
        if int(num_of_token) - self.n_logical != 1 or attentions[0].shape[2] != 1:
            raise RuntimeError("TurnCastKVCache expects one new token per decoding call")
        position = int(num_of_token) - 1
        for layer_idx, attention in enumerate(attentions):
            self.stats[layer_idx].add_row(attention[0, :, -1, :], position)
        self.seg.append(SEG_ID["user"] if self.phase == "q" else SEG_ID["assistant"])
        self.turn.append(self.turn_now)
        self.n_logical = int(num_of_token)
        return past_key_values

    def finalize_turn(self, past_key_values):
        if past_key_values is None or self.stats is None:
            return past_key_values
        return self._compress(past_key_values, stage="turn")

    # -- scoring -------------------------------------------------------------
    def snapshots(self, past_key_values):
        snaps = []
        for layer_idx, (key, _) in enumerate(past_key_values):
            snap = self.stats[layer_idx].snapshot(key.shape[2])
            if snap["pos"].shape[1] != key.shape[2]:
                raise RuntimeError(
                    f"TurnCast layer {layer_idx}: statistics cover {snap['pos'].shape[1]} entries, "
                    f"KV has {key.shape[2]}")
            snaps.append(snap)
        return snaps

    def features(self, past_key_values, snaps):
        """Features of every cached entry: list over layers of [H, S_l, F]."""
        device = past_key_values[0][0].device
        meta = conversation_meta(self.seg, self.turn, self.turn_now, self.image_start, self.image_len, device)
        layer_vals, layer_present = [], []
        for snap in snaps:
            values, present = layer_consensus(snap, meta["n_logical"])
            layer_vals.append(values)
            layer_present.append(present)
        glob = global_consensus(layer_vals, layer_present)
        feats = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            log_k = key[0].float().norm(dim=-1).clamp(min=1e-6).log()
            log_v = value[0].float().norm(dim=-1).clamp(min=1e-6).log()
            feats.append(layer_features(snaps[layer_idx], log_k, log_v, layer_vals[layer_idx], glob, meta))
        return feats

    @torch.no_grad()
    def score(self, feats):
        heads = feats[0].shape[0]
        flat = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats])
        ids = torch.cat([
            (layer_idx * heads + torch.arange(heads, device=f.device)).view(-1, 1).expand(heads, f.shape[1]).reshape(-1)
            for layer_idx, f in enumerate(feats)
        ])
        device = self.scorer.feat_mean.device
        logits = self.scorer(flat.to(device), ids.to(device))
        out, start = [], 0
        for f in feats:
            size = f.shape[0] * f.shape[1]
            out.append(logits[start: start + size].view(f.shape[0], f.shape[1]).to(f.device))
            start += size
        self.scored_entries += start
        return out

    def _compress(self, past_key_values, stage):
        lengths = [k.shape[2] for k, _ in past_key_values]
        capacity = max(1, int(self.n_logical * (1.0 - self.ratio)))
        budgets = pyramid_budgets(self.layer_num, self.ratio, self.n_logical, lengths)
        compress = self.ratio > 0 and budgets is not None and any(b < l for b, l in zip(budgets, lengths))
        use_forecast = stage == "prefill" or self.boundary == "forecast"
        scores = None
        if compress or self.record:
            snaps = self.snapshots(past_key_values)
            feats = None
            if self.record or (compress and use_forecast and self.score_override is None):
                feats = self.features(past_key_values, snaps)
            if self.score_override is not None:
                scores = self.score_override(self, stage, feats, snaps)
            elif compress:
                scores = self.score(feats) if use_forecast else window_scores(snaps)
            if self.record:
                self.records.append({
                    "stage": stage, "turn": self.turn_now, "n_logical": self.n_logical,
                    "features": [f.cpu() for f in feats], "pos": [s["pos"].cpu() for s in snaps],
                })

        if stage == "prefill" and self.boundary == "window" and not self.record:
            # Turn boundaries use only the last-8-row window: stop accumulating turn statistics.
            for stats in self.stats:
                stats.fold()
                stats.track_turns = False
        new_past, kept = [], []
        for layer_idx, (key, value) in enumerate(past_key_values):
            stats = self.stats[layer_idx]
            if stage == "turn":
                stats.end_turn(lengths[layer_idx])
            if not compress or budgets[layer_idx] >= lengths[layer_idx]:
                stats.reset_window()
                new_past.append([key, value])
                kept.append(lengths[layer_idx])
                continue
            layer_scores = scores[layer_idx].clone()
            layer_scores[stats.pos[:, : stats.n] == 0] = float("inf")     # attention sink
            index = layer_scores.topk(budgets[layer_idx], dim=-1).indices.sort(dim=-1).values
            stats.gather(index)
            if self.record:
                self.records[-1].setdefault("kept", {})[layer_idx] = stats.pos[:, : stats.n].cpu()
            gather = index.to(key.device)[None, :, :, None]
            new_past.append([
                key.gather(2, gather.expand(key.shape[0], -1, -1, key.shape[-1])),
                value.gather(2, gather.expand(value.shape[0], -1, -1, value.shape[-1])),
            ])
            kept.append(budgets[layer_idx])

        self.last_layer_budgets = kept
        self.last_finalize_stats = {
            "stage": stage, "turn": self.turn_now + 1, "logical_len": int(self.n_logical),
            "target_per_layer": int(capacity),
            "rule": "forecast" if use_forecast else "pyramidkv_window",
            "kv_before_mean": float(sum(lengths) / len(lengths)),
            "kv_after_mean": float(sum(kept) / len(kept)), "layer_budgets": kept,
        }
        return new_past