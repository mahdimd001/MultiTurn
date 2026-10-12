#!/usr/bin/env python
"""Efficiency benchmark for the continuous multi-turn KV-cache evaluator.

This drives the same pipeline as ``eval_rogue_final.py`` and reuses its helpers
(``build_kv_cache``, ``finalize_turn_cache``, ``select_next_token``, ...):

* turn 1 is one prefill with the image, then the cache policy runs;
* later user messages and every turn separator are fed ONE TOKEN AT A TIME
  through the model and the cache policy, exactly as the ROUGE evaluator does;
* boundary methods compress in ``finalize_turn`` after each turn.

For every (context length, method, ratio) and EVERY TURN it measures TTFT, TPOT,
latency, throughput, peak memory, KV-cache memory and end-to-end time.  ConvBench
has 3 turns, so each configuration gets a Turn 1, Turn 2 and Turn 3 result.

Per-turn metrics (wall clock, batch size 1, greedy decoding by default)
--------------------------------------------------------------------------
  ttft        Turn start -> first answer token id on the host.
              Turn 1: image preprocessing + tokenization + prefill (vision
              encoder + LLM) + cache construction and compression + sampling.
              Turn >= 2: tokenization + feeding the new user message one token
              at a time (as eval_rogue_final.py does) + sampling.
  tpot        Time per output token (TTPT): (t_last - t_first) / (n - 1).
              Each interval covers one forward pass, the cache policy's
              per-step update/eviction and sampling.
  itl p50/p90/p99   Distribution of the individual inter-token intervals.
  latency     Turn start -> last answer token (= ttft + (n - 1) * tpot).
  boundary    Last answer token -> turn committed: forwarding that token, the
              separator and ``finalize_turn`` (where boundary methods compress).
  turn_e2e    Turn start -> turn committed (= latency + boundary).
  e2e_cum     Conversation start (incl. image file load) -> end of this turn.
  decode_tps  Decode throughput, (n - 1) / (t_last - t_first) = 1 / tpot.
  turn_output_tps   generated tokens / turn_e2e.
  turn_total_tps    (tokens fed this turn + generated tokens) / turn_e2e.
  peak memory torch.cuda.max_memory_allocated during this turn (reset at the
              start of each turn; turn 1 includes the prefill).  Also reported
              minus the memory held before the conversation (model weights).
              On MPS/CPU this is a sampled lower bound.
  KV memory   Physical K/V bytes, summed over layers (layers can differ):
              kv_end at the end of the turn (after boundary compression),
              kv_peak = largest retained cache during the turn, kv_transient
              = largest fresh K/V before the policy ran, kv_full_end = what an
              uncompressed cache would hold at the same logical length.
  cache state Bytes of the policy's own bookkeeping tensors (scores, masks...).
Conversation-level totals (e2e, overall throughput, overall peak) are also kept.

Context length (--context-lengths)
  native  The dataset as-is, identical to the ROUGE run.
  N       A long-context version of the dataset in which every conversation's
          turn-1 prefill (system prompt + 576 image tokens + text) is exactly N
          tokens.  The samples are GENERATED FIRST as dataset files by
          make_long_context.py (same tokenizer and template as the model), saved
          as <long-context-dir>/<data>_ctxN.json, and then sent to the model like
          any other dataset.  Existing files are reused; eval_rogue_final.py can
          read the same files for ROUGE at the same context lengths.

Budget (--ratios)
  Same convention as eval_rogue_final.py: ratio is the PRUNED fraction of the
  logical length, so 0.8 keeps about 20%.  ``kv_budget`` = 1 - ratio is also
  reported.  ``full`` is the uncompressed baseline (ratio is ignored).

Generation
  By default every turn generates exactly --gen-tokens tokens, ignoring EOS, so
  every method does the same amount of work and timings are comparable.  Add
  --stop-at-eos to stop at EOS/separator like the ROUGE run (then
  --gen-tokens is a ceiling and lengths can differ between methods).

Attention
  Every method, including "full", runs on the patched LLaMA attention from
  tgv_kv_legacy_nova.py with logical positions.  With the stock transformers
  4.31 attention, h2o/local_cache/elastic_cache crash on the first decode step
  after compression (RoPE table sized by the physical cache length), and a
  shared kernel keeps the comparison about the cache policy alone.
  Attention maps are only materialised for policies that read them
  (--output-attentions always reproduces the ROUGE run, which always asks).
  --prefill-attention per_layer (default) folds each layer's turn-1 map into
  the cache via observe_layer() and drops it right after that layer, as
  official TGV-KV does, so peak memory never holds all 32 maps; the scoring
  then runs inside the prefill forward (prefill_forward_ms) and
  prefill_policy_ms is the pruning only.  all_layers reproduces the original
  behaviour (all maps returned at once, +32*heads*L^2*2 bytes at the peak).

Reading peak memory
  Every method runs the full, uncompressed turn-1 prefill before it can
  compress, so turn 1 is never below "full"'s prefill peak. Savings show in
  turns 2+ and grow with context. "full"'s peak usually comes at the last
  decode step: with the legacy tuple cache each step builds the new KV while
  the old one is still alive (about 2x the final KV). peak_over_base_gib
  removes the ~13.2 GiB of LLaVA-7B weights.

Outputs (in --output-dir)
  <exp>_summary_per_turn.csv one row per (context, method, ratio, turn)  <- main table
  <exp>_summary.csv          one row per (context, method, ratio), whole conversation
  <exp>_turns.jsonl          raw per-turn rows (one per conversation and turn)
  <exp>_conversations.jsonl  raw per-conversation rows
  <exp>_meta.json            arguments, model, device and library versions
Run with --summarize-only to merge every *_conversations.jsonl in the output
directory (e.g. from a SLURM array) into all_summary*.csv.
"""

import argparse
import contextlib
import csv
import gc
import glob
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ALL_METHODS = [
    "full",
    "local_cache",
    "h2o",
    "elastic_cache",
    "tgv_kv",
    "turn_conditioned_layerwise_merge",
    "turn_conditioned_layerwise_eviction",
]
# These policies never read attention maps, so by default they do not pay for
# materialising them (use --output-attentions always to match the ROUGE run).
ATTENTION_FREE_METHODS = {"full", "local_cache"}

GIB = float(1024 ** 3)
MIB = float(1024 ** 2)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_context_length(value):
    text = str(value).strip().lower()
    if text in {"native", "none"}:
        return "native"
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use 'native' or a whole token count such as 2048."
        ) from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Context lengths must be positive.")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    model = parser.add_argument_group("model and data (same as eval_rogue_final.py)")
    model.add_argument("--model-path", type=str, default="/lustre/hdd/LAS/jannesar-lab/tmahmud/models/llava-v1.5-7b")
    model.add_argument("--model-base", type=str, default=None)
    model.add_argument(
        "--data-path",
        type=str,
        default="./ConvBench/convbench.json",
        help="Multi-turn JSON (items with 'image' and 'turns'), e.g. "
        "ConvBench/convbench.json. Omit to use a synthetic image and questions.",
    )
    model.add_argument("--image-path", type=str, default="./ConvBench/visit_bench_images")
    model.add_argument("--device", type=str, default="cuda")
    model.add_argument("--conv-mode", type=str, default=None)
    model.add_argument("--load-8bit", action="store_true")
    model.add_argument("--load-4bit", action="store_true")
    model.add_argument("--image-aspect-ratio", type=str, default="pad")

    sweep = parser.add_argument_group("sweep")
    sweep.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        choices=ALL_METHODS,
        help="'full' is the uncompressed baseline.",
    )
    sweep.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.5, 0.8],
        help="Pruned fractions (0.8 keeps ~20%%). Ignored for 'full'.",
    )
    sweep.add_argument(
        "--context-lengths",
        nargs="+",
        type=parse_context_length,
        default=["native", 1024, 2048],
        help="'native' and/or turn-1 prefill lengths in tokens, e.g. 1024 2048 3072.",
    )
    sweep.add_argument("--gen-tokens", type=int, default=128, help="Tokens generated per turn.")
    sweep.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Stop each turn at EOS/separator like the ROUGE run (--gen-tokens becomes a cap).",
    )
    sweep.add_argument(
        "--num-turns",
        type=int,
        default=None,
        help="Force this many turns per conversation (truncate, or extend with "
        "generic follow-up questions). Default: the dataset's turns (3 for ConvBench).",
    )
    sweep.add_argument("--num-samples", type=int, default=2, help="Measured conversations per config.")
    sweep.add_argument("--sample-offset", type=int, default=0)
    sweep.add_argument("--warmup", type=int, default=1, help="Unmeasured conversations per config.")

    longctx = parser.add_argument_group("long-context samples (make_long_context.py)")
    longctx.add_argument(
        "--long-context-dir",
        type=str,
        default=None,
        help="Where <data>_ctxN.json files are read from / written to. "
        "Default: <data-path directory>/long_context.",
    )
    longctx.add_argument(
        "--regenerate-long-context",
        action="store_true",
        help="Rebuild the ctxN files even if they exist.",
    )
    longctx.add_argument(
        "--filler-file",
        type=str,
        default=None,
        help="Passage text for new ctxN files (default: other conversations' reference answers).",
    )

    policy = parser.add_argument_group("cache policy (same as eval_rogue_final.py)")
    policy.add_argument("--start-size", type=int, default=1)
    policy.add_argument(
        "--recent-size",
        type=int,
        default=None,
        help="Default: sized automatically to cover the whole conversation. "
        "elastic_cache allocates its score buffer from start_size + recent_size, "
        "so the ROUGE default of 2047 fails above ~2k tokens.",
    )
    policy.add_argument("--alpha", type=float, default=1.0)
    policy.add_argument("--beta", type=float, default=0.8)
    policy.add_argument("--cross-turn-weight", type=float, default=1.0)
    policy.add_argument("--temperature", type=float, default=0.0)
    policy.add_argument("--top_p", "--top-p", type=float, default=None)

    measure = parser.add_argument_group("measurement")
    measure.add_argument(
        "--output-attentions",
        choices=["auto", "always"],
        default="auto",
        help="auto: request attention maps only for policies that read them. "
        "always: request them for every method, exactly like eval_rogue_final.py.",
    )
    measure.add_argument(
        "--prefill-attention",
        choices=["per_layer", "all_layers"],
        default="per_layer",
        help="per_layer: each layer's turn-1 attention map is folded into the cache "
        "and dropped right after that layer (needs observe_layer() in the cache "
        "classes). all_layers: transformers returns all 32 maps at once, as the "
        "original eval_rogue_final.py did.",
    )
    measure.add_argument(
        "--breakdown",
        choices=["auto", "on", "off"],
        default="auto",
        help="Split each step into model-forward and cache-policy time. auto = on "
        "for CUDA (uses CUDA events, no extra synchronisation); 'on' elsewhere "
        "adds a synchronisation per step.",
    )
    measure.add_argument("--seed", type=int, default=0)
    measure.add_argument("--save-text", action="store_true", help="Store decoded answers in the turn rows.")

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=str, default="logs_efficiency")
    output.add_argument("--exp-name", type=str, default="efficiency")
    output.add_argument("--resume", action="store_true", help="Skip configs already measured in <exp>_conversations.jsonl.")
    output.add_argument("--summarize-only", action="store_true", help="Merge all JSONL results in --output-dir and exit.")
    output.add_argument("--fail-fast", action="store_true", help="Raise on the first non-OOM error.")
    return parser


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _clean(values):
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def mean(values):
    values = _clean(values)
    return sum(values) / len(values) if values else None


def stdev(values):
    values = _clean(values)
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def vmax(values):
    values = _clean(values)
    return max(values) if values else None


def percentile(values, q):
    values = sorted(_clean(values))
    if not values:
        return None
    position = (len(values) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def scale(value, factor):
    return None if value is None else value * factor


def safe_div(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------


class DeviceMonitor:
    """Synchronisation and memory statistics for CUDA, MPS or CPU."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.kind = self.device.type
        self._sampled_peak = 0
        self._process = None
        if self.kind == "cpu":
            try:
                import psutil

                self._process = psutil.Process()
            except ImportError:
                self._process = None

    @property
    def exact_peak(self):
        return self.kind == "cuda"

    def sync(self):
        if self.kind == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.kind == "mps":
            torch.mps.synchronize()

    def empty_cache(self):
        gc.collect()
        if self.kind == "cuda":
            torch.cuda.empty_cache()
        elif self.kind == "mps":
            torch.mps.empty_cache()

    def allocated(self):
        if self.kind == "cuda":
            return int(torch.cuda.memory_allocated(self.device))
        if self.kind == "mps":
            return int(torch.mps.current_allocated_memory())
        if self._process is not None:
            return int(self._process.memory_info().rss)
        return 0

    def reset_peak(self):
        if self.kind == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampled_peak = self.allocated()

    def observe(self):
        """Sample current usage (only needed where no exact peak counter exists)."""
        if self.kind != "cuda":
            self._sampled_peak = max(self._sampled_peak, self.allocated())

    def peak(self):
        if self.kind == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        self.observe()
        return self._sampled_peak

    def peak_reserved(self):
        if self.kind == "cuda":
            return int(torch.cuda.max_memory_reserved(self.device))
        if self.kind == "mps":
            return int(torch.mps.driver_allocated_memory())
        return None


class StepClock:
    """Accumulates model-forward vs cache-policy time across decode steps.

    On CUDA it records events, so it adds no host synchronisation; intervals
    are resolved after the turn's own synchronisation point.
    """

    def __init__(self, monitor, enabled):
        self.monitor = monitor
        self.enabled = bool(enabled)
        self.use_events = self.enabled and monitor.kind == "cuda"
        self.reset()

    def reset(self):
        self.forward_s = 0.0
        self.policy_s = 0.0
        self._pending = []

    def mark(self):
        if not self.enabled:
            return None
        if self.use_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        self.monitor.sync()
        return time.perf_counter()

    def add(self, start, middle, end):
        if not self.enabled:
            return
        if self.use_events:
            self._pending.append((start, middle, end))
        else:
            self.forward_s += middle - start
            self.policy_s += end - middle

    def collect(self):
        """Resolve pending events; call only after a device synchronisation."""
        if not self.enabled:
            return None, None
        for start, middle, end in self._pending:
            self.forward_s += start.elapsed_time(middle) / 1000.0
            self.policy_s += middle.elapsed_time(end) / 1000.0
        self._pending = []
        return self.forward_s, self.policy_s


def kv_footprint(past_key_values):
    """Physical K/V bytes and per-layer sequence lengths of a legacy cache."""
    total = 0
    lengths = []
    for layer in past_key_values:
        key, value = layer[0], layer[1]
        total += key.numel() * key.element_size() + value.numel() * value.element_size()
        lengths.append(int(key.shape[2]))
    return total, lengths


def _state_bytes(obj, seen, depth=0):
    if depth > 4 or obj is None:
        return 0
    if torch.is_tensor(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return obj.numel() * obj.element_size()
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (list, tuple)):
        return sum(_state_bytes(x, seen, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return sum(_state_bytes(x, seen, depth + 1) for x in obj.values())
    return 0


def cache_state_bytes(kv_cache):
    if kv_cache is None:
        return 0
    seen = set()
    return sum(_state_bytes(value, seen) for value in vars(kv_cache).values())


class KVTracker:
    """Largest retained / transient KV size, over the conversation and per turn."""

    def __init__(self):
        self.max_transient = 0
        self.max_retained = 0
        self.turn_max_transient = 0
        self.turn_max_retained = 0

    def transient_bytes(self, size):
        self.max_transient = max(self.max_transient, size)
        self.turn_max_transient = max(self.turn_max_transient, size)

    def transient(self, past_key_values):
        self.transient_bytes(kv_footprint(past_key_values)[0])

    def retained(self, past_key_values):
        size, lengths = kv_footprint(past_key_values)
        self.max_retained = max(self.max_retained, size)
        self.turn_max_retained = max(self.turn_max_retained, size)
        return size, lengths

    def start_turn(self, current_bytes):
        self.turn_max_transient = current_bytes
        self.turn_max_retained = current_bytes


def is_oom(exc):
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

FOLLOW_UP_QUESTIONS = (
    "Can you explain your previous answer in more detail?",
    "Which details in the image support that answer?",
    "Summarize our conversation so far in two sentences.",
    "Is there anything in the image you have not mentioned yet?",
)

SYNTHETIC_QUESTIONS = (
    "Describe this image in detail.",
    "What colors and textures stand out the most?",
    "What might have happened just before this picture was taken?",
)


def read_raw_items(args):
    """The dataset as stored on disk (or synthetic items without --data-path)."""
    if not args.data_path:
        count = max(1, args.num_samples + args.sample_offset)
        return [
            {
                "image": None,
                "id": f"synthetic-{i}",
                "turns": [{"question": q, "answer": ""} for q in SYNTHETIC_QUESTIONS],
            }
            for i in range(count)
        ]
    with open(args.data_path, "r") as handle:
        data = json.load(handle)
    return list(data.values()) if isinstance(data, dict) else data


def parse_items(raw_items, source):
    items = []
    for index, raw in enumerate(raw_items):
        if raw.get("turns"):
            turns = [
                {"question": t.get("question", ""), "answer": t.get("answer", "")}
                for t in raw["turns"]
            ]
        elif "question" in raw:
            turns = [{"question": raw["question"], "answer": raw.get("answer", "")}]
        else:
            continue
        image = raw.get("image") or raw.get("imagename")
        if not image and raw.get("image", "") is not None:
            continue
        long_context = raw.get("long_context") or {}
        items.append(
            {
                "image": image,
                "id": raw.get("id", image),
                "turns": turns,
                "index": index,
                "target_prefill_tokens": long_context.get("target_prefill_tokens"),
            }
        )
    if not items:
        raise ValueError(f"No usable items (need an image and questions) in {source}.")
    return items


def conversation_questions(item, num_turns):
    questions = [t["question"] for t in item["turns"]]
    if num_turns is None:
        return questions
    if num_turns < 1:
        raise ValueError("--num-turns must be at least 1.")
    while len(questions) < num_turns:
        questions.append(FOLLOW_UP_QUESTIONS[(len(questions) - 1) % len(FOLLOW_UP_QUESTIONS)])
    return questions[:num_turns]


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

ANSWER_MARK = "@@PREVIOUS_ANSWER@@"


class ConversationState:
    __slots__ = ("past_key_values", "kv_cache", "logical_len", "next_logits")

    def __init__(self):
        self.past_key_values = None
        self.kv_cache = None
        self.logical_len = 0
        self.next_logits = None


class Benchmark:
    def __init__(self, args):
        # Heavy imports live here so --help/--summarize-only stay lightweight.
        global build_kv_cache, effective_context_length
        global finalize_turn_cache, infer_actual_image_span, infer_conv_mode
        global load_image, select_next_token, stop_token_sequences
        global enable_legacy_tgv_attention, set_legacy_logical_position_limit
        global DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, SeparatorStyle, conv_templates
        global get_model_name_from_path, process_images, tokenizer_image_token
        global load_pretrained_model

        from eval_rogue_final import (
            build_kv_cache,
            effective_context_length,
            finalize_turn_cache,
            infer_actual_image_span,
            infer_conv_mode,
            load_image,
            select_next_token,
            stop_token_sequences,
        )
        from kv_cache_final import (
            enable_legacy_tgv_attention,
            set_legacy_logical_position_limit,
        )
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model

        self.args = args
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {args.device} was requested but is unavailable.")
        if args.gen_tokens < 1:
            raise ValueError("--gen-tokens must be at least 1.")

        self.model_name = get_model_name_from_path(args.model_path)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            loader_context_length,
        ) = load_pretrained_model(
            args.model_path,
            args.model_base,
            self.model_name,
            args.load_8bit,
            args.load_4bit,
            device=args.device,
        )
        self.model.eval()
        self.device = self.model.device
        self.dtype = getattr(self.model, "dtype", next(self.model.parameters()).dtype)
        args.conv_mode = infer_conv_mode(self.model_name, args.conv_mode)
        self.template = conv_templates[args.conv_mode]
        # The LLaVA loader reports 2048 whenever the config lacks
        # max_sequence_length (true for llava-v1.5), although the model has
        # max_position_embeddings positions. Use the real position limit for
        # overflow warnings and the --stop-at-eos cap.
        self.loader_context_length = effective_context_length(self.model, loader_context_length)
        self.model_context_length = (
            getattr(self.model.config, "max_position_embeddings", None)
            or self.loader_context_length
        )

        self.monitor = DeviceMonitor(self.device)
        breakdown = args.breakdown == "on" or (
            args.breakdown == "auto" and self.device.type == "cuda"
        )
        self.clock = StepClock(self.monitor, breakdown)

        # Stock transformers 4.31 LlamaAttention sizes its RoPE table by the
        # PHYSICAL cache length but indexes it with the LOGICAL position_ids
        # the evaluator passes, so h2o/local_cache/elastic_cache fail on the
        # first decode step after compression. eval_rogue_final.py patches the
        # attention only for tgv_kv and the turn-conditioned methods. Here every
        # method (including "full") runs on the same patched attention with
        # the logical position limit set each step, so the attention kernel is
        # identical across methods and only the cache policy differs.
        enable_legacy_tgv_attention()

        # Layer-by-layer prefill attention (the per-layer hook added to
        # tgv_kv_legacy_nova.py): each layer's map is folded into the cache and
        # dropped right after that layer, like official TGV-KV's
        # collect_online_prefill_attention.
        self.online_attention = None
        if args.prefill_attention == "per_layer":
            try:
                from tgv_kv_legacy_nova import enable_online_attention, online_attention
            except ImportError as exc:
                raise SystemExit(
                    "tgv_kv_legacy_nova.py has no enable_online_attention/online_attention. "
                    "Apply the layer-by-layer changes, or pass --prefill-attention all_layers."
                ) from exc
            enable_online_attention(self.model)
            self.online_attention = online_attention

        self.image_tokens = self._count_image_tokens()
        self.stop_sequences = stop_token_sequences(self.template.copy(), self.tokenizer)
        self.separator_ids, _ = self.turn_segment_ids("placeholder")
        self._warned = set()
        self.raw_items = read_raw_items(args)
        self.datasets = {}
        self.dataset_paths = {}
        self.prepare_datasets()

    # ------------------------------------------------------------------ setup

    def warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[WARNING] {message}", flush=True)

    def _count_image_tokens(self):
        try:
            tower = self.model.get_vision_tower()
            return int(tower.num_patches)
        except Exception:
            return 576

    def prepare_datasets(self):
        """Load the native dataset and generate/load one long-context file per N."""
        args = self.args
        source = args.data_path or "synthetic items"
        long_dir = args.long_context_dir or (
            os.path.join(os.path.dirname(args.data_path) or ".", "long_context")
            if args.data_path
            else os.path.join(args.output_dir, "long_context")
        )
        counter = None
        for context_length in args.context_lengths:
            if context_length == "native":
                self.datasets["native"] = parse_items(self.raw_items, source)
                self.dataset_paths["native"] = args.data_path
                continue
            try:
                import make_long_context 
            except ImportError as exc:
                raise SystemExit(
                    f"--context-lengths {context_length} needs make_long_context.py next to "
                    "eval_efficiency.py (or run with --context-lengths native)."
                ) from exc
            path = make_long_context.long_context_path(long_dir, args.data_path, context_length)
            if os.path.exists(path) and not args.regenerate_long_context:
                with open(path, "r") as handle:
                    raw = json.load(handle)
                meta = (raw[0].get("long_context") or {}) if raw else {}
                if meta and (
                    meta.get("image_tokens") != self.image_tokens
                    or meta.get("conv_mode") != args.conv_mode
                    or meta.get("target_prefill_tokens") != context_length
                ):
                    raise SystemExit(
                        f"{path} was generated for image_tokens={meta.get('image_tokens')}, "
                        f"conv_mode={meta.get('conv_mode')}, target={meta.get('target_prefill_tokens')}; "
                        f"this model needs {self.image_tokens}/{args.conv_mode}/{context_length}. "
                        "Pass --regenerate-long-context or a different --long-context-dir."
                    )
                print(f"ctx={context_length}: using {len(raw)} samples from {path}", flush=True)
            else:
                if counter is None:
                    counter = make_long_context.PrefillCounter(
                        self.tokenizer, self.template, self.image_tokens, args.conv_mode
                    )
                start = time.perf_counter()
                raw, report = make_long_context.build_long_context_dataset(
                    self.raw_items, counter, context_length,
                    seed=args.seed, filler_file=args.filler_file,
                )
                make_long_context.write_dataset(path, raw)
                print(
                    f"ctx={context_length}: generated {report['written']} samples "
                    f"({time.perf_counter() - start:.1f}s) -> {path}",
                    flush=True,
                )
            self.datasets[context_length] = parse_items(raw, path)
            self.dataset_paths[context_length] = path

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def turn_segment_ids(self, question):
        """Token ids of [separator after the previous answer] and [new user turn]."""
        conv = self.template.copy()
        conv.append_message(conv.roles[0], "question")
        conv.append_message(conv.roles[1], ANSWER_MARK)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        segment = prompt[prompt.rindex(ANSWER_MARK) + len(ANSWER_MARK) :]
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop and segment.startswith(stop):
            return self._encode(stop), self._encode(segment[len(stop) :])
        return [], self._encode(segment)

    def first_turn_ids(self, question):
        """Turn-1 prompt ids, built exactly like eval_rogue_final.py."""
        if DEFAULT_IMAGE_TOKEN not in question:
            question = question + "\n" + DEFAULT_IMAGE_TOKEN
        conv = self.template.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return list(tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX))

    def make_policy_config(self, method, ratio, max_logical_len):
        args = self.args
        recent = args.recent_size
        if recent is None:
            recent = max(2047, int(max_logical_len) + 64)
        elif method == "elastic_cache" and args.start_size + recent + 1 < max_logical_len:
            self.warn_once(
                ("recent", recent),
                f"--recent-size {recent} is smaller than the conversation "
                f"(~{max_logical_len} tokens); elastic_cache will fail. "
                "Omit --recent-size to size it automatically.",
            )
        return SimpleNamespace(
            method=method,
            ratio=ratio,
            start_size=args.start_size,
            recent_size=recent,
            alpha=args.alpha,
            beta=args.beta,
            cross_turn_weight=args.cross_turn_weight,
        )

    def load_item_image(self, item):
        if item["image"] is None:
            rng = np.random.default_rng(12345)
            from PIL import Image

            pixels = rng.integers(0, 256, size=(336, 336, 3), dtype=np.uint8)
            return Image.fromarray(pixels, mode="RGB")
        return load_image(os.path.join(self.args.image_path, item["image"]))

    # --------------------------------------------------------------- stepping

    def step(self, state, token_ids, need_attentions, tracker):
        """One forward + cache update; mirrors eval_rogue_final.forward_one_token."""
        position_ids = torch.tensor(
            [[state.logical_len]], dtype=torch.long, device=token_ids.device
        )
        set_legacy_logical_position_limit(self.model, state.logical_len + 1)

        start = self.clock.mark()
        outputs = self.model(
            input_ids=token_ids,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_attentions=need_attentions,
        )
        middle = self.clock.mark()
        state.logical_len += outputs.logits.shape[1]
        past = outputs.past_key_values
        tracker.transient(past)
        if state.kv_cache is not None:
            past = state.kv_cache(
                past, num_of_token=state.logical_len, attentions=outputs.attentions
            )
        end = self.clock.mark()
        self.clock.add(start, middle, end)

        state.past_key_values = past
        state.next_logits = outputs.logits[:, -1, :]
        tracker.retained(past)
        self.monitor.observe()

    def feed(self, state, ids, need_attentions, tracker):
        if not ids:
            return
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        for position in range(tokens.shape[1]):
            self.step(state, tokens[:, position : position + 1], need_attentions, tracker)

    def decode(self, state, limit, need_attentions, tracker):
        """Generate one answer; same order as eval_rogue_final.generate_one_turn."""
        args = self.args
        generated, times = [], []
        committed_stop, stop_reason = [], "max_tokens"

        next_id = select_next_token(state.next_logits, args.temperature, args.top_p)
        generated.append(int(next_id.item()))
        times.append(time.perf_counter())
        while True:
            self.step(state, next_id, need_attentions, tracker)
            if args.stop_at_eos:
                matched = next(
                    (
                        seq
                        for seq in self.stop_sequences
                        if len(generated) >= len(seq) and generated[-len(seq) :] == seq
                    ),
                    None,
                )
                if matched is not None:
                    committed_stop, stop_reason = list(matched), "eos_or_separator"
                    break
            if len(generated) >= limit:
                break
            next_id = select_next_token(state.next_logits, args.temperature, args.top_p)
            generated.append(int(next_id.item()))
            times.append(time.perf_counter())
        return generated, committed_stop, times, stop_reason

    # ---------------------------------------------------------- conversation

    def run_conversation(self, config, item, sample_index):
        args = self.args
        monitor = self.monitor
        method = config["method"]
        need_attentions = config["output_attentions"]
        questions = conversation_questions(item, args.num_turns)

        seed = (args.seed + sample_index) % (2 ** 32)  # warm-ups use negative indices
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        monitor.empty_cache()
        monitor.sync()
        base_memory = monitor.allocated()
        monitor.reset_peak()
        tracker = KVTracker()
        state = ConversationState()
        turn_rows = []
        overhead = 0.0  # bookkeeping time, excluded from every timing
        input_tokens = 0
        generated_tokens = 0
        max_state_bytes = 0

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with torch.no_grad():
                t_conversation = time.perf_counter()
                image = self.load_item_image(item)
                t_turn_start = time.perf_counter()
                image_load_s = t_turn_start - t_conversation

                # ---------------- turn 1: preprocessing + prefill + policy
                image_tensor = process_images([image], self.image_processor, args).to(
                    self.device, dtype=self.dtype
                )
                ids = self.first_turn_ids(questions[0])
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                placeholders = ids.count(IMAGE_TOKEN_INDEX)
                expected_prefill = len(ids) - placeholders + placeholders * self.image_tokens
                expected_span = (
                    (ids.index(IMAGE_TOKEN_INDEX), self.image_tokens) if placeholders else (None, 0)
                )
                user_lengths = [len(self.turn_segment_ids(q)[1]) for q in questions[1:]]
                max_logical = (
                    expected_prefill
                    + len(questions) * (args.gen_tokens + len(self.separator_ids))
                    + sum(user_lengths)
                )
                monitor.sync()
                t_prefill = time.perf_counter()

                # The cache exists BEFORE the prefill so that, in per_layer mode,
                # each layer's attention is folded into it and dropped as soon
                # as that layer finishes (never all 32 maps at once).
                if method != "full":
                    policy = self.make_policy_config(method, config["ratio"], max_logical)
                    state.kv_cache = build_kv_cache(
                        policy,
                        self.model_name,
                        image_start=expected_span[0],
                        image_token_len=expected_span[1],
                    )
                sink = None
                if config["prefill_attention"] == "per_layer":
                    sink = getattr(state.kv_cache, "observe_layer", None)
                    if sink is None:
                        raise RuntimeError(
                            f"{type(state.kv_cache).__name__} has no observe_layer(); apply the "
                            "layer-by-layer changes to the repo, or pass --prefill-attention all_layers."
                        )
                with self.online_attention(self.model, sink) if sink else contextlib.nullcontext():
                    outputs = self.model(
                        input_ids=input_ids,
                        images=image_tensor,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=need_attentions,
                    )
                monitor.sync()
                t_forward = time.perf_counter()

                raw = outputs.past_key_values
                state.logical_len = int(raw[0][0].size(2))
                prefill_tokens = state.logical_len
                full_prefill_bytes, _ = kv_footprint(raw)
                kv_bytes_per_token = full_prefill_bytes / max(1, prefill_tokens)
                tracker.transient_bytes(full_prefill_bytes)
                image_start, image_token_len = infer_actual_image_span(input_ids, raw)
                if prefill_tokens != expected_prefill or (image_start, image_token_len) != expected_span:
                    raise RuntimeError(
                        f"prefill expanded to {prefill_tokens} tokens / image span "
                        f"{(image_start, image_token_len)}, expected {expected_prefill} / {expected_span}"
                    )
                if sink is not None and any(a is not None for a in (outputs.attentions or ())):
                    raise RuntimeError("The per-layer attention hook did not drop the attention maps.")

                if state.kv_cache is not None:
                    raw = state.kv_cache(
                        raw,
                        num_of_token=state.logical_len,
                        attentions=None if sink is not None else outputs.attentions,
                    )
                # ElasticCache with observe_layer() only scores the prefill and
                # LocalCache never reads attention: decode steps skip the maps.
                decode_attentions = need_attentions and not (
                    args.output_attentions == "auto"
                    and method == "elastic_cache"
                    and hasattr(state.kv_cache, "observe_layer")
                )
                state.past_key_values = raw
                state.next_logits = outputs.logits[:, -1, :].clone()
                del outputs, raw
                monitor.sync()
                t_policy = time.perf_counter()

                # ---- accounting (excluded from all timings)
                t_accounting = time.perf_counter()
                kv_after_prefill, _ = tracker.retained(state.past_key_values)
                prefill_peak = monitor.peak()  # turn-1 peak keeps accumulating
                target = item.get("target_prefill_tokens")
                if target is not None and prefill_tokens != target:
                    self.warn_once(
                        ("target", config["context_length"]),
                        f"ctx={config['context_length']}: prefill is {prefill_tokens} tokens, "
                        f"the sample was generated for {target} (different tokenizer/model?). "
                        "Rows record the measured length; --regenerate-long-context rebuilds.",
                    )
                if image_token_len and image_token_len != self.image_tokens:
                    self.warn_once(
                        "image_tokens",
                        f"image expands to {image_token_len} tokens, not {self.image_tokens}.",
                    )
                if (
                    not args.stop_at_eos
                    and self.model_context_length
                    and max_logical > self.model_context_length
                ):
                    self.warn_once(
                        ("overflow", config["context_length"]),
                        f"conversations at context {config['context_length']} reach "
                        f"~{max_logical} positions, beyond the model's "
                        f"{self.model_context_length}; timings are still valid but "
                        "outputs past that point are out of distribution.",
                    )
                accounting_s = time.perf_counter() - t_accounting
                overhead += accounting_s
                t_turn_start += accounting_s  # keep bookkeeping out of turn-1 TTFT
                input_tokens += prefill_tokens

                prefill_info = {
                    "prefill_tokens": prefill_tokens,
                    "target_prefill_tokens": target,
                    "image_tokens": image_token_len,
                    "preprocess_s": t_prefill - t_turn_start + accounting_s,
                    "prefill_forward_s": t_forward - t_prefill,
                    "prefill_policy_s": t_policy - t_forward,
                    "prefill_tps": prefill_tokens / (t_policy - t_prefill),
                    "kv_after_prefill_bytes": kv_after_prefill,
                    "peak_prefill_mem_bytes": prefill_peak,
                }

                # ---------------- turns
                for turn_index, question in enumerate(questions):
                    self.clock.reset()
                    if turn_index == 0:
                        user_tokens = prefill_tokens
                    else:
                        t_turn_start = time.perf_counter()
                        _, user_ids = self.turn_segment_ids(question)
                        user_tokens = len(user_ids)
                        input_tokens += user_tokens
                        self.feed(state, user_ids, decode_attentions, tracker)

                    limit = args.gen_tokens
                    if args.stop_at_eos and self.model_context_length:
                        limit = max(1, min(limit, self.model_context_length - state.logical_len - 8))
                    generated, committed_stop, times, stop_reason = self.decode(
                        state, limit, decode_attentions, tracker
                    )

                    # Boundary: separator + finalize_turn (boundary compression).
                    separator = list(self.separator_ids)
                    if committed_stop and separator[: len(committed_stop)] == committed_stop:
                        separator = separator[len(committed_stop) :]
                    self.feed(state, separator, decode_attentions, tracker)
                    input_tokens += len(separator)
                    monitor.sync()
                    t_finalize = time.perf_counter()
                    if state.kv_cache is not None:
                        state.past_key_values = finalize_turn_cache(
                            state.kv_cache, state.past_key_values
                        )
                    monitor.sync()
                    t_turn_end = time.perf_counter()

                    # ---- accounting (excluded from all timings)
                    t_accounting = time.perf_counter()
                    turn_peak = monitor.peak()
                    turn_reserved = monitor.peak_reserved()
                    e2e_cum = t_turn_end - t_conversation - overhead
                    forward_s, policy_s = self.clock.collect()
                    kv_end, lengths = tracker.retained(state.past_key_values)
                    kv_full_end = int(round(kv_bytes_per_token * state.logical_len))
                    state_bytes = cache_state_bytes(state.kv_cache)
                    max_state_bytes = max(max_state_bytes, state_bytes)
                    n = len(generated)
                    generated_tokens += n
                    turn_input = user_tokens + len(separator)
                    turn_e2e = t_turn_end - t_turn_start
                    intervals = [b - a for a, b in zip(times, times[1:])]
                    decode_span = times[-1] - times[0]
                    row = {
                        "turn": turn_index + 1,
                        "user_tokens": user_tokens,
                        "turn_input_tokens": turn_input,
                        "generated_tokens": n,
                        "answer_tokens": n - len(committed_stop),
                        "stop_reason": stop_reason,
                        # time
                        "ttft_s": times[0] - t_turn_start,
                        "tpot_s": decode_span / (n - 1) if n > 1 else None,
                        "decode_span_s": decode_span,
                        "latency_s": times[-1] - t_turn_start,
                        "boundary_s": t_turn_end - times[-1],
                        "finalize_s": t_turn_end - t_finalize,
                        "turn_e2e_s": turn_e2e,
                        "e2e_cum_s": e2e_cum,
                        "itl_p50_s": percentile(intervals, 50),
                        "itl_p90_s": percentile(intervals, 90),
                        "itl_p99_s": percentile(intervals, 99),
                        "itl_max_s": max(intervals) if intervals else None,
                        "itl_s": [round(x, 7) for x in intervals],
                        "step_forward_s": forward_s,
                        "step_policy_s": policy_s,
                        # throughput
                        "decode_tps": (n - 1) / decode_span if n > 1 and decode_span > 0 else None,
                        "turn_output_tps": n / turn_e2e,
                        "turn_total_tps": (turn_input + n) / turn_e2e,
                        # memory
                        "peak_mem_bytes": turn_peak,
                        "peak_mem_over_base_bytes": turn_peak - base_memory,
                        "peak_reserved_bytes": turn_reserved,
                        "peak_mem_exact": monitor.exact_peak,
                        "logical_len_end": state.logical_len,
                        "kv_end_bytes": kv_end,
                        "kv_peak_bytes": tracker.turn_max_retained,
                        "kv_transient_peak_bytes": tracker.turn_max_transient,
                        "kv_full_end_bytes": kv_full_end,
                        "kv_ratio": kv_end / kv_full_end if kv_full_end else None,
                        "kv_end_tokens_mean": sum(lengths) / len(lengths),
                        "kv_end_tokens_min": min(lengths),
                        "kv_end_tokens_max": max(lengths),
                        "cache_state_bytes": state_bytes,
                    }
                    if turn_index == 0:
                        row.update(prefill_info)
                    if args.save_text:
                        row["answer"] = self.tokenizer.decode(
                            generated[: n - len(committed_stop)], skip_special_tokens=True
                        )
                    turn_rows.append(row)

                    # Next turn starts its own peak-memory and KV windows.
                    tracker.start_turn(kv_end)
                    monitor.reset_peak()
                    overhead += time.perf_counter() - t_accounting

                e2e_s = turn_rows[-1]["e2e_cum_s"]
                peak = max(r["peak_mem_bytes"] for r in turn_rows)
                reserved = [r["peak_reserved_bytes"] for r in turn_rows]
                conversation = {
                    "status": "ok",
                    "num_turns": len(questions),
                    "prefill_tokens": prefill_tokens,
                    "target_prefill_tokens": target,
                    "image_tokens": image_token_len,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "final_logical_len": state.logical_len,
                    "image_load_s": image_load_s,
                    "e2e_s": e2e_s,
                    "e2e_output_tps": generated_tokens / e2e_s,
                    "e2e_total_tps": (input_tokens + generated_tokens) / e2e_s,
                    "prefill_tps": prefill_info["prefill_tps"],
                    "base_mem_bytes": base_memory,
                    "peak_mem_bytes": peak,
                    "peak_prefill_mem_bytes": prefill_peak,
                    "peak_mem_over_base_bytes": peak - base_memory,
                    "peak_reserved_bytes": max(reserved) if None not in reserved else None,
                    "peak_mem_exact": monitor.exact_peak,
                    "kv_bytes_per_token_full": kv_bytes_per_token,
                    "kv_after_prefill_bytes": kv_after_prefill,
                    "kv_max_retained_bytes": tracker.max_retained,
                    "kv_max_transient_bytes": tracker.max_transient,
                    "kv_final_bytes": turn_rows[-1]["kv_end_bytes"],
                    "kv_full_final_bytes": turn_rows[-1]["kv_full_end_bytes"],
                    "kv_compression": turn_rows[-1]["kv_ratio"],
                    "cache_state_max_bytes": max_state_bytes,
                }
                return conversation, turn_rows
        finally:
            state.past_key_values = None
            state.kv_cache = None
            state.next_logits = None
            if gc_was_enabled:
                gc.enable()
            monitor.empty_cache()

    # ------------------------------------------------------------------ sweep

    def configs(self):
        args = self.args
        for context_length in args.context_lengths:
            for method in args.methods:
                ratios = [0.0] if method == "full" else args.ratios
                for ratio in ratios:
                    reads_attention = (
                        args.output_attentions == "always" or method not in ATTENTION_FREE_METHODS
                    )
                    yield {
                        "model": self.model_name,
                        "context_length": context_length,
                        "method": method,
                        "ratio": float(ratio),
                        "kv_budget": 1.0 - float(ratio),
                        "gen_tokens": args.gen_tokens,
                        "num_turns_arg": args.num_turns,
                        "stop_at_eos": bool(args.stop_at_eos),
                        "temperature": args.temperature,
                        "output_attentions": reads_attention,
                        "prefill_attention": (
                            "none"
                            if not reads_attention
                            else "per_layer"
                            if args.prefill_attention == "per_layer" and method != "full"
                            else "all_layers"
                        ),
                    }

    def run(self, paths):
        args = self.args
        done = completed_configs(paths["conversations"]) if args.resume else {}
        configs = list(self.configs())
        for config_index, config in enumerate(configs, start=1):
            key = config_key(config)
            label = (
                f"[{config_index}/{len(configs)}] ctx={config['context_length']} "
                f"method={config['method']} ratio={config['ratio']}"
            )
            if done.get(key, 0) >= args.num_samples:
                print(f"{label}: already measured, skipping", flush=True)
                continue

            print(label, flush=True)
            dataset = self.datasets[config["context_length"]]
            items = [
                dataset[(args.sample_offset + i) % len(dataset)]
                for i in range(args.num_samples)
            ]
            try:
                for warm in range(args.warmup):
                    self.run_conversation(config, items[warm % len(items)], -1 - warm)
            except Exception as exc:  # noqa: BLE001 - keep sweeping
                if self._record_failure(config, exc, paths, label, sample_index=-1):
                    continue

            for sample_index, item in enumerate(items):
                try:
                    conversation, turns = self.run_conversation(config, item, sample_index)
                except Exception as exc:  # noqa: BLE001 - keep sweeping
                    if self._record_failure(config, exc, paths, label, sample_index):
                        break
                    continue
                base = dict(
                    config,
                    sample_index=sample_index,
                    item_id=str(item.get("id")),
                    data_file=self.dataset_paths.get(config["context_length"]),
                )
                append_jsonl(paths["conversations"], [dict(base, **conversation)])
                append_jsonl(paths["turns"], [dict(base, **turn) for turn in turns])
                per_turn = " | ".join(
                    f"T{t['turn']} ttft={t['ttft_s'] * 1e3:.0f}ms "
                    f"tpot={_fmt(t['tpot_s'], 1e3)}ms kv={t['kv_end_bytes'] / MIB:.0f}MiB "
                    f"peak={t['peak_mem_bytes'] / GIB:.2f}GiB"
                    for t in turns
                )
                print(
                    f"    sample {sample_index}: prefill={conversation['prefill_tokens']} "
                    f"e2e={conversation['e2e_s']:.2f}s peak={conversation['peak_mem_bytes'] / GIB:.2f}GiB "
                    f"| {per_turn}",
                    flush=True,
                )

    def _record_failure(self, config, exc, paths, label, sample_index):
        """Log a failed conversation. Returns True when the config should stop."""
        oom = is_oom(exc)
        status = "oom" if oom else f"error: {type(exc).__name__}: {exc}"
        print(f"    {label}: {status}", flush=True)
        if not oom and self.args.fail_fast:
            raise exc
        row = dict(config, sample_index=sample_index, item_id=None, status=status)
        append_jsonl(paths["conversations"], [row])
        self.monitor.empty_cache()
        return True  # an OOM or error usually repeats for the whole config


def _fmt(value, factor=1.0, digits=1):
    return "n/a" if value is None else f"{value * factor:.{digits}f}"


# --------------------------------------------------------------------------
# Files and summaries
# --------------------------------------------------------------------------

KEY_FIELDS = (
    "model",
    "context_length",
    "method",
    "ratio",
    "gen_tokens",
    "num_turns_arg",
    "stop_at_eos",
    "temperature",
    "output_attentions",
    "prefill_attention",
)


def config_key(row):
    return tuple(json.dumps(row.get(field), sort_keys=True) for field in KEY_FIELDS)


def append_jsonl(path, rows):
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def completed_configs(path):
    counts = {}
    for row in read_jsonl([path]):
        if row.get("status") == "ok":
            key = config_key(row)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _context_sort(value):
    return (0, 0) if value == "native" else (1, int(value))


def _method_sort(value):
    return ALL_METHODS.index(value) if value in ALL_METHODS else len(ALL_METHODS)


def _decode_tps(rows):
    spans = sum(r["decode_span_s"] for r in rows)
    steps = sum(max(0, r["generated_tokens"] - 1) for r in rows)
    return safe_div(steps, spans), safe_div(spans, steps)


def _policy_share(rows):
    forward = _clean([r.get("step_forward_s") for r in rows])
    policy = _clean([r.get("step_policy_s") for r in rows])
    if not policy:
        return None
    return safe_div(sum(policy), sum(forward) + sum(policy))


def summarize_turns(rows):
    """Metrics for one (config, turn) group; means over conversations."""
    decode_tps, tpot = _decode_tps(rows)
    intervals = [x for r in rows for x in r.get("itl_s", [])]
    first = rows[0]
    record = {
        "num_samples": len(rows),
        "user_tokens": mean([r["user_tokens"] for r in rows]),
        "turn_input_tokens": mean([r["turn_input_tokens"] for r in rows]),
        "generated_tokens": mean([r["generated_tokens"] for r in rows]),
        "logical_len_end": mean([r["logical_len_end"] for r in rows]),
        # time
        "ttft_ms": scale(mean([r["ttft_s"] for r in rows]), 1e3),
        "ttft_std_ms": scale(stdev([r["ttft_s"] for r in rows]), 1e3),
        "tpot_ms": scale(tpot, 1e3),
        "tpot_std_ms": scale(stdev([r["tpot_s"] for r in rows]), 1e3),
        "itl_p50_ms": scale(percentile(intervals, 50), 1e3),
        "itl_p90_ms": scale(percentile(intervals, 90), 1e3),
        "itl_p99_ms": scale(percentile(intervals, 99), 1e3),
        "latency_ms": scale(mean([r["latency_s"] for r in rows]), 1e3),
        "latency_std_ms": scale(stdev([r["latency_s"] for r in rows]), 1e3),
        "boundary_ms": scale(mean([r["boundary_s"] for r in rows]), 1e3),
        "finalize_ms": scale(mean([r["finalize_s"] for r in rows]), 1e3),
        "turn_e2e_s": mean([r["turn_e2e_s"] for r in rows]),
        "turn_e2e_std_s": stdev([r["turn_e2e_s"] for r in rows]),
        "e2e_cum_s": mean([r["e2e_cum_s"] for r in rows]),
        # throughput
        "decode_tps": decode_tps,
        "turn_output_tps": mean([r["turn_output_tps"] for r in rows]),
        "turn_total_tps": mean([r["turn_total_tps"] for r in rows]),
        # memory
        "peak_mem_gib": scale(mean([r["peak_mem_bytes"] for r in rows]), 1 / GIB),
        "peak_mem_max_gib": scale(vmax([r["peak_mem_bytes"] for r in rows]), 1 / GIB),
        "peak_over_base_gib": scale(mean([r["peak_mem_over_base_bytes"] for r in rows]), 1 / GIB),
        "peak_reserved_gib": scale(vmax([r.get("peak_reserved_bytes") for r in rows]), 1 / GIB),
        "peak_mem_exact": all(r.get("peak_mem_exact") for r in rows),
        "kv_end_mib": scale(mean([r["kv_end_bytes"] for r in rows]), 1 / MIB),
        "kv_peak_mib": scale(mean([r["kv_peak_bytes"] for r in rows]), 1 / MIB),
        "kv_transient_peak_mib": scale(mean([r["kv_transient_peak_bytes"] for r in rows]), 1 / MIB),
        "kv_full_end_mib": scale(mean([r["kv_full_end_bytes"] for r in rows]), 1 / MIB),
        "kv_ratio": mean([r["kv_ratio"] for r in rows]),
        "kv_end_tokens_mean": mean([r["kv_end_tokens_mean"] for r in rows]),
        "cache_state_mib": scale(mean([r["cache_state_bytes"] for r in rows]), 1 / MIB),
        "policy_share": _policy_share(rows),
    }
    if first["turn"] == 1 and "prefill_tokens" in first:
        record.update(
            {
                "prefill_tokens": mean([r["prefill_tokens"] for r in rows]),
                "prefill_forward_ms": scale(mean([r["prefill_forward_s"] for r in rows]), 1e3),
                "prefill_policy_ms": scale(mean([r["prefill_policy_s"] for r in rows]), 1e3),
                "prefill_tps": mean([r["prefill_tps"] for r in rows]),
                "peak_prefill_gib": scale(mean([r["peak_prefill_mem_bytes"] for r in rows]), 1 / GIB),
                "kv_after_prefill_mib": scale(mean([r["kv_after_prefill_bytes"] for r in rows]), 1 / MIB),
            }
        )
    return record


def summarize(conversations, turns):
    groups = {}
    for row in conversations:
        groups.setdefault(config_key(row), []).append(row)
    turn_groups = {}
    for row in turns:
        turn_groups.setdefault(config_key(row), []).append(row)

    summary, per_turn = [], []
    for key, rows in groups.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        ok_samples = {r["sample_index"] for r in ok}
        trows = [t for t in turn_groups.get(key, []) if t["sample_index"] in ok_samples]
        first = rows[0]
        ident = {field: first.get(field) for field in KEY_FIELDS}
        ident["kv_budget"] = 1.0 - float(first.get("ratio", 0.0))
        record = dict(ident)
        record["status"] = "ok" if ok else (rows[-1].get("status") or "failed")
        record["num_samples"] = len(ok)
        if ok:
            turn1 = [t for t in trows if t["turn"] == 1]
            later = [t for t in trows if t["turn"] > 1]
            decode_tps, tpot = _decode_tps(trows)
            intervals = [x for t in trows for x in t.get("itl_s", [])]
            record.update(
                {
                    "num_turns": mean([r["num_turns"] for r in ok]),
                    "prefill_tokens": mean([r["prefill_tokens"] for r in ok]),
                    "final_logical_len": mean([r["final_logical_len"] for r in ok]),
                    "generated_tokens": mean([r["generated_tokens"] for r in ok]),
                    "ttft_turn1_ms": scale(mean([t["ttft_s"] for t in turn1]), 1e3),
                    "ttft_later_ms": scale(mean([t["ttft_s"] for t in later]), 1e3),
                    "tpot_ms": scale(tpot, 1e3),
                    "itl_p99_ms": scale(percentile(intervals, 99), 1e3),
                    "latency_turn_ms": scale(mean([t["latency_s"] for t in trows]), 1e3),
                    "boundary_ms": scale(mean([t["boundary_s"] for t in trows]), 1e3),
                    "decode_tps": decode_tps,
                    "prefill_tps": mean([r["prefill_tps"] for r in ok]),
                    "e2e_s": mean([r["e2e_s"] for r in ok]),
                    "e2e_std_s": stdev([r["e2e_s"] for r in ok]),
                    "e2e_output_tps": mean([r["e2e_output_tps"] for r in ok]),
                    "e2e_total_tps": mean([r["e2e_total_tps"] for r in ok]),
                    "peak_mem_gib": scale(mean([r["peak_mem_bytes"] for r in ok]), 1 / GIB),
                    "peak_mem_max_gib": scale(vmax([r["peak_mem_bytes"] for r in ok]), 1 / GIB),
                    "peak_over_base_gib": scale(mean([r["peak_mem_over_base_bytes"] for r in ok]), 1 / GIB),
                    "peak_prefill_gib": scale(mean([r["peak_prefill_mem_bytes"] for r in ok]), 1 / GIB),
                    "peak_reserved_gib": scale(vmax([r.get("peak_reserved_bytes") for r in ok]), 1 / GIB),
                    "peak_mem_exact": all(r.get("peak_mem_exact") for r in ok),
                    "kv_after_prefill_mib": scale(mean([r["kv_after_prefill_bytes"] for r in ok]), 1 / MIB),
                    "kv_final_mib": scale(mean([r["kv_final_bytes"] for r in ok]), 1 / MIB),
                    "kv_full_final_mib": scale(mean([r["kv_full_final_bytes"] for r in ok]), 1 / MIB),
                    "kv_compression": mean([r["kv_compression"] for r in ok]),
                    "kv_max_retained_mib": scale(mean([r["kv_max_retained_bytes"] for r in ok]), 1 / MIB),
                    "kv_max_transient_mib": scale(mean([r["kv_max_transient_bytes"] for r in ok]), 1 / MIB),
                    "cache_state_mib": scale(vmax([r["cache_state_max_bytes"] for r in ok]), 1 / MIB),
                    "policy_share": _policy_share(trows),
                }
            )
            for turn in sorted({t["turn"] for t in trows}):
                rows_t = [t for t in trows if t["turn"] == turn]
                per_turn.append(dict(ident, turn=turn, **summarize_turns(rows_t)))
        summary.append(record)

    order = lambda r: (  # noqa: E731
        str(r.get("model")),
        _context_sort(r.get("context_length")),
        _method_sort(r.get("method")),
        float(r.get("ratio") or 0.0),
        str(r.get("prefill_attention")),
        r.get("turn", 0),
    )
    return sorted(summary, key=order), sorted(per_turn, key=order)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            )


def _all_layers(r):
    mode = r.get("prefill_attention")
    return mode == "all_layers" or (mode is None and bool(r.get("output_attentions")))


def _method_label(r):
    return r["method"] + ("*" if _all_layers(r) else "")


ALL_LAYERS_NOTE = (
    "* turn-1 attention maps of all 32 layers held at once "
    "(--prefill-attention all_layers, or results recorded before that option existed)"
)


def print_per_turn_table(per_turn, summary):
    header = (
        f"{'ctx':>7} {'method':<36} {'ratio':>5} {'turn':>4} {'TTFT ms':>9} {'TPOT ms':>8} "
        f"{'lat ms':>9} {'turn e2e s':>10} {'cum e2e s':>9} {'tok/s':>7} {'peak GiB':>8} "
        f"{'KV MiB':>8} {'KV pk MiB':>9} {'KV %':>6}"
    )
    print("\nPer-turn results (mean over conversations)")
    print(header)
    print("-" * len(header))
    failed = [r for r in summary if r["status"] != "ok"]
    previous = None
    for r in per_turn:
        config = (r["context_length"], r["method"], r["ratio"], r.get("prefill_attention"))
        if previous is not None and config != previous and r["turn"] == 1:
            print()
        previous = config
        print(
            f"{str(r['context_length']):>7} {_method_label(r):<36} {r['ratio']:>5.2f} "
            f"{r['turn']:>4} {_fmt(r['ttft_ms']):>9} {_fmt(r['tpot_ms'], 1, 2):>8} "
            f"{_fmt(r['latency_ms']):>9} {_fmt(r['turn_e2e_s'], 1, 3):>10} "
            f"{_fmt(r['e2e_cum_s'], 1, 3):>9} {_fmt(r['decode_tps']):>7} "
            f"{_fmt(r['peak_mem_gib'], 1, 2):>8} {_fmt(r['kv_end_mib']):>8} "
            f"{_fmt(r['kv_peak_mib']):>9} {_fmt(r['kv_ratio'], 100):>6}"
        )
    for r in failed:
        print(f"{str(r['context_length']):>7} {_method_label(r):<36} {r['ratio']:>5.2f}  {r['status']}")
    if any(_all_layers(r) for r in summary):
        print(ALL_LAYERS_NOTE)
    print()


def print_table(summary):
    header = (
        f"{'ctx':>7} {'method':<36} {'ratio':>5} {'prefill':>7} {'TTFT1 ms':>9} "
        f"{'TTFT2+ ms':>9} {'TPOT ms':>8} {'e2e s':>7} {'out tok/s':>9} "
        f"{'peak GiB':>8} {'KV MiB':>8} {'KV %':>6}"
    )
    print("Whole-conversation results")
    print(header)
    print("-" * len(header))
    for r in summary:
        if r["status"] != "ok":
            print(f"{str(r['context_length']):>7} {_method_label(r):<36} {r['ratio']:>5.2f}  {r['status']}")
            continue
        print(
            f"{str(r['context_length']):>7} {_method_label(r):<36} {r['ratio']:>5.2f} "
            f"{_fmt(r['prefill_tokens'], 1, 0):>7} {_fmt(r['ttft_turn1_ms']):>9} "
            f"{_fmt(r['ttft_later_ms']):>9} {_fmt(r['tpot_ms'], 1, 2):>8} "
            f"{_fmt(r['e2e_s'], 1, 2):>7} {_fmt(r['e2e_output_tps']):>9} "
            f"{_fmt(r['peak_mem_gib'], 1, 2):>8} {_fmt(r['kv_final_mib']):>8} "
            f"{_fmt(r['kv_compression'], 100):>6}"
        )
    if any(_all_layers(r) for r in summary):
        print(ALL_LAYERS_NOTE)
    print()


def output_paths(args):
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, args.exp_name)
    return {
        "conversations": f"{stem}_conversations.jsonl",
        "turns": f"{stem}_turns.jsonl",
        "summary": f"{stem}_summary.csv",
        "per_turn": f"{stem}_summary_per_turn.csv",
        "meta": f"{stem}_meta.json",
    }


def write_summaries(conversation_paths, turn_paths, summary_path, per_turn_path):
    summary, per_turn = summarize(read_jsonl(conversation_paths), read_jsonl(turn_paths))
    write_csv(summary_path, summary)
    write_csv(per_turn_path, per_turn)
    print_per_turn_table(per_turn, summary)
    print_table(summary)
    print(f"Per-turn summary:          {per_turn_path}")
    print(f"Whole-conversation summary: {summary_path}")


def write_meta(path, args, bench):
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001
        transformers_version = None
    device_name = None
    if bench.device.type == "cuda":
        device_name = torch.cuda.get_device_name(bench.device)
    meta = {
        "argv": sys.argv,
        "args": vars(args),
        "model_name": bench.model_name,
        "model_dtype": str(bench.dtype),
        "model_context_length": bench.model_context_length,
        "loader_context_length": bench.loader_context_length,
        "image_tokens": bench.image_tokens,
        "data_files": {str(k): v for k, v in bench.dataset_paths.items()},
        "device": str(bench.device),
        "device_name": device_name,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "python": platform.python_version(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as handle:
        json.dump(meta, handle, indent=2, default=str)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summarize_only:
        pattern = os.path.join(args.output_dir, "*_conversations.jsonl")
        conversation_paths = sorted(glob.glob(pattern))
        if not conversation_paths:
            raise SystemExit(f"No result files match {pattern}")
        turn_paths = [p.replace("_conversations.jsonl", "_turns.jsonl") for p in conversation_paths]
        write_summaries(
            conversation_paths,
            turn_paths,
            os.path.join(args.output_dir, "all_summary.csv"),
            os.path.join(args.output_dir, "all_summary_per_turn.csv"),
        )
        return

    paths = output_paths(args)
    bench = Benchmark(args)
    write_meta(paths["meta"], args, bench)
    print(
        f"model={bench.model_name} device={bench.device} dtype={bench.dtype} "
        f"image_tokens={bench.image_tokens} context={bench.model_context_length} "
        f"breakdown={'events' if bench.clock.use_events else bench.clock.enabled}",
        flush=True,
    )
    bench.run(paths)
    write_summaries([paths["conversations"]], [paths["turns"]], paths["summary"], paths["per_turn"])


if __name__ == "__main__":
    main()