"""Policy-only overhead of SeedKV vs PyramidKV at LLaVA-1.5-7B sizes (no model).

Feeds synthetic attention/KV of the real shapes (32 layers, 32 heads, head dim
128, 576 image tokens) through both caches for a two-turn conversation and
reports the wall time of the cache code only.

SeedKV's one extra model pass per later turn (regenerating the prefill KV) is
NOT in these numbers: it is a batched forward over the prefill, i.e. about the
Turn-1 prefill time that eval_efficiency.py reports as TTFT1.  Here the
regenerated KV is supplied by a stub, so "hydrate" is the bookkeeping alone.

Usage: python seedkv/bench_overhead.py [--device cuda] [--ratio 0.9]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache_pyramidkv import PyramidKVCache  # noqa: E402
from kv_cache_seedkv import SeedKVCache  # noqa: E402

DIM = 128


class StubSeedKV(SeedKVCache):
    """SeedKV whose regeneration returns a stored copy instead of running a model."""

    def _build_seed(self, past_key_values):
        self._stored = list(past_key_values)   # stands in for the model's regenerated KV
        self.seed_kind = "stub"
        self.charge = 72                     # LLaVA-1.5-7B: 576 x 1024 features / (2 x 4096) entries

    def _regenerate(self):
        self.rehydrations += 1
        return self._stored


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Clock:
    def __init__(self, device):
        self.device, self.times = device, {}

    def run(self, name, fn):
        sync(self.device)
        start = time.perf_counter()
        out = fn()
        sync(self.device)
        self.times.setdefault(name, []).append(time.perf_counter() - start)
        return out


def step(cache, past, n, device, dtype, clock, name):
    past = [(torch.cat([k, k[:, :, -1:]], 2), torch.cat([v, v[:, :, -1:]], 2)) for k, v in past]
    heads = past[0][0].shape[1]
    atts = tuple(torch.softmax(torch.randn(1, heads, 1, k.shape[2], device=device), -1).to(dtype) for k, _ in past)
    return clock.run(name, lambda: cache(past, num_of_token=n, attentions=atts))


def run(cache, layers, heads, prompt, question, answer, device, dtype):
    clock = Clock(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    past = [(torch.randn(1, heads, prompt, DIM, generator=g).to(device, dtype),
             torch.randn(1, heads, prompt, DIM, generator=g).to(device, dtype)) for _ in range(layers)]
    mask = torch.triu(torch.full((prompt, prompt), -1e4, device=device), 1)
    for layer in range(layers):
        att = torch.softmax(torch.randn(1, heads, prompt, prompt, device=device) + mask, -1).to(dtype)
        clock.run("prefill observe (all layers)", lambda: cache.observe_layer(layer, att))
    past = clock.run("prefill compress", lambda: cache(past, num_of_token=prompt, attentions=None))
    n = prompt
    if hasattr(cache, "begin_answer"):
        cache.begin_answer()
    for _ in range(answer):
        n += 1
        past = step(cache, past, n, device, dtype, clock, "decode step")
    past = clock.run("turn boundary", lambda: cache.finalize_turn(past))
    rest = sum(k.shape[2] for k, _ in past) / layers

    if hasattr(cache, "prepare_turn"):
        past = clock.run("hydrate (bookkeeping only)", lambda: cache.prepare_turn(past, n))
    for _ in range(question):
        n += 1
        past = step(cache, past, n, device, dtype, clock, "question step")
    if hasattr(cache, "begin_answer"):
        cache.begin_answer()
    n += 1
    past = step(cache, past, n, device, dtype, clock, "first answer step (SeedKV commits)")
    for _ in range(answer - 1):
        n += 1
        past = step(cache, past, n, device, dtype, clock, "decode step")
    past = clock.run("turn boundary", lambda: cache.finalize_turn(past))

    out = {}
    for name, values in clock.times.items():
        total = sum(values)
        out[name] = total if name.startswith("prefill observe") else total / len(values)
    return out, rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--question", type=int, default=32)
    parser.add_argument("--answer", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    layers, heads, prompt = 32, 32, 640
    pyramid = PyramidKVCache(layer_num=layers, ratio=args.ratio)
    seed = StubSeedKV(layers, args.ratio, image_start=35, image_token_len=576)
    a, rest_a = run(pyramid, layers, heads, prompt, args.question, args.answer, device, dtype)
    b, rest_b = run(seed, layers, heads, prompt, args.question, args.answer, device, dtype)
    print(f"device {device}, ratio {args.ratio}, {layers} layers x {heads} heads, prefill {prompt} tokens, "
          f"question {args.question}, answer {args.answer} tokens")
    print(f"{'cache code only, ms per event':40}{'PyramidKV':>12}{'SeedKV':>12}")
    for name in b:
        left = f"{1000 * a[name]:12.3f}" if name in a else f"{'-':>12}"
        print(f"{name:40}{left}{1000 * b[name]:12.3f}")
    print(f"{'entries per layer at rest (turn 1)':40}{rest_a:12.1f}{rest_b:12.1f}")
    def per_turn(times):
        return 1000 * (times["turn boundary"] + times.get("hydrate (bookkeeping only)", 0.0)
                       + args.question * times["question step"]
                       + times["first answer step (SeedKV commits)"]
                       + (args.answer - 1) * times["decode step"])

    per_turn_a, per_turn_b = per_turn(a), per_turn(b)
    print(f"{'cache code per later turn, total':40}{per_turn_a:12.1f}{per_turn_b:12.1f}")


if __name__ == "__main__":
    main()
