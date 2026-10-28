"""Policy-only overhead of TurnCast-KV vs PyramidKV at LLaVA-1.5-7B sizes (no model).

Feeds synthetic attention/KV of the real shapes (32 layers, 32 heads, head dim
128, 576 image tokens) through both caches: one Turn-1 prefill compression,
N decoding steps, one turn-boundary compression.  Reports wall time of the
cache code only (the model forward is identical for both methods).
Usage: python turncast/bench_overhead.py [--device cuda]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache_pyramidkv import PyramidKVCache  # noqa: E402
from kv_cache_turncast import TurnCastKVCache, TurnCastScorer  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(cache, layers, heads, prompt, steps, device, dtype, turncast):
    g = torch.Generator(device="cpu").manual_seed(0)
    past = [[torch.randn(1, heads, prompt, 128, generator=g).to(device, dtype),
             torch.randn(1, heads, prompt, 128, generator=g).to(device, dtype)] for _ in range(layers)]
    if turncast:
        cache.register_question("q")
    times = {}
    observe = 0.0
    mask = torch.triu(torch.full((prompt, prompt), -1e4, device=device), 1)
    for layer in range(layers):
        att = torch.softmax(torch.randn(1, heads, prompt, prompt, device=device) + mask, -1).to(dtype)
        sync(device)
        t = time.perf_counter()
        cache.observe_layer(layer, att)
        sync(device)
        observe += time.perf_counter() - t
    t1 = time.perf_counter()
    past = cache(past, num_of_token=prompt, attentions=None)
    sync(device)
    t2 = time.perf_counter()
    times["prefill_observe_s"] = observe
    times["prefill_compress_s"] = t2 - t1
    if turncast:
        cache.begin_answer()
    n = prompt
    decode = 0.0
    for _ in range(steps):
        past = [[torch.cat([k, k[:, :, -1:]], 2), torch.cat([v, v[:, :, -1:]], 2)] for k, v in past]
        n += 1
        atts = tuple(torch.softmax(torch.randn(1, heads, 1, k.shape[2], device=device), -1).to(dtype)
                     for k, _ in past)
        sync(device)
        t = time.perf_counter()
        past = cache(past, num_of_token=n, attentions=atts)
        sync(device)
        decode += time.perf_counter() - t
    times["decode_per_step_ms"] = 1000 * decode / steps
    sync(device)
    t = time.perf_counter()
    past = cache.finalize_turn(past)
    sync(device)
    times["boundary_compress_s"] = time.perf_counter() - t
    times["kv_after_mean"] = sum(k.shape[2] for k, _ in past) / layers
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.9)
    args = p.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    layers, heads, prompt = 32, 32, 640
    scorer = TurnCastScorer(layers, heads)
    params = sum(x.numel() for x in scorer.parameters())
    pyramid = PyramidKVCache(layer_num=layers, ratio=args.ratio, exact_budget=True)
    turncast = TurnCastKVCache(layers, args.ratio, scorer=scorer, image_start=35, image_token_len=576)
    turncast.attach_model(torch.nn.Linear(1, 1).to(device))
    a = run(pyramid, layers, heads, prompt, args.steps, device, dtype, False)
    b = run(turncast, layers, heads, prompt, args.steps, device, dtype, True)
    print(f"device {device}, ratio {args.ratio}, forecaster parameters {params}")
    print(f"{'':24}{'PyramidKV':>12}{'TurnCast':>12}")
    for key in a:
        print(f"{key:24}{a[key]:12.4f}{b[key]:12.4f}")
    entries = turncast.scored_entries
    macs = (31 + 8) * 64 + 64 * 64 + 64
    print(f"forecaster: {entries} entries scored over 2 compressions, ~{2 * macs * entries / 1e9:.2f} GFLOPs "
          f"(one LLaVA-7B decoding step is ~14 GFLOPs)")


if __name__ == "__main__":
    main()
