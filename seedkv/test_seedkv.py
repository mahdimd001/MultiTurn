"""
Correctness checks for SeedKV (CPU, any small LLaVA checkpoint; see make_standin.py).

Everything runs through the unchanged evaluators (eval_ppl_final.py and
eval_rouge_final2.py with the CPU shims of turncast/run_cpu.py).

 1. Exact regeneration   the KV rebuilt from the seed equals the original prefill KV.
 2. Small seed           vision features + token ids; its size and budget charge.
 3. Budget               KV + seed never exceeds L * (C + appended tokens) except
                         while a later turn's question is being read; at rest
                         no prefill entry is resident.
 4. Layout               every head keeps sorted, unique positions; the same
                         count in every head; the attention sink is kept
                         whenever prefill entries are resident.
 5. No-op at ratio 0     identical perplexity to the full cache.
 6. Cost                 same number of evaluator forwards as PyramidKV, plus
                         exactly one regeneration per later turn.
 7. Fallbacks            without prepare_turn() / without begin_answer().
 8. Generation           eval_rouge_final2.py runs; refresh_every works.
 9. Options              the retain-only ablation and the PyramidKVCache-matched schedule.
10. Determinism          two runs give the same perplexity.

Usage: python seedkv/test_seedkv.py --standin /path/standin [--samples 6]
"""
import argparse
import json
import os
import sys
import tempfile
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402
from harness import Recorder, base_argv, run, run_ppl, run_rouge  # noqa: E402

sys.path.insert(0, harness.REPO)
from kv_cache_seedkv import SeedKVCache, layer_budgets  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def ppl_argv(args, method, ratio, name, extra=()):
    return base_argv(args.standin, args.split, method, ratio, args.samples, args.work, name) + list(extra)


def nominal_budgets(events, ratio, layers):
    """L * (C(base) + appended) for every event, C(n) = int(n * (1 - ratio))."""
    out, base, logical = [], None, 0
    for event in events:
        if event["kind"] == "call":
            logical = event["logical"]
            if base is None or event["first"]:
                base = logical
        elif event["kind"] == "finalize":
            base = logical
        out.append(layers * (int(base * (1.0 - ratio)) + (logical - base)))
    return out


def mark_first_calls(events):
    seen = set()
    for event in events:
        event["first"] = event["kind"] == "call" and event["cache"] not in seen
        if event["kind"] == "call":
            seen.add(event["cache"])
    return events


def test_budget_schedule():
    for allocation in ("pyramid", "uniform"):
        for total, length in ((8 * 19 - 5, 186), (8 * 62 - 19, 620), (8 * 113, 1136), (8 * 7, 40)):
            budgets = layer_budgets(8, total, min(8, total // 8 - 1), [length] * 8, allocation)
            check(f"budget schedule {allocation} total={total}",
                  sum(budgets) == total and all(a >= b for a, b in zip(budgets, budgets[1:])),
                  f"{budgets}")


def test_structure(args, ratio):
    """Tests 1-4 and 6 from one instrumented run."""
    stash, problems, layout_checks = {}, [], [0]

    def inspect(kind, cache, before, after):
        if not isinstance(cache, SeedKVCache):
            return
        if kind == "call" and id(cache) not in stash:
            stash[id(cache)] = [(k.clone(), v.clone()) for k, v in before]      # the full prefill
        for layer, (key, _) in enumerate(after):
            pos = cache._full_pos(layer, key.shape[2], key.shape[1])
            layout_checks[0] += 1
            if pos.shape[1] != key.shape[2]:
                problems.append(f"{kind}: layer {layer} bookkeeping {pos.shape[1]} != KV {key.shape[2]}")
            if pos.shape[1] > 1 and not bool((pos[:, 1:] > pos[:, :-1]).all()):
                problems.append(f"{kind}: layer {layer} positions not sorted/unique")
            a_count = cache._a_count[layer]
            if a_count and not bool((pos[:, 0] == 0).all()):
                problems.append(f"{kind}: layer {layer} lost the attention sink")
            if a_count and not bool((pos[:, :a_count] < cache.n0).all()):
                problems.append(f"{kind}: layer {layer} prefill entries not at the front")
            if bool((pos[:, a_count:] < cache.n0).any()):
                problems.append(f"{kind}: layer {layer} has prefill entries outside the front block")
            if cache.state == "rest" and a_count:
                problems.append(f"{kind}: layer {layer} holds prefill entries at rest")

    with Recorder("eval_ppl_final.py", inspect=inspect) as seed:
        run("eval_ppl_final.py", ppl_argv(args, "seedkv", ratio, f"t_seed_{ratio}"))
    with Recorder("eval_ppl_final.py") as pyramid:
        run("eval_ppl_final.py", ppl_argv(args, "pyramidkv", ratio, f"t_pyr_{ratio}"))

    regenerations = sum(c.rehydrations for c in seed.caches)

    # 1. exact regeneration, 2. seed
    worst = 0.0
    for cache in seed.caches:
        full = stash[id(cache)]
        for (key, value), (ref_key, ref_value) in zip(cache._regenerate(), full):
            worst = max(worst, float((key - ref_key).abs().max()), float((value - ref_value).abs().max()))
    check(f"ratio {ratio}: regenerated prefill KV equals the original", worst == 0.0, f"max |diff| = {worst:.2e}")
    cache = seed.caches[0]
    full_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in stash[id(cache)])
    check(f"ratio {ratio}: seed is features + token ids", cache.seed_kind == "features+ids",
          f"{cache.seed_kind}, {cache.seed_bytes} bytes = {100.0 * cache.seed_bytes / full_bytes:.2f}% of the "
          f"prefill KV, charged as {cache.charge} KV entries = {cache.charge / cache.layer_num:.2f} per layer")

    # 3. budget
    layers = len(seed.events[0]["sizes"])
    events = mark_first_calls([e for e in seed.events if e["kind"] != "prepare"])
    nominal = nominal_budgets(events, float(ratio), layers)
    over, hydrated_tokens, peak = 0, 0, 0.0
    for event, budget in zip(events, nominal):
        total = sum(event["sizes"]) + seed.caches[event["cache"]].charge
        if event["state"] == "hydrated":
            hydrated_tokens += 1
            peak = max(peak, total / budget)
        elif total > budget:
            over += 1
    turns = sum(1 for e in events if e["kind"] == "finalize")
    check(f"ratio {ratio}: KV + seed within L*(C + appended) outside question reading", over == 0,
          f"{len(events)} cache calls checked, {over} over budget")
    print(f"       question reading: {hydrated_tokens / max(1, turns - len(seed.caches)):.1f} tokens per later turn "
          f"above budget, peak {peak:.1f}x the budget")
    pyr_events = mark_first_calls(pyramid.events)
    same = len(pyr_events) == len(events)
    if same:
        # Informational: PyramidKVCache rounds its pyramid down and cannot refill a
        # layer it shrank earlier, so its realised size can sit below the budget.
        diff = [(sum(a["sizes"]) + seed.caches[a["cache"]].charge - sum(b["sizes"])) / layers
                for a, b in zip(events, pyr_events) if a["state"] != "hydrated"]
        print(f"       SeedKV(+seed) minus PyramidKV, entries per layer outside question reading: "
              f"mean {sum(diff) / len(diff):+.1f}, max {max(diff):+.1f}")
    rest = [e for e in events if e["kind"] == "finalize"]
    rest_pyr = [e for e in pyr_events if e["kind"] == "finalize"]
    print(f"       at rest, entries per layer: SeedKV {sum(sum(e['sizes']) for e in rest) / len(rest) / layers:.1f} "
          f"(+{seed.caches[0].charge / layers:.1f} seed), PyramidKV "
          f"{sum(sum(e['sizes']) for e in rest_pyr) / len(rest_pyr) / layers:.1f}")

    # 4. layout
    check(f"ratio {ratio}: per-head layout", not problems,
          f"{layout_checks[0]} layer states checked" + (f"; first problem: {problems[0]}" if problems else ""))

    # 6. cost
    later_turns = turns - len(seed.caches)
    check(f"ratio {ratio}: evaluator forwards equal PyramidKV's", seed.forwards == pyramid.forwards,
          f"{seed.forwards} vs {pyramid.forwards}")
    check(f"ratio {ratio}: one regeneration per later turn", regenerations == later_turns,
          f"{regenerations} regenerations for {later_turns} later turns")


def test_noop(args):
    full = run_ppl(args.standin, args.split, "local_cache", 0.0, args.samples, args.work, "t_full")
    seed = run_ppl(args.standin, args.split, "seedkv", 0.0, args.samples, args.work, "t_seed_0")
    check("ratio 0: identical to the full cache", full["overall_ppl"] == seed["overall_ppl"],
          f"{seed['overall_ppl']:.6f} vs {full['overall_ppl']:.6f}")
    return full["overall_ppl"]


def test_fallbacks(args, ratio):
    reference = run_ppl(args.standin, args.split, "seedkv", ratio, args.samples, args.work, "t_ref")

    def no_prepare(cache):
        if isinstance(cache, SeedKVCache):
            cache.prepare_turn = lambda past, num_of_token=None: past

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Recorder("eval_ppl_final.py", on_build=no_prepare) as lazy:
            run("eval_ppl_final.py", ppl_argv(args, "seedkv", ratio, "t_lazy"))
    turns = sum(1 for e in lazy.events if e["kind"] == "finalize")
    regenerations = sum(c.rehydrations for c in lazy.caches)
    with open(os.path.join(args.work, "t_lazy.json")) as handle:
        lazy_ppl = json.load(handle)["overall_ppl"]
    check("fallback without prepare_turn(): regenerates inside the first call of the turn",
          regenerations == turns - len(lazy.caches),
          f"PPL {lazy_ppl:.4f} vs {reference['overall_ppl']:.4f} with prepare_turn()")

    def no_begin(cache):
        if isinstance(cache, SeedKVCache):
            cache.begin_answer = lambda: None
            cache.max_hydrated_tokens = 4

    with Recorder("eval_ppl_final.py", on_build=no_begin) as capped:
        run("eval_ppl_final.py", ppl_argv(args, "seedkv", ratio, "t_capped"))
    longest, current = 0, 0
    for event in capped.events:
        current = current + 1 if event["state"] == "hydrated" and event["kind"] == "call" else 0
        longest = max(longest, current)
    check("fallback without begin_answer(): commits after max_hydrated_tokens", longest == 3,
          f"longest uncompressed stretch {longest} cache calls with max_hydrated_tokens=4")


def test_generation(args, ratio):
    result = run_rouge(args.standin, args.split, "seedkv", ratio, args.samples, args.work, "t_rouge")
    check("eval_rouge_final2.py runs with SeedKV", result["num_turns_scored"] > 0,
          f"{result['num_turns_scored']} turns, ROUGE-L {result['avg_rouge_l_f']:.4f}")

    with Recorder("eval_rouge_final2.py") as refresh:
        run("eval_rouge_final2.py",
            ppl_argv(args, "seedkv", ratio, "t_refresh", ["--seedkv-refresh-every", 4, "--max-token-generated", 24]))
    turns = sum(1 for e in refresh.events if e["kind"] == "finalize")
    regenerations = sum(c.rehydrations for c in refresh.caches)
    check("refresh_every: extra regenerations during the answer",
          regenerations > turns - len(refresh.caches), f"{regenerations} regenerations in {turns} turns")


def test_options(args, ratio):
    """Retain-only ablation and the PyramidKVCache-matched schedule."""
    with Recorder("eval_ppl_final.py") as retain:
        run("eval_ppl_final.py", ppl_argv(args, "seedkv", ratio, "t_retain", ["--seedkv-no-regenerate"]))
    check("ablation --seedkv-no-regenerate: no regeneration, no seed",
          sum(c.rehydrations for c in retain.caches) == 0 and all(c.charge == 0 for c in retain.caches)
          and all(e["state"] == "compressed" for e in retain.events))
    events = mark_first_calls(retain.events)
    layers = len(events[0]["sizes"])
    over = sum(1 for e, b in zip(events, nominal_budgets(events, float(ratio), layers)) if sum(e["sizes"]) > b)
    check("ablation --seedkv-no-regenerate: within L*(C + appended)", over == 0, f"{len(events)} cache calls")

    with Recorder("eval_ppl_final.py") as matched:
        run("eval_ppl_final.py", ppl_argv(args, "seedkv", ratio, "t_matched",
                                          ["--seedkv-allocation", "pyramidkv", "--seedkv-no-regenerate",
                                           "--seedkv-no-seed-charge"]))
    with Recorder("eval_ppl_final.py") as pyramid:
        run("eval_ppl_final.py", ppl_argv(args, "pyramidkv", ratio, "t_pyr_matched"))
    first_a = [e["sizes"] for e in mark_first_calls(matched.events) if e["first"]]
    first_b = [e["sizes"] for e in mark_first_calls(pyramid.events) if e["first"]]
    check("--seedkv-allocation pyramidkv: same entries per layer as PyramidKVCache after the prefill",
          first_a == first_b, f"{first_a[0]} vs {first_b[0]}")


def test_determinism(args, ratio):
    first = run_ppl(args.standin, args.split, "seedkv", ratio, args.samples, args.work, "t_det1")
    second = run_ppl(args.standin, args.split, "seedkv", ratio, args.samples, args.work, "t_det2")
    check("deterministic", first["overall_ppl"] == second["overall_ppl"], f"PPL {first['overall_ppl']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standin", required=True)
    parser.add_argument("--split", default="mixed")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--work", default=None)
    args = parser.parse_args()
    args.work = args.work or tempfile.mkdtemp(prefix="seedkv_test_")
    harness.set_threads()
    torch.manual_seed(0)

    test_budget_schedule()
    test_noop(args)
    for ratio in (0.9, 0.5):
        test_structure(args, ratio)
    test_fallbacks(args, 0.9)
    test_generation(args, 0.9)
    test_options(args, 0.9)
    test_determinism(args, 0.9)

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
