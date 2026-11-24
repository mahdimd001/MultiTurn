"""
Correctness checks for GhostKV (CPU, any small LLaVA checkpoint; see make_standin.py).

Everything runs through the unchanged evaluators (eval_ppl_final.py and
eval_rouge_final2.py with the CPU shims of turncast/run_cpu.py).

 1. Exact partial regeneration
        a. all patches, text prefix as context     -> equals the original image KV
        b. a subset, all other patches as context  -> equals the original KV of the subset
        (the method's approximation is only what is missing from the context)
 2. Budget          KV + seed + ghost keys <= L * (C + appended tokens) at EVERY
                    cache call: a page fault never grows the cache.
 3. Layout          bookkeeping matches the KV; a head never holds a position
                    twice; the attention sink stays; an entry that was not paged
                    in holds exactly the original key of its position.
 4. Paging works    the token that faults reads the paged-in patches in the same
                    forward pass (real attention on them at the faulting layer).
 5. No-op at ratio 0, and no regeneration with paging off.
 6. Cost            same evaluator forwards as PyramidKV; faults and regenerated
                    tokens per turn (SeedKV regenerates the whole prefill).
 7. Calibration     runs with the full cache, does not change the model's output,
                    writes the locator file.
 8. Relevance map   (stand-in only: the data says where the asked object is)
                    the map's top patches are the object's patches.
 9. Generation      eval_rouge_final2.py runs.
10. Determinism     two runs give the same perplexity.

Usage: python seedkv/test_ghostkv.py --standin /path/standin [--locators FILE] [--samples 6]
"""
import argparse
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402
from harness import Recorder, base_argv, run, run_ppl, run_rouge  # noqa: E402

sys.path.insert(0, harness.REPO)
from kv_cache_ghostkv import GhostKVCache  # noqa: E402
from test_seedkv import mark_first_calls, nominal_budgets  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def ppl_argv(args, method, ratio, name, extra=()):
    return base_argv(args.standin, args.split, method, ratio, args.samples, args.work, name) + list(extra)


def ghost_extra(args, extra=()):
    return (["--ghostkv-locators", args.locators] if args.locators else []) + list(extra)


def load_ppl(args, name):
    with open(os.path.join(args.work, name + ".json")) as handle:
        return json.load(handle)["overall_ppl"]


def max_diff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def test_structure(args, ratio):
    """Tests 1-4, 6 and 8 from one instrumented run."""
    stash, problems, counts = {}, [], {"layout": 0, "native": 0, "paged": 0}
    paged_error, read_now, grow = [], [], []
    last_sizes = {}

    def inspect(kind, cache, before, after):
        if not isinstance(cache, GhostKVCache):
            return
        if kind == "call" and id(cache) not in stash:
            stash[id(cache)] = [(k.clone(), v.clone()) for k, v in before]      # the full prefill
            last_sizes[id(cache)] = None
        full = stash[id(cache)]
        sizes = [int(k.shape[2]) for k, _ in after]
        previous = last_sizes.get(id(cache))
        if kind == "call" and previous is not None:
            grow.append(max(a - b for a, b in zip(sizes, previous)))
        last_sizes[id(cache)] = sizes
        ever = sorted({patch for fault in cache.fault_log for patch in fault["patches"]})
        ever = torch.as_tensor(ever, dtype=torch.long) + cache.image_start
        for layer, (key, _) in enumerate(after):
            pos = cache._full_pos(layer, key.shape[2], key.shape[1])
            counts["layout"] += 1
            if pos.shape[1] != key.shape[2]:
                problems.append(f"{kind}: layer {layer} bookkeeping {pos.shape[1]} != KV {key.shape[2]}")
                continue
            ordered = pos.sort(dim=1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                problems.append(f"{kind}: layer {layer} a head holds a position twice")
            if not bool((pos == 0).any(dim=1).all()):
                problems.append(f"{kind}: layer {layer} lost the attention sink")
            # Content: an entry at a prefill position holds that position's key
            # (exactly, unless the patch was paged in at some point).
            prefill = pos < cache.n0
            original = full[layer][0][0].gather(
                1, pos.clamp(max=cache.n0 - 1)[:, :, None].expand(-1, -1, key.shape[3]))
            error = (key[0] - original).norm(dim=-1) / original.norm(dim=-1).clamp_min(1e-6)
            was_paged = prefill & torch.isin(pos, ever)
            native = prefill & ~was_paged
            counts["native"] += int(native.sum())
            if bool((error[native] > 1e-5).any()):
                problems.append(f"{kind}: layer {layer} an entry does not hold the key of its position")
            if was_paged.any():
                counts["paged"] += int(was_paged.sum())
                paged_error.append(float(error[was_paged].mean()))
        # 4. the faulting token reads what it paged in
        if kind == "call" and cache.fault_log and cache.fault_log[-1]["token"] == cache.n_logical - 1 \
                and not cache.fault_log[-1].get("seen"):
            fault = cache.fault_log[-1]
            fault["seen"] = True
            layer = fault["layer"]
            row = cache._window[layer][-1]                                   # [heads, 1, keys]
            pos = cache._pos[layer]
            groups = row.shape[0] // pos.shape[0]
            wanted = torch.as_tensor(fault["patches"]) + cache.image_start
            best = 0.0
            for head in cache._ghost[layer]["heads"].tolist():
                cols = torch.isin(pos[head], wanted).nonzero().flatten()
                best = max(best, float(row[head, 0, cols].sum()))
            read_now.append((best, fault["predicted"]))

    with Recorder("eval_ppl_final.py", inspect=inspect) as ghost:
        run("eval_ppl_final.py", ppl_argv(args, "ghostkv", ratio, f"g_{ratio}", ghost_extra(args)))
    with Recorder("eval_ppl_final.py") as pyramid:
        run("eval_ppl_final.py", ppl_argv(args, "pyramidkv", ratio, f"g_pyr_{ratio}"))

    # 1. exact partial regeneration
    cache = ghost.caches[-1]
    full = stash[id(cache)]
    patches = list(range(cache.image_len))
    image = slice(cache.image_start, cache.image_end)
    fresh = cache._regenerate_patches(patches, None)
    worst = max(max_diff((k, v), (fk[:, :, image], fv[:, :, image])) for (k, v), (fk, fv) in zip(fresh, full))
    check(f"ratio {ratio}: all patches regenerated after the text prefix equal the original image KV",
          worst < 1e-4, f"max |diff| = {worst:.2e}")
    subset = patches[5::7]
    rest = torch.as_tensor([p for p in patches if p not in set(subset)]) + cache.image_start
    context = []
    for key, value in full:
        heads = key.shape[1]
        positions = rest.view(1, -1).expand(heads, -1)
        context.append((key[:, :, rest], value[:, :, rest], positions, torch.ones_like(positions, dtype=torch.bool)))
    fresh = cache._regenerate_patches(subset, context)
    index = torch.as_tensor(subset) + cache.image_start
    worst = max(max_diff((k, v), (fk[:, :, index], fv[:, :, index])) for (k, v), (fk, fv) in zip(fresh, full))
    check(f"ratio {ratio}: a subset regenerated with the other patches as context equals the original",
          worst < 1e-4, f"{len(subset)} of {len(patches)} patches, max |diff| = {worst:.2e}")

    # 2. budget, at every call
    layers = cache.layer_num
    events = mark_first_calls([e for e in ghost.events if e["kind"] != "prepare"])
    nominal = nominal_budgets(events, float(ratio), layers)
    over = sum(1 for event, budget in zip(events, nominal)
               if sum(event["sizes"]) + ghost.caches[event["cache"]].charge > budget)
    worst_fill = max((sum(e["sizes"]) + ghost.caches[e["cache"]].charge) / b for e, b in zip(events, nominal))
    check(f"ratio {ratio}: KV + seed + ghost keys within L*(C + appended) at every cache call", over == 0,
          f"{len(events)} calls, {over} over budget, fullest {100 * worst_fill:.0f}% of the budget")
    check(f"ratio {ratio}: a page fault never grows the cache", max(grow) <= 1,
          f"largest growth per layer in one step: {max(grow)} entry (the new token)")
    print(f"       overhead: seed {cache.seed_bytes} bytes + ghost keys {cache.ghost_bytes} bytes of "
          f"{len(cache.locators)} head(s) = {cache.charge} KV entries = {cache.charge / layers:.2f} per layer; "
          f"up to {cache.page_slots} patches per fault")

    # 3. layout
    check(f"ratio {ratio}: per-head layout", not problems,
          f"{counts['layout']} layer states, {counts['native']} entries compared with the original keys"
          + (f"; first problem: {problems[0]}" if problems else ""))
    if paged_error:
        print(f"       keys of patches that were paged in vs their original keys: mean relative error "
              f"{sum(paged_error) / len(paged_error):.3f} ({counts['paged']} entries)")

    # 4. paging
    faults = sum(c.faults for c in ghost.caches)
    turns = sum(1 for e in events if e["kind"] == "finalize")
    check(f"ratio {ratio}: page faults happen", faults > 0, f"{faults} faults in {turns} turns")
    if read_now:
        ratio_read = sum(a for a, _ in read_now) / max(1e-9, sum(p for _, p in read_now))
        check(f"ratio {ratio}: the faulting token reads the paged-in patches in the same forward pass",
              ratio_read > 0.5,
              f"real attention on them at the faulting layer = {100 * ratio_read:.0f}% of the predicted "
              f"(virtual) attention, {len(read_now)} faults")

    # 6. cost
    check(f"ratio {ratio}: evaluator forwards equal PyramidKV's", ghost.forwards == pyramid.forwards,
          f"{ghost.forwards} vs {pyramid.forwards}")
    tokens = sum(c.regen_tokens for c in ghost.caches)
    patches_regenerated = sum(len(f["patches"]) for c in ghost.caches for f in c.fault_log)
    print(f"       cost per turn: {faults / turns:.2f} faults, {tokens / turns:.1f} tokens through the model "
          f"({patches_regenerated / turns:.1f} of them image patches); "
          f"SeedKV: {cache.n0} tokens per later turn")

    # 8. relevance map against the stand-in's ground truth
    with open(harness.standin_paths(args.standin, args.split)["data"]) as handle:
        items = json.load(handle)[: args.samples]
    side = int(round(cache.image_len ** 0.5))
    hits = {"relevance_answer": [], "peak": []}
    for item_cache, item in zip(ghost.caches, items):
        for turn, entry in zip(item["turns"], item_cache.turn_log):
            if "target" not in turn:
                continue
            row, col = turn["target"]
            cells = {(row + a) * side + col + b for a in (0, 1) for b in (0, 1)}
            for name, values in hits.items():
                if entry.get(name) is not None:
                    values.append(len(set(entry[name].topk(4).indices.tolist()) & cells) / 4.0)
    if hits["peak"]:
        mean = {name: sum(values) / max(1, len(values)) for name, values in hits.items()}
        check(f"ratio {ratio}: the relevance map points at the asked object", mean["peak"] > 0.6,
              f"top-4 patches inside the object: sharpest answer token {mean['peak']:.2f}, "
              f"all checked answer tokens {mean['relevance_answer']:.2f}; "
              f"chance {4.0 / cache.image_len:.2f}; {len(hits['peak'])} turns")


def test_noop(args):
    full = run_ppl(args.standin, args.split, "local_cache", 0.0, args.samples, args.work, "g_full")
    ghost = run_ppl(args.standin, args.split, "ghostkv", 0.0, args.samples, args.work, "g_0", ghost_extra(args))
    check("ratio 0: identical to the full cache", full["overall_ppl"] == ghost["overall_ppl"],
          f"{ghost['overall_ppl']:.6f} vs {full['overall_ppl']:.6f}")
    return full["overall_ppl"]


def test_paging_off(args, ratio):
    with Recorder("eval_ppl_final.py") as off:
        run("eval_ppl_final.py", ppl_argv(args, "ghostkv", ratio, "g_off", ["--ghostkv-no-paging"]))
    faults = sum(c.faults for c in off.caches)
    check(f"ratio {ratio}: no regeneration with --ghostkv-no-paging", faults == 0, f"{faults} faults")
    return load_ppl(args, "g_off")


def test_calibration(args, full_ppl):
    path = os.path.join(args.work, "locators_test.pt")
    run("eval_ppl_final.py", ppl_argv(args, "ghostkv", 0.9, "g_calib", ["--ghostkv-calibrate", path]))
    check("calibration: the model's output is unchanged (full cache)", load_ppl(args, "g_calib") == full_ppl,
          f"{load_ppl(args, 'g_calib'):.6f} vs {full_ppl:.6f}")
    ok = os.path.exists(path)
    detail = "no file"
    if ok:
        blob = torch.load(path)
        score = blob["score"]
        best = int(score.flatten().argmax())
        detail = (f"{blob['conversations']} conversations, {blob['steps']} answer tokens; best head "
                  f"(layer {best // score.shape[1]}, head {best % score.shape[1]})")
        ok = score.shape[0] > 0 and "top_mass_quantiles" in blob and bool(torch.isfinite(score).all())
    check("calibration: locator file written", ok, detail)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standin", required=True)
    parser.add_argument("--locators", default=None, help="calibration file; made on the fly if missing")
    parser.add_argument("--split", default="shift")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--ratios", nargs="+", default=["0.9", "0.5"])
    parser.add_argument("--work", default=None)
    args = parser.parse_args()
    harness.set_threads()
    args.work = args.work or tempfile.mkdtemp(prefix="ghostkv_test_")

    full_ppl = test_noop(args)
    made = test_calibration(args, full_ppl)
    if not args.locators:
        args.locators = made
        print(f"       using the locator file made by this test ({args.samples} conversations only)")

    for ratio in args.ratios:
        test_structure(args, ratio)

    ratio = args.ratios[0]
    off = test_paging_off(args, ratio)
    first = run_ppl(args.standin, args.split, "ghostkv", ratio, args.samples, args.work, "g_a", ghost_extra(args))
    second = run_ppl(args.standin, args.split, "ghostkv", ratio, args.samples, args.work, "g_b", ghost_extra(args))
    check(f"ratio {ratio}: deterministic", first["overall_ppl"] == second["overall_ppl"],
          f"{first['overall_ppl']:.6f}")
    pyramid = load_ppl(args, f"g_pyr_{ratio}")
    print(f"       perplexity at ratio {ratio} ({args.samples} conversations): full {full_ppl:.3f}, "
          f"GhostKV {first['overall_ppl']:.3f}, "
          f"paging off (no seed, no ghost keys) {off:.3f}, PyramidKV {pyramid:.3f}")

    out = run_rouge(args.standin, args.split, "ghostkv", ratio, args.samples, args.work, "g_gen",
                    ghost_extra(args), max_tokens=40)
    accuracy, _ = harness.exact_match(out)
    check(f"ratio {ratio}: eval_rouge_final2.py runs", len(out["results"]) == args.samples,
          f"exact answers {accuracy:.2f}, ROUGE-L {out['avg_rouge_l_f']:.3f}")
    print()
    print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
