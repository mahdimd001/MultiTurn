"""What does each scoring rule keep?  Share of kept entries per segment and turn age,
at each compression point (toy diagnostics; same harness as diagnose.py)."""
import argparse
import collections
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import diagnose  # noqa: E402
from run_cpu import patched_module  # noqa: E402
from kv_cache_turncast import SEGMENTS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--ratio", default="0.9")
    p.add_argument("--modes", nargs="+", default=["forecast", "window", "oracle"])
    p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    evaluator = patched_module("eval_ppl_final.py")
    labels = {}
    if any(m.startswith("oracle") for m in args.modes):
        labels = {"mean": diagnose.oracle_labels(evaluator, args, "mean")}
    for mode in args.modes:
        caches = []
        original = evaluator.build_cache

        def build(run_args, *a, _mode=mode, **k):
            cache = original(run_args, *a, **k)
            cache.record = True
            if _mode == "window":
                cache.score_override = lambda c, st, f, s: diagnose.window_scores(s)
            elif _mode == "oracle":
                cache.score_override = diagnose.make_oracle(labels["mean"])
            caches.append(cache)
            return cache

        evaluator.build_cache = build
        original_load = evaluator.load_image

        def load_image(path, _orig=original_load):
            diagnose.current["image"] = os.path.basename(path)
            diagnose.current["boundary"] = 0
            return _orig(path)

        evaluator.load_image = load_image
        sys.argv = ["eval_ppl_final.py", "--model-path", args.model_path, "--data-path", args.data_path,
                    "--image-path", args.image_path, "--device", "cpu", "--method", "turncast_kv",
                    "--ratio", args.ratio, "--turncast-scorer", args.scorer, "--output-dir", "/tmp/turncast_comp",
                    "--exp-name", mode, "--eval-samples", str(args.samples)]
        evaluator.main(evaluator.parse_cli())
        evaluator.build_cache, evaluator.load_image = original, original_load
        shares = collections.defaultdict(lambda: collections.Counter())
        for cache in caches:
            seg = torch.tensor(cache.seg)
            turn = torch.tensor(cache.turn)
            for b, record in enumerate(cache.records[:-1]):
                kept = record.get("kept", {})
                for layer, pos in kept.items():
                    key = f"point {b} ({record['stage']}, turn {record['turn']})"
                    for p_ in pos.reshape(-1).tolist():
                        name = SEGMENTS[int(seg[p_])]
                        age = record["turn"] - int(turn[p_])
                        label = name if name in ("sink", "prompt_text", "image") else f"{name}[-{age}]"
                        shares[key][label] += 1
        print(f"\n== {mode} @ {args.ratio}: share of kept entries")
        for key in sorted(shares):
            total = sum(shares[key].values())
            parts = ", ".join(f"{k} {v / total:.2f}" for k, v in sorted(shares[key].items()))
            print(f"  {key}: {parts}")


if __name__ == "__main__":
    main()
