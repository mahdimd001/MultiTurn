"""
Compare cache methods on the stand-in (or any LLaVA checkpoint) through the
unchanged evaluators, on CPU.

    python seedkv/run_compare.py --standin /path/standin --metric ppl \
        --methods full pyramidkv seedkv --ratios 0.9 0.8 --splits shift same mixed \
        --samples 100 --out results_ppl.json

``--metric ppl`` runs eval_ppl_final.py (perplexity of the reference answers).
``--metric gen`` runs eval_rouge_final2.py (greedy generation) and also reports
the share of answers that equal the reference exactly.

Results are appended to --out after every run, and finished runs are skipped,
so an interrupted sweep can be restarted.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402

# name -> (evaluator --method, extra arguments)
METHODS = {
    "full": ("local_cache", []),            # run at ratio 0: nothing is evicted
    "local_cache": ("local_cache", []),
    "h2o": ("h2o", []),
    "elastic_cache": ("elastic_cache", []),
    "tgv_kv": ("tgv_kv", []),
    "meda": ("meda", []),
    "pyramidkv": ("pyramidkv", []),
    "atlaskv": ("atlaskv", []),
    "turncast_kv": ("turncast_kv", []),     # needs --turncast-scorer (see --scorer)
    "turn_conditioned_layerwise_merge": ("turn_conditioned_layerwise_merge", []),
    "turn_conditioned_layerwise_eviction": ("turn_conditioned_layerwise_eviction", []),
    "turn_conditioned_coverage": ("turn_conditioned_coverage", []),
    "seedkv": ("seedkv", []),
    "seedkv_nocharge": ("seedkv", ["--seedkv-no-seed-charge"]),
    "seedkv_uniform": ("seedkv", ["--seedkv-allocation", "uniform"]),
    "seedkv_w16": ("seedkv", ["--seedkv-window-size", "16"]),
    "seedkv_refresh": ("seedkv", ["--seedkv-refresh-every", "16"]),
    "seedkv_refresh8": ("seedkv", ["--seedkv-refresh-every", "8"]),
    # Same entries per layer as this repo's PyramidKVCache (which holds more than L * C).
    "seedkv_as_pyramidkv": ("seedkv", ["--seedkv-allocation", "pyramidkv"]),
    "seedkv_retain_only_as_pyramidkv": ("seedkv", ["--seedkv-allocation", "pyramidkv", "--seedkv-no-regenerate"]),
    # GhostKV needs --locators (a calibration file for this model).
    "ghostkv": ("ghostkv", []),
    "ghostkv_no_paging": ("ghostkv", ["--ghostkv-no-paging"]),
    # Same entries per layer as this repo's PyramidKVCache (which holds more than L * C).
    "ghostkv_as_pyramidkv": ("ghostkv", ["--seedkv-allocation", "pyramidkv"]),
    "ghostkv_no_paging_as_pyramidkv": ("ghostkv", ["--ghostkv-no-paging", "--seedkv-allocation", "pyramidkv"]),
    "ghostkv_nocharge": ("ghostkv", ["--seedkv-no-seed-charge"]),
    "ghostkv_answer_only": ("ghostkv", ["--ghostkv-fault-on", "answer"]),
    "ghostkv_auto_heads": ("ghostkv", ["--ghostkv-locators", "none"]),
    "ghostkv_ctx_prefix": ("ghostkv", ["--ghostkv-context", "prefix"]),
    "ghostkv_ctx_resident": ("ghostkv", ["--ghostkv-context", "resident"]),
    "ghostkv_ctx_none": ("ghostkv", ["--ghostkv-context", "none"]),
    "ghostkv_t15": ("ghostkv", ["--ghostkv-threshold", "0.15"]),
    "ghostkv_t70": ("ghostkv", ["--ghostkv-threshold", "0.7"]),
    "ghostkv_2heads": ("ghostkv", ["--ghostkv-n-locators", "2", "--ghostkv-ghost-share", "1"]),
    "ghostkv_4heads": ("ghostkv", ["--ghostkv-n-locators", "4", "--ghostkv-ghost-share", "1"]),
    # Ablation: SeedKV's code, scoring and exact budgets with regeneration off.
    "seedkv_retain_only": ("seedkv", ["--seedkv-no-regenerate"]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standin", required=True)
    parser.add_argument("--metric", choices=["ppl", "gen"], default="ppl")
    parser.add_argument("--methods", nargs="+", default=["full", "pyramidkv", "seedkv"])
    parser.add_argument("--ratios", nargs="+", default=["0.9"])
    parser.add_argument("--splits", nargs="+", default=["shift", "same", "mixed"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--scorer", default=None, help="TurnCast-KV scorer trained for this model")
    parser.add_argument("--locators", default=None, help="GhostKV calibration file for this model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None)
    args = parser.parse_args()
    harness.set_threads()
    work = args.work or os.path.splitext(args.out)[0] + "_runs"
    os.makedirs(work, exist_ok=True)

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as handle:
            results = json.load(handle)

    for split in args.splits:
        for name in args.methods:
            method, extra = METHODS[name]
            extra = list(extra)
            if method == "ghostkv" and "--ghostkv-locators" not in extra and "--ghostkv-no-paging" not in extra:
                if not args.locators:
                    print(f"skip {name}: no --locators")
                    continue
                extra += ["--ghostkv-locators", args.locators]
            if method == "turncast_kv":
                if not args.scorer:
                    print(f"skip {name}: no --scorer")
                    continue
                extra += ["--turncast-scorer", args.scorer]
            for ratio in (["0.0"] if name == "full" else args.ratios):
                key = f"{args.metric}|{split}|{name}|{ratio}"
                if key in results:
                    continue
                run_name = key.replace("|", "_")
                start = time.time()
                try:
                    if args.metric == "ppl":
                        out = harness.run_ppl(args.standin, split, method, ratio, args.samples, work, run_name, extra)
                        row = {"ppl": out["overall_ppl"], "per_turn": out["per_turn_ppl"]}
                    else:
                        out = harness.run_rouge(args.standin, split, method, ratio, args.samples, work, run_name,
                                                extra, max_tokens=args.max_tokens)
                        accuracy, per_turn = harness.exact_match(out)
                        row = {"exact": accuracy, "exact_per_turn": per_turn,
                               "rouge": out["avg_rouge_l_f"], "rouge_per_turn": out["per_turn_rouge_l_f"]}
                except Exception as exc:  # a baseline that cannot run on this model
                    row = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                row["seconds"] = round(time.time() - start, 1)
                results[key] = row
                with open(args.out, "w") as handle:
                    json.dump(results, handle, indent=1)
                print(key, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
