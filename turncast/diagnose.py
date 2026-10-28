"""
Diagnose TurnCast-KV end to end (PPL of reference answers, per turn).

Runs the unchanged eval_ppl_final.py (CPU shims from run_cpu.py) with the
TurnCast cache and swaps the scoring rule:

  turncast        the method: forecaster at the Turn-1 compression, PyramidKV's
                  window rule at turn boundaries (boundary="window")
  forecast-all    forecaster at every compression (boundary="forecast")
  window          PyramidKV's rule everywhere, same exact budgets (isolates the scorer)
  oracle          the TRUE future attention of the conversation (full-cache pass)
                  at every compression: is "future attention" the right target?
  oracle-max      same, peak instead of mean attention per later turn
  oracle-val      same, attention x value norm

Usage: python turncast/diagnose.py --model-path M --data-path heldout.json --image-path I \
           --scorer scorer.pt --ratios 0.9 0.8 --modes turncast forecast-all window oracle
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import collect  # noqa: E402
from run_cpu import patched_module  # noqa: E402
from kv_cache_turncast import window_scores  # noqa: E402

current = {"image": None, "boundary": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--ratios", nargs="+", default=["0.9"])
    p.add_argument("--modes", nargs="+", default=["turncast", "forecast-all", "window", "oracle"])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--out", default="/tmp/turncast_diag")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))

    evaluator = patched_module("eval_ppl_final.py")
    labels = {}
    for mode in args.modes:
        if mode.startswith("oracle"):
            agg = mode.split("-")[1] if "-" in mode else "mean"
            labels[mode] = oracle_labels(evaluator, args, agg)

    results = {}
    for ratio in args.ratios:
        for mode in args.modes:
            original = evaluator.build_cache

            def build(run_args, *a, _mode=mode, **k):
                cache = original(run_args, *a, **k)
                if _mode == "forecast-all":
                    cache.boundary = "forecast"
                elif _mode == "window":
                    cache.score_override = lambda c, stage, f, s: window_scores(s)
                elif _mode.startswith("oracle"):
                    cache.score_override = make_oracle(labels[_mode])
                return cache

            evaluator.build_cache = build
            original_load = evaluator.load_image

            def load_image(path, _orig=original_load):
                current["image"] = os.path.basename(path)
                current["boundary"] = 0
                return _orig(path)

            evaluator.load_image = load_image
            sys.argv = ["eval_ppl_final.py", "--model-path", args.model_path, "--data-path", args.data_path,
                        "--image-path", args.image_path, "--device", "cpu", "--method", "turncast_kv",
                        "--ratio", ratio, "--turncast-scorer", args.scorer, "--output-dir", args.out,
                        "--exp-name", f"{mode}-{ratio}", "--eval-samples", str(args.samples)]
            evaluator.main(evaluator.parse_cli())
            evaluator.build_cache = original
            evaluator.load_image = original_load
            results[f"{mode}@{ratio}"] = json.load(open(os.path.join(args.out, f"{mode}-{ratio}.json")))
    print("\nPPL of reference answers (lower is better)")
    for key, res in results.items():
        turns = " ".join(f"T{t}={v:.3f}" for t, v in res["per_turn_ppl"].items())
        print(f"  {key:<20} overall {res['overall_ppl']:.3f}   {turns}")


def oracle_labels(evaluator, args, agg):
    """Full-cache future attention (all heads) at every compression point, per image."""
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from tgv_kv_legacy_nova import enable_online_attention, online_attention
    tokenizer, model, processor, _ = evaluator._test_load(args.model_path, None, "llava-v1.5-7b", False, False,
                                                          device="cpu")
    enable_online_attention(model)
    data = json.load(open(args.data_path))[: args.samples]
    image_len = int(model.get_vision_tower().num_patches)
    heads, layers = model.config.num_attention_heads, model.config.num_hidden_layers
    out = {}
    for item in data:
        ids, turns = collect.render(item, tokenizer, "llava_v1")
        image_pos = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0, 0])
        turns = collect.expand(turns, image_pos, image_len, 10 ** 9)
        points = collect.boundaries(turns)
        store = {}
        image = evaluator.load_image(os.path.join(args.image_path, item["image"]))
        tensor = process_images([image], processor, argparse.Namespace(image_aspect_ratio="pad")).float()
        with torch.no_grad(), online_attention(model, collect.make_sink(
                turns, points, [torch.arange(heads)] * layers, store, layers, dtype=torch.float32,
                label_agg="max" if agg == "max" else "mean")):
            past = model(input_ids=ids, images=tensor, use_cache=True, output_attentions=True).past_key_values
        per_point = []
        for b in range(len(points)):
            row = []
            for layer in range(layers):
                label = store[layer][b]["label"]
                if agg == "val":       # attention x value norm
                    label = label * past[layer][1][0, :, : label.shape[1]].norm(dim=-1)
                    label = label / label.sum(-1, keepdim=True)
                row.append(label)
            per_point.append(row)
        out[item["image"]] = per_point
    return out


def make_oracle(labels):
    def score(cache, stage, feats, snaps):
        per_point = labels[current["image"]]
        b = current["boundary"]
        current["boundary"] += 1
        if b >= len(per_point):            # last boundary: nothing is asked afterwards
            return [torch.zeros_like(s["q"]) for s in snaps]
        return [torch.log(per_point[b][layer].gather(1, snap["pos"]) + 1e-12)
                for layer, snap in enumerate(snaps)]
    return score


if __name__ == "__main__":
    main()
