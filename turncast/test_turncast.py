"""
Correctness checks for TurnCast-KV (CPU, any small LLaVA checkpoint).

1. Train/inference parity: the features TurnCastKVCache computes ONLINE while
   eval_ppl_final.py feeds a conversation token by token equal the features
   built OFFLINE from turncast/collect.py's single teacher-forced pass.
2. Budgets: at ratios 0.9 / 0.5 and both boundary rules, every compression keeps
   exactly L * C entries (or fewer when a layer is shorter), the attention sink
   is always kept, and the per-head positions stay sorted and unique.
3. No extra model passes: the number of model forwards equals PyramidKV's.

Usage: python turncast/test_turncast.py --model-path <llava dir> --data-path <json> --image-path <dir>
"""
import argparse
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "turncast"))
os.chdir(REPO)

import eval_ppl_final  # noqa: E402
import collect  # noqa: E402
import train  # noqa: E402
from kv_cache_turncast import FEATURE_NAMES, TurnCastKVCache, TurnCastScorer, save_scorer  # noqa: E402


def cpu_loader(*args, **kwargs):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, processor, ctx = load_pretrained_model(*args, **kwargs)
    model = model.float()
    model.get_vision_tower().to(dtype=torch.float32)
    return tokenizer, model.eval(), processor, ctx


def run_ppl(args, method, ratio, extra=None, items=1):
    """Run the unchanged eval_ppl_final.main and return the cache objects it built."""
    caches, forwards = [], [0]
    original_build = eval_ppl_final.build_cache

    def build(run_args, *a, **k):
        cache = original_build(run_args, *a, **k)
        if isinstance(cache, TurnCastKVCache):
            cache.record = True
        caches.append(cache)
        return cache

    eval_ppl_final.build_cache = build
    eval_ppl_final.load_pretrained_model = cpu_loader
    parser_args = [
        "--model-path", args.model_path, "--data-path", args.data_path, "--image-path", args.image_path,
        "--device", "cpu", "--method", method, "--ratio", str(ratio), "--eval-samples", str(items),
        "--output-dir", args.work, "--exp-name", f"test_{method}_{ratio}",
    ] + (extra or [])
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    original_forward = LlavaLlamaForCausalLM.forward

    def counting_forward(self, *a, **k):
        forwards[0] += 1
        return original_forward(self, *a, **k)

    LlavaLlamaForCausalLM.forward = counting_forward
    try:
        sys.argv = ["eval_ppl_final.py"] + parser_args
        ns = eval_ppl_final_parse()
        eval_ppl_final.main(ns)
    finally:
        LlavaLlamaForCausalLM.forward = original_forward
        eval_ppl_final.build_cache = original_build
    return caches, forwards[0]


def eval_ppl_final_parse():
    """Parse sys.argv with eval_ppl_final.py's own __main__ argument parser."""
    import textwrap
    source = open(os.path.join(REPO, "eval_ppl_final.py")).read()
    block = source[source.index('if __name__ == "__main__":'):].split("\n", 1)[1]
    captured = {}
    namespace = dict(eval_ppl_final.__dict__)
    namespace["main"] = lambda parsed: captured.setdefault("args", parsed)
    exec(textwrap.dedent(block), namespace)
    return captured["args"]


def parity(args):
    caches, _ = run_ppl(args, "turncast_kv", 0.0)
    cache = caches[0]
    online = cache.records

    # Offline: the collector's single pass over the same conversation.
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model  # noqa: F401
    import json
    item = json.load(open(args.data_path))[0]
    tokenizer, model, processor, _ = cpu_loader(args.model_path, None, "llava-v1.5-7b", False, False, device="cpu")
    from tgv_kv_legacy_nova import enable_online_attention, online_attention
    from llava.constants import IMAGE_TOKEN_INDEX
    enable_online_attention(model)
    ids, turns = collect.render(item, tokenizer, "llava_v1")
    image_len = int(model.get_vision_tower().num_patches)
    image_pos = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0, 0])
    turns = collect.expand(turns, image_pos, image_len, 10 ** 9)
    points = collect.boundaries(turns)
    heads = model.config.num_attention_heads
    layers = model.config.num_hidden_layers
    subset = [torch.arange(heads) for _ in range(layers)]
    store = {}
    image = eval_ppl_final.load_image(os.path.join(args.image_path, item["image"]))
    image_tensor = process_images([image], processor, types.SimpleNamespace(image_aspect_ratio="pad"))
    with torch.no_grad(), online_attention(
            model, collect.make_sink(turns, points, subset, store, layers, dtype=torch.float32)):
        out = model(input_ids=ids, images=image_tensor.float(), use_cache=True, output_attentions=True)
    past = out.past_key_values
    seg, turn_of = collect.segments(turns, image_pos, image_len)
    conv = {"n_layers": layers, "n_heads": heads, "heads": torch.stack(subset), "image_start": image_pos,
            "image_len": image_len, "seg": torch.tensor(seg), "turn": torch.tensor(turn_of)}
    worst = 0.0
    if len(online) < len(points):
        raise AssertionError(f"online recorded {len(online)} compression points, offline {len(points)}")
    for b, (stage, turn_now, size, _) in enumerate(points):
        per_layer = []
        for l in range(layers):
            entry = dict(store[l][b])
            pos = entry["pos"].long()
            entry["logk"] = past[l][0][0].norm(dim=-1).log().gather(1, pos)
            entry["logv"] = past[l][1][0].norm(dim=-1).log().gather(1, pos)
            per_layer.append(entry)
        sample = {"stage": stage, "turn_now": turn_now, "size": size, "layers": per_layer}
        offline = train.sample_features(conv, sample, torch.device("cpu"))
        record = online[b]
        if record["stage"] != stage or record["n_logical"] != size:
            raise AssertionError(f"point {b}: online {record['stage']}@{record['n_logical']} vs offline {stage}@{size}")
        diff = torch.stack([(on - off).abs().amax(dim=(0, 1)) for on, off in zip(record["features"], offline)])
        per_feature = diff.amax(0)
        worst = max(worst, float(per_feature.max()))
        bad = [(FEATURE_NAMES[i], float(per_feature[i])) for i in range(len(FEATURE_NAMES)) if per_feature[i] > 1e-3]
        print(f"  point {b} ({stage}, {size} entries): max |online - offline| = {float(per_feature.max()):.2e}"
              + (f"  OFF: {bad}" if bad else ""))
        if bad:
            raise AssertionError(f"feature mismatch at point {b}: {bad}")
    print(f"[ok] train/inference parity over {len(points)} compression points (max diff {worst:.1e})")


def budgets(args):
    layers = 32
    scorer_path = os.path.join(args.work, "random_scorer.pt")
    import json
    from llava.model.builder import load_pretrained_model  # noqa: F401
    cfg = json.load(open(os.path.join(args.model_path, "config.json")))
    layers, heads = cfg["num_hidden_layers"], cfg["num_key_value_heads"]
    torch.manual_seed(0)
    save_scorer(TurnCastScorer(layers, heads), scorer_path)
    _, pyramid_forwards = run_ppl(args, "pyramidkv", 0.9, ["--pyramid-exact-budget"])
    for ratio in (0.9, 0.5):
        for boundary in ("window", "forecast"):
            caches, forwards = run_ppl(args, "turncast_kv", ratio,
                                       ["--turncast-scorer", scorer_path, "--turncast-boundary", boundary])
            cache = caches[0]
            if ratio == 0.9 and forwards != pyramid_forwards:
                raise AssertionError(f"turncast ran {forwards} model forwards, PyramidKV {pyramid_forwards}")
            stats = cache.last_finalize_stats
            capacity = stats["target_per_layer"]
            kept = stats["layer_budgets"]
            if sum(kept) > layers * capacity:
                raise AssertionError(f"kept {sum(kept)} > {layers} x {capacity}")
            for layer, st in enumerate(cache.stats):
                pos = st.pos[:, : st.n]
                if not bool((pos[:, 0] == 0).all()):
                    raise AssertionError(f"layer {layer}: sink evicted")
                if not bool((pos[:, 1:] > pos[:, :-1]).all()):
                    raise AssertionError(f"layer {layer}: positions not strictly increasing")
            print(f"[ok] ratio {ratio} boundary {boundary}: last compression kept {sum(kept)} "
                  f"(L x C = {layers * capacity}), layers {min(kept)}..{max(kept)}, forwards {forwards}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--work", default="/tmp/turncast_test")
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    torch.set_num_threads(2)
    parity(a)
    budgets(a)


if __name__ == "__main__":
    main()
