"""
Collect TurnCast-KV training data: forecaster inputs and next-turn attention targets.

ONE teacher-forced forward pass per conversation (no generation, no cache
compression).  Under a full cache, the attention that turn t's tokens pay to an
earlier entry is the same whether the conversation is fed token by token (as the
evaluators do) or in one pass, so a single pass with output attentions yields:

  * inputs: at every compression point (after the Turn-1 prefill and after each
    answer), the running statistics LayerStats keeps online (same class, same
    code as kv_cache_turncast.TurnCastKVCache);
  * targets: the mean attention each later turn pays to every entry that exists
    at that point (each later turn weighted equally; for the prefill point the
    Turn-1 answer is the first "later turn"), normalized per (layer, head).

The conversation is rendered exactly like eval_ppl_final.py (same template, same
prefix-diff tokenization, reference answers).  To save space, every
conversation stores a random subset of heads per layer (--heads-per-layer); the
cross-head consensus inputs are computed from all heads before subsampling.

Split: conversations are shuffled with --seed; the first --train-count form the
training split, the rest the held-out split.  <out>/<dataset>/train.json and
heldout.json are written in the evaluator's input format, so the end-to-end
comparison runs on conversations the forecaster never saw.

Usage (Nova, one GPU, ~1 s per conversation for LLaVA-1.5-7B):
    python turncast/collect.py --model-path $MODEL --data-path ConvBench/convbench.json \
        --image-path $ROOT/ConvBench/visit_bench_images --dataset convbench --out-dir turncast_data
    python turncast/collect.py ... --split heldout --limit 60      # for the offline check
"""
import argparse
import json
import os
import random
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kv_cache_turncast import (  # noqa: E402
    LayerStats, SEG_ID, layer_consensus, prompt_segments, pyramid_budgets,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-base", default=None)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--dataset", required=True, help="name used for the output folder")
    p.add_argument("--out-dir", default="turncast_data")
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    p.add_argument("--train-count", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="collect at most this many conversations")
    p.add_argument("--heads-per-layer", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048, help="model context; later turns are dropped")
    p.add_argument("--answers-from", default=None,
                   help="optional eval_rouge_final2 predictions JSON: use generated answers, not references")
    p.add_argument("--simulate-ratios", default="",
                   help="comma list, e.g. 0,0.5,0.8,0.9: each conversation is replayed under one of these "
                        "compression ratios (0 = full cache); default: full cache only")
    p.add_argument("--label-agg", choices=["mean", "max"], default="mean",
                   help="target per later turn: mean or peak attention of its tokens")
    p.add_argument("--conv-mode", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-aspect-ratio", default="pad")
    return p.parse_args()


# ----------------------------------------------------------------------
# Rendering (identical to eval_ppl_final.py's turn loop)
# ----------------------------------------------------------------------
def render(item, tokenizer, conv_mode, answers=None):
    from eval_ppl_final import answer_only_prompt, require_prefix, tokenize_conversation_prompt
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    conv = conv_templates[conv_mode].copy()
    committed, turns = None, []
    for t, turn in enumerate(item["turns"]):
        question = turn["question"]
        answer = answers[t] if answers is not None else turn["answer"]
        if t == 0 and DEFAULT_IMAGE_TOKEN not in question:
            question = question + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        base_prompt = conv.get_prompt()
        base_ids = tokenize_conversation_prompt(base_prompt, tokenizer, "cpu")
        if t > 0:
            require_prefix(base_ids, committed, description=f"turn {t}")
        q_start = 0 if t == 0 else committed.shape[1]
        with_answer = conv.copy()
        with_answer.messages[-1][-1] = answer
        prompt_with_answer = with_answer.get_prompt()
        through = answer_only_prompt(base_prompt, prompt_with_answer, answer)
        ids_through = tokenize_conversation_prompt(through, tokenizer, "cpu")
        ids_full = tokenize_conversation_prompt(prompt_with_answer, tokenizer, "cpu")
        require_prefix(ids_through, base_ids, description=f"turn {t} answer")
        require_prefix(ids_full, ids_through, description=f"turn {t} separator")
        turns.append({"q": (q_start, base_ids.shape[1]), "a": (base_ids.shape[1], ids_full.shape[1])})
        conv, committed = with_answer, ids_full
    return committed, turns


def expand(turns, image_pos, image_len, max_length):
    """Map token-id ranges to expanded KV positions; drop turns beyond max_length."""
    shift = image_len - 1

    def e(x):
        return x if x <= image_pos else x + shift

    out = []
    for turn in turns:
        q = (e(turn["q"][0]) if turn["q"][0] > 0 else 0, e(turn["q"][1]))
        a = (e(turn["a"][0]), e(turn["a"][1]))
        if a[1] > max_length:
            break
        out.append({"q": q, "a": a})
    return out


def segments(turns, image_start, image_len):
    prompt_len = turns[0]["q"][1]
    seg = prompt_segments(prompt_len, image_start, image_len)
    turn_of = [0] * prompt_len
    for t, turn in enumerate(turns):
        if t > 0:
            seg += [SEG_ID["user"]] * (turn["q"][1] - turn["q"][0])
            turn_of += [t] * (turn["q"][1] - turn["q"][0])
        seg += [SEG_ID["assistant"]] * (turn["a"][1] - turn["a"][0])
        turn_of += [t] * (turn["a"][1] - turn["a"][0])
    return seg, turn_of


def boundaries(turns):
    """Compression points with at least one later turn: (stage, turn, size, future row blocks)."""
    points = [("prefill", 0, turns[0]["q"][1], [turns[0]["a"]] + [(t["q"][0], t["a"][1]) for t in turns[1:]])]
    for t in range(len(turns) - 1):
        future = [(u["q"][0], u["a"][1]) for u in turns[t + 1:]]
        points.append(("turn", t, turns[t]["a"][1], future))
    return points


# ----------------------------------------------------------------------
# Per-layer replay of the online statistics
# ----------------------------------------------------------------------
def make_sink(turns, points, heads_subset, store, n_layers, sim_ratio=None, dtype=torch.float16,
              label_agg="mean"):
    """Called by the attention hook with each layer's [1, H, N, N] attention.

    Replays the conversation through LayerStats exactly as TurnCastKVCache sees
    it.  With sim_ratio, the cache is also COMPRESSED at every compression point
    (oracle top-k under the evaluator's budget), and later attention rows are
    restricted to the surviving entries and renormalized, so the forecaster also
    learns from the pruned histories it meets at inference (later turns).
    """

    def sink(layer_idx, attention):
        att = attention[0]
        heads, device = att.shape[0], att.device
        stats = LayerStats(heads, device, att.shape[-1])
        keep = heads_subset[layer_idx].to(device)
        results = []
        lengths = None          # physical length of every layer (same bookkeeping in each layer)

        def add(start, end):
            nonlocal lengths
            if end <= start:
                return
            index = torch.cat([stats.pos[:, : stats.n],
                               torch.arange(start, end, device=device).expand(heads, -1)], dim=1)
            rows = att[:, start:end, :].gather(2, index[:, None, :].expand(heads, end - start, -1)).float()
            rows = rows / rows.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            stats.add_block(rows, torch.arange(start, end))
            if lengths is not None:
                lengths = [length + end - start for length in lengths]

        def boundary(stage, turn_now, size, future):
            nonlocal lengths
            n = stats.n
            snap = stats.snapshot(n)
            cons, present = layer_consensus(snap, size)
            pos = snap["pos"]
            blocks = [att[:, s:e, :].gather(2, pos[:, None, :].expand(heads, e - s, n)).float()
                      for s, e in future if e > s]
            # per later turn: mean (or peak) attention of its tokens; turns weighted equally
            blocks = [b.amax(dim=1) if label_agg == "max" else b.mean(dim=1) for b in blocks]
            label = torch.stack(blocks).mean(0)
            label = label / label.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            results.append({
                "q": snap["q"][keep].to(dtype).cpu(), "a": snap["a"][keep].to(dtype).cpu(),
                "w": snap["w"][keep].to(dtype).cpu(), "h": snap["h"][keep].to(dtype).cpu(),
                "q_has": snap["q_has"][keep].cpu(), "a_has": snap["a_has"][keep].cpu(),
                "h_has": snap["h_has"][keep].cpu(), "pos": pos[keep].int().cpu(),
                "cons": cons.to(dtype).cpu(), "present": present.cpu(), "label": label[keep].cpu(),
            })
            if stage == "turn":
                stats.end_turn(n)
            if sim_ratio:
                if lengths is None:
                    lengths = [n] * n_layers
                budgets = pyramid_budgets(n_layers, sim_ratio, size, lengths)
                if budgets is not None:
                    if budgets[layer_idx] < n:
                        score = label.clone()
                        score[pos == 0] = float("inf")
                        stats.gather(score.topk(budgets[layer_idx], dim=-1).indices.sort(dim=-1).values)
                    lengths = [min(b, length) for b, length in zip(budgets, lengths)]
            stats.reset_window()

        point = 0
        for t, turn in enumerate(turns):
            if point >= len(points):
                break
            stats.set_phase("q")
            add(*turn["q"])
            if t == 0:
                boundary(*points[0])
                point = 1
            stats.set_phase("a")
            add(*turn["a"])
            if point < len(points):
                boundary(*points[point])
                point += 1
        store[layer_idx] = results

    return sink


def main():
    args = parse_args()
    from eval_ppl_final import choose_conversation_mode, load_image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import get_model_name_from_path, process_images
    from llava.model.builder import load_pretrained_model
    from tgv_kv_legacy_nova import enable_online_attention, online_attention

    data = json.load(open(args.data_path))
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    split = {"train": sorted(order[: args.train_count]), "heldout": sorted(order[args.train_count:])}
    out_root = os.path.join(args.out_dir, args.dataset)
    os.makedirs(os.path.join(out_root, args.split), exist_ok=True)
    json.dump({"seed": args.seed, "train_count": args.train_count, **split},
              open(os.path.join(out_root, "split.json"), "w"))
    for name in ("train", "heldout"):
        json.dump([data[i] for i in split[name]], open(os.path.join(out_root, f"{name}.json"), "w"))

    answers_by_key = None
    if args.answers_from:
        answers_by_key = {}
        for item in json.load(open(args.answers_from))["results"]:
            key = (item["image"], item["turns"][0]["question"])
            answers_by_key[key] = [turn["prediction"] for turn in item["turns"]]

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, False, False, device=args.device)
    if args.device == "cpu":
        model = model.float()
        model.get_vision_tower().to(dtype=torch.float32)
    model.eval()
    enable_online_attention(model)
    conv_mode = choose_conversation_mode(model_name, args.conv_mode)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else model.config.num_attention_heads
    if n_heads != model.config.num_attention_heads:
        raise NotImplementedError("collector assumes one query head per KV head (LLaVA-1.5)")
    image_len = int(model.get_vision_tower().num_patches)
    heads_per_layer = min(args.heads_per_layer, n_heads)

    indices = split[args.split][: args.limit]
    sim_ratios = [float(r) for r in args.simulate_ratios.split(",") if r.strip()]
    rng = random.Random(args.seed + 7)
    generator = torch.Generator().manual_seed(args.seed + (0 if args.split == "train" else 1))
    started, written, skipped = time.time(), 0, 0
    for count, index in enumerate(indices):
        path = os.path.join(out_root, args.split, f"{index:05d}.pt")
        if os.path.exists(path):
            continue
        item = data[index]
        answers = None
        if answers_by_key is not None:
            answers = answers_by_key.get((item["image"], item["turns"][0]["question"]))
            if answers is None:
                skipped += 1
                continue
        ids, turns = render(item, tokenizer, conv_mode, answers)
        image_pos = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0, 0])
        turns = expand(turns, image_pos, image_len, args.max_length)
        if not turns:
            skipped += 1
            continue
        end = turns[-1]["a"][1] - (image_len - 1)
        points = boundaries(turns)
        heads_subset = [torch.randperm(n_heads, generator=generator)[:heads_per_layer].sort().values
                        for _ in range(n_layers)]
        store = {}
        sim_ratio = float(rng.choice(sim_ratios)) if sim_ratios else 0.0
        image = load_image(os.path.join(args.image_path, item["image"]))
        image_tensor = process_images([image], image_processor, args).to(
            model.device, dtype=next(model.parameters()).dtype)
        with torch.no_grad(), online_attention(
                model, make_sink(turns, points, heads_subset, store, n_layers, sim_ratio or None,
                label_agg=args.label_agg)):
            outputs = model(input_ids=ids[:, :end].to(model.device), images=image_tensor,
                            use_cache=True, output_attentions=True)
        past = outputs.past_key_values
        del outputs
        if past[0][0].shape[2] != turns[-1]["a"][1]:
            raise RuntimeError(f"item {index}: expanded length {past[0][0].shape[2]} != {turns[-1]['a'][1]}")
        seg, turn_of = segments(turns, image_pos, image_len)
        norms = []
        for layer in range(n_layers):
            heads = heads_subset[layer].to(past[layer][0].device)
            norms.append([past[layer][i][0, heads].float().norm(dim=-1).clamp(min=1e-6).log().cpu()
                          for i in (0, 1)])
        samples = []
        for b, (stage, turn_now, size, _) in enumerate(points):
            layers = []
            for layer in range(n_layers):
                entry = store[layer][b]
                pos = entry["pos"].long()
                entry["logk"] = norms[layer][0].gather(1, pos).half()
                entry["logv"] = norms[layer][1].gather(1, pos).half()
                layers.append(entry)
            samples.append({"stage": stage, "turn_now": turn_now, "size": size,
                            "sim_ratio": sim_ratio, "layers": layers})
        torch.save({
            "dataset": args.dataset, "index": index, "image": item["image"],
            "n_layers": n_layers, "n_heads": n_heads, "heads": torch.stack(heads_subset),
            "image_start": image_pos, "image_len": image_len,
            "seg": torch.tensor(seg, dtype=torch.int8), "turn": torch.tensor(turn_of, dtype=torch.int8),
            "samples": samples, "format": 2,
        }, path)
        written += 1
        del past, store
        if (count + 1) % 10 == 0:
            rate = (time.time() - started) / (count + 1)
            print(f"[{args.dataset}/{args.split}] {count + 1}/{len(indices)}  {rate:.2f}s/conv", flush=True)
    print(f"[{args.dataset}/{args.split}] wrote {written}, skipped {skipped}, "
          f"{time.time() - started:.0f}s total -> {os.path.join(out_root, args.split)}")


if __name__ == "__main__":
    main()
