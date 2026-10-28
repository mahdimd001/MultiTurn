"""
Train the TurnCast-KV forecaster and check it offline against PyramidKV.

Input: folders written by turncast/collect.py (one .pt file per conversation).
Output: scorer.pt (load with --turncast-scorer) and report.json.

Training: listwise distillation.  For every compression point and every
(layer, head), the forecaster's softmax over the cached entries is fitted to the
attention share the later turns actually paid (KL divergence).  Conversations
are split 90/10 into fit / validation; the epoch with the lowest validation KL
is kept.  Fixed settings (not tuned per run): AdamW, lr 2e-3, weight decay 1e-4,
8 epochs, cosine schedule, 64 hidden units.

Offline check (on --eval-dirs, e.g. the held-out conversations, or else on the
validation split): at ratios 0.9 / 0.8 / 0.5 and the evaluator's budget rule,
the share of the later turns' attention that the KEPT entries receive, for
  pyramidkv          PyramidKV scores (last-8 window, maxpool 7) + its floored schedule
  pyramidkv_exact    the same scores with an exact L*C budget
  turn_attention     this turn's attention (question + answer), exact pyramid budget
  turncast           the forecaster, exact pyramid budget
  oracle             the true future attention (upper bound), exact pyramid budget
All of these see a full (uncompressed) history, so they rank single compression
steps; the end-to-end evaluators measure the compounding over turns.

Usage:
    python turncast/train.py --train-dirs turncast_data/convbench/train turncast_data/multiverse/train \
        --eval-dirs turncast_data/convbench/heldout turncast_data/multiverse/heldout --out turncast_scorer.pt
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kv_cache_turncast import (  # noqa: E402
    FEATURE_NAMES, WINDOW, TurnCastScorer, conversation_meta, global_consensus,
    layer_features, load_scorer, pyramid_budgets, pyramid_capacities, save_scorer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dirs", nargs="+", default=[])
    p.add_argument("--eval-dirs", nargs="*", default=[])
    p.add_argument("--out", default="turncast_scorer.pt")
    p.add_argument("--report", default=None, help="default: <out without .pt>_report.json")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate --out")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--groups-per-step", type=int, default=0,
                   help="0 = all (layer, head) groups of a compression point per step")
    p.add_argument("--feature-cache-gb", type=float, default=8.0)
    p.add_argument("--ratios", default="0.9,0.8,0.5")
    p.add_argument("--stages", default="prefill,turn",
                   help="compression points to train on (prefill = Turn-1 compression, turn = boundaries)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_conversations(dirs):
    convs = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.pt"))):
            conv = torch.load(path, map_location="cpu")
            conv["path"] = path
            convs.append(conv)
    return convs


def sample_features(conv, sample, device):
    """Per-layer [Hs, S_l, F] features, exactly as TurnCastKVCache computes them online."""
    size = sample["size"]
    meta = conversation_meta(conv["seg"][:size].long(), conv["turn"][:size].long(),
                             sample["turn_now"], conv["image_start"], conv["image_len"], device)
    layers = sample["layers"]
    cons = [layer["cons"].float().to(device) for layer in layers]
    present = [layer["present"].to(device) for layer in layers]
    glob_values = global_consensus(cons, present)
    feats = []
    for index, layer in enumerate(layers):
        snap = {name: layer[name].float().to(device) for name in ("q", "a", "w", "h")}
        snap.update({name: layer[name].to(device) for name in ("q_has", "a_has", "h_has")})
        snap["pos"] = layer["pos"].long().to(device)
        feats.append(layer_features(snap, layer["logk"].float().to(device), layer["logv"].float().to(device),
                                    cons[index], glob_values, meta))
    return feats


def sample_labels(sample, device):
    out = []
    for layer in sample["layers"]:
        y = layer["label"].float().to(device)
        out.append(y / y.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return out


def forward_layers(scorer, conv, feats, layer_ids):
    """Logits for the selected layers: list of [Hs, S_l]."""
    n_heads = conv["n_heads"]
    flat, ids = [], []
    for layer, f in zip(layer_ids, feats):
        heads = conv["heads"][layer].long().to(f.device)
        flat.append(f.reshape(-1, f.shape[-1]))
        ids.append((layer * n_heads + heads).view(-1, 1).expand(f.shape[0], f.shape[1]).reshape(-1))
    logits = scorer(torch.cat(flat), torch.cat(ids))
    out, start = [], 0
    for f in feats:
        size = f.shape[0] * f.shape[1]
        out.append(logits[start: start + size].view(f.shape[0], f.shape[1]))
        start += size
    return out


def kl_loss(logits, labels):
    """Mean over (layer, head) groups of KL(label || softmax(logits))."""
    total, groups = 0.0, 0
    for z, y in zip(logits, labels):
        logp = torch.log_softmax(z.float(), dim=-1)
        total = total + (torch.xlogy(y, y) - y * logp).sum(dim=-1).sum()
        groups += y.shape[0]
    return total / max(groups, 1)


class FeatureCache:
    def __init__(self, budget_gb, device):
        self.budget = budget_gb * 1e9
        self.used = 0
        self.items = {}
        self.device = device

    def get(self, key, conv, sample):
        if key in self.items:
            return [f.float() for f in self.items[key]]
        feats = sample_features(conv, sample, self.device)
        size = sum(f.numel() for f in feats) * 2
        if self.used + size <= self.budget:
            self.items[key] = [f.half() for f in feats]
            self.used += size
        return feats


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def fit_normalization(scorer, convs, device, limit=64):
    rows = []
    pairs = [(c, s) for c in convs for s in c["samples"]]
    random.Random(0).shuffle(pairs)
    for conv, sample in pairs[:limit]:
        feats = torch.cat([f.reshape(-1, len(FEATURE_NAMES)) for f in sample_features(conv, sample, device)])
        rows.append(feats[torch.randperm(feats.shape[0], device=device)[:20000]])
    rows = torch.cat(rows)
    scorer.feat_mean.copy_(rows.mean(0))
    scorer.feat_std.copy_(rows.std(0).clamp(min=1e-3))


def evaluate_kl(scorer, pairs, cache, device):
    total, count = 0.0, 0
    with torch.no_grad():
        for key, conv, sample in pairs:
            feats = cache.get(key, conv, sample)
            logits = forward_layers(scorer, conv, feats, range(len(feats)))
            total += float(kl_loss(logits, sample_labels(sample, device)))
            count += 1
    return total / max(count, 1)


def train(args, convs, device):
    rng = random.Random(args.seed)
    order = list(range(len(convs)))
    rng.shuffle(order)
    n_val = max(1, len(order) // 10)
    val_ids, fit_ids = set(order[:n_val]), order[n_val:]
    fit_pairs = [((i, j), convs[i], s) for i in fit_ids for j, s in enumerate(convs[i]["samples"])]
    val_pairs = [((i, j), convs[i], s) for i in sorted(val_ids) for j, s in enumerate(convs[i]["samples"])]
    n_layers, n_heads = convs[0]["n_layers"], convs[0]["n_heads"]
    torch.manual_seed(args.seed)
    scorer = TurnCastScorer(n_layers, n_heads, hidden=args.hidden).to(device)
    fit_normalization(scorer, [convs[i] for i in fit_ids], device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(fit_pairs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1))))
    cache = FeatureCache(args.feature_cache_gb, device)
    best, best_state, history = float("inf"), None, []
    print(f"fit on {len(fit_pairs)} compression points ({len(fit_ids)} conversations), "
          f"validate on {len(val_pairs)} ({len(val_ids)} conversations); "
          f"{sum(p.numel() for p in scorer.parameters())} parameters", flush=True)
    for epoch in range(args.epochs):
        started = time.time()
        rng.shuffle(fit_pairs)
        scorer.train()
        running = 0.0
        for key, conv, sample in fit_pairs:
            feats = cache.get(key, conv, sample)
            labels = sample_labels(sample, device)
            layer_ids = list(range(len(feats)))
            if args.groups_per_step:
                heads = feats[0].shape[0]
                layer_ids = sorted(rng.sample(layer_ids, max(1, min(len(layer_ids), args.groups_per_step // heads))))
                feats, labels = [feats[i] for i in layer_ids], [labels[i] for i in layer_ids]
            loss = kl_loss(forward_layers(scorer, conv, feats, layer_ids), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss)
        scorer.eval()
        val = evaluate_kl(scorer, val_pairs, cache, device)
        history.append({"epoch": epoch + 1, "train_kl": running / len(fit_pairs), "val_kl": val})
        print(f"epoch {epoch + 1}: train KL {running / len(fit_pairs):.4f}  val KL {val:.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in scorer.state_dict().items()}
    scorer.load_state_dict(best_state)
    scorer.eval()
    return scorer, [convs[i] for i in sorted(val_ids)], history


# ----------------------------------------------------------------------
# Offline check
# ----------------------------------------------------------------------
def floored_pyramid(n_layers, ratio, n_logical, lengths):
    """PyramidKV's own (floored) schedule: past entries per layer + window."""
    capacity = max(1, int(n_logical * (1.0 - ratio)))
    window = max(1, min(WINDOW, capacity - 1)) if capacity > 1 else 1
    caps = pyramid_capacities(n_layers, capacity, window, max(lengths), exact=False)
    return caps, window


def topk_keep(scores, budget, pos):
    scores = scores.clone()
    scores[pos == 0] = float("inf")
    return scores.topk(budget, dim=-1).indices


def offline_eval(scorer, convs, ratios, device):
    methods = ["pyramidkv", "pyramidkv_exact", "turn_attention", "turncast", "oracle"]
    totals = {r: {m: {"prefill": [], "turn": []} for m in methods} for r in ratios}
    memory = {r: {m: [0, 0] for m in methods} for r in ratios}
    with torch.no_grad():
        for conv in convs:
            for sample in conv["samples"]:
                size, stage = sample["size"], sample["stage"]
                layers = sample["layers"]
                n_layers = len(layers)
                labels = sample_labels(sample, device)
                feats = sample_features(conv, sample, device)
                logits = forward_layers(scorer, conv, feats, range(n_layers))
                lengths = [y.shape[1] for y in labels]
                for ratio in ratios:
                    exact = pyramid_budgets(n_layers, ratio, size, lengths)
                    if exact is None:
                        continue
                    floored, window = floored_pyramid(n_layers, ratio, size, lengths)
                    for layer in range(n_layers):
                        y, length = labels[layer], lengths[layer]
                        pos = layers[layer]["pos"].long().to(device)
                        w = layers[layer]["w"].float().to(device)
                        past = length - window
                        for name, caps in (("pyramidkv", floored), ("pyramidkv_exact", [b - window for b in exact])):
                            pooled = F.max_pool1d(w[:, :past].unsqueeze(1), 7, 1, 3).squeeze(1)
                            keep = torch.cat([pooled.topk(max(0, min(caps[layer], past)), dim=-1).indices,
                                              torch.arange(past, length, device=device).expand(y.shape[0], -1)], -1)
                            totals[ratio][name][stage].append(y.gather(1, keep).sum(-1))
                            memory[ratio][name][0] += keep.shape[1]
                            memory[ratio][name][1] += length
                        turn_score = torch.logaddexp(layers[layer]["q"].float(), layers[layer]["a"].float()).to(device)
                        for name, scores, budget in (
                            ("turn_attention", turn_score, exact[layer]),
                            ("turncast", logits[layer], exact[layer]),
                            ("oracle", y, exact[layer]),
                        ):
                            keep = topk_keep(scores, min(budget, length), pos)
                            totals[ratio][name][stage].append(y.gather(1, keep).sum(-1))
                            memory[ratio][name][0] += keep.shape[1]
                            memory[ratio][name][1] += length
    report = {}
    for ratio in ratios:
        report[str(ratio)] = {}
        for name in methods:
            row = {}
            for stage in ("prefill", "turn"):
                vals = totals[ratio][name][stage]
                row[stage] = float(torch.cat(vals).mean()) if vals else float("nan")
            both = totals[ratio][name]["prefill"] + totals[ratio][name]["turn"]
            row["all"] = float(torch.cat(both).mean()) if both else float("nan")
            kept, total = memory[ratio][name]
            row["kept_fraction"] = kept / max(1, total)
            report[str(ratio)][name] = row
    return report


def print_report(title, report):
    print(f"\n{title}: share of later turns' attention kept (higher is better)")
    for ratio, rows in report.items():
        print(f"  ratio {ratio}")
        print(f"    {'method':<16} {'prefill':>8} {'turn':>8} {'all':>8} {'kept':>8}")
        for name, row in rows.items():
            print(f"    {name:<16} {row['prefill']:8.4f} {row['turn']:8.4f} {row['all']:8.4f} "
                  f"{row['kept_fraction']:8.4f}")


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    ratios = [float(r) for r in args.ratios.split(",")]
    report = {"args": vars(args)}
    if args.eval_only:
        scorer = load_scorer(args.out, device)
        val_convs = []
    else:
        convs = load_conversations(args.train_dirs)
        stages = set(args.stages.split(","))
        for conv in convs:
            conv["samples"] = [s for s in conv["samples"] if s["stage"] in stages]
        convs = [c for c in convs if c["samples"]]
        if not convs:
            raise SystemExit("no training conversations found")
        scorer, val_convs, history = train(args, convs, device)
        save_scorer(scorer, args.out, extra={"history": history, "train_dirs": args.train_dirs,
                                             "conversations": len(convs)})
        report["history"] = history
        print(f"saved {args.out}")
        report["validation"] = offline_eval(scorer, val_convs, ratios, device)
        print_report("validation split", report["validation"])
    if args.eval_dirs:
        for d in args.eval_dirs:
            convs = load_conversations([d])
            if convs:
                report[d] = offline_eval(scorer, convs, ratios, device)
                print_report(f"{d} ({len(convs)} conversations)", report[d])
    path = args.report or os.path.splitext(args.out)[0] + "_report.json"
    json.dump(report, open(path, "w"), indent=2)
    print(f"report -> {path}")


if __name__ == "__main__":
    main()