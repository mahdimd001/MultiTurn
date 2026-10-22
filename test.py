"""
Read a MultiVerse JSON file and return the samples with the max number of turns.

Usage:
    python max_turn_samples.py multiverse_ppl_full.json
    python max_turn_samples.py multiverse_ppl_full.json --out max_turn_samples.json
"""

import argparse
import json
from collections import Counter


def get_max_turn_samples(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    max_turns = max(len(s["turns"]) for s in data)
    samples = [s for s in data if len(s["turns"]) == max_turns]
    return data, max_turns, samples


def main():
    ap = argparse.ArgumentParser()
    path="/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/MultiVerse_for_ppl/multiverse_ppl_full.json"
    ap.add_argument("--out", default=None, help="Optional: save the max-turn samples to this JSON file")
    args = ap.parse_args()

    data, max_turns, samples = get_max_turn_samples(path)

    print("Turn count distribution:", dict(sorted(Counter(len(s["turns"]) for s in data).items())))
    print(f"Max turns: {max_turns}")
    print(f"Samples with max turns: {len(samples)}")
    for s in samples:
        print(f"  {s['image']}  (multiverse_index={s['metadata']['multiverse_index']})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()