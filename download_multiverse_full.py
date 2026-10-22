"""
Download passing2961/MultiVerse from Hugging Face and export it in the same
JSON format as multiverse_ppl.json, but with ALL turns for every sample.

Output format (per sample):
{
  "image": "multiverse_000000.png",
  "turns": [{"question": ..., "answer": ...}, ...],      # every turn
  "metadata": {
    "multiverse_index", "character", "scenario", "goal",
    "main_goal_category", "sub_goal_category",
    "main_task_category", "sub_task_category",
    "image_main_category", "image_sub_category",
    "original_complete_turns"
  }
}

Usage:
    pip install datasets pillow
    python download_multiverse_full.py --out_dir multiverse_full

    # optional: reuse the exact image file names from your existing 3-turn file
    python download_multiverse_full.py --out_dir multiverse_full --reference multiverse_ppl.json
"""

import argparse
import json
import os

from datasets import load_dataset

USER_ROLES = {"user", "human", "speaker a", "a"}
ASSISTANT_ROLES = {"assistant", "ai assistant", "gpt", "bot", "model", "speaker b", "b"}

METADATA_KEYS = [
    "character", "scenario", "goal",
    "main_goal_category", "sub_goal_category",
    "main_task_category", "sub_task_category",
    "image_main_category", "image_sub_category",
]


def to_utterance_list(conv):
    """Normalize the 'conversation' field to a list of (speaker, utterance)."""
    if isinstance(conv, dict):  # dict of parallel lists: {"speaker": [...], "utterance": [...]}
        return list(zip(conv.get("speaker", []), conv.get("utterance", [])))
    if isinstance(conv, list):  # list of dicts: [{"speaker": ..., "utterance": ...}, ...]
        return [(c.get("speaker"), c.get("utterance")) for c in conv]
    raise TypeError(f"Unexpected conversation type: {type(conv)}")


def build_turns(conv):
    """Pair each USER utterance with the following ASSISTANT utterance."""
    turns, pending_q = [], None
    for idx, (spk, utt) in enumerate(to_utterance_list(conv)):
        role = (spk or "").strip().lower()
        if role in USER_ROLES:
            is_user = True
        elif role in ASSISTANT_ROLES:
            is_user = False
        else:  # unknown labels: fall back to alternation (even = user, odd = assistant)
            is_user = idx % 2 == 0
        utt = (utt or "").strip()
        if is_user:
            pending_q = utt if pending_q is None else pending_q + "\n" + utt
        elif pending_q is not None:
            turns.append({"question": pending_q, "answer": utt})
            pending_q = None
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./MultiVerse_for_ppl/")
    ap.add_argument("--json_name", default="multiverse_ppl_full.json")
    ap.add_argument("--reference", default="/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/MultiVerse_for_ppl/multiverse_ppl.json",
                    help="Optional existing multiverse_ppl.json; image names are taken from it by multiverse_index")
    ap.add_argument("--no_images", action="store_true", help="Skip saving image files")
    args = ap.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ref_names = {}
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            for s in json.load(f):
                ref_names[str(s["metadata"]["multiverse_index"])] = s["image"]

    ds = load_dataset("passing2961/MultiVerse", split="train")
    print(ds)

    data = []
    for i, row in enumerate(ds):
        mv_index = str(row["index"])
        img_name = ref_names.get(mv_index, f"multiverse_{i:06d}.png")

        if not args.no_images and row.get("image") is not None:
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img = row["image"]
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
                img.save(img_path, format="PNG")

        turns = build_turns(row["conversation"])

        metadata = {"multiverse_index": mv_index}
        for k in METADATA_KEYS:
            metadata[k] = row.get(k)
        metadata["original_complete_turns"] = len(turns)

        data.append({"image": img_name, "turns": turns, "metadata": metadata})

        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(ds)}")

    out_path = os.path.join(args.out_dir, args.json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_turns = [len(s["turns"]) for s in data]
    print(f"Saved {len(data)} samples -> {out_path}")
    print(f"Turns per sample: min={min(n_turns)}, max={max(n_turns)}, "
          f"avg={sum(n_turns) / len(n_turns):.2f}")


if __name__ == "__main__":
    main()
