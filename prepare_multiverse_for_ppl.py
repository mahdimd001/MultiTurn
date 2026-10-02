#!/usr/bin/env python3
"""
Convert passing2961/MultiVerse into the JSON + image-folder format expected by
the current LLaVA multi-turn PPL evaluator.

Output JSON format:
[
  {
    "image": "multiverse_000000.png",
    "turns": [
      {"question": "...", "answer": "..."},
      ...
    ],
    "metadata": {...}
  }
]

By default:
  - requires at least 3 complete USER/ASSISTANT turns
  - keeps the first 3 turns, for direct T1/T2/T3 comparison with ConvBench
  - keeps exactly one image per dialogue (MultiVerse already has one)
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def normalize_conversation(conv):
    """
    Hugging Face Sequence features can appear either as:
      1) list[dict], or
      2) dict[str, list]
    Normalize both to list[dict].
    """
    if conv is None:
        return []

    if isinstance(conv, list):
        return conv

    if isinstance(conv, dict):
        speakers = conv.get("speaker", [])
        utterances = conv.get("utterance", [])
        checklists = conv.get("checklist", [None] * len(speakers))
        n = min(len(speakers), len(utterances))
        out = []
        for i in range(n):
            out.append({
                "speaker": speakers[i],
                "utterance": utterances[i],
                "checklist": checklists[i] if i < len(checklists) else None,
            })
        return out

    return []


def extract_turns(conv):
    """
    Keep complete USER -> ASSISTANT pairs only.
    MultiVerse normally alternates USER/ASSISTANT.
    """
    conv = normalize_conversation(conv)
    turns = []
    pending_user = None

    for msg in conv:
        speaker = str(msg.get("speaker", "")).strip().upper()
        utterance = str(msg.get("utterance", "")).strip()

        if not utterance:
            continue

        if speaker in {"USER", "HUMAN", "SPEAKER A"}:
            pending_user = utterance

        elif speaker in {"ASSISTANT", "AI ASSISTANT", "BOT", "SPEAKER B"}:
            if pending_user is not None:
                turns.append({
                    "question": pending_user,
                    "answer": utterance,
                })
                pending_user = None

    return turns


def save_image(image_obj, path):
    """
    HF Image features normally decode to PIL.Image.
    Also handle {'bytes': ..., 'path': ...} defensively.
    """
    if isinstance(image_obj, Image.Image):
        image = image_obj
    elif isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            from io import BytesIO
            image = Image.open(BytesIO(image_obj["bytes"]))
        elif image_obj.get("path"):
            image = Image.open(image_obj["path"])
        else:
            raise ValueError("Unsupported image dictionary")
    elif isinstance(image_obj, str):
        image = Image.open(image_obj)
    else:
        raise ValueError(f"Unsupported image type: {type(image_obj)}")

    # PNG is intentional: many MultiVerse samples contain diagrams,
    # tables, plots, text, or scientific figures.
    image.convert("RGB").save(path, format="PNG")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./MultiVerse_for_ppl")
    p.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Keep at most this many turns. Use 0 for all turns.",
    )
    p.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Skip dialogues with fewer complete USER/ASSISTANT turns.",
    )
    p.add_argument(
        "--task-category",
        default=None,
        help="Optional case-insensitive substring filter on main/sub task category.",
    )
    p.add_argument(
        "--goal-category",
        default=None,
        help="Optional case-insensitive substring filter on main/sub goal category.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N converted dialogues. 0 = no limit.",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Loading passing2961/MultiVerse from Hugging Face...")
    ds = load_dataset(
        "parquet",
        data_files="./MultiVerse_raw/data/train-00000-of-00001.parquet",
        split="train",
    )
    print(f"Loaded {len(ds)} dialogues")

    converted = []
    skipped_short = 0
    skipped_filter = 0
    skipped_bad_image = 0

    for row_idx, row in enumerate(ds):
        main_task = str(row.get("main_task_category", ""))
        sub_task = str(row.get("sub_task_category", ""))
        main_goal = str(row.get("main_goal_category", ""))
        sub_goal = str(row.get("sub_goal_category", ""))

        if args.task_category:
            needle = args.task_category.lower()
            if needle not in (main_task + " " + sub_task).lower():
                skipped_filter += 1
                continue

        if args.goal_category:
            needle = args.goal_category.lower()
            if needle not in (main_goal + " " + sub_goal).lower():
                skipped_filter += 1
                continue

        turns = extract_turns(row.get("conversation"))

        if len(turns) < args.min_turns:
            skipped_short += 1
            continue

        if args.max_turns > 0:
            turns = turns[: args.max_turns]

        # Use a stable local filename. Avoid ':' and other index punctuation.
        image_name = f"multiverse_{row_idx:06d}.png"
        image_path = image_dir / image_name

        try:
            save_image(row.get("image"), image_path)
        except Exception as exc:
            print(f"[skip row {row_idx}] image error: {exc}")
            skipped_bad_image += 1
            continue

        converted.append({
            "image": image_name,
            "turns": turns,
            "metadata": {
                "multiverse_index": row.get("index"),
                "character": row.get("character"),
                "scenario": row.get("scenario"),
                "goal": row.get("goal"),
                "main_goal_category": row.get("main_goal_category"),
                "sub_goal_category": row.get("sub_goal_category"),
                "main_task_category": row.get("main_task_category"),
                "sub_task_category": row.get("sub_task_category"),
                "image_main_category": row.get("image_main_category"),
                "image_sub_category": row.get("image_sub_category"),
                "original_complete_turns": len(extract_turns(row.get("conversation"))),
            },
        })

        if args.limit > 0 and len(converted) >= args.limit:
            break

    output_json = out_dir / "multiverse_ppl.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print()
    print(f"Converted dialogues: {len(converted)}")
    print(f"Skipped (< {args.min_turns} turns): {skipped_short}")
    print(f"Skipped by category filter: {skipped_filter}")
    print(f"Skipped bad images: {skipped_bad_image}")
    print(f"JSON:   {output_json}")
    print(f"Images: {image_dir}")

    if converted:
        turn_counts = [len(x["turns"]) for x in converted]
        print(f"Turns kept: min={min(turn_counts)}, max={max(turn_counts)}")
        print()
        print("Example evaluator arguments:")
        print(f"  --data-path {output_json}")
        print(f"  --image-path {image_dir}")


if __name__ == "__main__":
    main()
