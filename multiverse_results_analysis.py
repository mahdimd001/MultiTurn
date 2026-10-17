#!/usr/bin/env python3
"""
ROUGE-L breakdown by MultiVerse category.

Joins one or more prediction JSON files (format: tgv_kv_ratio0.5_predictions.json)
with the MultiVerse metadata (passing2961/MultiVerse, split=train, 647 rows) and
reports ROUGE-L F by:
  1. main_goal_category
  2. main_task_category
(plus optional extras: image_main_category, goal x task cross-tab)

Join rule: "multiverse_000123.png" -> row 123 of the train split.
The join is sanity-checked by comparing each item's turn-1 question with the
first USER utterance in the dataset row; mismatches are reported.

Usage
-----
  pip install datasets pandas
  python rouge_by_category.py preds/*.json --out results/

  # offline: point at a local metadata export instead of downloading
  python rouge_by_category.py preds/*.json --meta multiverse_meta.jsonl --out results/

  # export the metadata once (no images) for offline use
  python rouge_by_category.py --dump-meta multiverse_meta.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import pandas as pd

CATEGORY_COLS = [
    "main_goal_category",
    "main_task_category",
    "image_main_category",
    "sub_goal_category",
    "sub_task_category",
]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_meta_from_hf(repo="passing2961/MultiVerse", split="train"):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    keep = [c for c in ["index", "conversation", *CATEGORY_COLS] if c in ds.column_names]
    ds = ds.select_columns(keep)  # drop images -> fast, small
    rows = []
    for pos, r in enumerate(ds):
        conv = r.get("conversation") or {}
        first_user = _first_user_utterance(conv)
        rows.append({"row": pos, "index": r.get("index"), "first_user": first_user,
                     **{c: r.get(c) for c in CATEGORY_COLS if c in r}})
    return pd.DataFrame(rows)


def _first_user_utterance(conv):
    # conversation is {"speaker": [...], "utterance": [...]} or a list of dicts
    if isinstance(conv, dict):
        spk, utt = conv.get("speaker", []), conv.get("utterance", [])
        for s, u in zip(spk, utt):
            if str(s).upper() == "USER":
                return u
        return utt[0] if utt else None
    if isinstance(conv, list) and conv:
        for t in conv:
            if isinstance(t, dict) and str(t.get("speaker", "")).upper() == "USER":
                return t.get("utterance")
    return None


def load_meta_local(path):
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "row" not in df.columns:
        df["row"] = range(len(df))
    return df


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
IMG_RE = re.compile(r"(\d+)")


def image_to_row(name):
    m = IMG_RE.findall(os.path.basename(name))
    if not m:
        raise ValueError(f"Cannot parse row id from image name: {name}")
    return int(m[-1])


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    exp = data.get("experiment") or os.path.splitext(os.path.basename(path))[0]
    recs = []
    for item in data["results"]:
        row = image_to_row(item["image"])
        turns = item.get("turns", [])
        q1 = next((t["question"] for t in turns if t.get("turn") == 1), None)
        for t in turns:
            recs.append({
                "experiment": exp,
                "image": item["image"],
                "row": row,
                "turn": int(t["turn"]),
                "rouge_l_f": float(t["rouge_l_f"]),
                "gen_tokens": t.get("generated_answer_tokens"),
                "refused_no_image": _looks_like_no_image(t.get("prediction", "")),
                "q1": q1,
            })
    return exp, pd.DataFrame(recs), data


NO_IMAGE_PAT = re.compile(r"(cannot|can't|unable to|don't have the ability to) (see|view|access) the image", re.I)


def _looks_like_no_image(text):
    return bool(NO_IMAGE_PAT.search(text or ""))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize(df, by):
    """Per-category table: #dialogues, #turns, turn-level mean, dialogue-level mean,
    per-turn means, std, and share of 'cannot see the image' answers."""
    g = df.groupby(by)
    turn_level = g["rouge_l_f"].agg(n_turns="count", rouge_l_turn_mean="mean", rouge_l_std="std")
    item_means = df.groupby([by, "row"])["rouge_l_f"].mean().groupby(by)
    item_level = item_means.agg(n_dialogues="count", rouge_l_dialogue_mean="mean")
    per_turn = df.pivot_table(index=by, columns="turn", values="rouge_l_f", aggfunc="mean")
    per_turn.columns = [f"turn{c}" for c in per_turn.columns]
    no_img = g["refused_no_image"].mean().rename("pct_no_image_answers") * 100
    out = item_level.join(turn_level).join(per_turn).join(no_img)
    out = out.sort_values("rouge_l_turn_mean", ascending=False)
    # overall row
    overall = {
        "n_dialogues": df["row"].nunique(),
        "n_turns": len(df),
        "rouge_l_turn_mean": df["rouge_l_f"].mean(),
        "rouge_l_std": df["rouge_l_f"].std(),
        "rouge_l_dialogue_mean": df.groupby("row")["rouge_l_f"].mean().mean(),
        "pct_no_image_answers": df["refused_no_image"].mean() * 100,
        **{f"turn{t}": v for t, v in df.groupby("turn")["rouge_l_f"].mean().items()},
    }
    out.loc["OVERALL"] = pd.Series(overall)
    cols = ["n_dialogues", "n_turns", "rouge_l_turn_mean", "rouge_l_dialogue_mean",
            "rouge_l_std"] + sorted(c for c in out.columns if c.startswith("turn")) + ["pct_no_image_answers"]
    out = out[cols]
    out[["n_dialogues", "n_turns"]] = out[["n_dialogues", "n_turns"]].astype(int)
    return out


def check_join(df, meta):
    if "first_user" not in meta.columns:
        return None
    m = df.drop_duplicates("row")[["row", "image", "q1"]].merge(meta[["row", "first_user"]], on="row", how="left")
    norm = lambda s: re.sub(r"\W+", " ", str(s or "")).strip().lower()[:80]
    m["ok"] = [norm(a) == norm(b) for a, b in zip(m["q1"], m["first_user"])]
    return m


def fmt_md(df, title):
    d = df.copy()
    for c in d.columns:
        if c.startswith("rouge") or c.startswith("turn"):
            d[c] = (d[c] * 100).map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        elif c == "pct_no_image_answers":
            d[c] = d[c].map(lambda x: f"{x:.1f}%")
    return f"### {title}\n\n(ROUGE-L F × 100)\n\n" + d.to_markdown() + "\n"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="*", help="prediction JSON file(s)")
    ap.add_argument("--meta", help="local metadata (jsonl/json/parquet/csv); default: download from HF")
    ap.add_argument("--out", default="rouge_by_category_out")
    ap.add_argument("--dump-meta", help="write HF metadata (no images) to this jsonl and exit")
    ap.add_argument("--extra", action="store_true", help="also report image_main_category and goal×task")
    args = ap.parse_args()

    meta = load_meta_local(args.meta) if args.meta else load_meta_from_hf()
    if args.dump_meta:
        meta.to_json(args.dump_meta, orient="records", lines=True, force_ascii=False)
        print(f"wrote {len(meta)} rows -> {args.dump_meta}")
        return
    if not args.predictions:
        ap.error("give at least one predictions file")

    os.makedirs(args.out, exist_ok=True)
    md_parts, comparison = [], {"main_goal_category": {}, "main_task_category": {}}

    for path in args.predictions:
        exp, df, raw = load_predictions(path)
        df = df.merge(meta.drop(columns=[c for c in ["index"] if c in meta.columns]),
                      on="row", how="left")

        missing = df[df["main_goal_category"].isna()]["image"].unique()
        if len(missing):
            print(f"[{exp}] WARNING: {len(missing)} images have no metadata, e.g. {missing[:3]}", file=sys.stderr)

        chk = check_join(df, meta)
        if chk is not None:
            bad = chk[~chk["ok"]]
            print(f"[{exp}] join check: {len(chk) - len(bad)}/{len(chk)} turn-1 questions match dataset rows")
            if len(bad):
                bad.to_csv(os.path.join(args.out, f"{exp}_join_mismatches.csv"), index=False)

        # sanity: our overall vs the file's own number
        print(f"[{exp}] overall ROUGE-L recomputed {df['rouge_l_f'].mean():.4f} "
              f"(file says {raw.get('avg_rouge_l_f')}, over all items it evaluated)")

        md_parts.append(f"## {exp}\n")
        groupings = ["main_goal_category", "main_task_category"]
        if args.extra:
            groupings.append("image_main_category")
        for by in groupings:
            tab = summarize(df.dropna(subset=[by]), by)
            tab.to_csv(os.path.join(args.out, f"{exp}_by_{by}.csv"))
            md_parts.append(fmt_md(tab, f"By {by}"))
            if by in comparison:
                comparison[by][exp] = tab["rouge_l_turn_mean"]

        if args.extra:
            ct = df.pivot_table(index="main_goal_category", columns="main_task_category",
                                values="rouge_l_f", aggfunc="mean")
            ct.to_csv(os.path.join(args.out, f"{exp}_goal_x_task.csv"))
            md_parts.append("### Goal × Task (ROUGE-L × 100)\n\n" + (ct * 100).round(2).to_markdown() + "\n")

        df.to_csv(os.path.join(args.out, f"{exp}_turn_level_joined.csv"), index=False)

    # side-by-side across experiments
    if len(args.predictions) > 1:
        for by, d in comparison.items():
            cmp_df = pd.DataFrame(d)
            cmp_df.to_csv(os.path.join(args.out, f"comparison_by_{by}.csv"))
            md_parts.insert(0, f"## Comparison by {by} (turn-level ROUGE-L × 100)\n\n"
                            + (cmp_df * 100).round(2).to_markdown() + "\n")

    report = "# ROUGE-L by MultiVerse category\n\n" + "\n".join(md_parts)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()