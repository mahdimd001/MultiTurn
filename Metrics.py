
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


METRIC_NAMES = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "CIDEr", "METEOR", "ROUGE-L"
]


def evaluate_group(records):
    """Compute corpus metrics and per-sample scores."""
    if not records:
        return None, []

    gts = {
        i: [{"caption": r["reference"]}]
        for i, r in enumerate(records)
    }
    res = {
        i: [{"caption": r["prediction"]}]
        for i, r in enumerate(records)
    }

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    summary = {"count": len(records)}
    individual = [
        {
            "conversation_id": r["conversation_id"],
            "turn": r["turn"],
            "image": r["image"],
        }
        for r in records
    ]

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Cider(), ["CIDEr"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
    ]

    for scorer, names in scorers:
        try:
            # Single-sample CIDEr has degenerate IDF statistics.
            if names == ["CIDEr"] and len(records) < 2:
                summary["CIDEr"] = None
                for row in individual:
                    row["CIDEr"] = None
                continue

            score, scores = scorer.compute_score(gts, res)

            if isinstance(score, (list, tuple)):
                for name, value, sample_scores in zip(
                    names, score, scores
                ):
                    summary[name] = float(value)
                    for i, v in enumerate(sample_scores):
                        individual[i][name] = float(v)
            else:
                summary[names[0]] = float(score)
                for i, v in enumerate(scores):
                    individual[i][names[0]] = float(v)

        finally:
            if isinstance(scorer, Meteor):
                # Close the Java subprocess.
                try:
                    scorer.meteor_p.stdin.close()
                    scorer.meteor_p.stdout.close()
                    scorer.meteor_p.wait(timeout=5)
                except Exception:
                    pass

    return summary, individual


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(input_path):
    input_path = Path(input_path).resolve()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    per_turn = defaultdict(list)

    for conversation_id, item in enumerate(data["results"]):
        image = item.get("image", "")

        for turn_data in item["turns"]:
            reference = turn_data.get("reference")
            prediction = turn_data.get("prediction")
            turn = int(turn_data["turn"])

            if not isinstance(reference, str) or not isinstance(
                prediction, str
            ):
                print(
                    f"Skipping conversation {conversation_id}, "
                    f"turn {turn}: missing text"
                )
                continue

            record = {
                "conversation_id": conversation_id,
                "image": image,
                "turn": turn,
                "reference": reference,
                "prediction": prediction,
            }

            records.append(record)
            per_turn[turn].append(record)

    if not records:
        raise ValueError("No valid prediction/reference pairs found.")

    print(f"Loaded {len(records)} turns from {len(data['results'])} conversations")

    # Evaluate all turns together.
    print("Computing overall metrics...")
    overall, individual = evaluate_group(records)

    # Evaluate each turn independently.
    turn_metrics = {}
    for turn in sorted(per_turn):
        print(f"Computing Turn {turn} metrics...")
        metrics, _ = evaluate_group(per_turn[turn])
        turn_metrics[str(turn)] = metrics

    output = {
        "experiment": data.get("experiment"),
        "method": data.get("method"),
        "ratio": data.get("ratio"),
        "num_conversations": len(data["results"]),
        "num_turns": len(records),
        "overall": overall,
        "per_turn": turn_metrics,
    }

    stem = input_path.stem
    output_dir = input_path.parent

    json_path = output_dir / f"{stem}_metrics.json"
    csv_path = output_dir / f"{stem}_metrics.csv"
    individual_path = output_dir / f"{stem}_individual_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    summary_rows = [
        {"group": "ALL", **overall}
    ]
    for turn, metrics in turn_metrics.items():
        summary_rows.append({
            "group": f"Turn {turn}",
            **metrics
        })

    save_csv(
        csv_path,
        summary_rows,
        ["group", "count"] + METRIC_NAMES
    )

    save_csv(
        individual_path,
        individual,
        ["conversation_id", "turn", "image"] + METRIC_NAMES
    )

    print("\n" + "=" * 75)
    print("EVALUATION RESULTS")
    print("=" * 75)

    header = f"{'Group':<12} {'Count':>6}"
    for name in METRIC_NAMES:
        header += f" {name:>9}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        line = f"{row['group']:<12} {row['count']:>6}"
        for name in METRIC_NAMES:
            value = row.get(name)
            line += (
                f" {value:>9.4f}"
                if value is not None
                else f" {'N/A':>9}"
            )
        print(line)

    print("\nSaved files:")
    print(json_path)
    print(csv_path)
    print(individual_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        default="/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/logs_rouge_llava/Convbench-0.5-elastic_cache/elastic_cache_ratio0.5_predictions.json",
        help="Path to predictions JSON file",

    )
    args = parser.parse_args()
    main(args.input)
