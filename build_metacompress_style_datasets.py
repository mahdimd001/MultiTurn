#!/usr/bin/env python3
"""Reconstruct three-turn GQA/VQA-v2 dialogues in MultiVerse JSON format.

IMPORTANT: This script DOES NOT reproduce the unpublished MetaCompress annotation
files or question ordering. It groups three existing independent questions for
one image. These are synthetic multi-turn conversations, not grounded follow-ups.

Sources:
- GQA: https://huggingface.co/datasets/vikhyatk/gqa  (testdev_balanced)
- VQA-v2: https://visualqa.org/download.html  (validation questions/answers)
- COCO: https://cocodataset.org/#download  (val2014 images)

Dependencies: pip install -U datasets pillow requests
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

GQA_REPO = "vikhyatk/gqa"
VQA_Q_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
VQA_A_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_IMAGE_URL = "https://images.cocodataset.org/val2014/{filename}"


def log(msg):
    print(msg, flush=True)


def download(url: str, dest: Path) -> Path:
    """Download with a resumable .part file; existing complete files are reused."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return dest
    tmp = Path(str(dest) + ".part")
    for attempt in range(1, 5):
        position = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={position}-"} if position else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                resume = position and r.status_code == 206
                if position and not resume:
                    log(f"  Server does not support resume; restarting {dest.name}")
                mode = "ab" if resume else "wb"
                with tmp.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return dest
        except requests.RequestException as e:
            log(f"  Download attempt {attempt}/4 failed: {e}")
            if attempt == 4:
                raise
            time.sleep(2 * attempt)
    return dest


def json_member(zpath: Path, suffix: str):
    with zipfile.ZipFile(zpath) as z:
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {suffix!r} in {zpath}, got {matches}")
        with z.open(matches[0]) as f:
            return json.load(f)


def sample_three(candidates, rng):
    """Choose three distinct question texts, each with a nonempty answer."""
    unique = {}
    for item in candidates:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            unique.setdefault(" ".join(q.casefold().split()), item)
    if len(unique) < 3:
        return None
    return rng.sample(list(unique.values()), 3)


def save_json(data, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp.replace(target)
    log(f"  Saved {len(data):,} dialogues: {target}")


def save_pil_jpeg(pil, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size:
        return
    if isinstance(pil, dict):
        if pil.get("bytes"):
            pil = Image.open(io.BytesIO(pil["bytes"]))
        elif pil.get("path"):
            pil = Image.open(pil["path"])
        else:
            raise ValueError("Unknown Hugging Face image representation")
    if isinstance(pil, bytes):
        pil = Image.open(io.BytesIO(pil))
    if not isinstance(pil, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(pil)}")
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(out, format="JPEG", quality=95)


def build_gqa(root: Path, limit: int, seed: int):
    from datasets import load_dataset

    out = root / "MT_GQA_reconstructed"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    log(f"GQA: streaming {GQA_REPO!r} split=val_balanced")
    ds = load_dataset(GQA_REPO, split="val_balanced", streaming=True)
    records = []
    for row_idx, row in enumerate(ds):
        choices = []
        for i, qa in enumerate(row["qa"]):
            choices.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),  # short GQA answer for exact-match eval
                "question_id": qa.get("question_id", i),
            })
        selected = sample_three(choices, rng)
        if selected is None:
            continue
        name = f"gqa_{row_idx:06d}.jpg"
        save_pil_jpeg(row["image"], images / name)
        metadata = {
            "dataset": "GQA", "source_split": "val_balanced",
            "source_row_index": row_idx,
            "source_question_ids": [x["question_id"] for x in selected],
            "construction": "reconstructed_independent_questions",
            "seed": seed,
        }
        records.append({
            "image": name,
            "turns": [{"question": x["question"], "answer": x["answer"]} for x in selected],
            "metadata": metadata,
        })
        if len(records) % 500 == 0:
            log(f"  GQA saved {len(records):,} images")
        if limit and len(records) >= limit:
            break
    if not records:
        raise RuntimeError("No eligible GQA conversations produced; check dataset schema")
    save_json(records, out / "mt_gqa.json")
    if limit and len(records) < limit:
        log(f"  NOTE: only {len(records):,} valid GQA image groups exist in the available split (requested {limit:,}).")
    return out


def image_filename(image_id: int) -> str:
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def extract_selected_images(archive, images: Path, required):
    """Read only images used in JSON, without unpacking the full COCO archive."""
    missing = {name for name in required if not (images / name).exists()}
    if not missing:
        return
    log(f"  Extracting {len(missing):,} COCO images from archive")
    with zipfile.ZipFile(archive) as z:
        available = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
        unavailable = missing - available.keys()
        if unavailable:
            raise FileNotFoundError(f"Missing {len(unavailable)} expected members in COCO archive; sample: {list(unavailable)[:3]}")
        for index, name in enumerate(sorted(missing), 1):
            with z.open(available[name]) as src, (images / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if index % 2000 == 0:
                log(f"  Extracted {index:,}/{len(missing):,}")


def download_selected_images(images: Path, required, workers: int):
    missing = [name for name in required if not (images / name).exists()]
    if not missing:
        return
    log(f"  Downloading {len(missing):,} individual COCO images ({workers} workers)")
    def fetch(name):
        download(COCO_IMAGE_URL.format(filename=name), images / name)
        return name
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, name) for name in missing]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 500 == 0:
                log(f"  Downloaded {i:,}/{len(missing):,}")


def build_vqa(root: Path, limit: int, seed: int, image_mode: str,
              coco_zip: Path | None, coco_dir: Path | None,
              workers: int, keep_coco_zip: bool):
    out = root / "MT_VQA_v2_reconstructed"
    images = out / "images"
    cache = root / "_downloads"
    images.mkdir(parents=True, exist_ok=True)
    log("VQA-v2: downloading official validation questions and annotations")
    qzip = download(VQA_Q_URL, cache / "v2_Questions_Val_mscoco.zip")
    azip = download(VQA_A_URL, cache / "v2_Annotations_Val_mscoco.zip")
    questions = json_member(qzip, "v2_OpenEnded_mscoco_val2014_questions.json")["questions"]
    annotations = json_member(azip, "v2_mscoco_val2014_annotations.json")["annotations"]
    ans_map = {int(a["question_id"]): a for a in annotations}
    grouped = defaultdict(list)
    for q in questions:
        a = ans_map.get(int(q["question_id"]))
        if a is None:
            continue
        grouped[int(q["image_id"])].append({
            "question": q["question"],
            "answer": a.get("multiple_choice_answer", ""),
            "question_id": int(q["question_id"]),
        })
    rng = random.Random(seed)
    image_ids = sorted(grouped)
    rng.shuffle(image_ids)
    records = []
    for image_id in image_ids:
        selected = sample_three(grouped[image_id], rng)
        if selected is None:
            continue
        name = image_filename(image_id)
        records.append({
            "image": name,
            "turns": [{"question": t["question"], "answer": t["answer"]} for t in selected],
            "metadata": {
                "dataset": "VQA-v2", "source_split": "val2014",
                "source_image_id": image_id,
                "source_question_ids": [t["question_id"] for t in selected],
                "construction": "reconstructed_independent_questions",
                "seed": seed,
            },
        })
        if limit and len(records) >= limit:
            break
    if not records:
        raise RuntimeError("No eligible VQA image groups produced")
    if limit and len(records) < limit:
        log(f"  NOTE: {len(records):,} valid images found (requested {limit:,}).")
    required = {rec["image"] for rec in records}
    if coco_dir is not None:
        log(f"  Reusing existing COCO images from {coco_dir}")
        for name in sorted(required):
            if (images / name).exists():
                continue
            src = coco_dir / name
            if not src.is_file():
                src = coco_dir / "val2014" / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing source COCO image: {name} under {coco_dir}")
            # Hard-link where possible to avoid duplicating gigabytes.
            try:
                (images / name).hardlink_to(src)
            except OSError:
                shutil.copy2(src, images / name)
    elif image_mode == "individual":
        download_selected_images(images, required, workers)
    else:
        owned_zip = coco_zip is None
        archive = coco_zip if coco_zip else cache / "val2014.zip"
        if any(not (images / name).is_file() for name in required):
            log("  COCO val2014 archive is several GB; ensure enough disk space")
            if not archive.is_file():
                download(COCO_ZIP_URL, archive)
            extract_selected_images(archive, images, required)
        if owned_zip and not keep_coco_zip and archive.exists():
            archive.unlink()  # only delete the file this script owns
    missing = [rec["image"] for rec in records if not (images / rec["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")
    save_json(records, out / "mt_vqa_v2.json")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["all", "gqa", "vqa"], default="all")
    p.add_argument("--output", type=Path, default=Path("./metacompress_data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-gqa", type=int, default=4061, help="0 = all eligible images")
    p.add_argument("--limit-vqa", type=int, default=25000, help="0 = all eligible images")
    p.add_argument("--vqa-image-mode", choices=["archive", "individual"], default="archive")
    p.add_argument("--coco-zip", type=Path, default=None, help="Optional existing val2014.zip")
    p.add_argument("--coco-dir", type=Path, default=None, help="Optional existing directory of COCO val2014 JPEGs")
    p.add_argument("--workers", type=int, default=8, help="Used in individual-image download mode")
    p.add_argument("--keep-coco-zip", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("all", "gqa"):
        build_gqa(args.output, args.limit_gqa, args.seed)
    if args.dataset in ("all", "vqa"):
        build_vqa(args.output, args.limit_vqa, args.seed, args.vqa_image_mode,
                  args.coco_zip, args.coco_dir, args.workers, args.keep_coco_zip)
    log("\nDone. Every JSON record refers to an image in its dataset's images/ folder.")
    log("WARNING: These are reconstructed same-image question sequences, NOT MetaCompress's unpublished dialogues.")


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
