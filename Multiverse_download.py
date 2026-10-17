"""
Download all images from the Hugging Face dataset passing2961/MultiVerse
and save them as multiverse_000000.png, multiverse_000001.png, ...

Requirements:
    pip install datasets pillow
"""

import csv
import os

from datasets import load_dataset

OUTPUT_DIR = "/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/MultiVerse_raw/images"
REPO_ID = "passing2961/MultiVerse"

# PNG can store these modes directly; anything else (e.g. CMYK, YCbCr) gets converted.
PNG_SAFE_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The dataset has a single "train" split (647 rows) with one image per row.
    ds = load_dataset(REPO_ID, split="train")
    print(f"Loaded {len(ds)} rows")

    mapping_path = os.path.join(OUTPUT_DIR, "mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "row", "index"])

        for i, example in enumerate(ds):
            img = example["image"]  # PIL.Image
            if img.mode not in PNG_SAFE_MODES:
                img = img.convert("RGBA" if "A" in img.mode else "RGB")

            filename = f"multiverse_{i:06d}.png"
            img.save(os.path.join(OUTPUT_DIR, filename), format="PNG")
            writer.writerow([filename, i, example["index"]])

            if (i + 1) % 50 == 0 or i + 1 == len(ds):
                print(f"Saved {i + 1}/{len(ds)}")

    print(f"Done. Images saved to '{OUTPUT_DIR}/' (row-to-file map in mapping.csv)")


if __name__ == "__main__":
    main()