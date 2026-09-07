"""
Run this script on your cluster to patch modeling_qwen.py so that the
attention mask addition is guarded against shape mismatches caused by
Qwen-VL's internal image-token expansion.

Usage:
    python patch_modeling_qwen.py

It will find modeling_qwen.py automatically, back it up, and apply the patch.
"""

import glob
import os
import shutil

# ── 1. Find the file ────────────────────────────────────────────────────────

search_roots = [
    os.path.expanduser("~"),
    "/tmp",
    "/var",
    "/lustre",
]

candidates = []
for root in search_roots:
    candidates += glob.glob(
        os.path.join(root, "**", "transformers_modules", "**", "modeling_qwen.py"),
        recursive=True,
    )
    candidates += glob.glob(
        os.path.join(root, "**", "Qwen-VL", "modeling_qwen.py"),
        recursive=True,
    )

if not candidates:
    raise FileNotFoundError(
        "Could not find modeling_qwen.py. "
        "Set MODELING_QWEN_PATH env var to its absolute path and re-run."
    )

path = os.environ.get("MODELING_QWEN_PATH", candidates[0])
print(f"Patching: {path}")

# ── 2. Back up ───────────────────────────────────────────────────────────────

backup = path + ".bak"
if not os.path.exists(backup):
    shutil.copy2(path, backup)
    print(f"Backup:   {backup}")
else:
    print(f"Backup already exists, skipping: {backup}")

# ── 3. Read source ───────────────────────────────────────────────────────────

with open(path, "r") as f:
    src = f.read()

# ── 4. Define patch ──────────────────────────────────────────────────────────

# Original line inside QWenAttention._attn:
OLD = "attn_weights = attn_weights + attention_mask"

# Replacement: guard against None and shape mismatch from image-token expansion.
NEW = (
    "if attention_mask is not None and attention_mask.shape[-1] == attn_weights.shape[-1]:\n"
    "                attn_weights = attn_weights + attention_mask"
)

# ── 5. Apply ─────────────────────────────────────────────────────────────────

if OLD not in src:
    raise RuntimeError(
        f"Could not find the target line in {path}.\n"
        "The model source may differ from the expected version.\n"
        "Please apply the patch manually (see instructions below).\n\n"
        "Find the line:\n"
        f"    {OLD}\n"
        "And replace it with:\n"
        f"    {NEW}\n"
    )

count = src.count(OLD)
if count > 1:
    print(f"WARNING: Found {count} occurrences of the target line; patching all.")

patched = src.replace(OLD, NEW)

with open(path, "w") as f:
    f.write(patched)

print(f"Patch applied ({count} occurrence(s) replaced).")
print("Done. You can now run eval_ppl_qwen.py normally.")
