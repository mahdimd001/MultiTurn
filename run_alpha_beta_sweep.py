#!/usr/bin/env python3
"""
Alpha/Beta sweep driver for MultimodalHelasticCache.

Fixes attn_weight (alpha) = 1.0 and sweeps conn_weight (beta) across several
values, at multiple compression ratios. Runs eval_ppl_sweep.py once per
(beta, ratio) combination, parses the PPL each run writes to its log file,
and collects everything into:

  - sweep_results/results.csv   (long format: one row per run -- easy for pandas/plotting)
  - sweep_results/results.json  (nested: results[ratio][beta] = ppl -- easy for quick lookups)
  - sweep_results/pivot.csv     (wide format: rows=beta, cols=ratio -- drops straight into a table)

Edit the CONFIG block below, then run:  python run_alpha_beta_sweep.py
"""

import os
import csv
import json
import subprocess

# ============================ CONFIG ============================
# --- what to sweep ---
ATTN_WEIGHT = 1.0                                                      # alpha, fixed
BETAS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]       # conn_weight values
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]            # compression ratios (higher = MORE compression)

# --- fixed eval args (EDIT THESE to your paths) ---
EVAL_SCRIPT = "eval_ppl_sweep.py"                   # the single-turn sweep eval
MODEL_PATH  = "/lustre/hdd/LAS/jannesar-lab/tmahmud/models/llava-v1.5-13b"
DATA_PATH   = "./mm-vet/mm-vet.json"             # <-- held-out dev split, NOT your final eval set
IMAGE_PATH  = "./mm-vet/images"
EVAL_SAMPLES = 218
METHOD = "elastic"
SEGREGATE_MERGE = True                              # pass --segregate-merge or not

# --- where results go ---
OUT_DIR = "sweep_results"
LOG_DIR = "logs_ppl_llava"                          # where eval_ppl_sweep.py writes its PPL txt files
# ===============================================================


def exp_name(beta, ratio):
    # unique, filesystem-safe name per run; matches the .txt the eval writes
    return f"sweep_attn{ATTN_WEIGHT}_beta{beta}_ratio{ratio}"


def read_ppl(name):
    """Read the LAST line of the eval's log file (each run appends one PPL)."""
    path = os.path.join(LOG_DIR, f"{name}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1])   # last line = this run's result
    except ValueError:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    rows = []                       # long-format rows for CSV
    nested = {}                     # nested[ratio][beta] = ppl

    total = len(BETAS) * len(RATIOS)
    done = 0

    for ratio in RATIOS:
        nested[ratio] = {}
        for beta in BETAS:
            done += 1
            name = exp_name(beta, ratio)
            print(f"\n[{done}/{total}] running: attn={ATTN_WEIGHT}, beta={beta}, ratio={ratio}")

            cmd = [
                "python", EVAL_SCRIPT,
                "--model-path", MODEL_PATH,
                "--data-path", DATA_PATH,
                "--image-path", IMAGE_PATH,
                "--eval-samples", str(EVAL_SAMPLES),
                "--method", METHOD,
                "--ratio", str(ratio),
                "--attn-weight", str(ATTN_WEIGHT),
                "--conn-weight", str(beta),
                "--exp-name", name,
            ]
            if SEGREGATE_MERGE:
                cmd.append("--segregate-merge")

            # run and stream output; capture failure without killing the whole sweep
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  !! run FAILED (returncode {result.returncode}) -- recording ppl=None and continuing")
                ppl = None
            else:
                ppl = read_ppl(name)
                print(f"  -> ppl = {ppl}")

            nested[ratio][beta] = ppl
            rows.append({
                "attn_weight": ATTN_WEIGHT,
                "conn_weight": beta,
                "ratio": ratio,
                "ppl": ppl,
                "exp_name": name,
            })

            # write incrementally after EVERY run, so a crash mid-sweep
            # never loses results already collected.
            _write_outputs(rows, nested)

    print(f"\nDone. Results in ./{OUT_DIR}/")
    print(f"  - results.csv  (long format, one row per run)")
    print(f"  - pivot.csv    (rows=beta, cols=ratio -- for tables)")
    print(f"  - results.json (nested lookup)")


def _write_outputs(rows, nested):
    # long-format CSV
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attn_weight", "conn_weight", "ratio", "ppl", "exp_name"])
        w.writeheader()
        w.writerows(rows)

    # nested JSON
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(nested, f, indent=2)

    # wide pivot CSV: rows = beta, columns = ratio
    with open(os.path.join(OUT_DIR, "pivot.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["conn_weight \\ ratio"] + [str(r) for r in RATIOS]
        w.writerow(header)
        for beta in BETAS:
            row = [beta] + [nested.get(r, {}).get(beta, "") for r in RATIOS]
            w.writerow(row)


if __name__ == "__main__":
    main()