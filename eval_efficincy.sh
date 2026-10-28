#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --job-name="rouge_multiverse"
#SBATCH --mail-user=msamani@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# === Step 1: Load Required Modules ===
module load python/3.10

# === Step 2: Activate Your Environment ===
source /lustre/hdd/LAS/jannesar-lab/msamani/pythonenv_elasticcache/bin/activate

echo "======================================================================"
echo "Running: Ratios=0.9 | Context Lengths=native"
python eval_efficiency.py --methods full local_cache h2o elastic_cache meda pyramidkv turncast_kv  tgv_kv  --ratios 0.9 --context-lengt
hs native --gen-tokens 512 --num-turns 3 --turncast-scorer turncast_data/turncast_scorer.pt

echo "Running: Ratios=0.9 | Context Lengths=2048"
python eval_efficiency.py --methods full local_cache h2o elastic_cache meda pyramidkv turncast_kv  tgv_kv  --ratios 0.9 --context-lengt
hs 2048 --gen-tokens 512 --num-turns 3 --turncast-scorer turncast_data/turncast_scorer.pt

echo "Running: Ratios=0.9 | Context Lengths=4096"
python eval_efficiency.py --methods full local_cache h2o elastic_cache meda pyramidkv turncast_kv  tgv_kv  --ratios 0.9 --context-lengt
hs 4096 --gen-tokens 512 --num-turns 3 --turncast-scorer turncast_data/turncast_scorer.pt

echo "Running: Ratios=0.9 | Context Lengths=8192"
python eval_efficiency.py --methods full local_cache h2o elastic_cache meda pyramidkv turncast_kv  tgv_kv  --ratios 0.9 --context-lengt
hs 8192 --gen-tokens 512 --num-turns 3 --turncast-scorer turncast_data/turncast_scorer.pt