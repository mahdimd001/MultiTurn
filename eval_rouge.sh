#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --job-name="eval_rouge_multi"
#SBATCH --mail-user=msamani@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# === Step 1: Load Required Modules ===
module load python/3.10

# === Step 2: Activate Your Environment ===
source /lustre/hdd/LAS/jannesar-lab/msamani/pythonenv_elasticcache/bin/activate

# === Step 3: Define Parameter Arrays ===
RATIOS=(0.5)
METHODS=(
    "turn_conditioned_layerwise_eviction"
    "turn_conditioned_coverage"
    "tgv_kv"
    "local_cache"
    "h2o"
    "elastic_cache"
    "meda"
    "pyramidkv"
)
DATASETS=("MultiVerse")

# === Step 4: Execute Nested Loops ===
for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" == "Convbench" ]; then
        DATA_PATH="./ConvBench/convbench.json"
        IMAGE_PATH="./ConvBench/visit_bench_images"
    else
        DATA_PATH="./MultiVerse_for_ppl/multiverse_ppl_full.json"
        IMAGE_PATH="./MultiVerse_raw/images"
    fi

    for RATIO in "${RATIOS[@]}"; do
        RATIO_INT=$(awk -v r="$RATIO" 'BEGIN {print r * 100}')

        for METHOD in "${METHODS[@]}"; do
            if [ "$METHOD" == "turn_conditioned_layerwise_eviction" ]; then
                ALLOCATION="sensitivity"
                SENS_PROFILE="./sensitivity_llava7b_r${RATIO_INT}.json"
            else
                ALLOCATION="turn"
                SENS_PROFILE="None"
            fi
            
            # Construct dynamic output directory 
            OUTPUT_DIR="./logs_rouge_llava/FULL/${DATASET}-${RATIO}-${METHOD}"

            echo "======================================================================"
            echo "Running: Dataset=$DATASET | Ratio=$RATIO | Method=$METHOD"
            echo "======================================================================"

            python /lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/eval_rouge_final2.py \
                --ratio "$RATIO" \
                --method "$METHOD" \
                --data-path "$DATA_PATH" \
                --image-path "$IMAGE_PATH" \
                --allocation-signal "$ALLOCATION" \
                --sensitivity-profile "$SENS_PROFILE" \
                --output-dir "$OUTPUT_DIR" 
        done
    done
done
