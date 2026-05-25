#!/bin/bash
# Batch eval for seed123 checkpoints missing eval results.
# b=2.0 and b=3.0 already done. This covers b=5.0 (add more below if needed).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_missing_seed123.sh

set -euo pipefail
cd /root/provenance_weight_training
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

EVAL_SCRIPT="eval_perplexity.py"
DATA_PATH="data/human/eval_holdout.jsonl"
OUTDIR="output/eval_results"

CHECKPOINTS=(
    # b=2.0 and b=3.0 eval already completed — uncomment to re-run if needed
    # "output/models/grid_b2_0_seed123/final"
    # "output/models/grid_b3_0_seed123/final"
    "output/models/grid_b5_0_seed123/final"
)

for CKPT in "${CHECKPOINTS[@]}"; do
    RUN_NAME=$(basename "$(dirname "$CKPT")")
    OUT_FILE="${OUTDIR}/${RUN_NAME}_eval.json"

    if [ -f "$OUT_FILE" ]; then
        echo "[SKIP] $OUT_FILE already exists"
        continue
    fi

    if [ ! -d "$CKPT" ]; then
        echo "[WARN] $CKPT not found, skipping (training may not be finished)"
        continue
    fi

    echo "[EVAL] $CKPT -> $OUT_FILE"
    python3 "$EVAL_SCRIPT" \
        --checkpoints "$CKPT" \
        --data_path "$DATA_PATH" \
        --max_docs 5000 \
        --output_path "$OUT_FILE"
done

echo "Done."
