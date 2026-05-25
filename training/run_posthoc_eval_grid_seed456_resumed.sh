#!/bin/bash
# Post-hoc eval: holdout perplexity for grid b=2.0/3.0/5.0 seed456 (resumed, no inline eval)
# Input: final/ checkpoint dirs from grid_b{2_0,3_0,5_0}_seed456
# Output: output/eval/posthoc_eval_grid_seed456_resumed.json
# Depends: eval_perplexity.py, data/human/eval_holdout.jsonl

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

MODEL_DIR="output/models"
EVAL_DATA="data/human/eval_holdout.jsonl"
OUTPUT="output/eval/posthoc_eval_grid_seed456_resumed.json"
MAX_DOCS="${1:-5000}"

mkdir -p output/eval

CHECKPOINTS=()
MISSING=()
for b in 2_0 3_0 5_0; do
    ckpt="${MODEL_DIR}/grid_b${b}_seed456/final"
    if [ -d "$ckpt" ]; then
        CHECKPOINTS+=("$ckpt")
    else
        MISSING+=("grid_b${b}_seed456")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "WARNING: Missing final/ dirs for: ${MISSING[*]}"
    echo "These will be skipped. Re-run after training completes."
fi

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found."
    exit 1
fi

echo "Evaluating ${#CHECKPOINTS[@]} checkpoints with max_docs=${MAX_DOCS}"
echo "Output: ${OUTPUT}"
echo ""

python eval_perplexity.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --data_path "${EVAL_DATA}" \
    --max_docs "${MAX_DOCS}" \
    --output_path "${OUTPUT}"

echo ""
echo "Done. Results in ${OUTPUT}"
