#!/bin/bash
# Post-hoc eval: hybrid_rho04_seed42
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

GPU_ID="${1:--1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME=~/.cache/huggingface

cd /root/provenance_weight_training

EXP_ID="hybrid_rho04_seed42"
FINAL_PATH="output/models/${EXP_ID}/final"
EVAL_DIR="output/eval_results"
mkdir -p "$EVAL_DIR"

if [ ! -f "${FINAL_PATH}/model.safetensors" ]; then
    echo "[ERROR] ${EXP_ID}: final checkpoint not ready (no ${FINAL_PATH}/model.safetensors)"
    echo "Training may still be running. Exiting."
    exit 1
fi

echo "=========================================="
echo "Evaluating ${EXP_ID} on GPU ${GPU_ID} at $(date)"
echo "=========================================="

python eval_perplexity.py \
    --checkpoints "$FINAL_PATH" \
    --data_path data/human/eval_holdout.jsonl \
    --output_path "${EVAL_DIR}/${EXP_ID}_eval.json"

echo "Completed ${EXP_ID} at $(date)"
echo ""
echo "=== Result ==="
cat "${EVAL_DIR}/${EXP_ID}_eval.json"
