#!/bin/bash
# Holdout PPL eval for adaptive_rho06_tau07 (3 seeds)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

GPU_ID="${1:--1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME=~/.cache/huggingface

cd /root/provenance_weight_training

SEEDS=(42 123 456)
MODEL_BASE="output/models"
EVAL_DIR="output/eval_results"
mkdir -p "$EVAL_DIR"

for SEED in "${SEEDS[@]}"; do
    EXP_ID="adaptive_rho06_tau07_seed${SEED}"
    FINAL_PATH="${MODEL_BASE}/${EXP_ID}/final"

    if [ ! -f "${FINAL_PATH}/model.safetensors" ]; then
        echo "[SKIP] ${EXP_ID}: no checkpoint at ${FINAL_PATH}"
        continue
    fi

    echo "=========================================="
    echo "PPL eval: ${EXP_ID} on GPU ${GPU_ID} at $(date)"
    echo "=========================================="

    python eval_perplexity.py \
        --checkpoints "$FINAL_PATH" \
        --data_path data/human/eval_holdout.jsonl \
        --output_path "${EVAL_DIR}/${EXP_ID}_eval.json"

    echo "Completed ${EXP_ID} at $(date)"
    echo ""
    echo "=== Result ==="
    cat "${EVAL_DIR}/${EXP_ID}_eval.json"
    echo ""
done

echo "All PPL evals done at $(date)"
