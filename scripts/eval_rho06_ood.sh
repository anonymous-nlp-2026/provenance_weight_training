#!/bin/bash
# OOD eval (openwebtext + wikipedia PPL) for adaptive_rho06_tau07 (3 seeds)
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

OOD_DATASETS=("data/eval_openwebtext.jsonl" "data/eval_wikipedia.jsonl")
OOD_NAMES=("openwebtext" "wikipedia")

for SEED in "${SEEDS[@]}"; do
    EXP_ID="adaptive_rho06_tau07_seed${SEED}"
    FINAL_PATH="${MODEL_BASE}/${EXP_ID}/final"

    if [ ! -f "${FINAL_PATH}/model.safetensors" ]; then
        echo "[SKIP] ${EXP_ID}: no checkpoint at ${FINAL_PATH}"
        continue
    fi

    for i in "${!OOD_DATASETS[@]}"; do
        DATA_PATH="${OOD_DATASETS[$i]}"
        DATA_NAME="${OOD_NAMES[$i]}"

        echo "=========================================="
        echo "OOD eval: ${EXP_ID} on ${DATA_NAME}, GPU ${GPU_ID} at $(date)"
        echo "=========================================="

        python eval_perplexity.py \
            --checkpoints "$FINAL_PATH" \
            --data_path "$DATA_PATH" \
            --output_path "${EVAL_DIR}/${EXP_ID}_ood_${DATA_NAME}.json"

        echo "Completed ${EXP_ID} / ${DATA_NAME} at $(date)"
        echo ""
        echo "=== Result ==="
        cat "${EVAL_DIR}/${EXP_ID}_ood_${DATA_NAME}.json"
        echo ""
    done
done

echo "All OOD evals done at $(date)"
