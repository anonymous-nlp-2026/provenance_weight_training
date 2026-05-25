#!/bin/bash
# Downstream eval (lm_eval) for adaptive_rho06_tau07 (3 seeds)
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
    echo "Downstream eval: ${EXP_ID} on GPU ${GPU_ID} at $(date)"
    echo "=========================================="

    python -m lm_eval \
        --model hf \
        --model_args pretrained=${FINAL_PATH},dtype=float16 \
        --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path "${EVAL_DIR}/${EXP_ID}_downstream/"

    echo "Completed ${EXP_ID} at $(date)"
    echo ""
done

echo "All downstream evals done at $(date)"
