#!/bin/bash
# submit_adaptive_v2.sh — Launch adaptive_v2 experiments
# Usage: bash submit_adaptive_v2.sh <gpu_id> <seed|all>
# Example: bash submit_adaptive_v2.sh 1 42
#          bash submit_adaptive_v2.sh 1 all

set -euo pipefail

GPU_ID="${1:?Usage: bash submit_adaptive_v2.sh <gpu_id> <seed|all>}"
SEED_ARG="${2:?Usage: bash submit_adaptive_v2.sh <gpu_id> <seed|all>}"

PROJECT_DIR="/root/provenance_weight_training"
cd "$PROJECT_DIR"

SEEDS=(42 123 456)

run_experiment() {
    local seed=$1
    local exp_id="adaptive_v2_seed${seed}"
    local output_dir="output/models/${exp_id}"

    echo "=== Starting ${exp_id} on GPU ${GPU_ID} ==="
    echo "  output_dir: ${output_dir}"
    echo "  seed: ${seed}, n_min: 1, ema_alpha: 0.1"

    source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir "${output_dir}" \
        --weighting_method adaptive \
        --model_name models/Qwen/Qwen3-0.6B \
        --num_train_tokens 200000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 2048 \
        --learning_rate 5e-5 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --save_total_limit 2 \
        --save_only_model \
        --contamination_ratio 0.4 \
        --n_min 1 \
        --ema_alpha 0.1 \
        --seed "${seed}" \
        --use_wandb \
        --wandb_run_name "${exp_id}" \
        --gradient_checkpointing

    echo "=== Finished ${exp_id} ==="
    echo ""
}

if [ "$SEED_ARG" = "all" ]; then
    for seed in "${SEEDS[@]}"; do
        run_experiment "$seed"
    done
    echo "All adaptive_v2 experiments complete."
else
    run_experiment "$SEED_ARG"
fi
