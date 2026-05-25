#!/bin/bash
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

for b in 0.1 0.2 0.3; do
    echo "=== Training grid b=$b ==="
    WANDB_MODE=disabled python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir output/models/grid_b_low_b${b}_seed42 \
        --weighting_method grid \
        --grid_b_value $b \
        --model_name models/Qwen/Qwen3-0.6B \
        --num_train_tokens 200000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 2048 \
        --learning_rate 5e-5 \
        --seed 42 \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 2 \
        --save_only_model \
        --contamination_ratio 0.4 \
        --bf16
done
echo "=== All 3 b-values complete ==="
