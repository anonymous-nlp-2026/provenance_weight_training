#!/bin/bash
# threshold_q03_seed42: Train on samples with q_score <= 0.3
# More aggressive filtering than detect_remove_baseline (q<=0.5)
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

FILTERED=data/scored_data_filtered_q03.jsonl
if [ ! -f "$FILTERED" ]; then
    echo "Generating filtered data (removing q_score > 0.3)..."
    python scripts/filter_high_qscore.py \
        --input data/scored_data.jsonl \
        --output "$FILTERED" \
        --threshold 0.3
fi

python training/pretrain_weighted.py \
    --data_path "$FILTERED" \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/threshold_q03_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method uniform \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name threshold_q03_seed42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: threshold_q03_seed42"
