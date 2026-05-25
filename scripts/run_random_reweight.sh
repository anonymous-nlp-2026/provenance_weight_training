#!/bin/bash
# p1_baseline_random_reweight: Random reweighting baseline
# Same as adaptive but q_scores shuffled per batch — proves detector signal matters
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/random_reweight_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --shuffle_scores \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name random_reweight_seed42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: random_reweight seed42"
