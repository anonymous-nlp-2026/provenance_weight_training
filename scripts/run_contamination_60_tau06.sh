#!/bin/bash
# p0_contamination_60_tau06: rho=0.6, tau=0.6 (lower threshold)
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data_60.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/contamination_60_tau06_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.6 \
    --n_min 2 \
    --contamination_ratio 0.6 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name contamination_60_tau06_seed42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: contamination_60_tau06 seed42"
