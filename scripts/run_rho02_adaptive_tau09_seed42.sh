#!/bin/bash
# rho=0.2 adaptive tau=0.9 n_min=2
# CUDA_VISIBLE_DEVICES set by submit_training_job
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled

cd /root/provenance_weight_training
mkdir -p ~/runs/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data_rho02.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/rho02_adaptive_tau09_seed42/ \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.9 \
    --n_min 2 \
    --contamination_ratio 0.2 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model \
    2>&1 | tee ~/runs/provenance_weight_training/rho02_adaptive_tau09_seed42.log

echo "DONE: rho02_adaptive_tau09 seed42"
