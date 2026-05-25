#!/bin/bash
# p0_batchsize_16: batch_size=16 generalization test, n_min=8 (coupling rule: floor(B/2))
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/batchsize16_nmin8_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.8 \
    --n_min 8 \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: batchsize16 seed42"
