#!/bin/bash
# ESS-only weighting (ablation: no alpha_eff constraint, only ESS stability bound)
# Cross-seed experiment for statistical significance (Claim 1)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled

cd /root/provenance_weight_training


SEED=123

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/ess_only_seed${SEED} \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method ess_only \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --n_min 2 \
    --seed ${SEED} \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: ess_only seed${SEED}"
