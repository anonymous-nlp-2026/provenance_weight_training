#!/bin/bash
# Grid-b seed456 提交脚本
# 用法:
#   bash scripts/submit_grid_b_seed456.sh <GPU_ID> <b_value|all>
#
# 示例:
#   bash scripts/submit_grid_b_seed456.sh 2 0.5    # GPU 2 上跑 b=0.5
#   bash scripts/submit_grid_b_seed456.sh 1 all    # GPU 1 上串行跑全部 6 个 b 值
#   bash scripts/submit_grid_b_seed456.sh 0 1.5    # GPU 0 上跑 b=1.5
#
# b 值范围: 0.5 1.0 1.5 2.0 3.0 5.0

set -e

if [ $# -ne 2 ]; then
    echo "用法: bash $0 <GPU_ID> <b_value|all>"
    echo "示例: bash $0 2 0.5"
    echo "      bash $0 1 all"
    exit 1
fi

GPU_ID=$1
B_ARG=$2
ALL_B_VALUES=(0.5 1.0 1.5 2.0 3.0 5.0)

# 验证 GPU ID
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "错误: GPU_ID 必须是非负整数，收到: $GPU_ID"
    exit 1
fi

# 确定要跑的 b 值列表
if [ "$B_ARG" = "all" ]; then
    B_VALUES=("${ALL_B_VALUES[@]}")
else
    # 验证 b 值合法
    VALID=false
    for v in "${ALL_B_VALUES[@]}"; do
        if [ "$B_ARG" = "$v" ]; then
            VALID=true
            break
        fi
    done
    if [ "$VALID" = false ]; then
        echo "错误: b 值必须是 ${ALL_B_VALUES[*]} 之一，或 'all'，收到: $B_ARG"
        exit 1
    fi
    B_VALUES=("$B_ARG")
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=$GPU_ID
cd /root/provenance_weight_training

echo "=========================================="
echo "GPU: $GPU_ID | seed: 456 | b values: ${B_VALUES[*]}"
echo "开始时间: $(date)"
echo "=========================================="

for b in "${B_VALUES[@]}"; do
    b_dir=$(echo $b | tr '.' '_')
    exp_id="grid_b${b_dir}_seed456"

    echo ""
    echo "=========================================="
    echo "Starting ${exp_id} (b=${b}) at $(date)"
    echo "=========================================="

    python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir output/models/${exp_id} \
        --weighting_method grid \
        --grid_b_value $b \
        --model_name models/Qwen/Qwen3-0.6B \
        --num_train_tokens 200000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 2048 \
        --learning_rate 5e-5 \
        --seed 456 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --save_total_limit 2 \
        --save_only_model \
        --contamination_ratio 0.4

    echo "Completed ${exp_id} at $(date)"
done

echo ""
echo "=========================================="
echo "All requested grid-b seed456 runs complete at $(date)"
echo "=========================================="
