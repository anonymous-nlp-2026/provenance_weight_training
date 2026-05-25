#!/bin/bash
set -e

cd /root/provenance_weight_training

# Network proxy
export no_proxy=localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn
export http_proxy=http://10.37.1.23:12798
export https_proxy=http://10.37.1.23:12798
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# Torch / HF
export TORCHDYNAMO_DISABLE=1
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/.cache/huggingface

echo "=== Starting detector training ==="
echo "Time: $(date)"
echo "Python: $(python --version)"

python detector/train_detector.py \
    --human_data_dir data/detector_input/human \
    --synthetic_data_dir data/detector_input/synthetic \
    --model_name answerdotai/ModernBERT-base \
    --output_dir output/detector \
    --max_length 512 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --warmup_ratio 0.1 \
    --eval_steps 500 \
    --seed 42

echo "=== Training finished ==="
echo "Time: $(date)"
