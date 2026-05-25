#!/bin/bash
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
# Generate rho=0.6 mixed+scored data (CPU only, no GPU needed)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

python scripts/prepare_rho06_data.py

echo ""
echo "Data files created:"
ls -lh data/mixed_data_rho06.jsonl data/scored_data_rho06.jsonl
echo ""
wc -l data/mixed_data_rho06.jsonl data/scored_data_rho06.jsonl
