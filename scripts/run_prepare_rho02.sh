#!/bin/bash
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training
python scripts/prepare_rho02_data.py
echo "DONE: scored_data_rho02.jsonl generated"
