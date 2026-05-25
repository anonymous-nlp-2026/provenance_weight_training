# Provenance Weight Training

Code for the paper "When Does Detect-and-Reweight Work? Epoch Confounds and Adaptive Reweighting in Continued LM Pretraining"

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Prepare training data with contamination scores in JSONL format
2. Each line should contain `text` and `q_score` fields
3. Example: `{"text": "...", "q_score": 0.85, "label": "human"}`

## Running Experiments

### Adaptive Reweighting Training (Ours)

```bash
python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/adaptive_seed42 \
    --model_name Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.7 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --bf16 \
    --seed 42
```

### Grid Search Baseline (Drayson et al.)

```bash
python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/grid_b1.0_seed42 \
    --model_name Qwen/Qwen3-0.6B \
    --weighting_method grid \
    --grid_b_value 1.0 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --bf16 \
    --seed 42
```

### Uniform Baseline

```bash
python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/uniform_seed42 \
    --model_name Qwen/Qwen3-0.6B \
    --weighting_method uniform \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --bf16 \
    --seed 42
```

### Evaluation

```bash
python evaluation/eval_pipeline.py \
    --model_path output/adaptive_seed42/final \
    --output_dir output/eval_results \
    --run_perplexity true \
    --run_lm_eval true \
    --lm_eval_tasks hellaswag,mmlu
```

## Weighting Methods

| Method | Description |
|--------|-------------|
| `uniform` | All samples equally weighted (baseline) |
| `grid` | Fixed b value per run; grid search over {0.5, 1.0, 1.5, 2.0, 3.0, 5.0} |
| `adaptive` | Batch-adaptive b* via alpha_eff convergence theory (ours) |
| `golden_ratio` | Oracle baseline |
| `random` | Random reweighting (ablation control) |

## License

MIT
