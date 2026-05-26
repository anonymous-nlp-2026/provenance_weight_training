# When Does Detect-and-Reweight Work for Synthetic Contamination?

Code for anonymous EMNLP 2026 submission.

## Abstract

As synthetic text pervades web corpora, detect-and-reweight methods that downweight likely-synthetic samples during training have emerged to prevent model collapse, yet when they actually help remains unclear. We investigate this question through controlled continued-pretraining experiments (Qwen3-0.6B, 132M tokens, 5 seeds) and identify three effectiveness conditions. First, detect-and-remove (D&R) outperforms all reweighting methods in our setting; epoch-matched controls attribute over 95% of the advantage to multi-epoch exposure on filtered data, identifying the mechanism behind D&R's effectiveness. Second, the detector must provide both sufficient discrimination and good calibration; high-AUC detectors with poor calibration degrade performance below uniform training. Third, minimum intervention, applying the smallest reweighting exponent that meets a quality floor, outperforms unconstrained weight maximization. As an illustrative application of the minimum-intervention principle, we derive a dual-constrained adaptive exponent b\* that improves over uniform training at contamination ratios rho >= 0.4, reaching conventional significance after multiple-comparison correction, while requiring no per-model tuning. These findings apply within our self-contamination setup at 0.6B scale; generalization to cross-model contamination and larger models remains to be established.

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
