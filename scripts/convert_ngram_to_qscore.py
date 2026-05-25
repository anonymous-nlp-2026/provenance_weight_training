import json
import numpy as np
from pathlib import Path

INPUT = "data/scored_data_ngram.jsonl"
OUTPUT_MAIN = "/root/provenance_weight_training/data/scored_data_ngram_qscore.jsonl"
OUTPUT_TMP = "data/scored_data_ngram_qscore.jsonl"

print("Loading data...")
records = []
ngram_ppls = []
with open(INPUT) as f:
    for line in f:
        rec = json.loads(line)
        records.append(rec)
        ngram_ppls.append(rec["ngram_ppl"])

ngram_ppls = np.array(ngram_ppls)
print(f"Loaded {len(records)} records")
print(f"ngram_ppl stats: min={ngram_ppls.min():.4f}, max={ngram_ppls.max():.4f}, "
      f"mean={ngram_ppls.mean():.4f}, median={np.median(ngram_ppls):.4f}")

# Percentile-based: rank each value, normalize to [0, 1]
# argsort twice gives rank; use average ranking for ties via scipy
from scipy.stats import rankdata
ranks = rankdata(ngram_ppls, method="average")
# percentile_rank in [0, 1]: fraction of values <= this value
percentile_ranks = (ranks - 1) / (len(ranks) - 1)
# Invert: lower ngram_ppl (more synthetic) -> higher q_score
q_scores = 1.0 - percentile_ranks

print(f"q_score stats: min={q_scores.min():.4f}, max={q_scores.max():.4f}, "
      f"mean={q_scores.mean():.4f}, median={np.median(q_scores):.4f}")

# Write output
print("Writing output...")
for out_path in [OUTPUT_MAIN, OUTPUT_TMP]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, rec in enumerate(records):
            rec["q_score"] = float(q_scores[i])
            # Keep ngram_ppl as-is
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written: {out_path}")

# Validation
depths = np.array([r["depth"] for r in records])
syn_mask = depths == 1
real_mask = depths == 0
print(f"\nValidation:")
print(f"  Synthetic (depth=1): n={syn_mask.sum()}, mean q_score={q_scores[syn_mask].mean():.4f}, "
      f"mean ngram_ppl={ngram_ppls[syn_mask].mean():.4f}")
print(f"  Real (depth=0):      n={real_mask.sum()}, mean q_score={q_scores[real_mask].mean():.4f}, "
      f"mean ngram_ppl={ngram_ppls[real_mask].mean():.4f}")
print(f"  Synthetic q_score > Real q_score: {q_scores[syn_mask].mean() > q_scores[real_mask].mean()}")
