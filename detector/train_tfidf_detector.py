"""
TF-IDF + Logistic Regression synthetic text detector.

Input:
  - data/detector_input/human/detector.jsonl  (label=0, human text)
  - data/detector_input/synthetic/synthetic.jsonl  (label=1, synthetic text)

Output (to output/detector_tfidf/):
  - tfidf_vectorizer.joblib, lr_model.joblib  — trained model
  - calibration_results.json  — AUC, ECE, accuracy, q-score stats, temperature T
  - predictions.jsonl  — val set predictions (text_preview, true_label, q_score)

Key params: TF-IDF bigrams, max_features=50000, sublinear_tf=True,
            LogisticRegression C=1.0 lbfgs, temperature scaling if ECE>0.05.
If AUC>0.99 and q-scores are binary, applies degradation strategies to
get AUC in 0.85-0.98 with smooth q-score distribution.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize_scalar
import joblib

PROJECT = Path("/root/provenance_weight_training")
HUMAN_PATH = PROJECT / "data/detector_input/human/detector.jsonl"
SYNTH_PATH = PROJECT / "data/detector_input/synthetic/synthetic.jsonl"
OUTPUT_DIR = PROJECT / "output/detector_tfidf"
SEED = 42


def load_data():
    texts, labels = [], []
    with open(HUMAN_PATH) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(0)
    with open(SYNTH_PATH) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(1)
    return texts, np.array(labels)


def truncate_texts(texts, max_words):
    return [" ".join(t.split()[:max_words]) for t in texts]


def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_conf - bin_acc)
    return ece


def qscore_stats(probs):
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(probs, bins=bins)
    mid_ratio = np.mean((probs > 0.1) & (probs < 0.9))
    stats = {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "median": float(np.median(probs)),
        "mid_range_ratio": float(mid_ratio),
        "histogram_bins": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)],
        "histogram_counts": hist.tolist(),
    }
    return stats


def temperature_scale(logits, labels):
    def nll(T):
        scaled = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return res.x


def run_experiment(config):
    label = config.get("label", "default")
    max_features = config.get("max_features", 50000)
    ngram_range = config.get("ngram_range", (1, 2))
    C = config.get("C", 1.0)
    max_words = config.get("max_words", None)

    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  max_features={max_features}, ngram_range={ngram_range}, C={C}, max_words={max_words}")
    print(f"{'='*60}")

    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples (human={np.sum(labels==0)}, synthetic={np.sum(labels==1)})")

    if max_words:
        texts = truncate_texts(texts, max_words)
        print(f"Truncated texts to first {max_words} words")

    texts_lower = [t.lower() for t in texts]

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        texts_lower, labels, np.arange(len(texts)),
        test_size=0.2, stratify=labels, random_state=SEED
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    Xt = vec.fit_transform(X_train)
    Xv = vec.transform(X_val)
    print(f"TF-IDF features: {Xt.shape[1]}")

    lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=SEED)
    lr.fit(Xt, y_train)

    probs_val = lr.predict_proba(Xv)[:, 1]
    logits_val = np.log(np.clip(probs_val, 1e-10, 1 - 1e-10) / np.clip(1 - probs_val, 1e-10, 1 - 1e-10))

    auc = roc_auc_score(y_val, probs_val)
    acc = accuracy_score(y_val, (probs_val >= 0.5).astype(int))
    ece = compute_ece(probs_val, y_val)
    stats = qscore_stats(probs_val)

    print(f"\nResults:")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ECE:      {ece:.4f}")
    print(f"  q-score mean={stats['mean']:.4f}, std={stats['std']:.4f}, median={stats['median']:.4f}")
    print(f"  mid-range (0.1-0.9) ratio: {stats['mid_range_ratio']:.4f}")
    print(f"  Histogram:")
    for b, c in zip(stats["histogram_bins"], stats["histogram_counts"]):
        bar = "#" * min(int(c / max(1, max(stats['histogram_counts'])) * 40), 40)
        print(f"    {b}: {c:6d} {bar}")

    T = 1.0
    ece_calibrated = ece
    probs_calibrated = probs_val
    if ece > 0.05:
        T = temperature_scale(logits_val, y_val)
        probs_calibrated = 1.0 / (1.0 + np.exp(-logits_val / T))
        ece_calibrated = compute_ece(probs_calibrated, y_val)
        stats_cal = qscore_stats(probs_calibrated)
        print(f"\n  Temperature scaling: T={T:.4f}")
        print(f"  ECE after calibration: {ece_calibrated:.4f}")
        print(f"  Calibrated q-score mean={stats_cal['mean']:.4f}, std={stats_cal['std']:.4f}")
        print(f"  Calibrated mid-range ratio: {stats_cal['mid_range_ratio']:.4f}")
        stats = stats_cal
        probs_val = probs_calibrated

    meets_target = (0.85 <= auc <= 0.98) and (stats["mid_range_ratio"] >= 0.05)
    print(f"\n  Meets target (AUC 0.85-0.98, mid-range >= 5%): {meets_target}")

    return {
        "label": label,
        "config": {k: str(v) for k, v in config.items()},
        "auc": auc,
        "accuracy": acc,
        "ece_raw": ece,
        "ece_calibrated": ece_calibrated,
        "temperature": T,
        "qscore_stats": stats,
        "meets_target": meets_target,
        "vec": vec,
        "lr": lr,
        "probs_val": probs_val,
        "y_val": y_val,
        "texts_val": X_val,
    }


def save_results(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["vec"], OUTPUT_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(result["lr"], OUTPUT_DIR / "lr_model.joblib")

    cal_results = {
        "label": result["label"],
        "config": result["config"],
        "auc": result["auc"],
        "accuracy": result["accuracy"],
        "ece_raw": result["ece_raw"],
        "ece_calibrated": result["ece_calibrated"],
        "temperature": result["temperature"],
        "qscore_stats": result["qscore_stats"],
        "meets_target": result["meets_target"],
    }
    with open(OUTPUT_DIR / "calibration_results.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    with open(OUTPUT_DIR / "predictions.jsonl", "w") as f:
        for text, label, prob in zip(result["texts_val"], result["y_val"], result["probs_val"]):
            preview = text[:200].replace("\n", " ")
            f.write(json.dumps({
                "text_preview": preview,
                "true_label": int(label),
                "q_score": round(float(prob), 6),
            }) + "\n")

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  tfidf_vectorizer.joblib, lr_model.joblib")
    print(f"  calibration_results.json")
    print(f"  predictions.jsonl ({len(result['y_val'])} rows)")


def main():
    configs = [
        {"label": "baseline", "max_features": 50000, "ngram_range": (1, 2), "C": 1.0},
        {"label": "fewer_features", "max_features": 10000, "ngram_range": (1, 2), "C": 1.0},
        {"label": "unigram_only", "max_features": 50000, "ngram_range": (1, 1), "C": 1.0},
        {"label": "strong_reg", "max_features": 50000, "ngram_range": (1, 2), "C": 0.01},
        {"label": "truncate_100w", "max_features": 50000, "ngram_range": (1, 2), "C": 1.0, "max_words": 100},
        {"label": "combo_degrade", "max_features": 10000, "ngram_range": (1, 1), "C": 0.01, "max_words": 100},
    ]

    best = None
    for cfg in configs:
        result = run_experiment(cfg)
        if result["meets_target"]:
            print(f"\n*** Config '{result['label']}' meets target! Saving. ***")
            best = result
            break
        if best is None or abs(result["auc"] - 0.92) < abs(best["auc"] - 0.92):
            best = result

    if not best["meets_target"]:
        print(f"\n*** No config met target. Saving best: '{best['label']}' (AUC={best['auc']:.4f}) ***")

    save_results(best)


if __name__ == "__main__":
    main()
