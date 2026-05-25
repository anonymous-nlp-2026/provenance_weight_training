"""n_min ablation analysis for provenance weight training.

Reads eval results for each n_min experiment, computes PPL statistics,
and generates an ablation curve (PDF + PNG).

Outputs:
  - analysis/nmin_ablation_curve.pdf
  - analysis/nmin_ablation_curve.png
  - analysis/nmin_ablation_summary.json
  - stdout summary
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("/root/provenance_weight_training")
MODELS_DIR = BASE_DIR / "output" / "models"
EVAL_RESULTS_DIR = BASE_DIR / "output" / "eval_results"
EVAL_DIR = BASE_DIR / "output" / "eval"
AGGREGATED_EVAL = BASE_DIR / "output" / "eval_perplexity_results.json"

NMIN_EXPERIMENTS = {
    1: {"model_dir": "nmin1_seed42",         "eval_key": "nmin1_seed42/final"},
    2: {"model_dir": "adaptive_fix_seed42",  "eval_key": "adaptive_fix_seed42/final"},
    3: {"model_dir": "adaptive_seed42",      "eval_key": "adaptive_seed42/final"},
    4: {"model_dir": "nmin4_seed42",         "eval_key": "nmin4_seed42/final"},
}

HARDCODED_FALLBACKS = {
    2: {"loss": 2.8708, "perplexity": 17.650, "source": "hardcoded (D018 confirmed)"},
}

UNIFORM_BASELINE = {
    "model_dir": "uniform_seed42",
    "ppl": 17.6939,
}


def find_eval_result(nmin, cfg):
    """Search multiple locations for eval results."""
    model_dir = cfg["model_dir"]

    # Location 1: output/models/<name>/eval_results.json
    p = MODELS_DIR / model_dir / "eval_results.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {**val, "source": str(p)}

    # Location 2: output/eval_results/<name>_eval.json
    p = EVAL_RESULTS_DIR / f"{model_dir}_eval.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {**val, "source": str(p)}

    # Location 3: output/eval/posthoc_eval_<name>.json
    p = EVAL_DIR / f"posthoc_eval_{model_dir}.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {**val, "source": str(p)}

    # Location 4: aggregated eval_perplexity_results.json
    if AGGREGATED_EVAL.exists():
        data = json.loads(AGGREGATED_EVAL.read_text())
        eval_key = cfg["eval_key"]
        if eval_key in data and "perplexity" in data[eval_key]:
            return {**data[eval_key], "source": str(AGGREGATED_EVAL)}

    # Location 5: hardcoded fallback
    if nmin in HARDCODED_FALLBACKS:
        return HARDCODED_FALLBACKS[nmin]

    return None


def check_training_status(cfg):
    model_path = MODELS_DIR / cfg["model_dir"]
    if not model_path.exists():
        return "NOT_STARTED"
    if (model_path / "final").exists():
        return "COMPLETE"
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    if checkpoints:
        return f"TRAINING (latest: {checkpoints[-1].name})"
    return "UNKNOWN"


def plot_ablation_curve(results, uniform_ppl, output_dir):
    nmins = sorted(results.keys())
    ppls = [results[n]["perplexity"] for n in nmins]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    ax.plot(nmins, ppls, "o-", color="#2171b5", linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor="#2171b5",
            zorder=5, label=r"Adaptive ($n_{\min}$ sweep)")

    ax.axhline(y=uniform_ppl, color="#d62728", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Uniform baseline ({uniform_ppl:.3f})")

    for n, p in zip(nmins, ppls):
        delta = uniform_ppl - p
        offset_y = 8 if delta > 0 else -14
        ax.annotate(f"{p:.3f}", (n, p), textcoords="offset points",
                    xytext=(0, offset_y), ha="center", fontsize=8.5,
                    color="#2171b5", fontweight="bold")

    ax.set_xlabel(r"$n_{\min}$ (minimum source count)", fontsize=11)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=11)
    ax.set_title(r"$n_{\min}$ Ablation — Provenance Weight Training", fontsize=12, pad=10)
    ax.set_xticks(nmins)
    ax.legend(fontsize=9, loc="upper left")

    ppl_min, ppl_max = min(ppls + [uniform_ppl]), max(ppls + [uniform_ppl])
    margin = (ppl_max - ppl_min) * 0.3
    ax.set_ylim(ppl_min - margin, ppl_max + margin)
    ax.set_xlim(min(nmins) - 0.3, max(nmins) + 0.3)

    ax.grid(True, alpha=0.3, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf_path = output_dir / "nmin_ablation_curve.pdf"
    png_path = output_dir / "nmin_ablation_curve.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    output_dir = BASE_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("n_min Ablation Analysis")
    print("=" * 70)

    # --- Collect results ---
    results = {}
    missing = []

    for nmin in sorted(NMIN_EXPERIMENTS.keys()):
        cfg = NMIN_EXPERIMENTS[nmin]
        status = check_training_status(cfg)
        result = find_eval_result(nmin, cfg)
        src = result.get("source", "?") if result else "N/A"
        if result and "perplexity" in result:
            results[nmin] = result
            print(f"  n_min={nmin}: PPL={result['perplexity']:.3f}  [{cfg['model_dir']}]  src={src}")
        else:
            missing.append((nmin, cfg, status))
            print(f"  n_min={nmin}: PENDING  [{cfg['model_dir']}]  status={status}")

    # --- Uniform baseline ---
    uniform_result = None
    for loc in [
        MODELS_DIR / "uniform_seed42" / "eval_results.json",
        EVAL_RESULTS_DIR / "uniform_seed42_eval.json",
        AGGREGATED_EVAL,
    ]:
        if loc.exists():
            data = json.loads(loc.read_text())
            for key, val in data.items():
                if "uniform_seed42" in key and "perplexity" in val:
                    uniform_result = val
                    break
            if uniform_result:
                break
    uniform_ppl = uniform_result["perplexity"] if uniform_result else UNIFORM_BASELINE["ppl"]
    print(f"\n  Uniform baseline: PPL={uniform_ppl:.3f}")

    if missing:
        print(f"\n  WARNING: {len(missing)} experiment(s) missing eval results:")
        for nmin, cfg, status in missing:
            print(f"    n_min={nmin}  {cfg['model_dir']}  ({status})")

    if not results:
        print("\nNo eval results found. Nothing to analyze.")
        sys.exit(1)

    # --- Statistics ---
    print("\n" + "=" * 70)
    print("n_min Ablation — Results")
    print("=" * 70)

    nmins = sorted(results.keys())
    ppls = [results[n]["perplexity"] for n in nmins]
    losses = [results[n].get("loss", float("nan")) for n in nmins]

    best_idx = int(np.argmin(ppls))
    worst_idx = int(np.argmax(ppls))
    ppl_mean = np.mean(ppls)
    ppl_std = np.std(ppls)
    ppl_range = max(ppls) - min(ppls)

    print(f"\n  Available data points: {len(results)}/{len(NMIN_EXPERIMENTS)}")
    print(f"  PPL range:  [{min(ppls):.3f}, {max(ppls):.3f}]")
    print(f"  PPL mean:   {ppl_mean:.3f} ± {ppl_std:.3f}")
    print(f"  PPL Δ(max-min): {ppl_range:.3f}")
    print(f"  Best n_min:  {nmins[best_idx]}  (PPL={ppls[best_idx]:.3f})")
    print(f"  Worst n_min: {nmins[worst_idx]}  (PPL={ppls[worst_idx]:.3f})")

    # --- Relative to uniform baseline ---
    print(f"\n  Relative to uniform baseline (PPL={uniform_ppl:.3f}):")
    print(f"  {'n_min':>5s}  {'PPL':>8s}  {'Δ':>8s}  {'%':>7s}  {'':>3s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*3}")
    for n in nmins:
        ppl = results[n]["perplexity"]
        delta = uniform_ppl - ppl
        pct = delta / uniform_ppl * 100
        marker = "▼" if delta > 0 else "▲"
        print(f"  {n:>5d}  {ppl:>8.3f}  {delta:>+8.3f}  {pct:>+6.2f}%  {marker}")

    # --- Hypothesis checks ---
    print(f"\n  Hypothesis checks:")
    if 2 in results and 3 in results:
        d23 = abs(results[2]["perplexity"] - results[3]["perplexity"])
        print(f"    n_min=2 vs n_min=3: ΔPPL={d23:.3f}  "
              f"({'~tied' if d23 < 0.02 else 'different'})")
    if 4 in results:
        d4u = abs(results[4]["perplexity"] - uniform_ppl)
        print(f"    n_min=4 vs uniform:  ΔPPL={d4u:.3f}  "
              f"({'degrades to uniform ✓' if d4u < 0.02 else 'still differs'})")
    else:
        print(f"    n_min=4 vs uniform:  PENDING (expect ~{uniform_ppl:.3f})")

    all_better = all(results[n]["perplexity"] < uniform_ppl for n in nmins)
    if all_better:
        print(f"\n  All tested n_min values outperform uniform baseline ✓")

    # --- Plot ---
    if len(results) >= 2:
        pdf_path, png_path = plot_ablation_curve(results, uniform_ppl, output_dir)
        print(f"\n  Plot saved:")
        print(f"    {pdf_path}")
        print(f"    {png_path}")
    else:
        print(f"\n  Need ≥2 data points to generate plot. Skipping.")

    # --- JSON summary ---
    summary = {
        "nmin_results": {str(n): {"ppl": results[n]["perplexity"],
                                   "loss": results[n].get("loss", None),
                                   "source": results[n].get("source", "unknown")}
                         for n in nmins},
        "uniform_baseline_ppl": uniform_ppl,
        "statistics": {
            "n_points": len(results),
            "ppl_mean": round(float(ppl_mean), 4),
            "ppl_std": round(float(ppl_std), 4),
            "ppl_range": round(float(ppl_range), 4),
            "best_nmin": nmins[best_idx],
            "best_ppl": round(ppls[best_idx], 4),
            "worst_nmin": nmins[worst_idx],
            "worst_ppl": round(ppls[worst_idx], 4),
        },
        "missing_nmin": [n for n, _, _ in missing],
    }
    json_path = output_dir / "nmin_ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
