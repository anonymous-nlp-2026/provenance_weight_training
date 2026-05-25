"""τ sensitivity analysis for provenance weight training.

Reads eval_results.json from each τ experiment, computes PPL statistics,
and generates a sensitivity curve (PDF + PNG) for the paper.

Inputs:
  - output/models/<name>/eval_results.json  (per-model eval results)
  - output/eval_results/<name>_eval.json    (posthoc eval fallback)
Outputs:
  - analysis/tau_sensitivity_curve.pdf
  - analysis/tau_sensitivity_curve.png
  - stdout summary

Dependencies: matplotlib, numpy (both in conda base)
"""
import argparse
import json
import math
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

TAU_EXPERIMENTS = {
    0.50: {"model_dir": "tau_0.5_seed42",       "eval_key": "tau_0.5_seed42/final"},
    0.60: {"model_dir": "tau_0.6_seed42",       "eval_key": "tau_0.6_seed42/final"},
    0.70: {"model_dir": "tau_0.7_seed42",       "eval_key": "tau_0.7_seed42/final"},
    0.80: {"model_dir": "adaptive_fix_seed42",  "eval_key": "adaptive_fix_seed42/final"},
    0.90: {"model_dir": "tau_09_seed42",        "eval_key": "tau_09_seed42/final"},
    0.95: {"model_dir": "tau_095_seed42",       "eval_key": "tau_095_seed42/final"},
}

UNIFORM_BASELINE = {
    "model_dir": "uniform_seed42",
    "ppl": 17.6939,  # from eval_perplexity_results.json
}


def find_eval_result(tau, cfg):
    """Search multiple locations for eval results of a given τ experiment."""
    model_dir = cfg["model_dir"]

    # Location 1: output/models/<name>/eval_results.json
    p = MODELS_DIR / model_dir / "eval_results.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {"source": str(p), **val}

    # Location 2: output/eval_results/<name>_eval.json
    p = EVAL_RESULTS_DIR / f"{model_dir}_eval.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {"source": str(p), **val}

    # Location 3: output/eval/posthoc_eval_<name>.json
    p = EVAL_DIR / f"posthoc_eval_{model_dir}.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key, val in data.items():
            if "perplexity" in val:
                return {"source": str(p), **val}

    # Location 4: output/eval_perplexity_results.json (aggregated)
    p = BASE_DIR / "output" / "eval_perplexity_results.json"
    if p.exists():
        data = json.loads(p.read_text())
        eval_key = cfg["eval_key"]
        if eval_key in data and "perplexity" in data[eval_key]:
            return {"source": str(p), **data[eval_key]}

    return None


def check_training_status(cfg):
    """Check if training is complete (has final/ dir or last checkpoint)."""
    model_path = MODELS_DIR / cfg["model_dir"]
    if not model_path.exists():
        return "NOT_STARTED"
    if (model_path / "final").exists():
        return "COMPLETE"
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    if checkpoints:
        return f"TRAINING (latest: {checkpoints[-1].name})"
    return "UNKNOWN"


def plot_sensitivity_curve(results, uniform_ppl, output_dir):
    """Generate publication-quality τ sensitivity curve."""
    taus = sorted(results.keys())
    ppls = [results[t]["perplexity"] for t in taus]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    ax.plot(taus, ppls, "o-", color="#2171b5", linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor="#2171b5",
            zorder=5, label=r"Adaptive ($\tau$ sweep)")

    ax.axhline(y=uniform_ppl, color="#d62728", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Uniform baseline ({uniform_ppl:.3f})")

    for t, p in zip(taus, ppls):
        offset_y = 8 if p < uniform_ppl else -14
        ax.annotate(f"{p:.3f}", (t, p), textcoords="offset points",
                    xytext=(0, offset_y), ha="center", fontsize=8,
                    color="#2171b5", fontweight="bold")

    ax.set_xlabel(r"Threshold $\tau$", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:.2f}" for t in taus])

    y_min = min(min(ppls), uniform_ppl) - 0.05
    y_max = max(max(ppls), uniform_ppl) + 0.05
    ax.set_ylim(y_min, y_max)

    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    pdf_path = output_dir / "tau_sensitivity_curve.pdf"
    png_path = output_dir / "tau_sensitivity_curve.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="τ sensitivity analysis for provenance weight training"
    )
    parser.add_argument("--output_dir", type=str,
                        default=str(BASE_DIR / "analysis"),
                        help="Directory for output plots (default: analysis/)")
    parser.add_argument("--run-eval", action="store_true",
                        help="Run eval_perplexity.py for experiments missing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect results ---
    results = {}
    missing = []

    print("=" * 70)
    print("τ Sensitivity Analysis — Data Collection")
    print("=" * 70)

    for tau in sorted(TAU_EXPERIMENTS.keys()):
        cfg = TAU_EXPERIMENTS[tau]
        status = check_training_status(cfg)
        result = find_eval_result(tau, cfg)

        if result:
            results[tau] = result
            print(f"  τ={tau:.2f}  PPL={result['perplexity']:.3f}  "
                  f"loss={result['loss']:.4f}  [{cfg['model_dir']}]")
        else:
            missing.append((tau, cfg, status))
            print(f"  τ={tau:.2f}  NO EVAL  status={status}  [{cfg['model_dir']}]")

    # --- Check uniform baseline ---
    uniform_result = None
    for loc in [
        MODELS_DIR / "uniform_seed42" / "eval_results.json",
        EVAL_RESULTS_DIR / "uniform_seed42_eval.json",
        BASE_DIR / "output" / "eval_perplexity_results.json",
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
        print(f"\n  WARNING: {len(missing)} experiments missing eval results:")
        for tau, cfg, status in missing:
            print(f"    τ={tau:.2f}  {cfg['model_dir']}  ({status})")
            if status == "COMPLETE" and args.run_eval:
                print(f"    → Run: python eval_perplexity.py "
                      f"--checkpoints output/models/{cfg['model_dir']}/final "
                      f"--data_path data/human/eval_holdout.jsonl "
                      f"--output_path output/models/{cfg['model_dir']}/eval_results.json")

    if not results:
        print("\nNo eval results found. Nothing to analyze.")
        sys.exit(1)

    # --- Statistics ---
    print("\n" + "=" * 70)
    print("τ Sensitivity Analysis — Results")
    print("=" * 70)

    taus = sorted(results.keys())
    ppls = [results[t]["perplexity"] for t in taus]
    losses = [results[t]["loss"] for t in taus]

    best_idx = int(np.argmin(ppls))
    worst_idx = int(np.argmax(ppls))
    ppl_mean = np.mean(ppls)
    ppl_std = np.std(ppls)
    ppl_range = max(ppls) - min(ppls)

    print(f"\n  Available τ points: {len(results)}/6")
    print(f"  PPL range:  [{min(ppls):.3f}, {max(ppls):.3f}]")
    print(f"  PPL mean:   {ppl_mean:.3f} ± {ppl_std:.3f}")
    print(f"  PPL Δ(max-min): {ppl_range:.3f}")
    print(f"  Best τ:     {taus[best_idx]:.2f}  (PPL={ppls[best_idx]:.3f})")
    print(f"  Worst τ:    {taus[worst_idx]:.2f}  (PPL={ppls[worst_idx]:.3f})")

    robust = ppl_range < 0.05
    print(f"\n  Robustness check (Δ < 0.05): {'PASS ✓' if robust else 'FAIL ✗'}  "
          f"(Δ={ppl_range:.4f})")

    print(f"\n  Relative to uniform baseline (PPL={uniform_ppl:.3f}):")
    for t in taus:
        ppl = results[t]["perplexity"]
        delta = uniform_ppl - ppl
        pct = delta / uniform_ppl * 100
        marker = "▼" if delta > 0 else "▲"
        print(f"    τ={t:.2f}:  PPL={ppl:.3f}  Δ={delta:+.3f} ({pct:+.2f}%) {marker}")

    all_better = all(results[t]["perplexity"] < uniform_ppl for t in taus)
    if all_better:
        print(f"\n  All τ values outperform uniform baseline ✓")

    # --- Plot ---
    if len(results) >= 2:
        pdf_path, png_path = plot_sensitivity_curve(results, uniform_ppl, output_dir)
        print(f"\n  Plot saved:")
        print(f"    {pdf_path}")
        print(f"    {png_path}")
    else:
        print("\n  Need ≥2 data points to generate plot. Skipping.")

    # --- JSON dump ---
    summary = {
        "tau_results": {str(t): {"ppl": results[t]["perplexity"],
                                  "loss": results[t]["loss"]}
                        for t in taus},
        "uniform_baseline_ppl": uniform_ppl,
        "statistics": {
            "n_points": len(results),
            "ppl_mean": round(ppl_mean, 4),
            "ppl_std": round(ppl_std, 4),
            "ppl_range": round(ppl_range, 4),
            "best_tau": taus[best_idx],
            "best_ppl": round(ppls[best_idx], 4),
            "worst_tau": taus[worst_idx],
            "worst_ppl": round(ppls[worst_idx], 4),
            "robust_default_claim": robust,
        },
        "missing_tau": [t for t, _, _ in missing],
    }
    json_path = output_dir / "tau_sensitivity_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
