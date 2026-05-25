#!/usr/bin/env python3
"""Extract per-batch figure data from step_metrics.jsonl files.

Handles restart-induced duplicates by identifying run boundaries
(step number decreases) and combining the initial run's early steps
with the final run's later steps.
"""
import json
import csv
import os
from collections import defaultdict

BASE = "/root/provenance_weight_training/output/models"
OUT = "/root/provenance_weight_training/artifacts/figure_data"


def classify_regime(b, alpha_eff, tau=0.8):
    if b == 0.0:
        return "clean"
    if alpha_eff >= tau - 0.005:
        return "alpha_eff_binding"
    return "ess_binding"


def find_run_boundaries(data):
    """Find indices where a new run starts (step number decreases)."""
    boundaries = [0]
    for i in range(1, len(data)):
        if data[i]["step"] < data[i - 1]["step"]:
            boundaries.append(i)
    return boundaries


def load_deduped(path):
    """Load step_metrics.jsonl with proper dedup.

    Strategy: identify run boundaries, use the initial run for early steps
    and the final run for later steps. Take first 4 entries per step
    (one per micro-batch in gradient accumulation).
    """
    with open(path) as f:
        data = [json.loads(line) for line in f]

    if len(data) == 0:
        return []

    boundaries = find_run_boundaries(data)

    if len(boundaries) == 1:
        # Single run, just take first 4 per step
        step_entries = defaultdict(list)
        for d in data:
            step_entries[d["step"]].append(d)
        result = []
        for step in sorted(step_entries.keys()):
            result.extend(step_entries[step][:4])
        return result

    # Multiple runs: use initial run for early steps, final run for late steps
    run0_end = boundaries[1]
    final_run_start = boundaries[-1]

    # Find the restart step (where the final run begins)
    restart_step = data[final_run_start]["step"]

    # Initial run: steps before restart_step
    initial = defaultdict(list)
    for d in data[:run0_end]:
        if d["step"] < restart_step:
            initial[d["step"]].append(d)

    # Final run: all steps from restart onwards
    final = defaultdict(list)
    for d in data[final_run_start:]:
        final[d["step"]].append(d)

    result = []
    for step in sorted(initial.keys()):
        result.extend(initial[step][:4])
    for step in sorted(final.keys()):
        result.extend(final[step][:4])

    return result


def extract_adaptive_fix(exp="adaptive_fix_seed42", tau=0.8):
    path = os.path.join(BASE, exp, "step_metrics.jsonl")
    raw = load_deduped(path)

    rows = []
    for d in raw:
        regime = classify_regime(d["b"], d["alpha_eff"], tau)
        rows.append({
            "step": d["step"],
            "b_star": d["b"],
            "alpha_eff": d["alpha_eff"],
            "ess": d["ess"],
            "regime": regime,
        })

    csv_path = os.path.join(OUT, "bstar_trajectory_seed42.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "b_star", "alpha_eff", "ess", "regime"])
        w.writeheader()
        w.writerows(rows)

    counts = defaultdict(int)
    b_by_regime = defaultdict(list)
    alpha_by_regime = defaultdict(list)
    ess_by_regime = defaultdict(list)
    for r in rows:
        counts[r["regime"]] += 1
        b_by_regime[r["regime"]].append(r["b_star"])
        alpha_by_regime[r["regime"]].append(r["alpha_eff"])
        ess_by_regime[r["regime"]].append(r["ess"])

    total = len(rows)
    stats = {
        "total_batches": total,
        "regime_pct": {k: round(100 * v / total, 2) for k, v in sorted(counts.items())},
        "regime_counts": dict(sorted(counts.items())),
        "mean_b_star_by_regime": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(b_by_regime.items())
        },
        "mean_alpha_eff_by_regime": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(alpha_by_regime.items())
        },
        "mean_ess_by_regime": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(ess_by_regime.items())
        },
    }

    json_path = os.path.join(OUT, "binding_stats_seed42.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"CSV: {csv_path} ({len(rows)} rows)")
    print(f"JSON: {json_path}")
    print(json.dumps(stats, indent=2))
    return rows


def extract_tau_sweep():
    experiments = {
        0.5: "tau_0.5_seed42",
        0.6: "tau_0.6_seed42",
        0.7: "tau_0.7_seed42",
        0.8: "adaptive_fix_seed42",
        0.9: "tau_09_seed42",
        0.95: "tau_095_seed42",
    }
    results = []
    for tau_val, exp_name in sorted(experiments.items()):
        path = os.path.join(BASE, exp_name, "step_metrics.jsonl")
        if not os.path.exists(path):
            print(f"SKIP {exp_name}: no step_metrics.jsonl")
            continue
        raw = load_deduped(path)
        bs = [d["b"] for d in raw]
        alphas = [d["alpha_eff"] for d in raw]
        results.append({
            "tau": tau_val,
            "experiment": exp_name,
            "n_batches": len(bs),
            "mean_b_star": round(sum(bs) / len(bs), 4),
            "mean_alpha_eff": round(sum(alphas) / len(alphas), 4),
        })
        print(f"tau={tau_val}: mean_b*={results[-1]['mean_b_star']:.4f}, "
              f"mean_alpha_eff={results[-1]['mean_alpha_eff']:.4f} ({len(bs)} batches)")

    json_path = os.path.join(OUT, "tau_sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(OUT, "tau_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "experiment", "n_batches", "mean_b_star", "mean_alpha_eff"])
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved: {json_path}, {csv_path}")


if __name__ == "__main__":
    print("=== Extracting adaptive_fix_seed42 ===")
    extract_adaptive_fix()
    print("\n=== Extracting tau sweep ===")
    extract_tau_sweep()
