"""Complete τ sweep analysis: PPL curve, b* statistics, claim verification."""
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
FIG_DIR = BASE_DIR / "analysis" / "figures"
CSV_DIR = BASE_DIR / "analysis"

TAU_EXPERIMENTS = {
    0.50: "tau_0.5_seed42",
    0.60: "tau_0.6_seed42",
    0.70: "tau_0.7_seed42",
    0.80: "adaptive_fix_seed42",
    0.90: "tau_09_seed42",
    0.95: "tau_095_seed42",
}

HARDCODED_PPL = {
    "tau_0.5_seed42": 17.688,
    "tau_0.6_seed42": 17.681,
    "adaptive_fix_seed42": 17.653,
}

UNIFORM_PPL = 17.694
GRID_BEST_PPL = 17.651


def load_ppl(exp_id):
    """Load PPL from eval_results.json, fallback to hardcoded."""
    for loc in [
        MODELS_DIR / exp_id / "eval_results.json",
        BASE_DIR / "output" / "eval_results" / f"{exp_id}_eval.json",
        BASE_DIR / "output" / "eval" / f"posthoc_eval_{exp_id}.json",
    ]:
        if loc.exists():
            data = json.loads(loc.read_text())
            for val in data.values():
                if isinstance(val, dict) and "perplexity" in val:
                    return val["perplexity"], str(loc)

    agg = BASE_DIR / "output" / "eval_perplexity_results.json"
    if agg.exists():
        data = json.loads(agg.read_text())
        key = f"{exp_id}/final"
        if key in data and "perplexity" in data[key]:
            return data[key]["perplexity"], str(agg)

    if exp_id in HARDCODED_PPL:
        return HARDCODED_PPL[exp_id], "hardcoded"
    return None, None


def load_b_values(exp_id):
    """Load all b values from step_metrics.jsonl."""
    path = MODELS_DIR / exp_id / "step_metrics.jsonl"
    if not path.exists():
        return None
    bs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "b" in rec:
                bs.append(rec["b"])
    return np.array(bs) if bs else None


def is_complete(exp_id):
    return (MODELS_DIR / exp_id / "final").exists()


def b_stats(bs):
    return {
        "mean": float(np.mean(bs)),
        "median": float(np.median(bs)),
        "std": float(np.std(bs)),
        "p25": float(np.percentile(bs, 25)),
        "p75": float(np.percentile(bs, 75)),
        "pct_zero": float(np.mean(bs == 0) * 100),
        "pct_above2": float(np.mean(bs > 2) * 100),
        "n": len(bs),
    }


def plot_ppl_curve(tau_ppl, output_dir):
    taus = sorted(tau_ppl.keys())
    ppls = [tau_ppl[t] for t in taus]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(taus, ppls, "o-", color="#2171b5", lw=2, ms=8,
            mfc="white", mew=2, mec="#2171b5", zorder=5,
            label=r"Adaptive ($\tau$ sweep)")
    ax.axhline(UNIFORM_PPL, color="#d62728", ls="--", lw=1.5, alpha=0.8,
               label=f"Uniform baseline ({UNIFORM_PPL:.3f})")
    ax.axhline(GRID_BEST_PPL, color="#2ca02c", ls=":", lw=1.5, alpha=0.8,
               label=f"Grid best b=0.5 ({GRID_BEST_PPL:.3f})")

    for t, p in zip(taus, ppls):
        ax.annotate(f"{p:.3f}", (t, p), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7.5, color="#333")

    ax.set_xlabel(r"$\tau$ (contamination threshold)", fontsize=11)
    ax.set_ylabel("Eval Perplexity (holdout)", fontsize=11)
    ax.set_title(r"$\tau$ Sensitivity Sweep", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_xticks(taus)
    ymin = min(ppls + [GRID_BEST_PPL]) - 0.015
    ymax = max(ppls + [UNIFORM_PPL]) + 0.015
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"tau_ppl_curve.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_b_median_curve(tau_bstats, output_dir):
    taus = sorted(tau_bstats.keys())
    medians = [tau_bstats[t]["median"] for t in taus]
    means = [tau_bstats[t]["mean"] for t in taus]
    p25s = [tau_bstats[t]["p25"] for t in taus]
    p75s = [tau_bstats[t]["p75"] for t in taus]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.fill_between(taus, p25s, p75s, alpha=0.15, color="#2171b5", label="IQR (p25–p75)")
    ax.plot(taus, medians, "s-", color="#2171b5", lw=2, ms=7,
            mfc="white", mew=2, mec="#2171b5", zorder=5, label=r"$b^*$ median")
    ax.plot(taus, means, "^--", color="#e6550d", lw=1.5, ms=6,
            mfc="white", mew=1.5, mec="#e6550d", zorder=4, label=r"$b^*$ mean")

    ax.set_xlabel(r"$\tau$ (contamination threshold)", fontsize=11)
    ax.set_ylabel(r"$b^*$ (down-weighting strength)", fontsize=11)
    ax.set_title(r"$\tau$ vs Down-weighting Strength", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.set_xticks(taus)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"tau_b_strength.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(tau_ppl, tau_bstats, output_path):
    header = "tau,ppl,b_mean,b_median,b_std,b_p25,b_p75,b_pct_zero,b_pct_above2,b_n"
    rows = [header]
    all_taus = sorted(set(list(tau_ppl.keys()) + list(tau_bstats.keys())))
    for t in all_taus:
        ppl = f"{tau_ppl[t]:.4f}" if t in tau_ppl else ""
        if t in tau_bstats:
            s = tau_bstats[t]
            bstr = (f"{s['mean']:.4f},{s['median']:.4f},{s['std']:.4f},"
                    f"{s['p25']:.4f},{s['p75']:.4f},{s['pct_zero']:.1f},{s['pct_above2']:.1f},{s['n']}")
        else:
            bstr = ",,,,,,,"
        rows.append(f"{t:.2f},{ppl},{bstr}")
    output_path.write_text("\n".join(rows) + "\n")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tau_ppl = {}
    tau_bstats = {}
    missing_ppl = []
    missing_b = []

    print("=" * 70)
    print("τ Sweep Complete Analysis")
    print("=" * 70)

    # --- Gather data ---
    for tau, exp_id in sorted(TAU_EXPERIMENTS.items()):
        status = "COMPLETE" if is_complete(exp_id) else "RUNNING/MISSING"
        ppl, source = load_ppl(exp_id)
        bs = load_b_values(exp_id)

        if ppl is not None:
            tau_ppl[tau] = ppl
        else:
            missing_ppl.append((tau, exp_id, status))

        if bs is not None:
            tau_bstats[tau] = b_stats(bs)
        else:
            missing_b.append((tau, exp_id))

        ppl_str = f"{ppl:.3f}" if ppl else "N/A"
        b_str = f"mean={tau_bstats[tau]['mean']:.3f}" if tau in tau_bstats else "N/A"
        src_str = f" ({source})" if source else ""
        print(f"  τ={tau:.2f}  {exp_id:25s}  {status:12s}  PPL={ppl_str}{src_str}  b*: {b_str}")

    if missing_ppl:
        print(f"\n  Missing PPL ({len(missing_ppl)}): {', '.join(f'τ={t:.2f}' for t,_,_ in missing_ppl)}")
    if missing_b:
        print(f"  Missing b* ({len(missing_b)}): {', '.join(f'τ={t:.2f}' for t,_ in missing_b)}")

    # --- b* statistics table ---
    if tau_bstats:
        print("\n" + "=" * 70)
        print("b* Statistics by τ")
        print("=" * 70)
        print(f"  {'τ':>5s} {'mean':>8s} {'median':>8s} {'std':>8s} {'p25':>8s} {'p75':>8s} {'%zero':>7s} {'%>2':>7s} {'n':>7s}")
        print("  " + "-" * 67)
        for tau in sorted(tau_bstats.keys()):
            s = tau_bstats[tau]
            print(f"  {tau:5.2f} {s['mean']:8.3f} {s['median']:8.3f} {s['std']:8.3f} "
                  f"{s['p25']:8.3f} {s['p75']:8.3f} {s['pct_zero']:6.1f}% {s['pct_above2']:6.1f}% {s['n']:7d}")

    # --- PPL analysis ---
    if tau_ppl:
        print("\n" + "=" * 70)
        print("PPL Analysis")
        print("=" * 70)
        taus = sorted(tau_ppl.keys())
        ppls = [tau_ppl[t] for t in taus]
        best_tau = taus[int(np.argmin(ppls))]

        print(f"  Uniform baseline:  {UNIFORM_PPL:.3f}")
        print(f"  Grid best (b=0.5): {GRID_BEST_PPL:.3f}")
        print(f"  Best τ:            {best_tau:.2f} (PPL={min(ppls):.3f})")
        print(f"  PPL range:         {max(ppls)-min(ppls):.4f}")
        print()
        for t in taus:
            p = tau_ppl[t]
            delta_u = UNIFORM_PPL - p
            marker = "< uniform" if delta_u > 0 else ">= uniform"
            print(f"    τ={t:.2f}  PPL={p:.3f}  Δ_uniform={delta_u:+.3f}  {marker}")

    # --- Claim 3 verification ---
    print("\n" + "=" * 70)
    print("Claim 3 Verification")
    print("=" * 70)
    claim_taus = [t for t in [0.6, 0.7, 0.8, 0.9, 0.95] if t in tau_ppl]
    missing_claim = [t for t in [0.6, 0.7, 0.8, 0.9, 0.95] if t not in tau_ppl]
    if missing_claim:
        print(f"  Missing data for τ: {missing_claim}")
    failures = [t for t in claim_taus if tau_ppl[t] > UNIFORM_PPL]
    if not claim_taus:
        print("  INCONCLUSIVE: no τ∈[0.6,0.95] data available")
    elif failures:
        print(f"  Claim 3 NOT SUPPORTED: τ values exceeding uniform: {failures}")
        for t in failures:
            print(f"    τ={t:.2f}  PPL={tau_ppl[t]:.3f} > {UNIFORM_PPL:.3f}")
    elif missing_claim:
        avail = ", ".join(f"τ={t:.2f}" for t in claim_taus)
        print(f"  Claim 3 PARTIALLY SUPPORTED ({len(claim_taus)}/{len(claim_taus)+len(missing_claim)}):")
        print(f"    Available ({avail}) all ≤ uniform({UNIFORM_PPL:.3f})")
        print(f"    Waiting on: {missing_claim}")
    else:
        print(f"  Claim 3 SUPPORTED: all τ∈[0.6,0.95] PPL ≤ uniform({UNIFORM_PPL:.3f})")
        print(f"  → τ is a robust empirical default")

    # --- Generate plots ---
    if len(tau_ppl) >= 2:
        plot_ppl_curve(tau_ppl, FIG_DIR)
        print(f"\n  PPL curve:   {FIG_DIR / 'tau_ppl_curve.pdf'}")

    if len(tau_bstats) >= 2:
        plot_b_median_curve(tau_bstats, FIG_DIR)
        print(f"  b* strength: {FIG_DIR / 'tau_b_strength.pdf'}")

    # --- CSV ---
    csv_path = CSV_DIR / "tau_sweep_summary.csv"
    write_csv(tau_ppl, tau_bstats, csv_path)
    print(f"  CSV:         {csv_path}")

    # --- JSON ---
    summary = {
        "tau_ppl": {f"{t:.2f}": tau_ppl[t] for t in sorted(tau_ppl)},
        "tau_bstats": {f"{t:.2f}": tau_bstats[t] for t in sorted(tau_bstats)},
        "baselines": {"uniform": UNIFORM_PPL, "grid_best_b05": GRID_BEST_PPL},
        "claim3": {
            "tested_taus": claim_taus,
            "missing_taus": missing_claim,
            "failures": failures,
            "supported": len(failures) == 0 and len(missing_claim) == 0,
            "partially_supported": len(failures) == 0 and len(missing_claim) > 0,
        },
    }
    json_path = CSV_DIR / "tau_sweep_full_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  JSON:        {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
