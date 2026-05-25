import numpy as np
from scipy import stats
import csv

def bca_bootstrap_ci(data, n_bootstrap=10000, alpha=0.05, seed=42):
    """BCa (Bias-Corrected and Accelerated) Bootstrap CI."""
    rng = np.random.RandomState(seed)
    n = len(data)
    theta_hat = np.mean(data)
    
    boot_means = np.array([np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)])
    
    z0 = stats.norm.ppf(np.mean(boot_means < theta_hat))
    
    jackknife = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jackknife)
    num = np.sum((jack_mean - jackknife)**3)
    den = 6 * (np.sum((jack_mean - jackknife)**2))**1.5
    a = num / den if den != 0 else 0
    
    z_alpha = stats.norm.ppf(alpha/2)
    z_1alpha = stats.norm.ppf(1 - alpha/2)
    
    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a*(z0 + z_alpha)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a*(z0 + z_1alpha)))
    
    ci_low = np.percentile(boot_means, 100*alpha1)
    ci_high = np.percentile(boot_means, 100*alpha2)
    
    return theta_hat, ci_low, ci_high, np.std(boot_means)

def paired_diff_ci(group_a, group_b, n_bootstrap=10000, alpha=0.05, seed=42):
    """BCa CI for paired difference (A - B)."""
    diffs = np.array(group_a) - np.array(group_b)
    return bca_bootstrap_ci(diffs, n_bootstrap, alpha, seed)

# === DATA (from registry) ===

data = {
    # Claim 1: Adaptive vs Uniform rho=0.4 (tau=0.7)
    "adaptive_rho04_tau07_holdout": [17.68, 17.666, 17.677],
    "uniform_rho04_holdout":       [17.6939, 17.6861, 17.7015],
    
    # Claim 1 OOD
    "adaptive_rho04_tau07_owt":  [18.348, 18.348, 18.344],
    "uniform_rho04_owt":         [18.365, 18.3678, 18.3865],
    "adaptive_rho04_tau07_wiki": [15.331, 15.345, 15.315],
    "uniform_rho04_wiki":        [15.335, 15.3768, 15.3594],
    
    # Claim 2: Detector ablation (rho=0.4, tau=0.8 for TF-IDF)
    "tfidf_adaptive_holdout":  [17.6527, 17.6432, 17.6566],
    "gpt2_adaptive_holdout":   [17.79, 17.78848, 17.79719],
    "ngram_adaptive_holdout":  [17.820214, 17.823, 17.82607],
    
    # Claim 3: rho=0.6 tau=0.7
    "adaptive_rho06_holdout":  [17.735, 17.727, 17.732],
    "uniform_rho06_holdout":   [17.766, 17.754, 17.77],
    
    # Claim 3 OOD (adaptive only — uniform rho06 incomplete)
    "adaptive_rho06_owt":  [18.408, 18.385, 18.411],
    "adaptive_rho06_wiki": [15.355, 15.329, 15.424],
}

results = []

# 1. Individual group CIs
for name, values in data.items():
    if values and len(values) == 3:
        mean, ci_lo, ci_hi, se = bca_bootstrap_ci(np.array(values))
        results.append({
            "group": name,
            "type": "individual",
            "mean": f"{mean:.5f}",
            "ci_95_low": f"{ci_lo:.5f}",
            "ci_95_high": f"{ci_hi:.5f}",
            "se": f"{se:.5f}",
            "width": f"{ci_hi - ci_lo:.5f}",
            "includes_zero": "",
        })

# 2. Paired difference CIs
pairs = [
    # Claim 1: adaptive vs uniform rho=0.4 (lower is better, so negative = adaptive wins)
    ("adaptive_rho04_tau07_holdout", "uniform_rho04_holdout", "Delta(adapt-unif) rho04 holdout"),
    ("adaptive_rho04_tau07_owt", "uniform_rho04_owt", "Delta(adapt-unif) rho04 OWT"),
    ("adaptive_rho04_tau07_wiki", "uniform_rho04_wiki", "Delta(adapt-unif) rho04 Wiki"),
    # Claim 2: TF-IDF vs GPT-2, TF-IDF vs N-gram
    ("tfidf_adaptive_holdout", "gpt2_adaptive_holdout", "Delta(tfidf-gpt2) holdout"),
    ("tfidf_adaptive_holdout", "ngram_adaptive_holdout", "Delta(tfidf-ngram) holdout"),
    ("gpt2_adaptive_holdout", "ngram_adaptive_holdout", "Delta(gpt2-ngram) holdout"),
    # Claim 3: adaptive vs uniform rho=0.6
    ("adaptive_rho06_holdout", "uniform_rho06_holdout", "Delta(adapt-unif) rho06 holdout"),
]

for key_a, key_b, label in pairs:
    if data.get(key_a) and data.get(key_b) and len(data[key_a]) == 3 and len(data[key_b]) == 3:
        mean, ci_lo, ci_hi, se = paired_diff_ci(data[key_a], data[key_b])
        includes_zero = ci_lo <= 0 <= ci_hi
        results.append({
            "group": label,
            "type": "paired_diff",
            "mean": f"{mean:.5f}",
            "ci_95_low": f"{ci_lo:.5f}",
            "ci_95_high": f"{ci_hi:.5f}",
            "se": f"{se:.5f}",
            "width": f"{ci_hi - ci_lo:.5f}",
            "includes_zero": str(includes_zero),
        })

# Output CSV
outpath = "output/bootstrap_ci_results.csv"
with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group", "type", "mean", "ci_95_low", "ci_95_high", "se", "width", "includes_zero"])
    writer.writeheader()
    writer.writerows(results)

# Print summary
print("=" * 80)
print("Bootstrap CI Results (BCa, B=10000, alpha=0.05)")
print("=" * 80)
print()

print("--- Individual Group CIs ---")
for r in results:
    if r["type"] == "individual":
        print(f"  {r['group']:40s}: {r['mean']} [{r['ci_95_low']}, {r['ci_95_high']}]  (w={r['width']})")

print()
print("--- Paired Difference CIs ---")
for r in results:
    if r["type"] == "paired_diff":
        sig = "SIGNIFICANT (CI excludes 0)" if r["includes_zero"] == "False" else "NOT significant (CI includes 0)"
        print(f"  {r['group']:40s}: {r['mean']} [{r['ci_95_low']}, {r['ci_95_high']}]  | {sig}")

print()
print(f"CSV saved to: {outpath}")

# Power assessment
print()
print("--- Statistical Power Assessment ---")
n_sig = sum(1 for r in results if r["type"] == "paired_diff" and r["includes_zero"] == "False")
n_total = sum(1 for r in results if r["type"] == "paired_diff")
print(f"Significant comparisons: {n_sig}/{n_total}")
print()
print("Note: With n=3 seeds, BCa bootstrap CIs are wide. Narrow CIs that exclude 0")
print("indicate robust effects; wide CIs suggest more seeds would strengthen claims.")
