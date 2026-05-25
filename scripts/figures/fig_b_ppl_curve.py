# Fig 2: b-PPL Curve — eval PPL vs provenance exponent b
# Data: grid sweep runs (seed42 full, seed123 b>=0.5), uniform & adaptive baselines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# Colorblind-safe palette (Okabe-Ito)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'

b_seed42 = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ppl_seed42 = [17.64, 17.651, 17.692, 17.718, 17.741, 17.774, 17.818]

b_seed123 = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ppl_seed123 = [17.638, 17.674, 17.701, 17.725, 17.76, 17.804]

uniform_42, uniform_123 = 17.694, 17.686
adaptive_42, adaptive_123 = 17.653, 17.643

fig, ax = plt.subplots(figsize=(5.5, 4))

ax.plot(b_seed42, ppl_seed42, 'o-', color=C_BLUE, lw=1.5, ms=5, label='Grid (seed 42)')
ax.plot(b_seed123, ppl_seed123, 's--', color=C_ORANGE, lw=1.5, ms=5, label='Grid (seed 123)')

# Highlight grid-best
best_idx_42 = int(np.argmin(ppl_seed42))
best_idx_123 = int(np.argmin(ppl_seed123))
ax.plot(b_seed42[best_idx_42], ppl_seed42[best_idx_42], '*', color=C_BLUE, ms=14, zorder=5)
ax.plot(b_seed123[best_idx_123], ppl_seed123[best_idx_123], '*', color=C_ORANGE, ms=14, zorder=5)

# Baselines
ax.axhline(np.mean([uniform_42, uniform_123]), color=C_RED, ls='--', lw=1.2, label=f'Uniform (avg {np.mean([uniform_42, uniform_123]):.3f})')
ax.axhline(np.mean([adaptive_42, adaptive_123]), color=C_GREEN, ls='-', lw=1.5, label=f'Adaptive (avg {np.mean([adaptive_42, adaptive_123]):.3f})')

ax.set_xlabel('Provenance exponent $b$')
ax.set_ylabel('Eval PPL (holdout)')
ax.set_xticks(b_seed42)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0.2, 5.3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = '/root/provenance_weight_training/output/figures/fig2_b_ppl_curve.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
