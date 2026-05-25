# Fig 3: τ Sweep — eval PPL vs adaptive threshold τ
# Data: adaptive runs with varying τ, seed42
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

C_BLUE = '#0072B2'
C_RED = '#D55E00'
C_GREEN_FILL = '#009E73'

tau = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ppl = [17.688, 17.681, 17.679, 17.653, 17.712, 17.723]
uniform_baseline = 17.694

fig, ax = plt.subplots(figsize=(5.5, 4))

# Safe range shading [0.5, 0.8]
ax.axvspan(0.5, 0.8, alpha=0.12, color=C_GREEN_FILL, label='Safe range')

ax.plot(tau, ppl, 'o-', color=C_BLUE, lw=1.5, ms=6, zorder=4)

# Highlight best
best_idx = int(np.argmin(ppl))
ax.plot(tau[best_idx], ppl[best_idx], '*', color=C_BLUE, ms=14, zorder=5)
ax.annotate(f'τ={tau[best_idx]}\nPPL={ppl[best_idx]:.3f}',
            xy=(tau[best_idx], ppl[best_idx]),
            xytext=(tau[best_idx]+0.05, ppl[best_idx]-0.015),
            fontsize=9, ha='left')

ax.axhline(uniform_baseline, color=C_RED, ls='--', lw=1.2, label=f'Uniform baseline ({uniform_baseline})')

ax.set_xlabel(r'Threshold $\tau$')
ax.set_ylabel('Eval PPL (holdout)')
ax.set_xticks(tau)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0.45, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = '/root/provenance_weight_training/output/figures/fig3_tau_sweep.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
