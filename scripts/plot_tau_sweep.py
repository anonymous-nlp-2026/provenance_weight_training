import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

# Data
tau = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
ppl = np.array([17.688, 17.6805, 17.6793, 17.6527, 17.712, 17.723])

uniform_ppl = 17.694
grid_best_ppl = 17.64

best_idx = 3  # tau=0.8

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Safe range shading (tau 0.5 to 0.8, PPL better than uniform)
ax.axvspan(0.5, 0.8, alpha=0.10, color='#56B4E9', zorder=0,
           label=r'Safe range ($\tau \leq 0.8$)')

# Reference lines
ax.axhline(uniform_ppl, color='#999999', linestyle='--', linewidth=1.2, zorder=1)
ax.text(0.955, uniform_ppl + 0.0015, 'Uniform baseline', fontsize=9,
        color='#999999', ha='right', va='bottom')

ax.axhline(grid_best_ppl, color='#009E73', linestyle=':', linewidth=1.2, zorder=1)
ax.text(0.955, grid_best_ppl - 0.0015, r'Grid-best ($\beta$=0.3)', fontsize=9,
        color='#009E73', ha='right', va='top')

# Main curve (non-best points)
mask = np.ones(len(tau), dtype=bool)
mask[best_idx] = False
ax.plot(tau[mask], ppl[mask], 'o-', color='#0072B2', markersize=7,
        zorder=3, label=r'$\tau$-PPL ($\rho$=0.4)')

# Connect all points with line
ax.plot(tau, ppl, '-', color='#0072B2', zorder=2)

# Best point (star marker)
ax.plot(tau[best_idx], ppl[best_idx], '*', color='#D55E00', markersize=16,
        zorder=4, markeredgecolor='white', markeredgewidth=0.8,
        label=r'Best ($\tau$=0.8, PPL=17.653)')

# Axis config
ax.set_xlabel(r'Temperature threshold $\tau$')
ax.set_ylabel('Held-out Perplexity')
ax.set_xlim(0.45, 1.0)
ax.set_ylim(17.63, 17.73)
ax.set_xticks(tau)
ax.set_xticklabels([str(t) for t in tau])

# Legend
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

# Light grid
ax.grid(True, alpha=0.2, axis='y', zorder=0)

plt.tight_layout()

outdir = '/root/provenance_weight_training/output/figures'
plt.savefig(f'{outdir}/tau_sweep_rho04.pdf')
plt.savefig(f'{outdir}/tau_sweep_rho04.png')
plt.close()

print(f'Saved PDF and PNG to {outdir}/')
