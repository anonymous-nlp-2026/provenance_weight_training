# Fig 4: ρ Sensitivity — grouped bar chart, 4 methods × 2 ρ values
# Data: uniform, grid b=0.5, adaptive τ=0.8, ESS-only at ρ=0.4 and ρ=0.6
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
C_ORANGE = '#E69F00'

methods = ['Uniform', 'Grid $b$=0.5', r'Adaptive $\tau$=0.8', 'ESS-only']
rho_04 = [17.694, 17.651, 17.653, 17.730]
rho_06 = [17.766, 17.704, 17.740, 17.770]

x = np.arange(len(methods))
width = 0.32

fig, ax = plt.subplots(figsize=(6, 4))

bars1 = ax.bar(x - width/2, rho_04, width, label=r'$\rho=0.4$', color=C_BLUE, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, rho_06, width, label=r'$\rho=0.6$', color=C_ORANGE, edgecolor='white', linewidth=0.5)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)

ax.set_ylabel('Eval PPL (holdout)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(17.6, 17.82)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out = '/root/provenance_weight_training/output/figures/fig4_rho_sensitivity.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
