import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

tau =      [0.5,    0.6,    0.7,    0.8,    0.9,    0.95]
ppl =      [17.688, 17.6805,17.6793,17.6527,17.712, 17.723]
b_mean =   [0.3923, 0.5899, 0.8943, 1.399,  2.36,   3.0775]

uniform_ppl = 17.6939
golden_ppl  = 17.654
ess_ppl     = 17.73

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

# Safe range background
ax1.axvspan(0.5, 0.8, color='#d4edda', alpha=0.45, zorder=0)
ax1.text(0.65, 17.734, 'safe range', fontsize=8, color='#2d6a4f',
         ha='center', va='bottom', style='italic')

# Reference lines
ax1.axhline(uniform_ppl, color='grey', ls='--', lw=1.0, zorder=1)
ax1.axhline(golden_ppl, color='#DAA520', ls='--', lw=1.0, zorder=1)

ax1.text(0.96, uniform_ppl + 0.0012, 'Uniform', fontsize=7, color='grey',
         ha='right', va='bottom', transform=ax1.get_yaxis_transform())
ax1.text(0.96, golden_ppl + 0.0012, 'Golden Ratio', fontsize=7, color='#DAA520',
         ha='right', va='bottom', transform=ax1.get_yaxis_transform())

# PPL line (left axis)
# Split into success / negative
success_idx = [0,1,2,3]
neg_idx = [4,5]

tau_s = [tau[i] for i in success_idx]
ppl_s = [ppl[i] for i in success_idx]
tau_n = [tau[i] for i in neg_idx]
ppl_n = [ppl[i] for i in neg_idx]

ax1.plot(tau, ppl, '-', color='#2166ac', lw=1.5, zorder=3)
ax1.plot(tau_s, ppl_s, 'o', color='#2166ac', ms=5, zorder=4, label='PPL (holdout)')
ax1.plot(tau_n, ppl_n, 'o', color='#aaaaaa', ms=5, zorder=4)

# Optimal marker
ax1.plot(0.8, 17.6527, '*', color='#2166ac', ms=11, zorder=5,
         markeredgecolor='black', markeredgewidth=0.5)
ax1.annotate('optimal', xy=(0.8, 17.6527), xytext=(0.72, 17.645),
             fontsize=7, color='#2166ac', ha='center',
             arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8))

ax1.set_xlabel(r'$\tau$', fontsize=11)
ax1.set_ylabel('PPL (holdout)', fontsize=10, color='#2166ac')
ax1.tick_params(axis='y', labelcolor='#2166ac', labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.set_xlim(0.45, 1.0)
ax1.set_ylim(17.635, 17.74)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Right axis: b*_mean
ax2 = ax1.twinx()

b_s = [b_mean[i] for i in success_idx]
b_n = [b_mean[i] for i in neg_idx]

ax2.plot(tau, b_mean, '--', color='#b2182b', lw=1.5, zorder=3)
ax2.plot(tau_s, b_s, 's', color='#b2182b', ms=5, markerfacecolor='white',
         markeredgewidth=1.2, zorder=4, label=r'$\bar{b}^*$')
ax2.plot(tau_n, b_n, 's', color='#aaaaaa', ms=5, markerfacecolor='white',
         markeredgewidth=1.2, zorder=4)

ax2.plot(0.8, 1.399, '*', color='#b2182b', ms=11, zorder=5,
         markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylabel(r'$\bar{b}^*$', fontsize=11, color='#b2182b')
ax2.tick_params(axis='y', labelcolor='#b2182b', labelsize=9)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
           loc='upper left', framealpha=0.85, edgecolor='none')

plt.tight_layout(pad=0.4)
fig.savefig('/root/provenance_weight_training/figures/tau_sweep.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('/root/provenance_weight_training/figures/tau_sweep.png',
            bbox_inches='tight', dpi=300)
plt.close()
print('DONE')
