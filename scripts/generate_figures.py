#!/usr/bin/env python3
"""Generate publication-ready figures for Provenance Weight Training paper."""
import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, 'figure_data.json')
DEFAULT_OUTPUT = '/root/provenance_weight_training/output/figures'


def load_data(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'{name}.{ext}')
        fig.savefig(p)
        print(f'  -> {p}')
    plt.close(fig)


def fig_tau_sweep(data, output_dir):
    """Figure 1: tau sensitivity sweep at rho=0.4."""
    d = data['tau_sweep']
    tau = np.array(d['tau'])
    ppl = np.array(d['ppl'])
    baseline = d['uniform_baseline']

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(tau, ppl, 'o-', color='#2166AC', linewidth=1.5, markersize=6,
            label='Adaptive (dual-constraint)', zorder=3)

    ax.axhline(y=baseline, color='#B2182B', linestyle='--', linewidth=1.2,
               label=f'Uniform baseline ({baseline:.4f})', zorder=2)

    ymin = min(ppl.min(), baseline) - 0.02
    ymax = max(ppl.max(), baseline) + 0.02
    ax.set_ylim(ymin, ymax)

    ax.axvspan(0.5, 0.8, alpha=0.07, color='#2166AC', zorder=1)
    mid_y = ymin + (ymax - ymin) * 0.05
    ax.text(0.65, mid_y, 'safe range', ha='center', va='bottom',
            fontsize=8, color='#2166AC', fontstyle='italic', alpha=0.8)

    opt_idx = int(np.argmin(ppl))
    ax.annotate(
        f'$\\tau^*={tau[opt_idx]}$\nPPL={ppl[opt_idx]:.4f}',
        xy=(tau[opt_idx], ppl[opt_idx]),
        xytext=(tau[opt_idx] + 0.07, ppl[opt_idx] + 0.012),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='#333', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.9))

    ax.set_xlabel(r'$\tau$ ($\alpha_{\mathrm{eff}}$ lower bound)')
    ax.set_ylabel('Eval Perplexity')
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()

    print('Fig 1: tau sweep')
    save_fig(fig, output_dir, 'fig1_tau_sweep')


def fig_b_curve(data, output_dir):
    """Figure 2: grid-b sweep with multi-seed mean/std."""
    d = data['grid_b_sweep']
    b_vals = np.array(d['b_values'])
    s42 = np.array(d['seed42'], dtype=float)
    s123 = np.array(d['seed123'], dtype=float)
    s456_raw = d['seed456']

    means = []
    stds = []
    for i in range(len(b_vals)):
        vals = [s42[i], s123[i]]
        if s456_raw[i] is not None:
            vals.append(s456_raw[i])
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.errorbar(b_vals, means, yerr=stds, fmt='o-', color='#2166AC',
                linewidth=1.5, markersize=6, capsize=3, capthick=1,
                label='Grid-b (mean ± std)', zorder=3)

    for b_val in d.get('pending_b', []):
        ax.axvline(x=b_val, color='#999', linestyle=':', linewidth=0.8,
                   alpha=0.5, zorder=1)
    if d.get('pending_b'):
        ax.axvline(x=d['pending_b'][0], color='#999', linestyle=':',
                   linewidth=0.8, alpha=0.5, label='Pending', zorder=1)

    ub = d['uniform_mean']
    ax.axhline(y=ub, color='#B2182B', linestyle='--', linewidth=1.2,
               label=f'Uniform ({ub:.4f})', zorder=2)

    ab = d['adaptive_mean']
    ax.axhline(y=ab, color='#4DAF4A', linestyle='-.', linewidth=1.5,
               label=f'Adaptive $b^*$ ({ab:.4f})', zorder=2)

    ax.set_xlabel('$b$ (downweighting strength)')
    ax.set_ylabel('Eval Perplexity')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.2, 5.5)
    fig.tight_layout()

    print('Fig 2: b curve')
    save_fig(fig, output_dir, 'fig2_b_curve')


def fig_method_comparison(data, output_dir):
    """Figure 3: method comparison bar chart with individual points."""
    d = data['method_comparison']
    keys = ['uniform', 'ess_only', 'adaptive_dual']
    labels = d['methods']

    means = [np.mean(d[k]) for k in keys]
    stds = [np.std(d[k], ddof=1) for k in keys]

    colors = ['#B2182B', '#F4A582', '#2166AC']
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.5, color=colors, edgecolor='#333',
                  linewidth=0.5, yerr=stds, capsize=4,
                  error_kw={'linewidth': 1, 'color': '#333'}, zorder=3)

    for i, k in enumerate(keys):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(d[k]))
        ax.scatter(x[i] + jitter, d[k], color='white', edgecolor=colors[i],
                   s=25, linewidth=1, zorder=4)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.003,
                f'{m:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Eval Perplexity')

    all_vals = [v for k in keys for v in d[k]]
    ax.set_ylim(min(all_vals) - 0.03, max(all_vals) + 0.04)

    delta_uni = means[0] - means[2]
    delta_ess = means[1] - means[2]
    ax.text(0.97, 0.95,
            f'$\\Delta$ vs Uniform: {delta_uni:+.4f}\n$\\Delta$ vs ESS-only: {delta_ess:+.4f}',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f0f0f0', ec='#ccc'))

    fig.tight_layout()

    print('Fig 3: method comparison')
    save_fig(fig, output_dir, 'fig3_method_comparison')


FIGURE_MAP = {
    'tau': fig_tau_sweep,
    'b_curve': fig_b_curve,
    'comparison': fig_method_comparison,
}


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='JSON data file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output directory')
    parser.add_argument('--figures', nargs='+', default=['all'],
                        choices=['all'] + list(FIGURE_MAP.keys()),
                        help='Which figures to generate')
    args = parser.parse_args()

    data = load_data(args.data)

    figs = list(FIGURE_MAP.keys()) if 'all' in args.figures else args.figures
    for name in figs:
        FIGURE_MAP[name](data, args.output)

    print(f'\nDone. Output: {args.output}')


if __name__ == '__main__':
    main()
