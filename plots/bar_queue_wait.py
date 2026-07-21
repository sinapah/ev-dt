import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt = pd.read_csv(os.path.join(root, 'datasets', 'ground_truth.csv'))
base = pd.read_csv(os.path.join(root, 'results', 'results_baseline.csv'))
dt = pd.read_csv(os.path.join(root, 'results', 'results_dt.csv'))

sites = ['Caltech', 'JPL']
colors = {'Ground truth': '#666666', 'Baseline': '#ff9900', 'DT-assisted': '#33aa33'}

def compute_site_metrics(df, prefix):
    q = df[f'sim_{prefix}_queue']
    nonzero = q[q > 0]
    return {
        'pct_queued': (q > 0).mean() * 100,
        'p95': q.quantile(0.95),
        'p99': q.quantile(0.99),
        'max': q.max(),
        'mean_cond': nonzero.mean() if len(nonzero) > 0 else 0,
        'total_queue_hours': q.sum(),
    }

def compute_gt_metrics(site):
    q = gt[gt['site_id'] == site]['average_queue_length']
    nonzero = q[q > 0]
    return {
        'pct_queued': (q > 0).mean() * 100,
        'p95': q.quantile(0.95),
        'p99': q.quantile(0.99),
        'max': q.max(),
        'mean_cond': nonzero.mean() if len(nonzero) > 0 else 0,
        'total_queue_hours': q.sum(),
    }

metrics = ['pct_queued', 'p95', 'mean_cond', 'total_queue_hours']
metric_labels = [
    '% hours with queue > 0',
    '95th percentile queue length',
    'Mean queue when non-zero',
    'Total queue-hours (thousands)',
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')

for row, site in enumerate(sites):
    prefix = site.lower()
    gt_m = compute_gt_metrics(site)
    bs_m = compute_site_metrics(base, prefix)
    dt_m = compute_site_metrics(dt, prefix)

    for col, (key, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row][col]
        factor = 1000 if key == 'total_queue_hours' else 1
        vals = {
            'Ground truth': gt_m[key] / factor,
            'Baseline': bs_m[key] / factor,
            'DT-assisted': dt_m[key] / factor,
        }
        x_pos = np.arange(len(vals))
        for i, (name, v) in enumerate(vals.items()):
            ax.bar(i, v, color=colors[name], alpha=0.85, label=name if row == 0 and col == 0 else '')
            ax.text(i, v * 1.02, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{site} — {label}', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([])
        ax.grid(axis='y', alpha=0.3)
        if row == 0:
            ax.set_xlabel('')
        if col == 0:
            ax.set_ylabel('Value')

handles = [plt.Rectangle((0,0),1,1, color=colors[n]) for n in colors]
fig.legend(handles, colors.keys(), loc='lower center', ncol=3, fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(root, 'plots', 'bar_queue_wait.png')
plt.savefig(out, dpi=150)
print(f'Saved {out}')
