import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt = pd.read_csv(os.path.join(root, 'datasets', 'ground_truth.csv'))
base = pd.read_csv(os.path.join(root, 'results', 'results_baseline.csv'))
dt = pd.read_csv(os.path.join(root, 'results', 'results_dt.csv'))

caltech_gt = gt[gt['site_id'] == 'Caltech']['arrivals_per_hour'].mean()
jpl_gt = gt[gt['site_id'] == 'JPL']['arrivals_per_hour'].mean()
caltech_base = base['routed_to_caltech'].mean()
jpl_base = base['routed_to_jpl'].mean()
caltech_dt = dt['routed_to_caltech'].mean()
jpl_dt = dt['routed_to_jpl'].mean()

sites = ['Caltech', 'JPL']
x = np.arange(len(sites))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, [caltech_gt, jpl_gt], width, label='Ground truth', color='gray', alpha=0.8)
ax.bar(x, [caltech_base, jpl_base], width, label='Baseline routing', color='orange', alpha=0.8)
ax.bar(x + width, [caltech_dt, jpl_dt], width, label='DT-assisted routing', color='green', alpha=0.8)
ax.set_ylabel('Mean arrivals per hour')
ax.set_title('Hourly EV arrivals by site and routing strategy')
ax.set_xticks(x)
ax.set_xticklabels(sites)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = os.path.join(root, 'plots', 'bar_arrivals.png')
plt.savefig(out, dpi=150)
print(f'Saved {out}')
