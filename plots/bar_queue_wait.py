import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt = pd.read_csv(os.path.join(root, 'datasets', 'ground_truth.csv'))
base = pd.read_csv(os.path.join(root, 'results', 'results_baseline.csv'))
dt = pd.read_csv(os.path.join(root, 'results', 'results_dt.csv'))

caltech_gt_q = gt[gt['site_id'] == 'Caltech']['average_queue_length'].mean()
jpl_gt_q = gt[gt['site_id'] == 'JPL']['average_queue_length'].mean()
caltech_gt_w = gt[gt['site_id'] == 'Caltech']['average_waiting_time_minutes'].mean()
jpl_gt_w = gt[gt['site_id'] == 'JPL']['average_waiting_time_minutes'].mean()

caltech_base_q = base['sim_caltech_queue'].mean()
jpl_base_q = base['sim_jpl_queue'].mean()
caltech_dt_q = dt['sim_caltech_queue'].mean()
jpl_dt_q = dt['sim_jpl_queue'].mean()

caltech_base_w = base['sim_caltech_wait'].mean()
jpl_base_w = base['sim_jpl_wait'].mean()
caltech_dt_w = dt['sim_caltech_wait'].mean()
jpl_dt_w = dt['sim_jpl_wait'].mean()

sites = ['Caltech', 'JPL']
x = np.arange(len(sites))
width = 0.25

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.bar(x - width, [caltech_gt_q, jpl_gt_q], width, label='Ground truth', color='gray', alpha=0.8)
ax1.bar(x, [caltech_base_q, jpl_base_q], width, label='Baseline routing', color='orange', alpha=0.8)
ax1.bar(x + width, [caltech_dt_q, jpl_dt_q], width, label='DT-assisted routing', color='green', alpha=0.8)
ax1.set_ylabel('Mean queue length')
ax1.set_title('Average queue length by site and routing strategy')
ax1.set_xticks(x)
ax1.set_xticklabels(sites)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x - width, [caltech_gt_w, jpl_gt_w], width, label='Ground truth', color='gray', alpha=0.8)
ax2.bar(x, [caltech_base_w, jpl_base_w], width, label='Baseline routing', color='orange', alpha=0.8)
ax2.bar(x + width, [caltech_dt_w, jpl_dt_w], width, label='DT-assisted routing', color='green', alpha=0.8)
ax2.set_ylabel('Mean waiting time (minutes)')
ax2.set_title('Average waiting time by site and routing strategy')
ax2.set_xticks(x)
ax2.set_xticklabels(sites)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = os.path.join(root, 'plots', 'bar_queue_wait.png')
plt.savefig(out, dpi=150)
print(f'Saved {out}')
