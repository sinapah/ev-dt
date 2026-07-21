import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt = pd.read_csv(os.path.join(root, 'datasets', 'ground_truth.csv'), parse_dates=['timestamp_hour'])
base = pd.read_csv(os.path.join(root, 'results', 'results_baseline.csv'), parse_dates=['timestamp'])
dt = pd.read_csv(os.path.join(root, 'results', 'results_dt.csv'), parse_dates=['timestamp'])

caltech_gt = gt[gt['site_id'] == 'Caltech'][['timestamp_hour', 'average_queue_length']].copy()
jpl_gt = gt[gt['site_id'] == 'JPL'][['timestamp_hour', 'average_queue_length']].copy()

for df in [caltech_gt, jpl_gt]:
    df['week'] = df['timestamp_hour'].dt.isocalendar().week.astype(int) \
                 + 52 * (df['timestamp_hour'].dt.year - df['timestamp_hour'].dt.year.min())

c_gt = caltech_gt.groupby('week')['average_queue_length'].quantile(0.95)
j_gt = jpl_gt.groupby('week')['average_queue_length'].quantile(0.95)
c_base = base.groupby(base['t'] // 168)['sim_caltech_queue'].quantile(0.95)
j_base = base.groupby(base['t'] // 168)['sim_jpl_queue'].quantile(0.95)
c_dt = dt.groupby(dt['t'] // 168)['sim_caltech_queue'].quantile(0.95)
j_dt = dt.groupby(dt['t'] // 168)['sim_jpl_queue'].quantile(0.95)

n = min(len(c_gt), len(c_base), len(c_dt))
week_dates = caltech_gt.groupby('week')['timestamp_hour'].first().iloc[:n]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(week_dates, c_gt.iloc[:n], label='Actual (ground truth)', color='gray', lw=1.5, alpha=0.7)
ax1.plot(week_dates, c_base.iloc[:n], label='Baseline routing', color='orange', lw=1.5)
ax1.plot(week_dates, c_dt.iloc[:n], label='DT-assisted routing', color='green', lw=1.5)
ax1.set_ylabel('P95 Queue Length')
ax1.set_title('Caltech — Queue Over Time (weekly P95)')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(week_dates, j_gt.iloc[:n], label='Actual (ground truth)', color='gray', lw=1.5, alpha=0.7)
ax2.plot(week_dates, j_base.iloc[:n], label='Baseline routing', color='orange', lw=1.5)
ax2.plot(week_dates, j_dt.iloc[:n], label='DT-assisted routing', color='green', lw=1.5)
ax2.set_xlabel('Date')
ax2.set_ylabel('P95 Queue Length')
ax2.set_title('JPL — Queue Over Time (weekly P95)')
ax2.legend(); ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(os.path.join(root, 'plots', 'queue_comparison.png'), dpi=150)
print('Saved queue_comparison.png')
