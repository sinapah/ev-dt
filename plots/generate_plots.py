import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from config import SITES

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def _ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def plot_fl_predictions(df, output_dir):
    _ensure_output_dir(output_dir)
    timestamps = pd.to_datetime(df['timestamp'])

    for site in SITES:
        prefix = site.lower()
        actual = df[f'ground_truth_{prefix}_arrivals'].values
        predicted = df[f'{prefix}_predicted_arrivals'].values

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(timestamps, actual, label='Actual arrivals', color='#2c7bb6', alpha=0.7, linewidth=0.8)
        ax.plot(timestamps, predicted, label='Predicted arrivals', color='#d7191c', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Arrivals per hour')
        ax.set_title(f'{site} — Actual vs Predicted Arrivals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig = ax.figure
        fig.autofmt_xdate()
        path = os.path.join(output_dir, f'fl_actual_vs_predicted_{site.lower()}.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(14, 5))
    for site in SITES:
        prefix = site.lower()
        actual = df[f'ground_truth_{prefix}_arrivals'].values
        predicted = df[f'{prefix}_predicted_arrivals'].values
        error = actual - predicted
        ax.plot(timestamps, error, label=f'{site} error', alpha=0.7, linewidth=0.6)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction Error (actual - predicted)')
    ax.set_title('Prediction Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig = ax.figure
    fig.autofmt_xdate()
    path = os.path.join(output_dir, 'fl_prediction_error_over_time.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dt_predictions(pred_errors_df, output_dir):
    _ensure_output_dir(output_dir)
    pred_errors_df = pred_errors_df.copy()
    pred_errors_df['timestamp'] = pd.to_datetime(pred_errors_df['timestamp'])

    fields = [
        ('queue', 'Queue Length', 'vehicles'),
        ('wait', 'Waiting Time', 'minutes'),
        ('util', 'Utilization', 'ratio'),
    ]

    for site in SITES:
        site_df = pred_errors_df[pred_errors_df['site'] == site]
        if len(site_df) == 0:
            continue
        timestamps = site_df['timestamp']

        for field, label, unit in fields:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(timestamps, site_df[f'actual_{field}'], label='Actual', color='#2c7bb6', alpha=0.7, linewidth=0.8)
            ax.plot(timestamps, site_df[f'predicted_{field}'], label='DT Predicted', color='#d7191c', alpha=0.7, linewidth=0.8)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{label} ({unit})')
            ax.set_title(f'{site} — Actual vs Predicted {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig = ax.figure
            fig.autofmt_xdate()
            path = os.path.join(output_dir, f'dt_{field}_comparison_{site.lower()}.png')
            fig.savefig(path, dpi=300)
            plt.close(fig)
            print(f"Saved {path}")


def plot_scheduler_comparison(df_dt, df_baseline, output_dir):
    _ensure_output_dir(output_dir)

    def _scenario_summary(df):
        all_queues = []
        all_waits = []
        for site in SITES:
            prefix = site.lower()
            all_queues.append(df[f'sim_{prefix}_queue'])
            all_waits.append(df[f'sim_{prefix}_wait'])
        combined_queue = pd.concat(all_queues, axis=1).sum(axis=1)
        combined_wait = pd.concat(all_waits, axis=1).mean(axis=1)
        util_caltech = df['sim_caltech_util']
        util_jpl = df['sim_jpl_util']
        balance = (util_caltech - util_jpl).abs()
        return {
            'avg_wait': combined_wait.mean(),
            'avg_queue': combined_queue.mean(),
            'peak_queue': combined_queue.max(),
            'avg_balance': balance.mean(),
        }

    base = _scenario_summary(df_baseline)
    dt = _scenario_summary(df_dt)

    categories = ['Avg Waiting Time\n(minutes)', 'Avg Queue Length\n(vehicles)', 'Peak Queue\n(vehicles)', 'Avg Load Imbalance\n(utilization diff)']
    keys = ['avg_wait', 'avg_queue', 'peak_queue', 'avg_balance']
    base_vals = [base[k] for k in keys]
    dt_vals = [dt[k] for k in keys]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, base_vals, width, label='Baseline', color='#e66101', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dt_vals, width, label='DT-Assisted', color='#5e3c99', alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value')
    ax.set_title('Baseline vs DT-Assisted: Scheduler Performance Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scheduler_comparison.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def generate_all_plots(df_dt, df_baseline, pred_errors_dt, output_dir):
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    print("\n[Plots 1-3] Federated Learning prediction plots...")
    plot_fl_predictions(df_dt, output_dir)

    print("\n[Plots 4-9] Digital Twin prediction plots...")
    plot_dt_predictions(pred_errors_dt, output_dir)

    print("\n[Plot 10] Scheduler comparison plot...")
    plot_scheduler_comparison(df_dt, df_baseline, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
