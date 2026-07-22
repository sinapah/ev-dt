import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.metrics import r2_score

from config import SITES


def compute_fl_metrics(df, output_dir):
    metrics = {}
    all_actual = []
    all_predicted = []

    for site in SITES:
        prefix = site.lower()
        actual = df[f'ground_truth_{prefix}_arrivals'].values
        predicted = df[f'{prefix}_predicted_arrivals'].values
        errors = actual - predicted

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        non_zero_mask = actual > 0
        if non_zero_mask.sum() > 0:
            mape = float(np.mean(np.abs(errors[non_zero_mask] / actual[non_zero_mask])) * 100)
        else:
            mape = float('nan')
        r2 = float(r2_score(actual, predicted))

        metrics[f'{site}_mae'] = round(mae, 4)
        metrics[f'{site}_rmse'] = round(rmse, 4)
        metrics[f'{site}_mape'] = round(mape, 4)
        metrics[f'{site}_r2'] = round(r2, 4)

        all_actual.extend(actual)
        all_predicted.extend(predicted)

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_errors = all_actual - all_predicted
    overall_mae = float(np.mean(np.abs(all_errors)))
    overall_rmse = float(np.sqrt(np.mean(all_errors ** 2)))
    non_zero_mask = all_actual > 0
    if non_zero_mask.sum() > 0:
        overall_mape = float(np.mean(np.abs(all_errors[non_zero_mask] / all_actual[non_zero_mask])) * 100)
    else:
        overall_mape = float('nan')
    overall_r2 = float(r2_score(all_actual, all_predicted))

    rows = []
    for site in SITES:
        rows.append({
            'site': site,
            'mae': metrics[f'{site}_mae'],
            'rmse': metrics[f'{site}_rmse'],
            'mape': metrics[f'{site}_mape'],
            'r2': metrics[f'{site}_r2'],
        })
    rows.append({
        'site': 'Overall',
        'mae': round(overall_mae, 4),
        'rmse': round(overall_rmse, 4),
        'mape': round(overall_mape, 4),
        'r2': round(overall_r2, 4),
    })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(output_dir, 'fl_metrics.csv'), index=False)
    print(f"FL metrics saved to {os.path.join(output_dir, 'fl_metrics.csv')}")
    return result_df


def compute_dt_metrics(pred_errors_df, output_dir):
    metrics = {}
    for site in SITES:
        site_df = pred_errors_df[pred_errors_df['site'] == site]
        if len(site_df) == 0:
            continue

        q_err = site_df['predicted_queue'] - site_df['actual_queue']
        w_err = site_df['predicted_wait'] - site_df['actual_wait']
        u_err = site_df['predicted_util'] - site_df['actual_util']

        metrics[f'{site}_queue_mae'] = round(float(np.mean(np.abs(q_err))), 4)
        metrics[f'{site}_queue_rmse'] = round(float(np.sqrt(np.mean(q_err ** 2))), 4)
        metrics[f'{site}_wait_mae'] = round(float(np.mean(np.abs(w_err))), 4)
        metrics[f'{site}_wait_rmse'] = round(float(np.sqrt(np.mean(w_err ** 2))), 4)
        metrics[f'{site}_util_mae'] = round(float(np.mean(np.abs(u_err))), 4)
        metrics[f'{site}_util_rmse'] = round(float(np.sqrt(np.mean(u_err ** 2))), 4)

    rows = []
    for site in SITES:
        rows.append({
            'site': site,
            'queue_mae': metrics[f'{site}_queue_mae'],
            'queue_rmse': metrics[f'{site}_queue_rmse'],
            'wait_mae': metrics[f'{site}_wait_mae'],
            'wait_rmse': metrics[f'{site}_wait_rmse'],
            'util_mae': metrics[f'{site}_util_mae'],
            'util_rmse': metrics[f'{site}_util_rmse'],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(output_dir, 'dt_metrics.csv'), index=False)
    print(f"DT metrics saved to {os.path.join(output_dir, 'dt_metrics.csv')}")
    return result_df


def compute_scheduler_metrics(df_dt, df_baseline, output_dir, df_disconnected=None):
    def _scenario_metrics(df, label):
        m = {}
        all_queues = []
        all_waits = []
        all_utils = []
        for site in SITES:
            prefix = site.lower()
            q = df[f'sim_{prefix}_queue']
            w = df[f'sim_{prefix}_wait']
            u = df[f'sim_{prefix}_util']
            all_queues.append(q)
            all_waits.append(w)
            all_utils.append(u)
            m[f'{site}_avg_queue'] = round(float(q.mean()), 4)
            m[f'{site}_avg_wait'] = round(float(w.mean()), 4)
            m[f'{site}_peak_queue'] = round(float(q.max()), 2)
            m[f'{site}_avg_util'] = round(float(u.mean()), 4)

        combined_queue = pd.concat(all_queues, axis=1).sum(axis=1)
        combined_wait = pd.concat(all_waits, axis=1).mean(axis=1)
        combined_util = pd.concat(all_utils, axis=1).mean(axis=1)

        m['avg_waiting_time'] = round(float(combined_wait.mean()), 4)
        m['avg_queue_length'] = round(float(combined_queue.mean()), 4)
        m['peak_queue_length'] = round(float(combined_queue.max()), 2)
        m['avg_utilization'] = round(float(combined_util.mean()), 4)

        util_caltech = df[f'sim_caltech_util']
        util_jpl = df[f'sim_jpl_util']
        balance = (util_caltech - util_jpl).abs()
        m['avg_load_imbalance'] = round(float(balance.mean()), 4)
        m['max_load_imbalance'] = round(float(balance.max()), 4)

        return m

    dt_m = _scenario_metrics(df_dt, 'DT_Assisted')
    base_m = _scenario_metrics(df_baseline, 'Baseline')

    if df_disconnected is not None:
        disc_m = _scenario_metrics(df_disconnected, 'DT_Disconnected')
        scenarios = {
            'Baseline': base_m,
            'DT_Assisted': dt_m,
            'DT_Disconnected': disc_m,
        }
    else:
        scenarios = {
            'Baseline': base_m,
            'DT_Assisted': dt_m,
        }

    rows = []
    for key in dt_m:
        row = {'metric': key}
        for scenario_name, scenario_data in scenarios.items():
            row[scenario_name] = scenario_data[key]
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(output_dir, 'scheduler_metrics.csv'), index=False)
    print(f"Scheduler metrics saved to {os.path.join(output_dir, 'scheduler_metrics.csv')}")
    return result_df


def compute_improvement_metrics(scheduler_metrics_df, output_dir):
    metrics = {}
    scenarios = [c for c in scheduler_metrics_df.columns if c != 'metric']
    baseline_col = 'Baseline'

    for scenario in scenarios:
        if scenario == baseline_col:
            continue
        for _, row in scheduler_metrics_df.iterrows():
            metric_name = row['metric']
            baseline = row[baseline_col]
            scen_val = row[scenario]
            if baseline != 0:
                improvement = 100.0 * (baseline - scen_val) / baseline
            else:
                improvement = 0.0
            metrics[f'{scenario}_{metric_name}'] = round(improvement, 2)

    rows = [
        {
            'metric': 'Waiting Time Reduction (%)',
            **{s: metrics.get(f'{s}_avg_waiting_time', 0) for s in scenarios if s != baseline_col},
        },
        {
            'metric': 'Queue Length Reduction (%)',
            **{s: metrics.get(f'{s}_avg_queue_length', 0) for s in scenarios if s != baseline_col},
        },
        {
            'metric': 'Peak Queue Reduction (%)',
            **{s: metrics.get(f'{s}_peak_queue_length', 0) for s in scenarios if s != baseline_col},
        },
        {
            'metric': 'Load Balance Improvement (%)',
            **{s: metrics.get(f'{s}_avg_load_imbalance', 0) for s in scenarios if s != baseline_col},
        },
    ]

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(output_dir, 'improvement_metrics.csv'), index=False)
    print(f"Improvement metrics saved to {os.path.join(output_dir, 'improvement_metrics.csv')}")
    return result_df


def compute_synthetic_validation(df_disconnected, output_dir):
    disc = df_disconnected[df_disconnected['connection_lost'] == True].copy()
    if len(disc) == 0:
        print("No disconnected timesteps found — skipping synthetic validation.")
        return pd.DataFrame()

    rows = []
    for site in SITES:
        prefix = site.lower()
        edge_arr = disc[f'{prefix}_predicted_arrivals'].values
        synth_arr = disc[f'{prefix}_synthetic_arrivals'].values
        gt_arr = disc[f'ground_truth_{prefix}_arrivals'].values

        edge_synth_ks = ks_2samp(edge_arr, synth_arr)
        synth_gt_ks = ks_2samp(synth_arr, gt_arr)

        edge_hist, bin_edges = np.histogram(edge_arr, bins=30, density=True)
        synth_hist, _ = np.histogram(synth_arr, bins=bin_edges, density=True)
        gt_hist, _ = np.histogram(gt_arr, bins=bin_edges, density=True)

        eps = 1e-12
        edge_synth_js = 0.5 * (
            entropy(edge_hist + eps, (edge_hist + synth_hist) / 2 + eps)
            + entropy(synth_hist + eps, (edge_hist + synth_hist) / 2 + eps)
        )
        synth_gt_js = 0.5 * (
            entropy(synth_hist + eps, (synth_hist + gt_hist) / 2 + eps)
            + entropy(gt_hist + eps, (synth_hist + gt_hist) / 2 + eps)
        )

        synth_vs_edge_mae = float(np.mean(np.abs(synth_arr - edge_arr)))
        synth_vs_gt_mae = float(np.mean(np.abs(synth_arr - gt_arr)))

        edge_ac = float(np.corrcoef(edge_arr[:-1], edge_arr[1:])[0, 1]) if len(edge_arr) > 1 else 0
        synth_ac = float(np.corrcoef(synth_arr[:-1], synth_arr[1:])[0, 1]) if len(synth_arr) > 1 else 0
        gt_ac = float(np.corrcoef(gt_arr[:-1], gt_arr[1:])[0, 1]) if len(gt_arr) > 1 else 0

        rows.append({
            'site': site,
            'edge_vs_synth_ks_stat': f'{edge_synth_ks.statistic:.4f}',
            'edge_vs_synth_ks_pval': f'{edge_synth_ks.pvalue:.4e}',
            'synth_vs_gt_ks_stat': f'{synth_gt_ks.statistic:.4f}',
            'synth_vs_gt_ks_pval': f'{synth_gt_ks.pvalue:.4e}',
            'edge_vs_synth_js_div': f'{edge_synth_js:.6f}',
            'synth_vs_gt_js_div': f'{synth_gt_js:.6f}',
            'synth_vs_edge_mae': round(synth_vs_edge_mae, 4),
            'synth_vs_gt_mae': round(synth_vs_gt_mae, 4),
            'edge_autocorr': round(edge_ac, 4),
            'synth_autocorr': round(synth_ac, 4),
            'gt_autocorr': round(gt_ac, 4),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(output_dir, 'synthetic_validation.csv'), index=False)
    print(f"Synthetic validation saved to {os.path.join(output_dir, 'synthetic_validation.csv')}")
    return result_df


def generate_summary_report(fl_metrics, dt_metrics, scheduler_metrics, improvement_metrics,
                            output_dir, synthetic_validation=None):
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE EVALUATION SUMMARY REPORT")
    lines.append("=" * 80)

    lines.append("\n--- 1. Federated Demand Prediction ---")
    for _, row in fl_metrics.iterrows():
        lines.append(f"  {row['site']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, MAPE={row['mape']:.2f}%, R²={row['r2']:.4f}")

    lines.append("\n--- 2. Digital Twin Prediction ---")
    for _, row in dt_metrics.iterrows():
        lines.append(f"  {row['site']}: Queue MAE={row['queue_mae']:.4f}, Queue RMSE={row['queue_rmse']:.4f}, "
                     f"Wait MAE={row['wait_mae']:.4f}, Wait RMSE={row['wait_rmse']:.4f}, "
                     f"Util MAE={row['util_mae']:.4f}, Util RMSE={row['util_rmse']:.4f}")

    lines.append("\n--- 3. Scheduler Impact ---")
    for _, row in scheduler_metrics.iterrows():
        parts = [f"  {row['metric']}"]
        for col in scheduler_metrics.columns:
            if col == 'metric':
                continue
            parts.append(f"{col}={row[col]}")
        lines.append(', '.join(parts))

    lines.append("\n--- 4. Improvement Metrics (vs Baseline) ---")
    for _, row in improvement_metrics.iterrows():
        parts = [f"  {row['metric']}"]
        for col in improvement_metrics.columns:
            if col == 'metric':
                continue
            parts.append(f"{col}={row[col]:.2f}%")
        lines.append(', '.join(parts))

    lines.append("\n--- 5. Key Findings ---")
    fl_overall = fl_metrics[fl_metrics['site'] == 'Overall']
    if len(fl_overall) > 0:
        r = fl_overall.iloc[0]
        lines.append(f"  FL Prediction: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, R²={r['r2']:.4f}")

    for _, row in improvement_metrics.iterrows():
        parts = [f"  {row['metric']}"]
        for col in improvement_metrics.columns:
            if col == 'metric':
                continue
            parts.append(f"{col}={row[col]:.2f}%")
        lines.append(', '.join(parts))

    if synthetic_validation is not None and len(synthetic_validation) > 0:
        lines.append("\n--- 6. Synthetic Data Validation (Disconnected Phase) ---")
        for _, row in synthetic_validation.iterrows():
            lines.append(f"  {row['site']}:")
            lines.append(f"    Edge vs Synth KS={row['edge_vs_synth_ks_stat']} "
                         f"(p={row['edge_vs_synth_ks_pval']}), JS={row['edge_vs_synth_js_div']}, "
                         f"MAE={row['synth_vs_edge_mae']}")
            lines.append(f"    Synth vs Truth KS={row['synth_vs_gt_ks_stat']} "
                         f"(p={row['synth_vs_gt_ks_pval']}), JS={row['synth_vs_gt_js_div']}, "
                         f"MAE={row['synth_vs_gt_mae']}")
            lines.append(f"    Autocorrelation: Edge={row['edge_autocorr']}, "
                         f"Synth={row['synth_autocorr']}, GroundTruth={row['gt_autocorr']}")

    report = '\n'.join(lines)
    print(report)

    report_path = os.path.join(output_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSummary report saved to {report_path}")

    return report


def run_full_evaluation(df_dt, df_baseline, pred_errors_dt, pred_errors_baseline, output_dir,
                        df_disconnected=None, pred_errors_disconnected=None):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 80)

    print("\n[1/4] Computing Federated Learning metrics...")
    fl_metrics = compute_fl_metrics(df_dt, output_dir)

    print("\n[2/4] Computing Digital Twin metrics...")
    dt_metrics = compute_dt_metrics(pred_errors_dt, output_dir)

    print("\n[3/4] Computing Scheduler metrics...")
    scheduler_metrics = compute_scheduler_metrics(
        df_dt, df_baseline, output_dir, df_disconnected=df_disconnected
    )

    print("\n[4/4] Computing Improvement metrics...")
    improvement_metrics = compute_improvement_metrics(scheduler_metrics, output_dir)

    synthetic_validation = None
    if df_disconnected is not None:
        print("\n[Extra] Computing Synthetic Data Validation...")
        synthetic_validation = compute_synthetic_validation(df_disconnected, output_dir)

    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    generate_summary_report(
        fl_metrics, dt_metrics, scheduler_metrics, improvement_metrics,
        output_dir, synthetic_validation=synthetic_validation
    )

    return {
        'fl_metrics': fl_metrics,
        'dt_metrics': dt_metrics,
        'scheduler_metrics': scheduler_metrics,
        'improvement_metrics': improvement_metrics,
        'synthetic_validation': synthetic_validation,
    }
