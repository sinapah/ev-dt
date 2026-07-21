import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import run_full_evaluation
from plots.generate_plots import generate_all_plots


def run_from_csv(results_dir, plots_dir):
    print("Loading results from CSV files...")
    df_dt = pd.read_csv(os.path.join(results_dir, 'results_dt.csv'))
    df_baseline = pd.read_csv(os.path.join(results_dir, 'results_baseline.csv'))
    pred_errors_dt = pd.read_csv(os.path.join(results_dir, 'prediction_errors_dt.csv'))
    pred_errors_baseline = pd.read_csv(os.path.join(results_dir, 'prediction_errors_baseline.csv'))

    run_full_evaluation(df_dt, df_baseline, pred_errors_dt, pred_errors_baseline, results_dir)
    generate_all_plots(df_dt, df_baseline, pred_errors_dt, plots_dir)


def run_simulation(train_steps=None, eval_steps=None):
    from simulation_runner import SimulationRunner
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner = SimulationRunner(os.path.join(root, 'datasets', 'ground_truth.csv'))
    df_dt, df_baseline = runner.run()
    return df_dt, df_baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation and generate plots for EV-DT simulation')
    parser.add_argument('--from-csv', action='store_true',
                       help='Regenerate metrics and plots from existing CSV results (skip simulation)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: ../results)')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Plots directory (default: ../plots)')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(root, 'results')
    plots_dir = args.plots_dir or os.path.join(root, 'plots')

    if args.from_csv:
        run_from_csv(results_dir, plots_dir)
    else:
        run_simulation()
