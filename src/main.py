import sys
sys.path.insert(0, 'src')

from simulation_runner import SimulationRunner


def main():
    runner = SimulationRunner('ground_truth.csv')
    # Use all data for training and evaluation
    df_dt, df_baseline, summary = runner.run()
    df_dt.to_csv('results_dt.csv', index=False)
    df_baseline.to_csv('results_baseline.csv', index=False)
    summary.to_csv('evaluation_summary.csv', index=False)
    print("\nResults saved to results_dt.csv, results_baseline.csv, evaluation_summary.csv")


if __name__ == '__main__':
    main()
