import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_runner import SimulationRunner

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    runner = SimulationRunner(os.path.join(root, 'datasets', 'ground_truth.csv'))
    df_dt, df_baseline, summary = runner.run()
    df_dt.to_csv(os.path.join(root, 'results', 'results_dt.csv'), index=False)
    df_baseline.to_csv(os.path.join(root, 'results', 'results_baseline.csv'), index=False)
    summary.to_csv(os.path.join(root, 'results', 'evaluation_summary.csv'), index=False)
    print("\nResults saved to results/*.csv")


if __name__ == '__main__':
    main()
