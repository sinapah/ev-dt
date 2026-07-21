import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_runner import SimulationRunner

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    runner = SimulationRunner(os.path.join(root, 'datasets', 'ground_truth.csv'))
    df_dt, df_baseline = runner.run()
    print("\nAll results, metrics, and plots saved successfully.")


if __name__ == '__main__':
    main()
