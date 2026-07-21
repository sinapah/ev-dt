import os

import numpy as np
import pandas as pd

from config import SITES, FL_AGGREGATION_INTERVAL
from digital_twin import DigitalTwin
from edge_node import EdgeNode
from environment import Environment
from federated_coordinator import FederatedCoordinator
from ggs_queue_simulator import GGsQueueSimulator
from scheduler import Scheduler
from session_pool import SessionPool

from evaluation import run_full_evaluation
from plots.generate_plots import generate_all_plots


class SimulationRunner:
    def __init__(self, data_path='ground_truth.csv'):
        self.env = Environment(data_path)
        self.edge_nodes = {site: EdgeNode(site) for site in SITES}
        self.coordinator = FederatedCoordinator()
        self.dt = DigitalTwin()
        self.scheduler = Scheduler()
        self.queue_sim = GGsQueueSimulator()
        data_dir = os.path.dirname(os.path.abspath(data_path))
        self.session_pool = SessionPool(data_dir)

    def compute_historical_split(self):
        self.env.reset()
        caltech_total = 0.0
        jpl_total = 0.0
        while not self.env.at_end():
            state = self.env.step()
            caltech_total += state['Caltech']['arrivals']
            jpl_total += state['JPL']['arrivals']
        self.env.reset()
        total = caltech_total + jpl_total
        return caltech_total / total if total > 0 else 0.5

    def train(self, max_steps=None):
        self.env.reset()
        print("Starting FL training phase...")
        step = 0
        while not self.env.at_end():
            state = self.env.step()
            for site in SITES:
                self.edge_nodes[site].update_history(state)
            if (step + 1) % FL_AGGREGATION_INTERVAL == 0:
                for site in SITES:
                    self.edge_nodes[site].train_local()
                self.coordinator.aggregate([self.edge_nodes[s] for s in SITES])
            step += 1
            if max_steps and step >= max_steps:
                break
        print(f"Training complete. Processed {step} timesteps.")
        hist_split = self.compute_historical_split()
        self.scheduler.set_historical_split(hist_split)
        print(f"Historical demand split (Caltech share): {hist_split:.3f}")

    def evaluate(self, max_steps=None, use_dt=True):
        self.env.reset()
        for site in SITES:
            self.edge_nodes[site].history = {k: [] for k in self.edge_nodes[site].history}

        first_gt = self.env.step()
        self.queue_sim.reset()

        sim_state = {
            site: {
                'arrivals': first_gt[site]['arrivals'],
                'queue_length': first_gt[site]['queue_length'],
                'waiting_time': first_gt[site]['waiting_time'],
                'utilization': first_gt[site]['utilization'],
                'service_time': first_gt[site]['service_time'],
                'active_sessions': first_gt[site]['active_sessions'],
                'completed_sessions': first_gt[site]['completed_sessions'],
            }
            for site in SITES
        }
        sim_state['timestamp'] = first_gt['timestamp']
        sim_state['total_incoming_demand'] = first_gt['total_incoming_demand']

        for site in SITES:
            self.edge_nodes[site].update_history(sim_state)

        results = []
        step = 0

        while True:
            predictions = self.coordinator.ensemble_predict(
                self.edge_nodes, timestamp=sim_state['timestamp']
            )

            sessions = self.session_pool.get_hour_sessions(step)
            total_demand = len(sessions)

            dt_output = self.dt.predict(sim_state, predictions, self.queue_sim, self.scheduler)
            routing = self.scheduler.route_sessions(
                sessions, dt_predictions=dt_output if use_dt else None
            )

            sim_next = self.queue_sim.step(routing)

            for site in SITES:
                self.dt.prediction_errors.append({
                    'step': step,
                    'timestamp': sim_state['timestamp'],
                    'site': site,
                    'predicted_queue': dt_output[site]['queue_length'],
                    'actual_queue': sim_next[site]['queue_length'],
                    'predicted_wait': dt_output[site]['waiting_time'],
                    'actual_wait': sim_next[site]['waiting_time'],
                    'predicted_util': dt_output[site]['utilization'],
                    'actual_util': sim_next[site]['utilization'],
                    'use_dt': use_dt,
                })

            row = {
                'timestamp': sim_state['timestamp'],
                't': step,
                'total_demand': total_demand,
                'ground_truth_caltech_arrivals': float(self.env._site_data['Caltech']['arrivals'][step]),
                'ground_truth_jpl_arrivals': float(self.env._site_data['JPL']['arrivals'][step]),
                'caltech_predicted_arrivals': predictions['Caltech'],
                'jpl_predicted_arrivals': predictions['JPL'],
                'caltech_dt_predicted_queue': dt_output['Caltech']['queue_length'],
                'jpl_dt_predicted_queue': dt_output['JPL']['queue_length'],
                'caltech_dt_predicted_wait': dt_output['Caltech']['waiting_time'],
                'jpl_dt_predicted_wait': dt_output['JPL']['waiting_time'],
                'caltech_dt_predicted_util': dt_output['Caltech']['utilization'],
                'jpl_dt_predicted_util': dt_output['JPL']['utilization'],
                'routed_to_caltech': len(routing['Caltech']),
                'routed_to_jpl': len(routing['JPL']),
                'sim_caltech_queue': sim_next['Caltech']['queue_length'],
                'sim_jpl_queue': sim_next['JPL']['queue_length'],
                'sim_caltech_wait': sim_next['Caltech']['waiting_time'],
                'sim_jpl_wait': sim_next['JPL']['waiting_time'],
                'sim_caltech_util': sim_next['Caltech']['utilization'],
                'sim_jpl_util': sim_next['JPL']['utilization'],
            }
            results.append(row)
            step += 1

            if self.env.at_end():
                break
            if max_steps and step >= max_steps:
                break

            next_gt = self.env.step()

            sim_state = {
                site: {
                    'arrivals': len(routing[site]),
                    'queue_length': sim_next[site]['queue_length'],
                    'waiting_time': sim_next[site]['waiting_time'],
                    'utilization': sim_next[site]['utilization'],
                    'service_time': next_gt[site]['service_time'],
                    'active_sessions': sim_next[site]['active_sessions'],
                    'completed_sessions': sim_next[site].get('completed_sessions', 0),
                }
                for site in SITES
            }
            sim_state['timestamp'] = next_gt['timestamp']
            sim_state['total_incoming_demand'] = total_demand

            for site in SITES:
                self.edge_nodes[site].update_history(sim_state)

        df = pd.DataFrame(results)
        return df

    def run(self, train_steps=None, eval_steps=None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(root, 'results')
        plots_dir = os.path.join(root, 'plots')

        self.train(max_steps=train_steps)
        print("\nEvaluating with DT-guided scheduler...")
        df_dt = self.evaluate(max_steps=eval_steps, use_dt=True)
        print(f"DT evaluation: {len(df_dt)} timesteps")
        pred_errors_dt = list(self.dt.prediction_errors)
        self.dt.prediction_errors = []

        print("\nEvaluating with baseline scheduler (no DT)...")
        df_baseline = self.evaluate(max_steps=eval_steps, use_dt=False)
        print(f"Baseline evaluation: {len(df_baseline)} timesteps")
        pred_errors_baseline = list(self.dt.prediction_errors)
        self.dt.prediction_errors = []

        df_dt.to_csv(os.path.join(results_dir, 'results_dt.csv'), index=False)
        df_baseline.to_csv(os.path.join(results_dir, 'results_baseline.csv'), index=False)
        pd.DataFrame(pred_errors_dt).to_csv(
            os.path.join(results_dir, 'prediction_errors_dt.csv'), index=False
        )
        pd.DataFrame(pred_errors_baseline).to_csv(
            os.path.join(results_dir, 'prediction_errors_baseline.csv'), index=False
        )

        run_full_evaluation(df_dt, df_baseline,
                           pd.DataFrame(pred_errors_dt),
                           pd.DataFrame(pred_errors_baseline),
                           results_dir)

        generate_all_plots(df_dt, df_baseline,
                          pd.DataFrame(pred_errors_dt),
                          plots_dir)

        return df_dt, df_baseline