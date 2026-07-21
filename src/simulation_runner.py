import numpy as np
import pandas as pd
from config import SITES, FL_AGGREGATION_INTERVAL
from environment import Environment
from edge_node import EdgeNode
from federated_coordinator import FederatedCoordinator
from digital_twin import DigitalTwin
from scheduler import Scheduler
from ggs_queue_simulator import GGsQueueSimulator


class SimulationRunner:
    def __init__(self, data_path='ground_truth.csv'):
        self.env = Environment(data_path)
        self.edge_nodes = {site: EdgeNode(site) for site in SITES}
        self.coordinator = FederatedCoordinator()
        self.dt = DigitalTwin()
        self.scheduler = Scheduler()
        self.queue_sim = GGsQueueSimulator()
        import pandas as pd
        gt_df = pd.read_csv(data_path)
        self.queue_sim.set_service_pools(gt_df)

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
        self.queue_sim.reset(first_gt)

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
            predictions = {}
            for site in SITES:
                predictions[site] = self.edge_nodes[site].predict_next_arrivals(
                    timestamp=sim_state['timestamp']
                )

            dt_output = self.dt.predict(sim_state, predictions)

            total_demand = int(round(sim_state['total_incoming_demand']))
            if use_dt:
                routing = self.scheduler.route_dt_guided(total_demand, dt_output)
            else:
                routing = self.scheduler.route_baseline(total_demand)

            service_times = {
                site: sim_state[site]['service_time']
                for site in SITES
            }
            routing_arrivals = {
                site: float(routing[site])
                for site in SITES
            }
            sim_next = self.queue_sim.step(routing_arrivals, service_times)

            row = {
                'timestamp': sim_state['timestamp'],
                't': step,
                'total_demand': total_demand,
                'ground_truth_caltech_arrivals': float(self.env._site_data['Caltech']['arrivals'][step]),
                'ground_truth_jpl_arrivals': float(self.env._site_data['JPL']['arrivals'][step]),
                'caltech_predicted_arrivals': predictions['Caltech'],
                'jpl_predicted_arrivals': predictions['JPL'],
                'caltech_dt_predicted_queue': dt_output['Caltech']['predicted_queue'],
                'jpl_dt_predicted_queue': dt_output['JPL']['predicted_queue'],
                'caltech_dt_predicted_wait': dt_output['Caltech']['predicted_waiting_time'],
                'jpl_dt_predicted_wait': dt_output['JPL']['predicted_waiting_time'],
                'caltech_dt_predicted_util': dt_output['Caltech']['predicted_utilization'],
                'jpl_dt_predicted_util': dt_output['JPL']['predicted_utilization'],
                'caltech_congestion': dt_output['Caltech']['congestion_score'],
                'jpl_congestion': dt_output['JPL']['congestion_score'],
                'routed_to_caltech': routing['Caltech'],
                'routed_to_jpl': routing['JPL'],
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
                    'arrivals': routing[site],
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
            sim_state['total_incoming_demand'] = next_gt['total_incoming_demand']

            for site in SITES:
                self.edge_nodes[site].update_history(sim_state)

        df = pd.DataFrame(results)
        return df

    def compute_metrics(self, df):
        metrics = {}
        for site in SITES:
            prefix = site.lower()
            actual = df[f'ground_truth_{prefix}_arrivals']
            predicted = df[f'{prefix}_predicted_arrivals']
            errors = actual - predicted
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            non_zero = actual[actual > 0]
            if len(non_zero) > 0:
                mape = float(np.mean(np.abs((non_zero - predicted[actual > 0]) / non_zero)) * 100)
            else:
                mape = float('nan')
            metrics[f'{site}_prediction_mae'] = round(mae, 4)
            metrics[f'{site}_prediction_rmse'] = round(rmse, 4)
            metrics[f'{site}_prediction_mape'] = round(mape, 4)
        for site in SITES:
            prefix = site.lower()
            actual_q = df[f'sim_{prefix}_queue']
            pred_q = df[f'{prefix}_dt_predicted_queue']
            metrics[f'{site}_queue_error'] = round(float(np.mean(np.abs(actual_q - pred_q))), 4)
            actual_w = df[f'sim_{prefix}_wait']
            pred_w = df[f'{prefix}_dt_predicted_wait']
            metrics[f'{site}_wait_error'] = round(float(np.mean(np.abs(actual_w - pred_w))), 4)
            actual_u = df[f'sim_{prefix}_util']
            pred_u = df[f'{prefix}_dt_predicted_util']
            metrics[f'{site}_util_error'] = round(float(np.mean(np.abs(actual_u - pred_u))), 4)
        metrics['avg_routed_to_caltech'] = round(float(df['routed_to_caltech'].mean()), 4)
        metrics['avg_routed_to_jpl'] = round(float(df['routed_to_jpl'].mean()), 4)
        balance = df['sim_caltech_util'] - df['sim_jpl_util']
        metrics['util_balance_mae'] = round(float(np.mean(np.abs(balance))), 4)

        all_queues = []
        for site in SITES:
            prefix = site.lower()
            q = df[f'sim_{prefix}_queue']
            all_queues.append(q)

            metrics[f'{site}_pct_hours_queued'] = round(float((q > 0).mean() * 100), 2)
            metrics[f'{site}_queue_p95'] = round(float(q.quantile(0.95)), 2)
            metrics[f'{site}_queue_p99'] = round(float(q.quantile(0.99)), 2)
            metrics[f'{site}_queue_max'] = round(float(q.max()), 2)
            nonzero = q[q > 0]
            metrics[f'{site}_queue_mean_cond'] = round(float(nonzero.mean()), 2) if len(nonzero) > 0 else 0.0
            metrics[f'{site}_total_queue_hours'] = round(float(q.sum()), 2)

        combined = pd.concat(all_queues, axis=1).sum(axis=1)
        metrics['total_pct_hours_queued'] = round(float((combined > 0).mean() * 100), 2)
        metrics['total_queue_p95'] = round(float(combined.quantile(0.95)), 2)
        metrics['total_queue_p99'] = round(float(combined.quantile(0.99)), 2)
        metrics['total_queue_max'] = round(float(combined.max()), 2)
        metrics['total_queue_mean_cond'] = round(float(combined[combined > 0].mean()), 2)
        metrics['total_queue_hours'] = round(float(combined.sum()), 2)
        return metrics

    def run(self, train_steps=None, eval_steps=None):
        self.train(max_steps=train_steps)
        print("\nEvaluating with DT-guided scheduler...")
        df_dt = self.evaluate(max_steps=eval_steps, use_dt=True)
        print(f"DT evaluation: {len(df_dt)} timesteps")
        print("\nEvaluating with baseline scheduler (no DT)...")
        df_baseline = self.evaluate(max_steps=eval_steps, use_dt=False)
        print(f"Baseline evaluation: {len(df_baseline)} timesteps")
        metrics_dt = self.compute_metrics(df_dt)
        metrics_base = self.compute_metrics(df_baseline)
        rows = []
        for k in metrics_dt:
            rows.append({'Metric': k, 'DT_Assisted': metrics_dt[k], 'Baseline': metrics_base[k]})
        summary = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        print(summary.to_string(index=False))
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
        return df_dt, df_baseline, summary
