import numpy as np
from config import SITES, NUM_CHARGERS


class DigitalTwin:
    def __init__(self):
        self.last_predictions = {}

    def predict(self, state, predicted_arrivals):
        results = {}
        for site in SITES:
            s = state[site]
            n_chargers = NUM_CHARGERS[site]
            avg_service = s['service_time']
            if avg_service > 0:
                service_capacity = n_chargers * (60.0 / avg_service)
            else:
                service_capacity = float(n_chargers)

            pred_arr = predicted_arrivals.get(site, 0)
            current_queue = s['queue_length']
            future_queue = max(0, current_queue + pred_arr - service_capacity)

            current_wait = s['waiting_time']
            if service_capacity > 0 and current_queue > 0:
                predicted_wait = current_wait * (future_queue / max(current_queue, 1))
            else:
                predicted_wait = future_queue / max(service_capacity, 1) * 5.0

            current_util = s['utilization']
            predicted_util = min(1.0, current_util + (pred_arr - service_capacity) / max(n_chargers, 1) * 0.1)
            predicted_util = max(0.0, predicted_util)

            congestion = self._congestion_score(future_queue, predicted_wait, predicted_util, site)

            results[site] = {
                'predicted_queue': future_queue,
                'predicted_waiting_time': predicted_wait,
                'predicted_utilization': predicted_util,
                'congestion_score': congestion,
            }
        self.last_predictions = results
        return results

    def _congestion_score(self, queue, wait, util, site):
        max_q = 100.0
        max_w = 120.0
        norm_q = min(queue / max_q, 1.0)
        norm_w = min(wait / max_w, 1.0)
        norm_u = util
        return 0.3 * norm_q + 0.4 * norm_w + 0.3 * norm_u
