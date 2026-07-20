import numpy as np
from config import SITES, NUM_CHARGERS


class QueueSimulator:
    def __init__(self):
        self.queue = {site: 0.0 for site in SITES}
        self.active_sessions = {site: 0.0 for site in SITES}

    def reset(self, initial_state):
        for site in SITES:
            self.queue[site] = initial_state[site]['queue_length']
            self.active_sessions[site] = initial_state[site]['active_sessions']

    def step(self, routed_arrivals, service_time_minutes):
        state = {}
        for site in SITES:
            n_chargers = NUM_CHARGERS[site]
            avg_service = service_time_minutes[site]
            if avg_service > 0:
                service_rate = n_chargers * (60.0 / avg_service)
                completed = min(self.active_sessions[site], service_rate)
            else:
                service_rate = float(n_chargers)
                completed = min(self.active_sessions[site], service_rate)

            arrivals = routed_arrivals[site]
            new_queue = max(0, self.queue[site] + arrivals - service_rate)
            new_active = min(n_chargers, self.active_sessions[site] + arrivals - completed)
            new_active = max(0, new_active)

            if new_queue > 0 and service_rate > 0:
                waiting_time = (new_queue / service_rate) * avg_service if avg_service > 0 else new_queue * 60.0
            else:
                waiting_time = 0.0

            utilization = new_active / n_chargers if n_chargers > 0 else 0.0

            self.queue[site] = new_queue
            self.active_sessions[site] = new_active

            state[site] = {
                'queue_length': new_queue,
                'waiting_time': waiting_time,
                'utilization': min(1.0, utilization),
                'active_sessions': new_active,
                'completed_sessions': completed,
                'service_time': avg_service,
            }
        return state
