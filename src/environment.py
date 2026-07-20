import numpy as np
from config import SITES, STATE_FIELDS
from data_loader import load_ground_truth, get_site_series


class Environment:
    def __init__(self, path='ground_truth.csv'):
        self.data = load_ground_truth(path)
        self.timestamps = self.data['timestamp_hour'].values
        self.n_steps = len(self.timestamps)
        self._current_idx = -1

        self._site_data = {}
        for site in SITES:
            self._site_data[site] = {
                f: get_site_series(self.data, site, f)
                for f in STATE_FIELDS + ['service_time', 'active_sessions', 'completed_sessions']
            }

    def reset(self):
        self._current_idx = -1

    def step(self):
        self._current_idx += 1
        return self._get_state()

    @property
    def current_idx(self):
        return self._current_idx

    def at_end(self):
        return self._current_idx >= self.n_steps - 1

    def _get_state(self):
        t = self._current_idx
        state = {'timestamp': self.timestamps[t], 't': t}
        for site in SITES:
            state[site] = {
                f: float(self._site_data[site][f][t])
                for f in STATE_FIELDS
            }
            state[site]['service_time'] = float(self._site_data[site]['service_time'][t])
            state[site]['active_sessions'] = float(self._site_data[site]['active_sessions'][t])
            state[site]['completed_sessions'] = float(self._site_data[site]['completed_sessions'][t])
        total_demand = sum(state[s]['arrivals'] for s in SITES)
        state['total_incoming_demand'] = total_demand
        return state
