import numpy as np
from scipy.stats import gaussian_kde

from config import SITES, KDE_HISTORY_WINDOW, KDE_N_SAMPLES, KDE_TEMP_SIGMA_HOUR, KDE_TEMP_SIGMA_DOW, KDE_BANDWIDTH


class DigitalTwin:
    def __init__(self):
        self.last_predictions = {}
        self.prediction_errors = []

        self.demand_history = []
        self._kde = None
        self._kde_dirty = False

    @staticmethod
    def _encode_timestamp(ts):
        hour = ts.astype('datetime64[h]').astype(int) % 24
        dow = ts.astype('datetime64[D]').astype(int) % 7
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * hour / 24.0),
            'dow_sin': np.sin(2 * np.pi * dow / 7.0),
            'dow_cos': np.cos(2 * np.pi * dow / 7.0),
            'hour': hour,
            'dow': dow,
        }

    def record_demand(self, natural_arrivals, timestamp):
        enc = self._encode_timestamp(timestamp)
        vector = [
            enc['hour_sin'], enc['hour_cos'],
            enc['dow_sin'], enc['dow_cos'],
            float(natural_arrivals.get('Caltech', 0)),
            float(natural_arrivals.get('JPL', 0)),
        ]
        self.demand_history.append(vector)
        if len(self.demand_history) > KDE_HISTORY_WINDOW:
            self.demand_history = self.demand_history[-KDE_HISTORY_WINDOW:]
        self._kde_dirty = True

    def _ensure_kde(self):
        if not self._kde_dirty and self._kde is not None:
            return
        if len(self.demand_history) < 10:
            self._kde = None
            return
        data = np.array(self.demand_history, dtype=np.float64).T
        try:
            self._kde = gaussian_kde(data, bw_method=KDE_BANDWIDTH)
            self._kde_dirty = False
        except Exception:
            self._kde = None

    def _synthesize(self, timestamp):
        self._ensure_kde()
        if self._kde is None or len(self.demand_history) < 10:
            fallback = self.demand_history[-1] if self.demand_history else [0, 0, 0, 0, 0, 0]
            return {
                'Caltech': max(0, fallback[4]),
                'JPL': max(0, fallback[5]),
            }

        target = self._encode_timestamp(timestamp)
        samples = self._kde.resample(KDE_N_SAMPLES)

        hour_samples = np.arctan2(samples[1], samples[0]) % (2 * np.pi)
        hour_samples = hour_samples * 24.0 / (2 * np.pi)
        dow_samples = np.arctan2(samples[3], samples[2]) % (2 * np.pi)
        dow_samples = dow_samples * 7.0 / (2 * np.pi)

        target_hour = np.arctan2(target['hour_sin'], target['hour_cos']) % (2 * np.pi)
        target_hour = target_hour * 24.0 / (2 * np.pi)
        target_dow = np.arctan2(target['dow_sin'], target['dow_cos']) % (2 * np.pi)
        target_dow = target_dow * 7.0 / (2 * np.pi)

        hour_diff = np.abs(hour_samples - target_hour)
        hour_diff = np.minimum(hour_diff, 24.0 - hour_diff)
        dow_diff = np.abs(dow_samples - target_dow)
        dow_diff = np.minimum(dow_diff, 7.0 - dow_diff)

        weights = np.exp(
            -(hour_diff ** 2) / (2 * KDE_TEMP_SIGMA_HOUR ** 2)
            - (dow_diff ** 2) / (2 * KDE_TEMP_SIGMA_DOW ** 2)
        )
        weights /= weights.sum()

        caltech_synth = float(np.sum(weights * samples[4]))
        jpl_synth = float(np.sum(weights * samples[5]))

        return {
            'Caltech': max(0.0, round(caltech_synth)),
            'JPL': max(0.0, round(jpl_synth)),
        }

    def predict(self, state, predicted_arrivals, ggs_sim, scheduler, use_synthetic=False):
        if use_synthetic or predicted_arrivals is None:
            predicted_arrivals = self._synthesize(state['timestamp'])
        self.last_used_predictions = dict(predicted_arrivals)

        saved = ggs_sim.get_state()

        synthetic_sessions = []
        for site in SITES:
            n = max(0, int(round(predicted_arrivals.get(site, 0))))
            avg_service = state[site].get('service_time', 60.0)
            for _ in range(n):
                synthetic_sessions.append({
                    'arrival_minute': 0.0,
                    'service_time_minutes': avg_service,
                    'natural_site': site,
                })

        baseline_routing = scheduler.route_sessions(synthetic_sessions, dt_predictions=None)
        results = ggs_sim.step(baseline_routing)
        ggs_sim.set_state(saved)

        self.last_predictions = results
        return results
