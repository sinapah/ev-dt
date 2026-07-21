import numpy as np
from collections import defaultdict
from config import FEATURE_LAGS, ROLLING_WINDOWS, STATE_FIELDS
from model import ForecastingModel


class EdgeNode:
    def __init__(self, site):
        self.site = site
        self.model = ForecastingModel()
        self.history = defaultdict(list)

    def update_history(self, state):
        for field in STATE_FIELDS:
            self.history[field].append(state[self.site][field])
        self.history['timestamps'].append(state['timestamp'])

    @staticmethod
    def _ts_hour(ts):
        return ts.astype('datetime64[h]').astype(int) % 24

    @staticmethod
    def _ts_dow(ts):
        return ts.astype('datetime64[D]').astype(int) % 7

    def _build_features(self, timestamp=None):
        n = len(self.history['arrivals'])
        feats = []
        for field, lags in FEATURE_LAGS.items():
            series = self.history[field]
            for lag in lags:
                feats.append(series[-lag] if n > lag else 0.0)
        arrivals = self.history['arrivals']
        for w in ROLLING_WINDOWS:
            feats.append(np.mean(arrivals[-w:]) if n >= w else 0.0)
        feats.append(np.sum(arrivals[-24:]) if n >= 24 else 0.0)
        ts = timestamp if timestamp else self.history['timestamps'][-1]
        feats.append(self._ts_hour(ts) / 23.0)
        feats.append(self._ts_dow(ts) / 6.0)
        feats.append(1.0 if self._ts_dow(ts) >= 5 else 0.0)
        return np.array(feats, dtype=np.float32)

    def predict_next_arrivals(self, timestamp=None):
        feats = self._build_features(timestamp)
        if not self.model.trained:
            return 0.0
        return max(0.0, self.model.predict(feats))

    def train_local(self):
        max_lag = max(v for lags in FEATURE_LAGS.values() for v in lags)
        min_rolling = max(ROLLING_WINDOWS)
        min_lookback = max(max_lag, min_rolling)
        n = len(self.history['arrivals'])
        if n < min_lookback + 1:
            return
        X_list, y_list = [], []
        for t in range(min_lookback, n):
            feats = []
            for field, lags in FEATURE_LAGS.items():
                series = self.history[field]
                for lag in lags:
                    feats.append(series[t - lag])
            arrivals = self.history['arrivals']
            for w in ROLLING_WINDOWS:
                feats.append(np.mean(arrivals[t - w:t]))
            feats.append(np.sum(arrivals[t - 24:t]))
            feats.append(self._ts_hour(self.history['timestamps'][t]) / 23.0)
            feats.append(self._ts_dow(self.history['timestamps'][t]) / 6.0)
            feats.append(1.0 if self._ts_dow(self.history['timestamps'][t]) >= 5 else 0.0)
            X_list.append(feats)
            y_list.append(self.history['arrivals'][t])
        if len(X_list) < 5:
            return
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        self.model.train(X, y)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
