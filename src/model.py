import numpy as np
import xgboost as xgb


class ForecastingModel:
    def __init__(self):
        self.model = None
        self.trained = False
        self._weights = None

    def train(self, X, y):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
        )
        self.model.fit(X, y)
        self.trained = True
        self._weights = self._extract_weights()

    def predict(self, X):
        if not self.trained:
            return 0.0
        return max(0.0, float(self.model.predict(X.reshape(1, -1))[0]))

    def _extract_weights(self):
        return self.model.get_booster().save_raw('json')

    def get_weights(self):
        if self._weights is None:
            return None
        return self._weights

    def set_weights(self, weights):
        booster = xgb.Booster()
        booster.load_model(bytearray(weights))
        self.model = xgb.XGBRegressor()
        self.model._Booster = booster
        self.trained = True
        self._weights = weights