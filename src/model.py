import numpy as np
from sklearn.linear_model import LinearRegression


class ForecastingModel:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self._weights = None

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        self._weights = self._extract_weights()

    def predict(self, X):
        if not self.trained:
            return 0.0
        return float(self.model.predict(X.reshape(1, -1))[0])

    def _extract_weights(self):
        return {
            'coef': self.model.coef_.copy(),
            'intercept': np.array([self.model.intercept_]),
        }

    def get_weights(self):
        if self._weights is None:
            return None
        return self._weights

    def set_weights(self, weights):
        self.model.coef_ = weights['coef'].copy()
        self.model.intercept_ = weights['intercept'][0]
        self.trained = True
        self._weights = weights
