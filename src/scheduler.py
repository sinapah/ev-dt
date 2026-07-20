import numpy as np
from config import SITES


class Scheduler:
    def __init__(self, historical_split=None):
        self.historical_split = historical_split

    def set_historical_split(self, caltech_share):
        self.historical_split = caltech_share

    def route_baseline(self, total_demand):
        if self.historical_split is None:
            caltech_share = 0.5
        else:
            caltech_share = self.historical_split
        caltech_routed = round(total_demand * caltech_share)
        jpl_routed = total_demand - caltech_routed
        return {'Caltech': caltech_routed, 'JPL': jpl_routed}

    def route_dt_guided(self, total_demand, dt_predictions):
        scores = {}
        for site in SITES:
            c = dt_predictions[site]['congestion_score']
            scores[site] = max(c, 0.01)
        inv = {s: 1.0 / scores[s] for s in SITES}
        total_inv = sum(inv.values())
        if total_inv == 0:
            return self.route_baseline(total_demand)
        caltech_share = inv['Caltech'] / total_inv
        caltech_routed = round(total_demand * caltech_share)
        jpl_routed = total_demand - caltech_routed
        return {'Caltech': caltech_routed, 'JPL': jpl_routed}
