from config import SITES, CAPTIVE_FRACTION


class Scheduler:
    def __init__(self, historical_split=None):
        self.historical_split = historical_split
        self.captive_fraction = CAPTIVE_FRACTION

    def set_historical_split(self, caltech_share):
        self.historical_split = caltech_share

    def route(self, arrivals, dt_predictions=None):
        hist = self.historical_split if self.historical_split is not None else 0.5

        captive = {}
        for site in SITES:
            captive[site] = int(round(arrivals[site] * self.captive_fraction))
        while sum(captive.values()) > sum(arrivals.values()):
            site = max(SITES, key=lambda s: captive[s])
            if captive[site] > 0:
                captive[site] -= 1

        flexible = int(round(sum(arrivals.values()) - sum(captive.values())))
        if flexible < 0:
            flexible = 0

        if dt_predictions is not None and flexible > 0:
            scores = {}
            for site in SITES:
                c = dt_predictions[site]['congestion_score']
                scores[site] = max(c, 0.01)
            inv = {s: 1.0 / scores[s] for s in SITES}
            total_inv = sum(inv.values())
            caltech_share = inv['Caltech'] / total_inv if total_inv > 0 else hist
        else:
            caltech_share = hist

        flex_cal = round(flexible * caltech_share)
        flex_jpl = flexible - flex_cal

        return {'Caltech': captive['Caltech'] + flex_cal, 'JPL': captive['JPL'] + flex_jpl}
