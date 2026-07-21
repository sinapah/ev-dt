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
                w = dt_predictions[site]['waiting_time']
                scores[site] = max(w, 0.01)
            inv = {s: 1.0 / scores[s] for s in SITES}
            total_inv = sum(inv.values())
            caltech_share = inv['Caltech'] / total_inv if total_inv > 0 else hist
        else:
            caltech_share = hist

        flex_cal = round(flexible * caltech_share)
        flex_jpl = flexible - flex_cal

        return {'Caltech': captive['Caltech'] + flex_cal, 'JPL': captive['JPL'] + flex_jpl}

    def route_sessions(self, sessions, dt_predictions=None):
        per_site = {site: [] for site in SITES}
        for s in sessions:
            per_site[s['natural_site']].append(s)

        for site in SITES:
            per_site[site].sort(key=lambda s: s['arrival_minute'])

        captive_counts = {}
        for site in SITES:
            captive_counts[site] = int(round(len(per_site[site]) * self.captive_fraction))
        while sum(captive_counts.values()) > sum(len(v) for v in per_site.values()):
            site = max(SITES, key=lambda s: captive_counts[s])
            if captive_counts[site] > 0:
                captive_counts[site] -= 1

        captive = {site: [] for site in SITES}
        flexible = []
        for site in SITES:
            n = captive_counts[site]
            site_sessions = per_site[site]
            captive[site] = site_sessions[:n]
            flexible.extend(site_sessions[n:])

        if dt_predictions is not None and len(flexible) > 0:
            scores = {}
            for site in SITES:
                scores[site] = max(dt_predictions[site]['waiting_time'], 0.01)
            inv = {s: 1.0 / scores[s] for s in SITES}
            total_inv = sum(inv.values())
            caltech_share = inv['Caltech'] / total_inv
        else:
            caltech_share = self.historical_split if self.historical_split is not None else 0.5

        flex_cal = round(len(flexible) * caltech_share)
        result = {
            'Caltech': captive['Caltech'] + flexible[:flex_cal],
            'JPL': captive['JPL'] + flexible[flex_cal:],
        }

        for site in SITES:
            result[site].sort(key=lambda s: s['arrival_minute'])

        return result
