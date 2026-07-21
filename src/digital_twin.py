from config import SITES


class DigitalTwin:
    def __init__(self):
        self.last_predictions = {}
        self.prediction_errors = []

    def predict(self, state, predicted_arrivals, ggs_sim, scheduler):
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
