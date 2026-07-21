from collections import deque

from config import SITES, NUM_CHARGERS


class GGsQueueSimulator:
    def __init__(self):
        self.num_chargers = dict(NUM_CHARGERS)
        self.chargers = {site: [None] * n for site, n in self.num_chargers.items()}
        self.queue = {site: deque() for site in SITES}
        self.current_time = 0.0

    def reset(self, initial_state=None):
        self.current_time = 0.0
        for site in SITES:
            n = self.num_chargers[site]
            self.chargers[site] = [None] * n
            self.queue[site] = deque()

    def get_state(self):
        return {
            'chargers': {site: list(c) for site, c in self.chargers.items()},
            'queue': {site: list(self.queue[site]) for site in SITES},
            'current_time': self.current_time,
        }

    def set_state(self, state):
        self.chargers = {site: list(state['chargers'][site]) for site in SITES}
        self.queue = {site: deque(state['queue'][site]) for site in SITES}
        self.current_time = state['current_time']

    def _drain_queue(self, site):
        n = self.num_chargers[site]
        while self.queue[site]:
            free_idx = next(
                (i for i in range(n) if self.chargers[site][i] is None),
                None
            )
            if free_idx is None:
                break
            svc_time, arr_time = self.queue[site].popleft()
            self.chargers[site][free_idx] = self.current_time + svc_time

    def step(self, per_site_sessions):
        hour_start = self.current_time
        hour_end = hour_start + 60.0
        results = {}

        for site in SITES:
            n = self.num_chargers[site]

            for i in range(n):
                if self.chargers[site][i] is not None and self.chargers[site][i] <= hour_start:
                    self.chargers[site][i] = None

            self._drain_queue(site)

            site_sessions = per_site_sessions.get(site, [])

            if len(site_sessions) == 0 and len(self.queue[site]) == 0 and \
                    all(c is None for c in self.chargers[site]):
                results[site] = {
                    'queue_length': 0.0,
                    'waiting_time': 0.0,
                    'utilization': 0.0,
                    'active_sessions': 0.0,
                    'completed_sessions': 0.0,
                }
                continue

            arrival_events = []
            for session in site_sessions:
                t = hour_start + session['arrival_minute']
                if t <= hour_end:
                    arrival_events.append((t, session['service_time_minutes']))

            events = [(t, 'arrival', st) for t, st in arrival_events]
            for i in range(n):
                et = self.chargers[site][i]
                if et is not None and hour_start < et <= hour_end:
                    events.append((et, 'departure', i))

            events.sort(key=lambda x: x[0])

            last_time = hour_start
            tw_queue = 0.0
            tw_busy = 0.0
            total_wait = 0.0
            completed = 0

            while events and events[0][0] <= hour_end:
                evt_time, evt_type, evt_data = events.pop(0)

                dt = evt_time - last_time
                tw_queue += len(self.queue[site]) * dt
                tw_busy += sum(1 for c in self.chargers[site] if c is not None) * dt
                last_time = evt_time

                if evt_type == 'arrival':
                    svc_time = evt_data
                    free_idx = next(
                        (i for i in range(n) if self.chargers[site][i] is None),
                        None
                    )
                    if free_idx is not None:
                        self.chargers[site][free_idx] = evt_time + svc_time
                        dep = evt_time + svc_time
                        if dep <= hour_end:
                            events.append((dep, 'departure', free_idx))
                            events.sort(key=lambda x: x[0])
                    else:
                        self.queue[site].append((svc_time, evt_time))

                elif evt_type == 'departure':
                    ci = evt_data
                    self.chargers[site][ci] = None
                    completed += 1

                    if self.queue[site]:
                        svc_time, arr_time = self.queue[site].popleft()
                        total_wait += evt_time - arr_time
                        self.chargers[site][ci] = evt_time + svc_time
                        dep = evt_time + svc_time
                        if dep <= hour_end:
                            events.append((dep, 'departure', ci))
                            events.sort(key=lambda x: x[0])

            rem = hour_end - last_time
            tw_queue += len(self.queue[site]) * rem
            tw_busy += sum(1 for c in self.chargers[site] if c is not None) * rem

            avg_queue = tw_queue / 60.0
            avg_util = tw_busy / (n * 60.0)
            avg_wait = total_wait / max(completed, 1)

            end_queue = len(self.queue[site])
            end_active = sum(1 for c in self.chargers[site] if c is not None)

            results[site] = {
                'queue_length': float(end_queue),
                'waiting_time': avg_wait,
                'utilization': min(1.0, avg_util),
                'active_sessions': float(end_active),
                'completed_sessions': float(completed),
            }

        self.current_time = hour_end
        return results
