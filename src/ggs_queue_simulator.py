import numpy as np
from collections import deque
from config import SITES, NUM_CHARGERS


class GGsQueueSimulator:
    def __init__(self):
        self.num_chargers = dict(NUM_CHARGERS)
        self.chargers = {site: [None] * n for site, n in self.num_chargers.items()}
        self.queue = {site: deque() for site in SITES}
        self.service_pool = {}
        self.current_time = 0.0

    def set_service_pools(self, gt_df):
        for site in SITES:
            vals = gt_df[
                (gt_df['site_id'] == site) &
                gt_df['average_service_time_minutes'].notna()
            ]['average_service_time_minutes'].values
            self.service_pool[site] = vals if len(vals) > 0 else np.array([240.0])

    def _sample_service_time(self, site):
        pool = self.service_pool.get(site)
        if pool is not None and len(pool) > 0:
            return float(np.random.choice(pool))
        return 240.0

    def reset(self, initial_state):
        self.current_time = 0.0
        for site in SITES:
            s = initial_state[site]
            n = self.num_chargers[site]
            active = int(round(s['active_sessions']))
            q_len = int(round(s['queue_length']))
            active = min(active, n)

            self.chargers[site] = [None] * n
            for i in range(active):
                st = self._sample_service_time(site)
                self.chargers[site][i] = np.random.uniform(0, st)

            self.queue[site] = deque()
            for _ in range(q_len):
                self.queue[site].append((self._sample_service_time(site), 0.0))

    def step(self, routed_arrivals, service_time_minutes=None):
        hour_start = self.current_time
        hour_end = hour_start + 60.0
        results = {}

        for site in SITES:
            n_arrivals = int(routed_arrivals.get(site, 0))
            n = self.num_chargers[site]

            for i in range(n):
                if self.chargers[site][i] is not None and self.chargers[site][i] <= hour_start:
                    self.chargers[site][i] = None

            if n_arrivals == 0 and len(self.queue[site]) == 0 and \
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
            t = hour_start
            for _ in range(n_arrivals):
                rate = max(n_arrivals / 60.0, 1e-6)
                interarrival = np.random.exponential(1.0 / rate)
                t += interarrival
                if t > hour_end:
                    break
                arrival_events.append((t, self._sample_service_time(site)))

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

            # Carry forward: update charger end_times that extend beyond this hour
            # (no change needed — end_times are already absolute and persist)

            results[site] = {
                'queue_length': float(end_queue),
                'waiting_time': avg_wait,
                'utilization': min(1.0, avg_util),
                'active_sessions': float(end_active),
                'completed_sessions': float(completed),
            }

        self.current_time = hour_end
        return results
