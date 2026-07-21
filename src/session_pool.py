import json
import os
from collections import defaultdict

import pandas as pd


def _load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    records = data['_items'] if '_items' in data else data
    df = pd.DataFrame(records)
    df['connectionTime'] = pd.to_datetime(df['connectionTime'])
    df['disconnectTime'] = pd.to_datetime(df['disconnectTime'])
    return df


class SessionPool:
    def __init__(self, data_dir='datasets'):
        caltech_df = _load_json(os.path.join(data_dir, 'caltech.json'))
        jpl_df = _load_json(os.path.join(data_dir, 'jpl.json'))

        caltech_min = caltech_df['connectionTime'].min()
        jpl_min = jpl_df['connectionTime'].min()
        self.common_start = max(caltech_min, jpl_min)
        self.sim_start = self.common_start.replace(minute=0, second=0, microsecond=0)

        self._hour_groups = defaultdict(list)
        for df, site in [(caltech_df, 'Caltech'), (jpl_df, 'JPL')]:
            df = df[df['connectionTime'] >= self.common_start].copy()
            for _, row in df.iterrows():
                conn_time = row['connectionTime']
                hour_start = conn_time.replace(minute=0, second=0, microsecond=0)
                hour_idx = int((hour_start - self.sim_start).total_seconds() / 3600)
                arrival_minute = (conn_time - hour_start).total_seconds() / 60.0
                service_minutes = (row['disconnectTime'] - row['connectionTime']).total_seconds() / 60.0
                self._hour_groups[hour_idx].append({
                    'arrival_minute': arrival_minute,
                    'service_time_minutes': service_minutes,
                    'natural_site': site,
                })

        for hour_idx, sessions in self._hour_groups.items():
            sessions.sort(key=lambda s: s['arrival_minute'])

    def get_hour_sessions(self, hour_idx):
        return list(self._hour_groups.get(hour_idx, []))

    @property
    def num_hours(self):
        return max(self._hour_groups.keys()) + 1 if self._hour_groups else 0
