import pandas as pd
import numpy as np
from config import SITES, COLUMN_MAP


def load_ground_truth(path='ground_truth.csv'):
    raw = pd.read_csv(path, parse_dates=['timestamp_hour'])
    raw = raw.sort_values(['timestamp_hour', 'site_id']).reset_index(drop=True)
    return _pivot_and_map(raw)


def _pivot_and_map(df):
    df = df[df['site_id'].isin(SITES)].copy()
    df.rename(columns=COLUMN_MAP, inplace=True)

    site_defaults = {}
    for site in SITES:
        site_mask = df['site_id'] == site
        valid = df.loc[site_mask & df['service_time'].notna(), 'service_time']
        site_defaults[site] = valid.median() if len(valid) > 0 else 240.0

    pivot_cols = [c for c in COLUMN_MAP.values() if c in df.columns]
    wide = df.pivot_table(
        index='timestamp_hour',
        columns='site_id',
        values=pivot_cols,
        aggfunc='first'
    )
    wide.columns = [f'{site}_{col}' for col, site in wide.columns]
    wide.reset_index(inplace=True)
    wide.sort_values('timestamp_hour', inplace=True)
    wide.reset_index(drop=True, inplace=True)
    for col in wide.columns:
        if col == 'timestamp_hour':
            continue
        if wide[col].dtype in (np.float64, np.float32, 'float64', 'float32'):
            is_service = col.endswith('_service_time')
            if is_service and wide[col].isna().any():
                site = col.split('_')[0]
                wide[col] = wide[col].fillna(site_defaults.get(site, 240.0))
            else:
                wide[col] = wide[col].fillna(0.0)
    return wide


def get_site_series(wide, site, field):
    col = f'{site}_{field}'
    if col not in wide.columns:
        raise KeyError(f"Column {col} not found in pivoted data")
    return wide[col].values
