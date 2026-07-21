SITES = ['Caltech', 'JPL']
NUM_CHARGERS = {'Caltech': 15, 'JPL': 15}
FL_AGGREGATION_INTERVAL = 24
CAPTIVE_FRACTION = 0.6

SERVICE_TIME_COL = 'average_service_time_minutes'
ARRIVALS_COL = 'arrivals_per_hour'
QUEUE_COL = 'average_queue_length'
UTIL_COL = 'charger_utilization'
WAIT_COL = 'average_waiting_time_minutes'

STATE_FIELDS = ['arrivals', 'queue_length', 'waiting_time', 'utilization']

FEATURE_LAGS = {
    'arrivals': [1, 2, 3, 24, 168],
    'queue_length': [1, 24],
    'utilization': [1, 24],
}

ROLLING_WINDOWS = [6, 24]

FEATURE_NAMES = (
    [f'arrivals_lag_{l}' for l in FEATURE_LAGS['arrivals']]
    + [f'queue_lag_{l}' for l in FEATURE_LAGS['queue_length']]
    + [f'util_lag_{l}' for l in FEATURE_LAGS['utilization']]
    + [f'arrivals_rolling_mean_{w}' for w in ROLLING_WINDOWS]
    + ['arrivals_daily_total']
    + ['hour', 'day_of_week', 'is_weekend']
)

COLUMN_MAP = {
    'arrivals_per_hour': 'arrivals',
    'average_queue_length': 'queue_length',
    'average_waiting_time_minutes': 'waiting_time',
    'charger_utilization': 'utilization',
    'average_service_time_minutes': 'service_time',
    'active_charging_sessions': 'active_sessions',
    'completed_sessions': 'completed_sessions',
    'maximum_queue_length': 'max_queue_length',
}
