import json
import heapq
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

def load_and_prepare_data(filepath):
    """Loads ACN JSON data and converts timestamps to datetime objects."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Handle case where JSON might have a root key like '_items' or is a direct list
    records = data['_items'] if '_items' in data else data
    
    df = pd.DataFrame(records)
    # Convert ISO 8601 strings to datetime
    df['connectionTime'] = pd.to_datetime(df['connectionTime'])
    df['disconnectTime'] = pd.to_datetime(df['disconnectTime'])
    
    # Calculate service time
    df['service_time_hours'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600.0
    
    # Sort chronologically by arrival time
    df = df.sort_values('connectionTime').reset_index(drop=True)
    return df

def simulate_site(df, num_chargers, site_id):
    """
    Runs a Discrete-Event Simulation for a single charging site and 
    tracks time-weighted metrics for exact hourly aggregation.
    """
    # Event loop structures
    # Priority Queue elements: (timestamp, event_type, data)
    # event_type: 0 for DEPARTURE (processed first if timestamps match), 1 for ARRIVAL
    events = []
    
    # Populate all arrival events
    for idx, row in df.iterrows():
        heapq.heappush(events, (row['connectionTime'], 1, row))
        
    fifo_queue = []
    busy_chargers = 0
    
    # Telemetry tracking structures
    hourly_stats = defaultdict(lambda: {
        'arrivals': 0,
        'completed': 0,
        'service_times': [],
        'waiting_times': [],
        'max_queue': 0,
        'queue_time_weighted': 0.0,
        'busy_charger_time_weighted': 0.0,
        'total_monitored_seconds': 0.0
    })
    
    # Track system state changes chronologically
    first_event_time = df['connectionTime'].min()
    last_state_time = first_event_time
    
    while events:
        current_time, event_type, data = heapq.heappop(events)
        
        # --- Time-Weighted Metric Accumulation ---
        # Distribute the duration since the last event into hourly buckets
        if current_time > last_state_time:
            time_delta = (current_time - last_state_time).total_seconds()
            temp_time = last_state_time
            
            while time_delta > 0:
                # Calculate time left in current hour
                next_hour = temp_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                seconds_to_next_hour = (next_hour - temp_time).total_seconds()
                chunk = min(time_delta, seconds_to_next_hour)
                
                hour_key = temp_time.replace(minute=0, second=0, microsecond=0)
                stats = hourly_stats[hour_key]
                
                stats['queue_time_weighted'] += len(fifo_queue) * chunk
                stats['busy_charger_time_weighted'] += busy_chargers * chunk
                stats['total_monitored_seconds'] += chunk
                stats['max_queue'] = max(stats['max_queue'], len(fifo_queue))
                
                time_delta -= chunk
                temp_time += timedelta(seconds=chunk)
                
            last_state_time = current_time

        # --- Event Processing & Queue Management ---
        if event_type == 1:  # ARRIVAL
            arrival_hour = current_time.replace(minute=0, second=0, microsecond=0)
            hourly_stats[arrival_hour]['arrivals'] += 1
            hourly_stats[arrival_hour]['service_times'].append((data['disconnectTime'] - data['connectionTime']).total_seconds() / 60.0)
            
            # Context state at arrival
            queue_len_at_arrival = len(fifo_queue)
            
            if busy_chargers < num_chargers:
                # Charger is free, start charging immediately
                busy_chargers += 1
                waiting_time_min = 0.0
                hourly_stats[arrival_hour]['waiting_times'].append(waiting_time_min)
                
                charging_end_time = current_time + (data['disconnectTime'] - data['connectionTime'])
                heapq.heappush(events, (charging_end_time, 0, None))
            else:
                # All chargers busy, enter FIFO queue
                fifo_queue.append((current_time, data['disconnectTime'] - data['connectionTime']))
                
        elif event_type == 0:  # DEPARTURE
            departure_hour = current_time.replace(minute=0, second=0, microsecond=0)
            hourly_stats[departure_hour]['completed'] += 1
            
            if fifo_queue:
                # Pull next EV from FIFO queue
                arr_time, service_duration = fifo_queue.pop(0)
                waiting_time_min = (current_time - arr_time).total_seconds() / 60.0
                
                # Log waiting time against its original arrival hour bucket
                hourly_stats[arr_time.replace(minute=0, second=0, microsecond=0)]['waiting_times'].append(waiting_time_min)
                
                charging_end_time = current_time + service_duration
                heapq.heappush(events, (charging_end_time, 0, None))
            else:
                busy_chargers -= 1

    # --- Post-Simulation Hourly Aggregation ---
    hourly_records = []
    for hour_stamp, stats in sorted(hourly_stats.items()):
        total_secs = stats['total_monitored_seconds'] if stats['total_monitored_seconds'] > 0 else 3600.0
        
        # Calculate Averages safely
        avg_service = sum(stats['service_times']) / len(stats['service_times']) if stats['service_times'] else 0.0
        avg_wait = sum(stats['waiting_times']) / len(stats['waiting_times']) if stats['waiting_times'] else 0.0
        
        # Time-weighted metrics
        avg_queue_len = stats['queue_time_weighted'] / total_secs
        active_sessions = stats['busy_charger_time_weighted'] / total_secs
        
        # Utilization = busy_charger_time / (total_chargers * monitoring_window)
        utilization = stats['busy_charger_time_weighted'] / (num_chargers * total_secs)
        
        hourly_records.append({
            'timestamp_hour': hour_stamp.strftime('%Y-%m-%d %H:%M:%S'),
            'site_id': site_id,
            'arrivals_per_hour': stats['arrivals'],
            'average_service_time_minutes': round(avg_service, 2),
            'average_waiting_time_minutes': round(avg_wait, 2),
            'average_queue_length': round(avg_queue_len, 2),
            'maximum_queue_length': stats['max_queue'],
            'charger_utilization': round(utilization, 4),
            'active_charging_sessions': round(active_sessions, 2),
            'completed_sessions': stats['completed']
        })
        
    return hourly_records

def main():
    # Configure parameters for each site
    site_configs = [
        {'filename': '../acndata_sessions_caltech.json', 'chargers': 10, 'id': 'Site_A'},
        {'filename': '../acndata_sessions_jpl.json', 'chargers': 15, 'id': 'Site_B'},
        {'filename': '../acndata_sessions_office1.json', 'chargers': 8, 'id': 'Site_C'}
    ]
    
    all_site_data = []
    
    for config in site_configs:
        print(f"Processing simulation for {config['id']}...")
        try:
            df = load_and_prepare_data(config['filename'])
            site_hourly_records = simulate_site(df, config['chargers'], config['id'])
            all_site_data.extend(site_hourly_records)
        except FileNotFoundError:
            print(f"Warning: File {config['filename']} not found. Skipping.")
            
    # Compile into a single DataFrame and export to CSV
    if all_site_data:
        final_df = pd.DataFrame(all_site_data)
        output_filename = 'hourly_charging_telemetry.csv'
        final_df.to_csv(output_filename, index=False)
        print(f"Success! Combined hourly dataset exported to '{output_filename}'.")
    else:
        print("No simulation data collected.")

if __name__ == '__main__':
    main()