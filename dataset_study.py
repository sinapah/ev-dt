import json
import os

def count_unique_stations(file_configs):
    all_stations = set()
    site_counts = {}

    for config in file_configs:
        filename = config['filename']
        site_id = config['id']
        
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
            
        with open(filename, 'r') as f:
            print(f"Attempting to read: {filename}...")
            data = json.load(f)
            
        # Extract records safely depending on format
        records = data['_items'] if '_items' in data else data
        
        # Gather all station IDs for this specific site
        site_stations = set(
            record['stationID'] for record in records 
            if 'stationID' in record and record['stationID'] is not None
        )
        
        site_counts[site_id] = len(site_stations)
        
        # Merge into the master set
        all_stations.update(site_stations)
        
    print("--- Station Counts per Site ---")
    for site, count in site_counts.items():
        print(f"{site}: {count} unique physical stations")
        
    print("\n--- Total Unique Stations Across All Datasets ---")
    print(f"Total: {len(all_stations)} unique station IDs")

if __name__ == '__main__':
    # Configuration matching your files
    files = [
        {'filename': './datasets/acndata_sessions_caltech.json', 'id': 'Site_A (Caltech)'},
        {'filename': './datasets/acndata_sessions_jpl.json', 'id': 'Site_B (JPL)'},
        {'filename': './datasets/acndata_sessions_office1.json', 'id': 'Site_C (Office 1)'}
    ]
    
    count_unique_stations(files)