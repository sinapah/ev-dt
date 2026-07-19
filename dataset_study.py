import json
import os

def count_unique_stations(file_configs, output_txt_path='station_counts_summary.txt'):
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
            
        records = data['_items'] if '_items' in data else data
        
        site_stations = set(
            record['stationID'] for record in records 
            if 'stationID' in record and record['stationID'] is not None
        )
        
        site_counts[site_id] = len(site_stations)
        
        all_stations.update(site_stations)
        
    output_lines = []
    output_lines.append("--- Station Counts per Site ---")
    for site, count in site_counts.items():
        output_lines.append(f"{site}: {count} unique physical stations")
        
    output_lines.append("\n--- Total Unique Stations Across All Datasets ---")
    output_lines.append(f"Total: {len(all_stations)} unique station IDs")
    
    final_output = "\n".join(output_lines)
    
    print("\n" + final_output)
    
    with open(output_txt_path, 'w') as out_file:
        out_file.write(final_output)
    print(f"\nSaved results successfully to '{output_txt_path}'")

if __name__ == '__main__':
    files = [
        {'filename': './datasets/caltech.json', 'id': 'Site_A (Caltech)'},
        {'filename': './datasets/jpl.json', 'id': 'Site_B (JPL)'},
        {'filename': './datasets/office001.json', 'id': 'Site_C (Office 1)'}
    ]
    
    count_unique_stations(files)