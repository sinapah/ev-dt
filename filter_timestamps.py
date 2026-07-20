import pandas as pd
import os

def filter_overlapping_timestamps(input_csv='hourly_charging_telemetry.csv', output_csv='hourly_telemetry_aligned.csv'):
    if not os.path.exists(input_csv):
        print(f"Error: Base telemetry file '{input_csv}' not found. Please run your simulation script first.")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter the input dataframe to only include Caltech and JPL rows
    target_sites = ['Caltech', 'JPL']
    df = df[df['site_id'].isin(target_sites)].copy()
    
    unique_sites = df['site_id'].nunique()
    print(f"Targeting {unique_sites} sites for alignment: {', '.join(target_sites)}")
    
    print("Filtering timestamps to find common operating periods...")
    
    # Group by timestamp and filter out any hours where both Caltech and JPL are not present
    aligned_df = df.groupby('timestamp_hour').filter(lambda x: x['site_id'].nunique() == len(target_sites))
    
    # Sort chronologically and by site for clean presentation
    aligned_df = aligned_df.sort_values(by=['timestamp_hour', 'site_id']).reset_index(drop=True)
    
    # Calculate drops for reporting
    original_hours = df['timestamp_hour'].nunique()
    aligned_hours = aligned_df['timestamp_hour'].nunique()
    
    print(f"\n--- Alignment Summary (Caltech & JPL Only) ---")
    print(f"Original unique hours with either site: {original_hours}")
    print(f"Aligned unique hours (both sites present): {aligned_hours}")
    print(f"Removed {original_hours - aligned_hours} hours where one of the two sites was missing.")
    print(f"Total rows in final aligned dataset: {len(aligned_df)}")
    
    # Save the synchronized dataset
    aligned_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Aligned federated dataset exported to '{output_csv}'.")

if __name__ == '__main__':
    filter_overlapping_timestamps()