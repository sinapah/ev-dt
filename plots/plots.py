import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_telemetry(input_csv='hourly_telemetry_aligned.csv', start_date=None, end_date=None):
    if not os.path.exists(input_csv):
        print(f"Error: Aligned file '{input_csv}' not found. Run your filtering script first.")
        return

    # Load data and ensure proper datetime parsing
    df = pd.read_csv(input_csv)
    df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
    
    # Optional: Filter for a specific time window to keep the plot readable
    if start_date and end_date:
        df = df[(df['timestamp_hour'] >= start_date) & (df['timestamp_hour'] <= end_date)]
        print(f"Filtered data to window: {start_date} to {end_date}")
    
    # Pivot datasets so each site has dedicated columns for easy plotting
    caltech_df = df[df['site_id'] == 'Caltech'].sort_values('timestamp_hour')
    jpl_df = df[df['site_id'] == 'JPL'].sort_values('timestamp_hour')
    
    if caltech_df.empty or jpl_df.empty:
        print("Error: Could not extract separate rows for Caltech and JPL. Check site_id naming.")
        return

    # Create the figure and the primary y-axis (Arrivals)
    fig, ax1 = plt.subplots(figsize=(14, 7), sharex=True)
    
    # Create a secondary y-axis sharing the same x-axis (Queue Length)
    ax2 = ax1.twinx()
    
    # --- Plot Metrics ---
    # Primary axis: Arrivals per hour (Solid lines)
    line1 = ax1.plot(caltech_df['timestamp_hour'], caltech_df['arrivals_per_hour'], 
                     color='darkorange', alpha=0.7, label='Caltech - Arrivals')
    line2 = ax1.plot(jpl_df['timestamp_hour'], jpl_df['arrivals_per_hour'], 
                     color='royalblue', alpha=0.7, label='JPL - Arrivals')
    
    # Secondary axis: Average Queue Length (Dashed lines to distinguish)
    line3 = ax2.plot(caltech_df['timestamp_hour'], caltech_df['average_queue_length'], 
                     color='red', linestyle='--', alpha=0.8, label='Caltech - Avg Queue')
    line4 = ax2.plot(jpl_df['timestamp_hour'], jpl_df['average_queue_length'], 
                     color='navy', linestyle='--', alpha=0.8, label='JPL - Avg Queue')
    
    # --- Axis Labeling & Formatting ---
    ax1.set_xlabel('Time (Hourly Intervals)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Arrivals per Hour', color='darkblue', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Queue Length (EVs)', color='darkred', fontsize=12, fontweight='bold')
    
    # Title adjust based on windowing
    window_str = f" ({start_date} to {end_date})" if start_date else " (Full Dataset Overview)"
    plt.title(f"Charging Park Dynamics: Caltech vs JPL{window_str}", fontsize=14, fontweight='bold', pad=15)
    
    # Combine legends from both axes seamlessly
    all_lines = line1 + line2 + line3 + line4
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    # Save chart image and display
    output_img = 'charging_dynamics_plot.png'
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully as '{output_img}'")
    plt.show()

if __name__ == '__main__':
    # Change these dates to view different periods!
    # A 2-week slice lets you clearly see daily variations vs queue growth
    sample_start = "2019-10-01 00:00:00"
    sample_end   = "2019-10-14 23:00:00"
    
    plot_telemetry(start_date=sample_start, end_date=sample_end)