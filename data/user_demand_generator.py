import re
from pathlib import Path

import numpy as np
import pandas as pd


def generate_fleet_demands(input_dir="../inputs", home_coords=(52.52, 13.41), work_coords=(52.55, 13.45)):
    """
    Generate demands for all EV IDs found in the inputs directory.
    Demands are triggered when the car arrives at Home or Work.
    """
    input_path = Path(input_dir)
    # Find all {ID}_lmd.csv files
    ev_files = [f for f in input_path.glob("*_lmd.csv") if re.match(r'^\d+_lmd\.csv', f.name)]
    
    for file in ev_files:
        ev_id = file.name.split('_')[0]
        df_ev = pd.read_csv(file)
        df_ev['date'] = pd.to_datetime(df_ev['date'])
        
        demands = []
        is_at_base_before = False
        
        for i, row in df_ev.iterrows():
            # Check if current location is a charging base (Home or Work)
            is_home = np.isclose(row['lat'], home_coords[0], atol=1e-4) and \
                      np.isclose(row['lon'], home_coords[1], atol=1e-4)
            is_work = np.isclose(row['lat'], work_coords[0], atol=1e-4) and \
                      np.isclose(row['lon'], work_coords[1], atol=1e-4)
            
            is_at_base_now = is_home or is_work
            
            # Demand Trigger: Just arrived at a base
            if is_at_base_now and not is_at_base_before:
                # Find the next time the car leaves this base (lat/lon changes)
                future_data = df_ev.iloc[i:]
                departure_rows = future_data[
                    (future_data['lat'] != row['lat']) | (future_data['lon'] != row['lon'])
                ]
                
                if not departure_rows.empty:
                    departure_time = departure_rows.iloc[0]['date']
                    
                    demands.append({
                        'ID': ev_id,
                        'arrival_time': row['date'],
                        'departure_time': departure_time,
                        'location_type': 'home' if is_home else 'work',
                        'target_soc': 0.8 if is_home else 0.5, 
                        'initial_soc': np.random.uniform(0.2, 0.4)
                    })
            
            is_at_base_before = is_at_base_now

        # Save individual demand file for each EV
        df_demand = pd.DataFrame(demands)
        output_file = input_path / f"{ev_id}_demand.csv"
        df_demand.to_csv(output_file, index=False)
        print(f"Generated {len(demands)} demands for EV {ev_id} -> {output_file}")

if __name__ == "__main__":
    generate_fleet_demands()