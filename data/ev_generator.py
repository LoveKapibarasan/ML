from datetime import datetime

import holidays
import numpy as np
import pandas as pd


def generate_fleet_data(ev_ids=[0, 1, 2], days=365):
    """
    Generate primary data for multiple EVs including calendar features.
    """
    start_date = datetime(2025, 1, 1)
    dates = pd.date_range(start=start_date, periods=days*24, freq='h')
    
    # German holidays (Berlin) to influence human behavior
    de_holidays = holidays.CountryHoliday('DE', prov='BE')
    
    HOME_COORDS = (52.5200, 13.4100)
    WORK_COORDS = (52.5500, 13.4500)
    
    all_ev_records = []

    for ev_id in ev_ids:
        # Add some randomness per EV (e.g., different work start times)
        start_shift = np.random.randint(-1, 2) 
        
        for date in dates:
            hour = date.hour
            weekday = date.weekday()
            is_holiday = date in de_holidays
            
            # Base state: Home
            lat, lon = HOME_COORDS
            plugged_in = True
            power_rating = 11.0
            
            # Logic: If it's a workday (not weekend and not a holiday)
            if weekday < 5 and not is_holiday:
                if (8 + start_shift) < hour < (18 + start_shift):
                    lat, lon = WORK_COORDS
                    plugged_in = True
                    power_rating = 3.6
                elif hour == (8 + start_shift) or hour == (18 + start_shift):
                    lat, lon = (52.5350, 13.4300)
                    plugged_in = False
                    power_rating = 0.0
            # Weekend or Holiday behavior
            else:
                if 11 <= hour <= 16: # Leisure outing
                    lat, lon = (52.6000, 13.5000)
                    plugged_in = False
                    power_rating = 0.0

            all_ev_records.append({
                'date': date,
                'ID': ev_id,
                'lat': lat,
                'lon': lon,
                'PluggedIn': plugged_in,
                'PowerRating': power_rating,
                'is_holiday': 1 if is_holiday else 0 # Feature for AI
            })

    df = pd.DataFrame(all_ev_records)
    return df

# Generate for 3 EVs
df_fleet = generate_fleet_data(ev_ids=[0, 1, 2])

# Save one big file or split by ID
# FleetRL often expects separate files: 0_lmd.csv, 1_lmd.csv...
for ev_id in df_fleet['ID'].unique():
    df_single = df_fleet[df_fleet['ID'] == ev_id]
    df_single.to_csv(f"../inputs/{ev_id}_lmd.csv", index=False)

print(f"Generated data for {len(df_fleet['ID'].unique())} vehicles.")