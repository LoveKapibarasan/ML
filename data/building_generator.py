import numpy as np
import pandas as pd


def generate_building_data(weather_csv_path, panel_area_m2=50, efficiency=0.2):
    """
    Generate building load and PV data based on weather radiation.
    """
    # Load weather data to sync timeline
    df_w = pd.read_csv(weather_csv_path)
    df_w['date'] = pd.to_datetime(df_w['date'])
    
    # 1. Generate Base Load (kW)
    # Simple model: Higher during day (8-20), lower at night
    def get_base_load(hour):
        if 8 <= hour <= 20:
            return np.random.uniform(15, 25) # Daytime peak
        else:
            return np.random.uniform(5, 10)   # Nighttime base
            
    df_w['hour'] = df_w['date'].dt.hour
    df_w['load'] = df_w['hour'].apply(get_base_load)
    
    # 2. Generate PV Generation (kW)
    # W/m2 * area * efficiency / 1000 = kW
    df_w['pv'] = (df_w['radiation_wm2'] * panel_area_m2 * efficiency) / 1000
    
    # 3. Final Structure
    building_df = df_w[['date', 'load', 'pv']].copy()
    
    return building_df

if __name__ == "__main__":
    # Ensure weather_2025.csv exists from previous step
    try:
        df_building = generate_building_data("../inputs/weather_2025.csv")
        df_building.to_csv("../inputs/load_lmd.csv", index=False)
        print("--- Building Data (Load & PV) Created ---")
        print(df_building.head(15))
    except FileNotFoundError:
        print("Error: weather_2025.csv not found. Run weather script first.")