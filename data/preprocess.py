import pandas as pd
from pathlib import Path

def preprocess_all_data(ev_id):
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "inputs"
    
    print(f"--- Preprocessing EV {ev_id} ---")

    df_price = pd.read_csv(path / "spot_2025_new.csv", sep=';', decimal=',')
    df_price.columns = ['date', 'price']
    df_price['price'] = df_price['price'] / 1000.0 # €/MWh -> €/kWh
    df_price['price_3h_future'] = df_price['price'].shift(-3)
    df_price['price_3h_future'] = df_price['price_3h_future'].ffill()
    df_ev = pd.read_csv(path / f"{ev_id}_lmd.csv")
    df_weather = pd.read_csv(path / "weather_2025.csv")
    df_holiday = pd.read_csv(path / "holidays_2025.csv")
    df_load = pd.read_csv(path / "load_lmd.csv")

    def clean_date(df):
        df.columns = [c.lower().strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        return df

    df_ev = clean_date(df_ev)
    df_price = clean_date(df_price)
    df_weather = clean_date(df_weather)
    df_holiday = clean_date(df_holiday)
    df_load = clean_date(df_load)

    df_final = df_ev.merge(df_price, on='date')\
                    .merge(df_weather, on='date')\
                    .merge(df_holiday, on='date')\
                    .merge(df_load, on='date')

    output_file = path / f"processed_data_{ev_id}.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Success: {output_file} created.")

    df_dem = pd.read_csv(path / f"{ev_id}_demand.csv")
    df_dem.columns = [c.lower().strip() for c in df_dem.columns]
    df_dem['arrival_time'] = pd.to_datetime(df_dem['arrival_time'], utc=True).dt.tz_localize(None)
    df_dem['departure_time'] = pd.to_datetime(df_dem['departure_time'], utc=True).dt.tz_localize(None)
    df_dem.to_csv(path / f"processed_demand_{ev_id}.csv", index=False)

if __name__ == "__main__":
    for i in range(3):
        preprocess_all_data(i)