import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


def fetch_annual_weather(lat=52.52, lon=13.41, year=2025):
    """
    Fetch 1 year of hourly weather data for RL features.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Define time range for the specific year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "shortwave_radiation", "relative_humidity_2m"],
        "timezone": "UTC"
    }

    print(f"Requesting data for {year}...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()
    hourly_humidity = hourly.Variables(2).ValuesAsNumpy()

    # Create timestamp range
    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Build DataFrame
    weather_df = pd.DataFrame({
        "date": date_range,
        "temp_c": hourly_temperature_2m,
        "radiation_wm2": hourly_shortwave_radiation,
        "humidity_pct": hourly_humidity
    })

    return weather_df

if __name__ == "__main__":
    # Fetching Berlin 2025 weather
    df_weather = fetch_annual_weather(lat=52.52, lon=13.41, year=2025)
    
    if df_weather is not None:
        print(df_weather.head())
        # Save for later merging
        df_weather.to_csv("../inputs/weather_2025.csv", index=False)
        print(f"Successfully saved {len(df_weather)} rows of weather data.")