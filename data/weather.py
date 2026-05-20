"""
Fetch hourly weather and sunshine data via Open-Meteo archive API.

Variables fetched
-----------------
  temp_c              : air temperature at 2 m  [°C]
  radiation_wm2       : global horizontal irradiance (GHI)  [W/m²]
  humidity_pct        : relative humidity at 2 m  [%]
  sunshine_duration_s : seconds of direct sunshine per hour (max 3600)  [s]

Usage (CLI)
-----------
  python weather.py --start 2026-05-11 --end 2026-05-19
  python weather.py --start 2025-01-01 --end 2025-12-31 --lat 52.52 --lon 13.41
"""

import argparse
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY_VARS = [
    "temperature_2m",  # [°C]
    "shortwave_radiation",  # GHI [W/m²]
    "relative_humidity_2m",  # [%]
    "sunshine_duration",  # direct sunshine per hour [s], max 3600
]


def fetch_weather(
    lat: float = 52.52,
    lon: float = 13.41,
    start_date: str = "2026-01-01",
    end_date: str = "2026-12-31",
) -> pd.DataFrame:
    """
    Fetch hourly weather + sunshine data from Open-Meteo archive.

    Parameters
    ----------
    lat, lon    : WGS-84 coordinates of the site
    start_date  : first day to fetch  (YYYY-MM-DD)
    end_date    : last  day to fetch  (YYYY-MM-DD)

    Returns
    -------
    DataFrame with UTC timestamps and columns:
      date, temp_c, radiation_wm2, humidity_pct, sunshine_duration_s
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    client = openmeteo_requests.Client(
        session=retry(cache_session, retries=5, backoff_factor=0.2)
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _HOURLY_VARS,
        "timezone": "UTC",
    }

    print(f"Fetching weather: {start_date} → {end_date}  (lat={lat}, lon={lon})")
    response = client.weather_api(_ARCHIVE_URL, params=params)[0]
    hourly = response.Hourly()

    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "temp_c": hourly.Variables(0).ValuesAsNumpy(),
            "radiation_wm2": hourly.Variables(1).ValuesAsNumpy(),
            "humidity_pct": hourly.Variables(2).ValuesAsNumpy(),
            "sunshine_duration_s": hourly.Variables(3).ValuesAsNumpy(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch weather + sunshine from Open-Meteo"
    )
    parser.add_argument(
        "--lat", type=float, default=52.52, help="Latitude  (default: Berlin)"
    )
    parser.add_argument(
        "--lon", type=float, default=13.41, help="Longitude (default: Berlin)"
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-12-31", help="End date   YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output CSV path (auto if omitted)")
    args = parser.parse_args()

    df = fetch_weather(args.lat, args.lon, args.start, args.end)

    year = args.start[:4]
    out_path = args.out or str(OUTPUT_DIR / f"weather_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(df.head())
