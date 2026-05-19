"""
Merge all data sources into a single processed CSV per EV / station.

Input files (all in inputs/)
------------------------------
  {ev_id}_lmd.csv        : hourly EV state from DB (ev_from_db.py)
  {ev_id}_demand.csv     : session demand from DB  (ev_from_db.py)
  spot_{year}_new.csv    : day-ahead electricity prices (tariff.py / ENTSO-E)
  weather_{year}.csv     : weather + sunshine (weather.py / Open-Meteo)
  holidays_{year}.csv    : holiday flags (holiday.py)

Output files (all in inputs/)
------------------------------
  processed_data_{ev_id}.csv   : merged hourly features
  processed_demand_{ev_id}.csv : demand schedule with parsed timestamps
"""

from pathlib import Path

import pandas as pd


def _clean_date(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df


def _detect_year(df_ev: pd.DataFrame) -> int:
    """Derive the data year from the EV state timestamps."""
    return pd.to_datetime(df_ev["date"]).dt.year.mode().iloc[0]


def preprocess(ev_id: int = 0) -> None:
    base = Path(__file__).resolve().parent.parent
    path = base / "inputs"

    print(f"--- Preprocessing EV/station {ev_id} ---")

    # ── EV state (from citrine DB via ev_from_db.py) ─────────────────────────
    df_ev = pd.read_csv(path / f"{ev_id}_lmd.csv")
    year  = _detect_year(df_ev)

    # ── Other data sources (API-fetched) ──────────────────────────────────────
    # Electricity spot prices — semicolon CSV, €/MWh → €/kWh
    price_file = path / f"spot_{year}_new.csv"
    if not price_file.exists():
        # Fallback: try the most recent available price file
        candidates = sorted(path.glob("spot_*_new.csv"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No spot price file found in {path}")
        price_file = candidates[0]
        print(f"  Warning: using fallback price file {price_file.name}")

    df_price = pd.read_csv(price_file, sep=";", decimal=",")
    df_price.columns = ["date", "price"]
    df_price["price"] /= 1000.0
    df_price["price_3h_future"] = df_price["price"].shift(-3).ffill()

    # Weather + sunshine
    weather_file = path / f"weather_{year}.csv"
    if not weather_file.exists():
        candidates = sorted(path.glob("weather_*.csv"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No weather file found in {path}")
        weather_file = candidates[0]
        print(f"  Warning: using fallback weather file {weather_file.name}")
    df_weather = pd.read_csv(weather_file)

    # Holidays
    holiday_file = path / f"holidays_{year}.csv"
    if not holiday_file.exists():
        candidates = sorted(path.glob("holidays_*.csv"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No holiday file found in {path}")
        holiday_file = candidates[0]
        print(f"  Warning: using fallback holiday file {holiday_file.name}")
    df_holiday = pd.read_csv(holiday_file)

    # ── Normalise timestamps ──────────────────────────────────────────────────
    df_ev      = _clean_date(df_ev)
    df_price   = _clean_date(df_price)
    df_weather = _clean_date(df_weather)
    df_holiday = _clean_date(df_holiday)

    # If price year differs from EV year, shift price dates to match.
    # (Hourly patterns are representative even across years.)
    price_year = pd.to_datetime(df_price["date"]).dt.year.mode().iloc[0]
    if price_year != year:
        offset = pd.DateOffset(years=(year - price_year))
        df_price["date"] = pd.to_datetime(df_price["date"]) + offset
        print(f"  Info: price dates shifted from {price_year} → {year}")

    # ── Merge on hourly date ──────────────────────────────────────────────────
    df_final = (
        df_ev
        .merge(df_price,   on="date")
        .merge(df_weather, on="date")
        .merge(df_holiday, on="date")
    )

    if df_final.empty:
        print("  Warning: merged DataFrame is empty — date ranges may not overlap.")
        print(f"    EV range   : {df_ev['date'].min()} → {df_ev['date'].max()}")
        print(f"    Price range: {df_price['date'].min()} → {df_price['date'].max()}")
        print(f"    Weather rng: {df_weather['date'].min()} → {df_weather['date'].max()}")

    out_data = path / f"processed_data_{ev_id}.csv"
    df_final.to_csv(out_data, index=False)
    print(f"  Merged {len(df_final)} rows → {out_data}")

    # ── Demand schedule ───────────────────────────────────────────────────────
    df_dem = pd.read_csv(path / f"{ev_id}_demand.csv")
    df_dem.columns = [c.lower().strip() for c in df_dem.columns]
    for col in ["arrival_time", "departure_time"]:
        df_dem[col] = pd.to_datetime(df_dem[col], utc=True).dt.tz_localize(None)
    out_dem = path / f"processed_demand_{ev_id}.csv"
    df_dem.to_csv(out_dem, index=False)
    print(f"  Demand {len(df_dem)} sessions → {out_dem}")


if __name__ == "__main__":
    preprocess(ev_id=0)
