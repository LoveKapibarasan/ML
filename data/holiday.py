"""
Generate hourly holiday flags from the official German public holiday calendar.

Usage (CLI)
-----------
  python holiday.py --year 2026
  python holiday.py --year 2025 --country DE --subdiv BE
"""

import argparse
from pathlib import Path

import holidays as hol_lib
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"


def fetch_holidays(
    year: int = 2026,
    country: str = "DE",
    subdiv: str = "BE",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Build an hourly DataFrame of holiday flags.

    Parameters
    ----------
    year     : calendar year
    country  : ISO 3166-1 alpha-2 country code
    subdiv   : subdivision / state code (e.g. "BE" for Berlin)
    start_date, end_date : optional override for the date range (YYYY-MM-DD)

    Returns
    -------
    DataFrame with columns: date (UTC), is_holiday
    """
    start = pd.Timestamp(start_date or f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(end_date or f"{year}-12-31 23:00", tz="UTC")
    dates = pd.date_range(start=start, end=end, freq="h")

    cal = hol_lib.country_holidays(country, subdiv=subdiv, years=year)
    flags = [1 if d.date() in cal else 0 for d in dates]

    return pd.DataFrame({"date": dates, "is_holiday": flags})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hourly holiday flags")
    parser.add_argument("--year", type=int, default=2026, help="Calendar year")
    parser.add_argument("--country", default="DE", help="Country code (default: DE)")
    parser.add_argument(
        "--subdiv", default="BE", help="State/subdivision (default: BE = Berlin)"
    )
    parser.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Override end date   YYYY-MM-DD")
    args = parser.parse_args()

    df = fetch_holidays(args.year, args.country, args.subdiv, args.start, args.end)

    out_path = OUTPUT_DIR / f"holidays_{args.year}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(df[df["is_holiday"] == 1].head())
