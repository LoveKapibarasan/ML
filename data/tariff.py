import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"


def fetch_market_prices(
    country_code="DE_LU", start_date="20250101", end_date="20251231"
):
    """
    Fetch Day-Ahead electricity prices from ENTSO-E API.
    """
    # Get API key from environment variable
    api_key = os.getenv("ENTSOE_TOKEN")
    if not api_key:
        raise ValueError("ENTSOE_TOKEN not found in environment variables.")

    client = EntsoePandasClient(api_key=api_key)

    # Define time range
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    try:
        # Query prices
        ts = client.query_day_ahead_prices(country_code, start=start, end=end)

        # Structure the DataFrame
        df_price = ts.to_frame(name="Deutschland/Luxemburg [€/MWh]")
        df_price.reset_index(inplace=True)
        df_price.rename(columns={"index": "date"}, inplace=True)

        return df_price

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def save_to_csv(df, filename: str | None = None):
    if df is None:
        return
    if filename is None:
        filename = str(OUTPUT_DIR / "spot_2025_new.csv")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", decimal=",", index=False)
    print(f"Data saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch day-ahead prices from ENTSO-E")
    parser.add_argument("--start", default="20250101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--country", default="DE_LU", help="ENTSO-E bidding zone")
    args = parser.parse_args()

    price_df = fetch_market_prices(args.country, args.start, args.end)
    year = args.start[:4]
    save_to_csv(price_df, filename=str(OUTPUT_DIR / f"spot_{year}_new.csv"))
