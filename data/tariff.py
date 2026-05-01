import os

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

# Load environment variables
load_dotenv()

def fetch_market_prices(country_code='DE_LU', start_date='20250101', end_date='20251231'):
    """
    Fetch Day-Ahead electricity prices from ENTSO-E API.
    """
    # Get API key from environment variable
    api_key = os.getenv('ENTSOE_TOKEN')
    if not api_key:
        raise ValueError("ENTSOE_TOKEN not found in environment variables.")

    client = EntsoePandasClient(api_key=api_key)

    # Define time range
    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')

    try:
        # Query prices
        ts = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        # Structure the DataFrame
        df_price = ts.to_frame(name='Deutschland/Luxemburg [€/MWh]')
        df_price.reset_index(inplace=True)
        df_price.rename(columns={'index': 'date'}, inplace=True)
        
        return df_price

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(df, filename='../inputs/spot_2025_new.csv'):
    """
    Save the DataFrame in a specific European CSV format (semicolon and comma decimal).
    """
    if df is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, sep=';', decimal=',', index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Example: Fetching 2025 data
    price_df = fetch_market_prices(
        country_code='DE_LU', 
        start_date='20250101', 
        end_date='20251231'
    )
    save_to_csv(price_df)