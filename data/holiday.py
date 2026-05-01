from pathlib import Path

import holidays
import pandas as pd


def generate_holiday_file(year=2025, country='DE', prov='BE'):
    """
    Generate a CSV file containing holiday flags for each date.
    """
    # Define timeline (Hourly to match other weather/market data)
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31 23:00:00"
    dates = pd.date_range(start=start_date, end=end_date, freq='h')

    # Get holiday definitions
    de_holidays = holidays.CountryHoliday(country, prov=prov)

    # Create list of flags
    # 1 if holiday, 0 if not
    holiday_flags = [1 if date in de_holidays else 0 for date in dates]

    # Build DataFrame
    df_holiday = pd.DataFrame({
        'date': dates,
        'is_holiday': holiday_flags
    })

    # File path setup (Same robust path logic)
    current_dir = Path(__file__).parent
    output_dir = current_dir.parent / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"holidays_{year}.csv"
    
    # Save
    df_holiday.to_csv(file_path, index=False)
    print(f"Holiday file created: {file_path}")
    print(df_holiday[df_holiday['is_holiday'] == 1].head()) # Check some holidays

if __name__ == "__main__":
    generate_holiday_file()