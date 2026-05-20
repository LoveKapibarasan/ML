"""
Build hourly EV state and demand schedule from citrine (OCPP) database.

Outputs
-------
  inputs/0_lmd.csv     : hourly EV state  (date, PluggedIn, PowerRating, is_holiday)
  inputs/0_demand.csv  : session schedule (arrival_time, departure_time, target_soc, initial_soc)

The date range is driven entirely by what is in the DB.
Run `python weather.py` and `python holiday.py` afterwards with the
printed date range so that all data covers the same period.
"""

import os
from pathlib import Path

import holidays as hol_lib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.25.30.7"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "citrine"),
    "user": os.getenv("DB_USER", "citrine"),
    "password": os.getenv("DB_PASSWORD"),
}

BATTERY_CAPACITY_KWH = 50.0
DEFAULT_POWER_RATING_KW = 11.0
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"


def _get_conn():
    return psycopg2.connect(**DB_CONFIG)


def _query(conn, sql) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall())


def build_ev_state(conn) -> pd.DataFrame:
    """
    Hourly EV state derived from MeterValues.

    - pluggedin  : True during hours that had an active session
    - powerrating: measured kW if available, else DEFAULT_POWER_RATING_KW when plugged in
    - is_holiday : German (Berlin) public holiday flag
    """
    df = _query(
        conn,
        """
        SELECT timestamp, "transactionId", "sampledValue"
        FROM "MeterValues"
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp ASC
    """,
    )
    if df.empty:
        raise RuntimeError("No MeterValues data in DB.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["pluggedin"] = df["transactionId"].notna()

    def extract_power_w(sv):
        if not isinstance(sv, list):
            return None
        for item in sv:
            if (
                item.get("measurand") == "Power.Active.Import"
                and item.get("phase") is None
            ):
                return float(item["value"])
        return None

    df["power_w"] = df["sampledValue"].apply(extract_power_w)
    df = df.set_index("timestamp")

    hourly_plugin = df["pluggedin"].resample("1h").max().rename("pluggedin")
    hourly_power = df["power_w"].resample("1h").mean()

    hourly = pd.concat([hourly_plugin, hourly_power], axis=1)
    hourly["PluggedIn"] = hourly["pluggedin"].fillna(False)
    hourly["PowerRating"] = hourly.apply(
        lambda r: (
            r["power_w"] / 1000.0
            if pd.notna(r["power_w"]) and r["power_w"] > 0
            else (DEFAULT_POWER_RATING_KW if r["PluggedIn"] else 0.0)
        ),
        axis=1,
    )
    hourly = (
        hourly[["PluggedIn", "PowerRating"]]
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    # Holiday flag (Berlin, Germany)
    years = hourly["date"].dt.year.unique().tolist()
    de_holidays = hol_lib.country_holidays("DE", subdiv="BE", years=years)
    hourly["is_holiday"] = hourly["date"].dt.date.apply(
        lambda d: 1 if d in de_holidays else 0
    )

    return hourly[["date", "PluggedIn", "PowerRating", "is_holiday"]]


def build_demand(conn) -> pd.DataFrame:
    """
    Session-based demand schedule from Transactions + MeterValues.

    - arrival_time  : first MeterValue timestamp of the session
    - departure_time: Transaction.endTime
    - target_soc    : capped ratio of totalKwh to battery capacity (min 0.5)
    - initial_soc   : random 0.1–0.3 (SoC at arrival not reported by OCPP station)
    """
    df = _query(
        conn,
        """
        SELECT
            t.id,
            t."totalKwh",
            t."endTime",
            MIN(mv.timestamp) AS arrival_time
        FROM "Transactions" t
        LEFT JOIN "MeterValues" mv ON mv."transactionDatabaseId" = t.id
        WHERE t."endTime" IS NOT NULL
          AND t."totalKwh" IS NOT NULL
          AND t."totalKwh" > 0
        GROUP BY t.id
        ORDER BY t."endTime" ASC
    """,
    )
    if df.empty:
        raise RuntimeError("No usable Transactions in DB.")

    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    df["departure_time"] = pd.to_datetime(df["endTime"], utc=True)

    # target_soc: at least 0.5 even for short test sessions
    df["target_soc"] = (df["totalKwh"].astype(float) / BATTERY_CAPACITY_KWH).clip(
        0.5, 1.0
    )
    rng = np.random.default_rng(42)
    df["initial_soc"] = rng.uniform(0.1, 0.3, size=len(df))

    return df[["arrival_time", "departure_time", "target_soc", "initial_soc"]]


def main():
    print(
        f"Connecting to {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['dbname']} ..."
    )
    conn = _get_conn()

    print("Building EV state time-series ...")
    df_lmd = build_ev_state(conn)
    lmd_path = OUTPUT_DIR / "0_lmd.csv"
    df_lmd.to_csv(lmd_path, index=False)
    print(f"  {len(df_lmd)} hourly rows → {lmd_path}")

    print("Building demand schedule ...")
    df_demand = build_demand(conn)
    demand_path = OUTPUT_DIR / "0_demand.csv"
    df_demand.to_csv(demand_path, index=False)
    print(f"  {len(df_demand)} sessions → {demand_path}")

    conn.close()

    start = df_lmd["date"].min().strftime("%Y-%m-%d")
    end = df_lmd["date"].max().strftime("%Y-%m-%d")
    year = df_lmd["date"].dt.year.iloc[0]
    print(f"\nDB date range: {start} → {end}  (year={year})")
    print("Next steps:")
    print(f"  python weather.py --start {start} --end {end}")
    print(f"  python holiday.py --year {year}")
    print(f"  python tariff.py  --start {start[:4]}0101 --end {start[:4]}1231")


if __name__ == "__main__":
    main()
