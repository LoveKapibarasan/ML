"""
Fetch real EV charging data from citrine (OCPP) database.

Extracted data:
  - meter_values.csv  : hourly energy consumption per connector (kWh)
  - transactions.csv  : EV charging sessions (start/end, total kWh, state)
"""

"""
Fetch real EV charging data from citrine (OCPP) database.

Extracted data:
  - meter_values.csv  : hourly energy consumption per connector (kWh)
  - transactions.csv  : EV charging sessions (start/end, total kWh, state)
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.25.30.7"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": "citrine",
    "user": "citrine",
    "password": os.getenv("DB_PASSWORD", "048910ee1e61799b85241a6e70c3f0d57c91302f7221481290d3b023a222e743"),
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"


def _query(conn, sql) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall())


def fetch_meter_values(conn) -> pd.DataFrame:
    """
    Extract hourly energy consumption (kWh) from MeterValues.
    Uses the total-phase Energy.Active.Import.Register reading.
    """
    df = _query(conn, """
        SELECT timestamp, "connectorId", "sampledValue"
        FROM "MeterValues"
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp ASC
    """)
    if df.empty:
        return df

    def extract_energy_wh(sv):
        try:
            items = sv if isinstance(sv, list) else []
            for item in items:
                if (
                    item.get("measurand") == "Energy.Active.Import.Register"
                    and item.get("phase") is None
                ):
                    return float(item["value"])
        except Exception:
            pass
        return None

    df["energy_wh"] = df["sampledValue"].apply(extract_energy_wh)
    df = df.dropna(subset=["energy_wh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["connectorId"] = df["connectorId"].fillna(0).astype(int)

    # Delta kWh between consecutive readings per connector
    df = df.sort_values(["connectorId", "timestamp"])
    df["delta_kwh"] = df.groupby("connectorId")["energy_wh"].diff() / 1000.0
    df["delta_kwh"] = df["delta_kwh"].clip(lower=0)

    # Resample to hourly per connector
    df = df.set_index("timestamp")
    hourly = (
        df.groupby("connectorId")["delta_kwh"]
        .resample("1h")
        .sum()
        .reset_index()
        .rename(columns={"timestamp": "date", "delta_kwh": "ev_energy_kwh"})
    )
    return hourly


def fetch_transactions(conn) -> pd.DataFrame:
    """Extract EV charging sessions from Transactions."""
    df = _query(conn, """
        SELECT
            id, "stationId", "transactionId", "chargingState",
            "totalKwh", "startTime", "endTime", "meterStart"
        FROM "Transactions"
        WHERE "startTime" IS NOT NULL OR "endTime" IS NOT NULL
        ORDER BY "endTime" DESC
    """)
    if df.empty:
        return df

    for col in ["startTime", "endTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def main():
    print(f"Connecting to {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['dbname']}...")
    with psycopg2.connect(**DB_CONFIG) as conn:
        print("Fetching MeterValues...")
        df_mv = fetch_meter_values(conn)
        if not df_mv.empty:
            out = OUTPUT_DIR / "meter_values.csv"
            df_mv.to_csv(out, index=False)
            print(f"  Saved {len(df_mv)} rows → {out}")
        else:
            print("  No MeterValues energy data found.")

        print("Fetching Transactions...")
        df_tx = fetch_transactions(conn)
        if not df_tx.empty:
            out = OUTPUT_DIR / "transactions.csv"
            df_tx.to_csv(out, index=False)
            print(f"  Saved {len(df_tx)} rows → {out}")
        else:
            print("  No Transaction data found.")


if __name__ == "__main__":
    main()
