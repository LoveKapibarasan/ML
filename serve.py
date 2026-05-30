"""
SAC Smart Charging Inference Server

Implements the SMARTCHARGING_ENDPOINT contract expected by citrineos-payment.
citrineos-payment polls this server every SMARTCHARGING_POLL_INTERVAL seconds
and forwards the returned ChargingProfile to CitrineOS via SetChargingProfile
(OCPP 2.0.1).

SoC guarantee
-------------
After the SAC model generates the 24-hour schedule, the server simulates the
resulting SoC trajectory.  If the target SoC will not be reached by departure,
it force-fills the cheapest remaining hours (cheapest first, negative prices
filled first) until the shortfall is covered.  The EVSE's actual max power is
read from the citrine DB (Power.Active.Import measurand, cached 24 h); the
MAX_POWER_KW env var is used as fallback.

API contract
------------
GET /schedule
  Query params:
    station_id    str   — OCPP station identifier
    evse_id       int   — OCPP EVSE ID (integer)
    desired_soc   float — target state-of-charge [0,1]   (default 0.8)
    departure_time str  — ISO 8601 datetime               (default now+24 h)
    current_soc   float — current battery SoC [0,1]       (default 0.2)
    user_id       str   — Keycloak user ID                (optional)

  Response:
    {
      "chargingProfile": {
        "id": <int>,
        "stackLevel": 0,
        "chargingProfilePurpose": "TxDefaultProfile",
        "chargingProfileKind": "Absolute",
        "chargingSchedule": [{
          "id": <int>,
          "chargingRateUnit": "W",
          "chargingSchedulePeriod": [
            {"startPeriod": 0,    "limit": <watts>},
            {"startPeriod": 3600, "limit": <watts>},
            ...  (SCHEDULE_HOURS periods, one per hour)
          ]
        }]
      }
    }

GET /health
  Response: {"status": "healthy", "model": "<path>", "max_power_kw": <float>}
"""

import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import holidays as hol_lib
import numpy as np
import openmeteo_requests
import pandas as pd
import psycopg2
import requests_cache
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from fastapi import FastAPI, HTTPException, Query
from retry_requests import retry
from stable_baselines3 import SAC

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
BATTERY_CAPACITY_KWH = 50.0
MAX_POWER_KW = float(os.getenv("MAX_POWER_KW", "11.0"))
WEATHER_LAT = float(os.getenv("WEATHER_LAT", "52.52"))
WEATHER_LON = float(os.getenv("WEATHER_LON", "13.41"))
INPUTS_DIR = Path(os.getenv("INPUTS_DIR", "inputs"))
SCHEDULE_HOURS = 24
_CHARGE_THRESHOLD = 0.05
_SOC_TOLERANCE = 0.02  # SoC within 2 % of target = "met"

_DB = dict(
    host=os.getenv("DB_HOST", "172.25.30.7"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "citrine"),
    user=os.getenv("DB_USER", "citrine"),
    password=os.getenv("DB_PASSWORD"),
)


# ── Model ──────────────────────────────────────────────────────────────────────
def _find_model() -> Path:
    for p in [
        Path(os.getenv("MODEL_PATH", "models/best_model")),
        Path("models/best_model"),
        Path("models/sac_smart_charger_final"),
    ]:
        if p.with_suffix(".zip").exists():
            return p
    raise FileNotFoundError("No trained model found — run train.py first.")


_model_path = _find_model()
model = SAC.load(str(_model_path))

# ── Weather (HTTP cache 1 h) ───────────────────────────────────────────────────
_wx_client = openmeteo_requests.Client(
    session=retry(
        requests_cache.CachedSession(".cache", expire_after=3600),
        retries=3,
        backoff_factor=0.2,
    )
)

app = FastAPI(title="SAC Smart Charging Server", version="1.0.0")


# ── EVSE max-power from DB (cache 24 h) ────────────────────────────────────────

_power_cache: dict[str, tuple[float, datetime]] = {}
_power_lock = threading.Lock()


def _get_evse_max_power_kw(station_id: str, ocpp_evse_id: int) -> float:
    """
    Return the maximum charging power (kW) for this EVSE.

    Queries citrine DB for the highest Power.Active.Import value observed on
    any transaction at this station.  Result is cached for 24 hours.
    Falls back to the MAX_POWER_KW environment variable on any error.
    """
    key = f"{station_id}:{ocpp_evse_id}"
    with _power_lock:
        hit = _power_cache.get(key)
        if hit and (datetime.now() - hit[1]).total_seconds() < 86_400:
            return hit[0]

    try:
        conn = psycopg2.connect(**_DB)
        with conn, conn.cursor() as cur:
            # Power.Active.Import is reported in W; convert to kW.
            # Filter by station so multi-station setups stay isolated.
            cur.execute(
                """
                SELECT MAX((sv->>'value')::float) / 1000.0
                FROM   "MeterValues"  mv
                JOIN   "Transactions" t  ON t.id = mv."transactionDatabaseId"
                CROSS  JOIN LATERAL jsonb_array_elements(mv."sampledValue") sv
                WHERE  t."stationId"    = %s
                  AND  sv->>'measurand' = 'Power.Active.Import'
                  AND  sv->>'phase'     IS NULL
                """,
                (station_id,),
            )
            row = cur.fetchone()
        conn.close()
        if row and row[0] and float(row[0]) > 0:
            kw = float(row[0])
            with _power_lock:
                _power_cache[key] = (kw, datetime.now())
            return kw
    except Exception:
        pass

    return MAX_POWER_KW


# ── Price data (ENTSO-E live + CSV fallback, cache 1 h) ───────────────────────

_prices_cache: dict = {"df": None, "fetched_at": None}
_prices_lock = threading.Lock()


def _fetch_prices_live() -> pd.DataFrame:
    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise ValueError("ENTSOE_TOKEN not set")
    client = EntsoePandasClient(api_key=token)
    start = pd.Timestamp.now(tz="UTC").normalize()
    end = start + pd.Timedelta(days=2)
    ts = client.query_day_ahead_prices("DE_LU", start=start, end=end)
    df = ts.to_frame(name="price")
    df.index = df.index.tz_convert(None)
    df["price"] /= 1000.0
    df["price_3h_future"] = df["price"].shift(-3).ffill()
    return df


def _load_prices_csv() -> pd.DataFrame:
    files = sorted(INPUTS_DIR.glob("spot_*_new.csv"), reverse=True)
    if not files:
        raise FileNotFoundError("No spot price file — run data/tariff.py")
    df = pd.read_csv(files[0], sep=";", decimal=",")
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["price"] /= 1000.0
    df["price_3h_future"] = df["price"].shift(-3).ffill()
    price_year = int(df["date"].dt.year.mode().iloc[0])
    cur_year = datetime.now().year
    if price_year != cur_year:
        df["date"] += pd.DateOffset(years=(cur_year - price_year))
    return df.set_index("date")


def _load_prices() -> pd.DataFrame:
    with _prices_lock:
        now = datetime.now()
        age = (
            (now - _prices_cache["fetched_at"]).total_seconds()
            if _prices_cache["fetched_at"]
            else 9999
        )
        if _prices_cache["df"] is not None and age < 3600:
            return _prices_cache["df"]
        try:
            df = _fetch_prices_live()
            _prices_cache.update({"df": df, "fetched_at": now})
            return df
        except Exception:
            pass
        if _prices_cache["df"] is not None:
            return _prices_cache["df"]
        df = _load_prices_csv()
        _prices_cache.update({"df": df, "fetched_at": now})
        return df


# ── Weather forecast (Open-Meteo, cache 1 h) ───────────────────────────────────


def _fetch_weather() -> pd.DataFrame:
    resp = _wx_client.weather_api(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,
            "hourly": [
                "temperature_2m",
                "shortwave_radiation",
                "relative_humidity_2m",
                "sunshine_duration",
            ],
            "timezone": "UTC",
            "forecast_days": 2,
        },
    )[0]
    h = resp.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    ).tz_localize(None)
    return pd.DataFrame(
        {
            "temp_c": h.Variables(0).ValuesAsNumpy(),
            "radiation_wm2": h.Variables(1).ValuesAsNumpy(),
            "sunshine_duration_s": h.Variables(3).ValuesAsNumpy(),
        },
        index=dates,
    )


# ── Observation / misc helpers ─────────────────────────────────────────────────


def _lookup(df: pd.DataFrame, t: datetime, col: str, default: float) -> float:
    t0 = t.replace(minute=0, second=0, microsecond=0)
    if t0 in df.index:
        return float(df.loc[t0, col])
    earlier = df[df.index <= t0]
    return float(earlier[col].iloc[-1]) if not earlier.empty else default


def _is_holiday(dt: datetime) -> float:
    cal = hol_lib.country_holidays("DE", subdiv="BE", years=dt.year)
    return 1.0 if dt.date() in cal else 0.0


def _build_obs(
    price: float,
    price_3h: float,
    temp_c: float,
    radiation: float,
    sunshine: float,
    holiday: float,
    soc: float,
    time_left_h: float,
    target_soc: float,
    power_kw: float,
) -> np.ndarray:
    """11-feature vector — must match EVChargingEnv observation space."""
    return np.array(
        [
            price,
            (temp_c + 20.0) / 60.0,
            min(radiation / 1000.0, 1.0),
            sunshine / 3600.0,
            holiday,
            1.0,  # plugged-in
            soc,
            float(np.clip(time_left_h / 24.0, 0.0, 1.0)),
            target_soc,
            min(power_kw / 22.0, 1.0),
            price_3h,
        ],
        dtype=np.float32,
    )


# ── SoC guarantee ──────────────────────────────────────────────────────────────


def _guarantee_soc(
    periods: list[dict],
    initial_soc: float,
    target_soc: float,
    dep_hour: int,
    prices: pd.DataFrame,
    now: datetime,
    max_power_kw: float,
    battery_capacity_kwh: float = BATTERY_CAPACITY_KWH,
) -> list[dict]:
    """
    Guarantee the target SoC is reached by the departure hour.

    1. Simulate the SoC trajectory using the periods generated by the SAC model.
    2. If the projected SoC at departure falls short of target − SOC_TOLERANCE,
       compute the remaining kWh needed.
    3. Fill the cheapest available hours (those with spare capacity) from cheapest
       to most expensive until the shortfall is covered.

    Negative spot prices are treated as the cheapest possible and are filled first
    (the station earns money while charging).
    """
    if dep_hour <= 0:
        return periods

    max_w = int(round(max_power_kw * 1000))
    n = min(dep_hour, len(periods))

    # ── 1. Simulate projected SoC ─────────────────────────────────────────────
    soc = initial_soc
    for h in range(n):
        rate = (periods[h]["limit"] / max_w) if max_w > 0 else 0.0
        soc = min(1.0, soc + rate * max_power_kw / battery_capacity_kwh)

    shortfall_kwh = max(0.0, (target_soc - soc) * battery_capacity_kwh)
    if shortfall_kwh < 0.05:  # within 50 Wh → already satisfied
        return periods

    # ── 2. Collect hours with spare charging capacity ─────────────────────────
    slots: list[tuple[float, int, int]] = []  # (price, hour_idx, cur_limit_w)
    for h in range(n):
        if periods[h]["limit"] < max_w:
            t = now + timedelta(hours=h)
            price = _lookup(prices, t, "price", default=0.0)
            slots.append((price, h, periods[h]["limit"]))

    slots.sort(key=lambda s: s[0])  # cheapest / most-negative first

    # ── 3. Fill cheapest slots until shortfall is covered ─────────────────────
    periods = [p.copy() for p in periods]
    for _price, h, cur_w in slots:
        if shortfall_kwh < 0.05:
            break
        headroom_kwh = (max_w - cur_w) / 1000.0
        fill_kwh = min(headroom_kwh, shortfall_kwh)
        periods[h]["limit"] = min(cur_w + int(round(fill_kwh * 1000)), max_w)
        shortfall_kwh -= fill_kwh

    return periods


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/schedule")
def get_schedule(
    station_id: str = Query(..., description="OCPP station ID"),
    evse_id: int = Query(..., description="OCPP EVSE ID (integer)"),
    desired_soc: float = Query(default=0.8, ge=0.0, le=1.0),
    departure_time: str | None = Query(default=None),
    current_soc: float = Query(default=0.2, ge=0.0, le=1.0),
    battery_capacity_kwh: float = Query(default=BATTERY_CAPACITY_KWH, gt=0.0),
    user_id: str | None = Query(default=None),
):
    now = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )

    # ── Parse departure time ──────────────────────────────────────────────────
    if departure_time:
        dep = pd.to_datetime(departure_time)
        if dep.tzinfo is not None:
            dep = dep.tz_convert(None)
        dep = dep.to_pydatetime().replace(tzinfo=None)
    else:
        dep = now + timedelta(hours=SCHEDULE_HOURS)

    dep_hour = max(0, min(int((dep - now).total_seconds() / 3600), SCHEDULE_HOURS))

    # ── Load real-time data ───────────────────────────────────────────────────
    try:
        prices = _load_prices()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        weather = _fetch_weather()
    except Exception:
        weather = pd.DataFrame()

    # ── Get EVSE max power from DB ────────────────────────────────────────────
    max_power_kw = _get_evse_max_power_kw(station_id, evse_id)

    # ── SAC schedule simulation ───────────────────────────────────────────────
    soc = float(np.clip(current_soc, 0.0, 1.0))
    target_soc = float(np.clip(desired_soc, 0.0, 1.0))
    initial_soc = soc
    periods: list[dict] = []

    for h in range(SCHEDULE_HOURS):
        t = now + timedelta(hours=h)
        time_left_h = max(0.0, (dep - t).total_seconds() / 3600.0)

        price = _lookup(prices, t, "price", default=0.0)
        price_3h = _lookup(prices, t, "price_3h_future", default=0.0)

        if not weather.empty:
            temp = _lookup(weather, t, "temp_c", default=15.0)
            rad = _lookup(weather, t, "radiation_wm2", default=0.0)
            sun = _lookup(weather, t, "sunshine_duration_s", default=0.0)
        else:
            temp, rad, sun = 15.0, 0.0, 0.0

        holiday = _is_holiday(t)

        if time_left_h <= 0 or soc >= target_soc:
            rate = 0.0
        else:
            obs = _build_obs(
                price,
                price_3h,
                temp,
                rad,
                sun,
                holiday,
                soc,
                time_left_h,
                target_soc,
                max_power_kw,
            )
            action, _ = model.predict(obs, deterministic=True)
            rate = float(np.clip(action[0], 0.0, 1.0))
            if rate < _CHARGE_THRESHOLD:
                rate = 0.0

        limit_w = int(round(rate * max_power_kw * 1000))
        periods.append({"startPeriod": h * 3600, "limit": limit_w})

        if rate > 0:
            soc = min(1.0, soc + rate * max_power_kw / battery_capacity_kwh)

    # ── SoC guarantee: fill cheapest hours if target will not be met ──────────
    periods = _guarantee_soc(
        periods,
        initial_soc,
        target_soc,
        dep_hour,
        prices,
        now,
        max_power_kw,
        battery_capacity_kwh,
    )

    profile_id = abs(hash(f"{station_id}:{evse_id}:{now.date()}")) % 2_000_000 + 1

    return {
        "chargingProfile": {
            "id": profile_id,
            "stackLevel": 0,
            "chargingProfilePurpose": "TxDefaultProfile",
            "chargingProfileKind": "Absolute",
            "chargingSchedule": [
                {
                    "id": profile_id,
                    "chargingRateUnit": "W",
                    "chargingSchedulePeriod": periods,
                }
            ],
        }
    }


@app.get("/health")
def health():
    max_kw = _power_cache.get("cached_default", (MAX_POWER_KW,))[0]
    return {"status": "healthy", "model": str(_model_path), "max_power_kw": max_kw}
