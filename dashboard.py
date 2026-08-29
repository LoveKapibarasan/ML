"""Data layer for the operations dashboard served by :mod:`serve`.

This module holds everything the dashboard needs that is *not* the SAC
schedule itself: the historical "what actually happened" series read
from the citrine (OCPP 2.0.1) database, and the assembly of the JSON
payload returned by ``GET /api/dashboard``.

It deliberately does **not** import :mod:`serve`. The dependency runs one
way only (``serve`` → ``dashboard``), so there is no import cycle and the
functions here stay unit-testable without loading the SAC model. Anything
the payload builder needs from the serving side — the price lookup, the
computed schedule — is passed in as an argument.

Database access is allowed to fail: the fetch functions raise
:class:`DashboardDataError` when citrine is unreachable, which lets the
caller tell "the DB is down" apart from "this station has no data yet"
and degrade to a forecast-only page with a warning banner.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd
import psycopg2
import psycopg2.extras

# ── Config ─────────────────────────────────────────────────────────────────────
HISTORY_HOURS = 24
SESSION_LIMIT = 10
_ACTUALS_TTL_S = 300  # 5 min — MeterValues arrive at most once a minute

_actuals_cache: dict[str, tuple[list[dict], datetime]] = {}
_actuals_lock = threading.Lock()


class DashboardDataError(RuntimeError):
    """Raised when the citrine database cannot be queried."""


# ── citrine DB ─────────────────────────────────────────────────────────────────


def _as_utc(dt: datetime) -> datetime:
    """Returns ``dt`` as a timezone-aware UTC datetime.

    The serving side works in naive UTC; ``MeterValues.timestamp`` is
    ``timestamptz``. Query bounds are therefore made explicit rather than
    left to the server's session timezone.

    Args:
        dt: Naive (assumed UTC) or timezone-aware datetime.

    Returns:
        datetime.datetime: The same instant, tagged as UTC.
    """
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def fetch_actuals(
    db_config: dict,
    station_id: str,
    since: datetime,
    until: datetime,
) -> list[dict]:
    """Reads the measured hourly charging power for one station.

    Averages the ``Power.Active.Import`` measurand (total phase) per hour
    over the requested window, the same reading
    :func:`serve._get_evse_max_power_kw` and ``data/ev_from_db.py`` use.

    ``sampledValue`` is declared ``json`` (not ``jsonb``) in citrine, so
    the cast is required: ``jsonb_array_elements`` has no ``json``
    overload and the query errors without it.
    Energy is approximated as ``power_kw x 1 h``, which matches the
    hourly resolution the model and the spot price both work at.

    Args:
        db_config: ``psycopg2.connect`` keyword arguments.
        station_id: OCPP station identifier.
        since: Start of the window (inclusive), naive UTC or aware.
        until: End of the window (exclusive), naive UTC or aware.

    Returns:
        list[dict]: One entry per hour that had readings, each with
        ``hour`` (naive UTC :class:`~datetime.datetime`), ``power_kw``
        and ``energy_kwh``. Empty if the station has no data in the
        window.

    Raises:
        DashboardDataError: If the database cannot be reached or queried.
    """
    try:
        conn = psycopg2.connect(**db_config)
        with conn, conn.cursor() as cur:
            # Power.Active.Import is reported in W; convert to kW.
            # date_trunc on the UTC-shifted timestamp keeps the bucket
            # labels in the naive-UTC convention used by serve.py.
            cur.execute(
                """
                SELECT date_trunc('hour', mv.timestamp AT TIME ZONE 'UTC') AS hour,
                       AVG((sv->>'value')::float) / 1000.0                 AS power_kw
                FROM   "MeterValues"  mv
                JOIN   "Transactions" t  ON t.id = mv."transactionDatabaseId"
                CROSS  JOIN LATERAL jsonb_array_elements(mv."sampledValue"::jsonb) sv
                WHERE  t."stationId"    = %s
                  AND  sv->>'measurand' = 'Power.Active.Import'
                  AND  sv->>'phase'     IS NULL
                  AND  mv.timestamp >= %s
                  AND  mv.timestamp <  %s
                GROUP  BY 1
                ORDER  BY 1
                """,
                (station_id, _as_utc(since), _as_utc(until)),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        raise DashboardDataError(str(exc)) from exc

    out = []
    for hour, power_kw in rows:
        kw = float(power_kw or 0.0)
        out.append({"hour": hour, "power_kw": kw, "energy_kwh": kw})
    return out


def fetch_actuals_cached(
    db_config: dict,
    station_id: str,
    since: datetime,
    until: datetime,
) -> list[dict]:
    """Caches :func:`fetch_actuals` per station for 5 minutes.

    The dashboard polls once a minute; MeterValues arrive far more
    slowly than that, so repeated queries would only add DB load. Uses
    the same in-process dict + lock pattern as ``serve._power_cache``.

    Args:
        db_config: ``psycopg2.connect`` keyword arguments.
        station_id: OCPP station identifier, also the cache key.
        since: Start of the window (inclusive).
        until: End of the window (exclusive).

    Returns:
        list[dict]: As :func:`fetch_actuals`.

    Raises:
        DashboardDataError: If the database cannot be reached or queried.
    """
    key = f"{station_id}:{since.isoformat()}:{until.isoformat()}"
    now = datetime.now()
    with _actuals_lock:
        hit = _actuals_cache.get(key)
        if hit and (now - hit[1]).total_seconds() < _ACTUALS_TTL_S:
            return hit[0]

    rows = fetch_actuals(db_config, station_id, since, until)
    if rows:
        with _actuals_lock:
            _actuals_cache[key] = (rows, now)
    return rows


def fetch_recent_sessions(
    db_config: dict,
    station_id: str,
    limit: int = SESSION_LIMIT,
) -> list[dict]:
    """Reads the most recent charging sessions for one station.

    Same columns as ``data/db_fetcher.fetch_transactions``, narrowed to a
    single station and to the newest ``limit`` rows.

    Args:
        db_config: ``psycopg2.connect`` keyword arguments.
        station_id: OCPP station identifier.
        limit: Maximum number of sessions to return.

    Returns:
        list[dict]: Newest first, each with ``id``, ``start``, ``end``,
        ``total_kwh`` and ``charging_state``. Empty if the station has
        no sessions.

    Raises:
        DashboardDataError: If the database cannot be reached or queried.
    """
    try:
        conn = psycopg2.connect(**db_config)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, "startTime", "endTime", "totalKwh", "chargingState"
                FROM   "Transactions"
                WHERE  "stationId" = %s
                ORDER  BY COALESCE("endTime", "startTime") DESC NULLS LAST
                LIMIT  %s
                """,
                (station_id, limit),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        raise DashboardDataError(str(exc)) from exc

    return [
        {
            "id": row[0],
            "start": _iso(row[1]),
            "end": _iso(row[2]),
            "total_kwh": float(row[3]) if row[3] is not None else None,
            "charging_state": row[4],
        }
        for row in rows
    ]


# ── Payload assembly ───────────────────────────────────────────────────────────


def _iso(dt) -> str | None:
    """Formats a datetime as an ISO 8601 string, or ``None``.

    Args:
        dt: Datetime, or anything falsy.

    Returns:
        str | None: ISO 8601 representation, or ``None`` if ``dt`` is
        not a datetime.
    """
    return dt.isoformat() if isinstance(dt, datetime) else None


def build_actual_series(
    actuals: list[dict],
    since: datetime,
    hours: int,
    price_at: Callable[[datetime], float],
) -> list[dict]:
    """Expands measured hours into a gap-free hourly series.

    Hours with no meter reading are emitted with zero power and
    ``plugged_in`` false, so the chart has one point per hour and the
    gaps read as "not charging" rather than as missing data.

    Args:
        actuals: Rows from :func:`fetch_actuals`.
        since: First hour of the window (naive UTC).
        hours: Number of hourly buckets to emit.
        price_at: Callable returning the spot price (EUR/kWh) at a
            given hour, or ``None`` where no price is known.

    Returns:
        list[dict]: ``hours`` entries with ``t``, ``power_kw``,
        ``energy_kwh``, ``plugged_in``, ``price_eur_kwh`` and
        ``cost_eur``. The last two are ``None`` for hours the price
        table does not reach — an unknown cost is reported as unknown
        rather than as zero.
    """
    by_hour = {row["hour"]: row for row in actuals}
    series = []
    for h in range(hours):
        t = since + timedelta(hours=h)
        row = by_hour.get(t)
        power_kw = round(row["power_kw"], 3) if row else 0.0
        energy_kwh = round(row["energy_kwh"], 3) if row else 0.0
        price = price_at(t)
        series.append(
            {
                "t": t.isoformat(),
                "power_kw": power_kw,
                "energy_kwh": energy_kwh,
                "plugged_in": row is not None,
                "price_eur_kwh": None if price is None else round(price, 6),
                "cost_eur": None if price is None else round(energy_kwh * price, 4),
            }
        )
    return series


def _baseline_cost_eur(
    hours: list[dict],
    initial_soc: float,
    target_soc: float,
    dep_hour: int,
    max_power_kw: float,
    battery_capacity_kwh: float,
) -> float:
    """Costs the always-on strategy over the same horizon.

    Charges at full power from hour 0 until the target SoC is reached or
    the vehicle departs — the ``always_on`` baseline of ``benchmark.py``,
    evaluated on this request's actual prices so the saving shown on the
    dashboard is comparable to the schedule beside it.

    Args:
        hours: Per-hour rows from ``serve.compute_schedule``.
        initial_soc: State of charge at hour 0, ``[0, 1]``.
        target_soc: Desired state of charge at departure, ``[0, 1]``.
        dep_hour: Hour index at which the vehicle departs.
        max_power_kw: Station's maximum power rating in kW.
        battery_capacity_kwh: Battery capacity in kWh.

    Returns:
        float: Cost in EUR of the always-on schedule.
    """
    soc = initial_soc
    cost = 0.0
    for h in range(min(dep_hour, len(hours))):
        if soc >= target_soc:
            break
        step_soc = min(target_soc, soc + max_power_kw / battery_capacity_kwh)
        energy_kwh = (step_soc - soc) * battery_capacity_kwh
        cost += energy_kwh * hours[h]["price_eur_kwh"]
        soc = step_soc
    return cost


def build_payload(
    *,
    station_id: str,
    evse_id: int,
    schedule: dict,
    actual_series: list[dict],
    sessions: list[dict],
    model_path: str,
    now: datetime,
    departure: datetime,
    current_soc: float,
    target_soc: float,
    battery_capacity_kwh: float,
    data_sources: dict,
    warnings: list[str],
) -> dict:
    """Assembles the ``GET /api/dashboard`` response.

    Args:
        station_id: OCPP station identifier.
        evse_id: OCPP EVSE identifier.
        schedule: Result of ``serve.compute_schedule``.
        actual_series: Result of :func:`build_actual_series`.
        sessions: Result of :func:`fetch_recent_sessions`.
        model_path: Path of the served SAC model, for display.
        now: Timestamp of hour 0 of the forecast (naive UTC).
        departure: Departure time (naive UTC).
        current_soc: State of charge at hour 0, ``[0, 1]``.
        target_soc: Desired state of charge at departure, ``[0, 1]``.
        battery_capacity_kwh: Battery capacity in kWh.
        data_sources: Provenance flags shown in the footer, e.g.
            ``{"prices": "live", "weather": "ok", "db": "unavailable"}``.
        warnings: Human-readable degradation notices for the UI banner.

    Returns:
        dict: The full dashboard payload (see the module docstring of
        :mod:`serve` for the shape).
    """
    hours = schedule["hours"]
    max_power_kw = schedule["max_power_kw"]
    dep_hour = schedule["dep_hour"]

    planned_energy = sum(h["energy_kwh"] for h in hours)
    planned_cost = sum(h["cost_eur"] for h in hours)
    avg_price = (planned_cost / planned_energy) if planned_energy > 0 else 0.0

    baseline_cost = _baseline_cost_eur(
        hours,
        current_soc,
        target_soc,
        dep_hour,
        max_power_kw,
        battery_capacity_kwh,
    )
    savings = baseline_cost - planned_cost

    actual_energy = sum(row["energy_kwh"] for row in actual_series)
    # Hours with no known price contribute energy but not cost, so the
    # figure never claims a cost it cannot substantiate.
    priced = [row for row in actual_series if row["cost_eur"] is not None]
    actual_cost = sum(row["cost_eur"] for row in priced)
    unpriced_kwh = sum(
        row["energy_kwh"] for row in actual_series if row["cost_eur"] is None
    )

    return {
        "station_id": station_id,
        "evse_id": evse_id,
        "generated_at": now.isoformat(),
        "model": model_path,
        "max_power_kw": round(max_power_kw, 3),
        "battery_capacity_kwh": battery_capacity_kwh,
        "current_soc": round(current_soc, 4),
        "target_soc": round(target_soc, 4),
        "departure_time": departure.isoformat(),
        "departure_hour_index": dep_hour,
        "kpi": {
            "projected_soc_at_departure": round(schedule["projected_soc"], 4),
            "soc_target_met": schedule["soc_target_met"],
            "planned_energy_kwh": round(planned_energy, 3),
            "planned_cost_eur": round(planned_cost, 4),
            "avg_price_eur_kwh": round(avg_price, 6),
            "forced_fill_hours": sum(1 for h in hours if h["forced"]),
            "baseline_cost_eur": round(baseline_cost, 4),
            "savings_eur": round(savings, 4),
            "savings_pct": (
                round(100.0 * savings / baseline_cost, 1) if baseline_cost > 0 else 0.0
            ),
            "actual_energy_kwh": round(actual_energy, 3),
            "actual_cost_eur": round(actual_cost, 4),
            "actual_unpriced_kwh": round(unpriced_kwh, 3),
        },
        "forecast": hours,
        "actuals": actual_series,
        "sessions": sessions,
        "data_sources": data_sources,
        "warnings": warnings,
    }


def merge_timeline(payload: dict) -> pd.DataFrame:
    """Flattens a dashboard payload into one hourly frame for plotting.

    Concatenates the measured history and the forecast onto a single
    time axis so every chart in the UI shares one x scale. The
    ``series`` column is what the power chart colours by.

    Args:
        payload: A response from ``GET /api/dashboard``.

    Returns:
        pandas.DataFrame: One row per hour, ordered oldest first, with
        columns ``t``, ``t_end``, ``phase``, ``series``,
        ``price_eur_kwh``, ``power_kw``, ``soc``, ``cost_eur``,
        ``forced``, ``temp_c`` and ``radiation_wm2``. Unknown values are
        ``NaN`` rather than zero.
    """
    rows = []
    for a in payload["actuals"]:
        rows.append(
            {
                "t": a["t"],
                "phase": "past",
                "series": "measured",
                "price_eur_kwh": a["price_eur_kwh"],
                "power_kw": a["power_kw"],
                "soc": None,
                "cost_eur": a["cost_eur"],
                "forced": False,
                "temp_c": None,
                "radiation_wm2": None,
            }
        )
    for f in payload["forecast"]:
        rows.append(
            {
                "t": f["t"],
                "phase": "future",
                "series": "forced" if f["forced"] else "planned",
                "price_eur_kwh": f["price_eur_kwh"],
                "power_kw": f["limit_w"] / 1000.0,
                "soc": f["soc_end"],
                "cost_eur": f["cost_eur"],
                "forced": f["forced"],
                "temp_c": f["temp_c"],
                "radiation_wm2": f["radiation_wm2"],
            }
        )

    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"])
    # Columns that mix None with floats land as object dtype, which both
    # Altair and st.dataframe render as the literal "None". Coerce so a
    # missing value is NaN and shows as a blank.
    for column in (
        "price_eur_kwh",
        "power_kw",
        "soc",
        "cost_eur",
        "temp_c",
        "radiation_wm2",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # A 5-minute shy end gives adjacent bars a visible gap without a border.
    df["t_end"] = df["t"] + pd.Timedelta(minutes=55)
    return df.sort_values("t").reset_index(drop=True)
