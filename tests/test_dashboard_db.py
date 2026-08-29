"""Tests for the citrine DB layer of :mod:`dashboard`.

Kept apart from ``test_dashboard.py`` so the shared runtime fixture,
which stubs these very functions out, is not in play here.
"""

from datetime import datetime, timedelta, timezone

import pytest

import dashboard

UNREACHABLE = dict(
    host="127.0.0.1", port=1, dbname="nope", user="nope", password="nope"
)


def test_fetch_actuals_raises_when_the_database_is_unreachable():
    now = datetime.now(timezone.utc)

    with pytest.raises(dashboard.DashboardDataError):
        dashboard.fetch_actuals(UNREACHABLE, "cp001", now - timedelta(hours=1), now)


def test_fetch_recent_sessions_raises_when_the_database_is_unreachable():
    with pytest.raises(dashboard.DashboardDataError):
        dashboard.fetch_recent_sessions(UNREACHABLE, "cp001")


def test_cached_actuals_propagate_the_error_rather_than_masking_it():
    now = datetime.now(timezone.utc)

    with pytest.raises(dashboard.DashboardDataError):
        dashboard.fetch_actuals_cached(
            UNREACHABLE, "cp001", now - timedelta(hours=1), now
        )


def test_build_actual_series_fills_gaps_and_keeps_unknown_prices_unknown():
    since = datetime(2026, 8, 28, 0, 0)
    rows = [{"hour": since + timedelta(hours=1), "power_kw": 6.0, "energy_kwh": 6.0}]

    series = dashboard.build_actual_series(
        rows, since, 3, lambda t: 0.25 if t.hour == 1 else None
    )

    assert [row["power_kw"] for row in series] == [0.0, 6.0, 0.0]
    assert [row["plugged_in"] for row in series] == [False, True, False]
    assert series[1]["cost_eur"] == pytest.approx(1.5)
    assert series[0]["price_eur_kwh"] is None and series[0]["cost_eur"] is None


def test_baseline_cost_charges_at_full_power_until_the_target_is_met():
    hours = [{"price_eur_kwh": 0.20} for _ in range(24)]

    # 0.2 -> 0.8 of a 50 kWh battery is 30 kWh; at 11 kW that is 3 full hours
    # of charging (11 + 11 + 8), so 30 kWh x EUR 0.20.
    cost = dashboard._baseline_cost_eur(hours, 0.2, 0.8, 24, 11.0, 50.0)

    assert cost == pytest.approx(30.0 * 0.20, abs=1e-6)


def test_baseline_cost_stops_at_departure():
    hours = [{"price_eur_kwh": 0.20} for _ in range(24)]

    cost = dashboard._baseline_cost_eur(hours, 0.2, 0.8, 2, 11.0, 50.0)

    assert cost == pytest.approx(22.0 * 0.20, abs=1e-6)
