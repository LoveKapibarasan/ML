from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import dashboard
import serve


@pytest.fixture(autouse=True)
def _patched(patch_runtime_dependencies):
    """Applies the shared runtime patches (see ``tests/conftest.py``)."""
    return patch_runtime_dependencies


AUTH = {"Authorization": "Bearer test-token"}


def _get(client, **params):
    return client.get(
        "/api/dashboard",
        params={"station_id": "ACE0797425", "evse_id": 1, **params},
        headers=AUTH,
    )


# ── Payload ────────────────────────────────────────────────────────────────────


def test_dashboard_payload_covers_the_full_horizon():
    client = TestClient(serve.app)

    response = _get(client, current_soc=0.2, desired_soc=0.8, history_hours=24)

    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == serve.SCHEDULE_HOURS
    assert len(body["actuals"]) == 24
    assert body["station_id"] == "ACE0797425"
    assert body["max_power_kw"] == 11.0

    hours = body["forecast"]
    assert [h["hour"] for h in hours] == list(range(serve.SCHEDULE_HOURS))
    # SoC only ever rises, and each hour picks up where the previous left off.
    assert all(h["soc_end"] >= h["soc_start"] for h in hours)
    assert all(
        hours[i + 1]["soc_start"] == pytest.approx(hours[i]["soc_end"], abs=1e-4)
        for i in range(len(hours) - 1)
    )


def test_dashboard_kpis_are_present_and_consistent():
    client = TestClient(serve.app)

    body = _get(client, current_soc=0.2, desired_soc=0.8).json()
    kpi = body["kpi"]

    for key in (
        "projected_soc_at_departure",
        "soc_target_met",
        "planned_energy_kwh",
        "planned_cost_eur",
        "avg_price_eur_kwh",
        "forced_fill_hours",
        "baseline_cost_eur",
        "savings_eur",
        "savings_pct",
        "actual_energy_kwh",
        "actual_cost_eur",
    ):
        assert key in kpi

    planned = sum(h["energy_kwh"] for h in body["forecast"])
    assert kpi["planned_energy_kwh"] == pytest.approx(planned, abs=1e-3)
    assert kpi["savings_eur"] == pytest.approx(
        kpi["baseline_cost_eur"] - kpi["planned_cost_eur"], abs=1e-3
    )
    # Flat 0.30 EUR/kWh prices, so the average price must come back flat too.
    assert kpi["avg_price_eur_kwh"] == pytest.approx(0.30, abs=1e-6)


def test_soc_guarantee_marks_the_hours_it_forced():
    client = TestClient(serve.app)
    departure = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()

    # 4 h at the fake policy's 0.5 rate delivers 22 kWh — short of the 30 kWh
    # needed for 0.2 -> 0.8, so the guarantee has to top hours up.
    body = _get(
        client, current_soc=0.2, desired_soc=0.8, departure_time=departure
    ).json()

    forced = [h for h in body["forecast"] if h["forced"]]
    assert forced, "expected the SoC guarantee to raise at least one hour"
    assert body["kpi"]["forced_fill_hours"] == len(forced)
    assert all(h["hour"] < body["departure_hour_index"] for h in forced)
    assert body["kpi"]["soc_target_met"] is True


def test_actuals_are_priced_and_gap_filled():
    client = TestClient(serve.app)
    now = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )

    def fake_actuals(_db, _station, since, _until):
        return [
            {"hour": since + timedelta(hours=2), "power_kw": 7.0, "energy_kwh": 7.0}
        ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dashboard, "fetch_actuals_cached", fake_actuals)
        body = _get(client, history_hours=6).json()

    actuals = body["actuals"]
    assert len(actuals) == 6
    assert actuals[0]["t"] == (now - timedelta(hours=6)).isoformat()
    assert actuals[2]["power_kw"] == 7.0
    assert actuals[2]["plugged_in"] is True
    assert actuals[2]["cost_eur"] == pytest.approx(7.0 * 0.30, abs=1e-4)
    # Hours without a meter reading read as "not charging", not as missing.
    assert all(row["power_kw"] == 0.0 for i, row in enumerate(actuals) if i != 2)
    assert body["kpi"]["actual_energy_kwh"] == pytest.approx(7.0, abs=1e-3)
    assert body["kpi"]["actual_cost_eur"] == pytest.approx(7.0 * 0.30, abs=1e-3)


def test_hours_the_price_table_does_not_reach_report_no_cost():
    """An unpriced hour must read as unknown, never as a EUR 0.00 hour."""
    client = TestClient(serve.app)

    def fake_actuals(_db, _station, since, _until):
        return [{"hour": since, "power_kw": 5.0, "energy_kwh": 5.0}]

    # The fixture price table starts 24 h back; ask for 48 h of history.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dashboard, "fetch_actuals_cached", fake_actuals)
        body = _get(client, history_hours=48).json()

    first = body["actuals"][0]
    assert first["power_kw"] == 5.0
    assert first["price_eur_kwh"] is None
    assert first["cost_eur"] is None
    assert body["kpi"]["actual_energy_kwh"] == pytest.approx(5.0, abs=1e-3)
    assert body["kpi"]["actual_cost_eur"] == 0.0
    assert body["kpi"]["actual_unpriced_kwh"] == pytest.approx(5.0, abs=1e-3)


# ── Degradation ────────────────────────────────────────────────────────────────


def test_forecast_still_renders_when_the_database_is_down():
    client = TestClient(serve.app)

    def boom(*_args, **_kwargs):
        raise dashboard.DashboardDataError("connection refused")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dashboard, "fetch_actuals_cached", boom)
        mp.setattr(dashboard, "fetch_recent_sessions", boom)
        response = _get(client)

    assert response.status_code == 200
    body = response.json()
    assert body["data_sources"]["db"] == "unavailable"
    assert body["sessions"] == []
    assert all(row["power_kw"] == 0.0 for row in body["actuals"])
    assert any(w.startswith("database_unreachable") for w in body["warnings"])
    assert len(body["forecast"]) == serve.SCHEDULE_HOURS


def test_missing_weather_is_reported_but_not_fatal():
    client = TestClient(serve.app)

    def no_weather():
        raise RuntimeError("open-meteo down")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(serve, "_fetch_weather", no_weather)
        body = _get(client).json()

    assert body["data_sources"]["weather"] == "unavailable"
    assert "weather_unavailable" in body["warnings"]
    assert len(body["forecast"]) == serve.SCHEDULE_HOURS


def test_dashboard_returns_503_when_price_data_is_missing(monkeypatch):
    client = TestClient(serve.app)

    def raise_missing_prices():
        raise FileNotFoundError("No spot price file")

    monkeypatch.setattr(serve, "_load_prices", raise_missing_prices)
    response = _get(client)

    assert response.status_code == 503
    assert "No spot price file" in response.json()["detail"]


def test_dashboard_rejects_invalid_soc():
    client = TestClient(serve.app)

    assert _get(client, desired_soc=1.5).status_code == 422


# ── Access control ─────────────────────────────────────────────────────────────


def test_dashboard_api_rejects_a_request_without_a_token():
    client = TestClient(serve.app)

    response = client.get(
        "/api/dashboard", params={"station_id": "ACE0797425", "evse_id": 1}
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_dashboard_api_rejects_a_wrong_token():
    client = TestClient(serve.app)

    response = client.get(
        "/api/dashboard",
        params={"station_id": "ACE0797425", "evse_id": 1},
        headers={"Authorization": "Bearer not-the-token"},
    )

    assert response.status_code == 401


def test_dashboard_api_is_disabled_when_the_server_has_no_token(monkeypatch):
    """Fails closed: an unconfigured server refuses rather than opens up."""
    client = TestClient(serve.app)
    monkeypatch.setattr(serve, "DASHBOARD_API_TOKEN", "")

    response = _get(client)

    assert response.status_code == 503
    assert "DASHBOARD_API_TOKEN" in response.json()["detail"]


def test_schedule_is_not_affected_by_the_dashboard_token():
    """The OCPP contract citrineos-payment depends on stays unauthenticated."""
    client = TestClient(serve.app)

    response = client.get(
        "/schedule", params={"station_id": "ACE0797425", "evse_id": 1}
    )

    assert response.status_code == 200


def test_dashboard_requires_a_station():
    client = TestClient(serve.app)

    response = client.get("/api/dashboard", params={"evse_id": 1}, headers=AUTH)

    assert response.status_code == 422
    assert "station_id" in response.json()["detail"]
