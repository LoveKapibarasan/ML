from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import serve


@pytest.fixture(autouse=True)
def _patched(patch_runtime_dependencies):
    """Applies the shared runtime patches (see ``tests/conftest.py``)."""
    return patch_runtime_dependencies


def test_health_returns_model_and_status():
    client = TestClient(serve.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"]
    assert body["max_power_kw"] > 0


def test_schedule_returns_ocpp_charging_profile():
    client = TestClient(serve.app)
    departure = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()

    response = client.get(
        "/schedule",
        params={
            "station_id": "ACE0797425",
            "evse_id": 1,
            "desired_soc": 0.8,
            "current_soc": 0.2,
            "departure_time": departure,
        },
    )

    assert response.status_code == 200
    profile = response.json()["chargingProfile"]
    assert profile["chargingProfilePurpose"] == "TxDefaultProfile"
    assert profile["chargingProfileKind"] == "Absolute"
    schedule = profile["chargingSchedule"][0]
    assert schedule["chargingRateUnit"] == "W"
    periods = schedule["chargingSchedulePeriod"]
    assert len(periods) == serve.SCHEDULE_HOURS
    assert periods[0]["startPeriod"] == 0
    assert periods[-1]["startPeriod"] == (serve.SCHEDULE_HOURS - 1) * 3600
    assert all(0 <= row["limit"] <= 11_000 for row in periods)


def test_schedule_rejects_invalid_soc():
    client = TestClient(serve.app)

    response = client.get(
        "/schedule",
        params={"station_id": "ACE0797425", "evse_id": 1, "desired_soc": 1.5},
    )

    assert response.status_code == 422


def test_schedule_returns_503_when_price_data_is_missing(monkeypatch):
    client = TestClient(serve.app)

    def raise_missing_prices():
        raise FileNotFoundError("No spot price file")

    monkeypatch.setattr(serve, "_load_prices", raise_missing_prices)
    response = client.get(
        "/schedule",
        params={"station_id": "ACE0797425", "evse_id": 1},
    )

    assert response.status_code == 503
    assert "No spot price file" in response.json()["detail"]


def test_importing_serve_does_not_require_a_checkpoint(monkeypatch):
    """CI has no models/ — importing must not be the thing that loads it."""
    import serve as serve_module

    monkeypatch.setattr(serve_module, "_model_path", None)
    monkeypatch.setattr(serve_module, "model", None)

    def no_checkpoint():
        raise FileNotFoundError("No trained model found")

    monkeypatch.setattr(serve_module, "_find_model", no_checkpoint)

    # Reporting is tolerant …
    assert serve_module.model_path() == "not found"
    assert TestClient(serve_module.app).get("/health").status_code == 200
    # … but actually serving is not.
    with pytest.raises(FileNotFoundError):
        serve_module.get_model()


def test_get_model_returns_an_already_assigned_policy(monkeypatch):
    sentinel = object()
    import serve as serve_module

    monkeypatch.setattr(serve_module, "model", sentinel)

    def explode():
        raise AssertionError("must not reload when a policy is already set")

    monkeypatch.setattr(serve_module, "_find_model", explode)

    assert serve_module.get_model() is sentinel
