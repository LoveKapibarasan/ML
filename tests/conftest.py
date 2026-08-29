import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (must follow the sys.path insert)

# On a host that has an Infisical identity, importing serve would otherwise
# fetch the real vault and inject it into the whole pytest process. Mark the
# load as already done so the suite runs against fixtures only.
config._loaded = True


class FakeModel:
    """Stands in for the trained SAC policy: a constant half-rate action."""

    def predict(self, _obs, deterministic=True):
        return [0.5], None


@pytest.fixture
def patch_runtime_dependencies(monkeypatch):
    """Replaces every external dependency of :mod:`serve` with a fixture.

    Patches the SAC policy, the ENTSO-E price feed, the Open-Meteo
    forecast, the EVSE power lookup and both citrine DB reads, so the
    API tests exercise the server's own logic without a model file, a
    network or a database.

    Yields:
        dict: The ``prices`` and ``weather`` frames the server will see
        and the ``api_token`` ``/api/dashboard`` expects, so a test can
        assert against the same values.
    """
    import dashboard
    import serve

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Reaches one day back, like the live ENTSO-E window, so measured
    # history can be priced too.
    price_index = pd.date_range(
        start=now.replace(tzinfo=None) - pd.Timedelta(hours=24),
        periods=24 + serve.SCHEDULE_HOURS + 4,
        freq="h",
    )
    prices = pd.DataFrame(
        {
            "price": [0.30] * len(price_index),
            "price_3h_future": [0.30] * len(price_index),
        },
        index=price_index,
    )
    weather = pd.DataFrame(
        {
            "temp_c": [18.0] * len(price_index),
            "radiation_wm2": [100.0] * len(price_index),
            "sunshine_duration_s": [600.0] * len(price_index),
        },
        index=price_index,
    )

    monkeypatch.setattr(serve, "DASHBOARD_API_TOKEN", "test-token")
    monkeypatch.setattr(serve, "model", FakeModel())
    monkeypatch.setattr(serve, "_load_prices", lambda: prices)
    monkeypatch.setattr(serve, "_fetch_weather", lambda: weather)
    monkeypatch.setattr(serve, "_get_evse_max_power_kw", lambda *_args: 11.0)
    monkeypatch.setattr(dashboard, "fetch_actuals_cached", lambda *_a, **_k: [])
    monkeypatch.setattr(dashboard, "fetch_recent_sessions", lambda *_a, **_k: [])

    yield {
        "prices": prices,
        "weather": weather,
        "now": now,
        "api_token": "test-token",
    }
