"""Tests for the Streamlit dashboard's pure helpers.

The Streamlit page itself is exercised by running it; what is unit
tested here is everything underneath: the timeline flattening in
:mod:`dashboard` and the translation table in :mod:`dashboard_app`.
"""

import pandas as pd
import pytest

import dashboard
import dashboard_app


def _payload():
    return {
        "target_soc": 0.8,
        "current_soc": 0.2,
        "departure_time": "2026-08-28T12:00:00",
        "generated_at": "2026-08-28T06:00:00",
        "actuals": [
            {
                "t": "2026-08-28T04:00:00",
                "power_kw": 0.0,
                "energy_kwh": 0.0,
                "plugged_in": False,
                "price_eur_kwh": 0.11,
                "cost_eur": 0.0,
            },
            {
                "t": "2026-08-28T05:00:00",
                "power_kw": 8.0,
                "energy_kwh": 8.0,
                "plugged_in": True,
                "price_eur_kwh": None,
                "cost_eur": None,
            },
        ],
        "forecast": [
            {
                "t": "2026-08-28T06:00:00",
                "hour": 0,
                "price_eur_kwh": -0.02,
                "limit_w": 11000,
                "soc_end": 0.42,
                "cost_eur": -0.22,
                "forced": False,
                "temp_c": 15.0,
                "radiation_wm2": 120.0,
            },
            {
                "t": "2026-08-28T07:00:00",
                "hour": 1,
                "price_eur_kwh": 0.09,
                "limit_w": 5500,
                "soc_end": 0.53,
                "cost_eur": 0.49,
                "forced": True,
                "temp_c": 16.0,
                "radiation_wm2": 240.0,
            },
        ],
    }


# ── Timeline ───────────────────────────────────────────────────────────────────


def test_merge_timeline_orders_history_before_forecast():
    df = dashboard.merge_timeline(_payload())

    assert list(df["phase"]) == ["past", "past", "future", "future"]
    assert df["t"].is_monotonic_increasing
    assert list(df["series"]) == ["measured", "measured", "planned", "forced"]


def test_merge_timeline_converts_watts_to_kw():
    df = dashboard.merge_timeline(_payload())

    assert df.loc[2, "power_kw"] == pytest.approx(11.0)
    assert df.loc[3, "power_kw"] == pytest.approx(5.5)


def test_merge_timeline_keeps_unknown_values_as_nan_not_zero():
    df = dashboard.merge_timeline(_payload())

    assert pd.isna(df.loc[1, "price_eur_kwh"])
    assert pd.isna(df.loc[1, "cost_eur"])
    # SoC and weather exist only for the forecast half.
    assert df["soc"].isna().tolist() == [True, True, False, False]
    assert df["temp_c"].isna().tolist() == [True, True, False, False]


def test_merge_timeline_leaves_a_gap_between_adjacent_bars():
    df = dashboard.merge_timeline(_payload())

    span = df.loc[0, "t_end"] - df.loc[0, "t"]
    assert span == pd.Timedelta(minutes=55)
    assert (df["t_end"] < df["t"].shift(-1)).iloc[:-1].all()


# ── i18n ───────────────────────────────────────────────────────────────────────


def test_every_language_defines_the_same_keys():
    english = set(dashboard_app.I18N["en"])

    for code, table in dashboard_app.I18N.items():
        assert set(table) == english, f"{code} is out of sync with en"


def test_translation_placeholders_match_across_languages():
    import re

    holders = lambda s: set(re.findall(r"{(\w+)}", s))  # noqa: E731

    for key, english in dashboard_app.I18N["en"].items():
        for code, table in dashboard_app.I18N.items():
            assert holders(table[key]) == holders(english), f"{code}.{key}"


def test_tr_substitutes_and_falls_back_to_english():
    assert dashboard_app.tr("ja", "d_target", v="80%") == "目標 80%"
    assert dashboard_app.tr("de", "now_line") == dashboard_app.I18N["en"]["now_line"]


# ── Charts ─────────────────────────────────────────────────────────────────────


def test_charts_build_without_touching_streamlit():
    payload = _payload()
    df = dashboard.merge_timeline(payload)
    now = pd.to_datetime(payload["generated_at"])
    colors = dashboard_app.PALETTE["light"]

    for chart in (
        dashboard_app.price_chart(df, now, "en", colors),
        dashboard_app.power_chart(df, now, "en", colors),
        dashboard_app.soc_chart(df, payload, now, "en", colors),
        dashboard_app.weather_chart(df, "temp_c", "°C", "en", colors),
    ):
        spec = chart.to_dict()
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")


def test_power_chart_labels_all_three_series_in_the_legend():
    payload = _payload()
    df = dashboard.merge_timeline(payload)
    colors = dashboard_app.PALETTE["light"]

    spec = dashboard_app.power_chart(
        df, pd.to_datetime(payload["generated_at"]), "ja", colors
    ).to_dict()
    bars = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "bar")

    assert bars["encoding"]["color"]["scale"]["domain"] == [
        dashboard_app.I18N["ja"]["s_measured"],
        dashboard_app.I18N["ja"]["s_planned"],
        dashboard_app.I18N["ja"]["s_forced"],
    ]
    # Forced-fill shares the planned hue, so opacity carries the difference.
    assert "opacity" in bars["encoding"]
