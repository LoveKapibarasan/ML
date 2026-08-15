from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from envs.charging_env import EVChargingEnv

START = datetime(2026, 1, 1, 0, 0, 0)


def _full_data(n, price=0.20, pluggedin=True, powerrating=11.0):
    idx = [START + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame(
        {
            "date": idx,
            "price": [price] * n,
            "temp_c": [18.0] * n,
            "radiation_wm2": [100.0] * n,
            "sunshine_duration_s": [600.0] * n,
            "is_holiday_x": [0] * n,
            "pluggedin": [pluggedin] * n,
            "powerrating": [powerrating] * n,
            "price_3h_future": [price] * n,
        }
    )


def _demand_data(departure_time, target_soc=0.8, initial_soc=0.2):
    return pd.DataFrame(
        {
            "arrival_time": [START],
            "departure_time": [departure_time],
            "target_soc": [target_soc],
            "initial_soc": [initial_soc],
        }
    )


def _make_env(monkeypatch, full_data, demand_data):
    monkeypatch.setattr(
        EVChargingEnv, "_load_data", lambda self, ev_id, input_dir: (full_data, demand_data)
    )
    return EVChargingEnv()


def test_reset_returns_11_dim_observation_and_initial_soc(monkeypatch):
    full = _full_data(3)
    demand = _demand_data(departure_time=START + timedelta(hours=2), initial_soc=0.35)
    env = _make_env(monkeypatch, full, demand)

    obs, info = env.reset()

    assert obs.shape == (11,)
    assert obs.dtype == np.float32
    assert obs[6] == pytest.approx(0.35)  # SoC
    assert info == {}


def test_action_and_observation_space_shapes(monkeypatch):
    full = _full_data(2)
    demand = _demand_data(departure_time=START + timedelta(hours=1))
    env = _make_env(monkeypatch, full, demand)

    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0
    assert env.observation_space.shape == (11,)


def test_step_below_threshold_does_not_charge_or_cost(monkeypatch):
    full = _full_data(3, price=0.20)
    demand = _demand_data(departure_time=START + timedelta(hours=2), initial_soc=0.2)
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    obs, reward, done, truncated, _ = env.step(np.array([0.01]))  # below 0.05 threshold

    assert env.soc == pytest.approx(0.2)
    assert reward == pytest.approx(0.0)
    assert done is False
    assert truncated is False


def test_step_charges_when_plugged_in_above_threshold(monkeypatch):
    full = _full_data(3, price=0.20, powerrating=11.0)
    demand = _demand_data(departure_time=START + timedelta(hours=2), initial_soc=0.2)
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    _, reward, _, _, _ = env.step(np.array([1.0]))  # full power, above threshold

    expected_soc = 0.2 + (1.0 * 11.0) / env.battery_capacity
    assert env.soc == pytest.approx(expected_soc)
    assert reward == pytest.approx(-(1.0 * 11.0 * 0.20))


def test_step_does_not_charge_when_unplugged(monkeypatch):
    full = _full_data(3, price=0.20, pluggedin=False)
    demand = _demand_data(departure_time=START + timedelta(hours=2), initial_soc=0.2)
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    _, reward, _, _, _ = env.step(np.array([1.0]))

    assert env.soc == pytest.approx(0.2)
    assert reward == pytest.approx(0.0)


def test_departure_bonus_when_target_met(monkeypatch):
    full = _full_data(2, price=0.20)
    demand = _demand_data(
        departure_time=START, target_soc=0.1, initial_soc=0.5
    )  # already above target
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    _, reward, done, _, _ = env.step(np.array([0.0]))  # no charging cost

    assert reward == pytest.approx(20.0)
    assert done is True


def test_departure_penalty_when_target_missed(monkeypatch):
    full = _full_data(2, price=0.20)
    demand = _demand_data(departure_time=START, target_soc=0.9, initial_soc=0.1)
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    _, reward, done, _, _ = env.step(np.array([0.0]))

    assert reward == pytest.approx(-100.0 * 0.8)
    assert done is True


def test_episode_terminates_only_at_last_step(monkeypatch):
    full = _full_data(3)
    demand = _demand_data(departure_time=START + timedelta(hours=10))  # never departs
    env = _make_env(monkeypatch, full, demand)
    env.reset()

    _, _, done0, _, _ = env.step(np.array([0.0]))
    assert done0 is False
    _, _, done1, _, _ = env.step(np.array([0.0]))
    assert done1 is True  # current_step reaches len(full_data) - 1
