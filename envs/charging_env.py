from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class EVChargingEnv(gym.Env):
    """Gymnasium environment for scheduling EV charging at minimum cost.

    The agent chooses a continuous charging rate each hour. It is rewarded
    for charging when electricity is cheap and penalized for missing the
    driver's target state-of-charge (SoC) by the departure hour.

    Action space:
        ``Box(1,)`` continuous, range ``[0, 1]`` — charging rate as a
        fraction of the station's maximum power rating (``0`` = off,
        ``1`` = full power).

    Observation space:
        ``Box(11,)``, unbounded (some features are raw, signed values).
        Feature layout, in index order:

        0. Spot price — raw €/kWh (can be negative).
        1. Air temperature — ``(°C + 20) / 60``.
        2. Solar irradiance — ``W/m² / 1000``.
        3. Sunshine duration — ``s / 3600``, clipped to ``[0, 1]``.
        4. Public holiday flag — ``0`` or ``1``.
        5. Plugged-in flag — ``0`` or ``1``.
        6. Battery SoC — ``[0, 1]``.
        7. Time left to departure — ``hours / 24``, clamped to ``[0, 1]``.
        8. Target SoC — ``[0, 1]``.
        9. Max power rating — ``kW / 22``.
        10. Spot price 3 hours ahead — raw €/kWh (can be negative).

    Reward:
        Per step: ``-(charging_rate * power_rating * price)`` in euros.
        On the departure step: ``+20`` if ``SoC >= target_SoC - 0.02``,
        otherwise ``-100 * shortfall``.

    Attributes:
        full_data (pandas.DataFrame): Hourly price/weather/plug-state rows,
            loaded by :meth:`_load_data`.
        demand_data (pandas.DataFrame): One row per charging session
            (arrival, departure, target/initial SoC).
        battery_capacity (float): Battery capacity in kWh, fixed at 50.0.
        soc (float): Current state of charge, in ``[0, 1]``.
    """

    #: Minimum charging rate treated as "off" (avoids infinitesimal charges).
    _CHARGE_THRESHOLD = 0.05

    def __init__(self, ev_id: int = 0, input_dir: str = "./inputs"):
        """Initializes the environment and loads its input data.

        Args:
            ev_id: Identifier used to locate ``processed_data_{ev_id}.csv``
                and ``processed_demand_{ev_id}.csv`` under ``input_dir``.
            input_dir: Directory containing the preprocessed input CSVs.
        """
        super().__init__()

        self.full_data, self.demand_data = self._load_data(ev_id, input_dir)
        self.battery_capacity = 50.0  # kWh

        # Continuous charging rate: 0.0 (off) → 1.0 (full power)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Unbounded — normalised values can fall outside [0, 1] for price signals
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self.reset()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self, ev_id: int, input_dir: str):
        """Loads the hourly data and demand-session tables from disk.

        Args:
            ev_id: Identifier used to locate the ``processed_data_{ev_id}.csv``
                and ``processed_demand_{ev_id}.csv`` files.
            input_dir: Directory containing the preprocessed input CSVs.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: ``(full_data,
            demand_data)`` — the hourly price/weather/plug-state table and
            the per-session demand table, respectively.
        """
        path = Path(input_dir).resolve()
        df = pd.read_csv(path / f"processed_data_{ev_id}.csv", parse_dates=["date"])
        df_demand = pd.read_csv(
            path / f"processed_demand_{ev_id}.csv",
            parse_dates=["arrival_time", "departure_time"],
        )
        return df, df_demand

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Builds the 11-feature observation vector for the current step.

        Returns:
            numpy.ndarray: ``float32`` array of shape ``(11,)`` — see the
            class docstring for the feature layout.
        """
        row = self.full_data.iloc[self.current_step]
        demand = self.demand_data.iloc[self.current_demand_idx]

        time_left = (demand["departure_time"] - row["date"]).total_seconds() / 3600.0

        return np.array(
            [
                row["price"],  # 0  €/kWh
                (row["temp_c"] + 20.0) / 60.0,  # 1  temperature
                min(row["radiation_wm2"] / 1000.0, 1.0),  # 2  irradiance
                row["sunshine_duration_s"] / 3600.0,  # 3  sunshine
                float(row["is_holiday_x"]),  # 4  holiday
                float(
                    str(row["pluggedin"]).lower() not in ("false", "0", "")
                ),  # 5  plugged in
                float(self.soc),  # 6  SoC
                float(np.clip(time_left / 24.0, 0.0, 1.0)),  # 7  time left
                float(demand["target_soc"]),  # 8  target SoC
                min(row["powerrating"] / 22.0, 1.0),  # 9  power rating
                row["price_3h_future"],  # 10 future price
            ],
            dtype=np.float32,
        )

    # ── reward ────────────────────────────────────────────────────────────────

    def _calculate_reward(
        self,
        rate: float,
        row: pd.Series,
        is_departure: bool,
        demand: pd.Series,
    ) -> float:
        """Computes the reward for one step.

        Args:
            rate: Charging rate in ``[0, 1]``, as a fraction of the
                station's power rating.
            row: The current row of :attr:`full_data` (price, power
                rating, plug state, etc.).
            is_departure: Whether this step is the session's departure
                step.
            demand: The current row of :attr:`demand_data` (holds
                ``target_soc`` for the shortfall penalty).

        Returns:
            float: Reward in euros — negative charging cost, plus (on
            departure) a ``+20`` bonus or a proportional shortfall
            penalty.
        """
        reward = 0.0

        pluggedin = str(row["pluggedin"]).lower() not in ("false", "0", "")
        if rate > self._CHARGE_THRESHOLD and pluggedin:
            actual_power = rate * row["powerrating"]  # kW
            cost = actual_power * row["price"]  # €  (negative price = free energy)
            reward -= cost

        if is_departure:
            shortfall = demand["target_soc"] - self.soc
            if shortfall <= 0.02:
                reward += 20.0
            else:
                reward -= 100.0 * shortfall

        return reward

    # ── step / reset ──────────────────────────────────────────────────────────

    def step(self, action):
        """Advances the environment by one hour.

        Args:
            action: Array-like containing a single charging rate in
                ``[0, 1]``; values outside that range are clipped.

        Returns:
            tuple: ``(observation, reward, terminated, truncated, info)``
            following the Gymnasium API. ``truncated`` is always
            ``False``; ``info`` is always ``{}``.
        """
        rate = float(
            np.clip(np.asarray(action).flat[0], 0.0, 1.0)
        )  # scalar charging rate
        row = self.full_data.iloc[self.current_step]
        demand = self.demand_data.iloc[self.current_demand_idx]

        is_departure = row["date"] >= demand["departure_time"]
        reward = self._calculate_reward(rate, row, is_departure, demand)

        # Update SoC
        pluggedin = str(row["pluggedin"]).lower() not in ("false", "0", "")
        if rate > self._CHARGE_THRESHOLD and pluggedin:
            added = (rate * row["powerrating"]) / self.battery_capacity
            self.soc = min(1.0, self.soc + added)

        # Advance demand session on departure
        if is_departure and self.current_demand_idx < len(self.demand_data) - 1:
            self.current_demand_idx += 1
            self.soc = float(
                self.demand_data.iloc[self.current_demand_idx]["initial_soc"]
            )

        self.current_step += 1
        done = self.current_step >= len(self.full_data) - 1

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        """Resets the environment to the first row of the first session.

        Args:
            seed: Optional random seed, forwarded to
                ``gymnasium.Env.reset``.
            options: Unused; accepted for Gymnasium API compatibility.

        Returns:
            tuple[numpy.ndarray, dict]: ``(observation, info)``, with
            ``info`` always ``{}``.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.current_demand_idx = 0
        self.soc = float(self.demand_data.iloc[0]["initial_soc"])
        return self._get_obs(), {}
