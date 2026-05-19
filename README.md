# EV Smart Charging — Reinforcement Learning

A real-data reinforcement learning pipeline that trains an agent to schedule
EV charging at minimum cost using live spot-market prices, OCPP session data,
and weather forecasts.

---

## Overview

The agent controls the **charging rate** (0 – 100 % of station capacity) for
a single EV charging station.  Its goal is to:

1. Fill the battery to the target state-of-charge before each departure.
2. Do so as cheaply as possible by shifting load to hours with low (or
   negative) spot prices.

The algorithm is **SAC (Soft Actor-Critic)** — an off-policy, maximum-entropy
method that handles continuous actions efficiently and avoids premature
convergence through automatic entropy tuning.

---

## Data Sources

| Layer | Source | Fetcher | Output file |
|-------|--------|---------|-------------|
| EV state & sessions | [citrine OCPP backend](https://github.com/citrineos) — PostgreSQL (cluster-postgis) | `data/ev_from_db.py` | `inputs/0_lmd.csv`, `inputs/0_demand.csv` |
| Electricity prices | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — Day-Ahead prices (DE_LU) | `data/tariff.py` | `inputs/spot_{year}_new.csv` |
| Weather & sunshine | [Open-Meteo Archive API](https://open-meteo.com/) — Berlin (52.52 N, 13.41 E) | `data/weather.py` | `inputs/weather_{year}.csv` |
| Public holidays | [Python `holidays` library](https://github.com/vacanza/python-holidays) — Germany / Berlin | `data/holiday.py` | `inputs/holidays_{year}.csv` |

### EV Data (citrine DB)

Fetched directly from the OCPP 2.0.1 backend at `172.25.30.7:5432`:

- **MeterValues** → hourly `PluggedIn` flag and `PowerRating` per connector  
  (Energy.Active.Import.Register readings, 1-minute resolution → resampled to 1 h)
- **Transactions** → session schedule: `arrival_time`, `departure_time`, `totalKwh`

### Weather & Sunshine Data (Open-Meteo)

Four hourly variables retrieved from the Open-Meteo archive endpoint:

| Variable | Unit | Description |
|----------|------|-------------|
| `temp_c` | °C | Air temperature at 2 m |
| `radiation_wm2` | W/m² | Global Horizontal Irradiance (GHI) |
| `humidity_pct` | % | Relative humidity at 2 m |
| `sunshine_duration_s` | s/h | Seconds of direct sunshine per hour (max 3 600) |

### Electricity Prices (ENTSO-E)

Day-ahead spot prices for the Germany / Luxembourg bidding zone (DE_LU),
in €/MWh converted to €/kWh.  Requires the environment variable
`ENTSOE_TOKEN` to be set.

---

## Observation Space  (11 features)

| # | Feature | Normalisation |
|---|---------|---------------|
| 0 | Spot price (current hour) | raw €/kWh — can be negative |
| 1 | Air temperature | (°C + 20) / 60 |
| 2 | Solar irradiance | W/m² / 1 000 |
| 3 | Sunshine duration | s / 3 600 → [0, 1] |
| 4 | Public holiday flag | {0, 1} |
| 5 | Plugged-in flag | {0, 1} |
| 6 | Battery SoC | [0, 1] |
| 7 | Time until departure | hours / 24, clipped [0, 1] |
| 8 | Target SoC | [0, 1] |
| 9 | Station max power rating | kW / 22 |
| 10 | Spot price +3 h ahead | raw €/kWh — can be negative |

## Action Space

`Box(0.0, 1.0, shape=(1,))` — continuous charging rate as a fraction of the
station's maximum power.  Values below 0.05 are treated as off.

## Reward Function

```
r_step      = −(rate × powerrating × spot_price)      [€]
r_departure = +20        if SoC ≥ target − 0.02
            = −100 × (target − SoC)   otherwise
```

Negative spot prices are exploited automatically: the agent earns money by
charging when prices are negative.

---

## Algorithm — SAC vs PPO

| | PPO (previous) | SAC (current) |
|--|--|--|
| Policy type | On-policy | Off-policy |
| Action space | Discrete (0 / 1) | Continuous [0, 1] |
| Sample efficiency | Low — discards old data | High — experience replay buffer |
| Entropy | Fixed coefficient | Auto-tuned |
| Charging granularity | Binary on/off | Variable rate |

---

## Project Structure

```
ML/
├── data/
│   ├── ev_from_db.py        # EV state + sessions from citrine DB
│   ├── db_fetcher.py        # Raw MeterValues / Transactions export
│   ├── weather.py           # Weather + sunshine from Open-Meteo
│   ├── holiday.py           # German public holidays
│   ├── tariff.py            # Day-ahead prices from ENTSO-E
│   └── preprocess.py        # Merge all sources → processed_data_0.csv
├── envs/
│   └── charging_env.py      # Gymnasium environment
├── inputs/                  # Generated CSV files (git-ignored)
├── models/                  # Saved model checkpoints
├── train.py                 # SAC training entry point
└── test.py                  # Backtest + plot
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Fetch data

```bash
# EV session data from citrine DB
python data/ev_from_db.py
# → prints the date range, e.g. 2026-05-11 → 2026-05-19

# Weather + sunshine for the same period
python data/weather.py --start 2026-05-11 --end 2026-05-19

# Public holidays
python data/holiday.py --year 2026

# Electricity prices (requires ENTSOE_TOKEN env var)
ENTSOE_TOKEN=<your_token> python data/tariff.py
```

### 2. Preprocess

```bash
python data/preprocess.py
# → inputs/processed_data_0.csv  (hourly features)
# → inputs/processed_demand_0.csv (session schedule)
```

### 3. Train

```bash
python train.py
# Checkpoints saved to models/best_model.zip (EvalCallback)
# Final model saved to models/sac_smart_charger_final.zip
# TensorBoard logs in sac_ev_logs/
```

```bash
tensorboard --logdir sac_ev_logs/
```

### 4. Backtest

```bash
python test.py
# → test_results.csv
# → test_results.png  (price vs SoC chart)
```

---

## Notes

- **Price year mismatch**: if no spot price file exists for the current data
  year, `preprocess.py` automatically shifts the most recent available price
  file to the correct year.  Run `tariff.py` to replace it with real data.
- **Battery capacity**: 50 kWh (configurable in `EVChargingEnv`).
- **Station**: `cp001` (EVerest software charger, OCPP 2.0.1).
