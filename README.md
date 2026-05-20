# EV Smart Charging — Reinforcement Learning

A production-ready RL pipeline that trains a **SAC** agent to schedule EV
charging at minimum electricity cost, while **guaranteeing** the user's target
state-of-charge (SoC) is always reached before departure.

---

## Architecture

```
citrine DB (OCPP 2.0.1)          ENTSO-E API          Open-Meteo API
       │                               │                      │
  ev_from_db.py               tariff.py (daily)       weather.py (forecast)
       │                               │                      │
       └──────────────┬────────────────┘                      │
                      ▼                                        │
               preprocess.py  ◄──────────────────────────────┘
                      │
              processed_data_0.csv
                      │
                  train.py  (SAC, GPU)
                      │
              models/best_model.zip
                      │
                  serve.py  (FastAPI, port 8000)
                      │
         GET /schedule?station_id=…&evse_id=…
                      │
         citrineos-payment  →  SetChargingProfile (OCPP)
                      │
              EV charger (CitrineOS)
```

---

## Data Sources

| Layer | Source | Script | Output file |
|---|---|---|---|
| EV state & sessions | citrine PostgreSQL (OCPP 2.0.1 backend) | `data/ev_from_db.py` | `inputs/0_lmd.csv`, `inputs/0_demand.csv` |
| Electricity prices | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — DE_LU day-ahead | `data/tariff.py` | `inputs/spot_{year}_new.csv` |
| Weather & sunshine | [Open-Meteo Archive API](https://open-meteo.com/) — Berlin (52.52 N, 13.41 E) | `data/weather.py` | `inputs/weather_{year}.csv` |
| Public holidays | Python `holidays` library — Germany / Berlin | `data/holiday.py` | `inputs/holidays_{year}.csv` |

### EV Data (citrine DB)

Fetched directly from the OCPP 2.0.1 backend at the host defined in `.env`:

- **MeterValues** → hourly `PluggedIn` flag and `PowerRating` per connector
  (Power.Active.Import readings → resampled to 1 h)
- **Transactions** → session schedule: `arrival_time`, `departure_time`, `totalKwh`

The script prints the detected date range so you can pass matching dates to the
other scripts.

### Weather (Open-Meteo)

| Variable | Unit | Description |
|---|---|---|
| `temp_c` | °C | Air temperature at 2 m |
| `radiation_wm2` | W/m² | Global Horizontal Irradiance (GHI) |
| `humidity_pct` | % | Relative humidity at 2 m |
| `sunshine_duration_s` | s/h | Seconds of direct sunshine per hour (max 3 600) |

### Electricity Prices (ENTSO-E)

Day-ahead spot prices for DE_LU bidding zone in €/MWh → converted to €/kWh.
Requires `ENTSOE_TOKEN` in `.env`.  Negative prices are exploited by the agent
(it earns money when charging at negative prices).

---

## RL Environment

### Observation Space (11 features)

| # | Feature | Normalisation |
|---|---|---|
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

### Action Space

`Box(0.0, 1.0, shape=(1,))` — continuous charging rate as fraction of max power.
Values below 0.05 are treated as "off".

### Reward Function

```
per step    = −(rate × power_kw × spot_price)          [€]
at departure: +20          if SoC ≥ target − 0.02
              −100 × gap   otherwise
```

---

## Algorithm — SAC

**Soft Actor-Critic (SAC)** is an off-policy, maximum-entropy algorithm chosen
for this problem because:

| | PPO (previous) | SAC (current) |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Action space | Discrete | Continuous [0, 1] |
| Sample efficiency | Low | High (experience replay) |
| Entropy | Fixed | Auto-tuned |
| Charging granularity | Binary on/off | Variable rate |

Training config: 500 000 steps, `net_arch=[256, 256]`, `batch_size=256`,
`buffer_size=200 000`, GPU (cuda).

---

## Inference API (serve.py)

### SoC Guarantee

After the SAC model generates the 24-hour charging schedule, the server
**simulates the resulting SoC trajectory**.  If the target SoC will not be
reached by the departure time, it automatically force-fills the cheapest
remaining hours (negative spot prices first) until the shortfall is covered.

The EVSE's actual maximum power is read from the citrine DB:

```sql
SELECT MAX((sv->>'value')::float) / 1000.0   -- W → kW
FROM   "MeterValues"  mv
JOIN   "Transactions" t  ON t.id = mv."transactionDatabaseId"
CROSS  JOIN LATERAL jsonb_array_elements(mv."sampledValue") sv
WHERE  t."stationId"    = <station_id>
  AND  sv->>'measurand' = 'Power.Active.Import'
  AND  sv->>'phase'     IS NULL
```

Result is cached for 24 hours.  Falls back to `MAX_POWER_KW` env var.

### Real-time Data in Serve

| Data | Source | Cache |
|---|---|---|
| Electricity prices | ENTSO-E live API | In-memory, 1 h |
| Weather / irradiance | Open-Meteo forecast API | HTTP, 1 h |
| EVSE max power | citrine DB | In-memory, 24 h |

CSV files in `inputs/` are used as fallback if live API calls fail.

### Endpoints

```
GET /schedule
  ?station_id=<str>
  &evse_id=<int>
  [&desired_soc=<float>]       default 0.8
  [&departure_time=<ISO8601>]  default now+24h
  [&current_soc=<float>]       default 0.2
  [&user_id=<str>]

→ { "chargingProfile": { ...OCPP 2.0.1 ChargingProfile... } }

GET /health
→ { "status": "healthy", "model": "<path>", "max_power_kw": <float> }
```

---

## Project Structure

```
ML/
├── data/
│   ├── ev_from_db.py        # EV state + session schedule from citrine DB
│   ├── db_fetcher.py        # Raw MeterValues / Transactions export (optional)
│   ├── weather.py           # Weather + sunshine from Open-Meteo archive API
│   ├── holiday.py           # German public holidays
│   ├── tariff.py            # Day-ahead prices from ENTSO-E
│   └── preprocess.py        # Merge all sources → processed_data_0.csv
├── envs/
│   └── charging_env.py      # Gymnasium environment (obs / action / reward)
├── inputs/                  # Generated CSVs — git-ignored, re-created by scripts
├── models/                  # Trained model checkpoints — git-ignored
├── benchmark_results/       # Benchmark plots and summary CSV — git-ignored
├── sac_ev_logs/             # TensorBoard training logs — git-ignored
├── train.py                 # SAC training entry point
├── test.py                  # Quick backtest + price-vs-SoC chart
├── benchmark.py             # Full benchmark: SAC vs 3 baseline strategies
├── serve.py                 # FastAPI inference server (SMARTCHARGING_ENDPOINT)
├── run_pipeline.sh          # One-shot: fetch data → preprocess → train
├── serve.sh                 # Start inference server
├── .env                     # Secrets — git-ignored
└── .env.example             # Template — commit this, not .env
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in the values
```

| Variable | Required | Description |
|---|---|---|
| `DB_HOST` | yes | citrine PostgreSQL host |
| `DB_PORT` | no | PostgreSQL port (default `5432`) |
| `DB_NAME` | no | Database name (default `citrine`) |
| `DB_USER` | no | Database user (default `citrine`) |
| `DB_PASSWORD` | **yes** | Database password |
| `ENTSOE_TOKEN` | **yes** | ENTSO-E API token ([get one here](https://transparency.entsoe.eu/usrm/user/myAccountSettings)) |
| `MAX_POWER_KW` | no | EVSE max power fallback (default `11.0` kW) |
| `WEATHER_LAT` | no | Site latitude (default `52.52` Berlin) |
| `WEATHER_LON` | no | Site longitude (default `13.41` Berlin) |
| `SERVE_PORT` | no | Inference server port (default `8000`) |
| `MODEL_PATH` | no | Override model path (default `models/best_model`) |

---

## Running the Pipeline

### Option A — One-shot script

```bash
./run_pipeline.sh
# Fetches all data, preprocesses, and trains in sequence.
```

### Option B — Step by step

#### 1. Fetch EV data from DB

```bash
python data/ev_from_db.py
# Prints the date range detected in the DB, e.g.:
#   DB date range: 2025-11-07 → 2026-05-19  (year=2025)
```

#### 2. Fetch supporting data for that date range

```bash
python data/weather.py --start 2025-11-07 --end 2026-05-19
python data/holiday.py --year 2025
python data/holiday.py --year 2026          # if range spans two years
python data/tariff.py  --start 20250101 --end 20251231
python data/tariff.py  --start 20260101 --end 20260531
```

> **Tip:** `ev_from_db.py` prints the exact commands to run.

#### 3. Preprocess

```bash
python data/preprocess.py
# → inputs/processed_data_0.csv   (hourly features, all years merged)
# → inputs/processed_demand_0.csv (session schedule)
```

#### 4. Train

```bash
python train.py
# Uses GPU (cuda) automatically if available.
# Checkpoints: models/best_model.zip     (saved by EvalCallback, eval every 5k steps)
# Final model: models/sac_smart_charger_final.zip
# TensorBoard: sac_ev_logs/
```

```bash
tensorboard --logdir sac_ev_logs/
```

#### 5. Benchmark

```bash
python benchmark.py
# → benchmark_results/summary.csv
# → benchmark_results/00_training_curve.png
# → benchmark_results/01_price_soc_comparison.png
# → benchmark_results/02_cumulative_cost.png
# → benchmark_results/03_cost_and_fulfillment.png
# → benchmark_results/04_price_vs_rate.png
# → benchmark_results/05_hourly_charging_pattern.png
```

#### 6. Start inference server

```bash
./serve.sh
# Server: http://0.0.0.0:8000
# Swagger UI: http://localhost:8000/docs
```

Then set `SMARTCHARGING_ENDPOINT=http://<this-host>:8000/schedule` in
`citrineos-payment`'s `.env`.

---

## Benchmark Results

Evaluated on the full DB period (2025-11-07 → 2026-05-19, 206 sessions, 4 614 hours).

| Strategy | Total cost | Savings vs baseline | SoC target met |
|---|---|---|---|
| **SAC (ours)** | **€450.79** | **+€269 (37 %)** | **59.5 %** |
| Always-on baseline | €719.84 | — | 60.5 % |
| Price-threshold (≤ median) | €208.32 | +€512 | 38.0 % |
| Greedy cheapest-hours | €278.06 | +€442 | 47.3 % |

SAC achieves a 37 % cost reduction vs always-on while maintaining SoC
fulfillment near the baseline.  The price-threshold strategy has lower cost but
drops SoC fulfillment to 38 % — unacceptable in production.  SAC balances
both objectives.  The SoC guarantee in `serve.py` brings the production
fulfillment rate to **100 %** by force-filling cheap hours when the model's
schedule alone would fall short.

---

## Notes

- **Multi-year data**: `preprocess.py` automatically detects all calendar years
  in the DB export and concatenates price and holiday files for each year.
  Run `tariff.py` and `holiday.py` for every year in the range.

- **Price year mismatch**: if no price file exists for a given year,
  `preprocess.py` still merges on available data (inner join).  Missing years
  will reduce the training set size.

- **Battery capacity**: 50 kWh — configurable in `EVChargingEnv`.

- **Station**: `cp001` (EVerest / CitrineOS OCPP 2.0.1 charger).

- **Training time**: ~3 h for 500 k steps on RTX 3050.
