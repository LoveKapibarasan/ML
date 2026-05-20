#!/usr/bin/env bash
# Data fetch → preprocess → train (full pipeline)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

echo "=== [1/5] Fetching EV data from DB ==="
python data/ev_from_db.py

START=$(python -c "
import pandas as pd
df = pd.read_csv('inputs/0_lmd.csv')
print(pd.to_datetime(df['date']).min().strftime('%Y-%m-%d'))
")
END=$(python -c "
import pandas as pd
df = pd.read_csv('inputs/0_lmd.csv')
print(pd.to_datetime(df['date']).max().strftime('%Y-%m-%d'))
")
YEAR="${START:0:4}"
echo "    Date range: $START → $END  (year=$YEAR)"

echo "=== [2/5] Fetching weather ($START → $END) ==="
python data/weather.py --start "$START" --end "$END"

echo "=== [3/5] Fetching holidays (year $YEAR) ==="
python data/holiday.py --year "$YEAR"

echo "=== [4/5] Fetching electricity prices (year $YEAR) ==="
python data/tariff.py --start "${YEAR}0101" --end "${YEAR}1231"

echo "=== [5/5] Preprocessing ==="
python data/preprocess.py

echo ""
echo "Data ready. Starting SAC training..."
python train.py
