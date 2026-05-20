#!/usr/bin/env bash
# Start the SAC smart charging inference server
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8000}"

echo "Starting SAC inference server on http://$HOST:$PORT ..."
echo "  SMARTCHARGING_ENDPOINT should be set to: http://<this-host>:$PORT/schedule"
echo ""
uvicorn serve:app --host "$HOST" --port "$PORT" --workers 1
