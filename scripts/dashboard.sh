#!/usr/bin/env bash
# SAC Smart Charging – Streamlit operations dashboard – service manager
# Usage: ./dashboard.sh {start|stop|restart|status|logs}
#
# Runs as its own process next to serve.py and reads it over HTTP, so the
# inference server must be running (or SERVE_URL must point at one).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/.dashboard.pid"
LOG_FILE="$ROOT_DIR/dashboard.log"

HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8501}"
export SERVE_URL="${SERVE_URL:-http://127.0.0.1:${SERVE_PORT:-8000}}"

_activate() {
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
}

_is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
    if _is_running; then
        echo "Already running (PID $(cat "$PID_FILE"))."
        return 0
    fi
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
       netstat -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "ERROR: Port $PORT is already in use."
        exit 1
    fi
    _activate
    echo "[$(date '+%F %T')] Starting dashboard on http://$HOST:$PORT (API: $SERVE_URL) ..." | tee -a "$LOG_FILE"
    nohup streamlit run "$ROOT_DIR/dashboard_app.py" \
        --server.address "$HOST" \
        --server.port "$PORT" \
        --server.headless true \
        --client.toolbarMode minimal \
        --browser.gatherUsageStats false \
        >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if _is_running; then
        echo "Started (PID $(cat "$PID_FILE"))."
        echo "  Logs : $LOG_FILE"
        echo "  UI   : http://$HOST:$PORT"
        echo "  API  : $SERVE_URL/api/dashboard"
    else
        echo "ERROR: Dashboard failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

cmd_stop() {
    if ! _is_running; then
        echo "Not running."
        rm -f "$PID_FILE"
        return 0
    fi
    PID=$(cat "$PID_FILE")
    echo "Stopping PID $PID ..."
    kill "$PID"
    for _ in $(seq 10); do
        _is_running || break
        sleep 1
    done
    if _is_running; then
        echo "Force-killing PID $PID ..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "Stopped."
}

cmd_status() {
    if _is_running; then
        echo "Running (PID $(cat "$PID_FILE")) on http://$HOST:$PORT → $SERVE_URL"
    else
        echo "Not running."
    fi
}

cmd_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No log file at $LOG_FILE"
        return 1
    fi
    tail -f "$LOG_FILE"
}

CMD="${1:-start}"
case "$CMD" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_stop; cmd_start ;;
    status)  cmd_status  ;;
    logs)    cmd_logs    ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
