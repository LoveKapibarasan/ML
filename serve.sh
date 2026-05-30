#!/usr/bin/env bash
# SAC Smart Charging Inference Server – service manager
# Usage: ./serve.sh {start|stop|restart|status|logs}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.serve.pid"
LOG_FILE="$SCRIPT_DIR/serve.log"

HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8000}"

_activate() {
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.venv/bin/activate"
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
        echo "ERROR: Port $PORT is already in use (systemd service running?)."
        echo "  Stop it first: sudo systemctl stop sac-charging"
        exit 1
    fi
    _activate
    echo "[$(date '+%F %T')] Starting SAC inference server on http://$HOST:$PORT ..." | tee -a "$LOG_FILE"
    nohup uvicorn serve:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1 \
        >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    if _is_running; then
        echo "Started (PID $(cat "$PID_FILE"))."
        echo "  Logs : $LOG_FILE"
        echo "  API  : http://$HOST:$PORT/schedule"
        echo "  Health: http://$HOST:$PORT/health"
    else
        echo "ERROR: Server failed to start. Check $LOG_FILE"
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
    # wait up to 10 s
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
        echo "Running (PID $(cat "$PID_FILE")) on http://$HOST:$PORT"
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
