#!/usr/bin/env bash
# Register serve.py as a systemd service.
# Usage: sudo ./install_service.sh
set -euo pipefail

SERVICE_NAME="sac-charging"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Resolve paths relative to this script — no hardcoded home dirs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_UVICORN="$SCRIPT_DIR/.venv/bin/uvicorn"

# When called via "sudo ./install_service.sh", $SUDO_USER is the real user.
# Fallback: owner of the script directory via ls (portable across GNU/BusyBox).
if [[ -n "${SUDO_USER:-}" ]]; then
    RUN_USER="$SUDO_USER"
else
    RUN_USER="$(ls -ld "$SCRIPT_DIR" | awk '{print $3}')"
fi

HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8000}"

# ── Guards ────────────────────────────────────────────────────────────────────
if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: Run as root: sudo $0"
    exit 1
fi

if ! command -v systemctl &>/dev/null; then
    echo "ERROR: systemd not found on this system."
    exit 1
fi

if [[ ! -x "$VENV_UVICORN" ]]; then
    echo "ERROR: uvicorn not found at $VENV_UVICORN"
    echo "  Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ── Generate unit file ────────────────────────────────────────────────────────
# EnvironmentFile is intentionally omitted:
# serve.py calls load_dotenv() and reads .env from WorkingDirectory at startup.
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=SAC Smart Charging Inference Server
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${VENV_UVICORN} serve:app --host ${HOST} --port ${PORT} --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

# ── Enable and start ──────────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable  "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "Installed : $UNIT_FILE"
echo "Running as: $RUN_USER"
echo ""
echo "  systemctl status  $SERVICE_NAME"
echo "  systemctl stop    $SERVICE_NAME"
echo "  systemctl restart $SERVICE_NAME"
echo "  journalctl -u     $SERVICE_NAME -f"
