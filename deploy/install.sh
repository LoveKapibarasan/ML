#!/usr/bin/env bash
# Installs the API and the dashboard as systemd services.
#
#   sudo ./deploy/install.sh            # install, enable and start both
#   sudo ./deploy/install.sh --no-start # install and enable only
#
# The unit files in this directory are the source of truth; this script only
# fills in the paths and ports and copies them into /etc/systemd/system.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
UNIT_DIR=/etc/systemd/system

# "sudo ./deploy/install.sh" leaves the real user in $SUDO_USER; fall back to
# whoever owns the checkout.
RUN_USER="${SERVICE_USER:-${SUDO_USER:-$(stat -c '%U' "$ROOT_DIR")}}"

API_HOST="${SERVE_HOST:-0.0.0.0}"
API_PORT="${SERVE_PORT:-8000}"
DASH_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASH_PORT="${DASHBOARD_PORT:-8501}"

START=1
[[ "${1:-}" == "--no-start" ]] && START=0

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run with sudo." >&2
    exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/uvicorn" ]]; then
    echo "ERROR: $ROOT_DIR/.venv is missing or incomplete." >&2
    echo "  python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "ERROR: $ROOT_DIR/.env not found (needs at least the Infisical identity)." >&2
    exit 1
fi

_install_unit() {
    local src="$DEPLOY_DIR/$1" dst="$UNIT_DIR/$1"
    sed -e "s|__USER__|$RUN_USER|g" \
        -e "s|__ROOT__|$ROOT_DIR|g" \
        -e "s|__API_HOST__|$API_HOST|g" \
        -e "s|__API_PORT__|$API_PORT|g" \
        -e "s|__DASH_HOST__|$DASH_HOST|g" \
        -e "s|__DASH_PORT__|$DASH_PORT|g" \
        "$src" > "$dst"
    chmod 644 "$dst"
    echo "  wrote $dst"
}

echo "Installing services for $RUN_USER from $ROOT_DIR ..."
_install_unit sac-charging.service
_install_unit sac-dashboard.service

systemctl daemon-reload
systemctl enable sac-charging.service sac-dashboard.service >/dev/null
echo "  enabled at boot"

if [[ $START -eq 1 ]]; then
    systemctl restart sac-charging.service
    systemctl restart sac-dashboard.service
    sleep 3
    echo
    systemctl --no-pager --lines=0 status sac-charging.service  | head -4
    systemctl --no-pager --lines=0 status sac-dashboard.service | head -4
fi

cat <<TXT

Done.
  API       : http://$API_HOST:$API_PORT/health
  Dashboard : http://$DASH_HOST:$DASH_PORT

  systemctl status  sac-charging sac-dashboard
  systemctl restart sac-dashboard
  journalctl -u sac-dashboard -f
TXT
