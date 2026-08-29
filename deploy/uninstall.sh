#!/usr/bin/env bash
# Stops, disables and removes both systemd services. Leaves the checkout,
# the venv and .env alone.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run with sudo." >&2
    exit 1
fi

for unit in sac-dashboard.service sac-charging.service; do
    systemctl disable --now "$unit" 2>/dev/null || true
    rm -f "/etc/systemd/system/$unit"
    echo "  removed $unit"
done
systemctl daemon-reload
echo "Done."
