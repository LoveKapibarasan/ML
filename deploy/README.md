# deploy/

systemd units for the two long-running processes, kept in git so the
deployed configuration is reviewable and reproducible.

| Unit | Process | Default |
|---|---|---|
| `sac-charging.service` | `uvicorn serve:app` — the inference API | `0.0.0.0:8000` |
| `sac-dashboard.service` | `streamlit run dashboard_app.py` — the operations dashboard | `0.0.0.0:8501` |

The unit files carry `__PLACEHOLDER__` tokens; `install.sh` fills in the
checkout path, the service user and the ports, then writes the result to
`/etc/systemd/system`. **Edit the files here, never the installed copies** —
the next install overwrites them.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # fill in the Infisical machine identity
sudo ./deploy/install.sh
```

Overridable via the environment: `SERVICE_USER`, `SERVE_HOST`, `SERVE_PORT`,
`DASHBOARD_HOST`, `DASHBOARD_PORT`.

```bash
sudo ./deploy/install.sh --no-start   # install and enable without starting
sudo ./deploy/uninstall.sh            # stop, disable and remove both
```

## Operating

```bash
systemctl status  sac-charging sac-dashboard
systemctl restart sac-dashboard
journalctl -u sac-dashboard -f
```

## Notes

- Both units read `EnvironmentFile=<root>/.env`, which on a deployed host holds
  **only** the Infisical machine identity.  `config.load()` pulls the rest from
  Infisical (`citrineos` project, `prod` environment, `/ml` folder).
- The dashboard's `ExecStartPre` runs `scripts/dashboard.sh secrets`, which
  writes `.streamlit/secrets.toml` from Infisical.  It exits non-zero when
  there are no Keycloak credentials, so **the unit fails to start rather than
  coming up unauthenticated**.
- The API loads the model on first request, not at import, so both units start
  quickly and a missing checkpoint surfaces as a request error rather than a
  boot loop.
- `scripts/serve.sh` and `scripts/dashboard.sh` still work for foreground and
  local use; they refuse to start if the port is already taken by the service.
