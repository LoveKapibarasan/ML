"""Runtime configuration: ``.env`` first, then Infisical.

The deployed host keeps only its Infisical machine-identity credentials in
``.env``; everything else — database password, ENTSO-E token, Keycloak
client, dashboard settings — lives in Infisical under the ``citrineos``
project, ``prod`` environment, ``/ml`` folder, and is pulled into the
process environment at startup.

Precedence is: anything already set in the environment (including
``.env``) wins, and Infisical fills the gaps. That keeps a local override
working while making the vault the source of truth in deployment.

If no Infisical credentials are present the fetch is skipped silently, so
tests and local development work from ``.env`` alone.

Also usable as a script, which is how ``scripts/dashboard.sh`` reads
values without needing the Infisical CLI::

    python config.py KEYCLOAK_CLIENT_ID KEYCLOAK_CLIENT_SECRET
"""

import os
import sys

import requests
from dotenv import load_dotenv

# citrineos project; override with INFISICAL_PROJECT_ID if it ever moves.
DEFAULT_PROJECT_ID = "98538adb-d5d9-46f6-9920-8f66af3122f0"
DEFAULT_ENVIRONMENT = "prod"
DEFAULT_PATH = "/ml"
REQUEST_TIMEOUT_S = 20

_loaded = False


class ConfigError(RuntimeError):
    """Raised when Infisical is configured but cannot be read."""


def _login(endpoint: str, client_id: str, client_secret: str) -> str:
    """Exchanges a machine identity for an Infisical access token.

    Args:
        endpoint: Infisical host, e.g. ``env.ai-charge.net``.
        client_id: Universal-auth client ID.
        client_secret: Universal-auth client secret.

    Returns:
        str: A short-lived access token.

    Raises:
        ConfigError: If the login is rejected.
    """
    response = requests.post(
        f"https://{endpoint}/api/v1/auth/universal-auth/login",
        json={"clientId": client_id, "clientSecret": client_secret},
        timeout=REQUEST_TIMEOUT_S,
    )
    if not response.ok:
        raise ConfigError(f"Infisical login failed: {response.status_code}")
    return response.json()["accessToken"]


def fetch_secrets() -> dict[str, str]:
    """Reads this service's secrets from Infisical.

    Returns:
        dict[str, str]: Key/value pairs from the configured folder, or an
        empty dict when no machine identity is configured.

    Raises:
        ConfigError: If credentials are present but the read fails —
            silently starting with a half-configured process would be
            worse than refusing.
    """
    endpoint = os.getenv("INFISICAL_ENDPOINT")
    client_id = os.getenv("INFISICAL_CLIENT_ID")
    client_secret = os.getenv("INFISICAL_CLIENT_SECRET")
    if not (endpoint and client_id and client_secret):
        return {}

    token = _login(endpoint, client_id, client_secret)
    response = requests.get(
        f"https://{endpoint}/api/v3/secrets/raw",
        params={
            "workspaceId": os.getenv("INFISICAL_PROJECT_ID", DEFAULT_PROJECT_ID),
            "environment": os.getenv("INFISICAL_ENV", DEFAULT_ENVIRONMENT),
            "secretPath": os.getenv("INFISICAL_PATH", DEFAULT_PATH),
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if not response.ok:
        raise ConfigError(f"Infisical read failed: {response.status_code}")
    return {s["secretKey"]: s["secretValue"] for s in response.json()["secrets"]}


def load(force: bool = False) -> dict[str, str]:
    """Loads ``.env`` and then fills the environment from Infisical.

    Safe to call more than once; the fetch happens only on the first
    call unless ``force`` is set.

    Args:
        force: Re-read even if a previous call already loaded.

    Returns:
        dict[str, str]: The keys taken from Infisical (empty if none).
    """
    global _loaded
    load_dotenv()
    if _loaded and not force:
        return {}

    secrets = fetch_secrets()
    for key, value in secrets.items():
        os.environ.setdefault(key, value)
    _loaded = True
    return secrets


def main(argv: list[str]) -> int:
    """Prints the requested configuration values, one per line.

    Args:
        argv: Key names to print.

    Returns:
        int: ``0`` if every key resolved, ``1`` otherwise.
    """
    load()
    missing = False
    for key in argv:
        value = os.getenv(key, "")
        if not value:
            missing = True
        print(value)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
