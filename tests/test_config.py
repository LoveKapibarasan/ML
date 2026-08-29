"""Tests for the Infisical-backed configuration loader."""

import os

import pytest
import requests

import config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Starts each test with no Infisical identity and a clean load flag."""
    for key in (
        "INFISICAL_ENDPOINT",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_ENV",
        "INFISICAL_PATH",
        # cleared too: a real deployment may already have these in the
        # environment, which would mask what the test is asserting
        "DB_PASSWORD",
        "ENTSOE_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "_loaded", False)


def _identity(monkeypatch):
    monkeypatch.setenv("INFISICAL_ENDPOINT", "vault.example")
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csec")


class _Response:
    def __init__(self, payload, ok=True, status=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status

    def json(self):
        return self._payload


def test_no_identity_means_no_network_call(monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("must not call Infisical without an identity")

    monkeypatch.setattr(requests, "post", explode)
    monkeypatch.setattr(requests, "get", explode)

    assert config.fetch_secrets() == {}


def test_secrets_are_read_from_the_configured_folder(monkeypatch):
    _identity(monkeypatch)
    seen = {}

    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response({"accessToken": "tok"})
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return _Response(
            {"secrets": [{"secretKey": "DB_PASSWORD", "secretValue": "s3"}]}
        )

    monkeypatch.setattr(requests, "get", fake_get)

    assert config.fetch_secrets() == {"DB_PASSWORD": "s3"}
    assert seen["url"] == "https://vault.example/api/v3/secrets/raw"
    assert seen["params"]["environment"] == "prod"
    assert seen["params"]["secretPath"] == "/ml"
    assert seen["params"]["workspaceId"] == config.DEFAULT_PROJECT_ID
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_folder_and_environment_are_overridable(monkeypatch):
    _identity(monkeypatch)
    monkeypatch.setenv("INFISICAL_ENV", "staging")
    monkeypatch.setenv("INFISICAL_PATH", "/somewhere-else")
    seen = {}
    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response({"accessToken": "t"})
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, params=None, **kw: (
            seen.update(params),
            _Response({"secrets": []}),
        )[1],
    )

    config.fetch_secrets()

    assert seen["environment"] == "staging"
    assert seen["secretPath"] == "/somewhere-else"


def test_a_rejected_login_is_an_error_not_a_silent_empty_config(monkeypatch):
    _identity(monkeypatch)
    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response(None, ok=False, status=401)
    )

    with pytest.raises(config.ConfigError, match="login failed"):
        config.fetch_secrets()


def test_a_failed_read_is_an_error(monkeypatch):
    _identity(monkeypatch)
    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response({"accessToken": "t"})
    )
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: _Response(None, ok=False, status=403)
    )

    with pytest.raises(config.ConfigError, match="read failed"):
        config.fetch_secrets()


def test_the_environment_wins_over_the_vault(monkeypatch):
    """A value already set locally is an override, not something to clobber."""
    _identity(monkeypatch)
    monkeypatch.setenv("DB_PASSWORD", "local-override")
    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response({"accessToken": "t"})
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: _Response(
            {
                "secrets": [
                    {"secretKey": "DB_PASSWORD", "secretValue": "from-vault"},
                    {"secretKey": "ENTSOE_TOKEN", "secretValue": "from-vault"},
                ]
            }
        ),
    )

    config.load(force=True)

    assert os.environ["DB_PASSWORD"] == "local-override"
    assert os.environ["ENTSOE_TOKEN"] == "from-vault"


def test_load_fetches_once(monkeypatch):
    _identity(monkeypatch)
    calls = []
    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _Response({"accessToken": "t"})
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: (calls.append(1), _Response({"secrets": []}))[1],
    )

    config.load()
    config.load()

    assert len(calls) == 1


def test_main_prints_requested_keys_and_flags_missing(monkeypatch, capsys):
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "ev-dash")

    assert config.main(["KEYCLOAK_CLIENT_ID"]) == 0
    assert capsys.readouterr().out == "ev-dash\n"

    assert config.main(["NOT_SET_ANYWHERE"]) == 1
    assert capsys.readouterr().out == "\n"
