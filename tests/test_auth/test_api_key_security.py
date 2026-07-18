# -*- coding: utf-8 -*-
"""API-key security: Fernet at rest, no plaintext in logs/exceptions."""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest
from cryptography.fernet import Fernet

from config.secret_redact import redact_secrets


@pytest.fixture(autouse=True)
def _iso(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    import config.user_store as US

    monkeypatch.setattr(US, "USER_DB_PATH", tmp_path / "users.db")
    yield


class TestSecretRedact:
    def test_redacts_apikey_query(self):
        raw = "https://newsapi.org/v2/everything?q=AAPL&apiKey=super-secret-news-key"
        out = redact_secrets(raw)
        assert "super-secret-news-key" not in out
        assert "REDACTED" in out

    def test_redacts_known_secret(self):
        key = "sk-ant-api03-THISISATEESTKEYVALUE1234567890abcdef"
        msg = f"Anthropic error for key {key}: invalid"
        out = redact_secrets(msg, known=[key])
        assert key not in out
        assert "REDACTED" in out

    def test_exception_str_with_url_does_not_leak(self):
        key = "news-KEY-should-not-appear"
        # Simulate requests.HTTPError-style message that embeds the URL.
        fake = Exception(
            f"401 Client Error: Unauthorized for url: "
            f"https://newsapi.org/v2/everything?apiKey={key}"
        )
        out = redact_secrets(str(fake), known=[key])
        assert key not in out


class TestFernetAtRest:
    def test_db_blob_is_not_plaintext(self, tmp_path):
        from config.user_store import load_user_api_keys, save_user_api_keys
        import config.user_store as US

        secret = "sk-ant-api03-PLAINTEXT-MUST-NOT-HIT-DISK-abcdef"
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": secret})
        con = sqlite3.connect(US.USER_DB_PATH)
        raw = con.execute(
            "SELECT keys FROM user_api_keys WHERE session_id=?",
            ("user:alice",),
        ).fetchone()[0]
        con.close()
        assert secret not in raw
        payload = json.loads(raw)
        assert payload["ANTHROPIC_API_KEY"] != secret
        assert payload["ANTHROPIC_API_KEY"].startswith("gAAAA")  # Fernet
        # Round-trip still works
        loaded = load_user_api_keys("user:alice")
        assert loaded["ANTHROPIC_API_KEY"] == secret

    def test_partial_update_keeps_other_keys_encrypted(self):
        from config.user_store import load_user_api_keys, save_user_api_keys
        import config.user_store as US

        save_user_api_keys(
            "user:alice",
            {
                "ANTHROPIC_API_KEY": "sk-ant-ALICE-AAAAAAAA",
                "OPENAI_API_KEY": "sk-OPENAI-BBBBBBBBBBBB",
            },
        )
        save_user_api_keys("user:alice", {"OPENAI_API_KEY": "sk-OPENAI-NEWNEWNEWNEW"})
        loaded = load_user_api_keys("user:alice")
        assert loaded["ANTHROPIC_API_KEY"] == "sk-ant-ALICE-AAAAAAAA"
        assert loaded["OPENAI_API_KEY"] == "sk-OPENAI-NEWNEWNEWNEW"
        con = sqlite3.connect(US.USER_DB_PATH)
        raw = con.execute(
            "SELECT keys FROM user_api_keys WHERE session_id=?",
            ("user:alice",),
        ).fetchone()[0]
        con.close()
        assert "sk-ant-ALICE" not in raw
        assert "sk-OPENAI" not in raw

    def test_get_keys_api_never_returns_secrets(self, monkeypatch, tmp_path):
        from config.user_store import save_user_api_keys
        from trading.auth import accounts as A

        monkeypatch.setenv("EVOLVE_AUTH_SECRET", "test-secret-keys")
        monkeypatch.setattr(A, "DB_PATH", tmp_path / "acc.db")
        A.create_user("alice", "hunter2secure", role="user")
        save_user_api_keys(
            "user:alice",
            {"ANTHROPIC_API_KEY": "sk-ant-api03-SHOULD-NOT-RETURN"},
        )
        from fastapi.testclient import TestClient
        from web.backend.main import app

        c = TestClient(app)
        tok = c.post(
            "/api/auth/token",
            data={"username": "alice", "password": "hunter2secure"},
        ).json()["access_token"]
        r = c.get(
            "/api/settings/keys",
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("anthropic") is True
        blob = json.dumps(body)
        assert "sk-ant" not in blob
        assert "SHOULD-NOT-RETURN" not in blob


class TestNewsFetcherNoLogLeak:
    def test_failed_request_log_redacts_key(self, monkeypatch, caplog):
        from trading.data import news_fetcher as NF

        secret = "news-SECRET-KEY-XYZ123456"

        monkeypatch.setattr(
            "config.api_keys.resolve_api_key",
            lambda name: secret,
        )

        class Boom:
            def raise_for_status(self):
                raise Exception(
                    f"401 for url: https://newsapi.org/v2/everything?apiKey={secret}"
                )

            def json(self):
                return {}

        def fake_get(*a, **k):
            # Ensure callers did not bake the key into the URL positional arg
            if a and isinstance(a[0], str):
                assert secret not in a[0]
            if k.get("params"):
                assert k["params"].get("apiKey") == secret
            return Boom()

        monkeypatch.setattr(
            "requests.get",
            fake_get,
        )
        with caplog.at_level(logging.DEBUG, logger=NF.logger.name):
            out = NF.fetch_recent_news("AAPL")
        assert out == []
        joined = " ".join(r.message for r in caplog.records)
        assert secret not in joined
