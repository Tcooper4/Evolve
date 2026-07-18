# -*- coding: utf-8 -*-
"""Live admin toggle for EVOLVE_SHARED_KEYS (persisted override)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from trading.auth import accounts as A
from trading.auth import admin_settings as AS


STRONG_PW = "CorrectHorse9"
ADMIN_PW = "AdminPassw0rd"


@pytest.fixture()
def acc_db(tmp_path, monkeypatch):
    db = tmp_path / "accounts.db"
    monkeypatch.setattr(A, "DB_PATH", db)
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "test-secret-shared-keys-toggle")
    monkeypatch.setenv("EVOLVE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    AS.invalidate_shared_keys_cache()
    yield db
    AS.invalidate_shared_keys_cache()


@pytest.fixture()
def clients(acc_db, monkeypatch):
    A.create_user("admin", ADMIN_PW, "Admin", role="admin", db_path=acc_db)
    A.create_user("bob", STRONG_PW, "Bob", role="user", db_path=acc_db)

    import config.user_store as US

    monkeypatch.setattr(US, "USER_DB_PATH", Path(acc_db).parent / "users.db")

    from web.backend.main import app

    c = TestClient(app)

    def _token(user: str, pw: str) -> dict:
        r = c.post(
            "/api/auth/token",
            data={"username": user, "password": pw},
        )
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return c, _token("admin", ADMIN_PW), _token("bob", STRONG_PW)


class TestSharedKeysOverride:
    def test_env_fallback_when_never_toggled(self, acc_db, monkeypatch):
        from config.api_keys import resolve_api_key

        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:nokey")
        monkeypatch.delenv("EVOLVE_SHARED_KEYS", raising=False)
        AS.invalidate_shared_keys_cache()
        assert AS.get_shared_keys_override() is None
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-SERVER"

        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "0")
        AS.invalidate_shared_keys_cache()
        assert resolve_api_key("ANTHROPIC_API_KEY") is None

    def test_persisted_off_blocks_env_even_when_env_on(self, acc_db, monkeypatch):
        from config.api_keys import resolve_api_key

        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "1")
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:nokey")

        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-SERVER"
        AS.set_shared_keys_override(False, updated_by="admin")
        assert resolve_api_key("ANTHROPIC_API_KEY") is None

    def test_persisted_on_restores_after_off(self, acc_db, monkeypatch):
        from config.api_keys import resolve_api_key

        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "0")  # env alone would block
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:nokey")

        AS.set_shared_keys_override(False, updated_by="admin")
        assert resolve_api_key("ANTHROPIC_API_KEY") is None
        AS.set_shared_keys_override(True, updated_by="admin")
        # Persisted on beats env off.
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-SERVER"

    def test_personal_key_still_wins_when_shared_off(self, acc_db, monkeypatch):
        from config.api_keys import resolve_api_key
        from config.user_store import save_user_api_keys

        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        AS.set_shared_keys_override(False, updated_by="admin")
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-ALICE"})
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-ALICE"


class TestAdminApi:
    def test_admin_can_get_and_toggle(self, clients, monkeypatch):
        c, H_admin, _H_bob = clients
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "1")

        r = c.get("/api/admin/shared-keys", headers=H_admin)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["allowed"] is True
        assert body["source"] == "env"
        assert body["persisted"] is False

        r2 = c.post(
            "/api/admin/shared-keys",
            json={"allowed": False},
            headers=H_admin,
        )
        assert r2.status_code == 200
        assert r2.json()["allowed"] is False
        assert r2.json()["source"] == "persisted"
        assert r2.json()["persisted"] is True

        from config.api_keys import resolve_api_key

        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:nokey")
        assert resolve_api_key("ANTHROPIC_API_KEY") is None

        r3 = c.post(
            "/api/admin/shared-keys",
            json={"allowed": True},
            headers=H_admin,
        )
        assert r3.status_code == 200
        assert r3.json()["allowed"] is True
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-SERVER"

    def test_non_admin_forbidden(self, clients):
        c, _H_admin, H_bob = clients
        assert c.get("/api/admin/shared-keys", headers=H_bob).status_code == 403
        assert (
            c.post(
                "/api/admin/shared-keys",
                json={"allowed": False},
                headers=H_bob,
            ).status_code
            == 403
        )


class TestHotPathPerf:
    def test_override_cache_is_cheap(self, acc_db, monkeypatch):
        """The added lookup must not dominate resolve_api_key.

        After the first read, the override is an in-memory hit — far under
        1 ms. A cold SQLite read for the toggle itself must also stay cheap.
        """
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        AS.set_shared_keys_override(True, updated_by="admin")

        AS.invalidate_shared_keys_cache()
        t_cold0 = time.perf_counter()
        assert AS.get_shared_keys_override() is True
        cold_ms = (time.perf_counter() - t_cold0) * 1000
        assert cold_ms < 100, f"cold shared_keys read too slow: {cold_ms:.1f} ms"

        n = 20_000
        t0 = time.perf_counter()
        for _ in range(n):
            assert AS.get_shared_keys_override() is True
        cached_us = ((time.perf_counter() - t0) / n) * 1_000_000
        assert cached_us < 50, f"cached override too slow: {cached_us:.1f} µs/call"

        # Incremental cost of shared_keys_allowed vs a pure env read.
        from config.api_keys import shared_keys_allowed, _env_shared_keys_allowed

        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "1")
        shared_keys_allowed()  # warm
        n2 = 10_000
        t_a0 = time.perf_counter()
        for _ in range(n2):
            shared_keys_allowed()
        with_override_us = ((time.perf_counter() - t_a0) / n2) * 1_000_000

        t_b0 = time.perf_counter()
        for _ in range(n2):
            _env_shared_keys_allowed()
        env_only_us = ((time.perf_counter() - t_b0) / n2) * 1_000_000

        # Cached path should stay in the same ballpark as getenv (tens of µs).
        assert with_override_us < 100, (
            f"shared_keys_allowed too slow: {with_override_us:.1f} µs "
            f"(env-only {env_only_us:.1f} µs)"
        )
