# -*- coding: utf-8 -*-
"""Per-user API key resolution: each account's stored keys are used for
its own requests; the server's env keys are only a policy-controlled
fallback; and user keys are never injected into the process-global
environment in multi-user mode (the cross-user quota-leak this design
replaces)."""

import importlib
import os

import pytest
from cryptography.fernet import Fernet


@pytest.fixture(autouse=True)
def _iso_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    import config.user_store as US
    from trading.auth import accounts as A
    from trading.auth import admin_settings as AS

    monkeypatch.setattr(US, "USER_DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "accounts.db")
    AS.invalidate_shared_keys_cache()
    monkeypatch.delenv("EVOLVE_SESSION_ID", raising=False)
    monkeypatch.delenv("EVOLVE_SHARED_KEYS", raising=False)
    yield
    AS.invalidate_shared_keys_cache()


def _fresh_llm_config():
    import config.llm_config as L
    importlib.reload(L)
    return L


class TestResolver:
    def test_each_user_gets_own_key(self, monkeypatch):
        from config.user_store import save_user_api_keys
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-ALICE"})
        save_user_api_keys("user:bob", {"ANTHROPIC_API_KEY": "sk-ant-BOB"})
        L = _fresh_llm_config()
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        assert L.get_llm_config().anthropic_api_key == "sk-ant-ALICE"
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:bob")
        assert L.get_llm_config().anthropic_api_key == "sk-ant-BOB"

    def test_shared_fallback_policy(self, monkeypatch):
        from config.api_keys import resolve_api_key
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SERVER")
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:nokey")
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-SERVER"
        monkeypatch.setenv("EVOLVE_SHARED_KEYS", "0")
        assert resolve_api_key("ANTHROPIC_API_KEY") is None

    def test_personal_mode_env_resolves(self, monkeypatch):
        from config.api_keys import resolve_api_key
        monkeypatch.delenv("EVOLVE_REQUIRE_LOGIN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-PERSONAL")
        assert resolve_api_key("ANTHROPIC_API_KEY") == "sk-ant-PERSONAL"

    def test_alias_resolution(self, monkeypatch):
        from config.api_keys import resolve_api_key
        from config.user_store import save_user_api_keys
        save_user_api_keys("user:alice", {"NEWS_API_KEY": "news-alice"})
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        assert resolve_api_key("NEWSAPI_KEY") == "news-alice"


class TestInjectionGuard:
    def test_env_injection_noop_in_live_mode(self, monkeypatch):
        from config.user_store import inject_user_keys_to_env, save_user_api_keys
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-ALICE"})
        inject_user_keys_to_env("user:alice")
        assert os.getenv("ANTHROPIC_API_KEY") is None

    def test_env_injection_still_works_in_personal_mode(self, monkeypatch):
        from config.user_store import inject_user_keys_to_env, save_user_api_keys
        monkeypatch.delenv("EVOLVE_REQUIRE_LOGIN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        save_user_api_keys("local", {"ANTHROPIC_API_KEY": "sk-ant-LOCAL"})
        inject_user_keys_to_env("local")
        assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-LOCAL"


class TestCacheReset:
    def test_reset_invalidates_single_user(self, monkeypatch):
        from config.user_store import save_user_api_keys
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-OLD"})
        L = _fresh_llm_config()
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        assert L.get_llm_config().anthropic_api_key == "sk-ant-OLD"
        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-NEW"})
        assert L.get_llm_config().anthropic_api_key == "sk-ant-OLD"  # cached
        L.reset_llm_config("user:alice")
        assert L.get_llm_config().anthropic_api_key == "sk-ant-NEW"
