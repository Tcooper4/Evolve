# -*- coding: utf-8 -*-
"""Phase 2 audit: every Settings API key routes via resolve_api_key and
produces an observable before/after change — no bare getenv bypasses.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from trading.auth import accounts as A
from trading.auth import admin_settings as AS


SETTINGS_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "NEWS_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "TWITTER_BEARER_TOKEN",
]


@pytest.fixture(autouse=True)
def _iso(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "test-secret-key-audit")
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "accounts.db")
    import config.user_store as US

    monkeypatch.setattr(US, "USER_DB_PATH", tmp_path / "users.db")
    AS.invalidate_shared_keys_cache()
    AS.set_shared_keys_override(False, updated_by="admin")  # no env fallback
    for k in SETTINGS_KEYS + ["NEWSAPI_KEY", "TWITTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    # Plant operator env keys that MUST stay blocked when shared is off
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-OPERATOR-SHOULD-NOT-LEAK")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-OPENAI-OPERATOR-SHOULD-NOT-LEAK")
    monkeypatch.setenv("NEWS_API_KEY", "news-OPERATOR-SHOULD-NOT-LEAK")
    monkeypatch.setenv("NEWSAPI_KEY", "newsapi-OPERATOR-SHOULD-NOT-LEAK")
    monkeypatch.setenv("REDDIT_CLIENT_ID", "reddit-id-OPERATOR")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "reddit-sec-OPERATOR")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "twitter-OPERATOR-SHOULD-NOT-LEAK")
    monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
    yield
    AS.invalidate_shared_keys_cache()


class TestNoGetenvBypass:
    """With shared keys OFF, env operator keys must not leak through helpers."""

    def test_twitter_bearer_ignores_env_when_shared_off(self):
        from trading.data.twitter_headlines import _bearer

        assert _bearer() is None

    def test_twitter_bearer_uses_personal_key(self):
        from config.user_store import save_user_api_keys
        from trading.data.twitter_headlines import _bearer

        save_user_api_keys(
            "user:alice", {"TWITTER_BEARER_TOKEN": "twitter-ALICE-PERSONAL"}
        )
        assert _bearer() == "twitter-ALICE-PERSONAL"

    def test_reddit_aggregator_ignores_env(self):
        from trading.data import news_aggregator as NA

        # Re-import path uses resolve; with no personal keys → empty
        assert NA._fetch_reddit("AAPL") == []

    def test_reddit_social_ignores_env(self):
        from trading.data.social_sentiment import _reddit_creds_from_runtime

        assert _reddit_creds_from_runtime() == ("", "")

    def test_ai_score_external_keys_ignore_env(self):
        from trading.analysis.ai_score import _has_external_api_keys

        assert _has_external_api_keys() is False

    def test_ai_score_external_keys_see_personal(self):
        from config.user_store import save_user_api_keys
        from trading.analysis.ai_score import _has_external_api_keys

        save_user_api_keys("user:alice", {"NEWS_API_KEY": "news-ALICE"})
        assert _has_external_api_keys() is True


class TestPerKeyResolveAndBehavior:
    def test_anthropic_save_changes_llm_config(self, monkeypatch):
        from config.user_store import save_user_api_keys
        import config.llm_config as L

        importlib.reload(L)
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        before = L.get_llm_config().anthropic_api_key
        assert before in (None, "") or "OPERATOR" not in (before or "")
        # shared off → operator blocked
        assert L.get_llm_config().anthropic_api_key in (None, "")

        save_user_api_keys("user:alice", {"ANTHROPIC_API_KEY": "sk-ant-ALICE-AUDIT"})
        L.reset_llm_config("user:alice")
        after = L.get_llm_config().anthropic_api_key
        assert after == "sk-ant-ALICE-AUDIT"

    def test_openai_save_changes_llm_config(self, monkeypatch):
        from config.user_store import save_user_api_keys
        import config.llm_config as L

        importlib.reload(L)
        monkeypatch.setenv("EVOLVE_SESSION_ID", "user:alice")
        assert L.get_llm_config().openai_api_key in (None, "")

        save_user_api_keys("user:alice", {"OPENAI_API_KEY": "sk-OPENAI-ALICE-AUDIT"})
        L.reset_llm_config("user:alice")
        assert L.get_llm_config().openai_api_key == "sk-OPENAI-ALICE-AUDIT"

    def test_news_key_changes_fetcher_behavior(self, monkeypatch):
        from config.user_store import save_user_api_keys
        from trading.data.news_fetcher import fetch_recent_news

        # Before: no personal key → empty (graceful)
        assert fetch_recent_news("AAPL") == []

        save_user_api_keys("user:alice", {"NEWS_API_KEY": "news-ALICE-AUDIT"})
        called = {}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "articles": [
                        {"title": "Alice-only headline", "description": "d"},
                    ]
                }

        def _fake_get(url, params=None, timeout=10):
            called["apiKey"] = (params or {}).get("apiKey")
            return _Resp()

        with patch("requests.get", side_effect=_fake_get):
            out = fetch_recent_news("AAPL")
        assert called["apiKey"] == "news-ALICE-AUDIT"
        assert out and out[0]["title"] == "Alice-only headline"

    def test_newsapi_alias_in_aggregator(self, monkeypatch):
        from config.api_keys import resolve_api_key
        from config.user_store import save_user_api_keys
        from trading.data.news_aggregator import _fetch_newsapi

        assert resolve_api_key("NEWSAPI_KEY") is None
        save_user_api_keys("user:alice", {"NEWS_API_KEY": "news-ALICE-AGG"})
        assert resolve_api_key("NEWSAPI_KEY") == "news-ALICE-AGG"

        class _Resp:
            status_code = 200

            def json(self):
                return {"articles": [{"title": "Agg", "url": "u", "source": {"name": "S"}}]}

        with patch("requests.get", return_value=_Resp()):
            rows = _fetch_newsapi("AAPL", max_items=3)
        assert rows and rows[0].get("title") == "Agg"

    def test_reddit_pair_changes_credential_resolver(self):
        from config.user_store import save_user_api_keys
        from trading.data.social_sentiment import _reddit_creds_from_runtime

        assert _reddit_creds_from_runtime() == ("", "")
        save_user_api_keys(
            "user:alice",
            {
                "REDDIT_CLIENT_ID": "rid-ALICE",
                "REDDIT_CLIENT_SECRET": "rsec-ALICE",
            },
        )
        assert _reddit_creds_from_runtime() == ("rid-ALICE", "rsec-ALICE")

    def test_twitter_missing_degrades_to_empty(self):
        from trading.data.twitter_headlines import get_breaking_headlines

        out = get_breaking_headlines(max_items=3)
        assert isinstance(out, list)
        assert out == []

    def test_news_context_missing_keys_empty_why(self):
        from trading.services.news_context import explain_headlines

        rows = explain_headlines(
            ["Test headline about markets"],
            user_id="user:alice",
        )
        assert len(rows) == 1
        assert rows[0]["why"] == ""
        assert "key" in (rows[0].get("note") or "").lower() or "LLM" in (rows[0].get("note") or "")


class TestBypassRegression:
    def test_twitter_source_has_no_getenv_fallback(self):
        src = Path("trading/data/twitter_headlines.py").read_text(encoding="utf-8")
        assert "os.getenv(" not in src
        assert "resolve_api_key" in src

    def test_news_aggregator_reddit_no_getenv(self):
        src = Path("trading/data/news_aggregator.py").read_text(encoding="utf-8")
        # Reddit block must not fall back to getenv
        assert "os.getenv(\"REDDIT" not in src
        assert "os.getenv('REDDIT" not in src

    def test_social_sentiment_no_environ_reddit(self):
        src = Path("trading/data/social_sentiment.py").read_text(encoding="utf-8")
        assert "REDDIT_CLIENT_ID" in src
        assert "os.environ.get(\"REDDIT" not in src
        assert "os.environ.get('REDDIT" not in src

    def test_ai_score_has_external_uses_resolve(self):
        src = Path("trading/analysis/ai_score.py").read_text(
            encoding="utf-8", errors="replace"
        )
        # Function body should call resolve_api_key, not os.environ.get loop
        start = src.index("def _has_external_api_keys")
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        assert "resolve_api_key" in body
        assert "os.environ.get" not in body
