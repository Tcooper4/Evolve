# -*- coding: utf-8 -*-
"""Per-user memory scoping: LONG_TERM and PREFERENCE entries were written
with session_id=None (single-user-era design so they'd survive restarts),
which in multi-user mode leaked everything - Bob's chat context included
Alice's trades, and one user changing the active LLM changed it for
everyone. Now scoped to a durable per-user identity with legacy NULL-row
fallback so pre-existing history stays readable."""

import json
import sqlite3
import uuid

import pytest

from trading.memory.memory_store import MemoryStore, MemoryType


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_SESSION_ID", raising=False)
    return MemoryStore(db_path=tmp_path / "mem.db")


def _as(monkeypatch, user):
    monkeypatch.setenv("EVOLVE_SESSION_ID", user)


class TestPreferenceIsolation:
    def test_each_user_owns_their_preferences(self, store, monkeypatch):
        _as(monkeypatch, "user:alice")
        store.upsert_preference("active_llm", {"provider": "claude"})
        _as(monkeypatch, "user:bob")
        store.upsert_preference("active_llm", {"provider": "gpt4"})
        assert store.get_preference("active_llm") == {"provider": "gpt4"}
        _as(monkeypatch, "user:alice")
        assert store.get_preference("active_llm") == {"provider": "claude"}

    def test_personal_mode_durable_across_restart(self, store, tmp_path,
                                                   monkeypatch):
        monkeypatch.delenv("EVOLVE_SESSION_ID", raising=False)
        store.upsert_preference("active_llm", {"provider": "ollama"})
        fresh = MemoryStore(db_path=tmp_path / "mem.db")  # simulated restart
        assert fresh.get_preference("active_llm") == {"provider": "ollama"}


class TestLongTermIsolation:
    def test_trades_do_not_cross_users(self, store, monkeypatch):
        _as(monkeypatch, "user:alice")
        store.upsert(MemoryType.LONG_TERM, "trades", "t1",
                     {"order": "SPY put spread"}, category="orders")
        _as(monkeypatch, "user:bob")
        assert store.list(MemoryType.LONG_TERM, namespace="trades") == []
        _as(monkeypatch, "user:alice")
        assert len(store.list(MemoryType.LONG_TERM, namespace="trades")) == 1

    def test_short_term_still_run_scoped(self, store):
        store.upsert(MemoryType.SHORT_TERM, "ctx", "k", {"v": 1})
        assert len(store.list(MemoryType.SHORT_TERM, namespace="ctx")) == 1


class TestLegacyFallback:
    def _seed_legacy(self, store):
        con = sqlite3.connect(store.db_path)
        for mtype, ns, key, val in (
            ("preference", "global", "theme", "dark"),
            ("long_term", "lessons", "l1", {"lesson": "history"}),
        ):
            con.execute(
                "INSERT INTO memory_entries (id, memory_type, namespace,"
                " session_id, key, category, value_json, created_at,"
                " updated_at) VALUES (?, ?, ?, NULL, ?, 'x', ?,"
                " datetime('now'), datetime('now'))",
                (str(uuid.uuid4()), mtype, ns, key, json.dumps(val)),
            )
        con.commit()
        con.close()

    def test_legacy_rows_readable_and_shadowed(self, store, monkeypatch):
        store.upsert_preference("warmup", 1)  # ensure schema
        self._seed_legacy(store)
        _as(monkeypatch, "user:bob")
        assert store.get_preference("theme") == "dark"          # fallback
        assert len(store.list(MemoryType.LONG_TERM, namespace="lessons")) == 1
        store.upsert_preference("theme", "light")
        assert store.get_preference("theme") == "light"          # shadowed
