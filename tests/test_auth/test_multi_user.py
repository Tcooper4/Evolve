# -*- coding: utf-8 -*-
"""Tests for multi-user mode: account store, login gating, and per-user
watchlist isolation (incl. the legacy-table migration and the cross-user
alert-sweep stamping)."""

import importlib
import sqlite3
from pathlib import Path

import pytest

from trading.auth import accounts as A


@pytest.fixture()
def acc_db(tmp_path):
    return tmp_path / "accounts.db"


class TestAccounts:
    def test_create_authenticate_lifecycle(self, acc_db):
        A.create_user("alice", "supersecret1", "Alice", db_path=acc_db)
        assert A.authenticate("alice", "supersecret1", db_path=acc_db)
        assert not A.authenticate("alice", "wrong", db_path=acc_db)
        assert not A.authenticate("nobody", "supersecret1", db_path=acc_db)
        A.set_active("alice", False, db_path=acc_db)
        assert not A.authenticate("alice", "supersecret1", db_path=acc_db)
        A.set_active("alice", True, db_path=acc_db)
        A.set_password("alice", "newsecret99", db_path=acc_db)
        assert A.authenticate("alice", "newsecret99", db_path=acc_db)

    def test_validation(self, acc_db):
        with pytest.raises(ValueError):
            A.create_user("bad name", "supersecret1", db_path=acc_db)
        with pytest.raises(ValueError):
            A.create_user("alice", "short", db_path=acc_db)
        A.create_user("alice", "supersecret1", db_path=acc_db)
        with pytest.raises(ValueError):
            A.create_user("alice", "supersecret1", db_path=acc_db)

    def test_hashes_not_plaintext(self, acc_db):
        A.create_user("alice", "supersecret1", db_path=acc_db)
        con = sqlite3.connect(acc_db)
        stored = con.execute(
            "SELECT password_hash FROM accounts WHERE username='alice'"
        ).fetchone()[0]
        assert "supersecret1" not in stored
        assert stored.startswith("$2")
        creds = A.credentials_dict(db_path=acc_db)
        assert creds["usernames"]["alice"]["password"] == stored


class TestWatchlistIsolation:
    def _legacy_db(self, tmp_path):
        db = tmp_path / "wl.db"
        con = sqlite3.connect(db)
        con.execute(
            """CREATE TABLE watchlist (symbol TEXT PRIMARY KEY,
            added_at TEXT NOT NULL, alert_price_above REAL,
            alert_price_below REAL, alert_rsi_below REAL,
            alert_rsi_above REAL, note TEXT, last_triggered TEXT)"""
        )
        con.execute(
            "INSERT INTO watchlist (symbol, added_at) VALUES ('SPY','2026-01-01')"
        )
        con.execute(
            """CREATE TABLE watchlist_alerts_log (id INTEGER PRIMARY KEY
            AUTOINCREMENT, symbol TEXT NOT NULL, alert_type TEXT NOT NULL,
            trigger_value REAL, current_value REAL, triggered_at TEXT NOT NULL)"""
        )
        con.commit()
        con.close()
        return db

    def _module_with_db(self, db):
        import trading.data.watchlist as W
        importlib.reload(W)
        old = W.DB_PATH
        W.DB_PATH = Path(db)
        W._init_db()          # runs the user_id migration
        W.DB_PATH = old
        return W

    def test_migration_preserves_legacy_rows_under_local(self, tmp_path):
        db = self._legacy_db(tmp_path)
        W = self._module_with_db(db)
        local = W.WatchlistManager(db_path=db, user_id="local")
        assert [r["symbol"] for r in local.get_all()] == ["SPY"]

    def test_per_user_isolation(self, tmp_path):
        db = self._legacy_db(tmp_path)
        W = self._module_with_db(db)
        alice = W.WatchlistManager(db_path=db, user_id="user:alice")
        bob = W.WatchlistManager(db_path=db, user_id="user:bob")
        alice.add_ticker("AAPL", alert_price_above=100.0)
        bob.add_ticker("TSLA")
        assert [r["symbol"] for r in alice.get_all()] == ["AAPL"]
        assert [r["symbol"] for r in bob.get_all()] == ["TSLA"]
        alice.remove_ticker("TSLA")  # cannot delete bob's row
        assert [r["symbol"] for r in bob.get_all()] == ["TSLA"]

    def test_alert_sweep_covers_all_users_and_stamps_owner(self, tmp_path):
        db = self._legacy_db(tmp_path)
        W = self._module_with_db(db)
        alice = W.WatchlistManager(db_path=db, user_id="user:alice")
        bob = W.WatchlistManager(db_path=db, user_id="user:bob")
        alice.add_ticker("AAPL", alert_price_above=100.0)
        fired = bob.check_alerts({"AAPL": {"price": 150.0, "rsi": 50.0}})
        assert any(a["symbol"] == "AAPL" for a in fired)
        stamped = [r for r in alice.get_all() if r["symbol"] == "AAPL"][0]
        assert stamped["last_triggered"], "owner's row must be stamped"
        assert all(r["symbol"] != "AAPL" for r in bob.get_all())


class TestGateModes:
    def test_personal_mode_is_noop(self, monkeypatch):
        monkeypatch.delenv("EVOLVE_REQUIRE_LOGIN", raising=False)
        from trading.auth.gate import login_required, require_login

        assert not login_required()
        assert require_login() is None

    def test_live_mode_flag(self, monkeypatch):
        from trading.auth.gate import login_required

        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "1")
        assert login_required()
        monkeypatch.setenv("EVOLVE_REQUIRE_LOGIN", "0")
        assert not login_required()
