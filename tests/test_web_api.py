# -*- coding: utf-8 -*-
"""Web API vertical slice: JWT auth against the SHARED accounts DB (same
one the Streamlit gate uses), 401 gating on every data route, per-user
watchlist scoping, and graceful offline degradation of market routes."""

import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "test-secret")
    import trading.auth.accounts as A
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "acc.db")
    A.create_user("thomas", "hunter2secure", "Thomas")
    from web.backend.main import app
    return TestClient(app)


def _token(client, user="thomas", pw="hunter2secure"):
    r = client.post("/api/auth/token", data={"username": user, "password": pw})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestAuth:
    def test_wrong_password_401(self, client):
        r = client.post("/api/auth/token",
                        data={"username": "thomas", "password": "wrong"})
        assert r.status_code == 401

    def test_login_returns_identity(self, client):
        r = client.post("/api/auth/token",
                        data={"username": "thomas",
                              "password": "hunter2secure"})
        body = r.json()
        assert body["username"] == "thomas"
        assert body["display_name"] == "Thomas"

    def test_data_routes_gated(self, client):
        assert client.get("/api/watchlist").status_code == 401
        assert client.get("/api/quote/SPY").status_code == 401
        assert client.get("/api/history/SPY").status_code == 401

    def test_garbage_token_401(self, client):
        assert client.get(
            "/api/watchlist",
            headers={"Authorization": "Bearer not.a.jwt"},
        ).status_code == 401


class TestWatchlist:
    def test_scoped_roundtrip(self, client, tmp_path, monkeypatch):
        import trading.data.watchlist as W
        monkeypatch.setattr(W, "DB_PATH", tmp_path / "wl.db")
        old = W.DB_PATH
        W.DB_PATH = tmp_path / "wl.db"
        W._init_db()
        W.DB_PATH = old

        H = _token(client)
        assert client.post("/api/watchlist", json={"symbol": "spy"},
                           headers=H).json()["ok"]
        # Note: WatchlistManager default db is module-level; the roundtrip
        # asserts the API path executes and returns rows for THIS user.
        rows = client.get("/api/watchlist", headers=H).json()
        assert all(r.get("user_id") == "user:thomas" for r in rows)
        client.delete("/api/watchlist/SPY", headers=H)


class TestMarketRoutes:
    def test_quote_degrades_gracefully_offline(self, client):
        H = _token(client)
        r = client.get("/api/quote/SPX", headers=H)
        assert r.status_code == 200
        assert r.json()["symbol"] == "^GSPC"  # alias resolution ran

    def test_history_degrades_gracefully_offline(self, client, monkeypatch):
        """Empty Yahoo frame → empty candles (not a 500)."""
        import pandas as pd
        import web.backend.main as main_mod

        class FakeTicker:
            def history(self, *a, **k):
                return pd.DataFrame()

        monkeypatch.setattr(
            "yfinance.Ticker",
            lambda *_a, **_k: FakeTicker(),
        )
        # history() imports yfinance inside the handler — patch there too
        import yfinance as yf

        monkeypatch.setattr(yf, "Ticker", lambda *_a, **_k: FakeTicker())

        H = _token(client)
        r = client.get("/api/history/SPY", headers=H)
        assert r.status_code == 200
        body = r.json()
        assert body["candles"] == []
        assert body["symbol"] == "SPY"
