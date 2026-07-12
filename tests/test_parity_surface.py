# -*- coding: utf-8 -*-
"""Coverage for the parity-build surface (Cursor-assisted commit
ad065c54): the paper-portfolio cash integration this session added, a
systematic auth+no-crash sweep over every new route, and targeted
correctness checks on the pieces most likely to be silently wrong
(position semantics, Monte Carlo percentile ordering).
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Cash integration: hand-verified accounting, now formalized
# ---------------------------------------------------------------------------

@pytest.fixture()
def pp(monkeypatch, tmp_path):
    import trading.portfolio.paper_portfolio as PP
    monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
    return PP


class TestCashIntegration:
    """Before this session, cashbook and paper_portfolio were two
    disconnected ledgers: buying $1,000 of AAPL left the displayed cash
    balance untouched at $100,000. These tests lock in that a trade
    actually moves cash, atomically, with the same money."""

    def test_starts_at_default_and_buy_deducts(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        assert p.get_cash() == 100_000.0
        r = p.record_trade("AAPL", "buy", 10, 100)
        assert r["cash_balance"] == 99_000.0
        assert p.get_cash() == 99_000.0

    def test_insufficient_cash_rejected_and_unchanged(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        bad = p.record_trade("AAPL", "buy", 100_000, 100)
        assert bad["success"] is False
        assert "insufficient" in bad["error"]
        assert p.get_cash() == 100_000.0  # untouched by the rejected trade

    def test_sell_credits_cash(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("AAPL", "buy", 10, 100)
        p.record_trade("AAPL", "sell", 5, 120)
        assert p.get_cash() == 99_000.0 + 600.0

    def test_equity_is_cash_plus_market_value(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("AAPL", "buy", 10, 100)  # cash -> 99,000
        s = p.get_summary(price_fn=lambda _: 130.0)  # 10 * 130 = 1300
        assert s["total_equity"] == 99_000.0 + 1300.0

    def test_manual_adjust_and_floor(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.adjust_cash(500)
        assert p.get_cash() == 100_500.0
        p.adjust_cash(-999_999)
        assert p.get_cash() == 0.0  # no paper margin/debt

    def test_cash_isolated_per_user(self, pp):
        a = pp.PaperPortfolio(user_id="user:a")
        b = pp.PaperPortfolio(user_id="user:b")
        a.record_trade("AAPL", "buy", 10, 100)
        assert a.get_cash() == 99_000.0
        assert b.get_cash() == 100_000.0


class TestLimitOrders:
    """Before this session, limit orders were written to a list and
    never checked against anything - not a real order book. These
    verify the executor actually fills orders when price crosses."""

    def test_buy_limit_does_not_fill_above_target(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.place_limit_order("AAPL", "buy", 10, 100)
        filled = p.check_limit_orders(price_fn=lambda _: 105.0)
        assert filled == []
        assert p.get_cash() == 100_000.0

    def test_buy_limit_fills_when_price_crosses(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.place_limit_order("AAPL", "buy", 10, 100)
        filled = p.check_limit_orders(price_fn=lambda _: 98.0)
        assert len(filled) == 1 and filled[0]["filled_price"] == 98.0
        assert p.get_cash() == 100_000.0 - 980.0
        assert p.get_positions()[0]["quantity"] == 10
        assert p.get_limit_orders(include_filled=False) == []

    def test_sell_limit_fills_at_or_above_target(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("AAPL", "buy", 5, 90)
        p.place_limit_order("AAPL", "sell", 5, 110)
        assert p.check_limit_orders(price_fn=lambda _: 105.0) == []
        filled = p.check_limit_orders(price_fn=lambda _: 115.0)
        assert len(filled) == 1

    def test_oversell_limit_rejected_at_placement(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        bad = p.place_limit_order("AAPL", "sell", 999, 200)
        assert bad["success"] is False

    def test_cancel_removes_open_order(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        r = p.place_limit_order("SPY", "buy", 1, 500)
        c = p.cancel_limit_order(r["id"])
        assert c["success"] is True
        assert p.get_limit_orders(include_filled=False) == []


class TestCashbookAPIWiredToPortfolio:
    """The cash a user sees in the Cashbook page must be the SAME cash
    that moves when they trade from the Portfolio page - the whole
    point of unifying the two ledgers."""

    def _client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
        import trading.auth.accounts as A
        import trading.portfolio.paper_portfolio as PP
        monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
        monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
        A.create_user("thomas", "hunter2secure")
        from fastapi.testclient import TestClient

        from web.backend.main import app
        c = TestClient(app)
        tok = c.post("/api/auth/token",
                     data={"username": "thomas",
                           "password": "hunter2secure"}).json()["access_token"]
        return c, {"Authorization": f"Bearer {tok}"}

    def test_buy_via_portfolio_route_moves_cashbook_cash(
        self, monkeypatch, tmp_path
    ):
        c, H = self._client(monkeypatch, tmp_path)
        assert c.get("/api/cashbook", headers=H).json()["cash"] == 100_000.0
        c.post("/api/portfolio/trade",
              json={"symbol": "AAPL", "side": "buy", "quantity": 10,
                    "price": 100}, headers=H)
        cb = c.get("/api/cashbook", headers=H).json()
        assert cb["cash"] == 99_000.0

    def test_portfolio_equity_matches_cashbook_cash_plus_value(
        self, monkeypatch, tmp_path
    ):
        c, H = self._client(monkeypatch, tmp_path)
        c.post("/api/portfolio/trade",
              json={"symbol": "AAPL", "side": "buy", "quantity": 10,
                    "price": 100}, headers=H)
        cash = c.get("/api/cashbook", headers=H).json()["cash"]
        port = c.get("/api/portfolio", headers=H).json()
        assert port["total_equity"] == cash + port["total_market_value"]

    def test_limit_order_roundtrip_via_api(self, monkeypatch, tmp_path):
        c, H = self._client(monkeypatch, tmp_path)
        r = c.post("/api/cashbook/limit",
                   json={"symbol": "SPY", "side": "buy", "quantity": 2,
                         "limit_price": 1.0}, headers=H).json()
        assert r["success"] is True
        oid = r["limit_orders"][0]["id"]
        c2 = c.delete(f"/api/cashbook/limit/{oid}", headers=H)
        assert c2.json()["success"] is True

    def test_cashbook_gated(self, monkeypatch, tmp_path):
        c, _ = self._client(monkeypatch, tmp_path)
        assert c.get("/api/cashbook").status_code == 401
        assert c.post("/api/cashbook/adjust", json={"amount": 1}).status_code == 401


# ---------------------------------------------------------------------------
# Position semantics: long-only "hold" must never go negative
# ---------------------------------------------------------------------------

class TestSignalToPositionSemantics:
    """Before this session, 'hold' mode let a sell signal flip strategies
    SHORT - wrong for RSI/MACD/SMA/Bollinger's buy/sell-to-flat
    semantics. Every long-only backtest was silently modeling exposure
    that shouldn't exist."""

    def _signals(self):
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=6, freq="D")
        return idx, pd.DataFrame({"signal": [1, 0, -1, 0, 1, 0]}, index=idx)

    def test_hold_mode_is_long_only(self):
        from trading.optimization.strategy_backtest_objective import (
            signals_to_position)
        idx, signals = self._signals()
        pos = signals_to_position(signals, idx, signal_mode="hold")
        assert (pos >= 0).all()
        assert list(pos) == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    def test_flip_mode_allows_short(self):
        from trading.optimization.strategy_backtest_objective import (
            signals_to_position)
        idx, signals = self._signals()
        pos = signals_to_position(signals, idx, signal_mode="flip")
        assert list(pos) == [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]

    def test_raw_mode_is_verbatim(self):
        from trading.optimization.strategy_backtest_objective import (
            signals_to_position)
        idx, signals = self._signals()
        pos = signals_to_position(signals, idx, signal_mode="raw")
        assert list(pos) == [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]

    def test_excess_vs_bh_registered_as_maximize_metric(self):
        from trading.optimization.strategy_backtest_objective import _MAXIMIZE
        assert "excess_vs_bh" in _MAXIMIZE


# ---------------------------------------------------------------------------
# Monte Carlo: percentile ordering must hold by construction
# ---------------------------------------------------------------------------

class TestMonteCarloSanity:
    def test_percentiles_ordered_and_deterministic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
        import numpy as np
        import pandas as pd

        import trading.auth.accounts as A
        monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
        A.create_user("thomas", "hunter2secure")

        rng = np.random.default_rng(11)
        idx = pd.date_range("2024-01-01", periods=400, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400)))
        hist = pd.DataFrame({"Close": close}, index=idx)

        import trading.data.price_cache as PC
        monkeypatch.setattr(PC, "get_history", lambda *a, **k: hist)

        from fastapi.testclient import TestClient

        from web.backend.main import app
        c = TestClient(app)
        tok = c.post("/api/auth/token",
                     data={"username": "thomas",
                           "password": "hunter2secure"}).json()["access_token"]
        H = {"Authorization": f"Bearer {tok}"}
        r = c.post("/api/monte-carlo",
                  json={"symbol": "SPY", "n_simulations": 200,
                        "horizon_days": 30}, headers=H).json()
        assert r["success"] is True
        assert r["final_p5"] <= r["final_p50"] <= r["final_p95"]
        assert r["n_simulations"] == 200


# ---------------------------------------------------------------------------
# Systematic sweep: every parity route, unauthenticated -> 401,
# authenticated -> never 500 even with no network (graceful offline).
# ---------------------------------------------------------------------------

PARITY_GET_ROUTES = [
    "/api/pulse",
    "/api/pulse/SPY",
    "/api/news/SPY",
    "/api/forecast/SPY",
    "/api/risk/SPY",
    "/api/strategies",
    "/api/settings/prefs",
    "/api/chart-events/SPY",
    "/api/causal/SPY",
    "/api/patterns/SPY",
    "/api/playbook/SPY",
    "/api/earnings/SPY",
    "/api/options/SPY",
    "/api/strategy-params/RSIStrategy/SPY",
    "/api/ic/SPY",
    "/api/cashbook",
    "/api/alerts",
]

PARITY_POST_ROUTES = [
    ("/api/briefing", {}),
    ("/api/pairs", {}),
    ("/api/tune-models", {"symbol": "SPY", "n_trials": 5, "models": []}),
    ("/api/gnn", {"symbols": []}),
    ("/api/monte-carlo", {"symbol": "SPY", "n_simulations": 50}),
    ("/api/optimize", {"strategy": "RSIStrategy", "symbol": "SPY",
                       "max_evaluations": 5}),
    ("/api/allocate", {"symbols": ["SPY", "QQQ"]}),
    ("/api/cashbook/adjust", {"amount": 1.0}),
    ("/api/cashbook/limit", {"symbol": "SPY", "side": "buy",
                             "quantity": 1, "limit_price": 1.0}),
    ("/api/alerts", {"symbol": "SPY", "condition": "price_above",
                     "threshold": 100.0}),
    ("/api/settings/prefs", {"scoring_style": "balanced"}),
]


@pytest.fixture()
def parity_client(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
    import trading.auth.accounts as A
    import trading.portfolio.paper_portfolio as PP
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
    A.create_user("thomas", "hunter2secure")
    from fastapi.testclient import TestClient

    from web.backend.main import app
    c = TestClient(app)
    tok = c.post("/api/auth/token",
                 data={"username": "thomas",
                       "password": "hunter2secure"}).json()["access_token"]
    return c, {"Authorization": f"Bearer {tok}"}


class TestParitySurfaceSweep:
    """Every route added in the parity-build commit: unauthenticated
    must 401 (nothing new escapes the auth gate), and authenticated
    must never 500 even fully offline (graceful degradation, matching
    the pattern established everywhere else in this codebase)."""

    @pytest.mark.parametrize("path", PARITY_GET_ROUTES)
    def test_get_route_gated_and_no_crash(self, parity_client, path):
        c, H = parity_client
        assert c.get(path).status_code == 401, f"{path} not gated"
        r = c.get(path, headers=H)
        assert r.status_code != 500, f"{path} -> 500: {r.text[:200]}"

    @pytest.mark.parametrize("path,body", PARITY_POST_ROUTES)
    def test_post_route_gated_and_no_crash(self, parity_client, path, body):
        c, H = parity_client
        assert c.post(path, json=body).status_code == 401, f"{path} not gated"
        r = c.post(path, json=body, headers=H)
        assert r.status_code != 500, f"{path} -> 500: {r.text[:200]}"

    def test_delete_alert_gated_and_no_crash(self, parity_client):
        c, H = parity_client
        assert c.delete("/api/alerts/nonexistent").status_code == 401
        r = c.delete("/api/alerts/nonexistent", headers=H)
        assert r.status_code != 500

    def test_delete_strategy_params_gated_and_no_crash(self, parity_client):
        c, H = parity_client
        assert c.delete("/api/strategy-params/RSIStrategy/SPY").status_code == 401
        r = c.delete("/api/strategy-params/RSIStrategy/SPY", headers=H)
        assert r.status_code != 500

    def test_model_backtest_gated_and_no_crash(self, parity_client):
        c, H = parity_client
        assert c.post("/api/backtest/model", json={"symbol": "SPY"}).status_code == 401
        r = c.post("/api/backtest/model", json={"symbol": "SPY"}, headers=H)
        assert r.status_code != 500
