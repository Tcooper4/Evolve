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
    "/api/strategy-overlay/SPY?strategy=RSIStrategy&period=6mo",
    "/api/causal/SPY",
    "/api/patterns/SPY",
    "/api/playbook/SPY",
    "/api/earnings/SPY",
    "/api/options/SPY",
    "/api/options/gex/SPY",
    "/api/options/skew/SPY",
    "/api/options/context/SPY",
    "/api/strategy-params/RSIStrategy/SPY",
    "/api/ic/SPY",
    "/api/cashbook",
    "/api/alerts",
    "/api/portfolio/risk",
    "/api/recs",
    "/api/edgar/SPY",
    "/api/market-signals",
    "/api/news/breaking",
]

PARITY_POST_ROUTES = [
    ("/api/briefing", {}),
    ("/api/pairs", {}),
    ("/api/tune-models", {"symbol": "SPY", "n_trials": 5, "models": []}),
    ("/api/gnn", {"symbols": []}),
    ("/api/monte-carlo", {"symbol": "SPY", "n_simulations": 50}),
    ("/api/backtest/options-structure", {
        "symbol": "SPY", "strategy": "iron_condor", "period": "1y",
        "sweep": False,
    }),
    ("/api/optimize", {"strategy": "RSIStrategy", "symbol": "SPY",
                       "max_evaluations": 5}),
    ("/api/allocate", {"symbols": ["SPY", "QQQ"]}),
    ("/api/cashbook/adjust", {"amount": 1.0}),
    ("/api/cashbook/limit", {"symbol": "SPY", "side": "buy",
                             "quantity": 1, "limit_price": 1.0}),
    ("/api/alerts", {"symbol": "SPY", "condition": "price_above",
                     "threshold": 100.0}),
    ("/api/settings/prefs", {"scoring_style": "balanced"}),
    ("/api/recs", {"symbol": "SPY", "score": 7.0, "price_at_rec": 100.0,
                   "capture_guidance": False}),
    ("/api/recs/outcome", {
        "symbol": "SPY", "real_pnl": 10.0, "real_strategy": "shares",
    }),
    ("/api/market-signals/gpr", {}),
    ("/api/news/context", {"titles": ["Markets steady ahead of data"]}),
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


class TestStructuredPatternData:
    """get_pattern_analysis used to discard the detector's per-pattern
    start_date/end_date/type/confidence down to a text-only summary,
    making it impossible to plot patterns on a chart. Structured data
    is now exposed alongside the summary."""

    def test_patterns_field_has_dated_structured_entries(self, monkeypatch):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(4)
        N = 130
        idx = pd.date_range("2025-01-01", periods=N, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, N)))
        hist = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.01,
            "Low": close * 0.99, "Close": close,
            "Volume": rng.integers(1e6, 3e6, N)}, index=idx)

        class FakeTicker:
            def __init__(self, *a, **k):
                pass

            def history(self, **k):
                return hist

        monkeypatch.setattr("yfinance.Ticker", FakeTicker)
        from trading.services.agent_tools import get_pattern_analysis
        r = get_pattern_analysis("SPY")
        assert r["success"] is True
        assert "patterns" in r and isinstance(r["patterns"], list)
        if r["patterns"]:
            p0 = r["patterns"][0]
            assert {"name", "type", "confidence", "start_date",
                   "end_date", "description"} <= set(p0)
            assert p0["start_date"] is not None  # chartable, not just prose


class TestRecommendationTracker:
    """Tracked ideas: save without buying, capture rec-time price, compute
    honest performance-since; one live rec per symbol; per-user."""

    def test_lifecycle_and_performance_math(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        r = p.track_recommendation(
            "NVDA", score=8.2, price_at_rec=100.0, capture_guidance=False,
        )
        assert r["success"]
        recs = p.get_recommendations(price_fn=lambda _: 112.0)
        assert recs[0]["change_pct"] == 12.0
        assert recs[0]["score"] == 8.2
        # re-track replaces (one live rec per symbol)
        p.track_recommendation(
            "NVDA", score=6.0, price_at_rec=112.0, capture_guidance=False,
        )
        assert len(p.get_recommendations(price_fn=lambda _: None)) == 1
        rid = p.get_recommendations(price_fn=lambda _: None)[0]["id"]
        assert p.delete_recommendation(rid)["success"]

    def test_offline_prices_none_safe(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.track_recommendation(
            "SPY", price_at_rec=500.0, capture_guidance=False,
        )
        recs = p.get_recommendations(price_fn=lambda _: None)
        assert recs[0]["last_price"] is None and recs[0]["change_pct"] is None

    def test_paper_buy_marks_acted_and_full_exit_closes(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.track_recommendation(
            "NVDA", score=8.0, price_at_rec=100.0, capture_guidance=False,
        )
        buy = p.record_trade("NVDA", "buy", 5, 105.0)
        assert buy["success"]
        assert buy.get("recommendation", {}).get("status") == "acted"
        recs = p.get_recommendations(price_fn=lambda _: 110.0)
        openish = [r for r in recs if r["symbol"] == "NVDA" and r["status"] == "acted"]
        assert len(openish) == 1
        assert openish[0]["acted_price"] == 105.0
        sell = p.record_trade("NVDA", "sell", 5, 120.0)
        assert sell["success"]
        assert sell.get("recommendation", {}).get("status") == "closed"
        closed = [r for r in p.get_recommendations(price_fn=lambda _: None)
                  if r["symbol"] == "NVDA" and r["status"] == "closed"]
        assert len(closed) == 1
        assert closed[0]["closed_price"] == 120.0
        assert closed[0]["change_pct"] == 20.0  # 100 → 120

    def test_isolated_per_user(self, pp):
        pp.PaperPortfolio(user_id="user:a").track_recommendation(
            "SPY", capture_guidance=False,
        )
        assert pp.PaperPortfolio(user_id="user:b").get_recommendations(
            price_fn=lambda _: None) == []

    def test_real_outcome_win_loss_and_match_summary(self, pp):
        """Hand-check: match vs mismatch buckets + Kelly small-sample caveat."""
        from trading.portfolio.kelly_sample_disclosure import SMALL_SAMPLE_N

        p = pp.PaperPortfolio(user_id="user:t")
        # 2 matched wins, 1 matched loss; 1 mismatched win, 1 mismatched loss
        for sym, sug, used, pnl in (
            ("AAPL", "iron_condor", "iron_condor", 140.0),
            ("MSFT", "iron_condor", "IC", 80.0),
            ("AMD", "iron_condor", "iron_condor", -200.0),
            ("TSLA", "put_credit_spread", "shares", 50.0),
            ("META", "call_credit_spread", "iron_condor", -30.0),
        ):
            r = p.track_recommendation(
                sym,
                price_at_rec=100.0,
                capture_guidance=False,
                structure_suggestion=sug,
            )
            out = p.record_real_outcome(
                r["id"], real_pnl=pnl, real_strategy=used, real_acted=True,
            )
            assert out["success"] is True
            assert out["won"] is (pnl > 0)

        recs = p.get_recommendations(price_fn=lambda _: None)
        aapl = next(x for x in recs if x["symbol"] == "AAPL")
        assert aapl["real_pnl"] == 140.0
        assert aapl["real_won"] is True

        summary = p.summarize_real_outcomes(recs)
        assert summary["n_with_outcome"] == 5
        # matched: AAPL, MSFT (IC→iron_condor), AMD → 2 wins / 3
        assert summary["matched_structure"]["n"] == 3
        assert summary["matched_structure"]["wins"] == 2
        assert summary["matched_structure"]["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert summary["matched_structure"]["avg_pnl"] == pytest.approx(
            (140 + 80 - 200) / 3, abs=0.02
        )
        # mismatched: TSLA, META → 1 win / 2
        assert summary["mismatched_structure"]["n"] == 2
        assert summary["mismatched_structure"]["win_rate"] == pytest.approx(0.5)
        # Under Kelly threshold → same caveat machinery fires
        assert summary["n_with_outcome"] < SMALL_SAMPLE_N
        assert summary["sample_size_flag"] in ("insufficient", "provisional")
        assert summary["sample_size_caveat"]
        assert str(SMALL_SAMPLE_N) in summary["sample_size_caveat"]

    def test_record_real_outcome_by_symbol(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.track_recommendation(
            "AAPL",
            price_at_rec=190.0,
            capture_guidance=False,
            structure_suggestion="iron_condor",
        )
        out = p.record_real_outcome(
            None, symbol="AAPL", real_pnl=140.0, real_strategy="iron_condor",
        )
        assert out["success"] is True
        assert out["won"] is True


class TestTradeStatsAndAccountRisk:
    """Kelly inputs come from the user's OWN closed paper trades - not
    generic numbers. Hand math: 2 wins avg $100, 1 loss $50 ->
    win rate 2/3, ratio 2.0 -> full Kelly 0.5, half 0.25."""

    def test_trade_stats_hand_math(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        for sym, sell in (("A", 20), ("B", 20), ("C", 5)):
            p.record_trade(sym, "buy", 10, 10)
            p.record_trade(sym, "sell", 10, sell)
        st = p.get_trade_stats()
        assert st["closed_trades"] == 3
        assert abs(st["win_rate"] - 2 / 3) < 1e-3
        assert abs(st["avg_win_loss_ratio"] - 2.0) < 1e-9

    def test_one_sided_history_not_usable(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("A", "buy", 10, 10)
        p.record_trade("A", "sell", 10, 20)  # only wins
        st = p.get_trade_stats()
        assert st["win_rate"] == 1.0
        assert st["avg_win_loss_ratio"] is None  # can't size from wins alone

    def test_account_risk_endpoint_kelly_from_ledger(self, parity_client,
                                                     monkeypatch, tmp_path):
        c, H = parity_client
        for sym, sell in (("A", 20), ("B", 20), ("C", 5)):
            c.post("/api/portfolio/trade",
                   json={"symbol": sym, "side": "buy", "quantity": 10,
                         "price": 10}, headers=H)
            c.post("/api/portfolio/trade",
                   json={"symbol": sym, "side": "sell", "quantity": 10,
                         "price": sell}, headers=H)
        r = c.get("/api/portfolio/risk", headers=H).json()
        assert r["success"] and r["trade_stats"]["closed_trades"] == 3
        assert abs(r["kelly"]["half_kelly_fraction"] - 0.25) < 1e-3

    def test_account_risk_open_position_offline_graceful(self, parity_client):
        c, H = parity_client
        c.post("/api/portfolio/trade",
               json={"symbol": "AAPL", "side": "buy", "quantity": 5,
                     "price": 100}, headers=H)
        r = c.get("/api/portfolio/risk", headers=H)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] and body["positions"] == 1
        # offline: metrics may be None but never a crash or fake numbers


class TestWalkForwardFolds:
    """Per-window results were computed then aggregated away; the route
    now exposes each fold so the UI can show stability across time."""

    def test_folds_in_model_backtest_response(self, parity_client, monkeypatch):
        import numpy as np
        import pandas as pd

        c, H = parity_client

        rng = np.random.default_rng(3)
        idx = pd.date_range("2024-01-01", periods=300, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 300)))
        hist = pd.DataFrame({"Close": close}, index=idx)

        class FakeTicker:
            def __init__(self, *a, **k):
                pass

            def history(self, **k):
                return hist

        monkeypatch.setattr("yfinance.Ticker", FakeTicker)

        class FakeWindow:
            def __init__(self, i):
                self.window_index = i
                self.train_start = idx[0]
                self.train_end = idx[100 + i]
                self.test_start = idx[101 + i]
                self.test_end = idx[120 + i]
                self.predictions = [1.0, 2.0, 3.0]
                self.actuals = [1.0, 2.1, 2.9]
                self.mae = 0.1
                self.mse = 0.02
                self.mape = 3.2
                self.directional_accuracy = 0.62

        class FakeResult:
            windows = [FakeWindow(0), FakeWindow(1)]
            model_performance = {"n_windows": 2, "mean_mae": 0.1,
                                 "mean_mape": 3.2,
                                 "mean_directional_accuracy": 0.62}

        class FakeValidator:
            def __init__(self, *a, **k):
                pass

            def run(self, *a, **k):
                return FakeResult()

        import trading.validation.walk_forward_utils as WF
        monkeypatch.setattr(WF, "WalkForwardValidator", FakeValidator)

        r = c.post("/api/backtest/model",
                   json={"symbol": "SPY", "model": "xgboost",
                         "period": "1y"}, headers=H).json()
        assert r["success"] is True
        assert len(r["folds"]) == 2
        f0 = r["folds"][0]
        assert {"window", "train_start", "test_end", "mae", "mape",
               "directional_accuracy"} <= set(f0)
        assert f0["directional_accuracy"] == 0.62


class TestBreakingNewsToolParity:
    """/api/news/breaking shipped UI-only - a real gap against the
    'guided analyst can do everything the codebase can do' principle.
    Now a chat/MCP tool too, degrading gracefully offline."""

    def test_registered_and_parity_holds(self):
        import asyncio

        import trading.services.mcp_server as M
        from agents.llm.agent import get_evolve_platform_tool_registry
        reg = {t["name"] for t in get_evolve_platform_tool_registry()}
        mcp_names = {t.name for t in asyncio.run(M.mcp.list_tools())}
        assert "get_breaking_news" in reg
        assert reg <= mcp_names

    def test_degrades_offline(self):
        from trading.services.agent_tools import get_breaking_news
        r = get_breaking_news()
        assert r["success"] in (True, False)
        assert isinstance(r["items"], list)


class TestMarketSignals:
    """GPR + EPS revision breadth: Dashboard pulse reads prefs/disk;
    Settings POSTs refresh and persist. Never invents a number."""

    def test_get_shape_and_offline_safe(self, parity_client):
        c, H = parity_client
        r = c.get("/api/market-signals", headers=H)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "gpr" in body and "revision_breadth" in body
        # None is honest when not loaded / offline
        assert body["gpr"] is None or (
            isinstance(body["gpr"], dict) and body["gpr"].get("current") is not None
        )

    def test_gpr_post_persists_to_prefs(self, parity_client, monkeypatch):
        c, H = parity_client
        fake = {
            "current": 142.0, "level": "ELEVATED", "trend": "RISING",
            "percentile": 80.0, "description": "test", "source": "test",
        }

        class FakeMF:
            def _get_gpr_index(self):
                return fake

        import trading.analysis.macro_factors as MF
        monkeypatch.setattr(MF, "MacroFactors", FakeMF)
        r = c.post("/api/market-signals/gpr", headers=H).json()
        assert r["success"] is True
        assert r["gpr"]["current"] == 142.0
        got = c.get("/api/market-signals", headers=H).json()
        assert got["gpr"]["current"] == 142.0
        assert got["gpr"]["level"] == "ELEVATED"


class TestNewsContextHedge:
    def test_sanitize_strips_advice(self):
        from trading.services.news_context import _sanitize_why
        assert _sanitize_why("Buy NVDA now") == ""
        assert _sanitize_why("This may reflect rate-cut expectations.") != ""

    def test_endpoint_offline_safe(self, parity_client, monkeypatch):
        c, H = parity_client
        import trading.services.news_context as NC
        monkeypatch.setattr(NC, "_complete", lambda *a, **k: "May reflect softer inflation data.")
        r = c.post("/api/news/context",
                   json={"titles": ["CPI cools more than expected"]},
                   headers=H).json()
        assert r["success"] is True
        assert r["items"][0]["why"]
        assert r["items"][0]["hedged"] is True

        monkeypatch.setattr(NC, "_complete", lambda *a, **k: "You should buy the dip")
        r2 = c.post("/api/news/context",
                    json={"titles": ["Stocks rally"]},
                    headers=H).json()
        assert r2["items"][0]["why"] == ""
