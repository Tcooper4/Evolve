# -*- coding: utf-8 -*-
"""Paper portfolio: average-cost accounting hand-verified, per-user
isolation, offline degradation, tool + API surface, and the route-order
regression (a mid-file '/' static mount swallowing later routes)."""

import pathlib
import tempfile

import pytest


@pytest.fixture()
def pp(monkeypatch, tmp_path):
    import trading.portfolio.paper_portfolio as PP
    monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
    return PP


class TestAccounting:
    def test_average_cost_and_realized(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("AAPL", "buy", 10, 100)
        p.record_trade("AAPL", "buy", 10, 110)
        pos = p.get_positions()[0]
        assert pos["quantity"] == 20 and abs(pos["avg_cost"] - 105) < 1e-9
        r = p.record_trade("AAPL", "sell", 5, 120)
        assert r["realized_pnl"] == 75.0
        pos = p.get_positions()[0]
        # average-cost method: sells never change the average
        assert pos["quantity"] == 15 and abs(pos["avg_cost"] - 105) < 1e-9

    def test_oversell_rejected_close_removes_row(self, pp):
        p = pp.PaperPortfolio(user_id="user:t")
        p.record_trade("SPY", "buy", 5, 600)
        assert p.record_trade("SPY", "sell", 6, 600)["success"] is False
        p.record_trade("SPY", "sell", 5, 610)
        assert p.get_positions() == []
        assert abs(p.get_realized_pnl() - 50.0) < 1e-9  # persists after close

    def test_isolation_and_offline_summary(self, pp):
        a = pp.PaperPortfolio(user_id="user:a")
        b = pp.PaperPortfolio(user_id="user:b")
        a.record_trade("SPY", "buy", 2, 600)
        assert b.get_positions() == []
        s = a.get_summary(price_fn=lambda _: None)
        assert s["all_prices_live"] is False
        assert s["total_unrealized_pnl"] == 0.0  # neutral at cost when unpriced
        live = a.get_summary(price_fn=lambda _: 615.0)
        assert live["total_unrealized_pnl"] == 30.0


class TestSurfaces:
    def test_tools_in_registry_and_mcp(self):
        import asyncio

        import trading.services.mcp_server as M
        from agents.llm.agent import get_evolve_platform_tool_registry
        reg = {t["name"] for t in get_evolve_platform_tool_registry()}
        mcp_names = {t.name for t in asyncio.run(M.mcp.list_tools())}
        assert {"get_portfolio", "record_paper_trade"} <= reg
        assert reg <= mcp_names

    def test_api_roundtrip_and_route_order(self, pp, monkeypatch, tmp_path):
        monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
        import trading.auth.accounts as A
        monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
        A.create_user("thomas", "hunter2secure")
        from fastapi.testclient import TestClient

        from web.backend.main import app
        c = TestClient(app)
        assert c.get("/api/portfolio").status_code == 401  # the mount bug
        tok = c.post("/api/auth/token",
                     data={"username": "thomas",
                           "password": "hunter2secure"}).json()["access_token"]
        H = {"Authorization": f"Bearer {tok}"}
        c.post("/api/portfolio/trade",
               json={"symbol": "AAPL", "side": "buy", "quantity": 10,
                     "price": 100}, headers=H)
        s = c.get("/api/portfolio", headers=H).json()
        assert s["positions"][0]["avg_cost"] == 100.0
