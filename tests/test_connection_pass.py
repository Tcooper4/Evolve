# -*- coding: utf-8 -*-
"""2026-07 connection pass: one tool registry as the single source of
truth for chat (both frontends) and MCP; the self-tuning loop's
champion/challenger discipline."""

import json

import numpy as np
import pandas as pd
import pytest


class TestUnifiedToolSurface:
    def test_chat_tools_are_registry_derived(self):
        from agents.llm.agent import get_evolve_platform_tool_registry
        from trading.services.chat_turn import STANDARD_TOOLS
        reg_names = {t["name"] for t in get_evolve_platform_tool_registry()}
        assert set(STANDARD_TOOLS) == reg_names

    def test_advisor_skill_pipeline_tools_all_callable(self):
        # the beginner-advisor playbook mandates these; the wiring gap
        # where chat couldn't call detect_market_regime is what this
        # test prevents from returning
        from agents.llm.agent import get_evolve_platform_tool_registry
        reg = {t["name"]: t for t in get_evolve_platform_tool_registry()}
        for name in ("detect_market_regime", "scan_universe", "get_ai_score",
                     "get_news", "get_risk_metrics"):
            assert name in reg and callable(reg[name]["function"]), name

    def test_new_tools_degrade_gracefully_offline(self):
        from trading.services import agent_tools as T
        assert T.get_watchlist()["success"]
        assert T.get_leaderboard()["success"]
        r = T.get_portfolio_allocation("SPY")  # one symbol -> honest error
        assert r["success"] is False and "two symbols" in r["error"]


class TestSelfTuneDiscipline:
    @pytest.fixture()
    def df(self):
        rng = np.random.default_rng(7)
        N = 504
        idx = pd.date_range("2024-07-01", periods=N, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, N)))
        return pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.005,
            "Low": close * 0.995, "Close": close,
            "Volume": rng.integers(1e6, 3e6, N).astype(float),
        }, index=idx)

    def test_champion_challenger_cycle(self, df, tmp_path, monkeypatch):
        import trading.services.self_tune as ST
        monkeypatch.setattr(ST, "STORE_PATH", tmp_path / "st.json")
        r = ST.tune_one("RSIStrategy", "TEST", df=df, max_evaluations=20)
        assert r["success"]
        assert isinstance(r["adopted"], bool)
        # adoption implies persisted params; non-adoption implies champion holds
        p = ST.get_adopted_params("RSIStrategy", "TEST")
        assert (p is not None) == r["adopted"]
        # journaled either way
        j = json.loads((tmp_path / "st.json").read_text())
        assert len(j["journal"]) == 1
        assert j["journal"][0]["challenger_oos"] is not None

    def test_adoption_requires_oos_margin(self, df, tmp_path, monkeypatch):
        # the gate that stops self-tuning from becoming an overfitting
        # machine: identical challenger cannot displace the champion
        import trading.services.self_tune as ST
        monkeypatch.setattr(ST, "STORE_PATH", tmp_path / "st.json")
        first = ST.tune_one("RSIStrategy", "TEST", df=df, max_evaluations=20)
        second = ST.tune_one("RSIStrategy", "TEST", df=df, max_evaluations=20)
        if first["adopted"] and second["challenger_oos"] is not None:
            # same data, same search: challenger ~= champion -> margin
            # gate must block churn
            assert second["adopted"] is False or (
                second["challenger_oos"]
                > (second["champion_oos"] or 0) + 0.05
            )
