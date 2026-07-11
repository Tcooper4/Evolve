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


class TestFullSurfaceParity:
    """The gap-audit findings, locked in structurally so the class of
    bug (hand-maintained capability subsets drifting) cannot recur."""

    def test_registry_is_subset_of_mcp(self):
        import asyncio

        import trading.services.mcp_server as M
        from agents.llm.agent import get_evolve_platform_tool_registry
        reg = {t["name"] for t in get_evolve_platform_tool_registry()}
        mcp_names = {t.name for t in asyncio.run(M.mcp.list_tools())}
        assert reg <= mcp_names, sorted(reg - mcp_names)

    def test_streamlit_chat_is_registry_derived(self):
        # the page must not carry its own hardcoded tool list
        src = open("pages/6_Chat.py").read()
        assert "available_tools=_standard_tools()" in src
        assert 'available_tools=[' not in src

    def test_kelly_tool_exists_and_is_correct(self):
        # the position-sizing skill references "Evolve's Kelly tool";
        # it now exists. Hand check: p=0.6, b=1.5 -> f = 0.6-0.4/1.5
        from trading.services.agent_tools import get_position_size
        r = get_position_size(0.6, 1.5, 10_000)
        assert abs(r["full_kelly_fraction"] - (0.6 - 0.4 / 1.5)) < 1e-4
        # dollars are computed from the UNROUNDED fraction
        expected = (0.6 - 0.4 / 1.5) / 2 * 10_000
        assert abs(r["half_kelly_dollars"] - expected) < 0.01
        assert get_position_size(0.4, 1.0)["full_kelly_fraction"] == 0.0

    def test_run_backtest_consults_adopted_params(self):
        # the learning loop must feed back: run_backtest passes adopted
        # champion params into strategy execution
        src = open("trading/services/agent_tools.py").read()
        assert "get_adopted_params" in src
        assert "parameters=_adopted" in src


class TestDeflatedSharpe:
    """Selection-bias correction (Bailey & Lopez de Prado 2014): the
    optimizer must report the probability its best Sharpe survives the
    number of trials it ran."""

    def test_hand_verified_math(self):
        from trading.optimization.deflated_sharpe import (
            _phi_inv, expected_max_sharpe, probabilistic_sharpe)
        assert abs(_phi_inv(0.975) - 1.959964) < 1e-4
        # more trials -> higher null benchmark (harder to impress)
        assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(300, 1.0)
        # clearly-above-benchmark SR over a long track -> near-certain
        assert probabilistic_sharpe(0.25, 0.14, n_obs=504) > 0.95
        # at-benchmark SR -> coin flip
        assert abs(probabilistic_sharpe(0.14, 0.14, n_obs=504) - 0.5) < 1e-9

    def test_attached_to_validated_runs_and_honest_on_noise(self):
        import numpy as np
        import pandas as pd

        import trading.strategies  # noqa: F401
        from trading.optimization.strategy_backtest_objective import (
            optimize_strategy_validated)
        rng = np.random.default_rng(7)
        N = 504
        idx = pd.date_range("2024-07-01", periods=N, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, N)))
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.005,
            "Low": close * 0.995, "Close": close,
            "Volume": rng.integers(1e6, 3e6, N).astype(float)}, index=idx)
        run = optimize_strategy_validated(
            "RSIStrategy", df, train_fraction=0.75, method="grid_search",
            metric="sharpe_ratio", max_evaluations=25)
        d = run.deflated_sharpe
        assert d is not None and d["n_trials"] >= 10
        # on a random walk the best pick must NOT be certified as real
        assert d["deflated_sharpe"] < 0.95
        # sentinel filter: null benchmark must be a sane per-day Sharpe
        assert abs(d["expected_max_sharpe_under_null"]) < 2.0
