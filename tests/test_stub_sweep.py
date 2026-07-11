# -*- coding: utf-8 -*-
"""2026-07 stub sweep: every class here was unconstructible (or a
routing default was a stub) before this pass. See commit for the
per-file archaeology."""

import numpy as np
import pandas as pd


class TestFormerlyUnconstructibleAgents:
    def test_all_four_construct_and_satisfy_contract(self):
        from trading.agents.model_evaluator_agent import ModelEvaluatorAgent
        from trading.agents.model_improver_agent import ModelImproverAgent
        from trading.agents.model_optimizer_agent import ModelOptimizerAgent
        from trading.agents.research_agent import ResearchAgent

        agents = [ResearchAgent(), ModelEvaluatorAgent(),
                  ModelOptimizerAgent(), ModelImproverAgent()]
        assert all(a.validate_config() for a in agents)
        assert all(len(a.get_capabilities()) > 0 for a in agents)

    def test_research_tool_degrades_offline(self):
        # research_agent is a LIVE chat-tool dependency
        # (trading/services/agent_tools.py) - it had never worked.
        from trading.agents.research_agent import ResearchAgent
        out = ResearchAgent().search_arxiv("momentum", max_results=1)
        assert isinstance(out, list)

    def test_legacy_optimizer_shims_construct_but_raise_on_use(self):
        from trading.agents.model_optimizer_agent import GeneticOptimizer
        g = GeneticOptimizer()  # was raise-on-construct
        try:
            g.optimize()
            raise AssertionError("shim should raise on use")
        except NotImplementedError as e:
            assert "trading/optimization" in str(e)  # points to the real cluster


class TestForecastRouterDetection:
    def _series(self, kind):
        idx = pd.date_range("2025-01-01", periods=252, freq="B")
        rng = np.random.default_rng(1)
        if kind == "trend":
            c = 100 * np.exp(0.001 * np.arange(252) + rng.normal(0, 0.002, 252))
        elif kind == "seasonal":
            c = 100 + 3 * np.sin(2 * np.pi * np.arange(252) / 5) + rng.normal(0, 0.2, 252)
        else:
            c = 100 + rng.normal(0, 0.5, 252)
        return pd.DataFrame({"Close": c}, index=idx)

    def test_trend_and_seasonality_no_longer_stubbed_false(self):
        from trading.models.forecast_router import ForecastRouter
        fr = ForecastRouter()
        assert fr._check_trend(self._series("trend"))
        assert not fr._check_trend(self._series("flat"))
        assert fr._check_seasonality(self._series("seasonal"))
        assert not fr._check_seasonality(self._series("flat"))


class TestMCPServerSurface:
    def test_ten_tools_exposed(self):
        import asyncio

        import trading.services.mcp_server as M
        tools = asyncio.run(M.mcp.list_tools())
        names = {t.name for t in tools}
        assert {"get_ai_score", "run_backtest", "scan_universe",
                "optimize_strategy_params"} <= names
        assert len(names) == 10
