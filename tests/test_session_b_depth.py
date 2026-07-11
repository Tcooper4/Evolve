# -*- coding: utf-8 -*-
"""Session B depth pass: agent.py internals, the three critic agents,
and the agent_manager retry loop. Every test encodes a bug found and
fixed by execution in 2026-07."""

import asyncio

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# agent.py internals
# ---------------------------------------------------------------------------

class TestPromptRouting:
    @pytest.fixture()
    def agent(self):
        from agents.llm.agent import PromptAgent
        return PromptAgent()

    def test_intent_priority_and_word_boundaries(self, agent):
        # Bugs: substring matching ('test' inside 'latest'), and generic
        # intents shadowing specific ones (backtest before critique).
        cases = [
            ("optimize my rsi strategy parameters", "optimize"),
            ("critique my last backtest", "critique_backtest"),
            ("what's the latest news on nvda", "general"),
            ("run a backtest of bollinger bands on TSLA", "backtest"),
        ]
        for prompt, want in cases:
            intent, _ = agent._parse_prompt_regex_fallback(prompt)
            assert intent == want, (prompt, intent)

    def test_domain_words_not_tickers(self, agent):
        # Bug: 'rsi strategy' -> ticker RSI, 'bollinger bands' -> BANDS.
        _, p = agent._parse_prompt_regex_fallback(
            "run a backtest of bollinger bands on TSLA")
        assert p["symbol"] == "TSLA"
        assert agent._resolve_symbol_from_prompt("optimize my rsi strategy") is None

    def test_company_map_beats_domain_stoplist(self, agent):
        assert agent._resolve_symbol_from_prompt(
            "forecast the price of apple") == "AAPL"


class TestFewShotCluster:
    @pytest.fixture()
    def agent(self):
        from agents.llm.agent import PromptAgent
        a = PromptAgent()

        class FakeEnc:
            def encode(self, texts):
                rng = np.random.default_rng(abs(hash(texts[0])) % 2**32)
                return rng.normal(size=(len(texts), 8))

        a.sentence_transformer = FakeEnc()
        a.prompt_examples = {"examples": [
            {"prompt": "forecast TSLA", "parsed_output": {"intent": "forecast"}},
            {"prompt": "malformed row"},  # missing parsed_output
            {"prompt": "backtest SPY", "parsed_output": {"intent": "backtest"}},
        ]}
        a.example_embeddings = FakeEnc().encode(["a", "b", "c"]) * np.array(
            [[3.0], [1.0], [0.5]])
        return a

    def test_numpy_truthiness_no_crash(self, agent):
        # Bug: `not self.example_embeddings` on ndarray raised ValueError
        # the moment an encoder was actually available.
        out = agent._find_similar_examples("forecast AAPL", top_k=3)
        assert isinstance(out, list)

    def test_malformed_example_skipped_not_fatal(self, agent):
        # Bug: one bad stored row raised KeyError inside the try and
        # silently disabled ALL retrieval.
        out = agent._find_similar_examples("forecast AAPL", top_k=3)
        assert len(out) == 2
        assert all("parsed_output" in o for o in out)

    def test_similarity_is_true_cosine(self, agent):
        out = agent._find_similar_examples("forecast AAPL", top_k=3)
        assert all(-1.001 <= o["similarity_score"] <= 1.001 for o in out)


# ---------------------------------------------------------------------------
# Critic agents
# ---------------------------------------------------------------------------

class TestCriticConstruction:
    def test_all_three_instantiable(self):
        # Bug class: BaseAgent grew abstract methods after these were
        # written; two of three critics raised TypeError at construction.
        from trading.agents.data_quality_agent import DataQualityAgent
        from trading.agents.execution_risk_agent import ExecutionRiskAgent
        from trading.agents.performance_critic_agent import PerformanceCriticAgent
        assert DataQualityAgent() and ExecutionRiskAgent() and PerformanceCriticAgent()


class TestDataQualityDetection:
    def test_planted_anomalies_detected_on_titlecase_data(self):
        # Bug: detectors read lowercase column names; every check
        # no-opped on the platform's Title-case dataframes forever.
        from trading.agents.data_quality_agent import DataQualityAgent
        dq = DataQualityAgent()
        rng = np.random.default_rng(5)
        N = 200
        idx = pd.date_range("2025-01-01", periods=N, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, N)))
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.004,
            "Low": close * 0.996, "Close": close.copy(),
            "Volume": rng.integers(1e6, 2e6, N).astype(float),
        }, index=idx)
        df.iloc[100, df.columns.get_loc("Close")] = close[100] * 1.5
        df.iloc[150, df.columns.get_loc("Volume")] = 5e8

        res = asyncio.run(dq.execute(action="assess_quality", data=df,
                                     symbol="TEST"))
        assert res.success
        assert (res.data or {}).get("anomalies_count", 0) >= 2

        clean = df.copy()
        clean.iloc[100, clean.columns.get_loc("Close")] = close[100]
        clean.iloc[150, clean.columns.get_loc("Volume")] = 1.5e6
        res2 = asyncio.run(dq.execute(action="assess_quality", data=clean,
                                      symbol="CLEAN"))
        assert res2.data["overall_score"] > res.data["overall_score"]


class TestExecutionRisk:
    def test_scenario_battery(self):
        from trading.agents.execution_risk_agent import ExecutionRiskAgent
        e = ExecutionRiskAgent()
        ctx = {"portfolio_value": 100_000}
        assert e.approve_trade("t1", "SPY", 0.05, "buy", 618.0,
                               ctx).approved_size > 0
        big = e.approve_trade("t2", "SPY", 0.50, "buy", 618.0, ctx)
        assert big.status.value == "rejected" and big.approved_size == 0.0
        e.portfolio_state["current_drawdown"] = 0.20
        assert e.approve_trade("t3", "AAPL", 0.05, "buy", 210.0,
                               ctx).status.value == "rejected"


class TestPerformanceCriticMath:
    def test_hand_verified_metrics(self):
        from trading.agents.performance_critic_agent import PerformanceCriticAgent
        p = PerformanceCriticAgent()
        N = 253
        idx = pd.date_range("2025-01-01", periods=N, freq="B")
        close = pd.Series(100 * (1.001 ** np.arange(N)), index=idx)
        td = pd.DataFrame({"close": close, "volume": 1e6}, index=idx)
        pm = p._calculate_performance_metrics(close * 1.0, td)
        assert abs(pm["total_return"] - ((1.001 ** 252) - 1)) < 1e-9
        # IR: zero for perfect tracking (bug: used std of ACTUAL returns)
        assert pm["information_ratio"] == 0.0
        rm = p._calculate_risk_metrics(close * 1.0, td)  # bug: config.get crash
        assert abs(rm["max_drawdown"]) < 1e-9


# ---------------------------------------------------------------------------
# Agent manager retry loop
# ---------------------------------------------------------------------------

class TestAgentManagerLoop:
    def test_constructs_despite_broken_agent_chain(self):
        # Bugs: hard redis import + module-level agent imports made the
        # manager unimportable/unconstructible.
        from trading.agents.agent_manager import EnhancedAgentManager
        m = EnhancedAgentManager()
        assert "performance_critic" in m.agent_registry

    def test_backoff_capped(self):
        from trading.agents.agent_manager import EnhancedAgentManager, RetryConfig
        m = EnhancedAgentManager()
        rc = RetryConfig(max_retries=5, base_delay=1.0, backoff_factor=2.0,
                         exponential_backoff=True, jitter=False, max_delay=5.0)
        assert [m._calculate_backoff_delay(a, rc) for a in (1, 2, 3, 4, 5)] == \
            [1.0, 2.0, 4.0, 5.0, 5.0]

    def test_success_is_success(self):
        # THE bug: bookkeeping read result.sharpe_ratio (nonexistent),
        # converting every successful run into a failure + retries.
        from trading.agents.agent_manager import EnhancedAgentManager, RetryConfig
        from trading.agents.base_agent_interface import AgentConfig, AgentResult
        m = EnhancedAgentManager()
        calls = {"n": 0}

        class Flaky:
            def __init__(self, *a, **k):
                self.config = AgentConfig(name="flaky", enabled=True)

            async def execute(self, **kw):
                calls["n"] += 1
                if calls["n"] < 3:
                    return AgentResult(success=False, error_message="transient")
                return AgentResult(success=True, data={"answer": 42})

        m.register_agent("flaky", Flaky)
        res = asyncio.run(m.execute_agent_with_retry(
            "flaky",
            retry_config=RetryConfig(max_retries=5, base_delay=0.01,
                                     jitter=False)))
        assert res.success and calls["n"] == 3
        assert res.data == {"answer": 42}
        assert m.agent_metrics["flaky"]["successful_executions"] == 1
