# -*- coding: utf-8 -*-
"""Agents-tier depth tests: the market-regime rules classifier (replacing
a RandomForest trained on synthetic noise) and the model selector's regime
detection (TRENDING_DOWN was unreachable, the trend threshold was on a
70%-per-DAY scale so directional regimes never fired, and abs(autocorr)
classified trending series as mean-reverting)."""

import numpy as np
import pandas as pd
import pytest


class TestRegimeRulesClassifier:
    @pytest.fixture()
    def agent(self):
        from trading.agents.market_regime_agent import MarketRegimeAgent
        return MarketRegimeAgent()

    def _m(self, vol, trend, mom):
        from trading.agents.market_regime_agent import RegimeMetrics
        return RegimeMetrics(volatility=vol, trend_strength=trend,
                             momentum=mom, volume_trend=0.0,
                             correlation=0.5, regime_confidence=0.5)

    @pytest.mark.parametrize("vol,trend,mom,expected", [
        (0.45, 0.05, 0.01, "volatile"),
        (0.15, 0.25, 0.02, "bull"),
        (0.20, -0.30, -0.03, "bear"),
        (0.18, 0.20, -0.01, "trending"),   # strong trend, momentum disagrees
        (0.12, 0.02, 0.00, "sideways"),
    ])
    def test_all_regimes_reachable(self, agent, vol, trend, mom, expected):
        regime, conf = agent.classify_regime(self._m(vol, trend, mom))
        assert regime.value == expected
        assert 0.5 <= conf <= 0.95

    def test_strategy_weights_normalize(self, agent):
        from trading.agents.market_regime_agent import MarketRegime
        strats = agent.get_recommended_strategies(MarketRegime.BULL, 0.9)
        if strats:
            assert sum(s["weight"] for s in strats) == pytest.approx(1.0)


class TestSelectorRegimeDetection:
    @pytest.fixture()
    def stub(self):
        import trading.agents.model_selector_agent as M

        class Stub:
            _calculate_trend_strength = (
                M.ModelSelectorAgent._calculate_trend_strength)
            _calculate_mean_reversion_strength = (
                M.ModelSelectorAgent._calculate_mean_reversion_strength)
            detect_market_regime = M.ModelSelectorAgent.detect_market_regime
        return Stub(), M.MarketRegime

    def _df(self, close):
        n = len(close)
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({"close": close, "volume": np.full(n, 1e6)},
                            index=idx)

    def test_downtrend_is_reachable(self, stub):
        a, MR = stub
        n = 200
        rng = np.random.default_rng(1)
        down = 100 * np.exp(np.cumsum(np.full(n, -0.0012)
                                      + rng.normal(0, 0.004, n)))
        assert a.detect_market_regime(self._df(down)) == MR.TRENDING_DOWN

    def test_uptrend_fires(self, stub):
        a, MR = stub
        n = 200
        rng = np.random.default_rng(1)
        up = 100 * np.exp(np.cumsum(np.full(n, 0.0012)
                                    + rng.normal(0, 0.004, n)))
        assert a.detect_market_regime(self._df(up)) == MR.TRENDING_UP

    def test_mean_reversion_needs_negative_autocorr(self, stub):
        a, MR = stub
        n = 200
        rng = np.random.default_rng(1)
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = -0.4 * r[i - 1] + rng.normal(0, 0.006)
        r = r - r.mean()
        assert a.detect_market_regime(
            self._df(100 * np.exp(np.cumsum(r)))) == MR.MEAN_REVERTING
        # Positive autocorr (momentum) must NOT read as mean reversion -
        # the old abs() implementation got this backwards.
        r2 = np.zeros(n)
        for i in range(1, n):
            r2[i] = +0.4 * r2[i - 1] + rng.normal(0, 0.006)
        r2 = r2 - r2.mean()
        got = a.detect_market_regime(self._df(100 * np.exp(np.cumsum(r2))))
        assert got != MR.MEAN_REVERTING

    def test_high_vol_dominates(self, stub):
        a, MR = stub
        n = 200
        rng = np.random.default_rng(2)
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
        assert a.detect_market_regime(self._df(wild)) == MR.VOLATILE


class TestPromptAgentDecisionPaths:
    """Targeted execution checks on agents/llm/agent.py's live extractors
    (full line-by-line of the 2,901-line module remains a Session B item)."""

    @pytest.fixture()
    def agent(self):
        from agents.llm.agent import PromptAgent
        return PromptAgent()

    def test_symbol_extraction_no_function_words(self, agent):
        out = agent._extract_symbols_from_prompt(
            "forecast AAPL and compare to MSFT please")
        assert sorted(out) == ["AAPL", "MSFT"]

    def test_lowercase_ticker_recall_kept(self, agent):
        assert agent._extract_symbols_from_prompt(
            "forecast aapl this week") == ["AAPL"]

    def test_sanitize_contract(self, agent):
        san = agent.sanitize_prompt(
            "hello <script>alert(1)</script> " + "x" * 5000)
        assert len(san) <= 4000
        assert "<script>" not in san
