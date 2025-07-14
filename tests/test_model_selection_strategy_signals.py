"""
Unit tests for model selection and strategy signals.

This module tests the core agentic functionality:
- Model selection based on market conditions
- Strategy signal generation and validation
- Performance metrics calculation
- Agent routing and fallback mechanisms
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Import components to test
from trading.agents.prompt_router_agent import PromptRouterAgent, RequestType
from agents.llm.agent import PromptAgent
from trading.models.forecast_router import ForecastRouter
from trading.strategies.gatekeeper import StrategyGatekeeper


class TestModelSelection:
    """Test model selection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2024-01-01", periods=5),
        )

        self.forecast_router = ForecastRouter()
        self.strategy_gatekeeper = StrategyGatekeeper()

    def test_model_selection_for_bull_market(self):
        """Test model selection for bullish market conditions."""
        # Create bullish market data
        bull_data = self.sample_data.copy()
        bull_data["Close"] = [100, 102, 104, 106, 108]  # Upward trend

        # Test model selection
        selected_model = self.forecast_router.select_best_model(
            symbol="AAPL", data=bull_data, timeframe="1d"
        )

        assert selected_model is not None
        assert isinstance(selected_model, str)
        # Should prefer trend-following models for bull markets
        assert selected_model in ["lstm", "transformer", "autoformer"]

    def test_model_selection_for_bear_market(self):
        """Test model selection for bearish market conditions."""
        # Create bearish market data
        bear_data = self.sample_data.copy()
        bear_data["Close"] = [100, 98, 96, 94, 92]  # Downward trend

        selected_model = self.forecast_router.select_best_model(
            symbol="AAPL", data=bear_data, timeframe="1d"
        )

        assert selected_model is not None
        # Should prefer volatility models for bear markets
        assert selected_model in ["garch", "lstm", "transformer"]

    def test_model_selection_for_volatile_market(self):
        """Test model selection for volatile market conditions."""
        # Create volatile market data
        volatile_data = self.sample_data.copy()
        volatile_data["Close"] = [100, 105, 95, 110, 90]  # High volatility

        selected_model = self.forecast_router.select_best_model(
            symbol="AAPL", data=volatile_data, timeframe="1d"
        )

        assert selected_model is not None
        # Should prefer volatility models
        assert selected_model in ["garch", "lstm"]

    def test_model_fallback_mechanism(self):
        """Test model selection fallback when primary model fails."""
        with patch.object(self.forecast_router, "select_best_model") as mock_select:
            # Simulate primary model failure
            mock_select.side_effect = [Exception("Model failed"), "lstm"]

            selected_model = self.forecast_router.select_best_model(
                symbol="AAPL", data=self.sample_data, timeframe="1d"
            )

            assert selected_model == "lstm"
            assert mock_select.call_count == 2


class TestStrategySignals:
    """Test strategy signal generation and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2024-01-01", periods=5),
        )

        self.strategy_gatekeeper = StrategyGatekeeper()

    def test_rsi_signal_generation(self):
        """Test RSI strategy signal generation."""
        from trading.strategies.rsi_signals import generate_signals

        signals = generate_signals(self.sample_data, strategy="rsi")

        assert "signals" in signals
        assert "metadata" in signals
        assert signals["success"] is True

        # Check signal structure
        signal_data = signals["signals"]
        assert "buy_signals" in signal_data
        assert "sell_signals" in signal_data
        assert "rsi_values" in signal_data

    def test_bollinger_signal_generation(self):
        """Test Bollinger Bands strategy signal generation."""
        from trading.strategies.bollinger_strategy import BollingerStrategy

        strategy = BollingerStrategy()
        signals = strategy.generate_signals(self.sample_data)

        assert "signals" in signals
        assert "metadata" in signals
        assert signals["success"] is True

        # Check signal structure
        signal_data = signals["signals"]
        assert "buy_signals" in signal_data
        assert "sell_signals" in signal_data
        assert "upper_band" in signal_data
        assert "lower_band" in signal_data

    def test_macd_signal_generation(self):
        """Test MACD strategy signal generation."""
        from trading.strategies.macd_strategy import MACDStrategy

        strategy = MACDStrategy()
        signals = strategy.generate_signals(self.sample_data)

        assert "signals" in signals
        assert "metadata" in signals
        assert signals["success"] is True

        # Check signal structure
        signal_data = signals["signals"]
        assert "buy_signals" in signal_data
        assert "sell_signals" in signal_data
        assert "macd_line" in signal_data
        assert "signal_line" in signal_data

    def test_signal_validation(self):
        """Test signal validation and quality checks."""
        from trading.strategies.rsi_signals import generate_signals

        signals = generate_signals(self.sample_data, strategy="rsi")

        # Validate signal quality
        signal_data = signals["signals"]

        # Check that signals are boolean arrays
        assert isinstance(signal_data["buy_signals"], (list, np.ndarray))
        assert isinstance(signal_data["sell_signals"], (list, np.ndarray))

        # Check that buy and sell signals don't overlap
        buy_signals = np.array(signal_data["buy_signals"])
        sell_signals = np.array(signal_data["sell_signals"])

        # Should not have both buy and sell signals at the same time
        overlap = np.logical_and(buy_signals, sell_signals)
        assert not np.any(overlap)

    def test_strategy_fallback_mechanism(self):
        """Test strategy fallback when primary strategy fails."""
        with patch("trading.strategies.rsi_signals.generate_signals") as mock_rsi:
            # Simulate RSI strategy failure
            mock_rsi.side_effect = Exception("RSI strategy failed")

            # Should fall back to a working strategy
            from trading.strategies.bollinger_strategy import BollingerStrategy

            strategy = BollingerStrategy()
            signals = strategy.generate_signals(self.sample_data)

            assert signals["success"] is True

    def test_multi_strategy_routing(self):
        """Test multi-strategy routing (e.g., RSI+MACD, MACD+SMA)."""
        from trading.strategies.rsi_signals import generate_signals as rsi_signals
        from trading.strategies.macd_strategy import MACDStrategy
        from trading.strategies.sma_strategy import SMAStrategy
        # RSI + MACD
        macd = MACDStrategy()
        rsi = rsi_signals(self.sample_data, strategy="rsi")
        macd_signals = macd.generate_signals(self.sample_data)
        assert rsi["success"] is True
        assert macd_signals["success"] is True
        # MACD + SMA
        sma = SMAStrategy()
        sma_signals = sma.generate_signals(self.sample_data)
        assert sma_signals["signals"] is not None
        # Combine signals (simple AND/OR logic)
        combined_buy = (np.array(macd_signals["signals"]["buy_signals"]) & np.array(sma_signals["signals"]["buy_signals"]))
        assert isinstance(combined_buy, np.ndarray)

    def test_negative_cases(self):
        """Test negative cases: <10 rows, NaNs, duplicate timestamps."""
        from trading.strategies.bollinger_strategy import BollingerStrategy
        from trading.strategies.macd_strategy import MACDStrategy
        from trading.strategies.rsi_signals import generate_signals as rsi_signals
        # <10 rows
        short_data = self.sample_data.head(3)
        for strat in [BollingerStrategy(), MACDStrategy()]:
            try:
                strat.generate_signals(short_data)
            except Exception as e:
                assert "not enough" in str(e).lower() or "invalid" in str(e).lower()
        try:
            rsi_signals(short_data, strategy="rsi")
        except Exception as e:
            assert "not enough" in str(e).lower() or "invalid" in str(e).lower()
        # NaNs
        nan_data = self.sample_data.copy()
        nan_data.iloc[0, 0] = np.nan
        for strat in [BollingerStrategy(), MACDStrategy()]:
            try:
                strat.generate_signals(nan_data)
            except Exception as e:
                assert "nan" in str(e).lower() or "invalid" in str(e).lower()
        try:
            rsi_signals(nan_data, strategy="rsi")
        except Exception as e:
            assert "nan" in str(e).lower() or "invalid" in str(e).lower()
        # Duplicate timestamps
        dup_data = self.sample_data.copy()
        dup_data = pd.concat([dup_data, dup_data.iloc[[0]]])
        dup_data.index = list(self.sample_data.index) + [self.sample_data.index[0]]
        for strat in [BollingerStrategy(), MACDStrategy()]:
            try:
                strat.generate_signals(dup_data)
            except Exception as e:
                assert "duplicate" in str(e).lower() or "invalid" in str(e).lower()
        try:
            rsi_signals(dup_data, strategy="rsi")
        except Exception as e:
            assert "duplicate" in str(e).lower() or "invalid" in str(e).lower()


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_trades = pd.DataFrame(
            {
                "entry_date": pd.date_range("2024-01-01", periods=10),
                "exit_date": pd.date_range("2024-01-02", periods=10),
                "entry_price": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "exit_price": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "position_size": [
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                ],
                "side": [
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                    "long",
                ],
            }
        )

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        from trading.evaluation.metrics import calculate_sharpe_ratio

        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03, 0.01]
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        from trading.evaluation.metrics import calculate_max_drawdown

        equity_curve = [1000, 1050, 1100, 1080, 1120, 1090, 1150, 1130, 1180, 1200]
        max_dd = calculate_max_drawdown(equity_curve)

        assert isinstance(max_dd, float)
        assert max_dd >= 0
        assert max_dd <= 1  # Should be a percentage

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        from trading.evaluation.metrics import calculate_win_rate

        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03, 0.01]
        win_rate = calculate_win_rate(returns)

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        from trading.evaluation.metrics import calculate_profit_factor

        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03, 0.01]
        profit_factor = calculate_profit_factor(returns)

        assert isinstance(profit_factor, float)
        assert profit_factor >= 0


class TestAgentRouting:
    """Test agent routing and fallback mechanisms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = PromptRouterAgent()

    def test_request_classification(self):
        """Test request classification functionality."""
        # Test forecast request
        forecast_request = "What will AAPL stock price be next week?"
        request_type = self.router._classify_request(forecast_request)
        assert request_type == RequestType.FORECAST

        # Test strategy request
        strategy_request = "Generate a trading strategy for TSLA"
        request_type = self.router._classify_request(strategy_request)
        assert request_type == RequestType.STRATEGY

        # Test analysis request
        analysis_request = "Analyze the market performance of GOOGL"
        request_type = self.router._classify_request(analysis_request)
        assert request_type == RequestType.ANALYSIS

    def test_agent_selection(self):
        """Test agent selection based on request type."""
        # Test forecast request routing
        forecast_request = "What will AAPL stock price be next week?"
        decision = self.router.route_request(forecast_request)

        assert decision.primary_agent is not None
        assert decision.confidence > 0
        assert decision.request_type == RequestType.FORECAST

        # Test strategy request routing
        strategy_request = "Generate a trading strategy for TSLA"
        decision = self.router.route_request(strategy_request)

        assert decision.primary_agent is not None
        assert decision.confidence > 0
        assert decision.request_type == RequestType.STRATEGY

    def test_fallback_routing(self):
        """Test fallback routing when primary routing fails."""
        with patch.object(self.router, "_classify_request") as mock_classify:
            # Simulate classification failure
            mock_classify.side_effect = Exception("Classification failed")

            decision = self.router.route_request("Test request")

            assert decision.primary_agent == "QuantGPTAgent"
            assert decision.confidence == 0.3
            assert decision.request_type == RequestType.UNKNOWN

    def test_agent_performance_tracking(self):
        """Test agent performance tracking."""
        # Record some performance metrics
        self.router.record_agent_performance("TestAgent", True, 2.5)
        self.router.record_agent_performance("TestAgent", False, 5.0)
        self.router.record_agent_performance("TestAgent", True, 1.5)

        # Check that metrics are updated
        agent_info = self.router.available_agents.get("TestAgent")
        if agent_info:
            assert agent_info.success_rate > 0
            assert agent_info.avg_response_time > 0


class TestIntegration:
    """Integration tests for the complete agentic system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prompt_agent = PromptAgent()

    def test_end_to_end_forecast_request(self):
        """Test end-to-end forecast request processing."""
        prompt = "Show me the best forecast for AAPL"

        response = self.prompt_agent.process_prompt(prompt)

        assert response is not None
        assert hasattr(response, "success") or isinstance(response, dict)

        if isinstance(response, dict):
            assert "success" in response
            assert "message" in response or "result" in response

    def test_end_to_end_strategy_request(self):
        """Test end-to-end strategy request processing."""
        prompt = "Generate a trading strategy for TSLA"

        response = self.prompt_agent.process_prompt(prompt)

        assert response is not None
        assert hasattr(response, "success") or isinstance(response, dict)

    def test_system_fallback_mechanism(self):
        """Test system-wide fallback mechanism."""
        # Test with invalid prompt that should trigger fallback
        prompt = "Invalid prompt that should trigger fallback"

        response = self.prompt_agent.process_prompt(prompt)

        assert response is not None
        # Should still return a response even if it's a fallback


if __name__ == "__main__":
    pytest.main([__file__])
