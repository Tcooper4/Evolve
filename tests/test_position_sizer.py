"""
Tests for PositionSizer

This module contains comprehensive tests for the PositionSizer class,
covering all sizing strategies, edge cases, and integration scenarios.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.base_agent_interface import AgentConfig
from trading.agents.execution_agent import ExecutionAgent, TradeDirection, TradeSignal
from trading.portfolio.position_sizer import (
    MarketContext,
    PortfolioContext,
    PositionSizer,
    SignalContext,
    SizingParameters,
    SizingStrategy,
)


class TestPositionSizer:
    """Test cases for PositionSizer class."""

    @pytest.fixture
    def position_sizer(self):
        """Create a PositionSizer instance for testing."""
        return PositionSizer()

    @pytest.fixture
    def market_context(self):
        """Create a sample market context."""
        return MarketContext(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.18,
            volume=50000000,
            market_regime="normal",
            correlation=0.3,
            liquidity_score=0.9,
            bid_ask_spread=0.0005,
        )

    @pytest.fixture
    def signal_context(self):
        """Create a sample signal context."""
        return SignalContext(
            confidence=0.7,
            forecast_certainty=0.6,
            strategy_performance=0.08,
            win_rate=0.6,
            avg_win=0.025,
            avg_loss=-0.015,
            sharpe_ratio=0.8,
            max_drawdown=0.12,
            signal_strength=0.7,
        )

    @pytest.fixture
    def portfolio_context(self):
        """Create a sample portfolio context."""
        return PortfolioContext(
            total_capital=100000.0,
            available_capital=50000.0,
            current_exposure=0.5,
            open_positions=2,
            daily_pnl=0.01,
            portfolio_volatility=0.15,
        )

    def test_initialization(self, position_sizer):
        """Test PositionSizer initialization."""
        assert position_sizer is not None
        assert hasattr(position_sizer, "config")
        assert hasattr(position_sizer, "sizing_history")
        assert position_sizer.config["default_strategy"] == SizingStrategy.FIXED_PERCENTAGE

    def test_initialization_with_config(self):
        """Test PositionSizer initialization with custom config."""
        config = {"default_strategy": SizingStrategy.KELLY_CRITERION, "risk_per_trade": 0.03, "max_position_size": 0.25}
        position_sizer = PositionSizer(config)

        assert position_sizer.config["default_strategy"] == SizingStrategy.KELLY_CRITERION
        assert position_sizer.config["risk_per_trade"] == 0.03
        assert position_sizer.config["max_position_size"] == 0.25

    def test_fixed_percentage_sizing(self, position_sizer, portfolio_context):
        """Test fixed percentage sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.FIXED_PERCENTAGE, risk_per_trade=0.02)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=Mock(),
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size == 0.02
        assert details["strategy"] == "fixed_percentage"
        assert details["risk_percentage"] > 0

    def test_kelly_criterion_sizing(self, position_sizer, signal_context, portfolio_context):
        """Test Kelly Criterion sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.KELLY_CRITERION, kelly_fraction=0.25)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "kelly_criterion"
        assert details["risk_percentage"] > 0

    def test_volatility_based_sizing(self, position_sizer, market_context, portfolio_context):
        """Test volatility-based sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.VOLATILITY_BASED, volatility_multiplier=1.0)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=market_context,
            signal_context=Mock(),
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "volatility_based"
        assert details["risk_percentage"] > 0

    def test_confidence_based_sizing(self, position_sizer, signal_context, portfolio_context):
        """Test confidence-based sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.CONFIDENCE_BASED, confidence_multiplier=1.0)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "confidence_based"
        assert details["risk_percentage"] > 0

    def test_forecast_certainty_sizing(self, position_sizer, signal_context, portfolio_context):
        """Test forecast certainty sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.FORECAST_CERTAINTY, confidence_multiplier=1.0)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "forecast_certainty"
        assert details["risk_percentage"] > 0

    def test_optimal_f_sizing(self, position_sizer, signal_context, portfolio_context):
        """Test Optimal F sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.OPTIMAL_F, optimal_f_risk=0.02)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "optimal_f"
        assert details["risk_percentage"] > 0

    def test_risk_parity_sizing(self, position_sizer, market_context, portfolio_context):
        """Test risk parity sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.RISK_PARITY, risk_per_trade=0.02)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=market_context,
            signal_context=Mock(),
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "risk_parity"
        assert details["risk_percentage"] > 0

    def test_martingale_sizing(self, position_sizer, portfolio_context):
        """Test Martingale sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.MARTINGALE, base_position_size=0.1)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=Mock(),
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "martingale"

    def test_anti_martingale_sizing(self, position_sizer, portfolio_context):
        """Test Anti-Martingale sizing strategy."""
        params = SizingParameters(strategy=SizingStrategy.ANTI_MARTINGALE, base_position_size=0.1)

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=Mock(),
            portfolio_context=portfolio_context,
            sizing_params=params,
        )

        assert position_size > 0
        assert details["strategy"] == "anti_martingale"

    def test_risk_adjustment(self, position_sizer, signal_context, portfolio_context):
        """Test risk adjustment functionality."""
        # Test with negative daily PnL
        portfolio_context.daily_pnl = -0.02

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert position_size > 0
        assert details["risk_percentage"] > 0

    def test_correlation_adjustment(self, position_sizer, signal_context, portfolio_context):
        """Test correlation adjustment functionality."""
        market_context = MarketContext(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.18,
            volume=50000000,
            correlation=0.8,  # High correlation
            liquidity_score=0.9,
            bid_ask_spread=0.0005,
        )

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert position_size > 0
        assert details["risk_percentage"] > 0

    def test_volatility_adjustment(self, position_sizer, signal_context, portfolio_context):
        """Test volatility adjustment functionality."""
        market_context = MarketContext(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.4,  # High volatility
            volume=50000000,
            correlation=0.3,
            liquidity_score=0.9,
            bid_ask_spread=0.0005,
        )

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert position_size > 0
        assert details["risk_percentage"] > 0

    def test_liquidity_adjustment(self, position_sizer, signal_context, portfolio_context):
        """Test liquidity adjustment functionality."""
        market_context = MarketContext(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.18,
            volume=50000000,
            correlation=0.3,
            liquidity_score=0.3,  # Low liquidity
            bid_ask_spread=0.01,  # Wide spread
        )

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=market_context,
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert position_size > 0
        assert details["risk_percentage"] > 0

    def test_position_size_constraints(self, position_sizer, signal_context, portfolio_context):
        """Test position size constraints."""
        # Test with very small available capital
        portfolio_context.available_capital = 1000.0

        position_size, details = position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert position_size > 0
        assert details["risk_percentage"] > 0
        assert details["position_value"] <= portfolio_context.available_capital

    def test_error_handling(self, position_sizer):
        """Test error handling in position sizing."""
        # Test with invalid parameters
        with pytest.raises(Exception):
            position_sizer.calculate_position_size(
                entry_price=150.0,
                stop_loss_price=147.0,
                market_context=None,
                signal_context=None,
                portfolio_context=None,
            )

    def test_conservative_fallback(self, position_sizer):
        """Test conservative fallback when errors occur."""
        # Mock an error scenario
        with patch.object(position_sizer, "_calculate_base_size", side_effect=Exception("Test error")):
            position_size, details = position_sizer.calculate_position_size(
                entry_price=150.0,
                stop_loss_price=147.0,
                market_context=Mock(),
                signal_context=Mock(),
                portfolio_context=Mock(),
            )

            assert position_size > 0
            assert details["strategy"] == "conservative_fallback"
            assert "error" in details

    def test_sizing_history(self, position_sizer, signal_context, portfolio_context):
        """Test sizing history tracking."""
        initial_history_length = len(position_sizer.sizing_history)

        position_sizer.calculate_position_size(
            entry_price=150.0,
            stop_loss_price=147.0,
            market_context=Mock(),
            signal_context=signal_context,
            portfolio_context=portfolio_context,
        )

        assert len(position_sizer.sizing_history) == initial_history_length + 1

        # Test history limit
        for _ in range(1000):
            position_sizer.calculate_position_size(
                entry_price=150.0,
                stop_loss_price=147.0,
                market_context=Mock(),
                signal_context=signal_context,
                portfolio_context=portfolio_context,
            )

        assert len(position_sizer.sizing_history) <= 1000

    def test_get_sizing_history(self, position_sizer, signal_context, portfolio_context):
        """Test getting sizing history."""
        # Add some sizing decisions
        for _ in range(5):
            position_sizer.calculate_position_size(
                entry_price=150.0,
                stop_loss_price=147.0,
                market_context=Mock(),
                signal_context=signal_context,
                portfolio_context=portfolio_context,
            )

        history = position_sizer.get_sizing_history(limit=3)
        assert len(history) == 3

        history = position_sizer.get_sizing_history(limit=10)
        assert len(history) == 5

    def test_get_sizing_summary(self, position_sizer, signal_context, portfolio_context):
        """Test getting sizing summary."""
        # Add some sizing decisions
        for _ in range(5):
            position_sizer.calculate_position_size(
                entry_price=150.0,
                stop_loss_price=147.0,
                market_context=Mock(),
                signal_context=signal_context,
                portfolio_context=portfolio_context,
            )

        summary = position_sizer.get_sizing_summary()

        assert "total_sizing_decisions" in summary
        assert "average_position_size" in summary
        assert "average_risk_percentage" in summary
        assert "strategy_usage" in summary
        assert summary["total_sizing_decisions"] == 5

    def test_sizing_parameters_serialization(self):
        """Test SizingParameters serialization."""
        params = SizingParameters(
            strategy=SizingStrategy.KELLY_CRITERION,
            risk_per_trade=0.02,
            max_position_size=0.2,
            confidence_multiplier=1.5,
            volatility_multiplier=1.0,
            kelly_fraction=0.25,
            optimal_f_risk=0.02,
            base_position_size=0.1,
        )

        # Test to_dict
        params_dict = params.to_dict()
        assert params_dict["strategy"] == "kelly_criterion"
        assert params_dict["risk_per_trade"] == 0.02
        assert params_dict["max_position_size"] == 0.2

        # Test from_dict
        restored_params = SizingParameters.from_dict(params_dict)
        assert restored_params.strategy == SizingStrategy.KELLY_CRITERION
        assert restored_params.risk_per_trade == 0.02
        assert restored_params.max_position_size == 0.2

    def test_context_serialization(self, market_context, signal_context, portfolio_context):
        """Test context serialization."""
        # Test MarketContext
        market_dict = market_context.to_dict()
        assert market_dict["symbol"] == "AAPL"
        assert market_dict["current_price"] == 150.0

        # Test SignalContext
        signal_dict = signal_context.to_dict()
        assert signal_dict["confidence"] == 0.7
        assert signal_dict["win_rate"] == 0.6

        # Test PortfolioContext
        portfolio_dict = portfolio_context.to_dict()
        assert portfolio_dict["total_capital"] == 100000.0
        assert portfolio_dict["available_capital"] == 50000.0


class TestPositionSizerIntegration:
    """Test cases for PositionSizer integration with ExecutionAgent."""

    @pytest.fixture
    def execution_agent(self):
        """Create an ExecutionAgent instance for testing."""
        config = AgentConfig(
            name="test_execution_agent",
            agent_type="execution",
            enabled=True,
            custom_config={
                "execution_mode": "simulation",
                "position_sizing_config": {
                    "default_strategy": "confidence_based",
                    "risk_per_trade": 0.02,
                    "max_position_size": 0.2,
                },
            },
        )
        return ExecutionAgent(config)

    @pytest.fixture
    def trade_signal(self):
        """Create a sample trade signal."""
        return TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test_strategy",
            confidence=0.75,
            entry_price=150.0,
            market_data={"forecast_certainty": 0.7, "signal_strength": 0.8, "sizing_strategy": "confidence_based"},
        )

    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        return {
            "AAPL": {
                "price": 150.0,
                "volatility": 0.18,
                "volume": 50000000,
                "liquidity_score": 0.9,
                "bid_ask_spread": 0.0005,
            }
        }

    def test_position_sizer_integration(self, execution_agent, trade_signal, market_data):
        """Test PositionSizer integration with ExecutionAgent."""
        # Test position size calculation
        position_size, sizing_details = execution_agent._calculate_position_size(
            trade_signal, trade_signal.entry_price, market_data
        )

        assert position_size > 0
        assert "strategy" in sizing_details
        assert "risk_percentage" in sizing_details
        assert "position_value" in sizing_details

    def test_different_sizing_strategies_integration(self, execution_agent, market_data):
        """Test different sizing strategies in ExecutionAgent."""
        strategies = ["fixed_percentage", "kelly_criterion", "volatility_based", "confidence_based"]

        for strategy in strategies:
            signal = TradeSignal(
                symbol="AAPL",
                direction=TradeDirection.LONG,
                strategy="test_strategy",
                confidence=0.7,
                entry_price=150.0,
                market_data={"sizing_strategy": strategy, "risk_per_trade": 0.02},
            )

            position_size, sizing_details = execution_agent._calculate_position_size(
                signal, signal.entry_price, market_data
            )

            assert position_size > 0
            assert sizing_details["strategy"] == strategy

    def test_manual_position_size(self, execution_agent, market_data):
        """Test manual position size override."""
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test_strategy",
            confidence=0.7,
            entry_price=150.0,
            size=100.0,  # Manual size
        )

        position_size, sizing_details = execution_agent._calculate_position_size(
            signal, signal.entry_price, market_data
        )

        assert position_size == 100.0
        assert sizing_details["strategy"] == "manual_size"

    def test_error_handling_integration(self, execution_agent, trade_signal, market_data):
        """Test error handling in ExecutionAgent integration."""
        # Test with invalid market data
        invalid_market_data = {}

        position_size, sizing_details = execution_agent._calculate_position_size(
            trade_signal, trade_signal.entry_price, invalid_market_data
        )

        assert position_size > 0
        assert "error" in sizing_details or sizing_details["strategy"] == "conservative_fallback"


if __name__ == "__main__":
    pytest.main([__file__])
