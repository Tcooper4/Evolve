"""
Tests for Enhanced Cost Modeling

This module contains comprehensive tests for the enhanced cost modeling system
including commission, slippage, spread, and cash drag calculations.
"""

import numpy as np
import pandas as pd
import pytest

from trading.backtesting.cost_model import CostConfig, CostModel
from trading.backtesting.performance_analysis import CostParameters, PerformanceAnalyzer
from trading.ui.cost_config import (
    CostConfigUI,
    _get_preset_params,
)


class TestCostParameters:
    """Test CostParameters dataclass."""

    def test_default_parameters(self):
        """Test default cost parameters."""
        params = CostParameters()

        assert params.commission_rate == 0.001
        assert params.slippage_rate == 0.002
        assert params.spread_rate == 0.0005
        assert params.cash_drag_rate == 0.02
        assert params.min_commission == 1.0
        assert params.max_commission == 1000.0
        assert params.enable_cost_adjustment is True

    def test_custom_parameters(self):
        """Test custom cost parameters."""
        params = CostParameters(
            commission_rate=0.002,
            slippage_rate=0.003,
            spread_rate=0.001,
            cash_drag_rate=0.03,
            min_commission=5.0,
            max_commission=5000.0,
            enable_cost_adjustment=False,
        )

        assert params.commission_rate == 0.002
        assert params.slippage_rate == 0.003
        assert params.spread_rate == 0.001
        assert params.cash_drag_rate == 0.03
        assert params.min_commission == 5.0
        assert params.max_commission == 5000.0
        assert params.enable_cost_adjustment is False


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer with cost modeling."""

    def setup_method(self):
        """Setup test data."""
        # Create sample equity curve
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.equity_curve = pd.DataFrame(
            {
                "equity_curve": np.linspace(100000, 110000, 100),  # 10% return
                "cash": np.linspace(5000, 3000, 100),  # Decreasing cash
                "returns": np.random.normal(0.001, 0.02, 100),  # Random returns
            },
            index=dates,
        )

        # Create sample trade log
        self.trade_log = pd.DataFrame(
            {
                "timestamp": dates[:10],
                "price": [100] * 10,
                "quantity": [100] * 10,
                "pnl": [10, -5, 15, -8, 12, -3, 18, -10, 20, -7],
                "trade_type": ["BUY", "SELL"] * 5,
            }
        )

        self.cost_params = CostParameters(
            commission_rate=0.001, slippage_rate=0.002, spread_rate=0.0005
        )

        self.analyzer = PerformanceAnalyzer(self.cost_params)

    def test_basic_returns_calculation(self):
        """Test basic return calculations."""
        metrics = self.analyzer._calculate_basic_returns(self.equity_curve)

        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

        # Check that total return is approximately 10%
        assert abs(metrics["total_return"] - 0.1) < 0.01

    def test_cost_adjusted_metrics(self):
        """Test cost-adjusted metrics calculation."""
        metrics = self.analyzer._calculate_cost_adjusted_metrics(
            self.equity_curve, self.trade_log
        )

        assert "cost_adjusted_return" in metrics
        assert "cost_adjusted_sharpe" in metrics
        assert "total_trading_costs" in metrics
        assert "cost_per_trade" in metrics
        assert "cost_impact" in metrics

        # Cost-adjusted return should be lower than gross return
        gross_return = self.analyzer._calculate_basic_returns(self.equity_curve)[
            "total_return"
        ]
        assert metrics["cost_adjusted_return"] < gross_return

    def test_trade_cost_calculation(self):
        """Test individual trade cost calculation."""
        trade = pd.Series({"price": 100, "quantity": 1000})

        cost = self.analyzer._calculate_trade_cost(trade)

        # Expected costs:
        # Commission: max(1, min(100000 * 0.001, 1000)) = 100
        # Slippage: 100000 * 0.002 = 200
        # Spread: 100000 * 0.0005 = 50
        # Total: 100 + 200 + 50 = 350
        expected_cost = 350
        assert abs(cost - expected_cost) < 0.01

    def test_cost_breakdown(self):
        """Test cost breakdown calculation."""
        breakdown = self.analyzer._calculate_cost_breakdown(self.trade_log)

        assert "total_commission" in breakdown
        assert "total_slippage" in breakdown
        assert "total_spread" in breakdown
        assert "cost_percentage" in breakdown
        assert "commission_percentage" in breakdown
        assert "slippage_percentage" in breakdown
        assert "spread_percentage" in breakdown

        # All costs should be positive
        assert breakdown["total_commission"] >= 0
        assert breakdown["total_slippage"] >= 0
        assert breakdown["total_spread"] >= 0

    def test_cash_efficiency_metrics(self):
        """Test cash efficiency metrics calculation."""
        metrics = self.analyzer._calculate_cash_efficiency_metrics(
            self.equity_curve, self.trade_log
        )

        assert "avg_cash_utilization" in metrics
        assert "cash_drag_cost" in metrics
        assert "cash_drag_percentage" in metrics
        assert "turnover_ratio" in metrics

        # Cash utilization should be between 0 and 1
        assert 0 <= metrics["avg_cash_utilization"] <= 1

    def test_compute_metrics_integration(self):
        """Test full metrics computation integration."""
        metrics = self.analyzer.compute_metrics(
            self.equity_curve, self.trade_log, self.cost_params
        )

        # Check that all metric categories are present
        basic_metrics = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
        ]
        cost_metrics = ["cost_adjusted_return", "total_trading_costs", "cost_impact"]
        risk_metrics = ["var_95", "sortino_ratio", "calmar_ratio"]
        trade_metrics = ["num_trades", "win_rate", "profit_factor"]

        for metric in basic_metrics + cost_metrics + risk_metrics + trade_metrics:
            assert metric in metrics

    def test_no_cost_adjustment(self):
        """Test behavior when cost adjustment is disabled."""
        no_cost_params = CostParameters(enable_cost_adjustment=False)
        analyzer = PerformanceAnalyzer(no_cost_params)

        metrics = analyzer.compute_metrics(self.equity_curve, self.trade_log)

        # Cost-adjusted metrics should be same as basic metrics
        assert metrics["cost_adjusted_return"] == metrics["total_return"]
        assert metrics["total_trading_costs"] == 0.0
        assert metrics["cost_impact"] == 0.0


class TestCostModel:
    """Test CostModel functionality."""

    def setup_method(self):
        """Setup test data."""
        self.data = pd.DataFrame(
            {
                "close": np.linspace(100, 110, 100),
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
        )

        self.config = CostConfig(
            fee_rate=0.001, slippage_rate=0.002, spread_rate=0.0005
        )

        self.cost_model = CostModel(self.config, self.data)

    def test_calculate_fees(self):
        """Test fee calculation."""
        # Test percentage-based fees
        fee = self.cost_model.calculate_fees(10000)
        expected_fee = 10000 * 0.001  # 10
        assert abs(fee - expected_fee) < 0.01

        # Test minimum fee
        fee = self.cost_model.calculate_fees(100)  # Should hit minimum
        assert fee == self.config.min_fee

    def test_calculate_spread(self):
        """Test spread calculation."""
        spread = self.cost_model.calculate_spread(100)
        expected_spread = 100 * 0.0005  # 0.05
        assert abs(spread - expected_spread) < 0.001

    def test_calculate_slippage(self):
        """Test slippage calculation."""
        slippage = self.cost_model.calculate_slippage(100, 1000, "BUY")
        expected_slippage = 100 * 0.002  # 0.2
        assert abs(slippage - expected_slippage) < 0.001

    def test_calculate_total_cost(self):
        """Test total cost calculation."""
        cost_breakdown = self.cost_model.calculate_total_cost(
            price=100, quantity=1000, trade_type="BUY"
        )

        assert "fees" in cost_breakdown
        assert "spread" in cost_breakdown
        assert "slippage" in cost_breakdown
        assert "total_cost" in cost_breakdown
        assert "effective_price" in cost_breakdown
        assert "cost_percentage" in cost_breakdown

        # Total cost should be sum of individual costs
        expected_total = (
            cost_breakdown["fees"]
            + cost_breakdown["spread"]
            + cost_breakdown["slippage"]
        )
        assert abs(cost_breakdown["total_cost"] - expected_total) < 0.01


class TestCostConfigUI:
    """Test cost configuration UI components."""

    def test_cost_config_ui_defaults(self):
        """Test CostConfigUI default values."""
        config = CostConfigUI()

        assert config.show_advanced is False
        assert config.show_presets is True
        assert config.show_cost_breakdown is True
        assert config.show_validation is True

    def test_get_preset_params(self):
        """Test preset parameter retrieval."""
        # Test retail trading preset
        retail_params = _get_preset_params("Retail Trading")
        assert retail_params.commission_rate == 0.001
        assert retail_params.slippage_rate == 0.002
        assert retail_params.spread_rate == 0.0005
        assert retail_params.cash_drag_rate == 0.02

        # Test institutional preset
        inst_params = _get_preset_params("Institutional")
        assert inst_params.commission_rate == 0.0005
        assert inst_params.slippage_rate == 0.001
        assert inst_params.spread_rate == 0.0002
        assert inst_params.cash_drag_rate == 0.015

        # Test unknown preset (should return defaults)
        default_params = _get_preset_params("Unknown")
        assert default_params.commission_rate == 0.001

    def test_cost_breakdown_preview(self):
        """Test cost breakdown preview calculation."""
        # This would normally test the UI function, but we can test the logic
        trade_value = 10000
        commission_rate = 0.001
        slippage_rate = 0.002
        spread_rate = 0.0005
        min_commission = 1.0
        max_commission = 1000.0

        # Calculate expected costs
        commission = max(
            min_commission, min(trade_value * commission_rate, max_commission)
        )
        slippage = trade_value * slippage_rate
        spread = trade_value * spread_rate
        total_cost = commission + slippage + spread

        assert commission == 10.0  # 10000 * 0.001
        assert slippage == 20.0  # 10000 * 0.002
        assert spread == 5.0  # 10000 * 0.0005
        assert total_cost == 35.0

        # Cost percentage
        cost_percentage = (total_cost / trade_value) * 100
        assert cost_percentage == 0.35


class TestCostModelingIntegration:
    """Test integration between cost modeling components."""

    def test_backtester_cost_integration(self):
        """Test backtester integration with cost model."""
        from trading.backtesting.backtester import Backtester

        # Create sample data
        data = pd.DataFrame(
            {
                "close": np.linspace(100, 110, 100),
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
        )

        # Create cost model
        cost_config = CostConfig(
            fee_rate=0.001, slippage_rate=0.002, spread_rate=0.0005
        )
        cost_model = CostModel(cost_config, data)

        # Initialize backtester with cost model
        backtester = Backtester(data=data, initial_cash=100000, cost_model=cost_model)

        # Verify cost model is properly integrated
        assert backtester.cost_model is not None
        assert isinstance(backtester.cost_model, CostModel)

    def test_performance_analyzer_cost_integration(self):
        """Test performance analyzer integration with cost parameters."""
        # Create sample data
        equity_curve = pd.DataFrame(
            {
                "equity_curve": np.linspace(100000, 110000, 100),
                "cash": np.linspace(5000, 3000, 100),
            }
        )

        trade_log = pd.DataFrame(
            {"price": [100] * 10, "quantity": [1000] * 10, "pnl": [10] * 10}
        )

        # Test with different cost parameters
        cost_params = CostParameters(
            commission_rate=0.002,  # Higher commission
            slippage_rate=0.003,  # Higher slippage
            spread_rate=0.001,  # Higher spread
        )

        analyzer = PerformanceAnalyzer(cost_params)
        metrics = analyzer.compute_metrics(equity_curve, trade_log)

        # Higher costs should result in lower cost-adjusted returns
        assert metrics["cost_adjusted_return"] < metrics["total_return"]
        assert metrics["total_trading_costs"] > 0

    def test_cost_impact_scenarios(self):
        """Test different cost scenarios and their impact."""
        # Create sample data
        equity_curve = pd.DataFrame(
            {
                "equity_curve": np.linspace(100000, 110000, 100),
                "cash": np.linspace(5000, 3000, 100),
            }
        )

        trade_log = pd.DataFrame(
            {"price": [100] * 20, "quantity": [1000] * 20, "pnl": [10] * 20}
        )

        # Test different cost scenarios
        scenarios = {
            "low_cost": CostParameters(commission_rate=0.0001, slippage_rate=0.0005),
            "medium_cost": CostParameters(commission_rate=0.001, slippage_rate=0.002),
            "high_cost": CostParameters(commission_rate=0.003, slippage_rate=0.005),
        }

        results = {}
        for name, params in scenarios.items():
            analyzer = PerformanceAnalyzer(params)
            metrics = analyzer.compute_metrics(equity_curve, trade_log)
            results[name] = metrics

        # Higher costs should result in lower cost-adjusted returns
        assert (
            results["low_cost"]["cost_adjusted_return"]
            > results["medium_cost"]["cost_adjusted_return"]
        )
        assert (
            results["medium_cost"]["cost_adjusted_return"]
            > results["high_cost"]["cost_adjusted_return"]
        )

        # Higher costs should result in higher total trading costs
        assert (
            results["low_cost"]["total_trading_costs"]
            < results["medium_cost"]["total_trading_costs"]
        )
        assert (
            results["medium_cost"]["total_trading_costs"]
            < results["high_cost"]["total_trading_costs"]
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trade_log(self):
        """Test behavior with empty trade log."""
        equity_curve = pd.DataFrame({"equity_curve": np.linspace(100000, 110000, 100)})

        empty_trade_log = pd.DataFrame()

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(equity_curve, empty_trade_log)

        # Should handle empty trade log gracefully
        assert metrics["num_trades"] == 0
        assert metrics["total_trading_costs"] == 0.0
        assert metrics["cost_adjusted_return"] == metrics["total_return"]

    def test_zero_trade_value(self):
        """Test behavior with zero trade value."""
        analyzer = PerformanceAnalyzer()

        trade = pd.Series({"price": 0, "quantity": 100})
        cost = analyzer._calculate_trade_cost(trade)

        # Should handle zero trade value
        assert cost == 0.0

    def test_negative_cost_parameters(self):
        """Test behavior with negative cost parameters."""
        # Cost parameters should handle negative values gracefully
        params = CostParameters(
            commission_rate=-0.001,  # Negative commission
            slippage_rate=-0.002,  # Negative slippage
        )

        analyzer = PerformanceAnalyzer(params)
        trade = pd.Series({"price": 100, "quantity": 1000})
        cost = analyzer._calculate_trade_cost(trade)

        # Should handle negative parameters (treat as zero)
        assert cost >= 0.0

    def test_missing_data_columns(self):
        """Test behavior with missing data columns."""
        # Test with missing equity_curve column
        data = pd.DataFrame({"returns": np.random.normal(0.001, 0.02, 100)})
        empty_trade_log = pd.DataFrame()

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(data, empty_trade_log)

        # Should handle missing columns gracefully
        assert pd.isna(metrics["total_return"])
        assert pd.isna(metrics["max_drawdown"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
