"""
Unit tests for Backtester.

Tests backtester functionality with synthetic signals and price data,
including edge cases and performance validation.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import backtester modules
try:
    from trading.backtesting.backtester import Backtester
    from trading.backtesting.performance_analyzer import PerformanceAnalyzer

    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False
    Backtester = Mock()
    PerformanceAnalyzer = Mock()


class TestBacktester:
    """Test suite for Backtester."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic price and signal data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create price data with trend and noise
        trend = np.linspace(100, 120, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        # Create synthetic signals (1=buy, -1=sell, 0=hold)
        signals = np.zeros(100)
        signals[10] = 1  # Buy signal
        signals[30] = -1  # Sell signal
        signals[50] = 1  # Buy signal
        signals[80] = -1  # Sell signal

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": close_prices + np.random.uniform(0, 3, 100),
                "Low": close_prices - np.random.uniform(0, 3, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
                "signal": signals,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def trending_data(self):
        """Create trending price data with signals."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Strong upward trend
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 1, 100)
        close_prices = trend + noise

        # Signals that follow the trend
        signals = np.zeros(100)
        signals[5] = 1  # Early buy
        signals[95] = -1  # Late sell

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": close_prices + np.random.uniform(0, 2, 100),
                "Low": close_prices - np.random.uniform(0, 2, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
                "signal": signals,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def volatile_data(self):
        """Create volatile price data with frequent signals."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Volatile price movement
        t = np.linspace(0, 4 * np.pi, 100)
        sine_wave = 100 + 15 * np.sin(t)  # Large oscillations
        noise = np.random.normal(0, 3, 100)
        close_prices = sine_wave + noise

        # Frequent signals
        signals = np.zeros(100)
        for i in range(10, 90, 10):  # Signal every 10 days
            signals[i] = 1 if i % 20 == 0 else -1

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": close_prices + np.random.uniform(0, 4, 100),
                "Low": close_prices - np.random.uniform(0, 4, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
                "signal": signals,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def empty_data(self):
        """Create empty DataFrame."""
        return pd.DataFrame()

    @pytest.fixture
    def no_signals_data(self):
        """Create data with no signals."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close_prices = np.linspace(100, 110, 50) + np.random.normal(0, 1, 50)

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": close_prices + np.random.uniform(0, 2, 50),
                "Low": close_prices - np.random.uniform(0, 2, 50),
                "Volume": np.random.uniform(1000000, 5000000, 50),
                "signal": np.zeros(50),  # No signals
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def backtester(self):
        """Create Backtester instance."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("Backtester not available")
        return Backtester()

    @pytest.fixture
    def performance_analyzer(self):
        """Create PerformanceAnalyzer instance."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("PerformanceAnalyzer not available")
        return PerformanceAnalyzer()

    def test_backtester_instantiation(self, backtester):
        """Test that Backtester instantiates correctly."""
        assert backtester is not None
        assert hasattr(backtester, "run_backtest")
        assert hasattr(backtester, "calculate_returns")
        assert hasattr(backtester, "calculate_metrics")

    def test_basic_backtest(self, backtester, synthetic_data):
        """Test basic backtest functionality."""
        result = backtester.run_backtest(synthetic_data)

        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert "equity_curve" in result
        assert "returns" in result
        assert "metrics" in result

    def test_equity_curve_calculation(self, backtester, synthetic_data):
        """Test equity curve calculation."""
        result = backtester.run_backtest(synthetic_data)

        equity_curve = result["equity_curve"]
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == len(synthetic_data)
        assert not equity_curve.isna().any()

        # Equity curve should be monotonically increasing or decreasing
        # (depending on strategy performance)
        assert (equity_curve.diff().dropna() >= 0).all() or (
            equity_curve.diff().dropna() <= 0
        ).all()

    def test_returns_calculation(self, backtester, synthetic_data):
        """Test returns calculation."""
        result = backtester.run_backtest(synthetic_data)

        returns = result["returns"]
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(synthetic_data)

        # Returns should be calculated for all periods
        valid_returns = returns.dropna()
        assert len(valid_returns) > 0

    def test_performance_metrics(self, backtester, synthetic_data):
        """Test performance metrics calculation."""
        result = backtester.run_backtest(synthetic_data)

        metrics = result["metrics"]
        assert isinstance(metrics, dict)

        # Check required metrics
        required_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_total_return_calculation(self, backtester, synthetic_data):
        """Test total return calculation."""
        result = backtester.run_backtest(synthetic_data)

        total_return = result["metrics"]["total_return"]
        assert isinstance(total_return, (int, float))

        # Total return should be reasonable (not infinite or NaN)
        assert not np.isnan(total_return)
        assert not np.isinf(total_return)
        assert total_return > -1  # Can't lose more than 100%

    def test_sharpe_ratio_calculation(self, backtester, synthetic_data):
        """Test Sharpe ratio calculation."""
        result = backtester.run_backtest(synthetic_data)

        sharpe_ratio = result["metrics"]["sharpe_ratio"]
        assert isinstance(sharpe_ratio, (int, float))

        # Sharpe ratio should be reasonable
        assert not np.isnan(sharpe_ratio)
        assert not np.isinf(sharpe_ratio)

    def test_max_drawdown_calculation(self, backtester, synthetic_data):
        """Test maximum drawdown calculation."""
        result = backtester.run_backtest(synthetic_data)

        max_drawdown = result["metrics"]["max_drawdown"]
        assert isinstance(max_drawdown, (int, float))

        # Max drawdown should be between 0 and 1 (0% to 100%)
        assert 0 <= max_drawdown <= 1
        assert not np.isnan(max_drawdown)

    def test_win_rate_calculation(self, backtester, synthetic_data):
        """Test win rate calculation."""
        result = backtester.run_backtest(synthetic_data)

        win_rate = result["metrics"]["win_rate"]
        assert isinstance(win_rate, (int, float))

        # Win rate should be between 0 and 1 (0% to 100%)
        assert 0 <= win_rate <= 1
        assert not np.isnan(win_rate)

    def test_trending_data_performance(self, backtester, trending_data):
        """Test performance on trending data."""
        result = backtester.run_backtest(trending_data)

        assert result["success"] is True

        # On trending data, strategy should perform reasonably
        total_return = result["metrics"]["total_return"]
        assert not np.isnan(total_return)
        assert not np.isinf(total_return)

    def test_volatile_data_performance(self, backtester, volatile_data):
        """Test performance on volatile data."""
        result = backtester.run_backtest(volatile_data)

        assert result["success"] is True

        # On volatile data, drawdown might be higher
        max_drawdown = result["metrics"]["max_drawdown"]
        assert 0 <= max_drawdown <= 1
        assert not np.isnan(max_drawdown)

    def test_empty_data_handling(self, backtester, empty_data):
        """Test handling of empty data."""
        result = backtester.run_backtest(empty_data)

        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["empty", "no data", "insufficient"]
        )

    def test_no_signals_handling(self, backtester, no_signals_data):
        """Test handling of data with no signals."""
        result = backtester.run_backtest(no_signals_data)

        if result["success"]:
            # If it succeeds, should have no trading activity
            equity_curve = result["equity_curve"]
            assert (equity_curve == equity_curve.iloc[0]).all()  # No change in equity

            total_return = result["metrics"]["total_return"]
            assert total_return == 0  # No return without trading
        else:
            # If it fails, should be due to no signals
            assert any(
                keyword in result["error"].lower()
                for keyword in ["no signals", "no trades", "no activity"]
            )

    def test_transaction_costs(self, backtester, synthetic_data):
        """Test backtest with transaction costs."""
        # Test with different transaction costs
        costs = [0, 0.001, 0.01, 0.05]  # 0%, 0.1%, 1%, 5%

        for cost in costs:
            result = backtester.run_backtest(synthetic_data, transaction_cost=cost)

            assert result["success"] is True
            total_return = result["metrics"]["total_return"]
            assert not np.isnan(total_return)
            assert not np.isinf(total_return)

    def test_initial_capital(self, backtester, synthetic_data):
        """Test backtest with different initial capital."""
        capitals = [10000, 50000, 100000, 1000000]

        for capital in capitals:
            result = backtester.run_backtest(synthetic_data, initial_capital=capital)

            assert result["success"] is True
            equity_curve = result["equity_curve"]

            # Initial equity should match initial capital
            assert abs(equity_curve.iloc[0] - capital) < 1e-6

    def test_position_sizing(self, backtester, synthetic_data):
        """Test different position sizing strategies."""
        sizing_strategies = ["fixed", "percentage", "kelly"]

        for strategy in sizing_strategies:
            result = backtester.run_backtest(synthetic_data, position_sizing=strategy)

            assert result["success"] is True
            assert "equity_curve" in result
            assert "metrics" in result

    def test_risk_management(self, backtester, synthetic_data):
        """Test risk management features."""
        # Test stop loss
        result = backtester.run_backtest(synthetic_data, stop_loss=0.05)
        assert result["success"] is True

        # Test take profit
        result = backtester.run_backtest(synthetic_data, take_profit=0.10)
        assert result["success"] is True

        # Test trailing stop
        result = backtester.run_backtest(synthetic_data, trailing_stop=0.03)
        assert result["success"] is True

    def test_performance_analyzer(self, performance_analyzer, synthetic_data):
        """Test performance analyzer functionality."""
        # Create sample backtest results
        equity_curve = pd.Series(
            [10000, 10100, 10200, 10150, 10300],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        metrics = performance_analyzer.calculate_metrics(equity_curve)

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    def test_rolling_metrics(self, backtester, synthetic_data):
        """Test rolling performance metrics."""
        result = backtester.run_backtest(synthetic_data)

        if "rolling_metrics" in result:
            rolling_metrics = result["rolling_metrics"]
            assert isinstance(rolling_metrics, pd.DataFrame)
            assert len(rolling_metrics) > 0

            # Check rolling Sharpe ratio
            if "rolling_sharpe" in rolling_metrics.columns:
                rolling_sharpe = rolling_metrics["rolling_sharpe"].dropna()
                assert len(rolling_sharpe) > 0
                assert not rolling_sharpe.isna().any()

    def test_trade_analysis(self, backtester, synthetic_data):
        """Test trade analysis functionality."""
        result = backtester.run_backtest(synthetic_data)

        if "trades" in result:
            trades = result["trades"]
            assert isinstance(trades, list)

            if trades:
                trade = trades[0]
                assert "entry_date" in trade
                assert "exit_date" in trade
                assert "entry_price" in trade
                assert "exit_price" in trade
                assert "return" in trade
                assert "pnl" in trade

    def test_benchmark_comparison(self, backtester, synthetic_data):
        """Test benchmark comparison."""
        # Create benchmark data (buy and hold)
        benchmark_data = synthetic_data.copy()
        benchmark_data["signal"] = 0
        benchmark_data.loc[benchmark_data.index[0], "signal"] = 1  # Buy at start

        result = backtester.run_backtest(synthetic_data, benchmark=benchmark_data)

        if "benchmark_comparison" in result:
            comparison = result["benchmark_comparison"]
            assert isinstance(comparison, dict)
            assert "strategy_return" in comparison
            assert "benchmark_return" in comparison
            assert "excess_return" in comparison

    def test_risk_metrics(self, backtester, synthetic_data):
        """Test additional risk metrics."""
        result = backtester.run_backtest(synthetic_data)

        metrics = result["metrics"]

        # Check for additional risk metrics
        risk_metrics = ["var_95", "cvar_95", "sortino_ratio", "calmar_ratio"]
        for metric in risk_metrics:
            if metric in metrics:
                value = metrics[metric]
                assert isinstance(value, (int, float))
                assert not np.isnan(value)
                assert not np.isinf(value)

    def test_data_validation(self, backtester):
        """Test data validation."""
        # Test with missing required columns
        invalid_data = pd.DataFrame(
            {"Close": [100, 101, 102], "Volume": [1000000, 1000000, 1000000]}
        )

        result = backtester.run_backtest(invalid_data)
        assert result["success"] is False
        assert "error" in result

    def test_edge_cases(self, backtester):
        """Test various edge cases."""
        # Test with single data point
        single_point = pd.DataFrame(
            {
                "Close": [100],
                "High": [101],
                "Low": [99],
                "Volume": [1000000],
                "signal": [0],
            },
            index=[pd.Timestamp("2023-01-01")],
        )

        result = backtester.run_backtest(single_point)
        # Should handle gracefully (either succeed or fail with clear error)
        assert isinstance(result, dict)
        assert "success" in result

    def test_concurrent_backtests(self, backtester, synthetic_data):
        """Test concurrent backtest execution."""
        import threading

        results = []
        errors = []

        def run_backtest():
            try:
                result = backtester.run_backtest(synthetic_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_backtest)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 5
        assert len(errors) == 0

        for result in results:
            assert result["success"] is True
            assert "equity_curve" in result
            assert "metrics" in result


if __name__ == "__main__":
    pytest.main([__file__])
