"""Tests for the backtester."""

import numpy as np
import pandas as pd
import pytest

from trading.backtesting.backtester import Backtester


class TestBacktester:
    @pytest.fixture
    def backtester(self):
        """Create a Backtester instance for testing."""
        return Backtester(initial_balance=10000, commission=0.001, slippage=0.0005)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    @pytest.fixture
    def sample_signals(self, sample_data):
        """Create sample trading signals for testing."""
        signals = pd.Series(0, index=sample_data.index)
        signals.iloc[10:20] = 1  # Buy signals
        signals.iloc[30:40] = -1  # Sell signals
        return signals

    def test_backtester_initialization(self, backtester):
        """Test that backtester initializes with correct parameters."""
        assert backtester.initial_balance == 10000
        assert backtester.commission == 0.001
        assert backtester.slippage == 0.0005
        assert backtester.name == "Backtester"

    def test_position_sizing(self, backtester, sample_data):
        """Test that position sizing works correctly."""
        price = sample_data["close"].iloc[0]
        position = backtester.calculate_position_size(price)

        assert isinstance(position, int)
        assert position > 0
        assert position * price <= backtester.initial_balance

    def test_trade_execution(self, backtester, sample_data, sample_signals):
        """Test that trades are executed correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)

        assert isinstance(trades, pd.DataFrame)
        assert "entry_price" in trades.columns
        assert "exit_price" in trades.columns
        assert "position" in trades.columns
        assert "pnl" in trades.columns
        assert len(trades) > 0

    def test_commission_calculation(self, backtester, sample_data, sample_signals):
        """Test that commissions are calculated correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)

        assert "commission" in trades.columns
        assert (trades["commission"] >= 0).all()
        assert (
            trades["commission"]
            == trades["position"].abs() * sample_data["close"] * backtester.commission
        ).all()

    def test_slippage_application(self, backtester, sample_data, sample_signals):
        """Test that slippage is applied correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)

        assert "slippage" in trades.columns
        assert (trades["slippage"] >= 0).all()
        assert (
            trades["slippage"]
            == trades["position"].abs() * sample_data["close"] * backtester.slippage
        ).all()

    def test_performance_calculation(self, backtester, sample_data, sample_signals):
        """Test that performance metrics are calculated correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)
        performance = backtester.calculate_performance(trades)

        assert isinstance(performance, dict)
        assert "total_return" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
        assert "win_rate" in performance
        assert all(isinstance(v, float) for v in performance.values())

    def test_equity_curve(self, backtester, sample_data, sample_signals):
        """Test that equity curve is calculated correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)
        equity_curve = backtester.calculate_equity_curve(trades)

        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == len(sample_data)
        assert (equity_curve >= 0).all()
        assert equity_curve.iloc[0] == backtester.initial_balance

    def test_risk_metrics(self, backtester, sample_data, sample_signals):
        """Test that risk metrics are calculated correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)
        risk_metrics = backtester.calculate_risk_metrics(trades)

        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "var" in risk_metrics
        assert "cvar" in risk_metrics
        assert all(isinstance(v, float) for v in risk_metrics.values())

    def test_trade_statistics(self, backtester, sample_data, sample_signals):
        """Test that trade statistics are calculated correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)
        stats = backtester.calculate_trade_statistics(trades)

        assert isinstance(stats, dict)
        assert "total_trades" in stats
        assert "winning_trades" in stats
        assert "losing_trades" in stats
        assert "avg_trade" in stats
        assert all(isinstance(v, (int, float)) for v in stats.values())

    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        with pytest.raises(ValueError):
            Backtester(initial_balance=0)  # Invalid initial balance
        with pytest.raises(ValueError):
            Backtester(commission=-0.001)  # Invalid commission
        with pytest.raises(ValueError):
            Backtester(slippage=-0.0005)  # Invalid slippage

    def test_empty_data_handling(self, backtester):
        """Test that empty data is handled correctly."""
        empty_data = pd.DataFrame()
        empty_signals = pd.Series()
        with pytest.raises(ValueError):
            backtester.execute_trades(empty_data, empty_signals)

    def test_missing_data_handling(self, backtester):
        """Test that missing data is handled correctly."""
        data = pd.DataFrame({"close": [100, np.nan, 101]})
        signals = pd.Series([1, 0, -1])
        with pytest.raises(ValueError):
            backtester.execute_trades(data, signals)

    def test_signal_validation(self, backtester, sample_data):
        """Test that signals are validated correctly."""
        invalid_signals = pd.Series([2, 0, -2], index=sample_data.index)
        with pytest.raises(ValueError):
            backtester.execute_trades(sample_data, invalid_signals)

    def test_position_limits(self, backtester, sample_data):
        """Test that position limits are enforced."""
        # Create signals that would exceed position limits
        signals = pd.Series(1, index=sample_data.index)
        trades = backtester.execute_trades(sample_data, signals)

        assert (
            trades["position"] * sample_data["close"] <= backtester.initial_balance
        ).all()

    def test_trade_sequencing(self, backtester, sample_data, sample_signals):
        """Test that trades are sequenced correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)

        # Check that buy signals precede sell signals
        buy_times = trades[trades["position"] > 0].index
        sell_times = trades[trades["position"] < 0].index

        if len(buy_times) > 0 and len(sell_times) > 0:
            assert buy_times[0] < sell_times[0]

    def test_performance_attribution(self, backtester, sample_data, sample_signals):
        """Test that performance attribution works correctly."""
        trades = backtester.execute_trades(sample_data, sample_signals)
        attribution = backtester.attribute_performance(trades)

        assert isinstance(attribution, dict)
        assert "pnl" in attribution
        assert "commission" in attribution
        assert "slippage" in attribution
        assert all(isinstance(v, float) for v in attribution.values())
