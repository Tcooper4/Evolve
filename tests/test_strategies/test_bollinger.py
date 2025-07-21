"""Tests for the Bollinger Bands strategy using pandas-ta."""

import numpy as np
import pandas as pd
import pytest

from trading.strategies.bollinger_strategy import BollingerStrategy


class TestBollingerStrategy:
    """Test Bollinger Bands strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create a Bollinger Bands strategy instance for testing."""
        return BollingerStrategy(window=20, num_std=2)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_bollinger_bands_calculation_with_pandas_ta(self, strategy, sample_data):
        """Test that Bollinger Bands are calculated correctly using pandas-ta."""
        # Calculate Bollinger Bands
        bb_result = strategy.calculate_bollinger_bands(sample_data["Close"])

        # Check that Bollinger Bands components exist
        assert "BBL_20_2.0" in bb_result.columns  # Lower band
        assert "BBM_20_2.0" in bb_result.columns  # Middle band (SMA)
        assert "BBU_20_2.0" in bb_result.columns  # Upper band
        assert "BBB_20_2.0" in bb_result.columns  # Bandwidth
        assert "BBP_20_2.0" in bb_result.columns  # Percentage B

        # Check data types and lengths
        assert isinstance(bb_result, pd.DataFrame)
        assert len(bb_result) == len(sample_data)

        # Middle band should not be all NaN
        middle_band = bb_result["BBM_20_2.0"].dropna()
        assert len(middle_band) > 0

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)

        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = signals.dropna()
        assert valid_signals.isin([1, 0, -1]).all()

    def test_buy_signal_generation(self, strategy):
        """Test that price touching lower band triggers buy signal."""
        # Create data that will generate buy signals
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create downward trend to touch lower band
        prices = np.linspace(120, 80, 50)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one buy signal
        assert (signals == 1).any(), "No buy signals generated for lower band touch"

    def test_sell_signal_generation(self, strategy):
        """Test that price touching upper band triggers sell signal."""
        # Create data that will generate sell signals
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create upward trend to touch upper band
        prices = np.linspace(80, 120, 50)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one sell signal
        assert (signals == -1).any(), "No sell signals generated for upper band touch"

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        # Test invalid window
        with pytest.raises(ValueError):
            BollingerStrategy(window=0, num_std=2)

        # Test invalid standard deviation
        with pytest.raises(ValueError):
            BollingerStrategy(window=20, num_std=0)

        # Test negative standard deviation
        with pytest.raises(ValueError):
            BollingerStrategy(window=20, num_std=-1)

    def test_empty_data_handling(self, strategy):
        """Test that strategy handles empty data correctly."""
        empty_data = pd.DataFrame(columns=["Close"])

        with pytest.raises(ValueError):
            strategy.generate_signals(empty_data)

    def test_missing_data_handling(self, strategy):
        """Test that strategy handles missing data correctly."""
        data = pd.DataFrame({"Close": [100, np.nan, 101, 102]})

        with pytest.raises(ValueError):
            strategy.generate_signals(data)

    def test_signal_consistency(self, strategy, sample_data):
        """Test that signals are consistent with Bollinger Bands values."""
        bb_result = strategy.calculate_bollinger_bands(sample_data["Close"])
        signals = strategy.generate_signals(sample_data)

        # Get non-NaN values
        mask = (
            bb_result["BBL_20_2.0"].notna()
            & bb_result["BBU_20_2.0"].notna()
            & signals.notna()
        )
        lower_band = bb_result.loc[mask, "BBL_20_2.0"]
        upper_band = bb_result.loc[mask, "BBU_20_2.0"]
        prices = sample_data.loc[mask, "Close"]
        signal_values = signals[mask]

        # Check that price touching lower band generates buy signals
        lower_touch_mask = prices <= lower_band * 1.01  # Within 1% of lower band
        if lower_touch_mask.any():
            assert (
                signal_values[lower_touch_mask] == 1
            ).all(), "Lower band touch should generate buy signals"

        # Check that price touching upper band generates sell signals
        upper_touch_mask = prices >= upper_band * 0.99  # Within 1% of upper band
        if upper_touch_mask.any():
            assert (
                signal_values[upper_touch_mask] == -1
            ).all(), "Upper band touch should generate sell signals"

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.window == 20
        assert strategy.num_std == 2
        assert strategy.name == "Bollinger Bands"

    def test_different_parameters(self, sample_data):
        """Test Bollinger Bands with different parameter combinations."""
        # Test with different window and std
        strategy1 = BollingerStrategy(window=10, num_std=1.5)
        strategy2 = BollingerStrategy(window=30, num_std=2.5)

        signals1 = strategy1.generate_signals(sample_data)
        signals2 = strategy2.generate_signals(sample_data)

        # Should have different signal patterns
        assert not signals1.equals(signals2)

    def test_bollinger_bands_properties(self, strategy, sample_data):
        """Test that Bollinger Bands have correct mathematical properties."""
        bb_result = strategy.calculate_bollinger_bands(sample_data["Close"])

        # Get non-NaN values
        mask = (
            bb_result["BBL_20_2.0"].notna()
            & bb_result["BBM_20_2.0"].notna()
            & bb_result["BBU_20_2.0"].notna()
        )
        lower_band = bb_result.loc[mask, "BBL_20_2.0"]
        middle_band = bb_result.loc[mask, "BBM_20_2.0"]
        upper_band = bb_result.loc[mask, "BBU_20_2.0"]

        # Lower band should be less than middle band
        assert (lower_band < middle_band).all()

        # Upper band should be greater than middle band
        assert (upper_band > middle_band).all()

        # Bands should be symmetric around middle band
        lower_distance = middle_band - lower_band
        upper_distance = upper_band - middle_band
        np.testing.assert_array_almost_equal(lower_distance, upper_distance, decimal=10)

    def test_percentage_b_calculation(self, strategy, sample_data):
        """Test that Percentage B is calculated correctly."""
        bb_result = strategy.calculate_bollinger_bands(sample_data["Close"])

        # Get non-NaN values
        mask = bb_result["BBP_20_2.0"].notna()
        percentage_b = bb_result.loc[mask, "BBP_20_2.0"]
        prices = sample_data.loc[mask, "Close"]
        lower_band = bb_result.loc[mask, "BBL_20_2.0"]
        upper_band = bb_result.loc[mask, "BBU_20_2.0"]

        # Percentage B should be between 0 and 1
        assert (percentage_b >= 0).all() and (percentage_b <= 1).all()

        # Manual calculation of Percentage B
        expected_pb = (prices - lower_band) / (upper_band - lower_band)
        np.testing.assert_array_almost_equal(percentage_b, expected_pb, decimal=10)

    def test_bandwidth_calculation(self, strategy, sample_data):
        """Test that Bandwidth is calculated correctly."""
        bb_result = strategy.calculate_bollinger_bands(sample_data["Close"])

        # Get non-NaN values
        mask = bb_result["BBB_20_2.0"].notna()
        bandwidth = bb_result.loc[mask, "BBB_20_2.0"]
        middle_band = bb_result.loc[mask, "BBM_20_2.0"]
        lower_band = bb_result.loc[mask, "BBL_20_2.0"]
        upper_band = bb_result.loc[mask, "BBU_20_2.0"]

        # Bandwidth should be positive
        assert (bandwidth > 0).all()

        # Manual calculation of Bandwidth
        expected_bandwidth = (upper_band - lower_band) / middle_band
        np.testing.assert_array_almost_equal(bandwidth, expected_bandwidth, decimal=10)

    def test_performance_evaluation(self, strategy, sample_data):
        """Test that the strategy can be evaluated for performance."""
        signals = strategy.generate_signals(sample_data)
        performance = strategy.evaluate_performance(sample_data, signals)

        # Check performance metrics
        assert isinstance(performance, dict)
        required_metrics = [
            "returns",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
        ]

        for metric in required_metrics:
            assert metric in performance
            assert isinstance(performance[metric], (int, float))

    def test_signal_timing(self, strategy, sample_data):
        """Test that signals are generated at the correct times."""
        signals = strategy.generate_signals(sample_data)

        # Signals should be generated after enough data is available
        # Bollinger Bands require window data points
        min_required = strategy.window

        # First signals should be NaN due to insufficient data
        assert signals.iloc[:min_required].isna().all()

        # Should have valid signals after minimum required data
        assert not signals.iloc[min_required:].isna().all()

    def test_edge_cases(self, strategy):
        """Test edge cases for Bollinger Bands calculation."""
        # Test with very short data
        short_data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError):
            strategy.generate_signals(short_data)

        # Test with constant price
        constant_data = pd.DataFrame({"Close": [100] * 30})

        signals = strategy.generate_signals(constant_data)
        assert isinstance(signals, pd.Series)

    def test_volatility_impact(self, strategy, sample_data):
        """Test that volatility affects Bollinger Bands width."""
        # Create high volatility data
        high_vol_data = sample_data.copy()
        high_vol_data["Close"] = high_vol_data["Close"] + np.random.normal(
            0, 5, len(high_vol_data)
        )

        # Create low volatility data
        low_vol_data = sample_data.copy()
        low_vol_data["Close"] = low_vol_data["Close"] + np.random.normal(
            0, 0.5, len(low_vol_data)
        )

        bb_high_vol = strategy.calculate_bollinger_bands(high_vol_data["Close"])
        bb_low_vol = strategy.calculate_bollinger_bands(low_vol_data["Close"])

        # High volatility should result in wider bands
        high_vol_bandwidth = bb_high_vol["BBB_20_2.0"].mean()
        low_vol_bandwidth = bb_low_vol["BBB_20_2.0"].mean()

        assert high_vol_bandwidth > low_vol_bandwidth

    def test_signal_distribution(self, strategy, sample_data):
        """Test that signals have reasonable distribution."""
        signals = strategy.generate_signals(sample_data)
        valid_signals = signals.dropna()

        if len(valid_signals) > 0:
            # Should have a mix of buy, sell, and hold signals
            signal_counts = valid_signals.value_counts()

            # Should have at least some hold signals (0)
            assert 0 in signal_counts.index

            # Total signals should equal valid signal count
            assert signal_counts.sum() == len(valid_signals)
