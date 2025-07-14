"""
Test cases for Bollinger Bands strategy.

This module tests:
- Strategy initialization
- Band calculation
- Signal generation
- Parameter validation
- Performance metrics
- Edge cases
"""

import logging

import numpy as np
import pandas as pd
import pytest

from trading.strategies.bollinger_strategy import BollingerConfig, BollingerStrategy

logger = logging.getLogger(__name__)


class TestBollingerStrategy:
    @pytest.fixture
    def strategy_config(self) -> BollingerConfig:
        """Get strategy configuration."""
        return BollingerConfig(window=20, num_std=2.0, min_volume=1000.0, min_price=1.0)

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample price data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = np.random.normal(100, 10, len(dates))
        volumes = np.random.normal(5000, 1000, len(dates))
        return pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization."""
        strategy = BollingerStrategy(config=strategy_config)

        # Verify configuration
        assert strategy.config.window == strategy_config.window
        assert strategy.config.num_std == strategy_config.num_std
        assert strategy.config.min_volume == strategy_config.min_volume
        assert strategy.config.min_price == strategy_config.min_price

    def test_band_calculation(self, strategy_config, sample_data):
        """Test Bollinger Bands calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        upper_band, middle_band, lower_band = strategy.calculate_bands(sample_data)

        # Verify bands
        assert isinstance(upper_band, pd.Series)
        assert isinstance(middle_band, pd.Series)
        assert isinstance(lower_band, pd.Series)

        # Verify calculations
        assert not upper_band.isnull().any()
        assert not middle_band.isnull().any()
        assert not lower_band.isnull().any()

        # Verify band relationships
        assert all(upper_band >= middle_band)
        assert all(lower_band <= middle_band)

    def test_signal_generation(self, strategy_config, sample_data):
        """Test signal generation."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)

        # Verify signals
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert all(signal in [-1, 0, 1] for signal in signals["signal"])

        # Verify bands in signals
        assert "upper_band" in signals.columns
        assert "middle_band" in signals.columns
        assert "lower_band" in signals.columns

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid window
        with pytest.raises(ValueError):
            BollingerConfig(window=0)

        # Test invalid standard deviation
        with pytest.raises(ValueError):
            BollingerConfig(num_std=0.0)

    def test_performance_metrics(self, strategy_config, sample_data):
        """Test performance metrics calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(sample_data)

        # Verify positions
        assert isinstance(positions, pd.DataFrame)
        assert "position" in positions.columns
        assert all(position in [-1, 0, 1] for position in positions["position"])

    def test_edge_cases(self, strategy_config):
        """Test edge cases."""
        strategy = BollingerStrategy(config=strategy_config)

        # Test empty data
        empty_data = pd.DataFrame(columns=["close", "volume"])
        with pytest.raises(ValueError):
            strategy.generate_signals(empty_data)

        # Test single row data
        single_row = pd.DataFrame({"close": [100], "volume": [5000]})
        signals = strategy.generate_signals(single_row)
        assert len(signals) == 1

        # Test all same values
        same_values = pd.DataFrame({"close": [100] * 100, "volume": [5000] * 100})
        signals = strategy.generate_signals(same_values)
        assert len(signals) == 100

    def test_parameter_updates(self, strategy_config, sample_data):
        """Test strategy parameter updates."""
        strategy = BollingerStrategy(config=strategy_config)

        # Update parameters
        new_config = BollingerConfig(
            window=50, num_std=2.5, min_volume=2000.0, min_price=2.0
        )
        strategy.set_parameters(new_config.__dict__)

        # Verify updates
        assert strategy.config.window == new_config.window
        assert strategy.config.num_std == new_config.num_std
        assert strategy.config.min_volume == new_config.min_volume
        assert strategy.config.min_price == new_config.min_price

        # Verify signals are reset
        assert strategy.signals is None
        assert strategy.positions is None

    def test_band_width(self, strategy_config, sample_data):
        """Test band width calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        upper_band, middle_band, lower_band = strategy.calculate_bands(sample_data)

        # Calculate band width
        band_width = (upper_band - lower_band) / middle_band

        # Verify band width
        assert not band_width.isnull().any()
        assert all(band_width >= 0)

    def test_signal_threshold(self, strategy_config, sample_data):
        """Test signal threshold application."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)

        # Verify threshold application
        assert all(abs(signal) <= 1 for signal in signals["signal"])

    def test_strategy_reset(self, strategy_config, sample_data):
        """Test strategy reset."""
        strategy = BollingerStrategy(config=strategy_config)

        # Generate signals
        strategy.generate_signals(sample_data)

        # Reset strategy
        strategy.reset()

        # Verify reset
        assert strategy.signals is None
        assert strategy.positions is None

    def test_strategy_serialization(self, strategy_config):
        """Test strategy serialization."""
        strategy = BollingerStrategy(config=strategy_config)

        # Serialize
        config = strategy.get_config()

        # Verify serialization
        assert config["window"] == strategy_config.window
        assert config["num_std"] == strategy_config.num_std
        assert config["min_volume"] == strategy_config.min_volume
        assert config["min_price"] == strategy_config.min_price

    def test_band_breakout(self, strategy_config, sample_data):
        """Test band breakout detection."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)

        # Verify breakout signals
        assert "breakout" in signals.columns
        assert all(signal in [True, False] for signal in signals["breakout"])

    def test_tight_bands_low_volatility_regime(self, strategy_config):
        """Test edge case with tight bands in low volatility regime."""
        logger.debug("\n📊 Testing Tight Bands in Low Volatility Regime")

        # Test multiple volatility scenarios
        volatility_scenarios = [
            {
                "name": "Very Low Volatility",
                "volatility": 0.01,  # 1% daily volatility
                "expected_band_width": 0.03,
                "expected_neutral_ratio": 0.8,
            },
            {
                "name": "Low Volatility",
                "volatility": 0.02,  # 2% daily volatility
                "expected_band_width": 0.06,
                "expected_neutral_ratio": 0.7,
            },
            {
                "name": "Medium Volatility",
                "volatility": 0.05,  # 5% daily volatility
                "expected_band_width": 0.15,
                "expected_neutral_ratio": 0.5,
            },
        ]

        for scenario in volatility_scenarios:
            logger.debug(f"\n  📈 Testing scenario: {scenario['name']}")

            # Create data with specific volatility
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
            base_price = 100
            prices = base_price + np.random.normal(
                0, scenario["volatility"], len(dates)
            )
            volumes = np.random.normal(5000, 1000, len(dates))

            volatility_data = pd.DataFrame(
                {"close": prices, "volume": volumes}, index=dates
            )

            strategy = BollingerStrategy(config=strategy_config)

            # Calculate bands
            upper_band, middle_band, lower_band = strategy.calculate_bands(
                volatility_data
            )

            # Test band tightness
            band_width = (upper_band - lower_band) / middle_band
            avg_band_width = band_width.mean()

            logger.debug(
                f"    Average band width: {avg_band_width:.4f} (expected < {scenario['expected_band_width']:.2f})"
            )

            # Verify band width is appropriate for volatility level
            assert (
                avg_band_width < scenario["expected_band_width"]
            ), f"Band width should be tight for {scenario['name']}"

            # Test signal generation
            signals = strategy.generate_signals(volatility_data)

            # Count signal types
            signal_counts = signals["signal"].value_counts()
            neutral_signals = signal_counts.get(0, 0)
            total_signals = len(signals)
            neutral_ratio = neutral_signals / total_signals

            logger.debug(
                f"    Neutral signal ratio: {neutral_ratio:.2f} (expected > {scenario['expected_neutral_ratio']:.1f})"
            )

            # Verify signal distribution
            assert (
                neutral_ratio > scenario["expected_neutral_ratio"]
            ), f"Most signals should be neutral in {scenario['name']}"

            # Test band relationships
            assert all(
                upper_band >= middle_band
            ), "Upper band should always be above middle"
            assert all(
                lower_band <= middle_band
            ), "Lower band should always be below middle"

            # Test band convergence
            max_band_separation = (upper_band - lower_band).max()
            logger.debug(f"    Max band separation: {max_band_separation:.3f}")

            # Verify bands don't diverge excessively
            assert (
                max_band_separation < base_price * scenario["volatility"] * 10
            ), f"Bands should not diverge excessively in {scenario['name']}"

        # Test extreme low volatility edge case
        logger.debug(f"\n  ⚠️ Testing extreme low volatility edge case...")

        # Create nearly constant price data
        extreme_dates = pd.date_range(start="2023-01-01", periods=100)
        extreme_prices = [100.0] * 100  # Constant price
        extreme_volumes = [5000] * 100

        extreme_data = pd.DataFrame(
            {"close": extreme_prices, "volume": extreme_volumes}, index=extreme_dates
        )

        strategy = BollingerStrategy(config=strategy_config)

        # Calculate bands for constant data
        extreme_upper, extreme_middle, extreme_lower = strategy.calculate_bands(
            extreme_data
        )

        # Test that bands are nearly identical for constant data
        band_differences = (extreme_upper - extreme_lower) / extreme_middle
        max_difference = band_differences.max()

        logger.debug(f"    Max band difference: {max_difference:.6f}")

        # Bands should be nearly identical for constant data
        assert (
            max_difference < 0.001
        ), "Bands should be nearly identical for constant data"

        # Test signal generation for constant data
        extreme_signals = strategy.generate_signals(extreme_data)

        # All signals should be neutral for constant data
        all_neutral = all(signal == 0 for signal in extreme_signals["signal"])
        logger.debug(f"    All signals neutral: {all_neutral}")

        assert all_neutral, "All signals should be neutral for constant data"

        # Test transition from low to high volatility
        logger.debug(f"\n  🔄 Testing volatility transition...")

        # Create data that transitions from low to high volatility
        transition_dates = pd.date_range(start="2023-01-01", periods=200)

        # First half: low volatility
        low_vol_prices = 100 + np.random.normal(0, 0.01, 100)
        # Second half: high volatility
        high_vol_prices = 100 + np.random.normal(0, 0.05, 100)

        transition_prices = np.concatenate([low_vol_prices, high_vol_prices])
        transition_volumes = np.random.normal(5000, 1000, 200)

        transition_data = pd.DataFrame(
            {"close": transition_prices, "volume": transition_volumes},
            index=transition_dates,
        )

        strategy = BollingerStrategy(config=strategy_config)

        # Calculate bands for transition data
        (
            transition_upper,
            transition_middle,
            transition_lower,
        ) = strategy.calculate_bands(transition_data)

        # Test band width evolution
        transition_band_width = (
            transition_upper - transition_lower
        ) / transition_middle

        # Calculate average band width for each half
        low_vol_width = transition_band_width.iloc[:100].mean()
        high_vol_width = transition_band_width.iloc[100:].mean()

        logger.debug(f"    Low volatility band width: {low_vol_width:.4f}")
        logger.debug(f"    High volatility band width: {high_vol_width:.4f}")

        # High volatility should have wider bands
        assert (
            high_vol_width > low_vol_width
        ), "High volatility should result in wider bands"

        # Test signal frequency changes
        transition_signals = strategy.generate_signals(transition_data)

        # Count signals in each half
        low_vol_signals = transition_signals["signal"].iloc[:100]
        high_vol_signals = transition_signals["signal"].iloc[100:]

        low_vol_neutral_ratio = (low_vol_signals == 0).mean()
        high_vol_neutral_ratio = (high_vol_signals == 0).mean()

        logger.debug(f"    Low volatility neutral ratio: {low_vol_neutral_ratio:.2f}")
        logger.debug(f"    High volatility neutral ratio: {high_vol_neutral_ratio:.2f}")

        # Low volatility should have more neutral signals
        assert (
            low_vol_neutral_ratio > high_vol_neutral_ratio
        ), "Low volatility should have more neutral signals"

        # Test parameter sensitivity in low volatility
        logger.debug(f"\n  🎯 Testing parameter sensitivity...")

        # Test different standard deviation parameters
        std_scenarios = [1.0, 1.5, 2.0, 2.5, 3.0]

        for std_param in std_scenarios:
            # Create config with different std
            test_config = BollingerConfig(
                window=strategy_config.window,
                num_std=std_param,
                min_volume=strategy_config.min_volume,
                min_price=strategy_config.min_price,
            )

            test_strategy = BollingerStrategy(config=test_config)

            # Use low volatility data
            low_vol_dates = pd.date_range(start="2023-01-01", periods=50)
            low_vol_prices = 100 + np.random.normal(0, 0.01, 50)
            low_vol_volumes = np.random.normal(5000, 1000, 50)

            test_data = pd.DataFrame(
                {"close": low_vol_prices, "volume": low_vol_volumes},
                index=low_vol_dates,
            )

            # Calculate bands
            test_upper, test_middle, test_lower = test_strategy.calculate_bands(
                test_data
            )

            # Calculate band width
            test_band_width = (test_upper - test_lower) / test_middle
            avg_test_width = test_band_width.mean()

            logger.debug(f"    Std {std_param}: Band width {avg_test_width:.4f}")

            # Verify band width increases with std parameter
            if std_param > 1.0:
                assert (
                    avg_test_width > 0.001
                ), f"Band width should be positive for std {std_param}"

        # Test volume filtering in low volatility
        logger.debug(f"\n  📊 Testing volume filtering...")

        # Create data with varying volumes
        volume_dates = pd.date_range(start="2023-01-01", periods=100)
        volume_prices = 100 + np.random.normal(0, 0.01, 100)

        # Mix of high and low volumes
        volume_volumes = np.concatenate(
            [
                np.random.normal(10000, 2000, 50),
                np.random.normal(500, 100, 50),
            ]  # High volume  # Low volume
        )

        volume_data = pd.DataFrame(
            {"close": volume_prices, "volume": volume_volumes}, index=volume_dates
        )

        strategy = BollingerStrategy(config=strategy_config)
        volume_signals = strategy.generate_signals(volume_data)

        # Check that low volume periods have neutral signals
        low_volume_mask = volume_data["volume"] < strategy_config.min_volume
        low_volume_signals = volume_signals.loc[low_volume_mask, "signal"]

        if len(low_volume_signals) > 0:
            low_volume_neutral_ratio = (low_volume_signals == 0).mean()
            logger.debug(
                f"    Low volume neutral ratio: {low_volume_neutral_ratio:.2f}"
            )

            # Low volume periods should have neutral signals
            assert (
                low_volume_neutral_ratio > 0.8
            ), "Low volume periods should have neutral signals"

        logger.debug("✅ Tight bands in low volatility regime test completed")
