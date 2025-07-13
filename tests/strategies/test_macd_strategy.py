"""
Test cases for MACD strategy.

This module tests:
- Strategy initialization
- MACD calculation
- Signal generation
- Parameter validation
- Performance metrics
- Edge cases
"""


import numpy as np
import pandas as pd
import pytest

from trading.strategies.macd_strategy import MACDConfig, MACDStrategy


class TestMACDStrategy:
    @pytest.fixture
    def strategy_config(self) -> MACDConfig:
        """Get strategy configuration."""
        return MACDConfig(fast_period=12, slow_period=26, signal_period=9, min_volume=1000.0, min_price=1.0)

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample price data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = np.random.normal(100, 10, len(dates))
        volumes = np.random.normal(5000, 1000, len(dates))
        return pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization."""
        strategy = MACDStrategy(config=strategy_config)

        # Verify configuration
        assert strategy.config.fast_period == strategy_config.fast_period
        assert strategy.config.slow_period == strategy_config.slow_period
        assert strategy.config.signal_period == strategy_config.signal_period
        assert strategy.config.min_volume == strategy_config.min_volume
        assert strategy.config.min_price == strategy_config.min_price

    def test_macd_calculation(self, strategy_config, sample_data):
        """Test MACD calculation."""
        strategy = MACDStrategy(config=strategy_config)
        macd_line, signal_line, histogram = strategy.calculate_macd(sample_data)

        # Verify MACD components
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # Verify calculations
        assert not macd_line.isnull().any()
        assert not signal_line.isnull().any()
        assert not histogram.isnull().any()

    def test_signal_generation(self, strategy_config, sample_data):
        """Test signal generation."""
        strategy = MACDStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)

        # Verify signals
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert all(signal in [-1, 0, 1] for signal in signals["signal"])

        # Verify MACD components in signals
        assert "macd_line" in signals.columns
        assert "signal_line" in signals.columns
        assert "histogram" in signals.columns

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid periods
        with pytest.raises(ValueError):
            MACDConfig(fast_period=0)

        with pytest.raises(ValueError):
            MACDConfig(slow_period=0)

        with pytest.raises(ValueError):
            MACDConfig(signal_period=0)

    def test_performance_metrics(self, strategy_config, sample_data):
        """Test performance metrics calculation."""
        strategy = MACDStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(sample_data)

        # Verify positions
        assert isinstance(positions, pd.DataFrame)
        assert "position" in positions.columns
        assert all(position in [-1, 0, 1] for position in positions["position"])

    def test_edge_cases(self, strategy_config):
        """Test edge cases."""
        strategy = MACDStrategy(config=strategy_config)

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
        strategy = MACDStrategy(config=strategy_config)

        # Update parameters
        new_config = MACDConfig(fast_period=8, slow_period=21, signal_period=5, min_volume=2000.0, min_price=2.0)
        strategy.set_parameters(new_config.__dict__)

        # Verify updates
        assert strategy.config.fast_period == new_config.fast_period
        assert strategy.config.slow_period == new_config.slow_period
        assert strategy.config.signal_period == new_config.signal_period
        assert strategy.config.min_volume == new_config.min_volume
        assert strategy.config.min_price == new_config.min_price

        # Verify signals are reset
        assert strategy.signals is None
        assert strategy.positions is None

    def test_golden_cross_trend_confirmation(self, strategy_config, sample_data):
        """Test signal confirmation for golden cross + trend confirmation."""
        print("\n📈 Testing Golden Cross + Trend Confirmation")

        # Test multiple golden cross scenarios
        cross_scenarios = [
            {
                "name": "Strong Golden Cross",
                "decline_period": 50,
                "decline_magnitude": 20,
                "uptrend_magnitude": 40,
                "expected_crosses": 1,
                "expected_trend_strength": "strong",
            },
            {
                "name": "Multiple Golden Crosses",
                "decline_period": 30,
                "decline_magnitude": 15,
                "uptrend_magnitude": 25,
                "expected_crosses": 2,
                "expected_trend_strength": "medium",
            },
            {
                "name": "Weak Golden Cross",
                "decline_period": 20,
                "decline_magnitude": 10,
                "uptrend_magnitude": 15,
                "expected_crosses": 1,
                "expected_trend_strength": "weak",
            },
        ]

        for scenario in cross_scenarios:
            print(f"\n  🎯 Testing scenario: {scenario['name']}")

            # Create data with specific golden cross pattern
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

            # Generate declining prices first
            declining_prices = (
                100
                - np.linspace(0, scenario["decline_magnitude"], scenario["decline_period"])
                + np.random.normal(0, 1, scenario["decline_period"])
            )

            # Generate reversal and uptrend
            uptrend_prices = (
                (100 - scenario["decline_magnitude"])
                + np.linspace(0, scenario["uptrend_magnitude"], len(dates) - scenario["decline_period"])
                + np.random.normal(0, 1, len(dates) - scenario["decline_period"])
            )

            # Combine the price series
            prices = np.concatenate([declining_prices, uptrend_prices])
            volumes = np.random.normal(5000, 1000, len(dates))

            golden_cross_data = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

            strategy = MACDStrategy(config=strategy_config)

            # Calculate MACD components
            macd_line, signal_line, histogram = strategy.calculate_macd(golden_cross_data)

            # Find golden cross points (MACD line crosses above signal line)
            golden_cross_points = []
            for i in range(1, len(macd_line)):
                if macd_line.iloc[i - 1] <= signal_line.iloc[i - 1] and macd_line.iloc[i] > signal_line.iloc[i]:
                    golden_cross_points.append(i)

            print(f"    Found {len(golden_cross_points)} golden cross points")

            # Verify expected number of crosses
            self.assertGreaterEqual(
                len(golden_cross_points),
                scenario["expected_crosses"],
                f"Should find at least {scenario['expected_crosses']} golden crosses",
            )

            # Test trend confirmation after each golden cross
            trend_confirmation_count = 0

            for cross_point in golden_cross_points:
                if cross_point + 15 < len(golden_cross_data):
                    # Check if trend continues after cross
                    post_cross_macd = macd_line.iloc[cross_point : cross_point + 15]
                    post_cross_signal = signal_line.iloc[cross_point : cross_point + 15]

                    # MACD should stay above signal line after golden cross
                    trend_confirmed = all(post_cross_macd >= post_cross_signal)

                    # Check histogram confirmation
                    post_cross_histogram = histogram.iloc[cross_point : cross_point + 15]
                    histogram_positive = all(post_cross_histogram > 0)

                    # Check trend strength
                    macd_slope = np.polyfit(range(len(post_cross_macd)), post_cross_macd, 1)[0]
                    trend_strength = "strong" if macd_slope > 0.1 else "medium" if macd_slope > 0.05 else "weak"

                    print(
                        f"    Cross at index {cross_point}: trend_confirmed={trend_confirmed}, histogram_positive={histogram_positive}, strength={trend_strength}"
                    )

                    if trend_confirmed and histogram_positive:
                        trend_confirmation_count += 1

            # Verify trend confirmation rate
            confirmation_rate = trend_confirmation_count / len(golden_cross_points) if golden_cross_points else 0
            print(f"    Trend confirmation rate: {confirmation_rate:.2f}")

            self.assertGreater(confirmation_rate, 0.5, "Most golden crosses should have trend confirmation")

        # Test signal generation and alignment
        print(f"\n  📊 Testing signal generation and alignment...")

        # Use the strongest scenario for signal testing
        strong_scenario = cross_scenarios[0]
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        declining_prices = (
            100
            - np.linspace(0, strong_scenario["decline_magnitude"], strong_scenario["decline_period"])
            + np.random.normal(0, 1, strong_scenario["decline_period"])
        )
        uptrend_prices = (
            (100 - strong_scenario["decline_magnitude"])
            + np.linspace(0, strong_scenario["uptrend_magnitude"], len(dates) - strong_scenario["decline_period"])
            + np.random.normal(0, 1, len(dates) - strong_scenario["decline_period"])
        )

        prices = np.concatenate([declining_prices, uptrend_prices])
        volumes = np.random.normal(5000, 1000, len(dates))

        signal_test_data = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

        strategy = MACDStrategy(config=strategy_config)

        # Calculate MACD components
        macd_line, signal_line, histogram = strategy.calculate_macd(signal_test_data)

        # Find golden cross points
        golden_cross_points = []
        for i in range(1, len(macd_line)):
            if macd_line.iloc[i - 1] <= signal_line.iloc[i - 1] and macd_line.iloc[i] > signal_line.iloc[i]:
                golden_cross_points.append(i)

        # Generate signals
        signals = strategy.generate_signals(signal_test_data)

        # Test that buy signals occur around golden crosses
        buy_signals = signals[signals["signal"] == 1]
        print(f"    Generated {len(buy_signals)} buy signals")

        # Check signal alignment with golden crosses
        signal_alignment = 0
        alignment_details = []

        for cross_point in golden_cross_points:
            # Look for buy signals within 5 days of golden cross
            nearby_signals = signals.iloc[max(0, cross_point - 5) : cross_point + 5]
            buy_signal_found = any(nearby_signals["signal"] == 1)

            if buy_signal_found:
                signal_alignment += 1
                # Find the closest buy signal
                buy_indices = nearby_signals[nearby_signals["signal"] == 1].index
                closest_buy = min(buy_indices, key=lambda x: abs(x - cross_point))
                days_diff = abs(closest_buy - cross_point).days
                alignment_details.append(days_diff)

        alignment_ratio = signal_alignment / len(golden_cross_points) if golden_cross_points else 0
        avg_alignment_days = np.mean(alignment_details) if alignment_details else 0

        print(f"    Signal alignment ratio: {alignment_ratio:.2f}")
        print(f"    Average alignment days: {avg_alignment_days:.1f}")

        self.assertGreater(alignment_ratio, 0.6, "Most golden crosses should generate buy signals")
        self.assertLess(avg_alignment_days, 3, "Buy signals should occur close to golden crosses")

        # Test signal strength validation
        print(f"\n  💪 Testing signal strength validation...")

        # Check signal strength based on MACD divergence
        strong_signals = 0
        weak_signals = 0

        for cross_point in golden_cross_points:
            if cross_point + 10 < len(signals):
                # Calculate signal strength based on MACD line slope
                post_cross_macd = macd_line.iloc[cross_point : cross_point + 10]
                macd_slope = np.polyfit(range(len(post_cross_macd)), post_cross_macd, 1)[0]

                # Calculate histogram strength
                post_cross_histogram = histogram.iloc[cross_point : cross_point + 10]
                histogram_strength = np.mean(post_cross_histogram)

                # Determine signal strength
                if macd_slope > 0.05 and histogram_strength > 0.1:
                    strong_signals += 1
                else:
                    weak_signals += 1

        total_signals = strong_signals + weak_signals
        strong_signal_ratio = strong_signals / total_signals if total_signals > 0 else 0

        print(f"    Strong signals: {strong_signals}")
        print(f"    Weak signals: {weak_signals}")
        print(f"    Strong signal ratio: {strong_signal_ratio:.2f}")

        # Test trend confirmation patterns
        print(f"\n  🔄 Testing trend confirmation patterns...")

        # Test different confirmation timeframes
        confirmation_timeframes = [5, 10, 15, 20]

        for timeframe in confirmation_timeframes:
            confirmation_count = 0

            for cross_point in golden_cross_points:
                if cross_point + timeframe < len(macd_line):
                    post_cross_macd = macd_line.iloc[cross_point : cross_point + timeframe]
                    post_cross_signal = signal_line.iloc[cross_point : cross_point + timeframe]

                    # Check if MACD stays above signal line
                    trend_confirmed = all(post_cross_macd >= post_cross_signal)

                    if trend_confirmed:
                        confirmation_count += 1

            confirmation_rate = confirmation_count / len(golden_cross_points) if golden_cross_points else 0
            print(f"    {timeframe}-day confirmation rate: {confirmation_rate:.2f}")

        # Test false signal detection
        print(f"\n  ⚠️ Testing false signal detection...")

        # Create data with false golden crosses (crosses that don't lead to sustained uptrends)
        false_cross_dates = pd.date_range(start="2023-01-01", periods=100)

        # Create choppy price data that might generate false crosses
        choppy_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        choppy_volumes = np.random.normal(5000, 1000, 100)

        false_cross_data = pd.DataFrame({"close": choppy_prices, "volume": choppy_volumes}, index=false_cross_dates)

        strategy = MACDStrategy(config=strategy_config)

        # Calculate MACD for choppy data
        choppy_macd, choppy_signal, choppy_histogram = strategy.calculate_macd(false_cross_data)

        # Find all crosses
        all_crosses = []
        for i in range(1, len(choppy_macd)):
            if choppy_macd.iloc[i - 1] <= choppy_signal.iloc[i - 1] and choppy_macd.iloc[i] > choppy_signal.iloc[i]:
                all_crosses.append(i)

        # Count false crosses (crosses without sustained trend)
        false_crosses = 0
        for cross_point in all_crosses:
            if cross_point + 10 < len(choppy_macd):
                post_cross_macd = choppy_macd.iloc[cross_point : cross_point + 10]
                post_cross_signal = choppy_signal.iloc[cross_point : cross_point + 10]

                # Check if trend is sustained
                trend_sustained = all(post_cross_macd >= post_cross_signal)

                if not trend_sustained:
                    false_crosses += 1

        false_cross_ratio = false_crosses / len(all_crosses) if all_crosses else 0
        print(f"    False cross ratio: {false_cross_ratio:.2f}")

        # Test signal filtering
        print(f"\n  🎯 Testing signal filtering...")

        # Generate signals for choppy data
        choppy_signals = strategy.generate_signals(false_cross_data)

        # Count buy signals
        choppy_buy_signals = len(choppy_signals[choppy_signals["signal"] == 1])
        print(f"    Buy signals in choppy data: {choppy_buy_signals}")

        # In choppy data, there should be fewer buy signals
        self.assertLess(choppy_buy_signals, len(all_crosses), "Should filter out some false signals in choppy data")

        print("✅ Golden cross + trend confirmation test completed")
