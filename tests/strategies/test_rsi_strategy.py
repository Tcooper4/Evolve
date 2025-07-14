"""Tests for RSI strategy."""

import logging
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trading.optimization.rsi_optimizer import RSIOptimizer
from trading.strategies.rsi_signals import generate_rsi_signals, load_optimized_settings

logger = logging.getLogger(__name__)


class TestRSIStrategy(unittest.TestCase):
    """Test cases for RSI strategy."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        cls.data = pd.DataFrame(
            {
                "open": np.random.normal(100, 1, len(dates)),
                "high": np.random.normal(101, 1, len(dates)),
                "low": np.random.normal(99, 1, len(dates)),
                "close": np.random.normal(100, 1, len(dates)),
                "volume": np.random.normal(1000000, 100000, len(dates)),
            },
            index=dates,
        )

        # Create optimizer instance
        cls.optimizer = RSIOptimizer(cls.data)

    def test_generate_rsi_signals(self):
        """Test RSI signal generation."""
        # Test with default parameters
        result = generate_rsi_signals(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("signal", result.columns)
        self.assertIn("returns", result.columns)
        self.assertIn("strategy_returns", result.columns)

        # Test with custom parameters
        result = generate_rsi_signals(
            self.data, period=10, buy_threshold=20, sell_threshold=80
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("signal", result.columns)

    def test_load_optimized_settings(self):
        """Test loading optimized settings."""
        # Test with non-existent ticker
        settings = load_optimized_settings("NONEXISTENT")
        self.assertIsNone(settings)

    def test_rsi_optimizer(self):
        """Test RSI optimizer."""
        # Test parameter optimization
        result = self.optimizer.optimize(n_trials=5)
        logger.info(
            f"Best params: period={result.period}, overbought={result.overbought}, oversold={result.oversold}, Sharpe={result.sharpe_ratio:.3f}"
        )
        self.assertIsInstance(result.period, int)
        self.assertIsInstance(result.overbought, float)
        self.assertIsInstance(result.oversold, float)
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("win_rate", result.metrics)
        self.assertGreaterEqual(result.sharpe_ratio, -10)  # sanity check
        self.assertLessEqual(result.sharpe_ratio, 10)  # sanity check

    def test_optimizer_visualization(self):
        """Test optimizer visualization."""
        # Get optimization results
        results = self.optimizer.optimize_rsi_parameters(objective="sharpe", n_top=1)
        result = results[0]

        # Test equity curve plot
        fig = self.optimizer.plot_equity_curve(result)
        self.assertIsInstance(fig, go.Figure)

        # Test drawdown plot
        fig = self.optimizer.plot_drawdown(result)
        self.assertIsInstance(fig, go.Figure)

        # Test signals plot
        fig = self.optimizer.plot_signals(result)
        self.assertIsInstance(fig, go.Figure)

    def test_oversold_overbought_threshold_variation_intraday(self):
        """Test oversold/overbought threshold variation with intraday data."""
        print("\n📊 Testing Oversold/Overbought Threshold Variation with Intraday Data")

        # Test multiple intraday timeframes
        timeframes = [
            {"freq": "H", "name": "Hourly", "periods": 24},
            {"freq": "30T", "name": "30-Minute", "periods": 48},
            {"freq": "15T", "name": "15-Minute", "periods": 96},
        ]

        for timeframe in timeframes:
            print(f"\n  ⏰ Testing {timeframe['name']} timeframe...")

            # Create intraday data with specific timeframe
            dates = pd.date_range(
                start="2023-01-01", end="2023-01-31", freq=timeframe["freq"]
            )

            # Create price data with clear oversold/overbought cycles
            base_price = 100
            prices = []

            for i in range(len(dates)):
                # Create cycles: oversold -> recovery -> overbought -> decline
                cycle_position = (i % timeframe["periods"]) / timeframe["periods"]

                if cycle_position < 0.25:  # Oversold period
                    price = base_price - 5 + np.random.normal(0, 0.5)
                elif cycle_position < 0.5:  # Recovery period
                    price = base_price - 2 + np.random.normal(0, 0.5)
                elif cycle_position < 0.75:  # Overbought period
                    price = base_price + 5 + np.random.normal(0, 0.5)
                else:  # Decline period
                    price = base_price + 2 + np.random.normal(0, 0.5)

                prices.append(price)

            intraday_data = pd.DataFrame(
                {
                    "open": prices,
                    "high": [p + abs(np.random.normal(0, 0.2)) for p in prices],
                    "low": [p - abs(np.random.normal(0, 0.2)) for p in prices],
                    "close": prices,
                    "volume": np.random.normal(1000000, 100000, len(dates)),
                },
                index=dates,
            )

            # Test different threshold combinations
            threshold_combinations = [
                {"buy_threshold": 20, "sell_threshold": 80, "name": "Standard"},
                {"buy_threshold": 30, "sell_threshold": 70, "name": "Conservative"},
                {"buy_threshold": 10, "sell_threshold": 90, "name": "Aggressive"},
                {"buy_threshold": 25, "sell_threshold": 75, "name": "Moderate"},
                {"buy_threshold": 15, "sell_threshold": 85, "name": "Balanced"},
                {
                    "buy_threshold": 35,
                    "sell_threshold": 65,
                    "name": "Very Conservative",
                },
            ]

            timeframe_results = {}

            for i, thresholds in enumerate(threshold_combinations):
                print(
                    f"\n    🔍 Testing {thresholds['name']} thresholds: {thresholds['buy_threshold']}/{thresholds['sell_threshold']}"
                )

                # Generate RSI signals with these thresholds
                result = generate_rsi_signals(
                    intraday_data,
                    period=14,
                    buy_threshold=thresholds["buy_threshold"],
                    sell_threshold=thresholds["sell_threshold"],
                )

                # Count signals
                buy_signals = (result["signal"] == 1).sum()
                sell_signals = (result["signal"] == -1).sum()
                neutral_signals = (result["signal"] == 0).sum()
                total_signals = len(result)

                # Calculate signal ratios
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals
                neutral_ratio = neutral_signals / total_signals

                print(f"      Buy signals: {buy_signals} ({buy_ratio:.2%})")
                print(f"      Sell signals: {sell_signals} ({sell_ratio:.2%})")
                print(f"      Neutral signals: {neutral_signals} ({neutral_ratio:.2%})")

                # Calculate performance metrics
                if "strategy_returns" in result.columns:
                    strategy_returns = result["strategy_returns"].dropna()
                    if len(strategy_returns) > 0:
                        # Annualize based on timeframe
                        periods_per_year = (
                            24 * 365
                            if timeframe["freq"] == "H"
                            else 48 * 365
                            if timeframe["freq"] == "30T"
                            else 96 * 365
                        )
                        sharpe_ratio = (
                            strategy_returns.mean()
                            / strategy_returns.std()
                            * np.sqrt(periods_per_year)
                        )
                        win_rate = (strategy_returns > 0).mean()
                        max_drawdown = (
                            strategy_returns.cumsum()
                            - strategy_returns.cumsum().cummax()
                        ).min()
                        total_return = strategy_returns.sum()

                        print(f"      Sharpe ratio: {sharpe_ratio:.3f}")
                        print(f"      Win rate: {win_rate:.2%}")
                        print(f"      Max drawdown: {max_drawdown:.3f}")
                        print(f"      Total return: {total_return:.3f}")

                        timeframe_results[thresholds["name"]] = {
                            "thresholds": thresholds,
                            "buy_ratio": buy_ratio,
                            "sell_ratio": sell_ratio,
                            "neutral_ratio": neutral_ratio,
                            "sharpe_ratio": sharpe_ratio,
                            "win_rate": win_rate,
                            "max_drawdown": max_drawdown,
                            "total_return": total_return,
                        }

                # Test threshold-specific assertions
                if thresholds["name"] == "Standard":
                    # Standard thresholds should have moderate signal activity
                    self.assertGreater(
                        buy_ratio,
                        0.05,
                        "Standard thresholds should generate some buy signals",
                    )
                    self.assertGreater(
                        sell_ratio,
                        0.05,
                        "Standard thresholds should generate some sell signals",
                    )

                elif thresholds["name"] == "Conservative":
                    # Conservative thresholds should have more signals
                    self.assertGreater(
                        buy_ratio,
                        0.1,
                        "Conservative thresholds should generate more buy signals",
                    )
                    self.assertGreater(
                        sell_ratio,
                        0.1,
                        "Conservative thresholds should generate more sell signals",
                    )

                elif thresholds["name"] == "Aggressive":
                    # Aggressive thresholds should have fewer signals
                    self.assertLess(
                        buy_ratio,
                        0.1,
                        "Aggressive thresholds should generate fewer buy signals",
                    )
                    self.assertLess(
                        sell_ratio,
                        0.1,
                        "Aggressive thresholds should generate fewer sell signals",
                    )

            # Compare performance across threshold combinations for this timeframe
            if len(timeframe_results) > 1:
                print(f"\n    📈 {timeframe['name']} Performance Comparison:")
                best_sharpe = max(
                    timeframe_results.values(), key=lambda x: x["sharpe_ratio"]
                )
                best_return = max(
                    timeframe_results.values(), key=lambda x: x["total_return"]
                )

                print(
                    f"      Best Sharpe: {best_sharpe['thresholds']['name']} ({best_sharpe['sharpe_ratio']:.3f})"
                )
                print(
                    f"      Best Return: {best_return['thresholds']['name']} ({best_return['total_return']:.3f})"
                )

                # Assert that different thresholds produce different results
                sharpe_ratios = [r["sharpe_ratio"] for r in timeframe_results.values()]
                self.assertGreater(
                    max(sharpe_ratios) - min(sharpe_ratios),
                    0.1,
                    f"Different thresholds should produce different performance for {timeframe['name']}",
                )

        # Test market condition sensitivity
        print(f"\n  🌊 Testing market condition sensitivity...")

        # Create different market conditions
        market_conditions = [
            {"name": "Trending Up", "trend": 0.001, "volatility": 0.02},
            {"name": "Trending Down", "trend": -0.001, "volatility": 0.02},
            {"name": "High Volatility", "trend": 0.0, "volatility": 0.05},
            {"name": "Low Volatility", "trend": 0.0, "volatility": 0.01},
        ]

        for condition in market_conditions:
            print(f"\n    📊 Testing {condition['name']} market condition...")

            # Create data with specific market condition
            condition_dates = pd.date_range(
                start="2023-01-01", end="2023-01-07", freq="H"
            )
            condition_prices = []
            base_price = 100

            for i in range(len(condition_dates)):
                # Add trend and volatility
                price = (
                    base_price
                    + i * condition["trend"]
                    + np.random.normal(0, condition["volatility"])
                )
                condition_prices.append(price)

            condition_data = pd.DataFrame(
                {
                    "open": condition_prices,
                    "high": [
                        p + abs(np.random.normal(0, 0.1)) for p in condition_prices
                    ],
                    "low": [
                        p - abs(np.random.normal(0, 0.1)) for p in condition_prices
                    ],
                    "close": condition_prices,
                    "volume": np.random.normal(1000000, 100000, len(condition_dates)),
                },
                index=condition_dates,
            )

            # Test with standard thresholds
            condition_result = generate_rsi_signals(
                condition_data, period=14, buy_threshold=20, sell_threshold=80
            )

            # Analyze signal distribution
            buy_signals = (condition_result["signal"] == 1).sum()
            sell_signals = (condition_result["signal"] == -1).sum()
            total_signals = len(condition_result)

            buy_ratio = buy_signals / total_signals
            sell_ratio = sell_signals / total_signals

            print(f"      Buy ratio: {buy_ratio:.2%}")
            print(f"      Sell ratio: {sell_ratio:.2%}")

            # Test market condition specific expectations
            if condition["name"] == "Trending Up":
                # In uptrend, should have more buy signals
                self.assertGreater(
                    buy_ratio, sell_ratio, "Uptrend should have more buy signals"
                )

            elif condition["name"] == "Trending Down":
                # In downtrend, should have more sell signals
                self.assertGreater(
                    sell_ratio, buy_ratio, "Downtrend should have more sell signals"
                )

            elif condition["name"] == "High Volatility":
                # High volatility should have more signals overall
                total_signal_ratio = buy_ratio + sell_ratio
                self.assertGreater(
                    total_signal_ratio,
                    0.15,
                    "High volatility should generate more signals",
                )

        # Test threshold sensitivity analysis
        print(f"\n  🎯 Testing threshold sensitivity analysis...")

        # Create baseline data
        baseline_dates = pd.date_range(start="2023-01-01", end="2023-01-15", freq="H")
        baseline_prices = 100 + np.cumsum(
            np.random.normal(0, 0.02, len(baseline_dates))
        )

        baseline_data = pd.DataFrame(
            {
                "open": baseline_prices,
                "high": [p + abs(np.random.normal(0, 0.1)) for p in baseline_prices],
                "low": [p - abs(np.random.normal(0, 0.1)) for p in baseline_prices],
                "close": baseline_prices,
                "volume": np.random.normal(1000000, 100000, len(baseline_dates)),
            },
            index=baseline_dates,
        )

        # Test fine-grained threshold variations
        buy_thresholds = [15, 20, 25, 30, 35]
        sell_thresholds = [65, 70, 75, 80, 85]

        sensitivity_results = []

        for buy_thresh in buy_thresholds:
            for sell_thresh in sell_thresholds:
                if buy_thresh < sell_thresh:  # Valid combination
                    result = generate_rsi_signals(
                        baseline_data,
                        period=14,
                        buy_threshold=buy_thresh,
                        sell_threshold=sell_thresh,
                    )

                    if "strategy_returns" in result.columns:
                        strategy_returns = result["strategy_returns"].dropna()
                        if len(strategy_returns) > 0:
                            sharpe_ratio = (
                                strategy_returns.mean()
                                / strategy_returns.std()
                                * np.sqrt(24 * 365)
                            )
                            buy_signals = (result["signal"] == 1).sum()
                            sell_signals = (result["signal"] == -1).sum()

                            sensitivity_results.append(
                                {
                                    "buy_threshold": buy_thresh,
                                    "sell_threshold": sell_thresh,
                                    "sharpe_ratio": sharpe_ratio,
                                    "buy_signals": buy_signals,
                                    "sell_signals": sell_signals,
                                }
                            )

        # Analyze sensitivity
        if sensitivity_results:
            best_combination = max(sensitivity_results, key=lambda x: x["sharpe_ratio"])
            print(
                f"    Best threshold combination: {best_combination['buy_threshold']}/{best_combination['sell_threshold']}"
            )
            print(f"    Best Sharpe ratio: {best_combination['sharpe_ratio']:.3f}")

            # Test sensitivity to threshold changes
            base_sharpe = best_combination["sharpe_ratio"]
            sensitivity_threshold = 0.1  # 10% sensitivity threshold

            for result in sensitivity_results:
                if result != best_combination:
                    sharpe_diff = (
                        abs(result["sharpe_ratio"] - base_sharpe) / base_sharpe
                    )
                    if sharpe_diff > sensitivity_threshold:
                        print(
                            f"    High sensitivity: {result['buy_threshold']}/{result['sell_threshold']} -> {sharpe_diff:.1%} change"
                        )

        # Test signal timing analysis
        print(f"\n  ⏱️ Testing signal timing analysis...")

        # Create data with known oversold/overbought periods
        timing_dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="H")
        timing_prices = []

        for i in range(len(timing_dates)):
            # Create clear oversold/overbought cycles
            cycle_position = (i % 24) / 24

            if cycle_position < 0.25:  # Oversold
                price = 95 + np.random.normal(0, 0.5)
            elif cycle_position < 0.5:  # Recovery
                price = 98 + np.random.normal(0, 0.5)
            elif cycle_position < 0.75:  # Overbought
                price = 105 + np.random.normal(0, 0.5)
            else:  # Decline
                price = 102 + np.random.normal(0, 0.5)

            timing_prices.append(price)

        timing_data = pd.DataFrame(
            {
                "open": timing_prices,
                "high": [p + abs(np.random.normal(0, 0.2)) for p in timing_prices],
                "low": [p - abs(np.random.normal(0, 0.2)) for p in timing_prices],
                "close": timing_prices,
                "volume": np.random.normal(1000000, 100000, len(timing_dates)),
            },
            index=timing_dates,
        )

        # Test different thresholds for timing accuracy
        timing_thresholds = [
            {"buy": 20, "sell": 80, "name": "Standard"},
            {"buy": 25, "sell": 75, "name": "Tighter"},
            {"buy": 15, "sell": 85, "name": "Wider"},
        ]

        for timing_thresh in timing_thresholds:
            timing_result = generate_rsi_signals(
                timing_data,
                period=14,
                buy_threshold=timing_thresh["buy"],
                sell_threshold=timing_thresh["sell"],
            )

            # Count signals in expected periods
            expected_buy_periods = 0
            expected_sell_periods = 0

            for i, signal in enumerate(timing_result["signal"]):
                cycle_position = (i % 24) / 24

                if (
                    cycle_position < 0.25 and signal == 1
                ):  # Buy signal in oversold period
                    expected_buy_periods += 1
                elif (
                    cycle_position > 0.5 and cycle_position < 0.75 and signal == -1
                ):  # Sell signal in overbought period
                    expected_sell_periods += 1

            total_buy_signals = (timing_result["signal"] == 1).sum()
            total_sell_signals = (timing_result["signal"] == -1).sum()

            buy_accuracy = (
                expected_buy_periods / total_buy_signals if total_buy_signals > 0 else 0
            )
            sell_accuracy = (
                expected_sell_periods / total_sell_signals
                if total_sell_signals > 0
                else 0
            )

            print(f"    {timing_thresh['name']} thresholds:")
            print(f"      Buy accuracy: {buy_accuracy:.2%}")
            print(f"      Sell accuracy: {sell_accuracy:.2%}")

        print("✅ Oversold/overbought threshold variation test completed")


if __name__ == "__main__":
    unittest.main()
