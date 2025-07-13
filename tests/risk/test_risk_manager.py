"""Test script for risk manager."""

import os
import unittest

import numpy as np
import pandas as pd

from trading.risk.risk_manager import RiskManager, RiskMetrics


class TestRiskManager(unittest.TestCase):
    """Test cases for risk manager."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create synthetic returns
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

        # Create synthetic benchmark returns
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)

        cls.returns = returns
        cls.benchmark_returns = benchmark_returns

        # Create test config
        cls.config = {"max_position_size": 0.2, "max_leverage": 1.0, "risk_free_rate": 0.02}

    def setUp(self):
        """Set up test case."""
        self.risk_manager = RiskManager(self.config)

    def test_initialization(self):
        """Test initialization."""
        self.assertIsNotNone(self.risk_manager)
        self.assertEqual(self.risk_manager.returns, None)
        self.assertEqual(self.risk_manager.benchmark_returns, None)
        self.assertEqual(len(self.risk_manager.metrics_history), 0)
        self.assertEqual(self.risk_manager.current_metrics, None)

    def test_update_returns(self):
        """Test returns update."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)

        self.assertIsNotNone(self.risk_manager.returns)
        self.assertIsNotNone(self.risk_manager.benchmark_returns)
        self.assertIsNotNone(self.risk_manager.current_metrics)
        self.assertEqual(len(self.risk_manager.metrics_history), 1)

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)
        metrics = self.risk_manager.current_metrics

        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.sortino_ratio, float)
        self.assertIsInstance(metrics.var_95, float)
        self.assertIsInstance(metrics.cvar_95, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertIsInstance(metrics.volatility, float)
        self.assertIsInstance(metrics.beta, float)
        self.assertIsInstance(metrics.correlation, float)
        self.assertIsInstance(metrics.kelly_fraction, float)

    def test_position_limits(self):
        """Test position limits calculation."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)
        limits = self.risk_manager.get_position_limits()

        self.assertIsNotNone(limits)
        self.assertIn("position_limit", limits)
        self.assertIn("leverage_limit", limits)
        self.assertIn("kelly_fraction", limits)

        # Test with no returns
        empty_manager = RiskManager(self.config)
        empty_limits = empty_manager.get_position_limits()
        self.assertEqual(empty_limits, {})

    def test_optimization(self):
        """Test position size optimization."""
        # Create synthetic data
        n_assets = 5
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (len(dates), n_assets)),
            index=dates,
            columns=[f"Asset_{i}" for i in range(n_assets)],
        )

        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        covariance = returns.cov()

        # Optimize
        weights = self.risk_manager.optimize_position_sizes(expected_returns, covariance)

        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), n_assets)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 1).all())

    def test_plotting(self):
        """Test metrics plotting."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)
        fig = self.risk_manager.plot_risk_metrics()

        self.assertIsNotNone(fig)

        # Test with no history
        empty_manager = RiskManager(self.config)
        empty_fig = empty_manager.plot_risk_metrics()
        self.assertIsNone(empty_fig)

    def test_cleanup(self):
        """Test results cleanup."""
        # Create test files
        results_dir = "trading/risk/results"
        os.makedirs(results_dir, exist_ok=True)

        for i in range(10):
            with open(os.path.join(results_dir, f"test_{i}.json"), "w") as f:
                f.write("{}")

        # Clean up
        self.risk_manager.cleanup_old_results(max_files=5)

        # Check number of remaining files
        remaining_files = len([f for f in os.listdir(results_dir) if f.endswith(".json")])
        self.assertLessEqual(remaining_files, 5)

    def test_report_export(self):
        """Test report export."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)

        # Test JSON export
        json_path = "trading/risk/results/test_report.json"
        self.risk_manager.export_risk_report(json_path)
        self.assertTrue(os.path.exists(json_path))

        # Test CSV export
        csv_path = "trading/risk/results/test_report.csv"
        self.risk_manager.export_risk_report(csv_path)
        self.assertTrue(os.path.exists(csv_path))

        # Test invalid format
        with self.assertRaises(ValueError):
            self.risk_manager.export_risk_report("test.txt")

    def test_benchmark_beta(self):
        """Test beta calculation with benchmark."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)
        metrics = self.risk_manager.current_metrics

        # Beta should not be 1.0 when using benchmark
        self.assertNotEqual(metrics.beta, 1.0)

        # Test without benchmark
        self.risk_manager.update_returns(self.returns)
        metrics = self.risk_manager.current_metrics
        self.assertEqual(metrics.beta, 1.0)

    def test_large_drawdown_and_recovery(self):
        """Include scenario with large drawdown followed by recovery to test reset logic."""
        print("\n📉 Testing Large Drawdown and Recovery Scenarios")

        # Test multiple drawdown and recovery scenarios
        scenarios = [
            {
                "name": "Single Large Drawdown",
                "returns": [0.01] * 30 + [-0.5] * 5 + [0.02] * 65,
                "expected_max_dd": -0.3,
                "expected_recovery": True,
            },
            {
                "name": "Multiple Drawdowns",
                "returns": [0.01] * 20 + [-0.3] * 3 + [0.02] * 20 + [-0.4] * 3 + [0.02] * 54,
                "expected_max_dd": -0.4,
                "expected_recovery": True,
            },
            {
                "name": "Gradual Recovery",
                "returns": [0.01] * 25 + [-0.6] * 5 + [0.005] * 70,
                "expected_max_dd": -0.4,
                "expected_recovery": True,
            },
            {
                "name": "No Recovery",
                "returns": [0.01] * 30 + [-0.5] * 5 + [-0.01] * 65,
                "expected_max_dd": -0.5,
                "expected_recovery": False,
            },
        ]

        for scenario in scenarios:
            print(f"\n  📊 Testing scenario: {scenario['name']}")

            # Create returns series
            dates = pd.date_range(start="2023-01-01", periods=len(scenario["returns"]))
            returns = pd.Series(scenario["returns"], index=dates)

            # Update risk manager
            self.risk_manager.update_returns(returns)
            metrics = self.risk_manager.current_metrics

            # Verify metrics exist
            self.assertIsNotNone(metrics, "Metrics should be calculated")

            # Verify max drawdown is as expected
            print(f"    Max drawdown: {metrics.max_drawdown:.3f} (expected < {scenario['expected_max_dd']:.1f})")
            self.assertLess(
                metrics.max_drawdown,
                scenario["expected_max_dd"],
                f"Max drawdown should be less than {scenario['expected_max_dd']}",
            )

            # Test rolling metrics calculation
            rolling_metrics = self.risk_manager.calculate_rolling_metrics(returns, window=20)

            # Verify rolling metrics structure
            self.assertIsNotNone(rolling_metrics, "Rolling metrics should be calculated")
            self.assertIn("max_drawdown", rolling_metrics.columns, "Should have max_drawdown column")
            self.assertIn("volatility", rolling_metrics.columns, "Should have volatility column")
            self.assertIn("sharpe_ratio", rolling_metrics.columns, "Should have sharpe_ratio column")

            # Test drawdown reset logic
            drawdowns = rolling_metrics["max_drawdown"]

            if scenario["expected_recovery"]:
                # After recovery, drawdown should reset (close to 0)
                final_drawdown = drawdowns.iloc[-1]
                print(f"    Final drawdown: {final_drawdown:.3f}")
                self.assertGreater(final_drawdown, -0.1, "Drawdown should reset after recovery")
            else:
                # No recovery scenario - drawdown should remain large
                final_drawdown = drawdowns.iloc[-1]
                print(f"    Final drawdown: {final_drawdown:.3f}")
                self.assertLess(final_drawdown, -0.2, "Drawdown should remain large without recovery")

        # Test risk management behavior during drawdown
        print(f"\n  🛡️ Testing risk management during drawdown...")

        # Create severe drawdown scenario
        dates = pd.date_range(start="2023-01-01", periods=100)
        severe_returns = pd.Series([0.01] * 20 + [-0.7] * 5 + [0.02] * 75, index=dates)

        # Update risk manager
        self.risk_manager.update_returns(severe_returns)

        # Test position limits during drawdown
        limits = self.risk_manager.get_position_limits()

        # Verify limits are calculated
        self.assertIsNotNone(limits, "Position limits should be calculated")
        self.assertIn("position_limit", limits, "Should have position_limit")
        self.assertIn("leverage_limit", limits, "Should have leverage_limit")
        self.assertIn("kelly_fraction", limits, "Should have kelly_fraction")

        # Verify limits are conservative during drawdown
        self.assertLess(limits["position_limit"], 0.5, "Position limit should be conservative during drawdown")
        self.assertLess(limits["leverage_limit"], 1.0, "Leverage limit should be conservative during drawdown")

        print(f"    Position limit: {limits['position_limit']:.3f}")
        print(f"    Leverage limit: {limits['leverage_limit']:.3f}")
        print(f"    Kelly fraction: {limits['kelly_fraction']:.3f}")

        # Test risk alerts during drawdown
        alerts = self.risk_manager.get_risk_alerts()

        # Verify alerts are generated
        self.assertIsNotNone(alerts, "Risk alerts should be generated")
        self.assertIsInstance(alerts, list, "Alerts should be a list")

        if alerts:
            print(f"    Risk alerts: {len(alerts)} alerts generated")
            for alert in alerts:
                print(f"      - {alert}")

        # Test recovery detection
        print(f"\n  🔍 Testing recovery detection...")

        # Create recovery scenario with monitoring
        recovery_dates = pd.date_range(start="2023-01-01", periods=150)
        recovery_returns = pd.Series([0.01] * 30 + [-0.5] * 5 + [0.02] * 115, index=recovery_dates)

        # Monitor recovery progress
        recovery_metrics = []
        for i in range(30, len(recovery_returns), 10):
            window_returns = recovery_returns.iloc[:i]
            self.risk_manager.update_returns(window_returns)
            metrics = self.risk_manager.current_metrics
            recovery_metrics.append(
                {
                    "period": i,
                    "max_drawdown": metrics.max_drawdown,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "volatility": metrics.volatility,
                }
            )

        # Verify recovery progression
        initial_dd = recovery_metrics[0]["max_drawdown"]
        final_dd = recovery_metrics[-1]["max_drawdown"]

        print(f"    Initial drawdown: {initial_dd:.3f}")
        print(f"    Final drawdown: {final_dd:.3f}")
        print(f"    Recovery improvement: {final_dd - initial_dd:.3f}")

        # Verify recovery occurred
        self.assertGreater(final_dd, initial_dd, "Drawdown should improve during recovery")

        # Test risk reset after recovery
        print(f"\n  🔄 Testing risk reset after recovery...")

        # Get position limits after recovery
        final_limits = self.risk_manager.get_position_limits()

        # Verify limits are more permissive after recovery
        self.assertGreater(
            final_limits["position_limit"], limits["position_limit"], "Position limit should increase after recovery"
        )
        self.assertGreater(
            final_limits["leverage_limit"], limits["leverage_limit"], "Leverage limit should increase after recovery"
        )

        print(f"    Post-recovery position limit: {final_limits['position_limit']:.3f}")
        print(f"    Post-recovery leverage limit: {final_limits['leverage_limit']:.3f}")

        # Test drawdown persistence
        print(f"\n  📈 Testing drawdown persistence...")

        # Create scenario with persistent drawdown
        persistent_dates = pd.date_range(start="2023-01-01", periods=200)
        persistent_returns = pd.Series([0.01] * 50 + [-0.4] * 10 + [-0.005] * 140, index=persistent_dates)

        # Update risk manager
        self.risk_manager.update_returns(persistent_returns)

        # Calculate rolling drawdown
        rolling_dd = self.risk_manager.calculate_rolling_metrics(persistent_returns, window=30)

        # Verify drawdown persists
        final_persistent_dd = rolling_dd["max_drawdown"].iloc[-1]
        print(f"    Persistent drawdown: {final_persistent_dd:.3f}")

        self.assertLess(final_persistent_dd, -0.2, "Drawdown should persist without recovery")

        # Test risk management adaptation
        print(f"\n  🎯 Testing risk management adaptation...")

        # Get adaptive limits
        adaptive_limits = self.risk_manager.get_adaptive_position_limits()

        # Verify adaptive limits are calculated
        self.assertIsNotNone(adaptive_limits, "Adaptive limits should be calculated")
        self.assertIn("conservative_limit", adaptive_limits, "Should have conservative limit")
        self.assertIn("aggressive_limit", adaptive_limits, "Should have aggressive limit")
        self.assertIn("recommended_limit", adaptive_limits, "Should have recommended limit")

        print(f"    Conservative limit: {adaptive_limits['conservative_limit']:.3f}")
        print(f"    Aggressive limit: {adaptive_limits['aggressive_limit']:.3f}")
        print(f"    Recommended limit: {adaptive_limits['recommended_limit']:.3f}")

        # Verify limits are reasonable
        self.assertLess(
            adaptive_limits["conservative_limit"],
            adaptive_limits["aggressive_limit"],
            "Conservative limit should be less than aggressive limit",
        )
        self.assertGreaterEqual(
            adaptive_limits["recommended_limit"],
            adaptive_limits["conservative_limit"],
            "Recommended limit should be >= conservative limit",
        )
        self.assertLessEqual(
            adaptive_limits["recommended_limit"],
            adaptive_limits["aggressive_limit"],
            "Recommended limit should be <= aggressive limit",
        )

        print("✅ Large drawdown and recovery test completed")


if __name__ == "__main__":
    unittest.main()
