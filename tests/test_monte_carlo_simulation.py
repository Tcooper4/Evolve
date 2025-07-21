"""
Tests for Monte Carlo Simulation Module

This module tests the functionality of the Monte Carlo simulation system.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trading.backtesting.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSimulator,
    run_monte_carlo_analysis,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMonteCarloSimulation(unittest.TestCase):
    """Test cases for Monte Carlo simulation functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Generate test returns
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.test_returns = pd.Series(np.random.normal(0.0005, 0.02, 100), index=dates)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_monte_carlo_config(self):
        """Test MonteCarloConfig initialization."""
        config = MonteCarloConfig()

        self.assertEqual(config.n_simulations, 1000)
        self.assertEqual(config.confidence_levels, [0.05, 0.50, 0.95])
        self.assertEqual(config.bootstrap_method, "historical")
        self.assertEqual(config.initial_capital, 10000.0)

        # Test custom config
        custom_config = MonteCarloConfig(
            n_simulations=500,
            confidence_levels=[0.01, 0.99],
            bootstrap_method="block",
            initial_capital=5000.0,
        )

        self.assertEqual(custom_config.n_simulations, 500)
        self.assertEqual(custom_config.confidence_levels, [0.01, 0.99])
        self.assertEqual(custom_config.bootstrap_method, "block")
        self.assertEqual(custom_config.initial_capital, 5000.0)

    def test_simulator_initialization(self):
        """Test MonteCarloSimulator initialization."""
        config = MonteCarloConfig()
        simulator = MonteCarloSimulator(config)

        self.assertEqual(simulator.config, config)
        self.assertEqual(simulator.results, {})
        self.assertIsNone(simulator.simulated_paths)
        self.assertIsNone(simulator.percentiles)

    def test_historical_bootstrap(self):
        """Test historical bootstrap method."""
        config = MonteCarloConfig(n_simulations=100, bootstrap_method="historical")
        simulator = MonteCarloSimulator(config)

        paths = simulator._bootstrap_historical_returns(self.test_returns, 100, 100)

        self.assertEqual(paths.shape, (100, 100))
        self.assertTrue(np.all(np.isfinite(paths)))

    def test_block_bootstrap(self):
        """Test block bootstrap method."""
        config = MonteCarloConfig(
            n_simulations=100, bootstrap_method="block", block_size=10
        )
        simulator = MonteCarloSimulator(config)

        paths = simulator._block_bootstrap_returns(self.test_returns, 100, 100)

        self.assertEqual(paths.shape, (100, 100))
        self.assertTrue(np.all(np.isfinite(paths)))

    def test_parametric_bootstrap(self):
        """Test parametric bootstrap method."""
        config = MonteCarloConfig(n_simulations=100, bootstrap_method="parametric")
        simulator = MonteCarloSimulator(config)

        paths = simulator._parametric_bootstrap_returns(self.test_returns, 100, 100)

        self.assertEqual(paths.shape, (100, 100))
        self.assertTrue(np.all(np.isfinite(paths)))

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        config = MonteCarloConfig()
        simulator = MonteCarloSimulator(config)

        # Create sample return paths
        return_paths = np.random.normal(0.001, 0.02, (10, 50))
        initial_capital = 10000.0

        portfolio_paths = simulator._calculate_portfolio_values(
            return_paths, initial_capital
        )

        self.assertEqual(portfolio_paths.shape, (50, 10))
        self.assertEqual(portfolio_paths.index[0], pd.Timestamp("2020-01-01").date())
        self.assertTrue(np.all(portfolio_paths.iloc[0] == initial_capital))

    def test_simulate_portfolio_paths(self):
        """Test portfolio path simulation."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        portfolio_paths = simulator.simulate_portfolio_paths(
            self.test_returns, initial_capital=10000.0, n_simulations=50
        )

        self.assertEqual(portfolio_paths.shape, (100, 50))
        self.assertEqual(simulator.results["n_simulations"], 50)
        self.assertEqual(simulator.results["n_periods"], 100)
        self.assertEqual(simulator.results["initial_capital"], 10000.0)
        self.assertIsNotNone(simulator.simulated_paths)

    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation first
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)

        # Calculate percentiles
        percentiles = simulator.calculate_percentiles([0.05, 0.50, 0.95])

        self.assertIsNotNone(percentiles)
        self.assertIn("P5", percentiles.columns)
        self.assertIn("P50", percentiles.columns)
        self.assertIn("P95", percentiles.columns)
        self.assertIn("Mean", percentiles.columns)
        self.assertEqual(len(percentiles), 100)  # Number of periods

    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        simulator.calculate_percentiles()

        stats = simulator.get_summary_statistics()

        # Check that all expected keys are present
        expected_keys = [
            "mean_final_value",
            "std_final_value",
            "mean_total_return",
            "std_total_return",
            "min_final_value",
            "max_final_value",
            "var_95",
            "var_99",
            "probability_of_loss",
        ]

        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertTrue(np.isfinite(stats[key]))

    def test_invalid_bootstrap_method(self):
        """Test error handling for invalid bootstrap method."""
        config = MonteCarloConfig(bootstrap_method="invalid_method")
        simulator = MonteCarloSimulator(config)

        with self.assertRaises(ValueError):
            simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)

    def test_empty_returns(self):
        """Test error handling for empty returns series."""
        config = MonteCarloConfig()
        simulator = MonteCarloSimulator(config)

        empty_returns = pd.Series(dtype=float)

        with self.assertRaises(ValueError):
            simulator.simulate_portfolio_paths(empty_returns, 10000.0, 50)

    def test_run_monte_carlo_analysis(self):
        """Test the convenience function."""
        results = run_monte_carlo_analysis(
            returns=self.test_returns,
            initial_capital=10000.0,
            n_simulations=50,
            bootstrap_method="historical",
            plot_results=False,
        )

        self.assertIsInstance(results, dict)
        self.assertIn("simulation_config", results)
        self.assertIn("summary_statistics", results)
        self.assertIn("percentile_analysis", results)
        self.assertIn("risk_metrics", results)

    def test_confidence_bands_visualization(self):
        """Test confidence bands visualization."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        simulator.calculate_percentiles()

        # Test plotting
        fig = simulator.plot_simulation_results(
            show_paths=True, n_paths_to_show=10, confidence_bands=True, save_path=None
        )

        self.assertIsNotNone(fig)

    def test_detailed_report(self):
        """Test detailed report generation."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        simulator.calculate_percentiles()

        report = simulator.create_detailed_report()

        self.assertIn("simulation_config", report)
        self.assertIn("summary_statistics", report)
        self.assertIn("percentile_analysis", report)
        self.assertIn("risk_metrics", report)

        # Check simulation config
        config_data = report["simulation_config"]
        self.assertEqual(config_data["n_simulations"], 50)
        self.assertEqual(config_data["bootstrap_method"], "historical")

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        simulator.calculate_percentiles()

        drawdowns = simulator._calculate_max_drawdowns()

        self.assertIsInstance(drawdowns, dict)
        self.assertIn("max_drawdown_P50", drawdowns)
        self.assertIn("max_drawdown_Mean", drawdowns)

        # Check that drawdowns are negative or zero
        for drawdown in drawdowns.values():
            self.assertLessEqual(drawdown, 0)

    def test_volatility_analysis(self):
        """Test volatility analysis."""
        config = MonteCarloConfig(n_simulations=50)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        simulator.calculate_percentiles()

        vol_analysis = simulator._calculate_volatility_analysis()

        self.assertIsInstance(vol_analysis, dict)
        self.assertIn("mean_volatility", vol_analysis)
        self.assertIn("volatility_std", vol_analysis)
        self.assertIn("min_volatility", vol_analysis)
        self.assertIn("max_volatility", vol_analysis)

        # Check that volatilities are positive
        for vol in vol_analysis.values():
            self.assertGreaterEqual(vol, 0)

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        confidence_levels = [0.01, 0.25, 0.75, 0.99]

        config = MonteCarloConfig(n_simulations=50, confidence_levels=confidence_levels)
        simulator = MonteCarloSimulator(config)

        # Run simulation
        simulator.simulate_portfolio_paths(self.test_returns, 10000.0, 50)
        percentiles = simulator.calculate_percentiles(confidence_levels)

        # Check that all requested percentiles are present
        for level in confidence_levels:
            percentile_key = f"P{int(level * 100)}"
            self.assertIn(percentile_key, percentiles.columns)

    def test_parameter_sensitivity(self):
        """Test sensitivity to different parameters."""
        # Test different numbers of simulations
        for n_sim in [10, 50, 100]:
            config = MonteCarloConfig(n_simulations=n_sim)
            simulator = MonteCarloSimulator(config)

            portfolio_paths = simulator.simulate_portfolio_paths(
                self.test_returns, 10000.0, n_sim
            )

            self.assertEqual(portfolio_paths.shape[1], n_sim)

        # Test different initial capitals
        for capital in [5000.0, 10000.0, 20000.0]:
            config = MonteCarloConfig()
            simulator = MonteCarloSimulator(config)

            portfolio_paths = simulator.simulate_portfolio_paths(
                self.test_returns, capital, 50
            )

            self.assertTrue(np.all(portfolio_paths.iloc[0] == capital))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
