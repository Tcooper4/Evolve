"""Test script for risk manager."""

import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.risk.risk_manager import RiskManager, RiskMetrics

class TestRiskManager(unittest.TestCase):
    """Test cases for risk manager."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create synthetic returns
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        
        # Create synthetic benchmark returns
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, len(dates)),
            index=dates
        )
        
        cls.returns = returns
        cls.benchmark_returns = benchmark_returns
        
        # Create test config
        cls.config = {
            'max_position_size': 0.2,
            'max_leverage': 1.0,
            'risk_free_rate': 0.02
        }
    
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
        self.assertIn('position_limit', limits)
        self.assertIn('leverage_limit', limits)
        self.assertIn('kelly_fraction', limits)
        
        # Test with no returns
        empty_manager = RiskManager(self.config)
        empty_limits = empty_manager.get_position_limits()
        self.assertEqual(empty_limits, {})
    
    def test_optimization(self):
        """Test position size optimization."""
        # Create synthetic data
        n_assets = 5
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (len(dates), n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
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
        results_dir = 'trading/risk/results'
        os.makedirs(results_dir, exist_ok=True)
        
        for i in range(10):
            with open(os.path.join(results_dir, f'test_{i}.json'), 'w') as f:
                f.write('{}')
        
        # Clean up
        self.risk_manager.cleanup_old_results(max_files=5)
        
        # Check number of remaining files
        remaining_files = len([f for f in os.listdir(results_dir) if f.endswith('.json')])
        self.assertLessEqual(remaining_files, 5)
    
    def test_report_export(self):
        """Test report export."""
        self.risk_manager.update_returns(self.returns, self.benchmark_returns)
        
        # Test JSON export
        json_path = 'trading/risk/results/test_report.json'
        self.risk_manager.export_risk_report(json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Test CSV export
        csv_path = 'trading/risk/results/test_report.csv'
        self.risk_manager.export_risk_report(csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.risk_manager.export_risk_report('test.txt')
    
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

if __name__ == '__main__':
    unittest.main() 