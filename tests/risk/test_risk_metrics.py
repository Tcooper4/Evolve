"""Test script for risk metrics with synthetic data validation."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.risk.risk_metrics import (
    calculate_rolling_metrics,
    calculate_advanced_metrics,
    calculate_regime_metrics,
    plot_risk_metrics,
    plot_drawdown_heatmap
)

class TestRiskMetrics(unittest.TestCase):
    """Test cases for risk metrics."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create synthetic returns
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create different regimes
        n_days = len(dates)
        
        # Bull market (first third)
        bull_returns = np.random.normal(0.001, 0.01, n_days//3)
        
        # Bear market (second third)
        bear_returns = np.random.normal(-0.001, 0.02, n_days//3)
        
        # Neutral market (last third)
        neutral_returns = np.random.normal(0.0001, 0.015, n_days - 2*(n_days//3))
        
        # Combine returns
        returns = np.concatenate([bull_returns, bear_returns, neutral_returns])
        
        # Create returns series
        cls.returns = pd.Series(returns, index=dates)
        
        # Create multi-asset returns
        n_assets = 5
        cls.multi_returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation."""
        metrics = calculate_rolling_metrics(self.returns)
        
        # Check shape
        self.assertEqual(len(metrics), len(self.returns))
        
        # Check columns
        expected_columns = [
            'volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'max_drawdown'
        ]
        self.assertTrue(all(col in metrics.columns for col in expected_columns))
        
        # Check values
        self.assertTrue(metrics['volatility'].notna().all())
        self.assertTrue(metrics['sharpe_ratio'].notna().all())
        self.assertTrue(metrics['sortino_ratio'].notna().all())
        self.assertTrue(metrics['calmar_ratio'].notna().all())
        self.assertTrue(metrics['max_drawdown'].notna().all())
        
        # Check ranges
        self.assertTrue((metrics['volatility'] >= 0).all())
        self.assertTrue((metrics['max_drawdown'] <= 0).all())
    
    def test_advanced_metrics(self):
        """Test advanced metrics calculation."""
        metrics = calculate_advanced_metrics(self.returns)
        
        # Check keys
        expected_keys = [
            'var_95',
            'cvar_95',
            'tail_risk',
            'skewness',
            'kurtosis'
        ]
        self.assertTrue(all(key in metrics for key in expected_keys))
        
        # Check values
        self.assertIsInstance(metrics['var_95'], float)
        self.assertIsInstance(metrics['cvar_95'], float)
        self.assertIsInstance(metrics['tail_risk'], float)
        self.assertIsInstance(metrics['skewness'], float)
        self.assertIsInstance(metrics['kurtosis'], float)
        
        # Check ranges
        self.assertLess(metrics['var_95'], 0)
        self.assertLess(metrics['cvar_95'], metrics['var_95'])
        self.assertTrue(0 <= metrics['tail_risk'] <= 1)
    
    def test_regime_metrics(self):
        """Test regime metrics calculation."""
        metrics = calculate_regime_metrics(self.returns)
        
        # Check keys
        expected_keys = [
            'regime',
            'sharpe_ratio',
            'volatility',
            'max_drawdown'
        ]
        self.assertTrue(all(key in metrics for key in expected_keys))
        
        # Check regime
        self.assertIn(metrics['regime'], ['bull', 'bear', 'neutral'])
        
        # Check values
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertIsInstance(metrics['volatility'], float)
        self.assertIsInstance(metrics['max_drawdown'], float)
        
        # Check ranges
        self.assertTrue(metrics['volatility'] >= 0)
        self.assertTrue(metrics['max_drawdown'] <= 0)
    
    def test_plot_risk_metrics(self):
        """Test risk metrics plotting."""
        metrics = calculate_rolling_metrics(self.returns)
        fig = plot_risk_metrics(metrics)
        
        # Check figure
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 6)  # 6 subplots
        
        # Check layout
        self.assertIsNotNone(fig.layout)
        self.assertIsNotNone(fig.layout.title)
    
    def test_plot_drawdown_heatmap(self):
        """Test drawdown heatmap plotting."""
        fig = plot_drawdown_heatmap(self.multi_returns)
        
        # Check figure
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # 1 heatmap
        
        # Check layout
        self.assertIsNotNone(fig.layout)
        self.assertIsNotNone(fig.layout.title)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty returns
        empty_returns = pd.Series([], index=pd.DatetimeIndex([]))
        with self.assertRaises(ValueError):
            calculate_rolling_metrics(empty_returns)
        
        # Single return
        single_return = pd.Series([0.001], index=[datetime.now()])
        with self.assertRaises(ValueError):
            calculate_rolling_metrics(single_return)
        
        # All zeros
        zero_returns = pd.Series([0] * 100, index=pd.date_range(start='2023-01-01', periods=100))
        metrics = calculate_rolling_metrics(zero_returns)
        self.assertTrue(metrics['volatility'].iloc[-1] == 0)
        self.assertTrue(np.isnan(metrics['sharpe_ratio'].iloc[-1]))
    
    def test_window_sizes(self):
        """Test different window sizes."""
        # Test small window
        metrics_small = calculate_rolling_metrics(self.returns, window=20)
        self.assertEqual(len(metrics_small), len(self.returns))
        
        # Test large window
        metrics_large = calculate_rolling_metrics(self.returns, window=500)
        self.assertEqual(len(metrics_large), len(self.returns))
        
        # Test window larger than data
        with self.assertRaises(ValueError):
            calculate_rolling_metrics(self.returns, window=len(self.returns) + 1)

if __name__ == '__main__':
    unittest.main() 