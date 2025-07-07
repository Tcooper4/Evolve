"""Tests for RSI strategy."""

import unittest
import pandas as pd
import numpy as np
import logging
from trading.strategies.rsi_signals import generate_rsi_signals, load_optimized_settings
from trading.optimization.rsi_optimizer import RSIOptimizer, RSIParameters
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class TestRSIStrategy(unittest.TestCase):
    """Test cases for RSI strategy."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        cls.data = pd.DataFrame({
            'open': np.random.normal(100, 1, len(dates)),
            'high': np.random.normal(101, 1, len(dates)),
            'low': np.random.normal(99, 1, len(dates)),
            'close': np.random.normal(100, 1, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
        
        # Create optimizer instance
        cls.optimizer = RSIOptimizer(cls.data)
    
    def test_generate_rsi_signals(self):
        """Test RSI signal generation."""
        # Test with default parameters
        result = generate_rsi_signals(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('signal', result.columns)
        self.assertIn('returns', result.columns)
        self.assertIn('strategy_returns', result.columns)
        
        # Test with custom parameters
        result = generate_rsi_signals(
            self.data,
            period=10,
            buy_threshold=20,
            sell_threshold=80
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('signal', result.columns)
    
    def test_load_optimized_settings(self):
        """Test loading optimized settings."""
        # Test with non-existent ticker
        settings = load_optimized_settings('NONEXISTENT')
        self.assertIsNone(settings)
    
    def test_rsi_optimizer(self):
        """Test RSI optimizer."""
        # Test parameter optimization
        result = self.optimizer.optimize(n_trials=5)
        logger.info(f"Best params: period={result.period}, overbought={result.overbought}, oversold={result.oversold}, Sharpe={result.sharpe_ratio:.3f}")
        self.assertIsInstance(result.period, int)
        self.assertIsInstance(result.overbought, float)
        self.assertIsInstance(result.oversold, float)
        self.assertIn('sharpe_ratio', result.metrics)
        self.assertIn('win_rate', result.metrics)
        self.assertGreaterEqual(result.sharpe_ratio, -10)  # sanity check
        self.assertLessEqual(result.sharpe_ratio, 10)      # sanity check
    
    def test_optimizer_visualization(self):
        """Test optimizer visualization."""
        # Get optimization results
        results = self.optimizer.optimize_rsi_parameters(
            objective='sharpe',
            n_top=1
        )
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

if __name__ == '__main__':
    unittest.main() 