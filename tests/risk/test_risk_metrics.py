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

    def test_max_drawdown_period_and_sharpe_sign_flip(self):
        """Add max drawdown period validation and Sharpe ratio sign flip edge case."""
        print("\n📊 Testing Max Drawdown Period and Sharpe Sign Flip Edge Cases")
        
        # Test multiple scenarios with different drawdown patterns
        scenarios = [
            {
                'name': 'Sharp Drop and Recovery',
                'returns': [0.01]*10 + [-0.2]*5 + [0.01]*35,
                'expected_dd_period': (10, 15),
                'expected_sign_flips': 2
            },
            {
                'name': 'Gradual Decline',
                'returns': [0.01]*20 + [-0.05]*10 + [0.01]*20,
                'expected_dd_period': (20, 30),
                'expected_sign_flips': 1
            },
            {
                'name': 'Multiple Drawdowns',
                'returns': [0.01]*15 + [-0.15]*3 + [0.01]*10 + [-0.1]*3 + [0.01]*19,
                'expected_dd_period': (15, 18),
                'expected_sign_flips': 3
            },
            {
                'name': 'Volatile Period',
                'returns': [0.01]*10 + [-0.1, 0.1, -0.1, 0.1, -0.1]*8 + [0.01]*10,
                'expected_dd_period': (10, 50),
                'expected_sign_flips': 8
            }
        ]
        
        for scenario in scenarios:
            print(f"\n  📈 Testing scenario: {scenario['name']}")
            
            # Create returns series
            dates = pd.date_range(start='2023-01-01', periods=len(scenario['returns']))
            returns = pd.Series(scenario['returns'], index=dates)
            
            # Calculate rolling metrics
            metrics = calculate_rolling_metrics(returns, window=10)
            
            # Test max drawdown period validation
            print(f"    Testing max drawdown period...")
            
            # Find the period of maximum drawdown
            min_dd_idx = metrics['max_drawdown'].idxmin()
            min_dd_period = min_dd_idx.dayofyear
            
            print(f"    Max drawdown occurred at period: {min_dd_period}")
            print(f"    Expected period range: {scenario['expected_dd_period']}")
            
            # Verify drawdown period is within expected range
            start_period, end_period = scenario['expected_dd_period']
            self.assertTrue(start_period <= min_dd_period <= end_period,
                          f"Max drawdown should occur between periods {start_period} and {end_period}")
            
            # Test drawdown magnitude
            max_dd_value = metrics['max_drawdown'].min()
            print(f"    Max drawdown value: {max_dd_value:.3f}")
            
            # Verify drawdown is negative
            self.assertLess(max_dd_value, 0, "Maximum drawdown should be negative")
            
            # Test Sharpe ratio sign flip detection
            print(f"    Testing Sharpe ratio sign flips...")
            
            sharpe = metrics['sharpe_ratio']
            
            # Calculate sign changes
            sign_changes = np.sign(sharpe).diff().fillna(0).abs().sum()
            print(f"    Number of sign flips: {sign_changes}")
            print(f"    Expected sign flips: {scenario['expected_sign_flips']}")
            
            # Verify sign flip count
            self.assertGreaterEqual(sign_changes, scenario['expected_sign_flips'],
                                  f"Should have at least {scenario['expected_sign_flips']} sign flips")
            
            # Test sign flip locations
            sign_flip_locations = np.where(np.sign(sharpe).diff().fillna(0) != 0)[0]
            print(f"    Sign flip locations: {sign_flip_locations}")
            
            # Verify sign flips are distributed
            if len(sign_flip_locations) > 1:
                flip_intervals = np.diff(sign_flip_locations)
                print(f"    Intervals between flips: {flip_intervals}")
                
                # Verify flips are not all at the beginning
                self.assertGreater(np.mean(flip_intervals), 2, "Sign flips should be distributed")
        
        # Test edge cases with extreme values
        print(f"\n  ⚠️ Testing extreme value edge cases...")
        
        # Test with very large drawdown
        extreme_dates = pd.date_range(start='2023-01-01', periods=30)
        extreme_returns = pd.Series([0.01]*10 + [-0.8]*5 + [0.01]*15, index=extreme_dates)
        
        extreme_metrics = calculate_rolling_metrics(extreme_returns, window=5)
        
        # Verify extreme drawdown is handled
        extreme_max_dd = extreme_metrics['max_drawdown'].min()
        print(f"    Extreme max drawdown: {extreme_max_dd:.3f}")
        
        self.assertLess(extreme_max_dd, -0.5, "Extreme drawdown should be properly calculated")
        
        # Test with very volatile returns
        volatile_dates = pd.date_range(start='2023-01-01', periods=50)
        volatile_returns = pd.Series([0.2, -0.2, 0.3, -0.3, 0.1, -0.1]*8 + [0.01]*2, index=volatile_dates)
        
        volatile_metrics = calculate_rolling_metrics(volatile_returns, window=5)
        
        # Verify high volatility is handled
        max_volatility = volatile_metrics['volatility'].max()
        print(f"    Max volatility: {max_volatility:.3f}")
        
        self.assertGreater(max_volatility, 0.1, "High volatility should be properly calculated")
        
        # Test Sharpe ratio with zero volatility
        print(f"\n  🔍 Testing zero volatility edge case...")
        
        zero_vol_dates = pd.date_range(start='2023-01-01', periods=20)
        zero_vol_returns = pd.Series([0.01]*20, index=zero_vol_dates)  # Constant returns
        
        zero_vol_metrics = calculate_rolling_metrics(zero_vol_returns, window=10)
        
        # Verify zero volatility handling
        zero_vol_sharpe = zero_vol_metrics['sharpe_ratio'].iloc[-1]
        print(f"    Sharpe ratio with zero volatility: {zero_vol_sharpe}")
        
        # Should be NaN or infinite when volatility is zero
        self.assertTrue(np.isnan(zero_vol_sharpe) or np.isinf(zero_vol_sharpe),
                       "Sharpe ratio should be NaN or infinite with zero volatility")
        
        # Test drawdown period with recovery
        print(f"\n  🔄 Testing drawdown recovery period...")
        
        recovery_dates = pd.date_range(start='2023-01-01', periods=60)
        recovery_returns = pd.Series([0.01]*20 + [-0.3]*5 + [0.02]*35, index=recovery_dates)
        
        recovery_metrics = calculate_rolling_metrics(recovery_returns, window=15)
        
        # Find drawdown and recovery periods
        drawdown_period = recovery_metrics['max_drawdown'].idxmin()
        recovery_period = recovery_metrics['max_drawdown'].iloc[-1]
        
        print(f"    Drawdown period: {drawdown_period.dayofyear}")
        print(f"    Final drawdown: {recovery_period:.3f}")
        
        # Verify recovery occurred
        self.assertGreater(recovery_period, recovery_metrics['max_drawdown'].min(),
                          "Should show recovery from maximum drawdown")
        
        # Test Sharpe ratio stability
        print(f"\n  📊 Testing Sharpe ratio stability...")
        
        # Create stable returns
        stable_dates = pd.date_range(start='2023-01-01', periods=100)
        stable_returns = pd.Series([0.001]*100, index=stable_dates)  # Small constant returns
        
        stable_metrics = calculate_rolling_metrics(stable_returns, window=20)
        
        # Calculate Sharpe ratio variance
        sharpe_variance = stable_metrics['sharpe_ratio'].var()
        print(f"    Sharpe ratio variance: {sharpe_variance:.6f}")
        
        # Verify stability (low variance)
        self.assertLess(sharpe_variance, 0.1, "Sharpe ratio should be stable with constant returns")
        
        # Test period validation with different window sizes
        print(f"\n  🎯 Testing period validation with different windows...")
        
        test_dates = pd.date_range(start='2023-01-01', periods=40)
        test_returns = pd.Series([0.01]*15 + [-0.2]*5 + [0.01]*20, index=test_dates)
        
        window_sizes = [5, 10, 15, 20]
        
        for window in window_sizes:
            window_metrics = calculate_rolling_metrics(test_returns, window=window)
            
            # Find max drawdown period
            window_min_dd_idx = window_metrics['max_drawdown'].idxmin()
            window_min_dd_period = window_min_dd_idx.dayofyear
            
            print(f"    Window {window}: Max DD at period {window_min_dd_period}")
            
            # Verify drawdown period is reasonable
            self.assertTrue(15 <= window_min_dd_period <= 20,
                          f"Max drawdown should occur in expected period for window {window}")
        
        # Test sign flip edge cases
        print(f"\n  🔄 Testing sign flip edge cases...")
        
        # Test with alternating positive/negative returns
        alternating_dates = pd.date_range(start='2023-01-01', periods=30)
        alternating_returns = pd.Series([0.1, -0.1]*15, index=alternating_dates)
        
        alternating_metrics = calculate_rolling_metrics(alternating_returns, window=5)
        
        # Count sign flips
        alternating_sharpe = alternating_metrics['sharpe_ratio']
        alternating_flips = np.sign(alternating_sharpe).diff().fillna(0).abs().sum()
        
        print(f"    Alternating returns sign flips: {alternating_flips}")
        
        # Should have many sign flips with alternating returns
        self.assertGreater(alternating_flips, 5, "Should have many sign flips with alternating returns")
        
        # Test with trend followed by reversal
        trend_reversal_dates = pd.date_range(start='2023-01-01', periods=50)
        trend_reversal_returns = pd.Series([0.02]*25 + [-0.02]*25, index=trend_reversal_dates)
        
        trend_reversal_metrics = calculate_rolling_metrics(trend_reversal_returns, window=10)
        
        # Count sign flips
        trend_reversal_sharpe = trend_reversal_metrics['sharpe_ratio']
        trend_reversal_flips = np.sign(trend_reversal_sharpe).diff().fillna(0).abs().sum()
        
        print(f"    Trend reversal sign flips: {trend_reversal_flips}")
        
        # Should have sign flip at trend reversal
        self.assertGreaterEqual(trend_reversal_flips, 1, "Should have sign flip at trend reversal")
        
        print("✅ Max drawdown period and Sharpe sign flip edge cases test completed")

if __name__ == '__main__':
    unittest.main() 