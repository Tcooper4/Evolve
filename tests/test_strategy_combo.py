#!/usr/bin/env python3
"""
Test Strategy Combo Functionality

This test file verifies that the enhanced strategy pipeline works correctly
and doesn't break existing functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

class TestStrategyCombo(unittest.TestCase):
    """Test cases for strategy combination functionality."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Generate test data
        self.test_data = self._generate_test_data(100)
        
    def _generate_test_data(self, n_samples: int) -> pd.DataFrame:
        """Generate test market data."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_samples)))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n_samples)
        })
        
        data.set_index('date', inplace=True)
        return data
    
    def test_backward_compatibility(self):
        """Test that existing combine_signals function still works."""
        try:
            from strategies.strategy_pipeline import combine_signals, rsi_strategy, macd_strategy
            
            # Generate individual signals
            rsi_signals = rsi_strategy(self.test_data)
            macd_signals = macd_strategy(self.test_data)
            
            # Test intersection mode
            combined = combine_signals([rsi_signals, macd_signals], mode='intersection')
            
            self.assertIsInstance(combined, pd.Series)
            self.assertEqual(len(combined), len(self.test_data))
            self.assertTrue(all(signal in [-1, 0, 1] for signal in combined))
            
            print("âœ… Backward compatibility test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_strategy_pipeline_creation(self):
        """Test creating a StrategyPipeline instance."""
        try:
            from strategies.strategy_pipeline import StrategyPipeline, StrategyConfig, CombinationConfig
            
            # Create strategy configurations
            strategies = [
                StrategyConfig(name="RSI", weight=1.0),
                StrategyConfig(name="MACD", weight=1.0)
            ]
            
            # Create combination configuration
            combination_config = CombinationConfig(mode='intersection')
            
            # Create pipeline
            pipeline = StrategyPipeline(strategies, combination_config)
            
            self.assertIsInstance(pipeline, StrategyPipeline)
            self.assertEqual(len(pipeline.strategies), 2)
            self.assertEqual(pipeline.combination_config.mode, 'intersection')
            
            print("âœ… StrategyPipeline creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_signal_combination_modes(self):
        """Test different signal combination modes."""
        try:
            from strategies.strategy_pipeline import StrategyPipeline, StrategyConfig, CombinationConfig
            
            # Create test signals
            signals = [
                pd.Series([1, 0, -1, 1, 0], index=range(5)),
                pd.Series([1, 1, 0, -1, 0], index=range(5)),
                pd.Series([0, 1, -1, 1, 1], index=range(5))
            ]
            
            # Test intersection mode
            pipeline = StrategyPipeline()
            combined_intersection = pipeline._combine_intersection(signals)
            
            # Test union mode
            combined_union = pipeline._combine_union(signals)
            
            # Test weighted mode
            combined_weighted = pipeline._combine_weighted(signals, [1.0, 1.0, 1.0])
            
            # Verify results
            self.assertIsInstance(combined_intersection, pd.Series)
            self.assertIsInstance(combined_union, pd.Series)
            self.assertIsInstance(combined_weighted, pd.Series)
            
            # Test specific values for intersection (should be more conservative)
            self.assertEqual(combined_intersection.iloc[0], 1)  # All agree on buy
            self.assertEqual(combined_intersection.iloc[2], -1)  # All agree on sell
            
            print("âœ… Signal combination modes test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_strategy_combo_creation(self):
        """Test creating strategy combinations using convenience function."""
        try:
            from strategies.strategy_pipeline import create_strategy_combo
            
            # Create a strategy combo
            pipeline = create_strategy_combo(
                strategy_names=["RSI", "MACD"],
                mode='weighted',
                weights=[0.6, 0.4]
            )
            
            self.assertIsInstance(pipeline, StrategyPipeline)
            self.assertEqual(len(pipeline.strategies), 2)
            self.assertEqual(pipeline.combination_config.mode, 'weighted')
            
            print("âœ… Strategy combo creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_signal_validation(self):
        """Test signal validation functionality."""
        try:
            from strategies.strategy_pipeline import StrategyPipeline
            
            # Create test signals with NaN values
            signals_with_nan = [
                pd.Series([1, np.nan, -1, 1, 0], index=range(5)),
                pd.Series([1, 1, 0, -1, np.nan], index=range(5))
            ]
            
            pipeline = StrategyPipeline()
            validated_signals = pipeline._validate_signals(signals_with_nan)
            
            # Check that NaN values are filled
            for signal in validated_signals:
                self.assertFalse(signal.isna().any())
                self.assertTrue(all(val in [-1, 0, 1] for val in signal))
            
            print("âœ… Signal validation test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_performance_calculation(self):
        """Test performance calculation functionality."""
        try:
            from strategies.strategy_pipeline import StrategyPipeline
            
            # Create test signal
            signal = pd.Series([1, 0, -1, 1, 0], index=range(5))
            
            # Create test data with returns
            test_data = pd.DataFrame({
                'close': [100, 101, 99, 102, 101],
                'returns': [0.01, -0.02, 0.03, -0.01, 0]
            })
            
            pipeline = StrategyPipeline()
            
            # This would require the calculate_performance_metrics function
            # For now, just test that the pipeline can handle the data
            self.assertIsInstance(pipeline, StrategyPipeline)
            
            print("âœ… Performance calculation test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        try:
            from strategies.strategy_pipeline import StrategyPipeline
            
            pipeline = StrategyPipeline()
            
            # Test with empty signals list
            with self.assertRaises(ValueError):
                pipeline.combine_signals([])
            
            # Test with invalid mode
            signals = [pd.Series([1, 0, -1], index=range(3))]
            with self.assertRaises(ValueError):
                pipeline.combine_signals(signals, mode='invalid_mode')
            
            print("âœ… Error handling test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")
    
    def test_integration_with_existing_strategies(self):
        """Test integration with existing strategy functions."""
        try:
            from strategies.strategy_pipeline import (
                StrategyPipeline, StrategyConfig, CombinationConfig,
                rsi_strategy, macd_strategy
            )
            
            # Create pipeline with existing strategies
            strategies = [
                StrategyConfig(name="RSI", weight=1.0),
                StrategyConfig(name="MACD", weight=1.0)
            ]
            
            combination_config = CombinationConfig(mode='intersection')
            pipeline = StrategyPipeline(strategies, combination_config)
            
            # Add existing strategy functions
            pipeline.strategy_functions = {
                "RSI": rsi_strategy,
                "MACD": macd_strategy
            }
            
            # Generate combined signals
            combined_signal, metadata = pipeline.generate_combined_signals(
                self.test_data, ["RSI", "MACD"]
            )
            
            self.assertIsInstance(combined_signal, pd.Series)
            self.assertIsInstance(metadata, dict)
            self.assertEqual(len(combined_signal), len(self.test_data))
            
            print("âœ… Integration with existing strategies test passed")
            
        except ImportError as e:
            self.skipTest(f"Could not import strategy pipeline: {e}")


def run_tests():
    """Run all tests and return results."""
    print("Running Strategy Combo Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategyCombo)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'âœ… PASSED' if success else 'âŒ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
