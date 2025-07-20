#!/usr/bin/env python3
"""
Example script showing how to use joblib caching with long-running model operations.

This demonstrates the caching functionality for expensive model operations
like LSTM and XGBoost forecasts.
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from model_cache import (
        get_model_cache,
        clear_model_cache,
        get_cache_info,
        cached_lstm_forecast,
        cached_xgboost_forecast,
        create_cached_forecast_function
    )
    print("✅ Successfully imported model caching utilities")
except ImportError as e:
    print(f"❌ Failed to import model caching utilities: {e}")
    sys.exit(1)

def create_sample_trading_data(n_samples=100):
    """Create realistic trading data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')

    # Create realistic price data with trend and volatility
    base_price = 100.0
    trend = np.linspace(0, 0.2, n_samples)  # 20% annual trend
    volatility = 0.02  # 2% daily volatility

    returns = np.random.normal(0.001, volatility, n_samples) + trend / 252
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'strategy': 'LSTM_Strategy',
        'signal': np.random.choice(['BUY', 'SELL', 'HOLD'], n_samples),
        'entry_price': prices,
        'exit_price': prices * (1 + np.random.uniform(-0.05, 0.05, n_samples)),
        'size': np.random.randint(1, 100, n_samples),
        'pnl': np.random.normal(0, 100, n_samples),
        'return': np.random.normal(0, 0.05, n_samples)
    }, index=dates)

    return data

def demonstrate_basic_caching():
    """Demonstrate basic caching functionality."""
    print("\n🔧 Basic Caching Demonstration")
    print("-" * 40)

    # Create sample data
    data = create_sample_trading_data(50)

    # Example 1: LSTM Forecast
    print("Example 1: LSTM Forecast Caching")

    lstm_config = {
        'input_size': 4,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 10,
        'feature_columns': ['entry_price', 'size', 'pnl', 'return'],
        'target_column': 'entry_price'
    }

    print("First LSTM forecast (will be slow)...")
    start_time = time.time()
    try:
        result1 = cached_lstm_forecast(data, lstm_config, horizon=10)
        time1 = time.time() - start_time
        print(f"First run took: {time1:.2f} seconds")
        print(f"Forecast shape: {result1['forecast'].shape}")
    except Exception as e:
        print(f"LSTM forecast failed: {e}")
        return False

    print("Second LSTM forecast (should be fast due to caching)...")
    start_time = time.time()
    try:
        result2 = cached_lstm_forecast(data, lstm_config, horizon=10)
        time2 = time.time() - start_time
        print(f"Second run took: {time2:.2f} seconds")

        # Verify results are identical
        if np.allclose(result1['forecast'], result2['forecast']):
            print("✅ Cached results are identical")
        else:
            print("❌ Cached results differ")

        # Check performance improvement
        if time2 < time1 * 0.5:
            print("✅ Significant performance improvement from caching")
        else:
            print("⚠️ Limited performance improvement")

    except Exception as e:
        print(f"Second LSTM forecast failed: {e}")
        return False

    return True

def demonstrate_xgboost_caching():
    """Demonstrate XGBoost caching functionality."""
    print("\n🔧 XGBoost Caching Demonstration")
    print("-" * 40)

    # Create sample data
    data = create_sample_trading_data(80)

    xgboost_config = {
        'auto_feature_engineering': False,
        'xgboost_params': {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }

    print("First XGBoost forecast (will be slow)...")
    start_time = time.time()
    try:
        result1 = cached_xgboost_forecast(data, xgboost_config, horizon=10)
        time1 = time.time() - start_time
        print(f"First run took: {time1:.2f} seconds")
        print(f"Forecast shape: {result1['forecast'].shape}")
    except Exception as e:
        print(f"XGBoost forecast failed: {e}")
        return False

    print("Second XGBoost forecast (should be fast due to caching)...")
    start_time = time.time()
    try:
        result2 = cached_xgboost_forecast(data, xgboost_config, horizon=10)
        time2 = time.time() - start_time
        print(f"Second run took: {time2:.2f} seconds")

        # Verify results are identical
        if np.allclose(result1['forecast'], result2['forecast']):
            print("✅ Cached results are identical")
        else:
            print("❌ Cached results differ")

        # Check performance improvement
        if time2 < time1 * 0.5:
            print("✅ Significant performance improvement from caching")
        else:
            print("⚠️ Limited performance improvement")

    except Exception as e:
        print(f"Second XGBoost forecast failed: {e}")
        return False

    return True

def demonstrate_cache_management():
    """Demonstrate cache management features."""
    print("\n🔧 Cache Management Demonstration")
    print("-" * 40)

    # Get cache information
    cache_info = get_cache_info()
    print("Current cache information:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")

    # Clear cache
    print("\nClearing cache...")
    clear_model_cache()

    # Get cache information after clearing
    cache_info_after = get_cache_info()
    print("Cache information after clearing:")
    for key, value in cache_info_after.items():
        print(f"  {key}: {value}")

    print("✅ Cache management demonstration completed")

def demonstrate_dynamic_caching():
    """Demonstrate dynamic caching with different parameters."""
    print("\n🔧 Dynamic Caching Demonstration")
    print("-" * 40)

    # Create sample data
    data = create_sample_trading_data(60)

    # Test with different horizons
    horizons = [5, 10, 15]

    for horizon in horizons:
        print(f"\nTesting forecast with horizon={horizon}")

        # Create config with horizon-specific parameters
        config = {
            'input_size': 4,
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': min(10, horizon),
            'feature_columns': ['entry_price', 'size', 'pnl', 'return'],
            'target_column': 'entry_price'
        }

        start_time = time.time()
        try:
            result = cached_lstm_forecast(data, config, horizon=horizon)
            time_taken = time.time() - start_time
            print(f"  Horizon {horizon}: {time_taken:.2f}s, Forecast shape: {result['forecast'].shape}")
        except Exception as e:
            print(f"  Horizon {horizon}: Failed - {e}")

    print("✅ Dynamic caching demonstration completed")

def main():
    """Run all caching demonstrations."""
    print("🚀 Model Caching Usage Examples")
    print("=" * 50)

    # Show initial cache state
    print("Initial cache state:")
    cache_info = get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")

    # Run demonstrations
    results = []

    results.append(demonstrate_basic_caching())
    results.append(demonstrate_xgboost_caching())
    demonstrate_cache_management()
    demonstrate_dynamic_caching()

    # Final cache state
    print("\nFinal cache state:")
    cache_info = get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Demonstration Summary:")
    print(f"✅ Successful demonstrations: {sum(results)}")
    print(f"❌ Failed demonstrations: {len(results) - sum(results)}")

    if all(results):
        print("🎉 All demonstrations completed successfully!")
        print("\n💡 Key Benefits of Model Caching:")
        print("  • Faster subsequent model runs")
        print("  • Reduced computational costs")
        print("  • Consistent results for same inputs")
        print("  • Automatic cache management")
    else:
        print("⚠️ Some demonstrations failed. Check error messages above.")

    print("\n🔧 Usage Tips:")
    print("  • Use @cache_model_operation decorator for expensive functions")
    print("  • Cache is automatically managed by joblib")
    print("  • Clear cache with clear_model_cache() when needed")
    print("  • Monitor cache size with get_cache_info()")

if __name__ == "__main__":
    main()

