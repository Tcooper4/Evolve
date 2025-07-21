"""
Example: Model Caching Usage

This module demonstrates how to use the model caching system for
improving performance and reducing computational overhead.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

try:
    from model_cache import (
        cached_lstm_forecast,
        cached_xgboost_forecast,
        clear_model_cache,
        create_cached_forecast_function,
        get_cache_info,
    )

    print("✅ Successfully imported model caching utilities")
except ImportError as e:
    print(f"❌ Failed to import model caching utilities: {e}")
    sys.exit(1)


def create_sample_trading_data(n_samples=100):
    """Create realistic trading data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

    # Create realistic price data with trend and volatility
    base_price = 100.0
    trend = np.linspace(0, 0.2, n_samples)  # 20% annual trend
    volatility = 0.02  # 2% daily volatility

    returns = np.random.normal(0.001, volatility, n_samples) + trend / 252
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "strategy": "LSTM_Strategy",
            "signal": np.random.choice(["BUY", "SELL", "HOLD"], n_samples),
            "entry_price": prices,
            "exit_price": prices * (1 + np.random.uniform(-0.05, 0.05, n_samples)),
            "size": np.random.randint(1, 100, n_samples),
            "pnl": np.random.normal(0, 100, n_samples),
            "return": np.random.normal(0, 0.05, n_samples),
        },
        index=dates,
    )

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
        "input_size": 4,
        "hidden_size": 32,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 10,
        "feature_columns": ["entry_price", "size", "pnl", "return"],
        "target_column": "entry_price",
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
        if np.allclose(result1["forecast"], result2["forecast"]):
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
    data = create_sample_trading_data(100)

    # Example 2: XGBoost Forecast
    print("Example 2: XGBoost Forecast Caching")

    xgb_config = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "feature_columns": ["entry_price", "size", "pnl", "return"],
        "target_column": "entry_price",
    }

    print("First XGBoost forecast...")
    start_time = time.time()
    try:
        result1 = cached_xgboost_forecast(data, xgb_config, horizon=5)
        time1 = time.time() - start_time
        print(f"First run took: {time1:.2f} seconds")
        print(f"Forecast shape: {result1['forecast'].shape}")
    except Exception as e:
        print(f"XGBoost forecast failed: {e}")
        return False

    print("Second XGBoost forecast (cached)...")
    start_time = time.time()
    try:
        result2 = cached_xgboost_forecast(data, xgb_config, horizon=5)
        time2 = time.time() - start_time
        print(f"Second run took: {time2:.2f} seconds")

        # Verify results are identical
        if np.allclose(result1["forecast"], result2["forecast"]):
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
    """Demonstrate cache management functionality."""
    print("\n🔧 Cache Management Demonstration")
    print("-" * 40)

    # Get cache info
    cache_info = get_cache_info()
    print(f"Current cache size: {cache_info['size']} items")
    print(f"Cache memory usage: {cache_info['memory_usage']:.2f} MB")

    # Clear cache
    print("Clearing cache...")
    clear_model_cache()

    # Check cache after clearing
    cache_info_after = get_cache_info()
    print(f"Cache size after clearing: {cache_info_after['size']} items")
    print(
        f"Cache memory usage after clearing: {cache_info_after['memory_usage']:.2f} MB"
    )

    return True


def demonstrate_dynamic_caching():
    """Demonstrate dynamic caching with custom functions."""
    print("\n🔧 Dynamic Caching Demonstration")
    print("-" * 40)

    # Create sample data
    data = create_sample_trading_data(75)

    # Create a custom forecast function
    def custom_forecast(data, config, horizon=5):
        """Custom forecast function for demonstration."""
        # Simulate some computation
        time.sleep(0.1)

        # Simple moving average forecast
        prices = data["entry_price"].values
        forecast = np.mean(prices[-config.get("window", 10):]) * np.ones(horizon)

        return {
            "forecast": forecast,
            "confidence": 0.8,
            "metadata": {"method": "custom_ma", "window": config.get("window", 10)},
        }

    # Create cached version
    cached_custom_forecast = create_cached_forecast_function(
        custom_forecast, cache_key_prefix="custom_ma"
    )

    config = {"window": 15}

    print("First custom forecast (will be slow)...")
    start_time = time.time()
    result1 = cached_custom_forecast(data, config, horizon=5)
    time1 = time.time() - start_time
    print(f"First run took: {time1:.2f} seconds")

    print("Second custom forecast (should be fast)...")
    start_time = time.time()
    result2 = cached_custom_forecast(data, config, horizon=5)
    time2 = time.time() - start_time
    print(f"Second run took: {time2:.2f} seconds")

    # Verify results are identical
    if np.allclose(result1["forecast"], result2["forecast"]):
        print("✅ Cached results are identical")
    else:
        print("❌ Cached results differ")

    # Check performance improvement
    if time2 < time1 * 0.5:
        print("✅ Significant performance improvement from caching")
    else:
        print("⚠️ Limited performance improvement")

    return True


def main():
    """Main demonstration function."""
    print("🚀 Model Caching Usage Examples")
    print("=" * 50)

    try:
        # Demonstrate basic caching
        if not demonstrate_basic_caching():
            print("❌ Basic caching demonstration failed")
            return

        # Demonstrate XGBoost caching
        if not demonstrate_xgboost_caching():
            print("❌ XGBoost caching demonstration failed")
            return

        # Demonstrate cache management
        if not demonstrate_cache_management():
            print("❌ Cache management demonstration failed")
            return

        # Demonstrate dynamic caching
        if not demonstrate_dynamic_caching():
            print("❌ Dynamic caching demonstration failed")
            return

        print("\n✅ All caching demonstrations completed successfully!")
        print("\n📊 Final cache statistics:")
        cache_info = get_cache_info()
        print(f"   Cache size: {cache_info['size']} items")
        print(f"   Memory usage: {cache_info['memory_usage']:.2f} MB")
        print(f"   Hit rate: {cache_info.get('hit_rate', 0):.2%}")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
