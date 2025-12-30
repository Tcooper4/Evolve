"""
Walk-Forward Backtest Example

This example demonstrates how to use the walk-forward backtest function
for model stability analysis and performance evaluation.
"""

import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample financial data for demonstration.

    Args:
        n_samples: Number of data points to generate

    Returns:
        DataFrame with sample financial data
    """
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

    # Generate price data with trend and noise
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    prices = trend + noise

    # Generate volume data
    volume = np.random.lognormal(10, 0.5, n_samples)

    # Generate technical indicators
    sma_20 = pd.Series(prices).rolling(20).mean().values
    rsi = (
        50
        + 30 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
        + np.random.normal(0, 5, n_samples)
    )
    rsi = np.clip(rsi, 0, 100)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "volume": volume,
            "sma_20": sma_20,
            "rsi": rsi,
            "returns": np.diff(prices, prepend=prices[0]) / prices,
            "volatility": pd.Series(prices).pct_change().rolling(20).std().values,
        }
    )

    # Remove NaN values
    data = data.dropna()

    logger.info(f"Generated sample data with shape: {data.shape}")
    return data


class SimpleLinearModel:
    """
    Simple linear model for demonstration purposes.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y, **kwargs):
        """Fit the linear model."""
        try:
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

            # Solve normal equation
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            self.intercept = coefficients[0]
            self.coefficients = coefficients[1:]

            return True
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            return False

    def predict(self, X):
        """Make predictions."""
        try:
            if self.coefficients is None:
                return np.zeros(X.shape[0])

            return self.intercept + np.dot(X, self.coefficients)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.zeros(X.shape[0])


class SimpleLSTMModel:
    """
    Simple LSTM-like model for demonstration purposes.
    """

    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.weights = None
        self.bias = None

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        try:
            # Simple linear transformation for demo
            if len(X.shape) == 3:  # If 3D (batch, sequence, features)
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(X_flat.shape[0]), X_flat])

            # Solve normal equation
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            self.bias = coefficients[0]
            self.weights = coefficients[1:]

            return True
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            return False

    def predict(self, X):
        """Make predictions."""
        try:
            if self.weights is None:
                return np.zeros(X.shape[0])

            # Flatten if 3D
            if len(X.shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            return self.bias + np.dot(X_flat, self.weights)
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            return np.zeros(X.shape[0])


def demonstrate_walk_forward_backtest():
    """Demonstrate walk-forward backtesting with different models."""
    logger.info("=== Walk-Forward Backtest Demo ===")

    # Generate sample data
    data = generate_sample_data(500)

    # Import the walk-forward backtest function
    from trading.backtesting.evaluator import walk_forward_backtest

    # Test with simple linear model
    logger.info("Testing Simple Linear Model...")
    linear_model = SimpleLinearModel()

    linear_results = walk_forward_backtest(
        model=linear_model,
        data=data,
        window_size=100,
        step_size=20,
        test_size=20,
        target_column="close",
        feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
    )

    # Display results
    print("\n=== Linear Model Results ===")
    print(
        f"Total windows processed: {linear_results['stability_report']['summary']['total_windows']}"
    )
    print(
        f"Successful windows: {linear_results['stability_report']['summary']['successful_windows']}"
    )
    print(
        f"Failed windows: {linear_results['stability_report']['summary']['failed_windows']}"
    )

    print("\nSharpe Ratio Statistics:")
    sharpe_stats = linear_results["stability_report"]["sharpe_ratio"]
    print(f"  Mean: {sharpe_stats['mean']:.4f}")
    print(f"  Std: {sharpe_stats['std']:.4f}")
    print(f"  Min: {sharpe_stats['min']:.4f}")
    print(f"  Max: {sharpe_stats['max']:.4f}")
    print(f"  Stability Score: {sharpe_stats['stability_score']:.4f}")

    print("\nTotal Return Statistics:")
    return_stats = linear_results["stability_report"]["total_return"]
    print(f"  Mean: {return_stats['mean']:.4f}")
    print(f"  Std: {return_stats['std']:.4f}")
    print(f"  Min: {return_stats['min']:.4f}")
    print(f"  Max: {return_stats['max']:.4f}")

    print("\nStability Analysis:")
    stability = linear_results["stability_report"]["stability_analysis"]
    print(f"  Sharpe Consistency: {stability['sharpe_consistency']:.4f}")
    print(f"  Return Consistency: {stability['return_consistency']:.4f}")
    print(f"  Overall Stability: {stability['overall_stability']:.4f}")

    # Test with LSTM-like model
    logger.info("\nTesting Simple LSTM Model...")
    lstm_model = SimpleLSTMModel(sequence_length=10)

    lstm_results = walk_forward_backtest(
        model=lstm_model,
        data=data,
        window_size=100,
        step_size=20,
        test_size=20,
        target_column="close",
        feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
    )

    # Display LSTM results
    print("\n=== LSTM Model Results ===")
    print(
        f"Total windows processed: {lstm_results['stability_report']['summary']['total_windows']}"
    )
    print(
        f"Successful windows: {lstm_results['stability_report']['summary']['successful_windows']}"
    )
    print(
        f"Failed windows: {lstm_results['stability_report']['summary']['failed_windows']}"
    )

    print("\nSharpe Ratio Statistics:")
    lstm_sharpe_stats = lstm_results["stability_report"]["sharpe_ratio"]
    print(f"  Mean: {lstm_sharpe_stats['mean']:.4f}")
    print(f"  Std: {lstm_sharpe_stats['std']:.4f}")
    print(f"  Min: {lstm_sharpe_stats['min']:.4f}")
    print(f"  Max: {lstm_sharpe_stats['max']:.4f}")
    print(f"  Stability Score: {lstm_sharpe_stats['stability_score']:.4f}")

    return linear_results, lstm_results


def demonstrate_model_comparison():
    """Demonstrate comparing multiple models using walk-forward backtesting."""
    logger.info("=== Model Comparison Demo ===")

    # Generate sample data
    data = generate_sample_data(500)

    # Import the evaluator
    from trading.backtesting.evaluator import ModelEvaluator

    # Create evaluator
    evaluator = ModelEvaluator(data)

    # Test multiple models
    models = {
        "Linear": SimpleLinearModel(),
        "LSTM": SimpleLSTMModel(sequence_length=10),
    }

    results = {}

    for model_name, model in models.items():
        logger.info(f"Testing {model_name} model...")

        result = evaluator.walk_forward_backtest(
            model=model,
            data=data,
            window_size=100,
            step_size=20,
            test_size=20,
            target_column="close",
            feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
        )

        results[model_name] = result

    # Compare results
    print("\n=== Model Comparison ===")
    comparison_data = []

    for model_name, result in results.items():
        stability_report = result["stability_report"]

        comparison_data.append(
            {
                "Model": model_name,
                "Sharpe_Mean": stability_report["sharpe_ratio"]["mean"],
                "Sharpe_Std": stability_report["sharpe_ratio"]["std"],
                "Sharpe_Stability": stability_report["sharpe_ratio"]["stability_score"],
                "Return_Mean": stability_report["total_return"]["mean"],
                "Return_Std": stability_report["total_return"]["std"],
                "Overall_Stability": stability_report["stability_analysis"][
                    "overall_stability"
                ],
                "Win_Rate_Mean": stability_report["win_rate"]["mean"],
                "Directional_Accuracy_Mean": stability_report["directional_accuracy"][
                    "mean"
                ],
            }
        )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format="%.4f"))

    # Find best model
    best_sharpe_idx = comparison_df["Sharpe_Mean"].idxmax()
    best_stability_idx = comparison_df["Overall_Stability"].idxmax()

    print(f"\nBest Sharpe Ratio: {comparison_df.loc[best_sharpe_idx, 'Model']}")
    print(f"Best Stability: {comparison_df.loc[best_stability_idx, 'Model']}")

    return results


def demonstrate_different_window_sizes():
    """Demonstrate the effect of different window sizes on stability."""
    logger.info("=== Window Size Analysis Demo ===")

    # Generate sample data
    data = generate_sample_data(500)

    # Import the walk-forward backtest function
    from trading.backtesting.evaluator import walk_forward_backtest

    # Test different window sizes
    window_sizes = [50, 100, 150, 200]
    model = SimpleLinearModel()

    results = {}

    for window_size in window_sizes:
        logger.info(f"Testing window size: {window_size}")

        result = walk_forward_backtest(
            model=model,
            data=data,
            window_size=window_size,
            step_size=20,
            test_size=20,
            target_column="close",
            feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
        )

        results[window_size] = result

    # Compare window sizes
    print("\n=== Window Size Comparison ===")
    window_comparison = []

    for window_size, result in results.items():
        stability_report = result["stability_report"]

        window_comparison.append(
            {
                "Window_Size": window_size,
                "Total_Windows": stability_report["summary"]["total_windows"],
                "Sharpe_Mean": stability_report["sharpe_ratio"]["mean"],
                "Sharpe_Std": stability_report["sharpe_ratio"]["std"],
                "Sharpe_Stability": stability_report["sharpe_ratio"]["stability_score"],
                "Overall_Stability": stability_report["stability_analysis"][
                    "overall_stability"
                ],
            }
        )

    window_df = pd.DataFrame(window_comparison)
    print(window_df.to_string(index=False, float_format="%.4f"))

    return results


def demonstrate_performance_degradation_detection():
    """Demonstrate detection of performance degradation over time."""
    logger.info("=== Performance Degradation Detection Demo ===")

    # Generate sample data with performance degradation
    np.random.seed(42)
    n_samples = 500

    # Create data where model performance degrades over time
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

    # Generate price data with increasing noise (degrading signal)
    trend = np.linspace(100, 150, n_samples)
    noise_level = np.linspace(1, 5, n_samples)  # Increasing noise
    noise = np.random.normal(0, 1, n_samples) * noise_level
    prices = trend + noise

    # Generate other features
    volume = np.random.lognormal(10, 0.5, n_samples)
    sma_20 = pd.Series(prices).rolling(20).mean().values
    rsi = (
        50
        + 30 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
        + np.random.normal(0, 5, n_samples)
    )
    rsi = np.clip(rsi, 0, 100)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "volume": volume,
            "sma_20": sma_20,
            "rsi": rsi,
            "returns": np.diff(prices, prepend=prices[0]) / prices,
            "volatility": pd.Series(prices).pct_change().rolling(20).std().values,
        }
    )

    data = data.dropna()

    # Run walk-forward backtest
    from trading.backtesting.evaluator import walk_forward_backtest

    model = SimpleLinearModel()

    result = walk_forward_backtest(
        model=model,
        data=data,
        window_size=100,
        step_size=20,
        test_size=20,
        target_column="close",
        feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
    )

    # Analyze degradation
    degradation_analysis = result["stability_report"]["trend_analysis"][
        "performance_degradation"
    ]

    print("\n=== Performance Degradation Analysis ===")
    print(f"Degradation Detected: {degradation_analysis['degradation_detected']}")
    print(f"Degradation Magnitude: {degradation_analysis['magnitude']:.4f}")
    print(f"Early Period Mean Sharpe: {degradation_analysis['early_period_mean']:.4f}")
    print(f"Late Period Mean Sharpe: {degradation_analysis['late_period_mean']:.4f}")
    print(f"Confidence: {degradation_analysis['confidence']:.4f}")

    # Trend analysis
    trend_analysis = result["stability_report"]["trend_analysis"]
    print(f"\nSharpe Trend: {trend_analysis['sharpe_trend']:.6f}")
    print(f"Return Trend: {trend_analysis['return_trend']:.6f}")

    return result


def demonstrate_detailed_analysis():
    """Demonstrate detailed analysis of walk-forward backtest results."""
    logger.info("=== Detailed Analysis Demo ===")

    # Generate sample data
    data = generate_sample_data(500)

    # Import the evaluator
    from trading.backtesting.evaluator import ModelEvaluator

    # Create evaluator
    evaluator = ModelEvaluator(data)

    # Run walk-forward backtest
    model = SimpleLinearModel()

    result = evaluator.walk_forward_backtest(
        model=model,
        data=data,
        window_size=100,
        step_size=20,
        test_size=20,
        target_column="close",
        feature_columns=["volume", "sma_20", "rsi", "returns", "volatility"],
    )

    # Detailed analysis
    print("\n=== Detailed Analysis ===")

    # Window-by-window analysis
    window_results = result["window_results"]
    print(f"Number of windows: {len(window_results)}")

    # Find best and worst windows
    sharpe_ratios = [w["metrics"]["sharpe_ratio"] for w in window_results]
    best_window_idx = np.argmax(sharpe_ratios)
    worst_window_idx = np.argmin(sharpe_ratios)

    print(f"\nBest Window (Index {best_window_idx}):")
    best_window = window_results[best_window_idx]
    print(f"  Sharpe Ratio: {best_window['metrics']['sharpe_ratio']:.4f}")
    print(f"  Total Return: {best_window['metrics']['total_return']:.4f}")
    print(f"  Win Rate: {best_window['metrics']['win_rate']:.4f}")
    print(
        f"  Directional Accuracy: {best_window['metrics']['directional_accuracy']:.4f}"
    )

    print(f"\nWorst Window (Index {worst_window_idx}):")
    worst_window = window_results[worst_window_idx]
    print(f"  Sharpe Ratio: {worst_window['metrics']['sharpe_ratio']:.4f}")
    print(f"  Total Return: {worst_window['metrics']['total_return']:.4f}")
    print(f"  Win Rate: {worst_window['metrics']['win_rate']:.4f}")
    print(
        f"  Directional Accuracy: {worst_window['metrics']['directional_accuracy']:.4f}"
    )

    # Consistency analysis
    stability_report = result["stability_report"]
    print(f"\nConsistency Analysis:")
    print(
        f"  Sharpe Consistency: {stability_report['stability_analysis']['sharpe_consistency']:.4f}"
    )
    print(
        f"  Return Consistency: {stability_report['stability_analysis']['return_consistency']:.4f}"
    )
    print(
        f"  Drawdown Consistency: {stability_report['stability_analysis']['drawdown_consistency']:.4f}"
    )

    # Performance distribution
    print(f"\nPerformance Distribution:")
    print(
        f"  Sharpe Ratio - Mean: {stability_report['sharpe_ratio']['mean']:.4f}, "
        f"Std: {stability_report['sharpe_ratio']['std']:.4f}"
    )
    print(
        f"  Total Return - Mean: {stability_report['total_return']['mean']:.4f}, "
        f"Std: {stability_report['total_return']['std']:.4f}"
    )
    print(
        f"  Max Drawdown - Mean: {stability_report['max_drawdown']['mean']:.4f}, "
        f"Std: {stability_report['max_drawdown']['std']:.4f}"
    )

    return result


def main():
    """Run all demonstration functions."""
    logger.info("Starting Walk-Forward Backtest Examples")
    logger.info("=" * 50)

    try:
        # Basic walk-forward backtest
        linear_results, lstm_results = demonstrate_walk_forward_backtest()
        logger.info("-" * 30)

        # Model comparison
        demonstrate_model_comparison()
        logger.info("-" * 30)

        # Window size analysis
        demonstrate_different_window_sizes()
        logger.info("-" * 30)

        # Performance degradation detection
        demonstrate_performance_degradation_detection()
        logger.info("-" * 30)

        # Detailed analysis
        demonstrate_detailed_analysis()
        logger.info("-" * 30)

        # Summary
        logger.info("=== SUMMARY ===")
        logger.info("All walk-forward backtest examples completed successfully!")
        logger.info("Key insights:")
        logger.info("- Walk-forward backtesting provides stability analysis")
        logger.info("- Different models can be compared systematically")
        logger.info("- Window size affects stability and performance")
        logger.info("- Performance degradation can be detected over time")
        logger.info("- Detailed analysis reveals model characteristics")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
