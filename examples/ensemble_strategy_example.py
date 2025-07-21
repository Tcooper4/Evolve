"""
Example: Using Weighted Ensemble Strategy

This example demonstrates how to use the WeightedEnsembleStrategy to combine
multiple trading strategies into a single ensemble signal.
"""

import numpy as np
import pandas as pd

# Import individual strategies
from trading.strategies import get_signals

# Import the ensemble strategy
from trading.strategies.ensemble import (
    EnsembleConfig,
    WeightedEnsembleStrategy,
    create_balanced_ensemble,
    create_conservative_ensemble,
    create_rsi_macd_bollinger_ensemble,
)


def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Generate realistic price data with some trend and volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.005, days)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, days),
        }
    )

    data.set_index("Date", inplace=True)
    return data


def example_basic_ensemble():
    """Example 1: Basic ensemble strategy usage."""
    print("=== Example 1: Basic Ensemble Strategy ===")

    # Generate sample data
    data = generate_sample_data(100)
    print(f"Generated {len(data)} days of sample data")

    # Create ensemble strategy with default weights (RSI: 40%, MACD: 40%, Bollinger: 20%)
    ensemble = create_rsi_macd_bollinger_ensemble()

    # Generate individual strategy signals
    strategy_signals = {}

    # RSI signals
    rsi_signals = get_signals("rsi", data)
    strategy_signals["rsi"] = rsi_signals["result"]
    print(f"RSI signals generated: {len(rsi_signals['result'])}")

    # MACD signals
    macd_signals = get_signals("macd", data)
    strategy_signals["macd"] = macd_signals["result"]
    print(f"MACD signals generated: {len(macd_signals['result'])}")

    # Bollinger Bands signals
    bollinger_signals = get_signals("bollinger", data)
    strategy_signals["bollinger"] = bollinger_signals["result"]
    print(f"Bollinger signals generated: {len(bollinger_signals['result'])}")

    # Combine signals using ensemble
    combined_signals = ensemble.combine_signals(strategy_signals)
    print(f"Combined ensemble signals: {len(combined_signals)}")

    # Display some results
    print("\nEnsemble Signal Summary:")
    print(f"Buy signals: {(combined_signals['signal'] > 0).sum()}")
    print(f"Sell signals: {(combined_signals['signal'] < 0).sum()}")
    print(f"Hold signals: {(combined_signals['signal'] == 0).sum()}")
    print(f"Average confidence: {combined_signals['confidence'].mean():.3f}")
    print(f"Average consensus: {combined_signals['consensus'].mean():.3f}")

    return ensemble, combined_signals


def example_custom_weights():
    """Example 2: Custom strategy weights."""
    print("\n=== Example 2: Custom Strategy Weights ===")

    # Generate sample data
    data = generate_sample_data(100)

    # Create custom weights (favor RSI and MACD more heavily)
    custom_weights = {
        "rsi": 0.5,  # 50% weight
        "macd": 0.3,  # 30% weight
        "bollinger": 0.2,  # 20% weight
    }

    # Create ensemble with custom configuration
    config = EnsembleConfig(
        strategy_weights=custom_weights,
        combination_method="weighted_average",
        confidence_threshold=0.7,  # Higher confidence threshold
        consensus_threshold=0.6,  # Higher consensus threshold
    )

    ensemble = WeightedEnsembleStrategy(config)

    # Generate signals using the unified interface
    ensemble_signals = get_signals("ensemble", data, strategy_weights=custom_weights)

    print(f"Custom ensemble signals generated: {len(ensemble_signals['result'])}")
    print(f"Configuration: {ensemble.get_parameters()}")

    return ensemble, ensemble_signals


def example_voting_method():
    """Example 3: Using voting method instead of weighted average."""
    print("\n=== Example 3: Voting Method ===")

    # Generate sample data
    data = generate_sample_data(100)

    # Create ensemble with voting method
    config = EnsembleConfig(
        strategy_weights={"rsi": 0.33, "macd": 0.33, "bollinger": 0.34},
        combination_method="voting",
        confidence_threshold=0.5,
        consensus_threshold=0.4,
    )

    ensemble = WeightedEnsembleStrategy(config)

    # Generate individual strategy signals
    strategy_signals = {}
    for strategy_name in ["rsi", "macd", "bollinger"]:
        signals = get_signals(strategy_name, data)
        strategy_signals[strategy_name] = signals["result"]

    # Combine using voting method
    combined_signals = ensemble.combine_signals(strategy_signals, method="voting")

    print(f"Voting ensemble signals generated: {len(combined_signals)}")
    print("Voting method results:")
    print(f"  Buy signals: {(combined_signals['signal'] > 0).sum()}")
    print(f"  Sell signals: {(combined_signals['signal'] < 0).sum()}")
    print(f"  Hold signals: {(combined_signals['signal'] == 0).sum()}")

    return ensemble, combined_signals


def example_dynamic_weight_update():
    """Example 4: Dynamically updating strategy weights."""
    print("\n=== Example 4: Dynamic Weight Updates ===")

    # Generate sample data
    data = generate_sample_data(100)

    # Create initial ensemble
    ensemble = create_balanced_ensemble()
    print(f"Initial weights: {ensemble.config.strategy_weights}")

    # Generate initial signals
    strategy_signals = {}
    for strategy_name in ["rsi", "macd", "bollinger"]:
        signals = get_signals(strategy_name, data)
        strategy_signals[strategy_name] = signals["result"]

    initial_signals = ensemble.combine_signals(strategy_signals)

    # Update weights based on some criteria (e.g., performance)
    new_weights = {
        "rsi": 0.6,  # Increase RSI weight
        "macd": 0.25,  # Decrease MACD weight
        "bollinger": 0.15,  # Decrease Bollinger weight
    }

    result = ensemble.update_weights(new_weights)
    print(f"Weight update result: {result['result']['status']}")
    print(f"New weights: {ensemble.config.strategy_weights}")

    # Generate signals with new weights
    updated_signals = ensemble.combine_signals(strategy_signals)

    print("Signal comparison:")
    print(f"  Initial buy signals: {(initial_signals['signal'] > 0).sum()}")
    print(f"  Updated buy signals: {(updated_signals['signal'] > 0).sum()}")

    return ensemble, updated_signals


def example_performance_analysis():
    """Example 5: Performance analysis of ensemble strategy."""
    print("\n=== Example 5: Performance Analysis ===")

    # Generate sample data
    data = generate_sample_data(100)

    # Create different ensemble configurations
    ensembles = {
        "Conservative": create_conservative_ensemble(),
        "Balanced": create_balanced_ensemble(),
        "RSI-Heavy": create_rsi_macd_bollinger_ensemble(),
    }

    # Generate signals for each ensemble
    results = {}

    for name, ensemble in ensembles.items():
        print(f"\nAnalyzing {name} ensemble...")

        # Generate individual strategy signals
        strategy_signals = {}
        for strategy_name in ["rsi", "macd", "bollinger"]:
            signals = get_signals(strategy_name, data)
            strategy_signals[strategy_name] = signals["result"]

        # Combine signals
        combined_signals = ensemble.combine_signals(strategy_signals)

        # Get performance metrics
        metrics = ensemble.get_performance_metrics()

        print(f"  Configuration: {ensemble.get_parameters()}")
        print(f"  Total signals: {metrics['result']['total_signals']}")
        print(f"  Buy signals: {metrics['result']['buy_signals']}")
        print(f"  Sell signals: {metrics['result']['sell_signals']}")
        print(f"  Average confidence: {metrics['result']['avg_confidence']:.3f}")
        print(f"  Average consensus: {metrics['result']['avg_consensus']:.3f}")

        results[name] = {
            "ensemble": ensemble,
            "signals": combined_signals,
            "metrics": metrics,
        }

    return results


def main():
    """Run all ensemble strategy examples."""
    print("Weighted Ensemble Strategy Examples")
    print("=" * 50)

    try:
        # Run all examples
        example_basic_ensemble()
        example_custom_weights()
        example_voting_method()
        example_dynamic_weight_update()
        example_performance_analysis()

        print(
            "Ensemble strategy completed. The combined approach should provide "
            "more robust signals than individual strategies."
        )

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
