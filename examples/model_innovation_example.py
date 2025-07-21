"""
Model Innovation Agent Example

This example demonstrates how to use the ModelInnovationAgent to automatically
discover and integrate new forecasting models into the ensemble.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from agents.model_innovation_agent import (
    InnovationConfig,
    create_model_innovation_agent,
)
from utils.service_utils import create_sample_market_data
from utils.weight_registry import get_registry_summary, get_weight_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample financial data for demonstration.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with features and target
    """
    logger.info(f"Creating sample data with {n_samples} samples")

    # Use shared utility for basic market data
    basic_data = create_sample_market_data(n_samples)
    basic_data["data"]

    # Generate dates
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")

    # Generate features with realistic financial patterns
    np.random.seed(42)

    # Price-based features
    base_price = 100
    price_returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
    prices = [base_price]
    for ret in price_returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Technical indicators
    prices_series = pd.Series(prices, index=dates)

    # Moving averages
    ma_20 = prices_series.rolling(window=20).mean()
    ma_50 = prices_series.rolling(window=50).mean()

    # Volatility
    volatility = prices_series.rolling(window=20).std()

    # RSI-like feature
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Volume (correlated with price movement)
    volume = np.random.lognormal(10, 0.5, n_samples) * (1 + abs(price_returns) * 10)

    # Create target (next day's return)
    target = pd.Series(price_returns).shift(-1)

    # Create DataFrame with enhanced features
    enhanced_data = pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "volatility": volatility,
            "rsi": rsi,
            "volume": volume,
            "price_change": prices_series.pct_change(),
            "target": target,
        }
    )

    # Remove NaN values
    enhanced_data = enhanced_data.dropna()

    logger.info(f"Created data with shape: {enhanced_data.shape}")
    return enhanced_data


def setup_initial_models():
    """Set up initial models in the registry for comparison."""
    logger.info("Setting up initial models in registry")

    registry = get_weight_registry()

    # Register some initial models
    initial_models = [
        {
            "name": "baseline_linear",
            "type": "linear",
            "weights": {"base_weight": 0.5},
            "performance": {
                "mse": 1.0,
                "sharpe_ratio": 0.3,
                "r2_score": 0.2,
                "max_drawdown": -0.1,
                "total_return": 0.05,
                "volatility": 0.15,
            },
        },
        {
            "name": "baseline_tree",
            "type": "tree",
            "weights": {"base_weight": 0.5},
            "performance": {
                "mse": 0.8,
                "sharpe_ratio": 0.4,
                "r2_score": 0.25,
                "max_drawdown": -0.08,
                "total_return": 0.06,
                "volatility": 0.12,
            },
        },
    ]

    for model_info in initial_models:
        registry.register_model(
            model_name=model_info["name"],
            model_type=model_info["type"],
            initial_weights=model_info["weights"],
        )
        registry.update_performance(model_info["name"], model_info["performance"])

    logger.info("Initial models registered")


def run_innovation_demo():
    """Run the model innovation demonstration."""
    logger.info("Starting Model Innovation Agent Demo")

    # Step 1: Create sample data
    data = create_sample_data(n_samples=500)

    # Step 2: Set up initial models
    setup_initial_models()

    # Step 3: Create innovation agent
    config = InnovationConfig(
        automl_time_budget=60,  # 1 minute for demo
        max_models_per_search=5,
        min_improvement_threshold=0.02,  # 2% improvement required
        enable_linear_models=True,
        enable_tree_models=True,
        enable_neural_models=False,  # Disable for demo (requires PyTorch)
        enable_ensemble_models=True,
    )

    agent = create_model_innovation_agent(config)

    # Step 4: Run innovation cycle
    logger.info("Running innovation cycle...")
    start_time = datetime.now()

    results = agent.run_innovation_cycle(data, target_col="target")

    end_time = datetime.now()
    cycle_duration = (end_time - start_time).total_seconds()

    # Step 5: Display results
    logger.info("=" * 60)
    logger.info("INNOVATION CYCLE RESULTS")
    logger.info("=" * 60)

    logger.info(f"Cycle Duration: {cycle_duration:.2f} seconds")
    logger.info(f"Candidates Discovered: {results['candidates_discovered']}")
    logger.info(f"Candidates Evaluated: {results['candidates_evaluated']}")
    logger.info(f"Improvements Found: {results['improvements_found']}")
    logger.info(f"Models Integrated: {results['models_integrated']}")

    if results["errors"]:
        logger.warning(f"Errors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            logger.warning(f"  - {error}")

    # Step 6: Get innovation statistics
    stats = agent.get_innovation_statistics()

    logger.info("\n" + "=" * 60)
    logger.info("INNOVATION STATISTICS")
    logger.info("=" * 60)

    logger.info(f"Total Innovation Cycles: {stats['total_cycles']}")
    logger.info(f"Total Models Integrated: {stats['total_models_integrated']}")
    logger.info(f"Total Evaluations: {stats['total_evaluations']}")

    logger.info(f"\nModel Type Distribution:")
    for model_type, count in stats["model_type_distribution"].items():
        logger.info(f"  {model_type}: {count}")

    logger.info(f"\nRecent Performance Improvements:")
    for improvement in stats["performance_improvements"][-3:]:  # Last 3
        logger.info(
            f"  {improvement['timestamp']}: {improvement['model_name']} ({improvement['model_type']})"
        )
        logger.info(f"    MSE: {improvement['metrics']['mse']:.4f}")
        logger.info(f"    Sharpe: {improvement['metrics']['sharpe_ratio']:.4f}")
        logger.info(f"    RÂ²: {improvement['metrics']['r2_score']:.4f}")

    # Step 7: Display final registry state
    registry_summary = get_registry_summary()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL REGISTRY STATE")
    logger.info("=" * 60)

    logger.info(f"Total Models: {registry_summary['total_models']}")
    logger.info(f"Model Types: {registry_summary['model_types']}")
    logger.info(f"Performance Records: {registry_summary['total_performance_records']}")
    logger.info(f"Optimizations: {registry_summary['total_optimizations']}")

    return results, stats


def run_continuous_innovation():
    """Run continuous innovation over multiple cycles."""
    logger.info("Starting Continuous Innovation Demo")

    # Create agent with longer time budget
    config = InnovationConfig(
        automl_time_budget=120,  # 2 minutes per cycle
        max_models_per_search=3,
        min_improvement_threshold=0.01,  # 1% improvement required
        enable_linear_models=True,
        enable_tree_models=True,
        enable_neural_models=False,
        enable_ensemble_models=True,
    )

    agent = create_model_innovation_agent(config)

    # Set up initial models
    setup_initial_models()

    # Run multiple cycles
    n_cycles = 3
    all_results = []

    for cycle in range(n_cycles):
        logger.info(f"\n{'=' * 20} CYCLE {cycle + 1}/{n_cycles} {'=' * 20}")

        # Create fresh data for each cycle
        data = create_sample_data(n_samples=300 + cycle * 50)
        _unused_var = data  # Placeholder, flake8 ignore: F841

        # Run innovation cycle
        results = agent.run_innovation_cycle(data, target_col="target")
        all_results.append(results)

        # Display cycle results
        logger.info(f"Cycle {cycle + 1} Results:")
        logger.info(f"  Models Integrated: {results['models_integrated']}")
        logger.info(f"  Improvements Found: {results['improvements_found']}")
        logger.info(f"  Cycle Time: {results.get('cycle_time_seconds', 0):.2f}s")

        # Get current statistics
        stats = agent.get_innovation_statistics()
        logger.info(f"  Total Models in Registry: {stats['total_models_integrated']}")

        # Wait between cycles (simulate real-world scenario)
        if cycle < n_cycles - 1:
            logger.info("Waiting 5 seconds before next cycle...")
            import time

            time.sleep(5)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("CONTINUOUS INNOVATION SUMMARY")
    logger.info("=" * 60)

    total_integrated = sum(r["models_integrated"] for r in all_results)
    total_improvements = sum(r["improvements_found"] for r in all_results)
    total_time = sum(r.get("cycle_time_seconds", 0) for r in all_results)

    logger.info(f"Total Cycles: {n_cycles}")
    logger.info(f"Total Models Integrated: {total_integrated}")
    logger.info(f"Total Improvements Found: {total_improvements}")
    logger.info(f"Total Time: {total_time:.2f} seconds")

    # Final registry state
    final_stats = agent.get_innovation_statistics()
    logger.info(f"Final Model Count: {final_stats['total_models_integrated']}")

    return all_results, final_stats


def demonstrate_model_evaluation():
    """Demonstrate individual model evaluation."""
    logger.info("Demonstrating Individual Model Evaluation")

    # Create sample data
    data = create_sample_data(n_samples=200)

    # Create agent
    agent = create_model_innovation_agent()

    # Discover some models
    candidates = agent.discover_models(data, target_col="target")

    if not candidates:
        logger.warning("No candidates discovered")
        return

    logger.info(f"Discovered {len(candidates)} candidates")

    # Evaluate each candidate
    for i, candidate in enumerate(candidates[:3]):  # Evaluate first 3
        logger.info(f"\nEvaluating candidate {i + 1}: {candidate.name}")

        evaluation = agent.evaluate_candidate(candidate, data, target_col="target")

        logger.info(f"  MSE: {evaluation.mse:.4f}")
        logger.info(f"  MAE: {evaluation.mae:.4f}")
        logger.info(f"  RÂ² Score: {evaluation.r2_score:.4f}")
        logger.info(f"  Sharpe Ratio: {evaluation.sharpe_ratio:.4f}")
        logger.info(f"  Max Drawdown: {evaluation.max_drawdown:.4f}")
        logger.info(f"  Total Return: {evaluation.total_return:.4f}")
        logger.info(f"  Training Time: {evaluation.training_time:.2f}s")
        logger.info(f"  Model Size: {evaluation.model_size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Innovation Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["single", "continuous", "evaluation"],
        default="single",
        help="Demo mode",
    )
    parser.add_argument(
        "--cycles", type=int, default=3, help="Number of cycles for continuous mode"
    )

    args = parser.parse_args()

    try:
        if args.mode == "single":
            results, stats = run_innovation_demo()
        elif args.mode == "continuous":
            results, stats = run_continuous_innovation()
        elif args.mode == "evaluation":
            demonstrate_model_evaluation()

        logger.info("\nDemo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
