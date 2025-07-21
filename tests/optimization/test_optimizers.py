"""Test cases for optimizers."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from trading.optimization.base_optimizer import BaseOptimizer
from trading.optimization.performance_logger import (
    PerformanceLogger,
    PerformanceMetrics,
)
from trading.optimization.strategy_selection_agent import StrategySelectionAgent

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler("trading/optimization/logs/optimization_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)


class TestStrategy:
    """Test strategy class."""

    def __init__(self):
        """Initialize test strategy."""
        self.name = "TestStrategy"

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Generate trading signals.

        Args:
            data: DataFrame with price data
            params: Strategy parameters

        Returns:
            Series with trading signals
        """
        # Simple moving average crossover
        short_window = params.get("short_window", 20)
        long_window = params.get("long_window", 50)

        # Calculate moving averages
        short_ma = data["price"].rolling(window=short_window).mean()
        long_ma = data["price"].rolling(window=long_window).mean()

        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal

        return signals


def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate test data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with test data
    """
    # Generate time index
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

    # Generate price series with trend and noise
    trend = np.linspace(0, 10, n_samples)
    noise = np.random.normal(0, 1, n_samples)
    prices = 100 + trend + noise

    # Generate volume series
    volumes = np.random.lognormal(10, 1, n_samples)

    # Create DataFrame
    df = pd.DataFrame({"price": prices, "volume": volumes}, index=dates)

    return df


def get_test_config() -> Dict[str, Any]:
    """Get test configuration.

    Returns:
        Dictionary with test configuration
    """
    return {
        "name": "test_optimizer",
        "max_iterations": 10,
        "early_stopping_patience": 2,
        "learning_rate": 0.01,
        "batch_size": 32,
        "is_multi_objective": True,
        "objectives": ["sharpe_ratio", "win_rate"],
        "objective_weights": {"sharpe_ratio": 0.6, "win_rate": 0.4},
        "use_lr_scheduler": True,
        "scheduler_type": "cosine",
        "min_lr": 0.0001,
        "warmup_steps": 0,
        "save_checkpoints": True,
        "checkpoint_dir": "test_checkpoints",
        "checkpoint_frequency": 2,
        "validation_split": 0.2,
        "cross_validation_folds": 2,
    }


@pytest.fixture
def test_data():
    """Fixture for test data."""
    return generate_test_data()


@pytest.fixture
def test_strategy():
    """Fixture for test strategy."""
    return TestStrategy()


@pytest.fixture
def test_config():
    """Fixture for test configuration."""
    return get_test_config()


@pytest.fixture
def test_optimizer(test_config):
    """Fixture for test optimizer."""
    return BaseOptimizer(test_config)


@pytest.fixture
def test_agent():
    """Fixture for test agent."""
    return StrategySelectionAgent()


@pytest.fixture
def test_logger():
    """Fixture for test logger."""
    return PerformanceLogger()


def test_optimizer_initialization(test_optimizer, test_config):
    """Test optimizer initialization."""
    assert test_optimizer.config.name == test_config["name"]
    assert test_optimizer.config.max_iterations == test_config["max_iterations"]
    assert test_optimizer.config.is_multi_objective == test_config["is_multi_objective"]
    assert test_optimizer.current_iteration == 0
    assert test_optimizer.best_metrics == {}
    assert test_optimizer.best_params is None
    assert test_optimizer.metrics_history == []
    assert test_optimizer.early_stopping_counter == 0


def test_strategy_evaluation(test_optimizer, test_strategy, test_data):
    """Test strategy evaluation."""
    # Test parameters
    params = {"short_window": 20, "long_window": 50}

    # Evaluate strategy
    metrics = test_optimizer.evaluate_strategy(test_strategy, params, test_data)

    # Check metrics
    assert "sharpe_ratio" in metrics
    assert "win_rate" in metrics
    assert "max_drawdown" in metrics
    assert "mse" in metrics
    assert "alpha" in metrics

    # Check metric ranges
    assert -10 <= metrics["sharpe_ratio"] <= 10
    assert 0 <= metrics["win_rate"] <= 1
    assert 0 <= metrics["max_drawdown"] <= 1
    assert metrics["mse"] >= 0


def test_agent_strategy_selection(test_agent, test_data):
    """Test strategy selection."""
    # Available strategies
    strategies = ["TestStrategy", "RSIStrategy", "MACDStrategy"]

    # Select strategy
    strategy, confidence, explanation = test_agent.select_strategy(
        test_data, strategies
    )

    # Check selection
    assert strategy in strategies
    assert 0 <= confidence <= 1
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_performance_logging(test_logger, test_strategy):
    """Test performance logging."""
    # Create test metrics
    metrics = PerformanceMetrics(
        timestamp=datetime.utcnow(),
        strategy=test_strategy.name,
        config={"param1": 1, "param2": 2},
        sharpe_ratio=1.5,
        win_rate=0.6,
        max_drawdown=0.2,
        mse=0.01,
        alpha=0.1,
        regime="test",
        reason="Test run",
    )

    # Log metrics
    test_logger.log_metrics(metrics)

    # Load metrics
    loaded_metrics = test_logger.load_metrics(strategy=test_strategy.name)

    # Check loaded metrics
    assert len(loaded_metrics) > 0
    assert loaded_metrics[-1].strategy == test_strategy.name
    assert loaded_metrics[-1].sharpe_ratio == 1.5
    assert loaded_metrics[-1].win_rate == 0.6


def test_optimizer_visualization(test_optimizer, test_strategy, test_data):
    """Test optimizer visualization."""
    # Test parameters
    params = {"short_window": 20, "long_window": 50}

    # Evaluate strategy
    metrics = test_optimizer.evaluate_strategy(test_strategy, params, test_data)

    # Log results
    test_optimizer.log_results(
        test_strategy.name, params, metrics, "Test visualization"
    )

    # Visualize results
    test_optimizer.visualize_results("test_results")

    # Check if visualization files were created
    assert os.path.exists("test_results")


def test_multi_objective_optimization(test_optimizer, test_strategy, test_data):
    """Test multi-objective optimization."""
    # Run optimization
    optimized_params = test_optimizer.optimize(test_strategy, test_data)

    # Check optimized parameters
    assert isinstance(optimized_params, dict)
    assert "short_window" in optimized_params
    assert "long_window" in optimized_params

    # Evaluate optimized strategy
    metrics = test_optimizer.evaluate_strategy(
        test_strategy, optimized_params, test_data
    )

    # Check metrics
    assert "sharpe_ratio" in metrics
    assert "win_rate" in metrics
    assert metrics["sharpe_ratio"] > 0
    assert 0 <= metrics["win_rate"] <= 1


def test_early_stopping(test_optimizer, test_strategy, test_data):
    """Test early stopping."""
    # Set early stopping parameters
    test_optimizer.config.early_stopping_patience = 2

    # Run optimization
    test_optimizer.optimize(test_strategy, test_data)

    # Check if optimization stopped early
    assert test_optimizer.current_iteration <= test_optimizer.config.max_iterations
    assert (
        test_optimizer.early_stopping_counter
        <= test_optimizer.config.early_stopping_patience
    )


def test_checkpointing(test_optimizer, test_strategy, test_data):
    """Test checkpointing."""
    # Enable checkpointing
    test_optimizer.config.save_checkpoints = True
    test_optimizer.config.checkpoint_dir = "test_checkpoints"

    # Run optimization
    test_optimizer.optimize(test_strategy, test_data)

    # Check if checkpoints were created
    assert os.path.exists(test_optimizer.config.checkpoint_dir)
    checkpoint_files = os.listdir(test_optimizer.config.checkpoint_dir)
    assert len(checkpoint_files) > 0


def test_cross_validation(test_optimizer, test_strategy, test_data):
    """Test cross-validation."""
    # Enable cross-validation
    test_optimizer.config.cross_validation_folds = 3

    # Run optimization
    test_optimizer.optimize(test_strategy, test_data)

    # Check if cross-validation was performed
    assert (
        len(test_optimizer.metrics_history)
        >= test_optimizer.config.cross_validation_folds
    )


def test_strategy_combination(test_optimizer, test_strategy, test_data):
    """Test strategy combination."""

    # Create combined strategy
    class CombinedStrategy:
        def __init__(self):
            self.name = "CombinedStrategy"
            self.strategies = [TestStrategy(), TestStrategy()]

        def generate_signals(
            self, data: pd.DataFrame, params: Dict[str, Any]
        ) -> pd.Series:
            # Generate signals from each strategy
            signals = []
            for i, strategy in enumerate(self.strategies):
                strategy_params = {
                    k: v for k, v in params.items() if k.startswith(f"strategy_{i}_")
                }
                strategy_signals = strategy.generate_signals(data, strategy_params)
                signals.append(strategy_signals)

            # Combine signals
            combined_signals = pd.concat(signals, axis=1).mean(axis=1)
            return combined_signals

    # Run optimization
    combined_strategy = CombinedStrategy()
    optimized_params = test_optimizer.optimize(combined_strategy, test_data)

    # Check optimized parameters
    assert isinstance(optimized_params, dict)
    assert "strategy_0_short_window" in optimized_params
    assert "strategy_1_short_window" in optimized_params


def test_overfitting_detection(test_optimizer, test_strategy, test_data):
    """Test overfitting detection."""
    # Run optimization
    optimized_params = test_optimizer.optimize(test_strategy, test_data)

    # Evaluate strategy
    metrics = test_optimizer.evaluate_strategy(
        test_strategy, optimized_params, test_data
    )

    # Check for overfitting
    assert metrics["sharpe_ratio"] < 5  # Unrealistic Sharpe ratio
    assert metrics["win_rate"] < 0.9  # Unrealistic win rate
    assert metrics["max_drawdown"] > 0.1  # Reasonable drawdown


def test_dynamic_strategy_crossover_grid_search(
    test_optimizer, test_strategy, test_data
):
    """Test dynamic strategy crossover tuning using grid search."""
    print("\n🎯 Testing Dynamic Strategy Crossover Grid Search")

    # Define comprehensive parameter grid for crossover optimization
    param_grid = {
        "short_window": [5, 10, 15, 20, 25],
        "long_window": [30, 40, 50, 60, 100],
        "crossover_threshold": [0.0, 0.1, 0.2, 0.3],
        "signal_strength": [0.5, 1.0, 1.5, 2.0],
    }

    # Track all results for analysis
    all_results = []
    best_score = -np.inf
    best_params = None
    best_metrics = None

    print(f"  🔍 Testing {len(param_grid['short_window'])} x {len(param_grid['long_window'])} x {len(param_grid['crossover_threshold'])} x {len(param_grid['signal_strength'])} = {len(param_grid['short_window']) *
                                                                                                                                                                                  len(param_grid['long_window']) *
                                                                                                                                                                                  len(param_grid['crossover_threshold']) *
                                                                                                                                                                                  len(param_grid['signal_strength'])} combinations")

    # Grid search with validation
    for short in param_grid["short_window"]:
        for long in param_grid["long_window"]:
            if short >= long:
                continue  # Skip invalid combinations

            for threshold in param_grid["crossover_threshold"]:
                for strength in param_grid["signal_strength"]:
                    params = {
                        "short_window": short,
                        "long_window": long,
                        "crossover_threshold": threshold,
                        "signal_strength": strength,
                    }

                    # Evaluate strategy with current parameters
                    metrics = test_optimizer.evaluate_strategy(
                        test_strategy, params, test_data
                    )

                    # Calculate composite score
                    sharpe = metrics.get("sharpe_ratio", 0)
                    win_rate = metrics.get("win_rate", 0)
                    max_dd = abs(metrics.get("max_drawdown", 0))

                    # Multi-objective scoring
                    score = sharpe * 0.5 + win_rate * 0.3 - max_dd * 0.2

                    result = {"params": params, "metrics": metrics, "score": score}
                    all_results.append(result)

                    # Track best result
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics

    # Verify grid search results
    assert best_params is not None, "Should find best parameters"
    assert best_score > -10, "Best score should be reasonable"
    assert len(all_results) > 0, "Should have results"

    print(f"  🏆 Best crossover params: {best_params}")
    print(f"  📊 Best score: {best_score:.3f}")
    print(
        f"  📈 Best metrics: Sharpe={
            best_metrics['sharpe_ratio']:.3f}, Win Rate={
            best_metrics['win_rate']:.3f}, Max DD={
                best_metrics['max_drawdown']:.3f}")

    # Test parameter sensitivity analysis
    print(f"\n  📊 Testing parameter sensitivity...")

    sensitivity_analysis = {}
    for param_name in [
        "short_window",
        "long_window",
        "crossover_threshold",
        "signal_strength",
    ]:
        param_values = param_grid[param_name]
        param_scores = []

        for value in param_values:
            # Find results with this parameter value
            relevant_results = [
                r for r in all_results if r["params"][param_name] == value
            ]
            if relevant_results:
                avg_score = np.mean([r["score"] for r in relevant_results])
                param_scores.append((value, avg_score))

        sensitivity_analysis[param_name] = param_scores

    # Verify sensitivity analysis
    for param_name, scores in sensitivity_analysis.items():
        assert len(scores) > 0, f"Should have sensitivity data for {param_name}"
        print(f"    {param_name}: {len(scores)} values tested")

    # Test crossover threshold optimization
    print(f"\n  🎯 Testing crossover threshold optimization...")

    threshold_results = sensitivity_analysis["crossover_threshold"]
    best_threshold = max(threshold_results, key=lambda x: x[1])

    print(f"  Best threshold: {best_threshold[0]} (score: {best_threshold[1]:.3f})")

    # Verify threshold optimization
    assert best_threshold[1] > -5, "Best threshold should have reasonable score"

    # Test signal strength optimization
    print(f"\n  💪 Testing signal strength optimization...")

    strength_results = sensitivity_analysis["signal_strength"]
    best_strength = max(strength_results, key=lambda x: x[1])

    print(f"  Best strength: {best_strength[0]} (score: {best_strength[1]:.3f})")

    # Verify strength optimization
    assert best_strength[1] > -5, "Best strength should have reasonable score"

    # Test parameter interaction analysis
    print(f"\n  🔗 Testing parameter interactions...")

    # Analyze short_window vs long_window interaction
    interaction_results = {}
    for short in [10, 20]:  # Test key values
        for long in [40, 60]:  # Test key values
            if short < long:
                relevant_results = [
                    r
                    for r in all_results
                    if r["params"]["short_window"] == short
                    and r["params"]["long_window"] == long
                ]
                if relevant_results:
                    avg_score = np.mean([r["score"] for r in relevant_results])
                    interaction_results[f"{short}_{long}"] = avg_score

    print(f"  Window interaction scores: {interaction_results}")

    # Verify interaction analysis
    assert len(interaction_results) > 0, "Should have interaction data"

    # Test optimization convergence
    print(f"\n  📈 Testing optimization convergence...")

    # Sort results by score to analyze convergence
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
    top_10_scores = [r["score"] for r in sorted_results[:10]]

    print(f"  Top 10 scores: {[f'{s:.3f}' for s in top_10_scores]}")

    # Verify convergence
    assert len(top_10_scores) > 0, "Should have top scores"
    assert max(top_10_scores) > min(top_10_scores), "Should have score variation"

    # Test parameter validation
    print(f"\n  ✅ Testing parameter validation...")

    # Verify best parameters are valid
    assert (
        best_params["short_window"] < best_params["long_window"]
    ), "Short window should be less than long window"
    assert (
        0 <= best_params["crossover_threshold"] <= 1
    ), "Crossover threshold should be between 0 and 1"
    assert best_params["signal_strength"] > 0, "Signal strength should be positive"

    # Test performance consistency
    print(f"\n  🔄 Testing performance consistency...")

    # Re-evaluate best parameters multiple times
    consistency_scores = []
    for _ in range(3):
        metrics = test_optimizer.evaluate_strategy(
            test_strategy, best_params, test_data
        )
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0)
        max_dd = abs(metrics.get("max_drawdown", 0))
        score = sharpe * 0.5 + win_rate * 0.3 - max_dd * 0.2
        consistency_scores.append(score)

    score_variance = np.var(consistency_scores)
    print(f"  Consistency scores: {[f'{s:.3f}' for s in consistency_scores]}")
    print(f"  Score variance: {score_variance:.6f}")

    # Verify consistency
    assert score_variance < 1.0, "Performance should be consistent across runs"

    print("✅ Dynamic strategy crossover grid search test completed")


if __name__ == "__main__":
    pytest.main([__file__])
