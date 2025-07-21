"""Test suite for optimization module."""

import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from trading.optimization.base_optimizer import OptimizerConfig
from trading.optimization.performance_logger import (
    PerformanceLogger,
    PerformanceMetrics,
)
from trading.optimization.strategy_optimizer import StrategyOptimizer
from trading.optimization.strategy_selection_agent import StrategySelectionAgent

# Test data generation


def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic price data for testing.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic price data
    """
    # Generate dates
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="1H")

    # Generate prices
    np.random.seed(42)
    prices = np.random.normal(100, 1, n_samples).cumsum() + 1000

    # Generate volumes
    volumes = np.random.lognormal(10, 1, n_samples)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices * (1 + np.random.uniform(0, 0.01, n_samples)),
            "low": prices * (1 - np.random.uniform(0, 0.01, n_samples)),
            "close": prices * (1 + np.random.normal(0, 0.005, n_samples)),
            "volume": volumes,
        }
    )

    return data


# Test strategy class


class TestStrategy:
    """Test strategy class for optimization testing."""

    def __init__(self, window: int = 20, threshold: float = 0.5):
        """Initialize test strategy.

        Args:
            window: Moving average window
            threshold: Signal threshold
        """
        self.window = window
        self.threshold = threshold
        self.signals = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit strategy to data.

        Args:
            data: DataFrame with price data
        """
        # Calculate moving average
        ma = data["close"].rolling(window=self.window).mean()

        # Generate signals
        self.signals = (data["close"] > ma * (1 + self.threshold)).astype(int)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            data: DataFrame with price data

        Returns:
            Series of predictions
        """
        if self.signals is None:
            self.fit(data)
        return self.signals

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate strategy performance.

        Args:
            data: DataFrame with price data

        Returns:
            Dictionary of performance metrics
        """
        signals = self.predict(data)
        returns = data["close"].pct_change()
        strategy_returns = returns * signals.shift(1)

        # Calculate metrics
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        win_rate = (strategy_returns > 0).mean()
        max_drawdown = (
            strategy_returns.cumsum() - strategy_returns.cumsum().cummax()
        ).min()
        mse = ((data["close"] - data["close"].shift(1)) ** 2).mean()
        alpha = strategy_returns.mean() - returns.mean()

        return {
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "mse": mse,
            "alpha": alpha,
        }


# Test configuration


@pytest.fixture
def optimizer_config() -> Dict[str, Any]:
    """Create test optimizer configuration.

    Returns:
        Dictionary with optimizer configuration
    """
    return {
        "optimizer_type": "bayesian",
        "n_initial_points": 3,
        "n_iterations": 10,
        "grid_search_points": 20,
        "primary_metric": "sharpe_ratio",
        "secondary_metrics": ["win_rate", "max_drawdown"],
        "metric_weights": {"sharpe_ratio": 0.6, "win_rate": 0.3, "max_drawdown": 0.1},
        "kernel_type": "matern",
        "kernel_length_scale": 1.0,
        "kernel_nu": 2.5,
        "grid_search_strategy": "random",
        "grid_search_batch_size": 5,
        "early_stopping_patience": 3,
    }


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Create test data.

    Returns:
        DataFrame with test data
    """
    return generate_test_data()


# Test cases


def test_strategy_optimizer_initialization(optimizer_config: Dict[str, Any]):
    """Test strategy optimizer initialization."""
    optimizer = StrategyOptimizer(optimizer_config)
    assert isinstance(optimizer.config, OptimizerConfig)
    assert optimizer.config.optimizer_type == "bayesian"
    assert optimizer.config.n_initial_points == 3
    assert optimizer.config.n_iterations == 10


def test_strategy_optimizer_bayesian_optimization(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test Bayesian optimization."""
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)

    # Check results
    assert isinstance(optimized_params, dict)
    assert "window" in optimized_params
    assert "threshold" in optimized_params
    assert optimized_params["window"] in param_space["window"]
    assert optimized_params["threshold"] in param_space["threshold"]


def test_strategy_optimizer_grid_search(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test grid search optimization."""
    optimizer_config["optimizer_type"] = "grid"
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)

    # Check results
    assert isinstance(optimized_params, dict)
    assert "window" in optimized_params
    assert "threshold" in optimized_params
    assert optimized_params["window"] in param_space["window"]
    assert optimized_params["threshold"] in param_space["threshold"]


def test_strategy_optimizer_random_search(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test random search optimization."""
    optimizer_config["optimizer_type"] = "random"
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)

    # Check results
    assert isinstance(optimized_params, dict)
    assert "window" in optimized_params
    assert "threshold" in optimized_params
    assert optimized_params["window"] in param_space["window"]
    assert optimized_params["threshold"] in param_space["threshold"]


def test_strategy_optimizer_early_stopping(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test early stopping."""
    optimizer_config["early_stopping_patience"] = 2
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)

    # Check results
    assert isinstance(optimized_params, dict)
    assert len(optimizer.history) <= optimizer_config["n_iterations"]


def test_strategy_optimizer_logging(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test optimization logging."""
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimizer.optimize(strategy, test_data, param_space)

    # Check log file
    log_path = "trading/optimization/logs/optimization_log.json"
    assert os.path.exists(log_path)

    # Check results file
    results_dir = "trading/optimization/results"
    assert os.path.exists(results_dir)
    results_files = [
        f for f in os.listdir(results_dir) if f.startswith("optimization_results_")
    ]
    assert len(results_files) > 0


def test_strategy_selection_agent(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test strategy selection agent."""
    # Create agent
    agent = StrategySelectionAgent()

    # Create test strategies
    strategies = {
        "sma": TestStrategy(window=20, threshold=0.5),
        "ema": TestStrategy(window=10, threshold=0.3),
        "rsi": TestStrategy(window=14, threshold=0.7),
    }

    # Get market regime
    regime = agent.get_market_regime(test_data)
    assert isinstance(regime, str)
    assert regime in ["trending", "ranging", "volatile"]

    # Select strategy
    selected_strategy = agent.select_strategy(strategies, test_data)
    assert isinstance(selected_strategy, str)
    assert selected_strategy in strategies.keys()

    # Check memory
    memory = agent.get_strategy_memory()
    assert isinstance(memory, dict)
    assert selected_strategy in memory


def test_performance_logger(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test performance logger."""
    # Create logger
    logger = PerformanceLogger()

    # Create test metrics
    metrics = PerformanceMetrics(
        timestamp=datetime.utcnow(),
        strategy="test_strategy",
        config={"window": 20, "threshold": 0.5},
        sharpe_ratio=1.5,
        win_rate=0.6,
        max_drawdown=-0.1,
        mse=0.01,
        alpha=0.02,
        regime="trending",
        reason="Test performance logging",
    )

    # Log metrics
    logger.log_metrics(metrics)

    # Check log file
    log_path = "trading/optimization/logs/performance_metrics.jsonl"
    assert os.path.exists(log_path)

    # Check memory
    memory = logger.get_strategy_memory()
    assert isinstance(memory, dict)
    assert "test_strategy" in memory

    # Check performance over time
    performance = logger.get_performance_over_time("test_strategy")
    assert isinstance(performance, pd.DataFrame)
    assert not performance.empty


def test_sandbox_optimization(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Test sandbox optimization."""
    # Create optimizer
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define parameter space
    param_space = {"window": [10, 20, 30], "threshold": [0.1, 0.5, 0.9]}

    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)

    # Check results
    assert isinstance(optimized_params, dict)
    assert "window" in optimized_params
    assert "threshold" in optimized_params

    # Check visualization files
    plots_dir = "trading/optimization/results/plots"
    assert os.path.exists(plots_dir)
    plot_files = os.listdir(plots_dir)
    assert len(plot_files) > 0


def test_optimized_params_strictly_better_than_defaults(
    optimizer_config: Dict[str, Any], test_data: pd.DataFrame
):
    """Assert that optimized strategy parameters yield strictly better performance than defaults."""
    print("\n🏆 Testing Optimized Parameters vs Defaults")

    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()

    # Define comprehensive parameter space
    param_space = {
        "window": [5, 10, 15, 20, 25, 30, 35, 40],
        "threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    # Test multiple default parameter sets
    default_param_sets = [
        {"window": 20, "threshold": 0.5},  # Standard defaults
        {"window": 14, "threshold": 0.3},  # Conservative defaults
        {"window": 30, "threshold": 0.7},  # Aggressive defaults
        {"window": 10, "threshold": 0.2},  # Short-term defaults
        {"window": 40, "threshold": 0.8},  # Long-term defaults
    ]

    for i, default_params in enumerate(default_param_sets):
        print(f"\n  📊 Testing default set {i + 1}: {default_params}")

        # Set default parameters
        strategy.window = default_params["window"]
        strategy.threshold = default_params["threshold"]
        strategy.fit(test_data)

        # Evaluate default performance
        default_metrics = strategy.evaluate(test_data)

        print(
            f"    Default metrics: Sharpe={default_metrics['sharpe_ratio']:.3f}, "
            f"Win Rate={default_metrics['win_rate']:.3f}, "
            f"Max DD={default_metrics['max_drawdown']:.3f}"
        )

        # Run optimization
        optimized_params = optimizer.optimize(strategy, test_data, param_space)

        # Set optimized parameters
        strategy.window = optimized_params["window"]
        strategy.threshold = optimized_params["threshold"]
        strategy.fit(test_data)

        # Evaluate optimized performance
        optimized_metrics = strategy.evaluate(test_data)

        print(f"    Optimized params: {optimized_params}")
        print(
            f"    Optimized metrics: Sharpe={optimized_metrics['sharpe_ratio']:.3f}, "
            f"Win Rate={optimized_metrics['win_rate']:.3f}, "
            f"Max DD={optimized_metrics['max_drawdown']:.3f}"
        )

        # Assert primary metric (sharpe_ratio) is strictly better
        assert optimized_metrics["sharpe_ratio"] > default_metrics["sharpe_ratio"], (f"Optimized Sharpe {
            optimized_metrics['sharpe_ratio']:.3f} not better than default {
            default_metrics['sharpe_ratio']:.3f} " f"for params {default_params} -> {optimized_params}")

        # Assert win rate is better or equal (with tolerance for noise)
        assert (
            optimized_metrics["win_rate"] >= default_metrics["win_rate"] - 0.05
        ), f"Optimized win rate {optimized_metrics['win_rate']:.3f} significantly worse than default {default_metrics['win_rate']:.3f}"

        # Assert max drawdown is not significantly worse
        assert (
            optimized_metrics["max_drawdown"] >= default_metrics["max_drawdown"] - 0.1), f"Optimized max drawdown {
            optimized_metrics['max_drawdown']:.3f} significantly worse than default {
            default_metrics['max_drawdown']:.3f}"

        # Calculate improvement percentages
        sharpe_improvement = (
            (
                (optimized_metrics["sharpe_ratio"] - default_metrics["sharpe_ratio"])
                / abs(default_metrics["sharpe_ratio"])
            )
            * 100
            if default_metrics["sharpe_ratio"] != 0
            else float("inf")
        )

        win_rate_improvement = (
            (
                (optimized_metrics["win_rate"] - default_metrics["win_rate"])
                / default_metrics["win_rate"]
            )
            * 100
            if default_metrics["win_rate"] != 0
            else 0
        )

        print(f"    Sharpe improvement: {sharpe_improvement:.1f}%")
        print(f"    Win rate improvement: {win_rate_improvement:.1f}%")

        # Verify meaningful improvement
        assert (
            sharpe_improvement > 5.0
        ), f"Insufficient Sharpe improvement: {sharpe_improvement:.1f}%"

    # Test optimization consistency across multiple runs
    print(f"\n  🔄 Testing optimization consistency...")

    consistency_results = []
    for run in range(3):
        strategy = TestStrategy()
        default_params = {"window": 20, "threshold": 0.5}

        # Default performance
        strategy.window = default_params["window"]
        strategy.threshold = default_params["threshold"]
        strategy.fit(test_data)
        default_metrics = strategy.evaluate(test_data)

        # Optimized performance
        optimized_params = optimizer.optimize(strategy, test_data, param_space)
        strategy.window = optimized_params["window"]
        strategy.threshold = optimized_params["threshold"]
        strategy.fit(test_data)
        optimized_metrics = strategy.evaluate(test_data)

        improvement = (
            optimized_metrics["sharpe_ratio"] - default_metrics["sharpe_ratio"]
        )
        consistency_results.append(improvement)

        print(f"    Run {run + 1}: Improvement = {improvement:.3f}")

    # Verify consistency
    improvement_variance = np.var(consistency_results)
    print(f"    Improvement variance: {improvement_variance:.6f}")

    assert (
        improvement_variance < 0.1
    ), f"Optimization results too inconsistent: variance = {improvement_variance:.6f}"

    # Test parameter space coverage
    print(f"\n  🎯 Testing parameter space coverage...")

    all_optimized_params = []
    for _ in range(5):
        strategy = TestStrategy()
        optimized_params = optimizer.optimize(strategy, test_data, param_space)
        all_optimized_params.append(optimized_params)

    # Check parameter diversity
    window_values = [p["window"] for p in all_optimized_params]
    threshold_values = [p["threshold"] for p in all_optimized_params]

    window_diversity = len(set(window_values))
    threshold_diversity = len(set(threshold_values))

    print(f"    Window diversity: {window_diversity}/{len(param_space['window'])}")
    print(
        f"    Threshold diversity: {threshold_diversity}/{len(param_space['threshold'])}"
    )

    # Verify exploration of parameter space
    assert (
        window_diversity >= 3
    ), f"Insufficient window parameter exploration: {window_diversity} unique values"
    assert (
        threshold_diversity >= 3
    ), f"Insufficient threshold parameter exploration: {threshold_diversity} unique values"

    # Test optimization with different data splits
    print(f"\n  📈 Testing optimization with different data splits...")

    # Split data into training and validation
    split_point = len(test_data) // 2
    train_data = test_data.iloc[:split_point]
    val_data = test_data.iloc[split_point:]

    strategy = TestStrategy()
    default_params = {"window": 20, "threshold": 0.5}

    # Default performance on validation
    strategy.window = default_params["window"]
    strategy.threshold = default_params["threshold"]
    strategy.fit(train_data)
    default_val_metrics = strategy.evaluate(val_data)

    # Optimize on training data
    optimized_params = optimizer.optimize(strategy, train_data, param_space)

    # Evaluate optimized on validation
    strategy.window = optimized_params["window"]
    strategy.threshold = optimized_params["threshold"]
    strategy.fit(train_data)
    optimized_val_metrics = strategy.evaluate(val_data)

    print(f"    Default validation Sharpe: {default_val_metrics['sharpe_ratio']:.3f}")
    print(
        f"    Optimized validation Sharpe: {optimized_val_metrics['sharpe_ratio']:.3f}"
    )

    # Verify generalization
    assert (
        optimized_val_metrics["sharpe_ratio"] > default_val_metrics["sharpe_ratio"]), f"Optimized parameters don't generalize: {
        optimized_val_metrics['sharpe_ratio']:.3f} vs {
            default_val_metrics['sharpe_ratio']:.3f}"

    print("✅ Optimized parameters strictly better than defaults test completed")


if __name__ == "__main__":
    pytest.main([__file__])
