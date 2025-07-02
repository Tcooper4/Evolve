"""Test cases for optimizers."""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import pytest
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from trading.optimization.base_optimizer import BaseOptimizer, OptimizerConfig
from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.optimization.performance_logger import PerformanceLogger, PerformanceMetrics

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

class TestStrategy:
    """Test strategy class."""
    
    def __init__(self):
        """Initialize test strategy."""
        self.name = "TestStrategy"
    
    def generate_signals(self, data: pd.DataFrame,
                        params: Dict[str, Any]) -> pd.Series:
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
    df = pd.DataFrame({
        "price": prices,
        "volume": volumes
    }, index=dates)
    
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
        "objective_weights": {
            "sharpe_ratio": 0.6,
            "win_rate": 0.4
        },
        "use_lr_scheduler": True,
        "scheduler_type": "cosine",
        "min_lr": 0.0001,
        "warmup_steps": 0,
        "save_checkpoints": True,
        "checkpoint_dir": "test_checkpoints",
        "checkpoint_frequency": 2,
        "validation_split": 0.2,
        "cross_validation_folds": 2
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
    params = {
        "short_window": 20,
        "long_window": 50
    }
    
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
    strategy, confidence, explanation = test_agent.select_strategy(test_data, strategies)
    
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
        reason="Test run"
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
    params = {
        "short_window": 20,
        "long_window": 50
    }
    
    # Evaluate strategy
    metrics = test_optimizer.evaluate_strategy(test_strategy, params, test_data)
    
    # Log results
    test_optimizer.log_results(
        test_strategy.name,
        params,
        metrics,
        "Test visualization"
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
    metrics = test_optimizer.evaluate_strategy(test_strategy, optimized_params, test_data)
    
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
    optimized_params = test_optimizer.optimize(test_strategy, test_data)
    
    # Check if optimization stopped early
    assert test_optimizer.current_iteration <= test_optimizer.config.max_iterations
    assert test_optimizer.early_stopping_counter <= test_optimizer.config.early_stopping_patience

def test_checkpointing(test_optimizer, test_strategy, test_data):
    """Test checkpointing."""
    # Enable checkpointing
    test_optimizer.config.save_checkpoints = True
    test_optimizer.config.checkpoint_dir = "test_checkpoints"
    
    # Run optimization
    optimized_params = test_optimizer.optimize(test_strategy, test_data)
    
    # Check if checkpoints were created
    assert os.path.exists(test_optimizer.config.checkpoint_dir)
    checkpoint_files = os.listdir(test_optimizer.config.checkpoint_dir)
    assert len(checkpoint_files) > 0

def test_cross_validation(test_optimizer, test_strategy, test_data):
    """Test cross-validation."""
    # Enable cross-validation
    test_optimizer.config.cross_validation_folds = 3
    
    # Run optimization
    optimized_params = test_optimizer.optimize(test_strategy, test_data)
    
    # Check if cross-validation was performed
    assert len(test_optimizer.metrics_history) >= test_optimizer.config.cross_validation_folds

def test_strategy_combination(test_optimizer, test_strategy, test_data):
    """Test strategy combination."""
    # Create combined strategy
    class CombinedStrategy:
        def __init__(self):
            self.name = "CombinedStrategy"
            self.strategies = [TestStrategy(), TestStrategy()]
        
        def generate_signals(self, data: pd.DataFrame,
                           params: Dict[str, Any]) -> pd.Series:
            # Generate signals from each strategy
            signals = []
            for i, strategy in enumerate(self.strategies):
                strategy_params = {
                    k: v for k, v in params.items()
                    if k.startswith(f"strategy_{i}_")
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
    metrics = test_optimizer.evaluate_strategy(test_strategy, optimized_params, test_data)
    
    # Check for overfitting
    assert metrics["sharpe_ratio"] < 5  # Unrealistic Sharpe ratio
    assert metrics["win_rate"] < 0.9  # Unrealistic win rate
    assert metrics["max_drawdown"] > 0.1  # Reasonable drawdown

if __name__ == "__main__":
    pytest.main([__file__]) 