"""Test suite for optimization module."""

import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from trading.optimization.strategy_optimizer import StrategyOptimizer, StrategyOptimizerConfig
from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.optimization.performance_logger import PerformanceLogger, PerformanceMetrics

# Test data generation
def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic price data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic price data
    """
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    # Generate prices
    np.random.seed(42)
    prices = np.random.normal(100, 1, n_samples).cumsum() + 1000
    
    # Generate volumes
    volumes = np.random.lognormal(10, 1, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.01, n_samples)),
        'close': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'volume': volumes
    })
    
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
        ma = data['close'].rolling(window=self.window).mean()
        
        # Generate signals
        self.signals = (data['close'] > ma * (1 + self.threshold)).astype(int)
        
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
        returns = data['close'].pct_change()
        strategy_returns = returns * signals.shift(1)
        
        # Calculate metrics
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        win_rate = (strategy_returns > 0).mean()
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        mse = ((data['close'] - data['close'].shift(1)) ** 2).mean()
        alpha = strategy_returns.mean() - returns.mean()
        
        return {
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'mse': mse,
            'alpha': alpha
        }

# Test configuration
@pytest.fixture
def optimizer_config() -> Dict[str, Any]:
    """Create test optimizer configuration.
    
    Returns:
        Dictionary with optimizer configuration
    """
    return {
        'optimizer_type': 'bayesian',
        'n_initial_points': 3,
        'n_iterations': 10,
        'grid_search_points': 20,
        'primary_metric': 'sharpe_ratio',
        'secondary_metrics': ['win_rate', 'max_drawdown'],
        'metric_weights': {
            'sharpe_ratio': 0.6,
            'win_rate': 0.3,
            'max_drawdown': 0.1
        },
        'kernel_type': 'matern',
        'kernel_length_scale': 1.0,
        'kernel_nu': 2.5,
        'grid_search_strategy': 'random',
        'grid_search_batch_size': 5,
        'early_stopping_patience': 3
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
    assert isinstance(optimizer.config, StrategyOptimizerConfig)
    assert optimizer.config.optimizer_type == 'bayesian'
    assert optimizer.config.n_initial_points == 3
    assert optimizer.config.n_iterations == 10

def test_strategy_optimizer_bayesian_optimization(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test Bayesian optimization."""
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check results
    assert isinstance(optimized_params, dict)
    assert 'window' in optimized_params
    assert 'threshold' in optimized_params
    assert optimized_params['window'] in param_space['window']
    assert optimized_params['threshold'] in param_space['threshold']

def test_strategy_optimizer_grid_search(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test grid search optimization."""
    optimizer_config['optimizer_type'] = 'grid'
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check results
    assert isinstance(optimized_params, dict)
    assert 'window' in optimized_params
    assert 'threshold' in optimized_params
    assert optimized_params['window'] in param_space['window']
    assert optimized_params['threshold'] in param_space['threshold']

def test_strategy_optimizer_random_search(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test random search optimization."""
    optimizer_config['optimizer_type'] = 'random'
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check results
    assert isinstance(optimized_params, dict)
    assert 'window' in optimized_params
    assert 'threshold' in optimized_params
    assert optimized_params['window'] in param_space['window']
    assert optimized_params['threshold'] in param_space['threshold']

def test_strategy_optimizer_early_stopping(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test early stopping."""
    optimizer_config['early_stopping_patience'] = 2
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check results
    assert isinstance(optimized_params, dict)
    assert len(optimizer.history) <= optimizer_config['n_iterations']

def test_strategy_optimizer_logging(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test optimization logging."""
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check log file
    log_path = "trading/optimization/logs/optimization_log.json"
    assert os.path.exists(log_path)
    
    # Check results file
    results_dir = "trading/optimization/results"
    assert os.path.exists(results_dir)
    results_files = [f for f in os.listdir(results_dir) if f.startswith("optimization_results_")]
    assert len(results_files) > 0

def test_strategy_selection_agent(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test strategy selection agent."""
    # Create agent
    agent = StrategySelectionAgent()
    
    # Create test strategies
    strategies = {
        'sma': TestStrategy(window=20, threshold=0.5),
        'ema': TestStrategy(window=10, threshold=0.3),
        'rsi': TestStrategy(window=14, threshold=0.7)
    }
    
    # Get market regime
    regime = agent.get_market_regime(test_data)
    assert isinstance(regime, str)
    assert regime in ['trending', 'ranging', 'volatile']
    
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
        strategy='test_strategy',
        config={'window': 20, 'threshold': 0.5},
        sharpe_ratio=1.5,
        win_rate=0.6,
        max_drawdown=-0.1,
        mse=0.01,
        alpha=0.02,
        regime='trending',
        reason='Test performance logging'
    )
    
    # Log metrics
    logger.log_metrics(metrics)
    
    # Check log file
    log_path = "trading/optimization/logs/performance_metrics.jsonl"
    assert os.path.exists(log_path)
    
    # Check memory
    memory = logger.get_strategy_memory()
    assert isinstance(memory, dict)
    assert 'test_strategy' in memory
    
    # Check performance over time
    performance = logger.get_performance_over_time('test_strategy')
    assert isinstance(performance, pd.DataFrame)
    assert not performance.empty

def test_sandbox_optimization(optimizer_config: Dict[str, Any], test_data: pd.DataFrame):
    """Test sandbox optimization."""
    # Create optimizer
    optimizer = StrategyOptimizer(optimizer_config)
    strategy = TestStrategy()
    
    # Define parameter space
    param_space = {
        'window': [10, 20, 30],
        'threshold': [0.1, 0.5, 0.9]
    }
    
    # Run optimization
    optimized_params = optimizer.optimize(strategy, test_data, param_space)
    
    # Check results
    assert isinstance(optimized_params, dict)
    assert 'window' in optimized_params
    assert 'threshold' in optimized_params
    
    # Check visualization files
    plots_dir = "trading/optimization/results/plots"
    assert os.path.exists(plots_dir)
    plot_files = os.listdir(plots_dir)
    assert len(plot_files) > 0

if __name__ == '__main__':
    pytest.main([__file__]) 