"""Manages multiple trading strategies with caching and dynamic loading."""

# Standard library imports
import importlib
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Third-party imports
import numpy as np
import pandas as pd

# Optional imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Local imports
from .exceptions import StrategyError, StrategyNotFoundError, StrategyValidationError
from ..core.performance import log_performance

@dataclass
class StrategyMetrics:
    """Strategy performance metrics with validation and serialization support."""
    returns: pd.Series
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    timestamp: str = datetime.utcnow().isoformat()
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not isinstance(self.returns, pd.Series):
            raise StrategyValidationError("returns must be a pandas Series")
        if not all(isinstance(x, (int, float)) for x in [
            self.sharpe_ratio, self.sortino_ratio, self.max_drawdown,
            self.win_rate, self.profit_factor, self.avg_trade,
            self.avg_win, self.avg_loss
        ]):
            raise StrategyValidationError("numeric metrics must be int or float")
        if not all(isinstance(x, int) for x in [
            self.total_trades, self.winning_trades, self.losing_trades
        ]):
            raise StrategyValidationError("trade counts must be integers")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        metrics_dict = asdict(self)
        metrics_dict['returns'] = self.returns.to_dict()
        return metrics_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyMetrics':
        """Create metrics from dictionary."""
        returns = pd.Series(data['returns'])
        data['returns'] = returns
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyMetrics':
        """Create metrics from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """Return a one-line summary of key metrics."""
        return (f"Sharpe: {self.sharpe_ratio:.2f}, Sortino: {self.sortino_ratio:.2f}, "
                f"Win Rate: {self.win_rate:.1%}, Trades: {self.total_trades}")

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration dictionary containing:
                - log_level: Logging level (default: INFO)
                - max_position_size: Maximum position size (default: 100000)
                - min_position_size: Minimum position size (default: 1000)
                - max_leverage: Maximum leverage (default: 1.0)
                - stop_loss: Stop loss percentage (default: 0.02)
                - take_profit: Take profit percentage (default: 0.04)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'log_level': os.getenv('STRATEGY_LOG_LEVEL', 'INFO'),
            'max_position_size': float(os.getenv('STRATEGY_MAX_POSITION_SIZE', 100000)),
            'min_position_size': float(os.getenv('STRATEGY_MIN_POSITION_SIZE', 1000)),
            'max_leverage': float(os.getenv('STRATEGY_MAX_LEVERAGE', 1.0)),
            'stop_loss': float(os.getenv('STRATEGY_STOP_LOSS', 0.02)),
            'take_profit': float(os.getenv('STRATEGY_TAKE_PROFIT', 0.04))
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with columns:
                - Signal: Trading signals (1: buy, -1: sell, 0: hold)
                - Position: Current position size
                - PnL: Profit and loss
                - Equity: Cumulative equity
        """
        pass
    
    @abstractmethod
    def evaluate_performance(self, signals: pd.DataFrame, prices: pd.DataFrame) -> StrategyMetrics:
        """Evaluate strategy performance.
        
        Args:
            signals: DataFrame with trading signals and positions
            prices: Market data with OHLCV columns
            
        Returns:
            StrategyMetrics object with performance metrics
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            data: Market data to validate
            
        Raises:
            StrategyError: If data validation fails
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise StrategyError(f"Data missing required columns: {required_columns}")
            
        if data.empty:
            raise StrategyError("Data is empty")
            
        if data.isnull().any().any():
            raise StrategyError("Data contains null values")
    
    def validate_signals(self, signals: pd.Series) -> None:
        """Validate trading signals.
        
        Args:
            signals: Trading signals to validate
            
        Raises:
            StrategyError: If signal validation fails
        """
        if signals.empty:
            raise StrategyError("Signals are empty")
            
        if signals.isnull().any():
            raise StrategyError("Signals contain null values")
            
        if not all(signal in [-1, 0, 1] for signal in signals):
            raise StrategyError("Signals must be -1 (sell), 0 (hold), or 1 (buy)")

class StrategyManager:
    """Manages multiple trading strategies with caching and dynamic loading."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy manager.
        
        Args:
            config: Configuration dictionary containing:
                - redis_host: Redis host (default: localhost)
                - redis_port: Redis port (default: 6379)
                - redis_db: Redis database (default: 0)
                - redis_password: Redis password
                - redis_ssl: Whether to use SSL (default: false)
                - log_level: Logging level (default: INFO)
                - results_dir: Directory for saving results (default: strategy_results)
                - max_strategies: Maximum number of strategies (default: 100)
                - min_performance_threshold: Minimum performance threshold (default: 0.5)
                - strategy_dir: Directory for strategy modules (default: trading/strategies)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            'redis_db': int(os.getenv('REDIS_DB', 0)),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'redis_ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
            'log_level': os.getenv('STRATEGY_MANAGER_LOG_LEVEL', 'INFO'),
            'results_dir': os.getenv('STRATEGY_RESULTS_DIR', 'strategy_results'),
            'max_strategies': int(os.getenv('MAX_STRATEGIES', 100)),
            'min_performance_threshold': float(os.getenv('MIN_PERFORMANCE_THRESHOLD', 0.5)),
            'strategy_dir': os.getenv('STRATEGY_DIR', 'trading/strategies')
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize Redis connection if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config['redis_host'],
                    port=self.config['redis_port'],
                    db=self.config['redis_db'],
                    password=self.config['redis_password'],
                    ssl=self.config['redis_ssl']
                )
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {str(e)}")
        
        # Initialize strategy storage
        self.strategies = {}
        self.active_strategies = set()
        self.ensemble_weights = {}
        
        # Load strategies
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load strategy modules from strategy directory."""
        strategy_dir = Path(self.config['strategy_dir'])
        if not strategy_dir.exists():
            self.logger.warning(f"Strategy directory not found: {strategy_dir}")
            return
            
        for file in strategy_dir.glob('*.py'):
            if file.name.startswith('_') or file.name == 'strategy_manager.py':
                continue
                
            try:
                module_name = f"trading.strategies.{file.stem}"
                module = importlib.import_module(module_name)
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Strategy) and 
                        obj != Strategy):
                        self.register_strategy(name, obj())
                        self.logger.info(f"Loaded strategy: {name}")
                        
            except Exception as e:
                self.logger.error(f"Error loading strategy from {file}: {str(e)}")
    
    def _save_strategies(self) -> None:
        """Save strategy configurations."""
        if not self.redis_client:
            return
            
        try:
            for name, strategy in self.strategies.items():
                config = {
                    'name': name,
                    'config': strategy.config,
                    'active': name in self.active_strategies,
                    'ensemble_weight': self.ensemble_weights.get(name, 0.0)
                }
                self.redis_client.hset('strategies', name, json.dumps(config))
                
            self.logger.info("Saved strategy configurations")
            
        except Exception as e:
            self.logger.error(f"Error saving strategies: {str(e)}")
    
    def register_strategy(self, name: str, strategy: Strategy) -> None:
        """Register a new strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            
        Raises:
            StrategyError: If strategy registration fails
        """
        if name in self.strategies:
            raise StrategyError(f"Strategy already registered: {name}")
            
        if len(self.strategies) >= self.config['max_strategies']:
            raise StrategyError("Maximum number of strategies reached")
            
        self.strategies[name] = strategy
        self.logger.info(f"Registered strategy: {name}")
        
        # Save to Redis if available
        if self.redis_client:
            try:
                config = {
                    'name': name,
                    'config': strategy.config,
                    'active': False,
                    'ensemble_weight': 0.0
                }
                self.redis_client.hset('strategies', name, json.dumps(config))
            except Exception as e:
                self.logger.error(f"Error saving strategy to Redis: {str(e)}")
    
    def get_strategy(self, name: str) -> Strategy:
        """Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance
            
        Raises:
            StrategyNotFoundError: If strategy not found
        """
        if name not in self.strategies:
            raise StrategyNotFoundError(f"Strategy not found: {name}")
        return self.strategies[name]
    
    def activate_strategy(self, name: str) -> None:
        """Activate a strategy.
        
        Args:
            name: Strategy name
            
        Raises:
            StrategyNotFoundError: If strategy not found
        """
        if name not in self.strategies:
            raise StrategyNotFoundError(f"Strategy not found: {name}")
            
        self.active_strategies.add(name)
        self.logger.info(f"Activated strategy: {name}")
        
        # Update Redis if available
        if self.redis_client:
            try:
                config = json.loads(self.redis_client.hget('strategies', name))
                config['active'] = True
                self.redis_client.hset('strategies', name, json.dumps(config))
            except Exception as e:
                self.logger.error(f"Error updating Redis: {str(e)}")
    
    def deactivate_strategy(self, name: str) -> None:
        """Deactivate a strategy.
        
        Args:
            name: Strategy name
            
        Raises:
            StrategyNotFoundError: If strategy not found
        """
        if name not in self.strategies:
            raise StrategyNotFoundError(f"Strategy not found: {name}")
            
        self.active_strategies.remove(name)
        self.logger.info(f"Deactivated strategy: {name}")
        
        # Update Redis if available
        if self.redis_client:
            try:
                config = json.loads(self.redis_client.hget('strategies', name))
                config['active'] = False
                self.redis_client.hset('strategies', name, json.dumps(config))
            except Exception as e:
                self.logger.error(f"Error updating Redis: {str(e)}")
    
    def set_ensemble(self, weights: Dict[str, float], strict: bool = True) -> None:
        """Set ensemble weights for strategies.
        
        Args:
            weights: Dictionary of strategy names and weights
            strict: Whether to enforce weight validation
            
        Raises:
            StrategyError: If weight validation fails
        """
        if strict:
            # Validate weights
            if not all(0 <= w <= 1 for w in weights.values()):
                raise StrategyError("Weights must be between 0 and 1")
                
            if not np.isclose(sum(weights.values()), 1.0):
                raise StrategyError("Weights must sum to 1")
                
            if not all(name in self.strategies for name in weights):
                raise StrategyError("Invalid strategy names in weights")
        
        self.ensemble_weights = weights
        self.logger.info(f"Set ensemble weights: {weights}")
        
        # Update Redis if available
        if self.redis_client:
            try:
                for name, weight in weights.items():
                    config = json.loads(self.redis_client.hget('strategies', name))
                    config['ensemble_weight'] = weight
                    self.redis_client.hset('strategies', name, json.dumps(config))
            except Exception as e:
                self.logger.error(f"Error updating Redis: {str(e)}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from active strategies.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Series of combined trading signals
            
        Raises:
            StrategyError: If signal generation fails
        """
        if not self.active_strategies:
            raise StrategyError("No active strategies")
            
        # Generate signals from each active strategy
        strategy_signals = {}
        for name in self.active_strategies:
            try:
                strategy = self.strategies[name]
                signals = strategy.generate_signals(data)
                strategy_signals[name] = signals['signal']
            except Exception as e:
                self.logger.error(f"Error generating signals for {name}: {str(e)}")
                continue
        
        if not strategy_signals:
            raise StrategyError("No valid signals generated")
        
        # Combine signals using ensemble weights
        if self.ensemble_weights:
            # Use weighted average of signals
            combined_signals = pd.Series(0, index=data.index)
            total_weight = 0
            
            for name, signals in strategy_signals.items():
                if name in self.ensemble_weights:
                    weight = self.ensemble_weights[name]
                    combined_signals += weight * signals
                    total_weight += weight
            
            if total_weight > 0:
                combined_signals /= total_weight
        else:
            # Use simple average of signals
            combined_signals = pd.concat(strategy_signals.values(), axis=1).mean(axis=1)
        
        # Round to nearest signal value
        combined_signals = combined_signals.round()
        
        return combined_signals
    
    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, StrategyMetrics]:
        """Evaluate performance of all strategies.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Dictionary of strategy names and metrics
            
        Raises:
            StrategyError: If evaluation fails
        """
        metrics = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                strategy_metrics = strategy.evaluate_performance(signals, data)
                metrics[name] = strategy_metrics
                
                # Log performance
                log_performance(name, strategy_metrics)
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
                continue
        
        return metrics
    
    def rank_strategies(self, metrics: Dict[str, StrategyMetrics]) -> List[tuple]:
        """Rank strategies by performance.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            List of (strategy_name, score) tuples sorted by score
        """
        scores = []
        
        for name, metric in metrics.items():
            # Calculate composite score
            score = (
                metric.sharpe_ratio * 0.4 +
                metric.sortino_ratio * 0.3 +
                metric.win_rate * 0.2 +
                (1 - abs(metric.max_drawdown)) * 0.1
            )
            
            scores.append((name, score))
        
        # Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save optimization results.
        
        Args:
            results: Results to save
            filename: Output filename
            
        Raises:
            StrategyError: If saving fails
        """
        try:
            results_dir = Path(self.config['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = results_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"Saved results to {filepath}")
            
        except Exception as e:
            raise StrategyError(f"Error saving results: {str(e)}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load optimization results.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
            
        Raises:
            StrategyError: If loading fails
        """
        try:
            filepath = Path(self.config['results_dir']) / filename
            with open(filepath, 'r') as f:
                results = json.load(f)
                
            self.logger.info(f"Loaded results from {filepath}")
            return results
            
        except Exception as e:
            raise StrategyError(f"Error loading results: {str(e)}")
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Raises:
            StrategyError: If caching fails
        """
        if not self.redis_client:
            return
            
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            self.logger.debug(f"Cached value for {key}")
            
        except Exception as e:
            raise StrategyError(f"Error setting cache: {str(e)}")
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            StrategyError: If cache retrieval fails
        """
        if not self.redis_client:
            return None
            
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            raise StrategyError(f"Error getting cache: {str(e)}")
    
    def simulate(self, data: pd.DataFrame, strategy_name: str, config: Optional[Dict] = None) -> StrategyMetrics:
        """Simulate strategy performance.
        
        Args:
            data: Market data with OHLCV columns
            strategy_name: Name of strategy to simulate
            config: Optional configuration overrides
            
        Returns:
            StrategyMetrics object
            
        Raises:
            StrategyError: If simulation fails
        """
        if strategy_name not in self.strategies:
            raise StrategyError(f"Strategy not found: {strategy_name}")
            
        try:
            strategy = self.strategies[strategy_name]
            
            # Update config if provided
            if config:
                strategy.config.update(config)
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Evaluate performance
            metrics = strategy.evaluate_performance(signals, data)
            
            # Log performance
            log_performance(strategy_name, metrics)
            
            return metrics
            
        except Exception as e:
            raise StrategyError(f"Error simulating {strategy_name}: {str(e)}") 