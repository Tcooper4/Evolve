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
                # Test connection
                self.redis_client.ping()
                self.logger.info("Successfully connected to Redis")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}. Using file-based storage.")
                self.redis_client = None
        
        # Initialize file-based storage
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy storage
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: Dict[str, Strategy] = {}
        self.ensemble_weights: Dict[str, float] = {}
        
        # Load existing strategies if available
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load strategy modules from the strategy directory."""
        strategy_dir = Path(self.config['strategy_dir'])
        if not strategy_dir.exists():
            self.logger.warning(f"Strategy directory {strategy_dir} does not exist")
            return
        
        for file in strategy_dir.glob('*.py'):
            if file.name.startswith('_') or file.name == 'strategy_manager.py':
                continue
            
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec is None or spec.loader is None:
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Strategy) and 
                        obj != Strategy):
                        self.register_strategy(name, obj())
                        self.logger.info(f"Loaded strategy: {name}")
            
            except Exception as e:
                self.logger.error(f"Failed to load strategy from {file}: {e}")
    
    def _save_strategies(self) -> None:
        """Save strategies to storage."""
        strategies = {
            name: {
                'active': name in self.active_strategies,
                'weight': self.ensemble_weights.get(name, 0.0)
            }
            for name in self.strategies.keys()
        }
        
        # Save to Redis if available
        if self.redis_client:
            try:
                self.redis_client.set('strategies', json.dumps(strategies))
            except Exception as e:
                self.logger.warning(f"Failed to save strategies to Redis: {e}")
        
        # Save to file as fallback
        try:
            with open(self.results_dir / 'strategies.json', 'w') as f:
                json.dump(strategies, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to save strategies to file: {e}")
    
    def register_strategy(self, name: str, strategy: Strategy) -> None:
        """Register a new strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            
        Raises:
            StrategyError: If strategy registration fails
        """
        try:
            if len(self.strategies) >= self.config['max_strategies']:
                raise StrategyError("Maximum number of strategies reached")
                
            if name in self.strategies:
                raise StrategyError(f"Strategy {name} already registered")
                
            self.strategies[name] = strategy
            
            self.logger.info(f"Registered strategy: {name}")
            
        except Exception as e:
            raise StrategyError(f"Failed to register strategy: {str(e)}")
    
    def get_strategy(self, name: str) -> Strategy:
        """Get a strategy by name."""
        if name not in self.strategies:
            raise StrategyNotFoundError(f"Strategy {name} not found")
        return self.strategies[name]
    
    def activate_strategy(self, name: str) -> None:
        """Activate a strategy.
        
        Args:
            name: Strategy name
            
        Raises:
            StrategyError: If strategy activation fails
        """
        try:
            if name not in self.strategies:
                raise StrategyError(f"Strategy {name} not found")
                
            if name not in self.active_strategies:
                self.active_strategies[name] = self.strategies[name]
                self.logger.info(f"Activated strategy: {name}")
                
        except Exception as e:
            raise StrategyError(f"Failed to activate strategy: {str(e)}")
    
    def deactivate_strategy(self, name: str) -> None:
        """Deactivate a strategy.
        
        Args:
            name: Strategy name
            
        Raises:
            StrategyError: If strategy deactivation fails
        """
        try:
            if name in self.active_strategies:
                del self.active_strategies[name]
                self.logger.info(f"Deactivated strategy: {name}")
                
        except Exception as e:
            raise StrategyError(f"Failed to deactivate strategy: {str(e)}")
    
    def set_ensemble(self, weights: Dict[str, float], strict: bool = True) -> None:
        """Set ensemble weights for active strategies.

        Args:
            weights: Dictionary of strategy weights
            strict: If ``True``, raise an error when the weights do not sum to
                ``1.0``. If ``False``, automatically normalise the provided
                weights so that they sum to ``1.0``.

        Raises:
            StrategyError: If ensemble configuration fails
        """
        try:
            if not all(name in self.active_strategies for name in weights.keys()):
                raise StrategyError("All strategies in weights must be active")

            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0):
                if strict:
                    raise StrategyError("Weights must sum to 1.0")
                if total_weight == 0:
                    raise StrategyError("Sum of weights must be non-zero for normalisation")
                weights = {name: weight / total_weight for name, weight in weights.items()}
                self.logger.info(f"Normalised ensemble weights: {weights}")

            self.ensemble_weights = weights
            self.logger.info(f"Set ensemble weights: {weights}")

        except Exception as e:
            raise StrategyError(f"Failed to set ensemble weights: {str(e)}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from active strategies.
        
        Args:
            data: Market data
            
        Returns:
            Combined trading signals
            
        Raises:
            StrategyError: If signal generation fails
        """
        try:
            if not self.active_strategies:
                raise StrategyError("No active strategies")
            
            signals = pd.DataFrame(index=data.index)
            
            for name in self.active_strategies:
                strategy = self.strategies[name]
                try:
                    # Validate data
                    strategy.validate_data(data)
                    
                    # Generate signals
                    strategy_signals = strategy.generate_signals(data)
                    strategy.validate_signals(strategy_signals['Signal'])
                    signals[name] = strategy_signals['Signal']
                    
                    # Calculate and store metrics
                    metrics = strategy.evaluate_performance(strategy_signals, data)
                    
                    # Store metrics in Redis
                    if self.redis_client:
                        self.redis_client.hset(
                            f'strategy_metrics:{name}',
                            metrics.timestamp,
                            metrics.to_json()
                        )
                    
                    self.logger.info(f"Generated signals for strategy {name}")
                    
                except Exception as e:
                    self.logger.error(f"Error generating signals for strategy {name}: {str(e)}")
                    signals[name] = 0
            
            if self.ensemble_weights:
                # Combine signals using ensemble weights
                combined_signals = pd.Series(0, index=data.index)
                for name, weight in self.ensemble_weights.items():
                    combined_signals += signals[name] * weight
                return combined_signals
            else:
                # Use simple majority voting
                return signals.mean(axis=1).round()
                
        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, StrategyMetrics]:
        """Evaluate all active strategies on the given data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Dictionary mapping strategy names to their metrics
        """
        try:
            metrics = {}
            for name, strategy in self.active_strategies.items():
                try:
                    # Generate signals
                    signals = strategy.generate_signals(data)
                    
                    # Calculate metrics
                    strategy_metrics = strategy.evaluate_performance(signals, data)
                    
                    # Log performance
                    log_performance(
                        ticker=data.name if hasattr(data, 'name') else 'unknown',
                        model=name,
                        strategy=strategy.__class__.__name__,
                        sharpe=strategy_metrics.sharpe_ratio,
                        drawdown=strategy_metrics.max_drawdown,
                        notes=f"Strategy evaluation for {name}"
                    )
                    
                    metrics[name] = strategy_metrics
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating strategy {name}: {str(e)}")
                    continue
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategies: {str(e)}")
            return {}
    
    def rank_strategies(self, metrics: Dict[str, StrategyMetrics]) -> List[tuple]:
        """Rank strategies based on their performance metrics.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            List of (strategy_name, score) tuples, sorted by score
            
        Raises:
            StrategyError: If strategy ranking fails
        """
        try:
            scores = []
            for name, metric in metrics.items():
                # Calculate composite score
                score = (
                    metric.sharpe_ratio * 0.3 +
                    metric.sortino_ratio * 0.2 +
                    (1 - abs(metric.max_drawdown)) * 0.2 +
                    metric.win_rate * 0.15 +
                    metric.profit_factor * 0.15
                )
                
                # Apply minimum performance threshold
                if score < self.config['min_performance_threshold']:
                    self.logger.warning(f"Strategy {name} below performance threshold: {score}")
                    continue
                    
                scores.append((name, score))
            
            return sorted(scores, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            raise StrategyError(f"Failed to rank strategies: {str(e)}")
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save strategy results to disk.
        
        Args:
            results: Results to save
            filename: Output filename
            
        Raises:
            StrategyError: If results saving fails
        """
        try:
            filepath = self.results_dir / filename
            
            # Convert metrics to dictionaries
            results_dict = {}
            for name, metrics in results.items():
                if isinstance(metrics, StrategyMetrics):
                    results_dict[name] = metrics.to_dict()
                else:
                    results_dict[name] = metrics
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=4)
                
            self.logger.info(f"Saved results to {filepath}")
            
        except Exception as e:
            raise StrategyError(f"Failed to save results: {str(e)}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load strategy results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
            
        Raises:
            StrategyError: If results loading fails
        """
        try:
            filepath = self.results_dir / filename
            
            with open(filepath, 'r') as f:
                results_dict = json.load(f)
            
            # Convert dictionaries to metrics
            results = {}
            for name, data in results_dict.items():
                if 'returns' in data:  # It's a metrics object
                    results[name] = StrategyMetrics.from_dict(data)
                else:
                    results[name] = data
            
            self.logger.info(f"Loaded results from {filepath}")
            return results
            
        except Exception as e:
            raise StrategyError(f"Failed to load results: {str(e)}")
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Cache a value using Redis if available."""
        if not self.redis_client:
            self.logger.debug("Redis not available, skipping cache")
            return
        
        try:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                value = value.to_json()
            elif isinstance(value, StrategyMetrics):
                value = value.to_json()
            elif not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)
            
            self.redis_client.setex(key, ttl, value)
            self.logger.debug(f"Cached value for key: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to cache value: {e}")
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cached value from Redis if available."""
        if not self.redis_client:
            self.logger.debug("Redis not available, skipping cache")
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            value = value.decode('utf-8')
            try:
                # Try to parse as JSON first
                return json.loads(value)
            except json.JSONDecodeError:
                # If not JSON, return as is
                return value
        except Exception as e:
            self.logger.warning(f"Failed to get cached value: {e}")
            return None
    
    def simulate(self, data: pd.DataFrame, strategy_name: str, config: Optional[Dict] = None) -> StrategyMetrics:
        """Simulate a strategy run and return metrics.
        
        Args:
            data: Market data with OHLCV columns
            strategy_name: Name of the strategy to simulate
            config: Optional configuration for the strategy
            
        Returns:
            StrategyMetrics object with performance metrics
        """
        strategy = self.get_strategy(strategy_name)
        
        # Try to get cached results
        cache_key = f"sim_{strategy_name}_{hash(str(data))}"
        cached_metrics = self.get_cache(cache_key)
        if cached_metrics:
            self.logger.info(f"Using cached results for {strategy_name}")
            return StrategyMetrics.from_dict(cached_metrics)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Evaluate performance
        metrics = strategy.evaluate_performance(signals, data)
        
        # Cache results
        self.set_cache(cache_key, metrics.to_dict())
        
        # Print summary
        self.logger.info(f"Strategy {strategy_name} simulation results:")
        self.logger.info(metrics.summary())
        
        return metrics 