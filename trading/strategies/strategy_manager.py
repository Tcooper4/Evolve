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
from trading.exceptions import StrategyError, StrategyNotFoundError, StrategyValidationError
from trading.core.performance import log_performance

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
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data.
        
        Args:
            data: Market data to validate
            
        Returns:
            Dictionary with validation status and details
            
        Raises:
            StrategyError: If data validation fails
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise StrategyError(f"Data missing required columns: {required_columns}")
                
            if data.empty:
                raise StrategyError("Data is empty")
                
            if data.isnull().any().any():
                raise StrategyError("Data contains null values")
            
            return {
                "status": "success",
                "message": "Data validation passed",
                "rows": len(data),
                "columns": len(data.columns)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def validate_signals(self, signals: pd.Series) -> Dict[str, Any]:
        """Validate trading signals.
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            Dictionary with validation status and details
            
        Raises:
            StrategyError: If signal validation fails
        """
        try:
            if not isinstance(signals, pd.Series):
                raise StrategyError("Signals must be a pandas Series")
                
            if signals.empty:
                raise StrategyError("Signals are empty")
                
            # Check for valid signal values
            valid_signals = [-1, 0, 1]
            invalid_signals = signals[~signals.isin(valid_signals)]
            if not invalid_signals.empty:
                raise StrategyError(f"Invalid signal values found: {invalid_signals.unique()}")
            
            return {
                "status": "success",
                "message": "Signal validation passed",
                "total_signals": len(signals),
                "buy_signals": (signals == 1).sum(),
                "sell_signals": (signals == -1).sum(),
                "hold_signals": (signals == 0).sum()
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

class StrategyManager:
    """Manages multiple trading strategies with caching and dynamic loading."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy manager.
        
        Args:
            config: Configuration dictionary containing:
                - strategies_dir: Directory containing strategy files
                - cache_enabled: Enable caching (default: True)
                - cache_ttl: Cache TTL in seconds (default: 3600)
                - max_strategies: Maximum number of strategies (default: 50)
                - auto_reload: Auto-reload strategies on changes (default: False)
        """
        self.config = config or {}
        
        # Strategy storage
        self.strategies = {}
        self.active_strategies = set()
        self.strategy_metrics = {}
        self.ensemble_weights = {}
        
        # Cache settings
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        self.cache = {}
        
        # Redis cache (optional)
        self.redis_client = None
        if REDIS_AVAILABLE and self.config.get('use_redis', False):
            try:
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    db=self.config.get('redis_db', 0),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
        
        # Performance tracking
        self.performance_history = []
        
        # Load strategies
        self._load_strategies()
        
        return {'success': True, 'message': 'Strategy manager initialized successfully', 'timestamp': datetime.now().isoformat()}

    def _load_strategies(self) -> None:
        """Load strategies from directory."""
        try:
            strategies_dir = self.config.get('strategies_dir', 'strategies')
            if not os.path.exists(strategies_dir):
                self.logger.warning(f"Strategies directory not found: {strategies_dir}")
                return {'success': False, 'error': 'Strategies directory not found', 'timestamp': datetime.now().isoformat()}
            
            for file in os.listdir(strategies_dir):
                if file.endswith('.py') and not file.startswith('_'):
                    strategy_name = file[:-3]
                    try:
                        module = importlib.import_module(f'strategies.{strategy_name}')
                        strategy_class = getattr(module, strategy_name.title().replace('_', ''))
                        self.strategies[strategy_name] = strategy_class()
                        self.logger.info(f"Loaded strategy: {strategy_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to load strategy {strategy_name}: {e}")
            
            return {'success': True, 'message': f'Loaded {len(self.strategies)} strategies', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _save_strategies(self) -> None:
        """Save strategy configurations."""
        try:
            config_file = self.config.get('config_file', 'strategy_config.json')
            config_data = {
                'active_strategies': list(self.active_strategies),
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return {'success': True, 'message': 'Strategy configuration saved', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def register_strategy(self, name: str, strategy: Strategy) -> Dict[str, Any]:
        """Register a new strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            
        Returns:
            Registration result
        """
        try:
            if name in self.strategies:
                return {'success': False, 'error': f'Strategy {name} already exists', 'timestamp': datetime.now().isoformat()}
            
            self.strategies[name] = strategy
            self.logger.info(f"Registered strategy: {name}")
            
            return {'success': True, 'message': f'Strategy {name} registered successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_strategy(self, name: str) -> Strategy:
        """Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance
        """
        try:
            if name not in self.strategies:
                return {'success': False, 'error': f'Strategy {name} not found', 'timestamp': datetime.now().isoformat()}
            
            return {'success': True, 'result': self.strategies[name], 'message': f'Strategy {name} retrieved successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def activate_strategy(self, name: str) -> Dict[str, Any]:
        """Activate a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Activation result
        """
        try:
            if name not in self.strategies:
                return {'success': False, 'error': f'Strategy {name} not found', 'timestamp': datetime.now().isoformat()}
            
            self.active_strategies.add(name)
            self._save_strategies()
            
            self.logger.info(f"Activated strategy: {name}")
            
            return {'success': True, 'message': f'Strategy {name} activated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def deactivate_strategy(self, name: str) -> Dict[str, Any]:
        """Deactivate a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Deactivation result
        """
        try:
            if name not in self.strategies:
                return {'success': False, 'error': f'Strategy {name} not found', 'timestamp': datetime.now().isoformat()}
            
            self.active_strategies.discard(name)
            self._save_strategies()
            
            self.logger.info(f"Deactivated strategy: {name}")
            
            return {'success': True, 'message': f'Strategy {name} deactivated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def set_ensemble(self, weights: Dict[str, float], strict: bool = True) -> Dict[str, Any]:
        """Set ensemble weights for strategies.
        
        Args:
            weights: Dictionary mapping strategy names to weights
            strict: If True, validate that all strategies exist
            
        Returns:
            Ensemble setup result
        """
        try:
            if strict:
                for strategy_name in weights:
                    if strategy_name not in self.strategies:
                        return {'success': False, 'error': f'Strategy {strategy_name} not found', 'timestamp': datetime.now().isoformat()}
            
            # Validate weights sum to 1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                return {'success': False, 'error': f'Weights must sum to 1.0, got {total_weight}', 'timestamp': datetime.now().isoformat()}
            
            self.ensemble_weights = weights.copy()
            self._save_strategies()
            
            self.logger.info(f"Set ensemble weights: {weights}")
            
            return {'success': True, 'message': 'Ensemble weights set successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals from active strategies.
        
        Args:
            data: Market data
            
        Returns:
            Combined signals from all active strategies
        """
        try:
            if not self.active_strategies:
                return {'success': False, 'error': 'No active strategies', 'timestamp': datetime.now().isoformat()}
            
            signals = {}
            for strategy_name in self.active_strategies:
                strategy = self.strategies[strategy_name]
                try:
                    strategy_signals = strategy.generate_signals(data)
                    signals[strategy_name] = strategy_signals
                except Exception as e:
                    self.logger.error(f"Error generating signals for {strategy_name}: {e}")
                    continue
            
            if not signals:
                return {'success': False, 'error': 'No signals generated', 'timestamp': datetime.now().isoformat()}
            
            # Combine signals using ensemble weights
            if self.ensemble_weights:
                combined_signals = pd.Series(0, index=data.index)
                for strategy_name, weight in self.ensemble_weights.items():
                    if strategy_name in signals:
                        combined_signals += weight * signals[strategy_name]['Signal']
            else:
                # Simple average
                combined_signals = pd.concat(signals.values(), axis=1)['Signal'].mean(axis=1)
            
            return {'success': True, 'result': combined_signals, 'message': 'Signals generated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, StrategyMetrics]:
        """Evaluate performance of all strategies.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary mapping strategy names to metrics
        """
        try:
            results = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    signals = strategy.generate_signals(data)
                    metrics = strategy.evaluate_performance(signals, data)
                    results[strategy_name] = metrics
                    self.strategy_metrics[strategy_name] = metrics
                except Exception as e:
                    self.logger.error(f"Error evaluating {strategy_name}: {e}")
                    continue
            
            return {'success': True, 'result': results, 'message': 'Strategy evaluation completed', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def rank_strategies(self, metrics: Dict[str, StrategyMetrics]) -> List[tuple]:
        """Rank strategies by performance.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            List of (strategy_name, sharpe_ratio) tuples sorted by Sharpe ratio
        """
        try:
            rankings = []
            for strategy_name, metric in metrics.items():
                rankings.append((strategy_name, metric.sharpe_ratio))
            
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return {'success': True, 'result': rankings, 'message': 'Strategies ranked successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def save_results(self, results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Save evaluation results to file.
        
        Args:
            results: Results to save
            filename: Output filename
            
        Returns:
            Save result
        """
        try:
            # Convert metrics to serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, StrategyMetrics):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {filename}")
            
            return {'success': True, 'message': f'Results saved to {filename}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> Dict[str, Any]:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Cache set result
        """
        try:
            if not self.cache_enabled:
                return {'success': False, 'error': 'Caching disabled', 'timestamp': datetime.now().isoformat()}
            
            if self.redis_client:
                # Use Redis
                self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                # Use local cache
                self.cache[key] = {
                    'value': value,
                    'expires': datetime.now().timestamp() + ttl
                }
            
            return {'success': True, 'message': 'Value cached successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_cache(self, key: str) -> Dict[str, Any]:
        """Get cache value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if not self.cache_enabled:
                return {'success': False, 'error': 'Caching disabled', 'timestamp': datetime.now().isoformat()}
            
            if self.redis_client:
                # Use Redis
                value = self.redis_client.get(key)
                if value:
                    return {'success': True, 'result': json.loads(value), 'message': 'Value retrieved from cache', 'timestamp': datetime.now().isoformat()}
            else:
                # Use local cache
                if key in self.cache:
                    cache_entry = self.cache[key]
                    if datetime.now().timestamp() < cache_entry['expires']:
                        return {'success': True, 'result': cache_entry['value'], 'message': 'Value retrieved from cache', 'timestamp': datetime.now().isoformat()}
                    else:
                        del self.cache[key]
            
            return {'success': False, 'error': 'Value not found in cache', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load evaluation results from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert back to StrategyMetrics objects
            results = {}
            for key, value in data.items():
                if isinstance(value, dict) and 'returns' in value:
                    results[key] = StrategyMetrics.from_dict(value)
                else:
                    results[key] = value
            
            self.logger.info(f"Results loaded from {filename}")
            
            return {'success': True, 'result': results, 'message': f'Results loaded from {filename}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def simulate(self, data: pd.DataFrame, strategy_name: str, config: Optional[Dict] = None) -> StrategyMetrics:
        """Simulate strategy performance.
        
        Args:
            data: Market data
            strategy_name: Strategy to simulate
            config: Optional configuration overrides
            
        Returns:
            Strategy performance metrics
        """
        try:
            if strategy_name not in self.strategies:
                return {'success': False, 'error': f'Strategy {strategy_name} not found', 'timestamp': datetime.now().isoformat()}
            
            strategy = self.strategies[strategy_name]
            
            # Apply configuration overrides
            if config:
                original_config = strategy.config.copy()
                strategy.config.update(config)
            
            try:
                signals = strategy.generate_signals(data)
                metrics = strategy.evaluate_performance(signals, data)
            finally:
                # Restore original configuration
                if config:
                    strategy.config = original_config
            
            return {'success': True, 'result': metrics, 'message': f'Simulation completed for {strategy_name}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def set_parameters(self, **kwargs) -> Dict[str, Any]:
        """Set strategy parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
            
        Returns:
            Dictionary with parameter update status
        """
        try:
            for param, value in kwargs.items():
                if param in self.config:
                    self.config[param] = value
                    self.logger.info(f"Updated parameter {param} = {value}")
                else:
                    self.logger.warning(f"Unknown parameter: {param}")
            
            return {'success': True, 'message': 'Parameters updated successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters.
        
        Returns:
            Dictionary with current parameters
        """
        try:
            return {'success': True, 'result': self.config.copy(), 'message': 'Parameters retrieved successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def reset_parameters(self) -> Dict[str, Any]:
        """Reset parameters to defaults.
        
        Returns:
            Dictionary with reset status
        """
        try:
            self.config = {
                'log_level': 'INFO',
                'max_position_size': 100000,
                'min_position_size': 1000,
                'max_leverage': 1.0,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
            
            return {'success': True, 'message': 'Parameters reset to defaults', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()} 