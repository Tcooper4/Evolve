"""
Strategy Registry

This module provides a unified registry for all trading strategies.
It allows for easy strategy discovery, registration, and execution.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StrategyResult:
    """Result of a strategy execution."""
    signals: pd.DataFrame
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime
    strategy_name: str

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize strategy."""
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get the parameter space for optimization."""
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters."""
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self.parameters.copy()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)

class RSIStrategy(BaseStrategy):
    """Relative Strength Index strategy."""
    
    def __init__(self):
        super().__init__("RSI", "Relative Strength Index strategy")
        self.parameters = {
            'period': 14,
            'overbought': 70,
            'oversold': 30
        }
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate RSI signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        signals['signal'] = 0
        
        # Buy signal when RSI crosses below oversold
        signals.loc[rsi < self.parameters['oversold'], 'signal'] = 1
        
        # Sell signal when RSI crosses above overbought
        signals.loc[rsi > self.parameters['overbought'], 'signal'] = -1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get RSI parameter space."""
        return {
            'period': (5, 30),
            'overbought': (60, 90),
            'oversold': (10, 40)
        }

class MACDStrategy(BaseStrategy):
    """MACD strategy."""
    
    def __init__(self):
        super().__init__("MACD", "Moving Average Convergence Divergence strategy")
        self.parameters = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate MACD signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=self.parameters['fast_period']).mean()
        ema_slow = data['Close'].ewm(span=self.parameters['slow_period']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.parameters['signal_period']).mean()
        histogram = macd_line - signal_line
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['macd'] = macd_line
        signals['signal_line'] = signal_line
        signals['histogram'] = histogram
        signals['signal'] = 0
        
        # Buy signal when MACD crosses above signal line
        signals.loc[macd_line > signal_line, 'signal'] = 1
        
        # Sell signal when MACD crosses below signal line
        signals.loc[macd_line < signal_line, 'signal'] = -1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get MACD parameter space."""
        return {
            'fast_period': (5, 20),
            'slow_period': (20, 50),
            'signal_period': (5, 20)
        }

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy."""
    
    def __init__(self):
        super().__init__("Bollinger", "Bollinger Bands strategy")
        self.parameters = {
            'period': 20,
            'std_dev': 2.0
        }
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
        
        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=self.parameters['period']).mean()
        std = data['Close'].rolling(window=self.parameters['period']).std()
        upper_band = sma + (std * self.parameters['std_dev'])
        lower_band = sma - (std * self.parameters['std_dev'])
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['sma'] = sma
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['signal'] = 0
        
        # Buy signal when price touches lower band
        signals.loc[data['Close'] <= lower_band, 'signal'] = 1
        
        # Sell signal when price touches upper band
        signals.loc[data['Close'] >= upper_band, 'signal'] = -1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get Bollinger Bands parameter space."""
        return {
            'period': (10, 30),
            'std_dev': (1.0, 3.0)
        }

class CCIStrategy(BaseStrategy):
    """Commodity Channel Index strategy."""
    
    def __init__(self):
        super().__init__("CCI", "Commodity Channel Index strategy")
        self.parameters = {
            'period': 20,
            'constant': 0.015,
            'oversold_threshold': -100.0,
            'overbought_threshold': 100.0
        }
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate CCI signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
        
        # Calculate CCI
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=self.parameters['period']).mean()
        mean_deviation = typical_price.rolling(window=self.parameters['period']).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma_tp) / (self.parameters['constant'] * mean_deviation)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['cci'] = cci
        signals['signal'] = 0
        
        # Buy signal when CCI crosses above oversold threshold
        signals.loc[(cci > self.parameters['oversold_threshold']) & 
                   (cci.shift(1) <= self.parameters['oversold_threshold']), 'signal'] = 1
        
        # Sell signal when CCI crosses below overbought threshold
        signals.loc[(cci < self.parameters['overbought_threshold']) & 
                   (cci.shift(1) >= self.parameters['overbought_threshold']), 'signal'] = -1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get CCI parameter space."""
        return {
            'period': (10, 30),
            'constant': (0.01, 0.03),
            'oversold_threshold': (-200, -50),
            'overbought_threshold': (50, 200)
        }

class ATRStrategy(BaseStrategy):
    """Average True Range strategy."""
    
    def __init__(self):
        super().__init__("ATR", "Average True Range strategy")
        self.parameters = {
            'period': 14,
            'multiplier': 2.0,
            'volatility_threshold': 0.02
        }
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate ATR signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
        
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.parameters['period']).mean()
        
        # Calculate ATR-based Bollinger Bands
        middle_band = data['Close'].rolling(window=self.parameters['period']).mean()
        upper_band = middle_band + (self.parameters['multiplier'] * atr)
        lower_band = middle_band - (self.parameters['multiplier'] * atr)
        
        # Calculate volatility filter
        volatility = data['Close'].pct_change().rolling(window=self.parameters['period']).std()
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['atr'] = atr
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['volatility'] = volatility
        signals['signal'] = 0
        
        # Buy signal when price touches lower band and volatility is high enough
        buy_condition = (data['Close'] <= lower_band) & (volatility >= self.parameters['volatility_threshold'])
        signals.loc[buy_condition, 'signal'] = 1
        
        # Sell signal when price touches upper band and volatility is high enough
        sell_condition = (data['Close'] >= upper_band) & (volatility >= self.parameters['volatility_threshold'])
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get ATR parameter space."""
        return {
            'period': (10, 20),
            'multiplier': (1.0, 3.0),
            'volatility_threshold': (0.01, 0.05)
        }

class StrategyRegistry:
    """Registry for managing trading strategies."""
    
    def __init__(self):
        """Initialize strategy registry."""
        self._strategies: Dict[str, BaseStrategy] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies."""
        self.register_strategy(RSIStrategy())
        self.register_strategy(MACDStrategy())
        self.register_strategy(BollingerBandsStrategy())
        self.register_strategy(CCIStrategy())
        self.register_strategy(ATRStrategy())
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy."""
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("Strategy must inherit from BaseStrategy")
        
        self._strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies."""
        return self._strategies.copy()
    
    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy names."""
        return list(self._strategies.keys())
    
    def execute_strategy(self, strategy_name: str, data: pd.DataFrame, 
                        parameters: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Execute a strategy with given data and parameters."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        # Set parameters if provided
        if parameters:
            strategy.set_parameters(parameters)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data, signals)
        
        # Create result
        result = StrategyResult(
            signals=signals,
            performance_metrics=performance_metrics,
            metadata={
                'strategy_name': strategy_name,
                'parameters': strategy.get_parameters(),
                'data_shape': data.shape
            },
            timestamp=datetime.now(),
            strategy_name=strategy_name
        )
        
        return result
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, 
                                     signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for strategy signals."""
        try:
            # Calculate returns
            returns = data['Close'].pct_change()
            strategy_returns = signals['signal'].shift(1) * returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
            win_rate = (strategy_returns > 0).mean()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': strategy_returns.std()
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }
    
    def get_strategy_parameter_space(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameter space for a strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        return strategy.get_parameter_space()

# Global registry instance
_strategy_registry = None

def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()
    return _strategy_registry

def register_strategy(strategy: BaseStrategy):
    """Register a strategy in the global registry."""
    registry = get_strategy_registry()
    registry.register_strategy(strategy)

def get_strategy(name: str) -> Optional[BaseStrategy]:
    """Get a strategy from the global registry."""
    registry = get_strategy_registry()
    return registry.get_strategy(name)

def execute_strategy(strategy_name: str, data: pd.DataFrame, 
                    parameters: Optional[Dict[str, Any]] = None) -> StrategyResult:
    """Execute a strategy using the global registry."""
    registry = get_strategy_registry()
    return registry.execute_strategy(strategy_name, data, parameters)
