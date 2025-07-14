"""
Strategy Registry

This module provides a unified registry for all trading strategies.
It allows for easy strategy discovery, registration, and execution.
Enhanced with dynamic discovery of strategy modules via introspection and file globbing.
"""

import importlib
import inspect
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

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

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get the parameter space for optimization."""

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters."""
        self.parameters.update(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self.parameters.copy()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        return all(col in data.columns for col in required_columns)


class StrategyDiscovery:
    """Handles dynamic discovery of strategy modules."""

    def __init__(self, strategies_dir: str = None):
        """Initialize strategy discovery."""
        if strategies_dir is None:
            # Default to the directory containing this file
            strategies_dir = os.path.dirname(os.path.abspath(__file__))

        self.strategies_dir = Path(strategies_dir)
        self.discovered_strategies: Dict[str, Type[BaseStrategy]] = {}
        self.discovery_log: List[Dict[str, Any]] = []

    def discover_strategies(
        self, recursive: bool = True, pattern: str = "*_strategy.py"
    ) -> Dict[str, Type[BaseStrategy]]:
        """Discover strategy modules using file globbing and introspection."""
        logger.info(f"🔍 Discovering strategies in: {self.strategies_dir}")

        # Find strategy files
        if recursive:
            strategy_files = list(self.strategies_dir.rglob(pattern))
        else:
            strategy_files = list(self.strategies_dir.glob(pattern))

        logger.info(f"📁 Found {len(strategy_files)} potential strategy files")

        for file_path in strategy_files:
            try:
                self._discover_strategies_from_file(file_path)
            except Exception as e:
                logger.error(f"❌ Failed to discover strategies from {file_path}: {e}")
                self.discovery_log.append(
                    {
                        "file": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        logger.info(f"✅ Discovered {len(self.discovered_strategies)} strategy classes")
        return self.discovered_strategies.copy()

    def _discover_strategies_from_file(self, file_path: Path):
        """Discover strategy classes from a single file."""
        try:
            # Convert file path to module path
            relative_path = file_path.relative_to(self.strategies_dir)
            module_path = str(relative_path).replace(os.sep, ".").replace(".py", "")

            # Import the module
            module_name = f"trading.strategies.{module_path}"
            module = importlib.import_module(module_name)

            # Find strategy classes in the module
            strategy_classes = self._find_strategy_classes(module)

            for class_name, strategy_class in strategy_classes.items():
                if class_name not in self.discovered_strategies:
                    self.discovered_strategies[class_name] = strategy_class
                    logger.info(
                        f"📦 Discovered strategy: {class_name} from {module_name}"
                    )

                    self.discovery_log.append(
                        {
                            "file": str(file_path),
                            "module": module_name,
                            "class": class_name,
                            "status": "success",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    logger.warning(
                        f"⚠️  Strategy {class_name} already discovered, skipping duplicate"
                    )

        except ImportError as e:
            logger.error(f"❌ Failed to import module {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error discovering strategies from {file_path}: {e}")
            raise

    def _find_strategy_classes(self, module) -> Dict[str, Type[BaseStrategy]]:
        """Find strategy classes in a module using introspection."""
        strategy_classes = {}

        for name, obj in inspect.getmembers(module):
            # Check if it's a class that inherits from BaseStrategy
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseStrategy)
                and obj != BaseStrategy
            ):
                strategy_classes[name] = obj

        return strategy_classes

    def discover_strategies_by_pattern(
        self, patterns: List[str] = None
    ) -> Dict[str, Type[BaseStrategy]]:
        """Discover strategies using multiple file patterns."""
        if patterns is None:
            patterns = [
                "*_strategy.py",
                "*_signals.py",
                "*_indicator.py",
                "strategy_*.py",
            ]

        all_strategies = {}

        for pattern in patterns:
            logger.info(f"🔍 Searching with pattern: {pattern}")
            strategies = self.discover_strategies(pattern=pattern)
            all_strategies.update(strategies)

        return all_strategies

    def get_discovery_log(self) -> List[Dict[str, Any]]:
        """Get the discovery log."""
        return self.discovery_log.copy()

    def clear_discovery_log(self):
        """Clear the discovery log."""
        self.discovery_log.clear()

    def validate_discovered_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Validate discovered strategies for completeness."""
        validation_results = {}

        for class_name, strategy_class in self.discovered_strategies.items():
            validation_result = {"valid": True, "errors": [], "warnings": []}

            try:
                # Check if class can be instantiated
                instance = strategy_class()

                # Check required methods
                required_methods = ["generate_signals", "get_parameter_space"]
                for method in required_methods:
                    if not hasattr(instance, method):
                        validation_result["errors"].append(
                            f"Missing required method: {method}"
                        )
                        validation_result["valid"] = False

                # Check if methods are callable
                if hasattr(instance, "generate_signals"):
                    if not callable(instance.generate_signals):
                        validation_result["errors"].append(
                            "generate_signals is not callable"
                        )
                        validation_result["valid"] = False

                if hasattr(instance, "get_parameter_space"):
                    if not callable(instance.get_parameter_space):
                        validation_result["errors"].append(
                            "get_parameter_space is not callable"
                        )
                        validation_result["valid"] = False

                # Check for description
                if not hasattr(instance, "description") or not instance.description:
                    validation_result["warnings"].append("No description provided")

            except Exception as e:
                validation_result["errors"].append(f"Instantiation failed: {str(e)}")
                validation_result["valid"] = False

            validation_results[class_name] = validation_result

        return validation_results


class RSIStrategy(BaseStrategy):
    """Relative Strength Index strategy."""

    def __init__(self):
        super().__init__("RSI", "Relative Strength Index strategy")
        self.parameters = {"period": 14, "overbought": 70, "oversold": 30}

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate RSI signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")

        # Calculate RSI
        delta = data["Close"].diff()
        gain = (
            (delta.where(delta > 0, 0)).rolling(window=self.parameters["period"]).mean()
        )
        loss = (
            (-delta.where(delta < 0, 0))
            .rolling(window=self.parameters["period"])
            .mean()
        )
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals["rsi"] = rsi
        signals["signal"] = 0

        # Buy signal when RSI crosses below oversold
        signals.loc[rsi < self.parameters["oversold"], "signal"] = 1

        # Sell signal when RSI crosses above overbought
        signals.loc[rsi > self.parameters["overbought"], "signal"] = -1

        return signals

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get RSI parameter space."""
        return {"period": (5, 30), "overbought": (60, 90), "oversold": (10, 40)}


class MACDStrategy(BaseStrategy):
    """MACD strategy."""

    def __init__(self):
        super().__init__("MACD", "Moving Average Convergence Divergence strategy")
        self.parameters = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate MACD signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")

        # Calculate MACD
        ema_fast = data["Close"].ewm(span=self.parameters["fast_period"]).mean()
        ema_slow = data["Close"].ewm(span=self.parameters["slow_period"]).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.parameters["signal_period"]).mean()
        histogram = macd_line - signal_line

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals["macd"] = macd_line
        signals["signal_line"] = signal_line
        signals["histogram"] = histogram
        signals["signal"] = 0

        # Buy signal when MACD crosses above signal line
        signals.loc[macd_line > signal_line, "signal"] = 1

        # Sell signal when MACD crosses below signal line
        signals.loc[macd_line < signal_line, "signal"] = -1

        return signals

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get MACD parameter space."""
        return {
            "fast_period": (5, 20),
            "slow_period": (20, 50),
            "signal_period": (5, 20),
        }


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy."""

    def __init__(self):
        super().__init__("Bollinger", "Bollinger Bands strategy")
        self.parameters = {"period": 20, "std_dev": 2.0}

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")

        # Calculate Bollinger Bands
        sma = data["Close"].rolling(window=self.parameters["period"]).mean()
        std = data["Close"].rolling(window=self.parameters["period"]).std()
        upper_band = sma + (std * self.parameters["std_dev"])
        lower_band = sma - (std * self.parameters["std_dev"])

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals["sma"] = sma
        signals["upper_band"] = upper_band
        signals["lower_band"] = lower_band
        signals["signal"] = 0

        # Buy signal when price touches lower band
        signals.loc[data["Close"] <= lower_band, "signal"] = 1

        # Sell signal when price touches upper band
        signals.loc[data["Close"] >= upper_band, "signal"] = -1

        return signals

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get Bollinger Bands parameter space."""
        return {"period": (10, 30), "std_dev": (1.0, 3.0)}


class CCIStrategy(BaseStrategy):
    """Commodity Channel Index strategy."""

    def __init__(self):
        super().__init__("CCI", "Commodity Channel Index strategy")
        self.parameters = {
            "period": 20,
            "constant": 0.015,
            "oversold_threshold": -100.0,
            "overbought_threshold": 100.0,
        }

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate CCI signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")

        # Calculate CCI
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        sma_tp = typical_price.rolling(window=self.parameters["period"]).mean()
        mean_deviation = typical_price.rolling(window=self.parameters["period"]).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma_tp) / (self.parameters["constant"] * mean_deviation)

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals["cci"] = cci
        signals["signal"] = 0

        # Buy signal when CCI crosses above oversold threshold
        signals.loc[
            (cci > self.parameters["oversold_threshold"])
            & (cci.shift(1) <= self.parameters["oversold_threshold"]),
            "signal",
        ] = 1

        # Sell signal when CCI crosses below overbought threshold
        signals.loc[
            (cci < self.parameters["overbought_threshold"])
            & (cci.shift(1) >= self.parameters["overbought_threshold"]),
            "signal",
        ] = -1

        return signals

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get CCI parameter space."""
        return {
            "period": (10, 30),
            "constant": (0.01, 0.03),
            "oversold_threshold": (-200, -50),
            "overbought_threshold": (50, 200),
        }


class ATRStrategy(BaseStrategy):
    """Average True Range strategy."""

    def __init__(self):
        super().__init__("ATR", "Average True Range strategy")
        self.parameters = {
            "period": 14,
            "multiplier": 2.0,
            "volatility_threshold": 0.02,
        }

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate ATR signals."""
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")

        # Calculate ATR
        high_low = data["High"] - data["Low"]
        high_close = np.abs(data["High"] - data["Close"].shift(1))
        low_close = np.abs(data["Low"] - data["Close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.parameters["period"]).mean()

        # Calculate ATR-based Bollinger Bands
        middle_band = data["Close"].rolling(window=self.parameters["period"]).mean()
        upper_band = middle_band + (self.parameters["multiplier"] * atr)
        lower_band = middle_band - (self.parameters["multiplier"] * atr)

        # Calculate volatility filter
        volatility = (
            data["Close"].pct_change().rolling(window=self.parameters["period"]).std()
        )

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals["atr"] = atr
        signals["upper_band"] = upper_band
        signals["lower_band"] = lower_band
        signals["volatility"] = volatility
        signals["signal"] = 0

        # Buy signal when price touches lower band and volatility is high enough
        buy_condition = (data["Close"] <= lower_band) & (
            volatility >= self.parameters["volatility_threshold"]
        )
        signals.loc[buy_condition, "signal"] = 1

        # Sell signal when price touches upper band and volatility is high enough
        sell_condition = (data["Close"] >= upper_band) & (
            volatility >= self.parameters["volatility_threshold"]
        )
        signals.loc[sell_condition, "signal"] = -1

        return signals

    def get_parameter_space(self) -> Dict[str, Any]:
        """Get ATR parameter space."""
        return {
            "period": (10, 20),
            "multiplier": (1.0, 3.0),
            "volatility_threshold": (0.01, 0.05),
        }


class StrategyRegistry:
    """Registry for managing trading strategies."""

    def __init__(self):
        """Initialize the strategy registry."""
        self.strategies: Dict[str, BaseStrategy] = {}
        self.discovery = StrategyDiscovery()
        self._register_default_strategies()
        self._discover_strategies()

    def _register_default_strategies(self):
        """Register default strategies."""
        default_strategies = [
            RSIStrategy(),
            MACDStrategy(),
            BollingerBandsStrategy(),
            CCIStrategy(),
            ATRStrategy(),
        ]

        for strategy in default_strategies:
            self.register_strategy(strategy)

    def _discover_strategies(self):
        """Discover and register additional strategies."""
        try:
            discovered_strategies = self.discovery.discover_strategies()

            for class_name, strategy_class in discovered_strategies.items():
                try:
                    # Create instance and register
                    strategy_instance = strategy_class()
                    self.register_strategy(strategy_instance)
                    logger.info(f"✅ Auto-registered discovered strategy: {class_name}")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to register discovered strategy {class_name}: {e}"
                    )

        except Exception as e:
            logger.error(f"❌ Strategy discovery failed: {e}")

    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy."""
        self.strategies[strategy.name] = strategy
        logger.info(f"📝 Registered strategy: {strategy.name}")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)

    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies."""
        return self.strategies.copy()

    def get_strategy_names(self) -> List[str]:
        """Get names of all registered strategies."""
        return list(self.strategies.keys())

    def execute_strategy(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StrategyResult:
        """Execute a strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")

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
                "strategy_name": strategy_name,
                "parameters": strategy.get_parameters(),
                "description": strategy.description,
            },
            timestamp=datetime.now(),
            strategy_name=strategy_name,
        )

        return result

    def _calculate_performance_metrics(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics for strategy results."""
        # Simple performance calculation
        if "signal" not in signals.columns:
            return {}

        # Calculate returns
        data_returns = data["Close"].pct_change()
        strategy_returns = signals["signal"].shift(1) * data_returns

        # Calculate metrics
        total_return = strategy_returns.sum()
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std()
            if strategy_returns.std() > 0
            else 0
        )
        max_drawdown = (
            strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()
        ).min()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": strategy_returns.std(),
            "win_rate": (strategy_returns > 0).mean(),
        }

    def get_strategy_parameter_space(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameter space for a strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        return strategy.get_parameter_space()

    def get_discovery_info(self) -> Dict[str, Any]:
        """Get information about strategy discovery."""
        return {
            "discovered_strategies": len(self.discovery.discovered_strategies),
            "registered_strategies": len(self.strategies),
            "discovery_log": self.discovery.get_discovery_log(),
            "validation_results": self.discovery.validate_discovered_strategies(),
        }

    def refresh_discovery(self):
        """Refresh strategy discovery."""
        logger.info("🔄 Refreshing strategy discovery...")
        self.discovery.clear_discovery_log()
        self.discovery.discover_strategies()
        logger.info("✅ Strategy discovery refreshed")


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


def execute_strategy(
    strategy_name: str, data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None
) -> StrategyResult:
    """Execute a strategy using the global registry."""
    registry = get_strategy_registry()
    return registry.execute_strategy(strategy_name, data, parameters)
