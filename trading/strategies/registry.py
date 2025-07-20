"""
Strategy Registry

This module provides dynamic discovery, registration, and execution of trading strategies.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

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

from .base_strategy import BaseStrategy

class StrategyDiscovery:
    """Handles dynamic discovery of strategy modules."""
    def __init__(self, strategies_dir: str = None):
        """Initialize strategy discovery."""
        if strategies_dir is None:
            strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.strategies_dir = Path(strategies_dir)
        self.discovered_strategies: Dict[str, Type[BaseStrategy]] = {}
        self.discovery_log: List[Dict[str, Any]] = []

    def discover_strategies(self, recursive: bool = True, pattern: str = "*_strategy.py") -> Dict[str, Type[BaseStrategy]]:
        """Discover strategy modules using file globbing and introspection."""
        logger.info(f"🔍 Discovering strategies in: {self.strategies_dir}")
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
                self.discovery_log.append({
                    "file": str(file_path),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
        logger.info(f"✅ Discovered {len(self.discovered_strategies)} strategy classes")
        return self.discovered_strategies.copy()

    def _discover_strategies_from_file(self, file_path: Path):
        """Discover strategy classes from a single file."""
        try:
            relative_path = file_path.relative_to(self.strategies_dir)
            module_path = str(relative_path).replace(os.sep, ".").replace(".py", "")
            module_name = f"trading.strategies.{module_path}"
            module = importlib.import_module(module_name)
            strategy_classes = self._find_strategy_classes(module)
            for class_name, strategy_class in strategy_classes.items():
                if class_name not in self.discovered_strategies:
                    self.discovered_strategies[class_name] = strategy_class
                    logger.info(f"📦 Discovered strategy: {class_name} from {module_name}")
                    self.discovery_log.append({
                        "file": str(file_path),
                        "module": module_name,
                        "class": class_name,
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                    })
                else:
                    logger.warning(f"⚠️  Strategy {class_name} already discovered, skipping duplicate")
        except ImportError as e:
            logger.error(f"❌ Failed to import module {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error discovering strategies from {file_path}: {e}")
            raise

    def _find_strategy_classes(self, module) -> Dict[str, Type[BaseStrategy]]:
        """Find strategy classes in a module."""
        strategy_classes = {}
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseStrategy) and attr is not BaseStrategy:
                strategy_classes[attr_name] = attr
        return strategy_classes

    def discover_strategies_by_pattern(self, patterns: List[str] = None) -> Dict[str, Type[BaseStrategy]]:
        """Discover strategies by multiple patterns."""
        patterns = patterns or ["*_strategy.py"]
        all_strategies = {}
        for pattern in patterns:
            all_strategies.update(self.discover_strategies(pattern=pattern))
        return all_strategies

    def get_discovery_log(self) -> List[Dict[str, Any]]:
        """Get the discovery log."""
        return self.discovery_log.copy()

    def clear_discovery_log(self):
        """Clear the discovery log."""
        self.discovery_log.clear()

    def validate_discovered_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Validate discovered strategies."""
        results = {}
        for name, cls in self.discovered_strategies.items():
            try:
                instance = cls(name)
                valid = hasattr(instance, "generate_signals") and hasattr(instance, "get_parameter_space")
                results[name] = {"valid": valid, "error": None}
            except Exception as e:
                results[name] = {"valid": False, "error": str(e)}
        return results

class StrategyRegistry:
    """Central registry for trading strategies."""
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self._register_default_strategies()
        self._discover_strategies()

    def _register_default_strategies(self):
        # Register built-in strategies here if needed
        pass

    def _discover_strategies(self):
        discovery = StrategyDiscovery()
        discovered = discovery.discover_strategies()
        for name, cls in discovered.items():
            try:
                instance = cls(name)
                self.strategies[name] = instance
            except Exception as e:
                logger.error(f"Failed to instantiate strategy {name}: {e}")

    def register_strategy(self, strategy: BaseStrategy):
        self.strategies[strategy.get_name()] = strategy

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        return self.strategies.get(name)

    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        return self.strategies.copy()

    def get_strategy_names(self) -> List[str]:
        return list(self.strategies.keys())

    def execute_strategy(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StrategyResult:
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        if parameters:
            strategy.set_parameters(parameters)
        signals = strategy.generate_signals(data, **(parameters or {}))
        performance_metrics = self._calculate_performance_metrics(data, signals)
        return StrategyResult(
            signals=signals,
            performance_metrics=performance_metrics,
            metadata={},
            timestamp=datetime.now(),
            strategy_name=strategy_name
        )

    def _calculate_performance_metrics(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
        # Dummy implementation for performance metrics
        if signals.empty or data.empty:
            return {"confidence": 0.0, "signal_count": 0}
        signal_count = len(signals)
        buy_signals = len(signals[signals > 0])
        sell_signals = len(signals[signals < 0])
        signal_strength = signals.abs().mean().mean() if not signals.empty else 0.0
        confidence = min(signal_strength, 1.0)
        return {
            "confidence": confidence,
            "signal_count": signal_count,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "signal_strength": signal_strength
        }

    def get_strategy_parameter_space(self, strategy_name: str) -> Dict[str, Any]:
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        return strategy.get_parameter_space()

    def get_discovery_info(self) -> Dict[str, Any]:
        return {
            "strategy_count": len(self.strategies),
            "strategy_names": self.get_strategy_names(),
        }

    def refresh_discovery(self):
        self.strategies.clear()
        self._register_default_strategies()
        self._discover_strategies()

# Global registry instance
_registry: Optional[StrategyRegistry] = None

def get_strategy_registry() -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry

def register_strategy(strategy: BaseStrategy):
    registry = get_strategy_registry()
    registry.register_strategy(strategy)

def get_strategy(name: str) -> Optional[BaseStrategy]:
    registry = get_strategy_registry()
    return registry.get_strategy(name)

def execute_strategy(
    strategy_name: str, data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None
) -> StrategyResult:
    registry = get_strategy_registry()
    return registry.execute_strategy(strategy_name, data, parameters) 