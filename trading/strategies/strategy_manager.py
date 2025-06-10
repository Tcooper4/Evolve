from typing import List, Dict, Any, Optional, Callable, Type, Union
import pandas as pd
import numpy as np
import logging
import importlib
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
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

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals.
        
        Args:
            data: Market data
            
        Returns:
            Series of trading signals (1: buy, -1: sell, 0: hold)
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, data: pd.DataFrame, signals: pd.Series) -> StrategyMetrics:
        """Calculate strategy performance metrics.
        
        Args:
            data: Market data
            signals: Trading signals
            
        Returns:
            StrategyMetrics object
        """
        pass

class StrategyManager:
    """Manages multiple trading strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create results directory
        self.results_dir = Path("strategy_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize strategy tracking
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: List[str] = []
        self.ensemble_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[StrategyMetrics]] = {}
    
    def register_strategy(self, name: str, strategy: Strategy) -> None:
        """Register a new strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        self.performance_history[name] = []
        self.logger.info(f"Registered strategy: {name}")
    
    def load_strategy(self, module_path: str, class_name: str, name: str) -> None:
        """Dynamically load a strategy from a module.
        
        Args:
            module_path: Path to the strategy module
            class_name: Name of the strategy class
            name: Name to register the strategy under
        """
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            strategy = strategy_class(self.config.get(name, {}))
            self.register_strategy(name, strategy)
            self.logger.info(f"Loaded strategy {name} from {module_path}.{class_name}")
        except Exception as e:
            self.logger.error(f"Failed to load strategy {name}: {str(e)}")
            raise
    
    def activate_strategy(self, name: str) -> None:
        """Activate a strategy.
        
        Args:
            name: Strategy name
        """
        if name in self.strategies and name not in self.active_strategies:
            self.active_strategies.append(name)
            self.logger.info(f"Activated strategy: {name}")
    
    def deactivate_strategy(self, name: str) -> None:
        """Deactivate a strategy.
        
        Args:
            name: Strategy name
        """
        if name in self.active_strategies:
            self.active_strategies.remove(name)
            self.logger.info(f"Deactivated strategy: {name}")
    
    def set_ensemble(self, weights: Dict[str, float]) -> None:
        """Set ensemble weights for active strategies.
        
        Args:
            weights: Dictionary of strategy weights
        """
        if not all(name in self.active_strategies for name in weights.keys()):
            raise ValueError("All strategies in weights must be active")
        
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        self.ensemble_weights = weights
        self.logger.info(f"Set ensemble weights: {weights}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from active strategies.
        
        Args:
            data: Market data
            
        Returns:
            Combined trading signals
        """
        if not self.active_strategies:
            raise ValueError("No active strategies")
        
        signals = pd.DataFrame(index=data.index)
        
        for name in self.active_strategies:
            strategy = self.strategies[name]
            try:
                strategy_signals = strategy.generate_signals(data)
                signals[name] = strategy_signals
                
                # Calculate and store metrics
                metrics = strategy.calculate_metrics(data, strategy_signals)
                self.performance_history[name].append(metrics)
                
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
    
    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, StrategyMetrics]:
        """Evaluate all active strategies.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of strategy metrics
        """
        results = {}
        for name in self.active_strategies:
            strategy = self.strategies[name]
            try:
                signals = strategy.generate_signals(data)
                metrics = strategy.calculate_metrics(data, signals)
                results[name] = metrics
                self.logger.info(f"Evaluated strategy {name}")
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {name}: {str(e)}")
        
        return results
    
    def rank_strategies(self, metrics: Dict[str, StrategyMetrics]) -> List[tuple]:
        """Rank strategies based on their performance metrics.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            List of (strategy_name, score) tuples, sorted by score
        """
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
            scores.append((name, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save strategy results to disk.
        
        Args:
            results: Results to save
            filename: Output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{filename}_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for name, metrics in results.items():
            serializable_results[name] = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_trade': metrics.avg_trade,
                'total_trades': metrics.total_trades
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        self.logger.info(f"Saved results to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load strategy results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
        """
        filepath = self.results_dir / filename
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        self.logger.info(f"Loaded results from {filepath}")
        
        return results 