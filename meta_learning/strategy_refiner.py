"""
Strategy Refiner

This module provides meta-learning capabilities for strategy refinement with
recency weighting, plug-and-play scoring, and comprehensive logging.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

import pandas as pd
import numpy as np
from dataclasses import dataclass

from trading.agents.base_agent_interface import BaseAgent, AgentResult, AgentConfig


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    timestamp: datetime
    recency_weight: float = 1.0


class ScoringFunction:
    """Base class for scoring functions"""
    
    def calculate_score(self, performance: StrategyPerformance) -> float:
        """Calculate score for a strategy performance"""
        raise NotImplementedError


class SharpeScoring(ScoringFunction):
    """Sharpe ratio based scoring"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_score(self, performance: StrategyPerformance) -> float:
        """Calculate Sharpe-based score"""
        if performance.volatility == 0:
            return 0.0
        
        excess_return = performance.total_return - self.risk_free_rate
        sharpe = excess_return / performance.volatility
        return max(0, sharpe) * performance.recency_weight


class MSEScoring(ScoringFunction):
    """Mean Squared Error based scoring (for forecasting)"""
    
    def __init__(self, target_return: float = 0.1):
        self.target_return = target_return
    
    def calculate_score(self, performance: StrategyPerformance) -> float:
        """Calculate MSE-based score"""
        mse = (performance.total_return - self.target_return) ** 2
        return max(0, 1 - mse) * performance.recency_weight


class WinRateScoring(ScoringFunction):
    """Win rate based scoring"""
    
    def calculate_score(self, performance: StrategyPerformance) -> float:
        """Calculate win rate based score"""
        return performance.win_rate * performance.recency_weight


class CompositeScoring(ScoringFunction):
    """Composite scoring using multiple metrics"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'sharpe': 0.3,
            'win_rate': 0.25,
            'calmar': 0.25,
            'sortino': 0.2
        }
    
    def calculate_score(self, performance: StrategyPerformance) -> float:
        """Calculate composite score"""
        scores = {
            'sharpe': max(0, performance.sharpe_ratio),
            'win_rate': performance.win_rate,
            'calmar': max(0, performance.calmar_ratio),
            'sortino': max(0, performance.sortino_ratio)
        }
        
        composite_score = sum(
            scores[metric] * weight 
            for metric, weight in self.weights.items()
        )
        
        return composite_score * performance.recency_weight


class StrategyRefiner(BaseAgent):
    """Agent for refining strategies using meta-learning."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Strategy performance history
        self.performance_history: List[StrategyPerformance] = []
        self.history_file = Path("logs/strategy_performance_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_performance_history()
        
        # Scoring functions
        self.scoring_functions = {
            'sharpe': SharpeScoring(),
            'mse': MSEScoring(),
            'win_rate': WinRateScoring(),
            'composite': CompositeScoring()
        }
        
        # Recency weighting parameters
        self.recency_decay_rate = 0.1  # Exponential decay rate
        self.max_history_days = 90  # Maximum days to consider
        
        # Strategy selection parameters
        self.top_strategies_count = 5
        self.min_performance_threshold = 0.5
        
        # Refinement parameters
        self.refinement_iterations = 3
        self.parameter_mutation_rate = 0.2
        
    def _load_performance_history(self) -> None:
        """Load strategy performance history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    
                for entry in data:
                    performance = StrategyPerformance(
                        strategy_name=entry['strategy_name'],
                        sharpe_ratio=entry['sharpe_ratio'],
                        max_drawdown=entry['max_drawdown'],
                        win_rate=entry['win_rate'],
                        total_return=entry['total_return'],
                        volatility=entry['volatility'],
                        calmar_ratio=entry['calmar_ratio'],
                        sortino_ratio=entry['sortino_ratio'],
                        timestamp=datetime.fromisoformat(entry['timestamp'])
                    )
                    self.performance_history.append(performance)
                    
        except Exception as e:
            self.logger.error(f"Failed to load performance history: {e}")
    
    def _save_performance_history(self) -> None:
        """Save strategy performance history to file."""
        try:
            data = []
            for performance in self.performance_history:
                data.append({
                    'strategy_name': performance.strategy_name,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'win_rate': performance.win_rate,
                    'total_return': performance.total_return,
                    'volatility': performance.volatility,
                    'calmar_ratio': performance.calmar_ratio,
                    'sortino_ratio': performance.sortino_ratio,
                    'timestamp': performance.timestamp.isoformat(),
                    'recency_weight': performance.recency_weight
                })
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")
    
    def _calculate_recency_weights(self) -> None:
        """Calculate recency weights for all performance records."""
        current_time = datetime.now()
        
        for performance in self.performance_history:
            days_old = (current_time - performance.timestamp).days
            
            if days_old > self.max_history_days:
                performance.recency_weight = 0.0
            else:
                # Exponential decay
                performance.recency_weight = math.exp(-self.recency_decay_rate * days_old)
    
    def add_strategy_performance(
        self,
        strategy_name: str,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_return: float,
        volatility: float,
        calmar_ratio: float = None,
        sortino_ratio: float = None,
        timestamp: datetime = None
    ) -> None:
        """Add a new strategy performance record."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate missing ratios if not provided
        if calmar_ratio is None:
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        if sortino_ratio is None:
            # Simplified sortino calculation
            sortino_ratio = total_return / volatility if volatility > 0 else 0
        
        performance = StrategyPerformance(
            strategy_name=strategy_name,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return=total_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            timestamp=timestamp
        )
        
        self.performance_history.append(performance)
        self._save_performance_history()
        
        self.logger.info(f"Added performance for strategy: {strategy_name}")
    
    def get_top_strategies(
        self, 
        scoring_method: str = 'composite',
        limit: int = None
    ) -> List[Tuple[str, float]]:
        """Get top performing strategies using specified scoring method."""
        if not self.performance_history:
            return []
        
        # Calculate recency weights
        self._calculate_recency_weights()
        
        # Filter out strategies with zero recency weight
        recent_performances = [
            p for p in self.performance_history 
            if p.recency_weight > 0
        ]
        
        if not recent_performances:
            return []
        
        # Get scoring function
        scoring_func = self.scoring_functions.get(scoring_method)
        if not scoring_func:
            self.logger.warning(f"Unknown scoring method: {scoring_method}, using composite")
            scoring_func = self.scoring_functions['composite']
        
        # Calculate scores
        strategy_scores = []
        for performance in recent_performances:
            score = scoring_func.calculate_score(performance)
            if score >= self.min_performance_threshold:
                strategy_scores.append((performance.strategy_name, score))
        
        # Sort by score (descending)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit
        if limit:
            strategy_scores = strategy_scores[:limit]
        
        # Log selection reason
        if strategy_scores:
            top_strategy, top_score = strategy_scores[0]
            self.logger.info(
                f"Selected top strategy '{top_strategy}' with score {top_score:.3f} "
                f"using {scoring_method} scoring method"
            )
        
        return strategy_scores
    
    def refine_strategies(
        self,
        strategy_configs: List[Dict[str, Any]],
        scoring_method: str = 'composite'
    ) -> List[Dict[str, Any]]:
        """Refine strategy configurations based on performance history."""
        if not strategy_configs:
            return []
        
        # Get top strategies for reference
        top_strategies = self.get_top_strategies(scoring_method, self.top_strategies_count)
        top_strategy_names = [name for name, _ in top_strategies]
        
        refined_configs = []
        
        for config in strategy_configs:
            strategy_name = config.get('name', 'unknown')
            
            # Check if this strategy is in top performers
            is_top_performer = strategy_name in top_strategy_names
            
            if is_top_performer:
                # Minor refinements for top performers
                refined_config = self._minor_refinement(config)
                self.logger.info(f"Applied minor refinement to top performer: {strategy_name}")
            else:
                # Major refinements for underperformers
                refined_config = self._major_refinement(config, top_strategies)
                self.logger.info(f"Applied major refinement to underperformer: {strategy_name}")
            
            refined_configs.append(refined_config)
        
        return refined_configs
    
    def _minor_refinement(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply minor refinements to top performing strategies."""
        refined_config = config.copy()
        
        # Small parameter adjustments
        if 'parameters' in refined_config:
            params = refined_config['parameters']
            
            # Adjust thresholds slightly
            for key in ['threshold', 'stop_loss', 'take_profit']:
                if key in params:
                    current_value = params[key]
                    # Small random adjustment (±5%)
                    adjustment = current_value * 0.05 * (np.random.random() - 0.5)
                    params[key] = max(0, current_value + adjustment)
        
        return refined_config
    
    def _major_refinement(
        self, 
        config: Dict[str, Any], 
        top_strategies: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Apply major refinements to underperforming strategies."""
        refined_config = config.copy()
        
        # Get best performing strategy as reference
        if top_strategies:
            best_strategy_name, _ = top_strategies[0]
            
            # Find best strategy's performance record
            best_performance = None
            for perf in self.performance_history:
                if perf.strategy_name == best_strategy_name:
                    best_performance = perf
                    break
            
            if best_performance:
                # Apply parameter mutations based on best performer
                refined_config = self._apply_parameter_mutations(
                    refined_config, best_performance
                )
        
        return refined_config
    
    def _apply_parameter_mutations(
        self, 
        config: Dict[str, Any], 
        best_performance: StrategyPerformance
    ) -> Dict[str, Any]:
        """Apply parameter mutations based on best performing strategy."""
        mutated_config = config.copy()
        
        if 'parameters' not in mutated_config:
            mutated_config['parameters'] = {}
        
        params = mutated_config['parameters']
        
        # Mutate parameters based on best performance characteristics
        if best_performance.win_rate > 0.6:
            # High win rate strategy - adjust for better risk management
            params['stop_loss'] = params.get('stop_loss', 0.02) * 0.8
            params['take_profit'] = params.get('take_profit', 0.04) * 1.2
        
        if best_performance.sharpe_ratio > 1.5:
            # High Sharpe ratio - adjust for better risk-adjusted returns
            params['position_size'] = params.get('position_size', 0.1) * 1.1
        
        if best_performance.max_drawdown < 0.1:
            # Low drawdown - adjust for better capital preservation
            params['max_position_size'] = params.get('max_position_size', 0.2) * 0.9
        
        # Random mutations
        for key in params:
            if np.random.random() < self.parameter_mutation_rate:
                current_value = params[key]
                if isinstance(current_value, (int, float)):
                    # Random adjustment (±20%)
                    adjustment = current_value * 0.2 * (np.random.random() - 0.5)
                    params[key] = max(0, current_value + adjustment)
        
        return mutated_config
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute strategy refinement process."""
        try:
            # Extract parameters
            scoring_method = kwargs.get('scoring_method', 'composite')
            strategy_configs = kwargs.get('strategy_configs', [])
            add_performance = kwargs.get('add_performance', {})
            
            # Add new performance data if provided
            if add_performance:
                self.add_strategy_performance(**add_performance)
            
            # Get top strategies
            top_strategies = self.get_top_strategies(scoring_method, self.top_strategies_count)
            
            # Refine strategies if configs provided
            refined_configs = []
            if strategy_configs:
                refined_configs = self.refine_strategies(strategy_configs, scoring_method)
            
            # Calculate performance statistics
            stats = self._calculate_performance_statistics()
            
            return AgentResult(
                success=True,
                data={
                    'top_strategies': top_strategies,
                    'refined_configs': refined_configs,
                    'total_strategies': len(self.performance_history),
                    'scoring_method': scoring_method
                },
                extra_metrics=stats
            )
            
        except Exception as e:
            self.logger.error(f"Strategy refinement failed: {e}")
            return AgentResult(
                success=False,
                error_message=str(e)
            )
    
    def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.performance_history:
            return {}
        
        # Calculate recency weights
        self._calculate_recency_weights()
        
        # Filter recent performances
        recent_performances = [
            p for p in self.performance_history 
            if p.recency_weight > 0
        ]
        
        if not recent_performances:
            return {}
        
        # Calculate statistics
        stats = {
            'total_strategies': len(self.performance_history),
            'recent_strategies': len(recent_performances),
            'avg_sharpe_ratio': np.mean([p.sharpe_ratio for p in recent_performances]),
            'avg_win_rate': np.mean([p.win_rate for p in recent_performances]),
            'avg_total_return': np.mean([p.total_return for p in recent_performances]),
            'avg_max_drawdown': np.mean([p.max_drawdown for p in recent_performances]),
            'best_strategy': max(recent_performances, key=lambda p: p.sharpe_ratio).strategy_name,
            'worst_strategy': min(recent_performances, key=lambda p: p.sharpe_ratio).strategy_name
        }
        
        return stats
    
    def register_scoring_function(self, name: str, scoring_func: ScoringFunction) -> None:
        """Register a custom scoring function."""
        self.scoring_functions[name] = scoring_func
        self.logger.info(f"Registered custom scoring function: {name}")
    
    def get_performance_history(self, strategy_name: str = None) -> List[StrategyPerformance]:
        """Get performance history, optionally filtered by strategy name."""
        if strategy_name:
            return [p for p in self.performance_history if p.strategy_name == strategy_name]
        return self.performance_history.copy() 