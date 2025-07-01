"""Self-Tuning Optimizer for Trading Strategies.

This module provides an adaptive optimizer that monitors trade performance
over time and automatically adjusts strategy parameters based on walk-forward metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationMetric(Enum):
    """Metrics used for optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"

@dataclass
class ParameterChange:
    """Record of a parameter change."""
    timestamp: datetime
    strategy: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    impact_score: float
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]] = None

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    strategy: str
    timestamp: datetime
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    old_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    improvement: Dict[str, float]
    confidence: float
    recommendations: List[str]

class SelfTuningOptimizer:
    """Self-tuning optimizer for trading strategies."""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_path: str = "logs/optimizer_history.json"):
        """Initialize self-tuning optimizer.
        
        Args:
            config: Configuration dictionary
            log_path: Path to save optimization history
        """
        self.config = config or {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Optimization settings
        self.evaluation_window = self.config.get('evaluation_window', 30)  # days
        self.min_trades_for_evaluation = self.config.get('min_trades_for_evaluation', 10)
        self.optimization_threshold = self.config.get('optimization_threshold', 0.05)  # 5% improvement
        self.max_parameter_changes = self.config.get('max_parameter_changes', 3)
        
        # Parameter bounds and step sizes
        self.parameter_bounds = self.config.get('parameter_bounds', {})
        self.parameter_steps = self.config.get('parameter_steps', {})
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.parameter_changes: List[ParameterChange] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # Load existing history
        self._load_history()
        
        logger.info(f"Self-tuning optimizer initialized with evaluation window: {self.evaluation_window} days")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _load_history(self):
        """Load optimization history from file."""
        try:
            if self.log_path.exists():
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    self.parameter_changes = [
                        ParameterChange(**change) for change in data.get('parameter_changes', [])
                    ]
                    self.optimization_history = [
                        OptimizationResult(**result) for result in data.get('optimization_history', [])
                    ]
                logger.info(f"Loaded optimization history: {len(self.performance_history)} records")
        except Exception as e:
            logger.warning(f"Failed to load optimization history: {e}")

    def _save_history(self):
        """Save optimization history to file."""
        try:
            data = {
                'performance_history': self.performance_history,
                'parameter_changes': [asdict(change) for change in self.parameter_changes],
                'optimization_history': [asdict(result) for result in self.optimization_history]
            }
            
            with open(self.log_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved optimization history to {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")

    def record_performance(self, 
                          strategy: str,
                          parameters: Dict[str, Any],
                          metrics: Dict[str, float],
                          trades: List[Dict[str, Any]]):
        """Record strategy performance for optimization.
        
        Args:
            strategy: Strategy name
            parameters: Current parameters
            metrics: Performance metrics
            trades: List of trades
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'parameters': parameters,
            'metrics': metrics,
            'num_trades': len(trades),
            'period_days': self.evaluation_window
        }
        
        self.performance_history.append(record)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=365)  # Keep 1 year
        self.performance_history = [
            record for record in self.performance_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        logger.info(f"Recorded performance for {strategy}: {metrics}")

    def should_optimize(self, strategy: str) -> bool:
        """Determine if strategy should be optimized.
        
        Args:
            strategy: Strategy name
            
        Returns:
            True if optimization is recommended
        """
        # Get recent performance records
        recent_records = [
            record for record in self.performance_history
            if record['strategy'] == strategy
        ]
        
        if len(recent_records) < 2:
            return False  # Need at least 2 records to compare
        
        # Check if performance is declining
        recent_metrics = recent_records[-1]['metrics']
        previous_metrics = recent_records[-2]['metrics']
        
        # Calculate performance change
        sharpe_change = (recent_metrics.get('sharpe_ratio', 0) - 
                        previous_metrics.get('sharpe_ratio', 0))
        
        return_change = (recent_metrics.get('total_return', 0) - 
                        previous_metrics.get('total_return', 0))
        
        # Optimize if performance is declining significantly
        return (sharpe_change < -self.optimization_threshold or 
                return_change < -self.optimization_threshold)
    
    def optimize_strategy(self, 
                         strategy: str,
                         current_parameters: Dict[str, Any],
                         current_metrics: Dict[str, float]) -> Optional[OptimizationResult]:
        """Optimize strategy parameters.
        
        Args:
            strategy: Strategy name
            current_parameters: Current parameters
            current_metrics: Current performance metrics
            
        Returns:
            Optimization result or None if no optimization needed
        """
        if not self.should_optimize(strategy):
            logger.info(f"No optimization needed for {strategy}")
            return None
        
        logger.info(f"Starting optimization for {strategy}")
        
        # Get parameter bounds and steps
        bounds = self.parameter_bounds.get(strategy, {})
        steps = self.parameter_steps.get(strategy, {})
        
        if not bounds:
            logger.warning(f"No parameter bounds defined for {strategy}")
            return None
        
        # Generate parameter variations
        variations = self._generate_parameter_variations(
            current_parameters, bounds, steps
        )
        
        # Evaluate variations using historical data
        best_variation = None
        best_metrics = current_metrics
        best_score = self._calculate_optimization_score(current_metrics)
        
        for variation in variations:
            # Simulate performance with new parameters
            simulated_metrics = self._simulate_performance(strategy, variation)
            score = self._calculate_optimization_score(simulated_metrics)
            
            if score > best_score:
                best_score = score
                best_variation = variation
                best_metrics = simulated_metrics
        
        if best_variation is None:
            logger.info(f"No better parameters found for {strategy}")

        # Calculate improvement
        improvement = {}
        for metric in current_metrics:
            if metric in best_metrics:
                improvement[metric] = best_metrics[metric] - current_metrics[metric]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            strategy, current_parameters, best_variation, improvement
        )
        
        # Create optimization result
        result = OptimizationResult(
            strategy=strategy,
            timestamp=datetime.now(),
            old_parameters=current_parameters,
            new_parameters=best_variation,
            old_metrics=current_metrics,
            new_metrics=best_metrics,
            improvement=improvement,
            confidence=self._calculate_confidence(improvement),
            recommendations=recommendations
        )
        
        self.optimization_history.append(result)
        self._save_history()
        
        logger.info(f"Optimization completed for {strategy}: {improvement}")
        
        return result
    
    def _generate_parameter_variations(self, 
                                     current_params: Dict[str, Any],
                                     bounds: Dict[str, Tuple[float, float]],
                                     steps: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate parameter variations for optimization.
        
        Args:
            current_params: Current parameters
            bounds: Parameter bounds
            steps: Parameter step sizes
            
        Returns:
            List of parameter variations
        """
        variations = []
        
        for param, (min_val, max_val) in bounds.items():
            if param not in current_params:
                continue
            
            current_val = current_params[param]
            step = steps.get(param, (max_val - min_val) / 10)
            
            # Generate variations around current value
            for multiplier in [-2, -1, 0.5, 1.5, 2]:
                new_val = current_val * multiplier
                if min_val <= new_val <= max_val:
                    variation = current_params.copy()
                    variation[param] = new_val
                    variations.append(variation)
        
        return variations[:self.max_parameter_changes * 2]  # Limit variations
    
    def _simulate_performance(self, 
                             strategy: str,
                             parameters: Dict[str, Any]) -> Dict[str, float]:
        """Simulate performance with given parameters.
        
        Args:
            strategy: Strategy name
            parameters: Parameters to test
            
        Returns:
            Simulated performance metrics
        """
        # Get historical performance data
        historical_records = [
            record for record in self.performance_history
            if record['strategy'] == strategy
        ]
        
        if not historical_records:
            # Return default metrics if no history
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.5,
                'profit_factor': 1.0
            }
        
        # Use weighted average of historical performance
        # with adjustment based on parameter similarity
        total_weight = 0
        weighted_metrics = {}
        
        for record in historical_records[-10:]:  # Use last 10 records
            similarity = self._calculate_parameter_similarity(
                parameters, record['parameters']
            )
            weight = similarity
            
            for metric, value in record['metrics'].items():
                if metric not in weighted_metrics:
                    weighted_metrics[metric] = 0
                weighted_metrics[metric] += value * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for metric in weighted_metrics:
                weighted_metrics[metric] /= total_weight
        
        return weighted_metrics
    
    def _calculate_parameter_similarity(self, 
                                      params1: Dict[str, Any],
                                      params2: Dict[str, Any]) -> float:
        """Calculate similarity between parameter sets.
        
        Args:
            params1: First parameter set
            params2: Second parameter set
            
        Returns:
            Similarity score (0-1)
        """
        if not params1 or not params2:
            return 0.0
        
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0
        
        similarities = []
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Calculate relative difference
                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        diff = abs(val1 - val2) / max_val
                        similarity = max(0, 1 - diff)
            else:
                # For non-numeric parameters, exact match
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """Calculate optimization score from metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Optimization score
        """
        # Weighted combination of metrics
        weights = {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': -0.2,  # Negative weight (lower is better)
            'win_rate': 0.15,
            'profit_factor': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.0)
            if metric == 'max_drawdown':
                # Convert drawdown to positive score (lower drawdown = higher score)
                score += weight * (1 - abs(value))
            else:
                score += weight * value
        
        return score
    
    def _calculate_confidence(self, improvement: Dict[str, float]) -> float:
        """Calculate confidence in optimization result.
        
        Args:
            improvement: Performance improvements
            
        Returns:
            Confidence score (0-1)
        """
        if not improvement:
            return 0.0
        
        # Calculate confidence based on magnitude and consistency of improvements
        positive_improvements = [imp for imp in improvement.values() if imp > 0]
        negative_improvements = [imp for imp in improvement.values() if imp < 0]
        
        if not positive_improvements:
            return 0.0
        
        # Base confidence on ratio of positive improvements
        positive_ratio = len(positive_improvements) / len(improvement)
        
        # Adjust for magnitude of improvements
        avg_positive = np.mean(positive_improvements) if positive_improvements else 0
        avg_negative = np.mean(negative_improvements) if negative_improvements else 0
        
        magnitude_factor = min(1.0, (avg_positive - avg_negative) / 0.1)  # Normalize to 10%
        
        confidence = positive_ratio * 0.7 + magnitude_factor * 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_recommendations(self, 
                                 strategy: str,
                                 old_params: Dict[str, Any],
                                 new_params: Dict[str, Any],
                                 improvement: Dict[str, float]) -> List[str]:
        """Generate recommendations based on optimization.
        
        Args:
            strategy: Strategy name
            old_params: Old parameters
            new_params: New parameters
            improvement: Performance improvements
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Parameter change recommendations
        for param, new_val in new_params.items():
            if param in old_params and old_params[param] != new_val:
                change_pct = ((new_val - old_params[param]) / old_params[param]) * 100
                recommendations.append(
                    f"Adjust {param} from {old_params[param]:.2f} to {new_val:.2f} "
                    f"({change_pct:+.1f}%)"
                )
        
        # Performance improvement recommendations
        if improvement.get('sharpe_ratio', 0) > 0.1:
            recommendations.append("Significant improvement in risk-adjusted returns")
        
        if improvement.get('total_return', 0) > 0.05:
            recommendations.append("Notable increase in total returns")
        
        if improvement.get('max_drawdown', 0) < -0.02:
            recommendations.append("Reduced maximum drawdown")
        
        if improvement.get('win_rate', 0) > 0.05:
            recommendations.append("Improved win rate")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Monitor performance closely with new parameters")
        
        return recommendations
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities.
        
        Returns:
            Dictionary with optimization summary
        """
        if not self.optimization_history:
            return {
                'total_optimizations': 0,
                'successful_optimizations': 0,
                'average_improvement': {},
                'recent_optimizations': []
            }
        
        successful_optimizations = [
            result for result in self.optimization_history
            if result.confidence > 0.5
        ]
        
        # Calculate average improvements
        avg_improvement = {}
        for result in successful_optimizations:
            for metric, improvement in result.improvement.items():
                if metric not in avg_improvement:
                    avg_improvement[metric] = []
                avg_improvement[metric].append(improvement)
        
        for metric in avg_improvement:
            avg_improvement[metric] = np.mean(avg_improvement[metric])
        
        return {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': len(successful_optimizations),
            'success_rate': len(successful_optimizations) / len(self.optimization_history),
            'average_improvement': avg_improvement,
            'recent_optimizations': [
                {
                    'strategy': result.strategy,
                    'timestamp': result.timestamp.isoformat(),
                    'confidence': result.confidence,
                    'improvement': result.improvement
                }
                for result in self.optimization_history[-5:]  # Last 5 optimizations
            ]
        }

# Global optimizer instance
self_tuning_optimizer = SelfTuningOptimizer()

def get_self_tuning_optimizer() -> SelfTuningOptimizer:
    """Get the global self-tuning optimizer instance."""
    return self_tuning_optimizer