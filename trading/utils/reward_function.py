"""
RewardFunction: Multi-objective reward calculation for model and strategy evaluation.
Optimizes for return, Sharpe, and consistency (win rate over drawdown).
"""

from typing import Dict, Any, Optional, List
import numpy as np

class RewardFunction:
    """
    Multi-objective reward function for evaluating models and strategies.
    Supports weighted aggregation of return, Sharpe, and consistency (win rate/drawdown).
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Dict of weights for each objective (return, sharpe, consistency)
        """
        self.weights = weights or {
            'return': 0.4,
            'sharpe': 0.4,
            'consistency': 0.2
        }

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def compute(self, metrics: Dict[str, Any]) -> float:
        """
        Compute the overall reward score from metrics.
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
        Returns:
            Aggregated reward score (float)
        """
        objectives = self.compute_objectives(metrics)
        return {'success': True, 'result': self.aggregate(objectives), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def compute_objectives(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute individual objective scores (normalized where possible).
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
        Returns:
            Dict of objective scores
        """
        total_return = metrics.get('total_return', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        max_drawdown = abs(metrics.get('max_drawdown', 1e-6)) or 1e-6  # Avoid div by zero
        # Consistency: win rate over drawdown (higher is better)
        consistency = win_rate / max_drawdown if max_drawdown > 0 else 0.0
        return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            'return': total_return,
            'sharpe': sharpe,
            'consistency': consistency
        }

    def aggregate(self, objectives: Dict[str, float]) -> float:
        """
        Aggregate objectives into a single reward score using weights.
        Args:
            objectives: Dict of objective scores
        Returns:
            Weighted sum (float)
        """
        return {'success': True, 'result': sum(self.weights.get(k, 0.0) * objectives.get(k, 0.0) for k in self.weights), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def multi_objective_vector(self, metrics: Dict[str, Any]) -> List[float]:
        """
        Return the vector of objectives for multi-objective optimization algorithms.
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
        Returns:
            List of objective values [return, sharpe, consistency]
        """
        obj = self.compute_objectives(metrics)
        return {'success': True, 'result': [obj['return'], obj['sharpe'], obj['consistency']], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set new weights for the objectives.
        Args:
            weights: Dict of weights for each objective
        """
        self.weights = weights.copy() 
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}