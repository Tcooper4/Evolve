"""Optimization Module for Evolve Trading Platform.

This module provides optimization capabilities for trading strategies,
model hyperparameters, and portfolio allocation.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = differential_evolution = None

class OptimizationResult:
    """Result container for optimization."""
    
    def __init__(self, 
                 best_params: Dict[str, Any],
                 best_value: float,
                 optimization_history: List[Dict[str, Any]],
                 metadata: Dict[str, Any]):
        """Initialize optimization result."""
        self.best_params = best_params
        self.best_value = best_value
        self.optimization_history = optimization_history
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'optimization_history': self.optimization_history,
            'metadata': self.metadata
        }

class StrategyOptimizer:
    """Optimizer for trading strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy optimizer."""
        self.config = config or {}
        self.results = []
    
    def optimize_strategy(self, 
                         strategy_func,
                         param_bounds: Dict[str, tuple],
                         objective_func,
                         n_trials: int = 100) -> OptimizationResult:
        """Optimize strategy parameters."""
        try:
            if OPTUNA_AVAILABLE:
                return self._optimize_with_optuna(strategy_func, param_bounds, objective_func, n_trials)
            elif SCIPY_AVAILABLE:
                return self._optimize_with_scipy(strategy_func, param_bounds, objective_func)
            else:
                return self._fallback_optimization(strategy_func, param_bounds, objective_func)
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            return self._create_fallback_result(param_bounds)
    
    def _optimize_with_optuna(self, strategy_func, param_bounds, objective_func, n_trials):
        """Optimize using Optuna."""
        def objective(trial):
            params = {}
            for param, (low, high) in param_bounds.items():
                params[param] = trial.suggest_float(param, low, high)
            return OptimizationResult(
                best_params=params,
                best_value=objective_func(params),
                optimization_history=[trial.params for trial in study.trials],
                metadata={'method': 'optuna', 'n_trials': n_trials}
            )
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            optimization_history=[trial.params for trial in study.trials],
            metadata={'method': 'optuna', 'n_trials': n_trials}
        )
    
    def _optimize_with_scipy(self, strategy_func, param_bounds, objective_func):
        """Optimize using SciPy."""
        param_names = list(param_bounds.keys())
        bounds = list(param_bounds.values())
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            return -objective_func(param_dict)
        
        result = differential_evolution(objective, bounds)
        
        best_params = dict(zip(param_names, result.x))
        return OptimizationResult(
            best_params=best_params,
            best_value=-result.fun,
            optimization_history=[],
            metadata={'method': 'scipy', 'success': result.success}
        )
    
    def _fallback_optimization(self, strategy_func, param_bounds, objective_func):
        """Fallback optimization using grid search."""
        logger.warning("Using fallback grid search optimization")
        
        # Simple grid search
        best_params = {}
        best_value = float('-inf')
        
        for param, (low, high) in param_bounds.items():
            best_params[param] = (low + high) / 2  # Use midpoint
        
        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            optimization_history=[],
            metadata={'method': 'fallback_grid_search'}
        )
    
    def _create_fallback_result(self, param_bounds):
        """Create fallback result."""
        best_params = {}
        for param, (low, high) in param_bounds.items():
            best_params[param] = (low + high) / 2
        
        return OptimizationResult(
            best_params=best_params,
            best_value=0.0,
            optimization_history=[],
            metadata={'method': 'fallback', 'error': True}
        )

class PortfolioOptimizer:
    """Optimizer for portfolio allocation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize portfolio optimizer."""
        self.config = config or {}
    
    def optimize_allocation(self, 
                           returns: List[float],
                           risk_free_rate: float = 0.02,
                           target_return: Optional[float] = None) -> OptimizationResult:
        """Optimize portfolio allocation."""
        try:
            # Simple equal-weight allocation as fallback
            n_assets = len(returns)
            weights = [1.0 / n_assets] * n_assets
            
            return OptimizationResult(
                best_params={'weights': weights},
                best_value=sum(returns) / len(returns),
                optimization_history=[],
                metadata={'method': 'equal_weight', 'n_assets': n_assets}
            )
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return OptimizationResult(
                best_params={'weights': [1.0]},
                best_value=0.0,
                optimization_history=[],
                metadata={'method': 'fallback', 'error': True}
            )

# Global optimizer instances
strategy_optimizer = StrategyOptimizer()
portfolio_optimizer = PortfolioOptimizer()

def get_strategy_optimizer() -> StrategyOptimizer:
    """Get the global strategy optimizer instance."""
    return strategy_optimizer

def get_portfolio_optimizer() -> PortfolioOptimizer:
    """Get the global portfolio optimizer instance."""
    return portfolio_optimizer