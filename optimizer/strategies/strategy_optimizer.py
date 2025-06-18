"""
Strategy Optimizer.

This module provides a unified interface for optimizing trading strategies using
different optimization methods (Grid, Bayesian, Genetic).
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
import pandas as pd
from ..core.optimizer_factory import OptimizerFactory, BaseOptimizer

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """Strategy optimizer that integrates with strategy switcher."""
    
    def __init__(self, strategy_switcher, memory_logger):
        """Initialize the strategy optimizer.
        
        Args:
            strategy_switcher: Instance of strategy switcher
            memory_logger: Instance of memory logger
        """
        self.strategy_switcher = strategy_switcher
        self.memory_logger = memory_logger
        self.optimizer_factory = OptimizerFactory()
    
    def optimize_strategy(self, strategy_name: str, optimizer_type: str,
                         param_space: Dict, data: Dict,
                         **optimizer_kwargs) -> Dict:
        """Optimize a strategy using the specified optimizer.
        
        Args:
            strategy_name: Name of the strategy to optimize
            optimizer_type: Type of optimizer to use (grid, bayesian, genetic)
            param_space: Dictionary defining the parameter space
            data: Dictionary containing training data
            **optimizer_kwargs: Additional arguments for optimizer initialization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting optimization of {strategy_name} using {optimizer_type}")
        
        # Create optimizer
        optimizer = self.optimizer_factory.create_optimizer(
            optimizer_type, **optimizer_kwargs
        )
        
        # Run optimization
        results = optimizer.optimize(strategy_name, param_space, data)
        
        # Log results
        self._log_optimization_results(strategy_name, optimizer_type, results)
        
        return results
    
    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizer types.
        
        Returns:
            List of optimizer type names
        """
        return list(self.optimizer_factory.get_available_optimizers().keys())
    
    def get_strategy_param_space(self, strategy_name: str) -> Dict:
        """Get the parameter space for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary defining the parameter space
        """
        # Get strategy class
        strategy_class = self.strategy_switcher.get_strategy_class(strategy_name)
        
        # Get parameter definitions
        param_space = {}
        for name, param in strategy_class.get_parameters().items():
            if param.get('optimizable', False):
                param_space[name] = (
                    param.get('min', 0),
                    param.get('max', 1)
                )
        
        return param_space
    
    def _log_optimization_results(self, strategy_name: str, optimizer_type: str,
                                results: Dict) -> None:
        """Log optimization results to memory.
        
        Args:
            strategy_name: Name of the optimized strategy
            optimizer_type: Type of optimizer used
            results: Dictionary containing optimization results
        """
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'strategy': strategy_name,
            'optimizer': optimizer_type,
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'all_results': results['all_results']
        }
        
        self.memory_logger.log_optimization(log_entry)
    
    def save_optimization_results(self, results: Dict, output_path: str) -> None:
        """Save optimization results to a file.
        
        Args:
            results: Dictionary containing optimization results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved optimization results to {output_path}")
    
    def load_optimization_results(self, input_path: str) -> Dict:
        """Load optimization results from a file.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Dictionary containing optimization results
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded optimization results from {input_path}")
        return results 