"""
Strategy Registry Module

This module handles strategy-related functionality and registry management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """Registry for trading strategies and configurations."""
    
    def __init__(self):
        """Initialize the strategy registry."""
        self.strategies = {}
        self.strategy_configs = {}
        self.performance_history = {}
        
    def register_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> bool:
        """Register a trading strategy.
        
        Args:
            strategy_name: Name of the strategy
            strategy_config: Strategy configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.strategies[strategy_name] = {
                'config': strategy_config,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            logger.info(f"Registered strategy: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_name}: {e}")
            return False
    
    def get_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get a strategy configuration.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dict: Strategy configuration or None if not found
        """
        return self.strategies.get(strategy_name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies.
        
        Returns:
            List: Names of registered strategies
        """
        return list(self.strategies.keys())
    
    def update_performance(self, strategy_name: str, performance_metrics: Dict[str, Any]) -> bool:
        """Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Performance metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            self.performance_history[strategy_name].append({
                'metrics': performance_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Updated performance for strategy: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update performance for strategy {strategy_name}: {e}")
            return False
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Optional[str]:
        """Get the best performing strategy based on a metric.
        
        Args:
            metric: Performance metric to use
            
        Returns:
            str: Name of best strategy or None
        """
        best_strategy = None
        best_score = float('-inf')
        
        for strategy_name, history in self.performance_history.items():
            if history:
                latest_metrics = history[-1]['metrics']
                score = latest_metrics.get(metric, float('-inf'))
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        return best_strategy
    
    def get_strategy_parameters(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get parameters for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dict: Strategy parameters or None
        """
        strategy = self.get_strategy(strategy_name)
        if strategy:
            return strategy['config'].get('parameters', {})
        return None

# Global instance
strategy_registry = StrategyRegistry() 