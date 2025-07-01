"""Performance monitor agent for tracking model and strategy performance."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading.base_agent import BaseMetaAgent
from trading.memory.performance_memory import PerformanceMemory
from trading.models import BaseModel
from trading.strategies import StrategyManager

class PerformanceMonitorAgent(BaseMetaAgent):
    """Agent for monitoring model and strategy performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the performance monitor agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("performance_monitor", config)
        
        # Initialize components
        self.performance_memory = PerformanceMemory()
        self.strategy_manager = StrategyManager()
        
        # Performance thresholds
        self.thresholds = {
            "accuracy": 0.6,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.2
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def run(self) -> Dict[str, Any]:
        """Run performance monitoring.
        
        Returns:
            Dict containing performance analysis results
        """
        results = {
            "models": self._analyze_models(),
            "strategies": self._analyze_strategies(),
            "suggested_actions": []
        }
        
        # Generate actions
        actions = self._generate_actions(results)
        results["suggested_actions"] = actions
        
        # Log results
        self.log_action("Performance monitoring completed", results)
        
        return results
    
    def _analyze_models(self) -> Dict[str, Any]:
        """Analyze model performance.
        
        Returns:
            Dict containing model analysis results
        """
        results = {}
        
        # Get model metrics
        for model_name in self._get_model_names():
            metrics = self.performance_memory.get_metrics(model_name)
            if metrics:
                results[model_name] = {
                    "accuracy": metrics.get("accuracy"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "last_update": metrics.get("timestamp")
                }
        
        return results
    
    def _analyze_strategies(self) -> Dict[str, Any]:
        """Analyze strategy performance.
        
        Returns:
            Dict containing strategy analysis results
        """
        results = {}
        
        # Get strategy metrics
        for strategy_name in self.strategy_manager.get_strategy_names():
            metrics = self.performance_memory.get_metrics(strategy_name)
            if metrics:
                results[strategy_name] = {
                    "accuracy": metrics.get("accuracy"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "last_update": metrics.get("timestamp")
                }
        
        return results
    
    def _generate_actions(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions based on performance analysis.
        
        Args:
            results: Results from performance analysis
            
        Returns:
            List of suggested actions
        """
        actions = []
        
        # Check model performance
        for model_name, metrics in results["models"].items():
            if metrics["accuracy"] < self.thresholds["accuracy"]:
                actions.append({
                    "type": "retrain",
                    "target": model_name,
                    "description": "Low accuracy",
                    "suggestion": "Retrain model with recent data"
                })
            
            if metrics["sharpe_ratio"] < self.thresholds["sharpe_ratio"]:
                actions.append({
                    "type": "optimize",
                    "target": model_name,
                    "description": "Low Sharpe ratio",
                    "suggestion": "Optimize model parameters"
                })
        
        # Check strategy performance
        for strategy_name, metrics in results["strategies"].items():
            if metrics["max_drawdown"] > self.thresholds["max_drawdown"]:
                actions.append({
                    "type": "adjust",
                    "target": strategy_name,
                    "description": "High drawdown",
                    "suggestion": "Adjust risk parameters"
                })
        
        return actions
    
    def _get_model_names(self) -> List[str]:
        """Get list of model names.
        
        Returns:
            List of model names
        """
        model_dir = Path("trading/models")
        return [f.stem for f in model_dir.glob("*.py") if f.stem != "__init__"]
    
    def _is_degrading(self, metrics: List[Dict[str, Any]], key: str) -> bool:
        """Check if performance is degrading.
        
        Args:
            metrics: List of historical metrics
            key: Metric key to check
            
        Returns:
            True if performance is degrading
        """
        if len(metrics) < 2:
            return False
        
        # Calculate trend
        values = [m[key] for m in metrics]
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        return trend < 0 