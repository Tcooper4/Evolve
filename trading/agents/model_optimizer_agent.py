"""
Model Optimizer Agent for Trading System

This agent optimizes model performance through various techniques including
hyperparameter tuning, feature selection, ensemble methods, and model architecture optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from enum import Enum
import uuid

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.memory.agent_memory import AgentMemory
from trading.models.model_registry import ModelRegistry
from trading.optimization.strategy_optimizer import BayesianOptimization, GridSearch

# Simple GeneticOptimizer stub to avoid import issues
class GeneticOptimizer:
    """Simple genetic optimizer stub."""
    def __init__(self, *args, **kwargs):
        pass
    
    def optimize(self, *args, **kwargs):
        return None

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Optimization types."""
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_SELECTION = "feature_selection"
    ENSEMBLE_OPTIMIZATION = "ensemble_optimization"
    ARCHITECTURE_OPTIMIZATION = "architecture_optimization"
    WEIGHT_OPTIMIZATION = "weight_optimization"

class OptimizationStatus(Enum):
    """Optimization status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationResult:
    """Result of model optimization."""
    optimization_id: str
    model_id: str
    optimization_type: OptimizationType
    start_timestamp: datetime
    end_timestamp: datetime
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement: Dict[str, float]
    optimization_params: Dict[str, Any]
    status: OptimizationStatus
    error_message: Optional[str] = None

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    optimization_type: OptimizationType
    objective_metric: str
    max_iterations: int
    timeout_seconds: int
    early_stopping_patience: int
    validation_split: float
    random_state: int
    custom_params: Dict[str, Any]

class ModelOptimizerAgent(BaseAgent):
    """
    Model Optimizer Agent that:
    - Optimizes model hyperparameters using various algorithms
    - Performs feature selection and engineering optimization
    - Optimizes ensemble weights and configurations
    - Tracks optimization history and improvements
    - Provides optimization recommendations and insights
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ModelOptimizerAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        
        self.config_dict = config.custom_config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.model_registry = ModelRegistry()
        
        # Optimization engines
        self.bayesian_optimizer = BayesianOptimization()
        self.genetic_optimizer = GeneticOptimizer()
        
        # Configuration
        self.default_timeout = self.config_dict.get('default_timeout', 3600)
        self.max_concurrent_optimizations = self.config_dict.get('max_concurrent_optimizations', 3)
        self.improvement_threshold = self.config_dict.get('improvement_threshold', 0.05)
        
        # Storage
        self.optimization_history: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationConfig] = {}
        self.current_optimization_id = None
        
        # Load existing data
        self._load_optimization_history()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model optimization logic.
        Args:
            **kwargs: action, model_id, optimization_config, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'optimize_model')
            
            if action == 'optimize_model':
                model_id = kwargs.get('model_id')
                optimization_config = kwargs.get('optimization_config')
                
                if not model_id or not optimization_config:
                    return AgentResult(success=False, error_message="Missing model_id or optimization_config")
                
                result = await self.optimize_model(model_id, optimization_config)
                return AgentResult(success=True, data={
                    "optimization_result": result.__dict__,
                    "improvement": result.improvement,
                    "status": result.status.value
                })
                
            elif action == 'get_optimization_history':
                model_id = kwargs.get('model_id')
                history = self.get_optimization_history(model_id)
                return AgentResult(success=True, data={
                    "optimization_history": [opt.__dict__ for opt in history]
                })
                
            elif action == 'get_optimization_status':
                optimization_id = kwargs.get('optimization_id')
                if not optimization_id:
                    return AgentResult(success=False, error_message="Missing optimization_id")
                
                status = self.get_optimization_status(optimization_id)
                return AgentResult(success=True, data={"optimization_status": status})
                
            elif action == 'cancel_optimization':
                optimization_id = kwargs.get('optimization_id')
                if not optimization_id:
                    return AgentResult(success=False, error_message="Missing optimization_id")
                
                cancelled = self.cancel_optimization(optimization_id)
                return AgentResult(success=True, data={"cancelled": cancelled})
                
            elif action == 'get_optimization_recommendations':
                model_id = kwargs.get('model_id')
                recommendations = self.get_optimization_recommendations(model_id)
                return AgentResult(success=True, data={"recommendations": recommendations})
                
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)
    
    async def optimize_model(self, model_id: str, optimization_config: Dict[str, Any]) -> OptimizationResult:
        """
        Optimize a model based on the configuration.
        
        Args:
            model_id: ID of the model to optimize
            optimization_config: Optimization configuration
            
        Returns:
            Optimization result
        """
        try:
            optimization_id = f"opt_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_optimization_id = optimization_id
            
            self.logger.info(f"Starting optimization for model {model_id}")
            
            # Create optimization config
            config = OptimizationConfig(
                optimization_type=OptimizationType(optimization_config.get('type', 'hyperparameter_tuning')),
                objective_metric=optimization_config.get('objective_metric', 'sharpe_ratio'),
                max_iterations=optimization_config.get('max_iterations', 100),
                timeout_seconds=optimization_config.get('timeout_seconds', self.default_timeout),
                early_stopping_patience=optimization_config.get('early_stopping_patience', 10),
                validation_split=optimization_config.get('validation_split', 0.2),
                random_state=optimization_config.get('random_state', 42),
                custom_params=optimization_config.get('custom_params', {})
            )
            
            # Get original model metrics
            original_metrics = self._get_model_metrics(model_id)
            
            # Start optimization
            start_time = datetime.now()
            self.active_optimizations[optimization_id] = config
            
            try:
                # Run optimization based on type
                if config.optimization_type == OptimizationType.HYPERPARAMETER_TUNING:
                    optimized_metrics = await self._optimize_hyperparameters(model_id, config)
                elif config.optimization_type == OptimizationType.FEATURE_SELECTION:
                    optimized_metrics = await self._optimize_features(model_id, config)
                elif config.optimization_type == OptimizationType.ENSEMBLE_OPTIMIZATION:
                    optimized_metrics = await self._optimize_ensemble(model_id, config)
                elif config.optimization_type == OptimizationType.ARCHITECTURE_OPTIMIZATION:
                    optimized_metrics = await self._optimize_architecture(model_id, config)
                else:
                    raise ValueError(f"Unsupported optimization type: {config.optimization_type}")
                
                end_time = datetime.now()
                
                # Calculate improvement
                improvement = self._calculate_improvement(original_metrics, optimized_metrics)
                
                # Create result
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    model_id=model_id,
                    optimization_type=config.optimization_type,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                    original_metrics=original_metrics,
                    optimized_metrics=optimized_metrics,
                    improvement=improvement,
                    optimization_params=config.custom_params,
                    status=OptimizationStatus.COMPLETED
                )
                
                # Store result
                self.optimization_history.append(result)
                self._store_optimization_result(result)
                
                self.logger.info(f"Completed optimization for model {model_id}: {improvement}")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    model_id=model_id,
                    optimization_type=config.optimization_type,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                    original_metrics=original_metrics,
                    optimized_metrics={},
                    improvement={},
                    optimization_params=config.custom_params,
                    status=OptimizationStatus.FAILED,
                    error_message=str(e)
                )
                
                self.optimization_history.append(result)
                self._store_optimization_result(result)
                
                raise
                
            finally:
                # Clean up
                if optimization_id in self.active_optimizations:
                    del self.active_optimizations[optimization_id]
            
        except Exception as e:
            self.logger.error(f"Error optimizing model {model_id}: {str(e)}")
            raise
    
    async def _optimize_hyperparameters(self, model_id: str, config: OptimizationConfig) -> Dict[str, float]:
        """Optimize model hyperparameters."""
        try:
            # Get model
            model = self.model_registry.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Define parameter space
            param_space = self._get_hyperparameter_space(model_id, config)
            
            # Define objective function
            def objective(params):
                try:
                    # Update model hyperparameters
                    model.update_hyperparameters(params)
                    
                    # Evaluate model
                    metrics = self._evaluate_model(model_id)
                    return metrics.get(config.objective_metric, 0.0)
                    
                except Exception as e:
                    self.logger.warning(f"Objective function failed: {e}")
                    return 0.0
            
            # Run optimization
            if len(param_space) <= 6:
                # Use Bayesian optimization for smaller spaces
                best_params = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.bayesian_optimizer.optimize,
                    objective,
                    param_space,
                    config.max_iterations
                )
            else:
                # Use genetic optimization for larger spaces
                best_params = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.genetic_optimizer.optimize,
                    objective,
                    param_space,
                    config.max_iterations
                )
            
            # Apply best parameters
            if best_params:
                model.update_hyperparameters(best_params)
            
            # Return final metrics
            return self._evaluate_model(model_id)
            
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise
    
    async def _optimize_features(self, model_id: str, config: OptimizationConfig) -> Dict[str, float]:
        """Optimize feature selection."""
        try:
            # Get model and data
            model = self.model_registry.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Get feature importance or correlation data
            feature_data = self._get_feature_data(model_id)
            
            # Define feature selection objective
            def objective(feature_subset):
                try:
                    # Update model with selected features
                    model.update_features(feature_subset)
                    
                    # Evaluate model
                    metrics = self._evaluate_model(model_id)
                    return metrics.get(config.objective_metric, 0.0)
                    
                except Exception as e:
                    self.logger.warning(f"Feature selection objective failed: {e}")
                    return 0.0
            
            # Run feature selection optimization
            best_features = await asyncio.get_event_loop().run_in_executor(
                None,
                self.genetic_optimizer.optimize_feature_selection,
                objective,
                feature_data,
                config.max_iterations
            )
            
            # Apply best features
            if best_features:
                model.update_features(best_features)
            
            # Return final metrics
            return self._evaluate_model(model_id)
            
        except Exception as e:
            self.logger.error(f"Error optimizing features: {str(e)}")
            raise
    
    async def _optimize_ensemble(self, model_id: str, config: OptimizationConfig) -> Dict[str, float]:
        """Optimize ensemble configuration."""
        try:
            # Get ensemble model
            model = self.model_registry.get_model(model_id)
            if not model or not hasattr(model, 'ensemble_weights'):
                raise ValueError(f"Model {model_id} is not an ensemble model")
            
            # Define ensemble optimization objective
            def objective(weights):
                try:
                    # Update ensemble weights
                    model.update_ensemble_weights(weights)
                    
                    # Evaluate ensemble
                    metrics = self._evaluate_model(model_id)
                    return metrics.get(config.objective_metric, 0.0)
                    
                except Exception as e:
                    self.logger.warning(f"Ensemble optimization objective failed: {e}")
                    return 0.0
            
            # Run weight optimization
            best_weights = await asyncio.get_event_loop().run_in_executor(
                None,
                self.bayesian_optimizer.optimize_weights,
                objective,
                len(model.ensemble_weights),
                config.max_iterations
            )
            
            # Apply best weights
            if best_weights:
                model.update_ensemble_weights(best_weights)
            
            # Return final metrics
            return self._evaluate_model(model_id)
            
        except Exception as e:
            self.logger.error(f"Error optimizing ensemble: {str(e)}")
            raise
    
    async def _optimize_architecture(self, model_id: str, config: OptimizationConfig) -> Dict[str, float]:
        """Optimize model architecture."""
        try:
            # Get model
            model = self.model_registry.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Define architecture search space
            arch_space = self._get_architecture_space(model_id, config)
            
            # Define architecture optimization objective
            def objective(architecture):
                try:
                    # Create new model with architecture
                    new_model = self._create_model_with_architecture(model_id, architecture)
                    
                    # Evaluate new model
                    metrics = self._evaluate_model(new_model.model_id)
                    return metrics.get(config.objective_metric, 0.0)
                    
                except Exception as e:
                    self.logger.warning(f"Architecture optimization objective failed: {e}")
                    return 0.0
            
            # Run architecture optimization
            best_architecture = await asyncio.get_event_loop().run_in_executor(
                None,
                self.genetic_optimizer.optimize_architecture,
                objective,
                arch_space,
                config.max_iterations
            )
            
            # Apply best architecture
            if best_architecture:
                self._apply_architecture(model_id, best_architecture)
            
            # Return final metrics
            return self._evaluate_model(model_id)
            
        except Exception as e:
            self.logger.error(f"Error optimizing architecture: {str(e)}")
            raise
    
    def _get_model_metrics(self, model_id: str) -> Dict[str, float]:
        """Get current model metrics."""
        try:
            # This would typically get metrics from the model evaluator
            # For now, return default metrics
            return {
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.1,
                'win_rate': 0.6,
                'profit_factor': 1.2
            }
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {}
    
    def _evaluate_model(self, model_id: str) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        try:
            # This would typically call the model evaluator
            # For now, return simulated metrics
            return {
                'sharpe_ratio': np.random.uniform(0.3, 1.5),
                'max_drawdown': np.random.uniform(0.05, 0.3),
                'win_rate': np.random.uniform(0.4, 0.8),
                'profit_factor': np.random.uniform(0.8, 2.0)
            }
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def _calculate_improvement(self, original_metrics: Dict[str, float], 
                             optimized_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement between original and optimized metrics."""
        try:
            improvement = {}
            
            for metric in original_metrics:
                if metric in optimized_metrics:
                    original_val = original_metrics[metric]
                    optimized_val = optimized_metrics[metric]
                    
                    if original_val != 0:
                        improvement[metric] = (optimized_val - original_val) / abs(original_val)
                    else:
                        improvement[metric] = 0.0
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement: {str(e)}")
            return {}
    
    def _get_hyperparameter_space(self, model_id: str, config: OptimizationConfig) -> Dict[str, Any]:
        """Get hyperparameter search space for a model."""
        # This would be model-specific
        return {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64, 128],
            'epochs': [50, 100, 200]
        }
    
    def _get_feature_data(self, model_id: str) -> Dict[str, Any]:
        """Get feature data for optimization."""
        # This would get feature importance or correlation data
        return {
            'features': ['feature1', 'feature2', 'feature3'],
            'importance': [0.8, 0.6, 0.4]
        }
    
    def _get_architecture_space(self, model_id: str, config: OptimizationConfig) -> Dict[str, Any]:
        """Get architecture search space."""
        return {
            'layers': [1, 2, 3, 4],
            'units': [64, 128, 256, 512],
            'dropout': [0.1, 0.2, 0.3, 0.4]
        }
    
    def _create_model_with_architecture(self, model_id: str, architecture: Dict[str, Any]):
        """Create a new model with given architecture."""
        # This would create a new model instance
        pass
    
    def _apply_architecture(self, model_id: str, architecture: Dict[str, Any]):
        """Apply architecture changes to existing model."""
        # This would modify the existing model
        pass
    
    def get_optimization_history(self, model_id: Optional[str] = None) -> List[OptimizationResult]:
        """Get optimization history for a model or all models."""
        if model_id:
            return [opt for opt in self.optimization_history if opt.model_id == model_id]
        return self.optimization_history.copy()
    
    def get_optimization_status(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an optimization."""
        try:
            # Find optimization in history
            for opt in self.optimization_history:
                if opt.optimization_id == optimization_id:
                    return {
                        'optimization_id': optimization_id,
                        'status': opt.status.value,
                        'start_time': opt.start_timestamp.isoformat(),
                        'end_time': opt.end_timestamp.isoformat() if opt.end_timestamp else None,
                        'improvement': opt.improvement,
                        'error_message': opt.error_message
                    }
            
            # Check if it's currently running
            if optimization_id in self.active_optimizations:
                return {
                    'optimization_id': optimization_id,
                    'status': 'running',
                    'start_time': datetime.now().isoformat(),
                    'end_time': None
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting optimization status: {str(e)}")
            return None
    
    def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancel an ongoing optimization."""
        try:
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
                
                # Update status in history
                for opt in self.optimization_history:
                    if opt.optimization_id == optimization_id:
                        opt.status = OptimizationStatus.CANCELLED
                        opt.end_timestamp = datetime.now()
                        break
                
                self.logger.info(f"Cancelled optimization: {optimization_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling optimization: {str(e)}")
            return False
    
    def get_optimization_recommendations(self, model_id: str) -> List[str]:
        """Get optimization recommendations for a model."""
        try:
            recommendations = []
            
            # Get model history
            model_history = [opt for opt in self.optimization_history if opt.model_id == model_id]
            
            if not model_history:
                recommendations.append("No optimization history found - consider hyperparameter tuning")
                return recommendations
            
            # Analyze recent optimizations
            recent_optimizations = sorted(model_history, key=lambda x: x.start_timestamp)[-5:]
            
            # Check for improvement trends
            improvements = [opt.improvement.get('sharpe_ratio', 0) for opt in recent_optimizations]
            
            if improvements and max(improvements) < self.improvement_threshold:
                recommendations.append("Recent optimizations show minimal improvement - consider different approach")
            
            # Check optimization types used
            types_used = set(opt.optimization_type for opt in recent_optimizations)
            
            if OptimizationType.FEATURE_SELECTION not in types_used:
                recommendations.append("Consider feature selection optimization")
            
            if OptimizationType.ENSEMBLE_OPTIMIZATION not in types_used:
                recommendations.append("Consider ensemble optimization if applicable")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {str(e)}")
            return ["Unable to generate recommendations"]
    
    def _store_optimization_result(self, result: OptimizationResult):
        """Store optimization result in memory."""
        try:
            self.memory.store(f'optimization_{result.optimization_id}', {
                'result': result.__dict__,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing optimization result: {str(e)}")
    
    def _load_optimization_history(self):
        """Load optimization history from memory."""
        try:
            # Load recent optimizations
            optimization_data = self.memory.get('optimization_history')
            if optimization_data:
                self.optimization_history = [OptimizationResult(**opt) for opt in optimization_data]
        except Exception as e:
            self.logger.error(f"Error loading optimization history: {str(e)}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'status': 'active',
            'last_update': datetime.now().isoformat(),
            'optimizations_completed': len(self.optimization_history),
            'active_optimizations': len(self.active_optimizations),
            'current_optimization': self.current_optimization_id
        } 