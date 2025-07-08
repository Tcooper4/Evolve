# -*- coding: utf-8 -*-
"""
Meta-Learning Feedback Agent for continuous model improvement and hyperparameter optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from trading.models.model_registry import ModelRegistry
from trading.models.base_model import BaseModel
from trading.optimization.bayesian_optimizer import BayesianOptimizer
from trading.optimization.core_optimizer import GeneticOptimizer
from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.memory.agent_memory import AgentMemory
from trading.agents.model_selector_agent import ModelSelectorAgent
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

@dataclass
class ModelFeedback:
    """Model performance feedback data."""
    model_name: str
    timestamp: datetime
    forecast_horizon: int
    actual_return: float
    predicted_return: float
    mse: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    market_regime: str
    hyperparameters: Dict[str, Any]
    confidence_score: float

@dataclass
class HyperparameterUpdate:
    """Hyperparameter update recommendation."""
    model_name: str
    parameter_name: str
    old_value: Any
    new_value: Any
    improvement_expected: float
    confidence: float
    reason: str

class MetaLearningFeedbackAgent(BaseAgent):
    """
    Agent responsible for:
    - Monitoring model performance after each trade
    - Automatically retuning hyperparameters
    - Replacing underperforming models
    - Updating ensemble weights based on performance
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="MetaLearningFeedbackAgent",
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
        self.model_selector = ModelSelectorAgent()
        # Performance tracking
        self.feedback_history: Dict[str, List[ModelFeedback]] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.hyperparameter_history: Dict[str, List[Dict[str, Any]]] = {}
        # Configuration
        self.performance_window = self.config_dict.get('performance_window', 30)
        self.retuning_frequency = self.config_dict.get('retuning_frequency', 'weekly')
        self.ensemble_update_frequency = self.config_dict.get('ensemble_update_frequency', 'weekly')
        self.min_performance_threshold = self.config_dict.get('min_performance_threshold', 0.5)
        self.max_hyperparameter_changes = self.config_dict.get('max_hyperparameter_changes', 5)
        # Optimizers
        self.bayesian_optimizer = BayesianOptimizer()
        self.genetic_optimizer = GeneticOptimizer()
        # Load existing data
        self._load_feedback_history()
        self._load_ensemble_weights()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the meta-learning feedback agent logic.
        Args:
            **kwargs: action, feedback, model_name, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'process_feedback')
            if action == 'process_feedback':
                feedback_data = kwargs.get('feedback')
                if not feedback_data:
                    return AgentResult(success=False, error_message="Missing feedback data")
                feedback = ModelFeedback(**feedback_data)
                await self.process_model_feedback(feedback)
                return AgentResult(success=True, data={"message": f"Processed feedback for {feedback.model_name}"})
            elif action == 'get_ensemble_weights':
                weights = self.get_ensemble_weights()
                return AgentResult(success=True, data={"ensemble_weights": weights})
            elif action == 'get_model_performance_summary':
                model_name = kwargs.get('model_name')
                if not model_name:
                    return AgentResult(success=False, error_message="Missing model_name")
                summary = self.get_model_performance_summary(model_name)
                return AgentResult(success=True, data={"performance_summary": summary})
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
        except Exception as e:
            return self.handle_error(e)

    async def process_model_feedback(self, feedback: ModelFeedback):
        """Process feedback from a model prediction."""
        try:
            # Store feedback
            self._store_feedback(feedback)
            
            # Update ensemble weights if needed
            await self._update_ensemble_weights()
            
            # Check if hyperparameter tuning is needed
            if self._should_retune_hyperparameters(feedback.model_name):
                await self._retune_hyperparameters(feedback.model_name)
            
            # Check if model replacement is needed
            if self._should_replace_model(feedback.model_name):
                await self._replace_underperforming_model(feedback.model_name)
            
            # Log feedback
            self.logger.info(f"Processed feedback for {feedback.model_name}: "
                           f"MSE={feedback.mse:.4f}, Sharpe={feedback.sharpe_ratio:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error processing model feedback: {str(e)}")
    
    def _store_feedback(self, feedback: ModelFeedback):
        """Store feedback in history."""
        try:
            if feedback.model_name not in self.feedback_history:
                self.feedback_history[feedback.model_name] = []
            
            self.feedback_history[feedback.model_name].append(feedback)
            
            # Keep only recent feedback
            cutoff_date = datetime.now() - timedelta(days=self.performance_window)
            self.feedback_history[feedback.model_name] = [
                f for f in self.feedback_history[feedback.model_name]
                if f.timestamp > cutoff_date
            ]
            
            # Store in memory
            memory_key = f"model_feedback_{feedback.model_name}"
            self.memory.store(memory_key, {
                'feedback': [f.__dict__ for f in self.feedback_history[feedback.model_name]],
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error storing feedback: {str(e)}")
    
    async def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance."""
        try:
            # Check if it's time to update weights
            last_update = self.memory.get('last_ensemble_update')
            if (last_update and 
                (datetime.now() - last_update['timestamp']).days < 7):
                return
            
            # Calculate performance scores for each model
            performance_scores = {}
            for model_name, feedback_list in self.feedback_history.items():
                if len(feedback_list) < 5:  # Need minimum feedback
                    continue
                
                recent_feedback = feedback_list[-10:]  # Last 10 predictions
                avg_sharpe = np.mean([f.sharpe_ratio for f in recent_feedback])
                avg_mse = np.mean([f.mse for f in recent_feedback])
                avg_win_rate = np.mean([f.win_rate for f in recent_feedback])
                
                # Calculate composite score
                performance_score = (
                    0.4 * max(0, avg_sharpe) +
                    0.3 * (1 - min(1, avg_mse)) +
                    0.3 * avg_win_rate
                )
                
                performance_scores[model_name] = performance_score
            
            if not performance_scores:
                return
            
            # Normalize weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                self.ensemble_weights = {
                    model: score / total_score 
                    for model, score in performance_scores.items()
                }
            else:
                # Equal weights if no positive performance
                num_models = len(performance_scores)
                self.ensemble_weights = {
                    model: 1.0 / num_models 
                    for model in performance_scores.keys()
                }
            
            # Store updated weights
            self.memory.store('ensemble_weights', {
                'weights': self.ensemble_weights,
                'timestamp': datetime.now()
            })
            
            self.memory.store('last_ensemble_update', {
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {str(e)}")
    
    def _should_retune_hyperparameters(self, model_name: str) -> bool:
        """Check if hyperparameter tuning is needed."""
        try:
            if model_name not in self.feedback_history:
                return False
            
            recent_feedback = self.feedback_history[model_name][-10:]
            if len(recent_feedback) < 5:
                return False
            
            # Check if performance is declining
            recent_mse = np.mean([f.mse for f in recent_feedback[-5:]])
            older_mse = np.mean([f.mse for f in recent_feedback[:5]])
            
            # Check if Sharpe ratio is declining
            recent_sharpe = np.mean([f.sharpe_ratio for f in recent_feedback[-5:]])
            older_sharpe = np.mean([f.sharpe_ratio for f in recent_feedback[:5]])
            
            # Trigger retuning if performance is declining significantly
            mse_decline = (recent_mse - older_mse) / max(older_mse, 1e-6)
            sharpe_decline = (older_sharpe - recent_sharpe) / max(abs(older_sharpe), 1e-6)
            
            return mse_decline > 0.2 or sharpe_decline > 0.3
            
        except Exception as e:
            self.logger.error(f"Error checking hyperparameter tuning need: {str(e)}")
            return False
    
    async def _retune_hyperparameters(self, model_name: str):
        """Retune hyperparameters for a model."""
        try:
            self.logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            # Get current model
            model = self.model_registry.get_model(model_name)
            if not model:
                return
            
            # Get recent feedback for optimization
            recent_feedback = self.feedback_history[model_name][-20:]
            if len(recent_feedback) < 10:
                return
            
            # Define optimization objective
            def objective(hyperparams):
                # Simulate performance with new hyperparameters
                # This is a simplified version - in practice, you'd retrain the model
                try:
                    # Update model hyperparameters
                    model.update_hyperparameters(hyperparams)
                    
                    # Calculate expected performance based on historical patterns
                    expected_mse = self._estimate_performance_improvement(
                        model_name, hyperparams, 'mse'
                    )
                    expected_sharpe = self._estimate_performance_improvement(
                        model_name, hyperparams, 'sharpe'
                    )
                    
                    # Return negative score (minimize)
                    return -(0.6 * expected_sharpe + 0.4 * (1 - expected_mse))
                    
                except Exception as e:
                    self.logger.error(f"Error in optimization objective: {str(e)}")
                    return 0.0
            
            # Define hyperparameter space
            param_space = self._get_hyperparameter_space(model_name)
            
            # Run optimization
            with ThreadPoolExecutor() as executor:
                if len(param_space) <= 3:
                    # Use Bayesian optimization for small spaces
                    best_params = await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        self.bayesian_optimizer.optimize,
                        objective, 
                        param_space, 
                        n_trials=20
                    )
                else:
                    # Use genetic optimization for larger spaces
                    best_params = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        self.genetic_optimizer.optimize,
                        objective,
                        param_space,
                        population_size=50,
                        generations=20
                    )
            
            # Apply new hyperparameters
            if best_params:
                old_params = model.get_hyperparameters()
                model.update_hyperparameters(best_params)
                
                # Store hyperparameter update
                self._store_hyperparameter_update(model_name, old_params, best_params)
                
                self.logger.info(f"Updated hyperparameters for {model_name}: {best_params}")
            
        except Exception as e:
            self.logger.error(f"Error retuning hyperparameters: {str(e)}")
    
    def _estimate_performance_improvement(self, 
                                        model_name: str, 
                                        hyperparams: Dict[str, Any],
                                        metric: str) -> float:
        """Estimate performance improvement from hyperparameter changes."""
        try:
            # This is a simplified estimation - in practice, you'd use more sophisticated methods
            # like meta-learning or surrogate models
            
            if model_name not in self.hyperparameter_history:
                return 0.5  # Default expectation
            
            # Find similar hyperparameter configurations
            similar_configs = []
            for hist_params in self.hyperparameter_history[model_name]:
                similarity = self._calculate_hyperparameter_similarity(hyperparams, hist_params)
                if similarity > 0.7:
                    similar_configs.append(hist_params)
            
            if not similar_configs:
                return 0.5
            
            # Estimate based on similar configurations
            if metric == 'mse':
                return np.mean([config.get('mse', 0.5) for config in similar_configs])
            elif metric == 'sharpe':
                return np.mean([config.get('sharpe_ratio', 0.5) for config in similar_configs])
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error estimating performance improvement: {str(e)}")
            return 0.5
    
    def _calculate_hyperparameter_similarity(self, 
                                           params1: Dict[str, Any], 
                                           params2: Dict[str, Any]) -> float:
        """Calculate similarity between two hyperparameter configurations."""
        try:
            common_keys = set(params1.keys()) & set(params2.keys())
            if not common_keys:
                return 0.0
            
            similarities = []
            for key in common_keys:
                val1, val2 = params1[key], params2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                else:
                    # Categorical similarity
                    similarity = 1.0 if val1 == val2 else 0.0
                
                similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating hyperparameter similarity: {str(e)}")
            return 0.0
    
    def _get_hyperparameter_space(self, model_name: str) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for a model."""
        try:
            # Define search spaces for different model types
            if 'lstm' in model_name.lower():
                return {
                    'hidden_size': [64, 128, 256, 512],
                    'num_layers': [1, 2, 3, 4],
                    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'learning_rate': [0.001, 0.01, 0.1]
                }
            elif 'transformer' in model_name.lower():
                return {
                    'num_heads': [4, 8, 16],
                    'num_layers': [2, 4, 6, 8],
                    'd_model': [128, 256, 512],
                    'dropout': [0.1, 0.2, 0.3]
                }
            elif 'xgboost' in model_name.lower():
                return {
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 500],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                # Default space
                return {'learning_rate': [0.001, 0.01, 0.1],
                        'batch_size': [32, 64, 128],
                        'epochs': [50, 100, 200]
                       }
                
        except Exception as e:
            self.logger.error(f"Error getting hyperparameter space: {str(e)}")
            return {}
    
    def _should_replace_model(self, model_name: str) -> bool:
        """Check if a model should be replaced."""
        try:
            if model_name not in self.feedback_history:
                return False
            
            recent_feedback = self.feedback_history[model_name][-20:]
            if len(recent_feedback) < 10:
                return False
            
            # Calculate recent performance
            recent_mse = np.mean([f.mse for f in recent_feedback[-10:]])
            recent_sharpe = np.mean([f.sharpe_ratio for f in recent_feedback[-10:]])
            recent_win_rate = np.mean([f.win_rate for f in recent_feedback[-10:]])
            
            # Check if performance is below thresholds
            if (recent_mse > 0.1 or  # High MSE
                recent_sharpe < -0.5 or  # Negative Sharpe
                recent_win_rate < 0.3):  # Low win rate
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking model replacement: {str(e)}")
            return False
    
    async def _replace_underperforming_model(self, model_name: str):
        """Replace an underperforming model with a better alternative."""
        try:
            self.logger.info(f"Replacing underperforming model: {model_name}")
            
            # Get available models
            available_models = self.model_registry.list_models()
            alternative_models = [m for m in available_models if m != model_name]
            
            if not alternative_models:
                return
            
            # Find best alternative
            best_alternative = None
            best_score = -float('inf')
            
            for alt_model in alternative_models:
                if alt_model in self.feedback_history:
                    recent_feedback = self.feedback_history[alt_model][-10:]
                    if len(recent_feedback) >= 5:
                        avg_sharpe = np.mean([f.sharpe_ratio for f in recent_feedback])
                        avg_mse = np.mean([f.mse for f in recent_feedback])
                        score = avg_sharpe - avg_mse
                        
                        if score > best_score:
                            best_score = score
                            best_alternative = alt_model
            
            if best_alternative:
                # Update ensemble weights to favor the better model
                self.ensemble_weights[best_alternative] = self.ensemble_weights.get(best_alternative, 0.1) + 0.2
                if model_name in self.ensemble_weights:
                    self.ensemble_weights[model_name] = max(0.05, self.ensemble_weights[model_name] - 0.1)
                
                # Normalize weights
                total_weight = sum(self.ensemble_weights.values())
                self.ensemble_weights = {k: v / total_weight for k, v in self.ensemble_weights.items()}
                
                self.logger.info(f"Replaced {model_name} with {best_alternative}")
                self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Error replacing model: {str(e)}")
    
    def _store_hyperparameter_update(self, 
                                   model_name: str, 
                                   old_params: Dict[str, Any], 
                                   new_params: Dict[str, Any]):
        """Store hyperparameter update history."""
        try:
            if model_name not in self.hyperparameter_history:
                self.hyperparameter_history[model_name] = []
            
            update_record = {
                'timestamp': datetime.now(),
                'old_params': old_params,
                'new_params': new_params,
                'model_name': model_name
            }
            
            self.hyperparameter_history[model_name].append(update_record)
            
            # Keep only recent updates
            cutoff_date = datetime.now() - timedelta(days=30)
            self.hyperparameter_history[model_name] = [
                record for record in self.hyperparameter_history[model_name]
                if record['timestamp'] > cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error storing hyperparameter update: {str(e)}")
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.ensemble_weights.copy()
    
    def get_model_performance_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a specific model."""
        try:
            if model_name not in self.feedback_history:
                return None
            
            feedback_list = self.feedback_history[model_name]
            if not feedback_list:
                return None
            
            recent_feedback = feedback_list[-10:]
            
            return {
                'model_name': model_name,
                'total_predictions': len(feedback_list),
                'recent_mse': np.mean([f.mse for f in recent_feedback]),
                'recent_sharpe': np.mean([f.sharpe_ratio for f in recent_feedback]),
                'recent_win_rate': np.mean([f.win_rate for f in recent_feedback]),
                'recent_max_drawdown': np.mean([f.max_drawdown for f in recent_feedback]),
                'last_updated': recent_feedback[-1].timestamp if recent_feedback else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return None
    
    def _load_feedback_history(self):
        """Load feedback history from memory."""
        try:
            feedback_file = Path("memory/feedback_history.json")
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    for model_name, feedback_list in data.items():
                        self.feedback_history[model_name] = [
                            ModelFeedback(**f) for f in feedback_list
                        ]
                        
        except Exception as e:
            self.logger.error(f"Error loading feedback history: {str(e)}")
    
    def _load_ensemble_weights(self):
        """Load ensemble weights from memory."""
        try:
            weights_data = self.memory.get('ensemble_weights')
            if weights_data:
                self.ensemble_weights = weights_data.get('weights', {})
                
        except Exception as e:
            self.logger.error(f"Error loading ensemble weights: {str(e)}")
    
    def save_feedback_history(self):
        """Save feedback history to file."""
        try:
            feedback_file = Path("memory/feedback_history.json")
            feedback_file.parent.mkdir(exist_ok=True)
            
            data = {}
            for model_name, feedback_list in self.feedback_history.items():
                data[model_name] = [f.__dict__ for f in feedback_list]
            
            with open(feedback_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving feedback history: {str(e)}")