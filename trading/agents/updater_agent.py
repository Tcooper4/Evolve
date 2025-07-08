"""
Updater Agent

This agent tunes model weights, retrains, or replaces bad models
based on performance critic results.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import shutil

# Local imports
from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.agents.model_builder_agent import ModelBuilderAgent, ModelBuildRequest
from trading.agents.performance_critic_agent import ModelEvaluationRequest, ModelEvaluationResult
from trading.optimization.strategy_optimizer import StrategyOptimizer
from utils.common_helpers import timer, handle_exceptions
from trading.memory.performance_memory import PerformanceMemory
from trading.memory.agent_memory import AgentMemory
from trading.utils.reward_function import RewardFunction

@dataclass
class UpdateRequest:
    """Request for model update."""
    model_id: str
    evaluation_result: ModelEvaluationResult
    update_type: str  # 'retrain', 'tune', 'replace', 'ensemble_adjust'
    priority: str = 'normal'  # 'low', 'normal', 'high', 'critical'
    request_id: Optional[str] = None

@dataclass
class UpdateResult:
    """Result of model update."""
    request_id: str
    model_id: str
    original_model_id: str
    update_timestamp: str
    update_type: str
    new_model_path: str
    new_model_id: str
    improvement_metrics: Dict[str, float]
    update_status: str = "success"
    error_message: Optional[str] = None

class UpdaterAgent(BaseAgent):
    """Agent responsible for updating models based on performance feedback."""
    
    # Agent metadata
    version = "1.0.0"
    description = "Tunes model weights, retrains, or replaces bad models based on performance critic results"
    author = "Evolve Trading System"
    tags = ["model-updating", "optimization", "retraining", "tuning"]
    capabilities = ["model_retraining", "hyperparameter_tuning", "model_replacement", "ensemble_adjustment"]
    dependencies = ["trading.agents.model_builder_agent", "trading.optimization"]
    
    def _setup(self) -> None:
        """Setup method called during initialization."""
        self.memory = PerformanceMemory()
        self.agent_memory = AgentMemory("trading/agents/agent_memory.json")
        
        # Initialize sub-agents
        self.model_builder = ModelBuilderAgent()
        
        # Update thresholds
        self.update_thresholds = {
            'critical_sharpe': 0.0,
            'critical_drawdown': -0.25,
            'critical_win_rate': 0.3,
            'retrain_sharpe': 0.3,
            'retrain_drawdown': -0.15,
            'tune_sharpe': 0.5,
            'tune_drawdown': -0.10
        }
        
        # Update history
        self.update_history: Dict[str, List[UpdateResult]] = {}
        
        # Model registry
        self.active_models: Dict[str, Dict[str, Any]] = {}
        
        self.reward_function = RewardFunction()
        
        self.logger.info("UpdaterAgent initialized")
    
        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat()
        }
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model updating logic.
        
        Args:
            **kwargs: Must contain either 'evaluation_result' or 'request'
            
        Returns:
            AgentResult: Result of the model updating execution
        """
        evaluation_result = kwargs.get('evaluation_result')
        request = kwargs.get('request')
        
        if not evaluation_result and not request:
            return AgentResult(
                success=False,
                error_message="Either evaluation_result or request is required"
            )
        
        try:
            if evaluation_result and not request:
                # Process evaluation and determine if update is needed
                request = self.process_evaluation(evaluation_result)
                if not request:
                    return AgentResult(
                        success=True,
                        data={"message": "No update needed for this model"}
                    )
            
            if not isinstance(request, UpdateRequest):
                return AgentResult(
                    success=False,
                    error_message="Request must be an UpdateRequest instance"
                )
            
            result = self.execute_update(request)
            
            if result.update_status == "success":
                return AgentResult(
                    success=True,
                    data={
                        "model_id": result.model_id,
                        "update_type": result.update_type,
                        "new_model_id": result.new_model_id,
                        "improvement_metrics": result.improvement_metrics,
                        "update_timestamp": result.update_timestamp
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=result.error_message or "Model update failed"
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if input is valid
        """
        evaluation_result = kwargs.get('evaluation_result')
        request = kwargs.get('request')
        
        if not evaluation_result and not request:
            return False
        
        if evaluation_result and not isinstance(evaluation_result, ModelEvaluationResult):
            return False
        
        if request and not isinstance(request, UpdateRequest):
            return False
        
        return True
    
    @handle_exceptions
    def process_evaluation(self, evaluation_result: ModelEvaluationResult) -> Optional[UpdateRequest]:
        """Process evaluation result and determine if update is needed.
        
        Args:
            evaluation_result: Model evaluation result
            
        Returns:
            Update request if needed, None otherwise
        """
        self.logger.info(f"Processing evaluation for model {evaluation_result.model_id}")
        
        # Analyze performance
        performance_score = self._calculate_performance_score(evaluation_result)
        risk_score = self._calculate_risk_score(evaluation_result)
        overall_score = (performance_score + risk_score) / 2
        
        # Determine update type and priority
        update_type, priority = self._determine_update_action(evaluation_result, overall_score)
        
        if update_type:
            request = UpdateRequest(
                model_id=evaluation_result.model_id,
                evaluation_result=evaluation_result,
                update_type=update_type,
                priority=priority,
                request_id=str(uuid.uuid4())
            )
            
            self.logger.info(f"Update needed for model {evaluation_result.model_id}: {update_type} ({priority})")
            return request
        else:
            self.logger.info(f"No update needed for model {evaluation_result.model_id}")

    @handle_exceptions
    def execute_update(self, request: UpdateRequest) -> UpdateResult:
        """Execute model update.
        
        Args:
            request: Update request
            
        Returns:
            Update result
        """
        self.logger.info(f"Executing {request.update_type} update for model {request.model_id}")
        
        try:
            if request.update_type == 'retrain':
                result = self._retrain_model(request)
            elif request.update_type == 'tune':
                result = self._tune_model(request)
            elif request.update_type == 'replace':
                result = self._replace_model(request)
            elif request.update_type == 'ensemble_adjust':
                result = self._adjust_ensemble_weights(request)
            else:
                raise ValueError(f"Unsupported update type: {request.update_type}")
            
            # Store update result
            self._store_update_result(result)
            
            # Update active models registry
            self._update_model_registry(result)
            
            # Compute reward improvement if possible
            reward = None
            if hasattr(result, 'improvement_metrics') and result.improvement_metrics:
                reward = self.reward_function.aggregate(result.improvement_metrics)
            # Log outcome to agent memory (add reward/improvement)
            self.agent_memory.log_outcome(
                agent="UpdaterAgent",
                run_type=request.update_type,
                outcome={
                    "model_id": request.model_id,
                    "update_type": request.update_type,
                    "priority": request.priority,
                    "status": result.update_status,
                    "improvement_metrics": getattr(result, 'improvement_metrics', {}),
                    "reward": reward,
                    "timestamp": getattr(result, 'update_timestamp', datetime.now().isoformat())
                }
            )
            
            self.logger.info(f"Successfully completed {request.update_type} update for model {request.model_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute update for model {request.model_id}: {str(e)}")
            self.agent_memory.log_outcome(
                agent="UpdaterAgent",
                run_type=request.update_type,
                outcome={
                    "model_id": request.model_id,
                    "update_type": request.update_type,
                    "priority": request.priority,
                    "status": "failed",
                    "error_message": str(e),
                    "reward": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return UpdateResult(
                request_id=request.request_id,
                model_id=request.model_id,
                original_model_id=request.model_id,
                update_timestamp=datetime.now().isoformat(),
                update_type=request.update_type,
                new_model_path="",
                new_model_id="",
                improvement_metrics={},
                update_status="failed",
                error_message=str(e)
            )
    
    def _calculate_performance_score(self, evaluation: ModelEvaluationResult) -> float:
        """Calculate performance score from evaluation.
        
        Args:
            evaluation: Model evaluation result
            
        Returns:
            Performance score (0-1)
        """
        metrics = evaluation.performance_metrics
        
        # Normalize metrics to 0-1 scale
        sharpe_score = min(max(metrics.get('sharpe_ratio', 0) / 2.0, 0), 1)
        return_score = min(max(metrics.get('total_return', 0) / 0.5, 0), 1)
        
        # Weighted average
        performance_score = (sharpe_score * 0.6 + return_score * 0.4)
        
        return performance_score
    
    def _calculate_risk_score(self, evaluation: ModelEvaluationResult) -> float:
        """Calculate risk score from evaluation.
        
        Args:
            evaluation: Model evaluation result
            
        Returns:
            Risk score (0-1, higher is better)
        """
        risk_metrics = evaluation.risk_metrics
        trading_metrics = evaluation.trading_metrics
        
        # Normalize metrics to 0-1 scale (inverted for risk)
        drawdown_score = 1 - min(abs(risk_metrics.get('max_drawdown', 0)) / 0.3, 1)
        win_rate_score = trading_metrics.get('win_rate', 0)
        sortino_score = min(max(risk_metrics.get('sortino_ratio', 0) / 1.0, 0), 1)
        
        # Weighted average
        risk_score = (drawdown_score * 0.4 + win_rate_score * 0.3 + sortino_score * 0.3)
        
        return risk_score
    
    def _determine_update_action(self, evaluation: ModelEvaluationResult, overall_score: float) -> Tuple[Optional[str], str]:
        """Determine update action based on evaluation.
        
        Args:
            evaluation: Model evaluation result
            overall_score: Overall performance score
            
        Returns:
            Tuple of (update_type, priority)
        """
        performance_metrics = evaluation.performance_metrics
        risk_metrics = evaluation.risk_metrics
        trading_metrics = evaluation.trading_metrics
        
        # Check for critical issues
        if (performance_metrics.get('sharpe_ratio', 0) <= self.update_thresholds['critical_sharpe'] or
            risk_metrics.get('max_drawdown', 0) <= self.update_thresholds['critical_drawdown'] or
            trading_metrics.get('win_rate', 0) <= self.update_thresholds['critical_win_rate']):
            return 'replace', 'critical'
        
        # Check for retraining needed
        if (performance_metrics.get('sharpe_ratio', 0) <= self.update_thresholds['retrain_sharpe'] or
            risk_metrics.get('max_drawdown', 0) <= self.update_thresholds['retrain_drawdown']):
            return 'retrain', 'high'
        
        # Check for tuning needed
        if (performance_metrics.get('sharpe_ratio', 0) <= self.update_thresholds['tune_sharpe'] or
            risk_metrics.get('max_drawdown', 0) <= self.update_thresholds['tune_drawdown']):
            return 'tune', 'normal'
        
        # Check for ensemble weight adjustment
        if overall_score < 0.6:
            return 'ensemble_adjust', 'low'
        
        return None, 'low'
    
    @timer
    def _retrain_model(self, request: UpdateRequest) -> UpdateResult:
        """Retrain model with improved hyperparameters.
        
        Args:
            request: Update request
            
        Returns:
            Update result
        """
        # Get model metadata
        model_metadata = self.memory.get_model_metadata(request.model_id)
        if not model_metadata:
            raise ValueError(f"No metadata found for model {request.model_id}")
        
        # Create retrain request with improved hyperparameters
        improved_config = self._generate_improved_config(model_metadata['model_config'])
        
        build_request = ModelBuildRequest(
            model_type=model_metadata['model_type'],
            data_path=self._get_latest_data_path(),
            target_column='close',  # This should be configurable
            hyperparameters=improved_config,
            request_id=f"retrain_{request.request_id}"
        )
        
        # Build new model
        build_result = self.model_builder.build_model(build_request)
        
        # Calculate improvement
        improvement_metrics = self._calculate_improvement(
            request.evaluation_result, build_result
        )
        
        return UpdateResult(
            request_id=request.request_id,
            model_id=request.model_id,
            original_model_id=request.model_id,
            update_timestamp=datetime.now().isoformat(),
            update_type='retrain',
            new_model_path=build_result.model_path,
            new_model_id=build_result.model_id,
            improvement_metrics=improvement_metrics
        )
    
    @timer
    def _tune_model(self, request: UpdateRequest) -> UpdateResult:
        """Tune model hyperparameters using optimization.
        
        Args:
            request: Update request
            
        Returns:
            Update result
        """
        # Get model metadata
        model_metadata = self.memory.get_model_metadata(request.model_id)
        if not model_metadata:
            raise ValueError(f"No metadata found for model {request.model_id}")
        
        # Initialize optimizer
        optimizer = StrategyOptimizer({
            'optimizer_type': 'bayesian',
            'n_trials': 50,
            'timeout': 3600
        })
        
        # Define optimization space
        optimization_space = self._define_optimization_space(model_metadata['model_type'])
        
        # Run optimization
        best_params = optimizer.optimize(
            objective_function=self._optimization_objective,
            param_space=optimization_space,
            data_path=self._get_latest_data_path()
        )
        
        # Build model with optimized parameters
        build_request = ModelBuildRequest(
            model_type=model_metadata['model_type'],
            data_path=self._get_latest_data_path(),
            target_column='close',
            hyperparameters=best_params,
            request_id=f"tune_{request.request_id}"
        )
        
        build_result = self.model_builder.build_model(build_request)
        
        # Calculate improvement
        improvement_metrics = self._calculate_improvement(
            request.evaluation_result, build_result
        )
        
        return UpdateResult(
            request_id=request.request_id,
            model_id=request.model_id,
            original_model_id=request.model_id,
            update_timestamp=datetime.now().isoformat(),
            update_type='tune',
            new_model_path=build_result.model_path,
            new_model_id=build_result.model_id,
            improvement_metrics=improvement_metrics
        )
    
    @timer
    def _replace_model(self, request: UpdateRequest) -> UpdateResult:
        """Replace model with a completely new one.
        
        Args:
            request: Update request
            
        Returns:
            Update result
        """
        # Get model metadata
        model_metadata = self.memory.get_model_metadata(request.model_id)
        if not model_metadata:
            raise ValueError(f"No metadata found for model {request.model_id}")
        
        # Try different model type if current one is failing
        new_model_type = self._select_alternative_model_type(model_metadata['model_type'])
        
        # Build new model
        build_request = ModelBuildRequest(
            model_type=new_model_type,
            data_path=self._get_latest_data_path(),
            target_column='close',
            request_id=f"replace_{request.request_id}"
        )
        
        build_result = self.model_builder.build_model(build_request)
        
        # Calculate improvement
        improvement_metrics = self._calculate_improvement(
            request.evaluation_result, build_result
        )
        
        return UpdateResult(
            request_id=request.request_id,
            model_id=request.model_id,
            original_model_id=request.model_id,
            update_timestamp=datetime.now().isoformat(),
            update_type='replace',
            new_model_path=build_result.model_path,
            new_model_id=build_result.model_id,
            improvement_metrics=improvement_metrics
        )
    
    @timer
    def _adjust_ensemble_weights(self, request: UpdateRequest) -> UpdateResult:
        """Adjust ensemble model weights.
        
        Args:
            request: Update request
            
        Returns:
            Update result
        """
        # Get ensemble configuration
        model_metadata = self.memory.get_model_metadata(request.model_id)
        if not model_metadata or model_metadata['model_type'] != 'ensemble':
            raise ValueError(f"Model {request.model_id} is not an ensemble")
        
        # Load ensemble configuration
        with open(model_metadata['model_path'], 'r') as f:
            ensemble_config = json.load(f)
        
        # Optimize weights based on recent performance
        optimized_weights = self._optimize_ensemble_weights(ensemble_config)
        
        # Create new ensemble configuration
        new_config = {
            'models': ensemble_config['models'],
            'voting_method': ensemble_config['voting_method']
        }
        
        # Update weights
        for i, model_info in enumerate(new_config['models']):
            model_info['weight'] = optimized_weights[i]
        
        # Save new configuration
        new_model_id = f"ensemble_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_model_path = Path("trading/models/built") / f"{new_model_id}.json"
        
        with open(new_model_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        return UpdateResult(
            request_id=request.request_id,
            model_id=request.model_id,
            original_model_id=request.model_id,
            update_timestamp=datetime.now().isoformat(),
            update_type='ensemble_adjust',
            new_model_path=str(new_model_path),
            new_model_id=new_model_id,
            improvement_metrics={'weight_optimization': 1.0}
        )
    
    def _generate_improved_config(self, original_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improved configuration based on original.
        
        Args:
            original_config: Original model configuration
            
        Returns:
            Improved configuration
        """
        improved_config = original_config.copy()
        
        # Apply improvements based on model type
        if 'hidden_dim' in improved_config:
            # LSTM improvements
            improved_config['hidden_dim'] = min(improved_config['hidden_dim'] * 1.2, 128)
            improved_config['dropout'] = max(improved_config['dropout'] * 0.9, 0.1)
            improved_config['learning_rate'] = improved_config['learning_rate'] * 0.8
        
        elif 'n_estimators' in improved_config:
            # XGBoost improvements
            improved_config['n_estimators'] = min(improved_config['n_estimators'] * 1.5, 200)
            improved_config['max_depth'] = min(improved_config['max_depth'] + 1, 8)
            improved_config['learning_rate'] = improved_config['learning_rate'] * 0.9
        
        return improved_config
    
    def _define_optimization_space(self, model_type: str) -> Dict[str, Any]:
        """Define optimization space for hyperparameter tuning.
        
        Args:
            model_type: Type of model
            
        Returns:
            Optimization space
        """
        if model_type == 'lstm':
            return {
                'hidden_dim': {'type': 'int', 'min': 32, 'max': 128},
                'num_layers': {'type': 'int', 'min': 1, 'max': 3},
                'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'learning_rate': {'type': 'float', 'min': 0.0001, 'max': 0.01},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]}
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': {'type': 'int', 'min': 50, 'max': 200},
                'max_depth': {'type': 'int', 'min': 3, 'max': 8},
                'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
                'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
                'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0}
            }
        else:
            return {}
    
    def _optimization_objective(self, params: Dict[str, Any], data_path: str) -> float:
        """Objective function for optimization.
        
        Args:
            params: Hyperparameters
            data_path: Path to data
            
        Returns:
            Objective value (lower is better)
        """
        try:
            # Build model with given parameters
            build_request = ModelBuildRequest(
                model_type='lstm',  # This should be configurable
                data_path=data_path,
                target_column='close',
                hyperparameters=params
            )
            
            build_result = self.model_builder.build_model(build_request)
            
            # Return negative Sharpe ratio (minimize)
            return -build_result.training_metrics.get('sharpe_ratio', 0)
            
        except Exception as e:
            self.logger.error(f"Optimization objective failed: {e}")
            return 1000.0  # High penalty for failures
    
    def _select_alternative_model_type(self, current_type: str) -> str:
        """Select alternative model type for replacement.
        
        Args:
            current_type: Current model type
            
        Returns:
            Alternative model type
        """
        alternatives = {
            'lstm': 'xgboost',
            'xgboost': 'lstm',
            'ensemble': 'lstm'
        }
        
        return {'success': True, 'result': alternatives.get(current_type, 'lstm'), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _optimize_ensemble_weights(self, ensemble_config: Dict[str, Any]) -> List[float]:
        """Optimize ensemble weights based on recent performance.
        
        Args:
            ensemble_config: Ensemble configuration
            
        Returns:
            Optimized weights
        """
        # This is a simplified implementation
        # In practice, you'd use optimization to find optimal weights
        num_models = len(ensemble_config['models'])
        return [1.0 / num_models] * num_models
    
    def _calculate_improvement(self, old_evaluation: ModelEvaluationResult, 
                             new_build_result: Any) -> Dict[str, float]:
        """Calculate improvement metrics.
        
        Args:
            old_evaluation: Old model evaluation
            new_build_result: New model build result
            
        Returns:
            Improvement metrics
        """
        # This is a simplified calculation
        # In practice, you'd compare actual performance metrics
        return {'success': True, 'result': {
            'sharpe_improvement': 0.1,
            'return_improvement': 0.05,
            'drawdown_improvement': 0.02
        }, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _get_latest_data_path(self) -> str:
        """Get path to latest data.
        
        Returns:
            Path to latest data file
        """
        # This should be configurable
        return "data/latest_market_data.csv"
    
    def _store_update_result(self, result: UpdateResult) -> None:
        """Store update result in memory.
        
        Args:
            result: Update result
        """
        # Store in local history
        if result.model_id not in self.update_history:
            self.update_history[result.model_id] = []
        
        self.update_history[result.model_id].append(result)
        
        # Store in performance memory
        metadata = {
            'model_id': result.model_id,
            'original_model_id': result.original_model_id,
            'update_timestamp': result.update_timestamp,
            'update_type': result.update_type,
            'new_model_path': result.new_model_path,
            'new_model_id': result.new_model_id,
            'improvement_metrics': result.improvement_metrics,
            'status': result.update_status
        }
        
        self.memory.store_update_result(result.model_id, metadata)
    
    def _update_model_registry(self, result: UpdateResult) -> None:
        """Update active models registry.
        
        Args:
            result: Update result
        """
        if result.update_status == 'success':
            # Remove old model
            if result.original_model_id in self.active_models:
                del self.active_models[result.original_model_id]
            
            # Add new model
            self.active_models[result.new_model_id] = {
                'model_path': result.new_model_path,
                'update_timestamp': result.update_timestamp,
                'update_type': result.update_type,
                'improvement_metrics': result.improvement_metrics
            }
    
    def get_update_history(self, model_id: str) -> List[UpdateResult]:
        """Get update history for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of update results
        """
        return {'success': True, 'result': self.update_history.get(model_id, []), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_active_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all active models.
        
        Returns:
            Dictionary of active models
        """
        return self.active_models.copy()
    
    def cleanup_old_models(self, max_age_days: int = 30) -> int:
        """Clean up old model files.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of models cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        for model_id, model_info in list(self.active_models.items()):
            update_timestamp = datetime.fromisoformat(model_info['update_timestamp'])
            if update_timestamp < cutoff_date:
                # Remove model file
                model_path = Path(model_info['model_path'])
                if model_path.exists():
                    model_path.unlink()
                
                # Remove from registry
                del self.active_models[model_id]
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old models")
        return cleaned_count