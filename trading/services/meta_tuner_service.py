"""
Meta Tuner Service

Service wrapper for the MetaTunerAgent, handling hyperparameter tuning requests
via Redis pub/sub communication.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional
import json
from pathlib import Path

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading.agents.meta_tuner_agent import MetaTunerAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)

class MetaTunerService(BaseService):
    """
    Service wrapper for MetaTunerAgent.
    
    Handles hyperparameter tuning requests and communicates results via Redis.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0):
        """Initialize the MetaTunerService."""
        super().__init__('meta_tuner', redis_host, redis_port, redis_db)
        
        # Initialize the agent
        self.agent = MetaTunerAgent()
        self.memory = AgentMemory()
        
        logger.info("MetaTunerService initialized")
        
    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming tuning requests.
        
        Args:
            data: Message data containing tuning request
            
        Returns:
            Response with tuning results or error
        """
        try:
            message_type = data.get('type', '')
            
            if message_type == 'tune_hyperparameters':
                return self._handle_tune_request(data)
            elif message_type == 'get_tuning_history':
                return self._handle_history_request(data)
            elif message_type == 'get_best_params':
                return self._handle_best_params_request(data)
            elif message_type == 'auto_tune':
                return self._handle_auto_tune_request(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {
                    'type': 'error',
                    'error': f"Unknown message type: {message_type}",
                    'original_message': data
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'original_message': data
            }
    
    def _handle_tune_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hyperparameter tuning request."""
        try:
            tune_data = data.get('data', {})
            
            # Extract tuning parameters
            model_type = tune_data.get('model_type')
            param_space = tune_data.get('param_space', {})
            optimization_method = tune_data.get('optimization_method', 'bayesian')
            n_trials = tune_data.get('n_trials', 50)
            cv_folds = tune_data.get('cv_folds', 5)
            
            if not model_type:
                return {
                    'type': 'error',
                    'error': 'model_type is required'
                }
            
            logger.info(f"Tuning {model_type} with {optimization_method} optimization")
            
            # Tune hyperparameters using the agent
            result = self.agent.tune_hyperparameters(
                model_type=model_type,
                param_space=param_space,
                optimization_method=optimization_method,
                n_trials=n_trials,
                cv_folds=cv_folds
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='meta_tuner',
                decision_type='tune_hyperparameters',
                details={
                    'model_type': model_type,
                    'optimization_method': optimization_method,
                    'n_trials': n_trials,
                    'cv_folds': cv_folds,
                    'best_score': result.get('best_score', 0),
                    'best_params': result.get('best_params', {})
                }
            )
            
            return {
                'type': 'hyperparameters_tuned',
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'status': 'failed'
            }
    
    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tuning history request."""
        try:
            history_data = data.get('data', {})
            
            # Extract parameters
            model_type = history_data.get('model_type')
            limit = history_data.get('limit', 10)
            
            if not model_type:
                return {
                    'type': 'error',
                    'error': 'model_type is required'
                }
            
            # Get tuning history
            history = self.agent.get_tuning_history(
                model_type=model_type,
                limit=limit
            )
            
            return {
                'type': 'tuning_history',
                'history': history,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Error getting tuning history: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_best_params_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle best parameters request."""
        try:
            params_data = data.get('data', {})
            
            # Extract parameters
            model_type = params_data.get('model_type')
            dataset_size = params_data.get('dataset_size', 'medium')
            
            if not model_type:
                return {
                    'type': 'error',
                    'error': 'model_type is required'
                }
            
            # Get best parameters
            best_params = self.agent.get_best_hyperparameters(
                model_type=model_type,
                dataset_size=dataset_size
            )
            
            return {
                'type': 'best_hyperparameters',
                'best_params': best_params,
                'model_type': model_type,
                'dataset_size': dataset_size
            }
            
        except Exception as e:
            logger.error(f"Error getting best parameters: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_auto_tune_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automatic tuning request."""
        try:
            auto_data = data.get('data', {})
            
            # Extract parameters
            model_types = auto_data.get('model_types', ['lstm', 'xgboost'])
            optimization_methods = auto_data.get('optimization_methods', ['bayesian'])
            max_trials_per_model = auto_data.get('max_trials_per_model', 30)
            
            logger.info(f"Starting auto-tune for models: {model_types}")
            
            # Perform automatic tuning
            result = self.agent.auto_tune_all_models(
                model_types=model_types,
                optimization_methods=optimization_methods,
                max_trials_per_model=max_trials_per_model
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='meta_tuner',
                decision_type='auto_tune',
                details={
                    'model_types': model_types,
                    'optimization_methods': optimization_methods,
                    'max_trials_per_model': max_trials_per_model,
                    'models_tuned': result.get('models_tuned', []),
                    'overall_improvement': result.get('overall_improvement', 0)
                }
            )
            
            return {
                'type': 'auto_tune_completed',
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in auto-tune: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'status': 'failed'
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()
            
            # Get recent tuning activities
            recent_tuning = [
                entry for entry in memory_stats.get('recent_decisions', [])
                if entry.get('agent_name') == 'meta_tuner'
            ]
            
            # Count by type
            tuning_types = {}
            for tuning in recent_tuning:
                tuning_type = tuning.get('decision_type', 'unknown')
                tuning_types[tuning_type] = tuning_types.get(tuning_type, 0) + 1
            
            # Calculate average improvement
            improvements = [
                tuning.get('details', {}).get('best_score', 0)
                for tuning in recent_tuning
                if tuning.get('decision_type') == 'tune_hyperparameters'
            ]
            avg_improvement = sum(improvements) / max(len(improvements), 1)
            
            return {
                'total_tuning_activities': len(recent_tuning),
                'tuning_types': tuning_types,
                'average_improvement': avg_improvement,
                'memory_entries': memory_stats.get('total_entries', 0),
                'recent_tuning': recent_tuning[:5]
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {'error': str(e)}