"""
Model Builder Service

Service wrapper for the ModelBuilderAgent, handling model building requests
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

from trading.agents.model_builder_agent import ModelBuilderAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)

class ModelBuilderService(BaseService):
    """
    Service wrapper for ModelBuilderAgent.
    
    Handles model building requests and communicates results via Redis.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0):
        """Initialize the ModelBuilderService."""
        super().__init__('model_builder', redis_host, redis_port, redis_db)
        
        # Initialize the agent
        self.agent = ModelBuilderAgent()
        self.memory = AgentMemory()
        
        logger.info("ModelBuilderService initialized")def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming model building requests.
        
        Args:
            data: Message data containing build request
            
        Returns:
            Response with model information or error
        """
        try:
            message_type = data.get('type', '')
            
            if message_type == 'build_model':
                return self._handle_build_request(data)
            elif message_type == 'list_models':
                return self._handle_list_request(data)
            elif message_type == 'get_model_info':
                return self._handle_info_request(data)
            elif message_type == 'delete_model':
                return self._handle_delete_request(data)
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
    
    def _handle_build_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model build request."""
        try:
            build_data = data.get('data', {})
            
            # Extract build parameters
            model_type = build_data.get('model_type', 'lstm')
            symbol = build_data.get('symbol', 'BTCUSDT')
            timeframe = build_data.get('timeframe', '1h')
            features = build_data.get('features', [])
            hyperparameters = build_data.get('hyperparameters', {})
            
            logger.info(f"Building {model_type} model for {symbol}")
            
            # Build the model using the agent
            model_info = self.agent.build_model(
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe,
                features=features,
                hyperparameters=hyperparameters
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='model_builder',
                decision_type='build_model',
                details={
                    'model_type': model_type,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'features': features,
                    'hyperparameters': hyperparameters,
                    'model_id': model_info.get('model_id')
                }
            )
            
            return {
                'type': 'model_built',
                'model_info': model_info,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'status': 'failed'
            }
    
    def _handle_list_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model list request."""
        try:
            models = self.agent.list_models()
            
            return {
                'type': 'models_listed',
                'models': models,
                'count': len(models)
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_info_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model info request."""
        try:
            model_id = data.get('data', {}).get('model_id')
            if not model_id:
                return {
                    'type': 'error',
                    'error': 'model_id is required'
                }
            
            model_info = self.agent.get_model_info(model_id)
            
            return {
                'type': 'model_info',
                'model_info': model_info
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_delete_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model delete request."""
        try:
            model_id = data.get('data', {}).get('model_id')
            if not model_id:
                return {
                    'type': 'error',
                    'error': 'model_id is required'
                }
            
            success = self.agent.delete_model(model_id)
            
            if success:
                # Log to memory
                self.memory.log_decision(
                    agent_name='model_builder',
                    decision_type='delete_model',
                    details={'model_id': model_id}
                )
                
                return {
                    'type': 'model_deleted',
                    'model_id': model_id,
                    'status': 'success'
                }
            else:
                return {
                    'type': 'error',
                    'error': f'Failed to delete model {model_id}',
                    'status': 'failed'
                }
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            models = self.agent.list_models()
            memory_stats = self.memory.get_stats()
            
            return {
                'total_models': len(models),
                'model_types': list(set(m.get('model_type', 'unknown') for m in models)),
                'memory_entries': memory_stats.get('total_entries', 0),
                'recent_decisions': memory_stats.get('recent_decisions', [])
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {'error': str(e)}