"""
QuantGPT Service

Service wrapper for the QuantGPT interface, handling natural language queries
via Redis pub/sub communication.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional
import json

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .base_service import BaseService
from .quant_gpt import QuantGPT
from ..memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class QuantGPTService(BaseService):
    """
    Service wrapper for QuantGPT.
    
    Handles natural language queries and communicates results via Redis.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, openai_api_key: str = None):
        """Initialize the QuantGPTService."""
        super().__init__('quant_gpt', redis_host, redis_port, redis_db)
        
        # Initialize QuantGPT
        self.quant_gpt = QuantGPT(
            openai_api_key=openai_api_key,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db
        )
        self.memory = AgentMemory()
        
        logger.info("QuantGPTService initialized")
    
    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming natural language queries.
        
        Args:
            data: Message data containing query
            
        Returns:
            Response with analysis results and GPT commentary
        """
        try:
            message_type = data.get('type', '')
            
            if message_type == 'process_query':
                return self._handle_query_request(data)
            elif message_type == 'get_query_history':
                return self._handle_history_request(data)
            elif message_type == 'get_available_symbols':
                return self._handle_symbols_request(data)
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
    
    def _handle_query_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle natural language query request."""
        try:
            query_data = data.get('data', {})
            
            # Extract query parameters
            query = query_data.get('query')
            
            if not query:
                return {
                    'type': 'error',
                    'error': 'query is required'
                }
            
            logger.info(f"Processing query: {query}")
            
            # Process the query using QuantGPT
            result = self.quant_gpt.process_query(query)
            
            # Log to memory
            self.memory.log_decision(
                agent_name='quant_gpt',
                decision_type='query_processed',
                details={
                    'query': query,
                    'intent': result.get('parsed_intent', {}).get('intent'),
                    'symbol': result.get('parsed_intent', {}).get('symbol'),
                    'status': result.get('status')
                }
            )
            
            return {
                'type': 'query_processed',
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'status': 'failed'
            }
    
    def _handle_history_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query history request."""
        try:
            history_data = data.get('data', {})
            
            # Extract parameters
            limit = history_data.get('limit', 10)
            symbol = history_data.get('symbol')
            
            # Get query history from memory
            memory_stats = self.memory.get_stats()
            
            # Filter by symbol if specified
            recent_queries = [
                entry for entry in memory_stats.get('recent_decisions', [])
                if entry.get('agent_name') == 'quant_gpt' and 
                   entry.get('decision_type') == 'query_processed'
            ]
            
            if symbol:
                recent_queries = [
                    query for query in recent_queries
                    if query.get('details', {}).get('symbol') == symbol
                ]
            
            # Limit results
            recent_queries = recent_queries[:limit]
            
            return {
                'type': 'query_history',
                'history': recent_queries,
                'symbol': symbol,
                'count': len(recent_queries)
            }
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_symbols_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle available symbols request."""
        try:
            return {
                'type': 'available_symbols',
                'symbols': self.quant_gpt.trading_context['available_symbols'],
                'timeframes': self.quant_gpt.trading_context['available_timeframes'],
                'periods': self.quant_gpt.trading_context['available_periods'],
                'models': self.quant_gpt.trading_context['available_models']
            }
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()
            
            # Get recent queries
            recent_queries = [
                entry for entry in memory_stats.get('recent_decisions', [])
                if entry.get('agent_name') == 'quant_gpt'
            ]
            
            # Count by intent
            intent_counts = {}
            for query in recent_queries:
                intent = query.get('details', {}).get('intent', 'unknown')
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # Count by symbol
            symbol_counts = {}
            for query in recent_queries:
                symbol = query.get('details', {}).get('symbol', 'unknown')
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            return {
                'total_queries': len(recent_queries),
                'intent_counts': intent_counts,
                'symbol_counts': symbol_counts,
                'memory_entries': memory_stats.get('total_entries', 0),
                'recent_queries': recent_queries[:5]
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {'error': str(e)}
    
    def stop(self):
        """Stop the service and clean up resources."""
        if hasattr(self, 'quant_gpt'):
            self.quant_gpt.close()
        super().stop() 