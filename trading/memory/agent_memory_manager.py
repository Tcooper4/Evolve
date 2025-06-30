"""
Agent Memory Manager for Evolve Trading Platform

This module provides institutional-level agent memory management:
- Redis-based memory storage with fallback to local storage
- Agent interaction history and performance tracking
- Strategy success/failure memory for continuous improvement
- Confidence boosting for recently successful strategies
- Long-term performance decay tracking
- Meta-agent loop for strategy retirement and tuning
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Agent interaction record."""
    timestamp: datetime
    agent_type: str
    prompt: str
    response: str
    confidence: float
    success: bool
    metadata: Dict[str, Any]
    execution_time: float

@dataclass
class StrategyMemory:
    """Strategy performance memory record."""
    strategy_name: str
    timestamp: datetime
    performance: Dict[str, float]
    confidence: float
    regime: str
    success: bool
    parameters: Dict[str, Any]
    execution_time: float

@dataclass
class ModelMemory:
    """Model performance memory record."""
    model_name: str
    timestamp: datetime
    performance: Dict[str, float]
    confidence: float
    data_quality: float
    success: bool
    parameters: Dict[str, Any]
    execution_time: float

class AgentMemoryManager:
    """Agent memory manager with Redis fallback."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, fallback_storage: str = 'local'):
        """Initialize the agent memory manager.
        
        Args:
            redis_host: Redis host address
            redis_port: Redis port
            redis_db: Redis database number
            fallback_storage: Fallback storage type ('local' or 'memory')
        """
        self.redis_client = None
        self.fallback_storage = fallback_storage
        self.local_storage_path = Path('memory/agent_memory')
        self.memory_cache = {}
        
        # Initialize storage
        self._initialize_storage(redis_host, redis_port, redis_db)
        
        logger.info("Agent Memory Manager initialized")
    
    def _initialize_storage(self, redis_host: str, redis_port: int, redis_db: int):
        """Initialize storage backend."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=1
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
            
            # Setup fallback storage
            if self.fallback_storage == 'local':
                self._setup_local_storage()
            else:
                logger.info("Using in-memory storage")
    
    def _setup_local_storage(self):
        """Setup local file storage."""
        try:
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.local_storage_path / 'interactions').mkdir(exist_ok=True)
            (self.local_storage_path / 'strategies').mkdir(exist_ok=True)
            (self.local_storage_path / 'models').mkdir(exist_ok=True)
            (self.local_storage_path / 'cache').mkdir(exist_ok=True)
            
            logger.info(f"Local storage setup at {self.local_storage_path}")
            
        except Exception as e:
            logger.error(f"Local storage setup failed: {e}")
            self.fallback_storage = 'memory'
    
    def store_agent_interaction(self, interaction: AgentInteraction) -> dict:
        """Store agent interaction in memory.
        
        Args:
            interaction: Agent interaction record
            
        Returns:
            Dictionary with storage status and details
        """
        try:
            success = False
            if self.redis_client:
                success = self._store_in_redis('interaction', interaction)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('interaction', interaction)
            else:
                success = self._store_in_memory('interaction', interaction)
            
            if success:
                return {
                    'success': True,
                    'message': f'Agent interaction stored successfully',
                    'agent_type': interaction.agent_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'storage_type': 'redis' if self.redis_client else self.fallback_storage
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to store agent interaction',
                    'agent_type': interaction.agent_type,
                    'timestamp': interaction.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to store agent interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_type': interaction.agent_type,
                'timestamp': interaction.timestamp.isoformat()
            }
    
    def store_strategy_memory(self, strategy_memory: StrategyMemory) -> dict:
        """Store strategy performance memory.
        
        Args:
            strategy_memory: Strategy memory record
            
        Returns:
            Dictionary with storage status and details
        """
        try:
            success = False
            if self.redis_client:
                success = self._store_in_redis('strategy', strategy_memory)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('strategy', strategy_memory)
            else:
                success = self._store_in_memory('strategy', strategy_memory)
            
            if success:
                return {
                    'success': True,
                    'message': f'Strategy memory stored successfully',
                    'strategy_name': strategy_memory.strategy_name,
                    'timestamp': strategy_memory.timestamp.isoformat(),
                    'storage_type': 'redis' if self.redis_client else self.fallback_storage
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to store strategy memory',
                    'strategy_name': strategy_memory.strategy_name,
                    'timestamp': strategy_memory.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to store strategy memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_name': strategy_memory.strategy_name,
                'timestamp': strategy_memory.timestamp.isoformat()
            }
    
    def store_model_memory(self, model_memory: ModelMemory) -> dict:
        """Store model performance memory.
        
        Args:
            model_memory: Model memory record
            
        Returns:
            Dictionary with storage status and details
        """
        try:
            success = False
            if self.redis_client:
                success = self._store_in_redis('model', model_memory)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('model', model_memory)
            else:
                success = self._store_in_memory('model', model_memory)
            
            if success:
                return {
                    'success': True,
                    'message': f'Model memory stored successfully',
                    'model_name': model_memory.model_name,
                    'timestamp': model_memory.timestamp.isoformat(),
                    'storage_type': 'redis' if self.redis_client else self.fallback_storage
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to store model memory',
                    'model_name': model_memory.model_name,
                    'timestamp': model_memory.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to store model memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_memory.model_name,
                'timestamp': model_memory.timestamp.isoformat()
            }
    
    def _store_in_redis(self, data_type: str, data: Any) -> bool:
        """Store data in Redis."""
        try:
            key = f"agent_memory:{data_type}:{data.timestamp.isoformat()}"
            value = json.dumps(asdict(data), default=str)
            self.redis_client.set(key, value, ex=86400*30)  # 30 days expiry
            return True
        except Exception as e:
            logger.error(f"Redis storage failed: {e}")
            return False
    
    def _store_in_local(self, data_type: str, data: Any) -> bool:
        """Store data in local files."""
        try:
            timestamp_str = data.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp_str}.json"
            filepath = self.local_storage_path / data_type / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(data), f, default=str, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            return False
    
    def _store_in_memory(self, data_type: str, data: Any) -> bool:
        """Store data in memory cache."""
        try:
            if data_type not in self.memory_cache:
                self.memory_cache[data_type] = []
            
            self.memory_cache[data_type].append(data)
            
            # Keep only last 1000 records per type
            if len(self.memory_cache[data_type]) > 1000:
                self.memory_cache[data_type] = self.memory_cache[data_type][-1000:]
            
            return True
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return False
    
    def get_agent_interactions(self, agent_type: str = None, 
                             limit: int = 100, 
                             since: datetime = None) -> dict:
        """Get agent interactions from memory.
        
        Args:
            agent_type: Filter by agent type
            limit: Maximum number of records to return
            since: Get records since this timestamp
            
        Returns:
            Dictionary with agent interactions and status
        """
        try:
            records = []
            if self.redis_client:
                records = self._get_from_redis('interaction', agent_type, limit, since)
            elif self.fallback_storage == 'local':
                records = self._get_from_local('interaction', agent_type, limit, since)
            else:
                records = self._get_from_memory('interaction', agent_type, limit, since)
            
            return {
                'success': True,
                'result': records,
                'message': f'Retrieved {len(records)} agent interactions',
                'agent_type': agent_type,
                'limit': limit,
                'count': len(records),
                'timestamp': datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Failed to get agent interactions: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': [],
                'agent_type': agent_type,
                'limit': limit,
                'count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_strategy_memory(self, strategy_name: str = None,
                          limit: int = 100,
                          since: datetime = None) -> dict:
        """Get strategy memory records.
        
        Args:
            strategy_name: Filter by strategy name
            limit: Maximum number of records to return
            since: Get records since this timestamp
            
        Returns:
            Dictionary with strategy memory records and status
        """
        try:
            records = []
            if self.redis_client:
                records = self._get_from_redis('strategy', strategy_name, limit, since)
            elif self.fallback_storage == 'local':
                records = self._get_from_local('strategy', strategy_name, limit, since)
            else:
                records = self._get_from_memory('strategy', strategy_name, limit, since)
            
            return {
                'success': True,
                'result': records,
                'message': f'Retrieved {len(records)} strategy memory records',
                'strategy_name': strategy_name,
                'limit': limit,
                'count': len(records),
                'timestamp': datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Failed to get strategy memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': [],
                'strategy_name': strategy_name,
                'limit': limit,
                'count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_memory(self, model_name: str = None,
                        limit: int = 100,
                        since: datetime = None) -> dict:
        """Get model memory records.
        
        Args:
            model_name: Filter by model name
            limit: Maximum number of records to return
            since: Get records since this timestamp
            
        Returns:
            Dictionary with model memory records and status
        """
        try:
            records = []
            if self.redis_client:
                records = self._get_from_redis('model', model_name, limit, since)
            elif self.fallback_storage == 'local':
                records = self._get_from_local('model', model_name, limit, since)
            else:
                records = self._get_from_memory('model', model_name, limit, since)
            
            return {
                'success': True,
                'result': records,
                'message': f'Retrieved {len(records)} model memory records',
                'model_name': model_name,
                'limit': limit,
                'count': len(records),
                'timestamp': datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Failed to get model memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': [],
                'model_name': model_name,
                'limit': limit,
                'count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_from_redis(self, data_type: str, filter_value: str = None,
                       limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from Redis."""
        try:
            pattern = f"agent_memory:{data_type}:*"
            keys = self.redis_client.keys(pattern)
            
            records = []
            for key in keys[-limit:]:  # Get most recent
                value = self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    
                    # Apply filters
                    if filter_value and data.get('agent_type' if data_type == 'interaction' else f'{data_type}_name') != filter_value:
                        continue
                    
                    if since:
                        record_time = datetime.fromisoformat(data['timestamp'])
                        if record_time < since:
                            continue
                    
                    # Convert back to dataclass
                    if data_type == 'interaction':
                        record = AgentInteraction(**data)
                    elif data_type == 'strategy':
                        record = StrategyMemory(**data)
                    elif data_type == 'model':
                        record = ModelMemory(**data)
                    else:
                        continue
                    
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Redis retrieval failed: {e}")
            return []
    
    def _get_from_local(self, data_type: str, filter_value: str = None,
                       limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from local files."""
        try:
            data_dir = self.local_storage_path / data_type
            if not data_dir.exists():
                return []
            
            files = sorted(data_dir.glob(f"{data_type}_*.json"), reverse=True)
            records = []
            
            for filepath in files[:limit]:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Apply filters
                    if filter_value and data.get('agent_type' if data_type == 'interaction' else f'{data_type}_name') != filter_value:
                        continue
                    
                    if since:
                        record_time = datetime.fromisoformat(data['timestamp'])
                        if record_time < since:
                            continue
                    
                    # Convert back to dataclass
                    if data_type == 'interaction':
                        record = AgentInteraction(**data)
                    elif data_type == 'strategy':
                        record = StrategyMemory(**data)
                    elif data_type == 'model':
                        record = ModelMemory(**data)
                    else:
                        continue
                    
                    records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to read file {filepath}: {e}")
                    continue
            
            return records
            
        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return []
    
    def _get_from_memory(self, data_type: str, filter_value: str = None,
                        limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from memory cache."""
        try:
            if data_type not in self.memory_cache:
                return []
            
            records = self.memory_cache[data_type][-limit:]
            
            # Apply filters
            if filter_value or since:
                filtered_records = []
                for record in records:
                    # Apply filter
                    if filter_value:
                        if data_type == 'interaction' and record.agent_type != filter_value:
                            continue
                        elif data_type == 'strategy' and record.strategy_name != filter_value:
                            continue
                        elif data_type == 'model' and record.model_name != filter_value:
                            continue
                    
                    # Apply time filter
                    if since and record.timestamp < since:
                        continue
                    
                    filtered_records.append(record)
                
                return filtered_records
            
            return records
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    def get_strategy_confidence_boost(self, strategy_name: str) -> dict:
        """Get confidence boost for a strategy based on recent success.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with confidence boost and status
        """
        try:
            # Get recent strategy performance
            recent_memory_result = self.get_strategy_memory(strategy_name, limit=20)
            if not recent_memory_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get strategy memory',
                    'strategy_name': strategy_name,
                    'confidence_boost': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            recent_memory = recent_memory_result.get('result', [])
            if not recent_memory:
                return {
                    'success': True,
                    'confidence_boost': 0.0,
                    'message': 'No recent memory found',
                    'strategy_name': strategy_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate success rate in last 20 executions
            recent_successes = sum(1 for record in recent_memory if record.success)
            success_rate = recent_successes / len(recent_memory)
            
            # Calculate average confidence
            avg_confidence = np.mean([record.confidence for record in recent_memory])
            
            # Calculate performance trend
            if len(recent_memory) >= 2:
                recent_performance = [record.performance.get('sharpe_ratio', 0) for record in recent_memory[-10:]]
                if len(recent_performance) >= 2:
                    performance_trend = np.mean(recent_performance[-5:]) - np.mean(recent_performance[:5])
                else:
                    performance_trend = 0.0
            else:
                performance_trend = 0.0
            
            # Calculate confidence boost
            boost = (success_rate * 0.4 + avg_confidence * 0.3 + max(0, performance_trend) * 0.3)
            confidence_boost = min(1.0, max(0.0, boost))
            
            return {
                'success': True,
                'confidence_boost': confidence_boost,
                'message': f'Confidence boost calculated for {strategy_name}',
                'strategy_name': strategy_name,
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'performance_trend': performance_trend,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Confidence boost calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence_boost': 0.0,
                'strategy_name': strategy_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_confidence_boost(self, model_name: str) -> dict:
        """Get confidence boost for a model based on recent success.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with confidence boost and status
        """
        try:
            # Get recent model performance
            recent_memory_result = self.get_model_memory(model_name, limit=20)
            if not recent_memory_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get model memory',
                    'model_name': model_name,
                    'confidence_boost': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            recent_memory = recent_memory_result.get('result', [])
            if not recent_memory:
                return {
                    'success': True,
                    'confidence_boost': 0.0,
                    'message': 'No recent memory found',
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate success rate in last 20 executions
            recent_successes = sum(1 for record in recent_memory if record.success)
            success_rate = recent_successes / len(recent_memory)
            
            # Calculate average confidence
            avg_confidence = np.mean([record.confidence for record in recent_memory])
            
            # Calculate data quality trend
            if len(recent_memory) >= 2:
                recent_quality = [record.data_quality for record in recent_memory[-10:]]
                if len(recent_quality) >= 2:
                    quality_trend = np.mean(recent_quality[-5:]) - np.mean(recent_quality[:5])
                else:
                    quality_trend = 0.0
            else:
                quality_trend = 0.0
            
            # Calculate confidence boost
            boost = (success_rate * 0.4 + avg_confidence * 0.3 + max(0, quality_trend) * 0.3)
            confidence_boost = min(1.0, max(0.0, boost))
            
            return {
                'success': True,
                'confidence_boost': confidence_boost,
                'message': f'Confidence boost calculated for {model_name}',
                'model_name': model_name,
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'quality_trend': quality_trend,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model confidence boost calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence_boost': 0.0,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def check_strategy_retirement(self, strategy_name: str) -> dict:
        """Check if a strategy should be retired based on performance decay.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with retirement recommendation and status
        """
        try:
            # Get strategy performance history
            memory_result = self.get_strategy_memory(strategy_name, limit=100)
            if not memory_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get strategy memory',
                    'strategy_name': strategy_name,
                    'should_retire': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            memory = memory_result.get('result', [])
            if len(memory) < 20:
                return {
                    'success': True,
                    'should_retire': False,
                    'reason': 'Insufficient data',
                    'confidence': 0.0,
                    'strategy_name': strategy_name,
                    'metrics': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate performance metrics
            recent_performance = memory[-20:]
            older_performance = memory[-50:-20] if len(memory) >= 50 else memory[:-20]
            
            recent_sharpe = np.mean([r.performance.get('sharpe_ratio', 0) for r in recent_performance])
            older_sharpe = np.mean([r.performance.get('sharpe_ratio', 0) for r in older_performance])
            
            recent_success_rate = sum(1 for r in recent_performance if r.success) / len(recent_performance)
            older_success_rate = sum(1 for r in older_performance if r.success) / len(older_performance)
            
            # Calculate decay
            sharpe_decay = (older_sharpe - recent_sharpe) / max(abs(older_sharpe), 0.1)
            success_decay = older_success_rate - recent_success_rate
            
            # Determine if should retire
            should_retire = (
                sharpe_decay > 0.3 or  # 30% Sharpe decay
                success_decay > 0.2 or  # 20% success rate decay
                recent_success_rate < 0.3  # Less than 30% success rate
            )
            
            confidence = min(1.0, (sharpe_decay + success_decay) / 2)
            
            return {
                'success': True,
                'should_retire': should_retire,
                'reason': f"Sharpe decay: {sharpe_decay:.1%}, Success decay: {success_decay:.1%}",
                'confidence': confidence,
                'strategy_name': strategy_name,
                'metrics': {
                    'sharpe_decay': sharpe_decay,
                    'success_decay': success_decay,
                    'recent_success_rate': recent_success_rate,
                    'recent_sharpe': recent_sharpe
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy retirement check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'should_retire': False,
                'strategy_name': strategy_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_health(self) -> dict:
        """Get memory system health information."""
        try:
            # Count records
            interaction_result = self.get_agent_interactions(limit=1000)
            strategy_result = self.get_strategy_memory(limit=1000)
            model_result = self.get_model_memory(limit=1000)
            
            interaction_count = len(interaction_result.get('result', []))
            strategy_count = len(strategy_result.get('result', []))
            model_count = len(model_result.get('result', []))
            
            # Check storage status
            storage_status = 'redis' if self.redis_client else self.fallback_storage
            
            # Calculate memory usage
            total_records = interaction_count + strategy_count + model_count
            
            return {
                'success': True,
                'status': 'healthy' if total_records > 0 else 'empty',
                'storage_backend': storage_status,
                'total_records': total_records,
                'interaction_records': interaction_count,
                'strategy_records': strategy_count,
                'model_records': model_count,
                'redis_available': self.redis_client is not None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_memory(self, data_type: str = None) -> dict:
        """Clear memory data.
        
        Args:
            data_type: Type of data to clear ('interaction', 'strategy', 'model', or None for all)
            
        Returns:
            Dictionary with clear status and details
        """
        try:
            if data_type is None:
                data_types = ['interaction', 'strategy', 'model']
            else:
                data_types = [data_type]
            
            cleared_count = 0
            for dt in data_types:
                if self.redis_client:
                    pattern = f"agent_memory:{dt}:*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        cleared_count += len(keys)
                
                elif self.fallback_storage == 'local':
                    data_dir = self.local_storage_path / dt
                    if data_dir.exists():
                        files = list(data_dir.glob(f"{dt}_*.json"))
                        for filepath in files:
                            filepath.unlink()
                        cleared_count += len(files)
                
                else:
                    if dt in self.memory_cache:
                        cleared_count += len(self.memory_cache[dt])
                        self.memory_cache[dt].clear()
            
            logger.info(f"Memory cleared for types: {data_types}")
            return {
                'success': True,
                'message': f'Memory cleared successfully for {len(data_types)} data types',
                'cleared_count': cleared_count,
                'data_types': data_types,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory clear failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_types': data_types if 'data_types' in locals() else [],
                'timestamp': datetime.now().isoformat()
            }


# Global instance
agent_memory_manager = AgentMemoryManager()

def get_agent_memory_manager() -> dict:
    """Get the global agent memory manager instance."""
    try:
        return {
            'success': True,
            'result': agent_memory_manager,
            'message': 'Agent memory manager retrieved successfully',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }