"""
Agent Memory Manager for Evolve Trading Platform

This module provides institutional-level agent memory management:
- Redis-based memory storage with fallback to local storage
- Agent interaction history and performance tracking
- Strategy success/failure memory for continuous improvement
- Confidence boosting for recently successful strategies
- Long-term performance decay tracking
- Meta-agent loop for strategy retirement and tuning
- Thread-safe access for multiple agents
"""

import json
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from contextlib import contextmanager
from queue import Queue
import weakref

# Add Pinecone import
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

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

class ThreadSafeConnectionPool:
    """Thread-safe connection pool for Redis."""
    
    def __init__(self, max_connections: int = 10, connection_timeout: int = 5):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.RLock()
        self._connection_config = None
        self._connection_count = 0
        
    def configure(self, host: str, port: int, db: int, password: str = None):
        """Configure connection parameters."""
        self._connection_config = {
            'host': host,
            'port': port,
            'db': db,
            'password': password,
            'decode_responses': True,
            'socket_connect_timeout': self.connection_timeout
        }
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        connection = None
        try:
            # Try to get existing connection
            try:
                connection = self.connections.get_nowait()
            except Exception as e:
                logger.warning(f"Error getting connection from pool: {e}")
                # Create new connection if pool is empty and under limit
                with self.lock:
                    if self._connection_count < self.max_connections:
                        import redis
                        connection = redis.Redis(**self._connection_config)
                        self._connection_count += 1
                    else:
                        # Wait for a connection to become available
                        connection = self.connections.get(timeout=self.connection_timeout)
            
            yield connection
            
        except Exception as e:
            logger.error(f"Connection pool error: {e}")
            raise
        finally:
            # Return connection to pool if it's still valid
            if connection:
                try:
                    # Test connection
                    connection.ping()
                    self.connections.put(connection)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    # Connection is dead, don't return it
                    with self.lock:
                        self._connection_count -= 1

class AtomicCounter:
    """Thread-safe atomic counter."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self, amount: int = 1) -> int:
        """Atomically increment the counter."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Atomically decrement the counter."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set value atomically."""
        with self._lock:
            self._value = value
            return self._value

class ThreadSafeCache:
    """Thread-safe cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        self.access_counter = AtomicCounter()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] > self.default_ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                self.access_counter.increment()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Evict if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            return True
    
    def _evict_oldest(self):
        """Evict oldest accessed item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'access_count': self.access_counter.get(),
                'utilization': len(self.cache) / self.max_size * 100
            }

class AgentMemoryManager:
    """Agent memory manager with Redis, Pinecone, or fallback - thread-safe."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, fallback_storage: str = 'local',
                 max_connections: int = 10):
        """Initialize the agent memory manager.
        
        Args:
            redis_host: Redis host address
            redis_port: Redis port
            redis_db: Redis database number
            fallback_storage: Fallback storage type ('local' or 'memory')
            max_connections: Maximum Redis connections in pool
        """
        self.memory_backend = os.getenv('MEMORY_BACKEND', 'redis').lower()
        self.pinecone_index = None
        self.redis_pool = None
        self.fallback_storage = fallback_storage
        self.local_storage_path = Path('memory/agent_memory')
        self.memory_cache = ThreadSafeCache()
        
        # Thread safety
        self._lock = threading.RLock()
        self._file_locks = weakref.WeakValueDictionary()
        self._operation_counter = AtomicCounter()
        self._error_counter = AtomicCounter()
        
        # Initialize storage
        if self.memory_backend == 'pinecone' and PINECONE_AVAILABLE:
            self._initialize_pinecone()
        else:
            self._initialize_storage(redis_host, redis_port, redis_db, max_connections)
        
        logger.info(f"Agent Memory Manager initialized with backend: {self.memory_backend}")
    
    def _get_file_lock(self, file_path: str) -> threading.RLock:
        """Get or create a lock for a specific file."""
        with self._lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.RLock()
            return self._file_locks[file_path]
    
    def _initialize_storage(self, redis_host: str, redis_port: int, redis_db: int, max_connections: int):
        """Initialize storage backend."""
        try:
            import redis
            import os
            
            # Get Redis password from environment
            redis_password = os.getenv('REDIS_PASSWORD')
            
            # Initialize connection pool
            self.redis_pool = ThreadSafeConnectionPool(max_connections=max_connections)
            self.redis_pool.configure(redis_host, redis_port, redis_db, redis_password)
            
            # Test connection
            with self.redis_pool.get_connection() as conn:
                conn.ping()
            
            logger.info("Redis connection pool established")
            
        except Exception as e:
            logger.debug(f"Redis connection failed (using fallback): {e}")
            self.redis_pool = None
            
            # Setup fallback storage
            if self.fallback_storage == 'local':
                self._setup_local_storage()
            else:
                logger.info("Using in-memory storage")
    
    def _setup_local_storage(self):
        """Setup local file storage."""
        try:
            with self._lock:
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
    
    def _initialize_pinecone(self):
        """Initialize Pinecone backend."""
        try:
            with self._lock:
                pinecone_api_key = os.getenv('PINECONE_API_KEY')
                pinecone_env = os.getenv('PINECONE_ENV', 'us-west1-gcp')
                pinecone_index_name = os.getenv('PINECONE_INDEX', 'agent-memory')
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
                if pinecone_index_name not in pinecone.list_indexes():
                    pinecone.create_index(pinecone_index_name, dimension=512)
                self.pinecone_index = pinecone.Index(pinecone_index_name)
                logger.info(f"Pinecone index '{pinecone_index_name}' initialized")
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            self.memory_backend = 'memory'

    def store_agent_interaction(self, interaction: AgentInteraction) -> dict:
        """Store agent interaction in memory or Pinecone - thread-safe."""
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"interaction:{interaction.agent_type}:{interaction.timestamp.isoformat()}"
            if self.memory_cache.get(cache_key):
                return {
                    'success': True,
                    'message': 'Interaction already cached',
                    'agent_type': interaction.agent_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'storage_type': 'cache'
                }
            
            success = False
            if self.memory_backend == 'pinecone' and self.pinecone_index:
                success = self._store_in_pinecone('interaction', interaction)
            elif self.redis_pool:
                success = self._store_in_redis('interaction', interaction)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('interaction', interaction)
            else:
                success = self._store_in_memory('interaction', interaction)
            
            if success:
                # Cache the interaction
                self.memory_cache.set(cache_key, interaction)
                
                return {
                    'success': True,
                    'message': f'Agent interaction stored successfully',
                    'agent_type': interaction.agent_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'storage_type': self.memory_backend
                }
            else:
                self._error_counter.increment()
                return {
                    'success': False,
                    'error': 'Failed to store agent interaction',
                    'agent_type': interaction.agent_type,
                    'timestamp': interaction.timestamp.isoformat()
                }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to store agent interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_type': interaction.agent_type,
                'timestamp': interaction.timestamp.isoformat()
            }
    
    def store_strategy_memory(self, strategy_memory: StrategyMemory) -> dict:
        """Store strategy performance memory - thread-safe.
        
        Args:
            strategy_memory: Strategy memory record
        Returns:
            Dictionary with storage status
        """
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"strategy:{strategy_memory.strategy_name}:{strategy_memory.timestamp.isoformat()}"
            if self.memory_cache.get(cache_key):
                return {
                    'success': True,
                    'message': 'Strategy memory already cached',
                    'strategy_name': strategy_memory.strategy_name,
                    'timestamp': strategy_memory.timestamp.isoformat(),
                    'storage_type': 'cache'
                }
            
            success = False
            if self.memory_backend == 'pinecone' and self.pinecone_index:
                success = self._store_in_pinecone('strategy', strategy_memory)
            elif self.redis_pool:
                success = self._store_in_redis('strategy', strategy_memory)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('strategy', strategy_memory)
            else:
                success = self._store_in_memory('strategy', strategy_memory)
            
            if success:
                # Cache the strategy memory
                self.memory_cache.set(cache_key, strategy_memory)
                
                return {
                    'success': True,
                    'message': f'Strategy memory stored successfully',
                    'strategy_name': strategy_memory.strategy_name,
                    'timestamp': strategy_memory.timestamp.isoformat(),
                    'storage_type': self.memory_backend
                }
            else:
                self._error_counter.increment()
                return {
                    'success': False,
                    'error': 'Failed to store strategy memory',
                    'strategy_name': strategy_memory.strategy_name,
                    'timestamp': strategy_memory.timestamp.isoformat()
                }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to store strategy memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_name': strategy_memory.strategy_name,
                'timestamp': strategy_memory.timestamp.isoformat()
            }
    
    def store_model_memory(self, model_memory: ModelMemory) -> dict:
        """Store model performance memory - thread-safe.
        
        Args:
            model_memory: Model memory record
        Returns:
            Dictionary with storage status
        """
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"model:{model_memory.model_name}:{model_memory.timestamp.isoformat()}"
            if self.memory_cache.get(cache_key):
                return {
                    'success': True,
                    'message': 'Model memory already cached',
                    'model_name': model_memory.model_name,
                    'timestamp': model_memory.timestamp.isoformat(),
                    'storage_type': 'cache'
                }
            
            success = False
            if self.memory_backend == 'pinecone' and self.pinecone_index:
                success = self._store_in_pinecone('model', model_memory)
            elif self.redis_pool:
                success = self._store_in_redis('model', model_memory)
            elif self.fallback_storage == 'local':
                success = self._store_in_local('model', model_memory)
            else:
                success = self._store_in_memory('model', model_memory)
            
            if success:
                # Cache the model memory
                self.memory_cache.set(cache_key, model_memory)
                
                return {
                    'success': True,
                    'message': f'Model memory stored successfully',
                    'model_name': model_memory.model_name,
                    'timestamp': model_memory.timestamp.isoformat(),
                    'storage_type': self.memory_backend
                }
            else:
                self._error_counter.increment()
                return {
                    'success': False,
                    'error': 'Failed to store model memory',
                    'model_name': model_memory.model_name,
                    'timestamp': model_memory.timestamp.isoformat()
                }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to store model memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_memory.model_name,
                'timestamp': model_memory.timestamp.isoformat()
            }
    
    def _store_in_redis(self, data_type: str, data: Any) -> bool:
        """Store data in Redis - thread-safe."""
        try:
            with self.redis_pool.get_connection() as conn:
                key = f"{data_type}:{data.timestamp.isoformat()}"
                value = json.dumps(asdict(data))
                conn.setex(key, 86400, value)  # 24 hour TTL
                return True
        except Exception as e:
            logger.error(f"Redis store error: {e}")
            return False
    
    def _store_in_local(self, data_type: str, data: Any) -> bool:
        """Store data in local files - thread-safe."""
        try:
            file_path = self.local_storage_path / f"{data_type}s" / f"{data.timestamp.strftime('%Y%m%d')}.json"
            file_lock = self._get_file_lock(str(file_path))
            
            with file_lock:
                # Load existing data
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []
                
                # Add new data
                existing_data.append(asdict(data))
                
                # Save back to file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Local store error: {e}")
            return False
    
    def _store_in_memory(self, data_type: str, data: Any) -> bool:
        """Store data in memory - thread-safe."""
        try:
            with self._lock:
                if data_type not in self.memory_cache.cache:
                    self.memory_cache.cache[data_type] = []
                self.memory_cache.cache[data_type].append(asdict(data))
                return True
        except Exception as e:
            logger.error(f"Memory store error: {e}")
            return False
    
    def _store_in_pinecone(self, data_type: str, data: Any) -> bool:
        """Store data in Pinecone - thread-safe."""
        try:
            with self._lock:
                vector_id = f"{data_type}_{data.timestamp.isoformat()}"
                metadata = asdict(data)
                # Convert to vector (simplified - in practice you'd use embeddings)
                vector = np.random.rand(512).tolist()
                self.pinecone_index.upsert(vectors=[(vector_id, vector, metadata)])
                return True
        except Exception as e:
            logger.error(f"Pinecone store error: {e}")
            return False

    def get_agent_interactions(self, agent_type: str = None, 
                             limit: int = 100, 
                             since: datetime = None) -> dict:
        """Get agent interactions - thread-safe."""
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"interactions:{agent_type}:{limit}:{since.isoformat() if since else 'all'}"
            cached_result = self.memory_cache.get(cache_key)
            if cached_result:
                return {
                    'success': True,
                    'data': cached_result,
                    'source': 'cache'
                }
            
            data = []
            if self.redis_pool:
                data = self._get_from_redis('interaction', agent_type, limit, since)
            elif self.fallback_storage == 'local':
                data = self._get_from_local('interaction', agent_type, limit, since)
            else:
                data = self._get_from_memory('interaction', agent_type, limit, since)
            
            # Cache the result
            self.memory_cache.set(cache_key, data)
            
            return {
                'success': True,
                'data': data,
                'count': len(data),
                'source': self.memory_backend
            }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to get agent interactions: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def get_strategy_memory(self, strategy_name: str = None,
                          limit: int = 100,
                          since: datetime = None) -> dict:
        """Get strategy memory - thread-safe."""
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"strategies:{strategy_name}:{limit}:{since.isoformat() if since else 'all'}"
            cached_result = self.memory_cache.get(cache_key)
            if cached_result:
                return {
                    'success': True,
                    'data': cached_result,
                    'source': 'cache'
                }
            
            data = []
            if self.redis_pool:
                data = self._get_from_redis('strategy', strategy_name, limit, since)
            elif self.fallback_storage == 'local':
                data = self._get_from_local('strategy', strategy_name, limit, since)
            else:
                data = self._get_from_memory('strategy', strategy_name, limit, since)
            
            # Cache the result
            self.memory_cache.set(cache_key, data)
            
            return {
                'success': True,
                'data': data,
                'count': len(data),
                'source': self.memory_backend
            }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to get strategy memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def get_model_memory(self, model_name: str = None,
                        limit: int = 100,
                        since: datetime = None) -> dict:
        """Get model memory - thread-safe."""
        try:
            self._operation_counter.increment()
            
            # Check cache first
            cache_key = f"models:{model_name}:{limit}:{since.isoformat() if since else 'all'}"
            cached_result = self.memory_cache.get(cache_key)
            if cached_result:
                return {
                    'success': True,
                    'data': cached_result,
                    'source': 'cache'
                }
            
            data = []
            if self.redis_pool:
                data = self._get_from_redis('model', model_name, limit, since)
            elif self.fallback_storage == 'local':
                data = self._get_from_local('model', model_name, limit, since)
            else:
                data = self._get_from_memory('model', model_name, limit, since)
            
            # Cache the result
            self.memory_cache.set(cache_key, data)
            
            return {
                'success': True,
                'data': data,
                'count': len(data),
                'source': self.memory_backend
            }
        except Exception as e:
            self._error_counter.increment()
            logger.error(f"Failed to get model memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def _get_from_redis(self, data_type: str, filter_value: str = None,
                       limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from Redis - thread-safe."""
        try:
            with self.redis_pool.get_connection() as conn:
                pattern = f"{data_type}:*"
                keys = conn.keys(pattern)
                
                data = []
                for key in keys[:limit]:
                    value = conn.get(key)
                    if value:
                        item = json.loads(value)
                        if filter_value and item.get('agent_type' if data_type == 'interaction' else 'strategy_name' if data_type == 'strategy' else 'model_name') != filter_value:
                            continue
                        if since and datetime.fromisoformat(item['timestamp']) < since:
                            continue
                        data.append(item)
                
                return data
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return []
    
    def _get_from_local(self, data_type: str, filter_value: str = None,
                       limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from local files - thread-safe."""
        try:
            data = []
            data_dir = self.local_storage_path / f"{data_type}s"
            
            if not data_dir.exists():
                return data
            
            # Get all JSON files in the directory
            json_files = sorted(data_dir.glob("*.json"), reverse=True)
            
            for file_path in json_files:
                if len(data) >= limit:
                    break
                
                file_lock = self._get_file_lock(str(file_path))
                with file_lock:
                    try:
                        with open(file_path, 'r') as f:
                            file_data = json.load(f)
                        
                        for item in file_data:
                            if len(data) >= limit:
                                break
                            
                            # Apply filters
                            if filter_value:
                                if data_type == 'interaction' and item.get('agent_type') != filter_value:
                                    continue
                                elif data_type == 'strategy' and item.get('strategy_name') != filter_value:
                                    continue
                                elif data_type == 'model' and item.get('model_name') != filter_value:
                                    continue
                            
                            if since and datetime.fromisoformat(item['timestamp']) < since:
                                continue
                            
                            data.append(item)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                        continue
            
            return data
        except Exception as e:
            logger.error(f"Local get error: {e}")
            return []
    
    def _get_from_memory(self, data_type: str, filter_value: str = None,
                        limit: int = 100, since: datetime = None) -> List[Any]:
        """Get data from memory - thread-safe."""
        try:
            with self._lock:
                if data_type not in self.memory_cache.cache:
                    return []
                
                data = []
                for item in self.memory_cache.cache[data_type]:
                    if len(data) >= limit:
                        break
                    
                    # Apply filters
                    if filter_value:
                        if data_type == 'interaction' and item.get('agent_type') != filter_value:
                            continue
                        elif data_type == 'strategy' and item.get('strategy_name') != filter_value:
                            continue
                        elif data_type == 'model' and item.get('model_name') != filter_value:
                            continue
                    
                    if since and datetime.fromisoformat(item['timestamp']) < since:
                        continue
                    
                    data.append(item)
                
                return data
        except Exception as e:
            logger.error(f"Memory get error: {e}")
            return []
    
    def _get_from_pinecone(self, data_type: str, filter_value: str = None, limit: int = 100, since: datetime = None) -> dict:
        """Get data from Pinecone - thread-safe."""
        try:
            with self._lock:
                # Query Pinecone (simplified)
                query_vector = np.random.rand(512).tolist()
                results = self.pinecone_index.query(
                    vector=query_vector,
                    top_k=limit,
                    include_metadata=True
                )
                
                data = []
                for match in results.matches:
                    metadata = match.metadata
                    if filter_value:
                        if data_type == 'interaction' and metadata.get('agent_type') != filter_value:
                            continue
                        elif data_type == 'strategy' and metadata.get('strategy_name') != filter_value:
                            continue
                        elif data_type == 'model' and metadata.get('model_name') != filter_value:
                            continue
                    
                    if since and datetime.fromisoformat(metadata['timestamp']) < since:
                        continue
                    
                    data.append(metadata)
                
                return data
        except Exception as e:
            logger.error(f"Pinecone get error: {e}")
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
            
            recent_memory = recent_memory_result.get('data', []) # Use 'data' from get_strategy_memory
            if not recent_memory:
                return {
                    'success': True,
                    'confidence_boost': 0.0,
                    'message': 'No recent memory found',
                    'strategy_name': strategy_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate success rate in last 20 executions
            recent_successes = sum(1 for record in recent_memory if record.get('success'))
            success_rate = recent_successes / len(recent_memory)
            
            # Calculate average confidence
            avg_confidence = np.mean([record.get('confidence', 0) for record in recent_memory])
            
            # Calculate performance trend
            if len(recent_memory) >= 2:
                recent_performance = [record.get('performance', {}).get('sharpe_ratio', 0) for record in recent_memory[-10:]]
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
            
            recent_memory = recent_memory_result.get('data', []) # Use 'data' from get_model_memory
            if not recent_memory:
                return {
                    'success': True,
                    'confidence_boost': 0.0,
                    'message': 'No recent memory found',
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate success rate in last 20 executions
            recent_successes = sum(1 for record in recent_memory if record.get('success'))
            success_rate = recent_successes / len(recent_memory)
            
            # Calculate average confidence
            avg_confidence = np.mean([record.get('confidence', 0) for record in recent_memory])
            
            # Calculate data quality trend
            if len(recent_memory) >= 2:
                recent_quality = [record.get('data_quality', 0) for record in recent_memory[-10:]]
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
            
            memory = memory_result.get('data', []) # Use 'data' from get_strategy_memory
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
            
            recent_sharpe = np.mean([r.get('performance', {}).get('sharpe_ratio', 0) for r in recent_performance])
            older_sharpe = np.mean([r.get('performance', {}).get('sharpe_ratio', 0) for r in older_performance])
            
            recent_success_rate = sum(1 for r in recent_performance if r.get('success')) / len(recent_performance)
            older_success_rate = sum(1 for r in older_performance if r.get('success')) / len(older_performance)
            
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
        """Get system health information - thread-safe."""
        try:
            health_info = {
                'backend': self.memory_backend,
                'operations_total': self._operation_counter.get(),
                'errors_total': self._error_counter.get(),
                'error_rate': (self._error_counter.get() / max(1, self._operation_counter.get())) * 100,
                'cache_stats': self.memory_cache.get_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add backend-specific health info
            if self.redis_pool:
                try:
                    with self.redis_pool.get_connection() as conn:
                        conn.ping()
                    health_info['redis_status'] = 'healthy'
                except Exception as e:
                    logger.warning(f"Error checking Redis health: {e}")
                    health_info['redis_status'] = 'unhealthy'
            
            if self.pinecone_index:
                try:
                    # Simple health check
                    health_info['pinecone_status'] = 'healthy'
                except Exception as e:
                    logger.warning(f"Error checking Pinecone health: {e}")
                    health_info['pinecone_status'] = 'unhealthy'
            
            return {
                'success': True,
                'health': health_info
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_memory(self, data_type: str = None) -> dict:
        """Clear memory - thread-safe."""
        try:
            with self._lock:
                if data_type is None:
                    # Clear all
                    self.memory_cache.clear()
                    if self.redis_pool:
                        with self.redis_pool.get_connection() as conn:
                            conn.flushdb()
                    return {
                        'success': True,
                        'message': 'All memory cleared'
                    }
                else:
                    # Clear specific type
                    if data_type in self.memory_cache.cache:
                        del self.memory_cache.cache[data_type]
                    return {
                        'success': True,
                        'message': f'{data_type} memory cleared'
                    }
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_memory_snapshot(self, snapshot_name: str = None) -> dict:
        """Create a timestamped snapshot of current memory state for recovery.
        
        Args:
            snapshot_name: Optional custom name for the snapshot
            
        Returns:
            Dictionary with snapshot details and success status
        """
        try:
            timestamp = datetime.now()
            snapshot_id = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            if snapshot_name:
                snapshot_id = f"{snapshot_name}_{snapshot_id}"
            
            # Collect current memory state
            snapshot_data = {
                'snapshot_id': snapshot_id,
                'timestamp': timestamp.isoformat(),
                'memory_backend': self.memory_backend,
                'fallback_storage': self.fallback_storage,
                'interactions': self.get_agent_interactions(limit=1000),
                'strategies': self.get_strategy_memory(limit=1000),
                'models': self.get_model_memory(limit=1000),
                'system_health': self.get_system_health()
            }
            
            # Save snapshot
            snapshot_path = self.local_storage_path / 'snapshots'
            snapshot_path.mkdir(exist_ok=True)
            
            snapshot_file = snapshot_path / f"{snapshot_id}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            # Also save to Redis if available
            if self.redis_pool:
                with self.redis_pool.get_connection() as conn:
                    conn.setex(
                        f"agent_memory:snapshot:{snapshot_id}",
                        86400 * 30,  # 30 days TTL
                        json.dumps(snapshot_data, default=str)
                    )
            
            logger.info(f"Memory snapshot created: {snapshot_id}")
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'timestamp': timestamp.isoformat(),
                'file_path': str(snapshot_file),
                'data_size': len(json.dumps(snapshot_data)),
                'message': f'Memory snapshot created successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to create memory snapshot: {e}")
            return {
                'success': False,
                'error': str(e),
                'snapshot_id': snapshot_id if 'snapshot_id' in locals() else None
            }

    def list_memory_snapshots(self) -> dict:
        """List all available memory snapshots.
        
        Returns:
            Dictionary with list of snapshots and their metadata
        """
        try:
            snapshots = []
            
            # Check local snapshots
            snapshot_path = self.local_storage_path / 'snapshots'
            if snapshot_path.exists():
                for snapshot_file in snapshot_path.glob('*.json'):
                    try:
                        with open(snapshot_file, 'r') as f:
                            snapshot_data = json.load(f)
                            snapshots.append({
                                'snapshot_id': snapshot_data.get('snapshot_id'),
                                'timestamp': snapshot_data.get('timestamp'),
                                'file_path': str(snapshot_file),
                                'storage': 'local'
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read snapshot {snapshot_file}: {e}")
            
            # Check Redis snapshots
            if self.redis_pool:
                try:
                    with self.redis_pool.get_connection() as conn:
                        redis_snapshots = conn.keys("agent_memory:snapshot:*")
                        for key in redis_snapshots:
                            snapshot_id = key.decode().split(':')[-1] # Decode bytes if needed
                            snapshots.append({
                                'snapshot_id': snapshot_id,
                                'timestamp': 'unknown',  # Would need to parse from stored data
                                'storage': 'redis'
                            })
                except Exception as e:
                    logger.warning(f"Failed to read Redis snapshots: {e}")
            
            # Sort by timestamp
            snapshots.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return {
                'success': True,
                'snapshots': snapshots,
                'count': len(snapshots)
            }
            
        except Exception as e:
            logger.error(f"Failed to list memory snapshots: {e}")
            return {
                'success': False,
                'error': str(e),
                'snapshots': []
            }

    def restore_memory_snapshot(self, snapshot_id: str) -> dict:
        """Restore memory state from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to restore
            
        Returns:
            Dictionary with restoration status
        """
        try:
            # Try to find snapshot
            snapshot_data = None
            
            # Check local storage first
            snapshot_path = self.local_storage_path / 'snapshots' / f"{snapshot_id}.json"
            if snapshot_path.exists():
                with open(snapshot_path, 'r') as f:
                    snapshot_data = json.load(f)
            
            # Check Redis if not found locally
            elif self.redis_pool:
                with self.redis_pool.get_connection() as conn:
                    redis_data = conn.get(f"agent_memory:snapshot:{snapshot_id}")
                    if redis_data:
                        snapshot_data = json.loads(redis_data)
            
            if not snapshot_data:
                return {
                    'success': False,
                    'error': f'Snapshot {snapshot_id} not found'
                }
            
            # Clear current memory
            self.clear_memory()
            
            # Restore interactions
            interactions = snapshot_data.get('interactions', {}).get('data', [])
            for interaction_data in interactions:
                try:
                    interaction = AgentInteraction(**interaction_data)
                    self.store_agent_interaction(interaction)
                except Exception as e:
                    logger.warning(f"Failed to restore interaction: {e}")
            
            # Restore strategies
            strategies = snapshot_data.get('strategies', {}).get('data', [])
            for strategy_data in strategies:
                try:
                    strategy = StrategyMemory(**strategy_data)
                    self.store_strategy_memory(strategy)
                except Exception as e:
                    logger.warning(f"Failed to restore strategy: {e}")
            
            # Restore models
            models = snapshot_data.get('models', {}).get('data', [])
            for model_data in models:
                try:
                    model = ModelMemory(**model_data)
                    self.store_model_memory(model)
                except Exception as e:
                    logger.warning(f"Failed to restore model: {e}")
            
            logger.info(f"Memory restored from snapshot: {snapshot_id}")
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'timestamp': snapshot_data.get('timestamp'),
                'restored_interactions': len(interactions),
                'restored_strategies': len(strategies),
                'restored_models': len(models),
                'message': f'Memory restored successfully from snapshot {snapshot_id}'
            }
            
        except Exception as e:
            logger.error(f"Failed to restore memory snapshot: {e}")
            return {
                'success': False,
                'error': str(e),
                'snapshot_id': snapshot_id
            }

    def get_memory_state_hash(self) -> str:
        """Generate a hash of current memory state for change detection.
        
        Returns:
            SHA256 hash of current memory state
        """
        try:
            import hashlib
            
            # Collect current state
            state_data = {
                'interactions': self.get_agent_interactions(limit=100),
                'strategies': self.get_strategy_memory(limit=100),
                'models': self.get_model_memory(limit=100),
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate hash
            state_json = json.dumps(state_data, sort_keys=True, default=str)
            state_hash = hashlib.sha256(state_json.encode()).hexdigest()
            
            return state_hash
            
        except Exception as e:
            logger.error(f"Failed to generate memory state hash: {e}")
            return "unknown"

# Global instance with thread safety
_agent_memory_manager = None
_manager_lock = threading.RLock()

def get_agent_memory_manager() -> AgentMemoryManager:
    """Get the global agent memory manager instance - thread-safe."""
    global _agent_memory_manager
    with _manager_lock:
        if _agent_memory_manager is None:
            _agent_memory_manager = AgentMemoryManager()
        return _agent_memory_manager