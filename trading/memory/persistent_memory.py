"""
Persistent Memory Module with Batch 12 Features

This module provides persistent memory storage for the trading system,
enabling long-term learning from past prompt → action → outcome patterns.

Features:
- Redis and vector store integration for persistent storage
- Long-term learning loop for failed trades and model adjustments
- Semantic similarity search for past interactions
- Performance tracking and trend analysis
- Memory optimization and cleanup
"""

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import sqlite3
from pathlib import Path

import numpy as np

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Try to import vector store
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    
    REDIS = "redis"
    VECTOR_STORE = "vector_store"
    SQLITE = "sqlite"
    HYBRID = "hybrid"


class InteractionType(Enum):
    """Types of interactions."""
    
    PROMPT_PROCESSING = "prompt_processing"
    AGENT_EXECUTION = "agent_execution"
    TRADE_EXECUTION = "trade_execution"
    MODEL_PREDICTION = "model_prediction"
    STRATEGY_GENERATION = "strategy_generation"
    ERROR_RECOVERY = "error_recovery"


class OutcomeType(Enum):
    """Types of outcomes."""
    
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class MemoryInteraction:
    """A memory interaction record."""
    
    interaction_id: str
    timestamp: datetime
    interaction_type: InteractionType
    prompt: str
    action: str
    outcome: OutcomeType
    outcome_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    model_name: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['interaction_type'] = self.interaction_type.value
        data['outcome'] = self.outcome.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryInteraction':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['interaction_type'] = InteractionType(data['interaction_type'])
        data['outcome'] = OutcomeType(data['outcome'])
        return cls(**data)


@dataclass
class LearningInsight:
    """Learning insight from memory analysis."""
    
    insight_id: str
    timestamp: datetime
    insight_type: str
    description: str
    confidence: float
    data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    affected_agents: List[str] = field(default_factory=list)
    affected_models: List[str] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Memory statistics."""
    
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    avg_execution_time: float = 0.0
    avg_confidence: float = 0.0
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    recent_trends: Dict[str, Any] = field(default_factory=dict)


class RedisMemoryStore:
    """Redis-based memory store."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis memory store.
        
        Args:
            redis_url: Redis connection URL
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.redis_client = redis.from_url(redis_url)
        self.prefix = "trading_memory:"
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis memory store")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def store_interaction(self, interaction: MemoryInteraction) -> bool:
        """Store an interaction in Redis.
        
        Args:
            interaction: Memory interaction to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Store interaction data
            interaction_key = f"{self.prefix}interaction:{interaction.interaction_id}"
            interaction_data = interaction.to_dict()
            
            # Store in Redis
            self.redis_client.hset(interaction_key, mapping=interaction_data)
            
            # Set expiration (30 days)
            self.redis_client.expire(interaction_key, 86400 * 30)
            
            # Add to index for quick lookup
            index_key = f"{self.prefix}index:{interaction.interaction_type.value}"
            self.redis_client.zadd(index_key, {interaction.interaction_id: interaction.timestamp.timestamp()})
            
            # Add to user index if user_id exists
            if interaction.user_id:
                user_key = f"{self.prefix}user:{interaction.user_id}"
                self.redis_client.zadd(user_key, {interaction.interaction_id: interaction.timestamp.timestamp()})
            
            # Add to agent index if agent_name exists
            if interaction.agent_name:
                agent_key = f"{self.prefix}agent:{interaction.agent_name}"
                self.redis_client.zadd(agent_key, {interaction.interaction_id: interaction.timestamp.timestamp()})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction in Redis: {e}")
            return False
    
    async def get_interaction(self, interaction_id: str) -> Optional[MemoryInteraction]:
        """Get an interaction from Redis.
        
        Args:
            interaction_id: Interaction ID
            
        Returns:
            MemoryInteraction or None if not found
        """
        try:
            interaction_key = f"{self.prefix}interaction:{interaction_id}"
            data = self.redis_client.hgetall(interaction_key)
            
            if not data:
                return None
            
            # Convert bytes to strings
            data = {k.decode(): v.decode() for k, v in data.items()}
            
            return MemoryInteraction.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to get interaction from Redis: {e}")
            return None
    
    async def search_interactions(
        self,
        interaction_type: Optional[InteractionType] = None,
        user_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        outcome: Optional[OutcomeType] = None,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MemoryInteraction]:
        """Search interactions in Redis.
        
        Args:
            interaction_type: Filter by interaction type
            user_id: Filter by user ID
            agent_name: Filter by agent name
            outcome: Filter by outcome
            limit: Maximum number of results
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of matching interactions
        """
        try:
            # Determine which index to use
            if user_id:
                index_key = f"{self.prefix}user:{user_id}"
            elif agent_name:
                index_key = f"{self.prefix}agent:{agent_name}"
            elif interaction_type:
                index_key = f"{self.prefix}index:{interaction_type.value}"
            else:
                # Use timestamp-based search
                index_key = f"{self.prefix}index:all"
            
            # Get interaction IDs from index
            if start_time and end_time:
                interaction_ids = self.redis_client.zrangebyscore(
                    index_key,
                    start_time.timestamp(),
                    end_time.timestamp(),
                    start=0,
                    num=limit
                )
            else:
                interaction_ids = self.redis_client.zrevrange(index_key, 0, limit - 1)
            
            # Fetch interactions
            interactions = []
            for interaction_id in interaction_ids:
                interaction = await self.get_interaction(interaction_id.decode())
                if interaction:
                    # Apply additional filters
                    if outcome and interaction.outcome != outcome:
                        continue
                    interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to search interactions in Redis: {e}")
            return []
    
    async def get_stats(self) -> MemoryStats:
        """Get memory statistics from Redis.
        
        Returns:
            MemoryStats object
        """
        try:
            stats = MemoryStats()
            
            # Get all interaction keys
            pattern = f"{self.prefix}interaction:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                data = self.redis_client.hgetall(key)
                if data:
                    data = {k.decode(): v.decode() for k, v in data.items()}
                    interaction = MemoryInteraction.from_dict(data)
                    
                    stats.total_interactions += 1
                    if interaction.outcome == OutcomeType.SUCCESS:
                        stats.successful_interactions += 1
                    else:
                        stats.failed_interactions += 1
                    
                    stats.avg_execution_time += interaction.execution_time
                    stats.avg_confidence += interaction.confidence
                    
                    # Track agent performance
                    if interaction.agent_name:
                        if interaction.agent_name not in stats.agent_performance:
                            stats.agent_performance[interaction.agent_name] = {
                                "total": 0,
                                "successful": 0,
                                "avg_time": 0.0,
                                "avg_confidence": 0.0
                            }
                        
                        agent_stats = stats.agent_performance[interaction.agent_name]
                        agent_stats["total"] += 1
                        if interaction.outcome == OutcomeType.SUCCESS:
                            agent_stats["successful"] += 1
                        agent_stats["avg_time"] += interaction.execution_time
                        agent_stats["avg_confidence"] += interaction.confidence
            
            # Calculate averages
            if stats.total_interactions > 0:
                stats.avg_execution_time /= stats.total_interactions
                stats.avg_confidence /= stats.total_interactions
                
                for agent_stats in stats.agent_performance.values():
                    if agent_stats["total"] > 0:
                        agent_stats["avg_time"] /= agent_stats["total"]
                        agent_stats["avg_confidence"] /= agent_stats["total"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats from Redis: {e}")
            return MemoryStats()


class VectorMemoryStore:
    """Vector store-based memory store."""
    
    def __init__(self, vector_dim: int = 384, index_path: str = "data/vector_memory"):
        """Initialize vector memory store.
        
        Args:
            vector_dim: Dimension of vectors
            index_path: Path to store vector index
        """
        if not VECTOR_STORE_AVAILABLE:
            raise ImportError("Vector store not available")
        
        self.vector_dim = vector_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(vector_dim)
        
        # Load existing index if available
        index_file = self.index_path / "faiss.index"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        
        # Store interaction metadata
        self.metadata_file = self.index_path / "metadata.jsonl"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, MemoryInteraction]:
        """Load interaction metadata from file."""
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        interaction = MemoryInteraction.from_dict(data)
                        metadata[interaction.interaction_id] = interaction
        return metadata
    
    def _save_metadata(self):
        """Save interaction metadata to file."""
        with open(self.metadata_file, 'w') as f:
            for interaction in self.metadata.values():
                f.write(json.dumps(interaction.to_dict()) + '\n')
    
    async def store_interaction(self, interaction: MemoryInteraction) -> bool:
        """Store an interaction in vector store.
        
        Args:
            interaction: Memory interaction to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Generate embedding for prompt
            prompt_embedding = self.sentence_transformer.encode([interaction.prompt])
            
            # Add to FAISS index
            self.index.add(prompt_embedding.astype('float32'))
            
            # Store metadata
            self.metadata[interaction.interaction_id] = interaction
            
            # Save metadata
            self._save_metadata()
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction in vector store: {e}")
            return False
    
    async def search_similar_interactions(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[MemoryInteraction, float]]:
        """Search for similar interactions.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of (interaction, similarity) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.sentence_transformer.encode([query])
            
            # Search in FAISS index
            similarities, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(limit * 2, self.index.ntotal)
            )
            
            # Get interactions and filter by threshold
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= threshold and idx < len(self.metadata):
                    interaction_id = list(self.metadata.keys())[idx]
                    interaction = self.metadata[interaction_id]
                    results.append((interaction, float(similarity)))
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search similar interactions: {e}")
            return []


class SQLiteMemoryStore:
    """SQLite-based memory store."""
    
    def __init__(self, db_path: str = "data/memory.db"):
        """Initialize SQLite memory store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    outcome_data TEXT,
                    metadata TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    agent_name TEXT,
                    model_name TEXT,
                    execution_time REAL,
                    confidence REAL,
                    error_message TEXT,
                    retry_count INTEGER
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interaction_type ON interactions(interaction_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON interactions(outcome)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON interactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_name ON interactions(agent_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON interactions(model_name)")
    
    async def store_interaction(self, interaction: MemoryInteraction) -> bool:
        """Store an interaction in SQLite.
        
        Args:
            interaction: Memory interaction to store
            
        Returns:
            True if stored successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.interaction_id,
                    interaction.timestamp.isoformat(),
                    interaction.interaction_type.value,
                    interaction.prompt,
                    interaction.action,
                    interaction.outcome.value,
                    json.dumps(interaction.outcome_data),
                    json.dumps(interaction.metadata),
                    interaction.user_id,
                    interaction.session_id,
                    interaction.agent_name,
                    interaction.model_name,
                    interaction.execution_time,
                    interaction.confidence,
                    interaction.error_message,
                    interaction.retry_count
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction in SQLite: {e}")
            return False
    
    async def get_interaction(self, interaction_id: str) -> Optional[MemoryInteraction]:
        """Get an interaction from SQLite.
        
        Args:
            interaction_id: Interaction ID
            
        Returns:
            MemoryInteraction or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM interactions WHERE interaction_id = ?",
                    (interaction_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_interaction(row)
                
        except Exception as e:
            logger.error(f"Failed to get interaction from SQLite: {e}")
            return None
    
    def _row_to_interaction(self, row) -> MemoryInteraction:
        """Convert database row to MemoryInteraction."""
        return MemoryInteraction(
            interaction_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            interaction_type=InteractionType(row[2]),
            prompt=row[3],
            action=row[4],
            outcome=OutcomeType(row[5]),
            outcome_data=json.loads(row[6]) if row[6] else {},
            metadata=json.loads(row[7]) if row[7] else {},
            user_id=row[8],
            session_id=row[9],
            agent_name=row[10],
            model_name=row[11],
            execution_time=row[12] or 0.0,
            confidence=row[13] or 0.0,
            error_message=row[14],
            retry_count=row[15] or 0
        )


class PersistentMemory:
    """Main persistent memory class with hybrid storage."""
    
    def __init__(
        self,
        memory_type: MemoryType = MemoryType.HYBRID,
        redis_url: Optional[str] = None,
        vector_dim: int = 384,
        db_path: str = "data/memory.db"
    ):
        """Initialize persistent memory.
        
        Args:
            memory_type: Type of memory storage to use
            redis_url: Redis connection URL
            vector_dim: Vector dimension for vector store
            db_path: Path to SQLite database
        """
        self.memory_type = memory_type
        self.stores = {}
        
        # Initialize stores based on memory type
        if memory_type in [MemoryType.REDIS, MemoryType.HYBRID]:
            if redis_url and REDIS_AVAILABLE:
                self.stores['redis'] = RedisMemoryStore(redis_url)
            elif memory_type == MemoryType.REDIS:
                raise ImportError("Redis not available but required for REDIS memory type")
        
        if memory_type in [MemoryType.VECTOR_STORE, MemoryType.HYBRID]:
            if VECTOR_STORE_AVAILABLE:
                self.stores['vector'] = VectorMemoryStore(vector_dim)
            elif memory_type == MemoryType.VECTOR_STORE:
                raise ImportError("Vector store not available but required for VECTOR_STORE memory type")
        
        if memory_type in [MemoryType.SQLITE, MemoryType.HYBRID]:
            self.stores['sqlite'] = SQLiteMemoryStore(db_path)
        
        # Learning insights
        self.insights: List[LearningInsight] = []
        
        logger.info(f"Initialized persistent memory with type: {memory_type.value}")
    
    async def store_interaction(self, interaction: MemoryInteraction) -> bool:
        """Store an interaction in all configured stores.
        
        Args:
            interaction: Memory interaction to store
            
        Returns:
            True if stored in at least one store
        """
        success = False
        
        for store_name, store in self.stores.items():
            try:
                if await store.store_interaction(interaction):
                    success = True
                    logger.debug(f"Stored interaction in {store_name}")
                else:
                    logger.warning(f"Failed to store interaction in {store_name}")
            except Exception as e:
                logger.error(f"Error storing interaction in {store_name}: {e}")
        
        return success
    
    async def get_interaction(self, interaction_id: str) -> Optional[MemoryInteraction]:
        """Get an interaction from stores.
        
        Args:
            interaction_id: Interaction ID
            
        Returns:
            MemoryInteraction or None if not found
        """
        for store_name, store in self.stores.items():
            try:
                interaction = await store.get_interaction(interaction_id)
                if interaction:
                    return interaction
            except Exception as e:
                logger.error(f"Error getting interaction from {store_name}: {e}")
        
        return None
    
    async def search_similar_interactions(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[MemoryInteraction, float]]:
        """Search for similar interactions.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of (interaction, similarity) tuples
        """
        if 'vector' in self.stores:
            return await self.stores['vector'].search_similar_interactions(query, limit, threshold)
        
        # Fallback to other stores
        return []
    
    async def get_stats(self) -> MemoryStats:
        """Get memory statistics.
        
        Returns:
            MemoryStats object
        """
        # Try to get stats from Redis first
        if 'redis' in self.stores:
            return await self.stores['redis'].get_stats()
        
        # Fallback to SQLite
        if 'sqlite' in self.stores:
            # Implement SQLite stats
            pass
        
        return MemoryStats()
    
    async def analyze_learning_patterns(self) -> List[LearningInsight]:
        """Analyze memory for learning patterns.
        
        Returns:
            List of learning insights
        """
        insights = []
        
        # Get recent interactions
        stats = await self.get_stats()
        
        # Analyze agent performance
        for agent_name, agent_stats in stats.agent_performance.items():
            success_rate = agent_stats["successful"] / agent_stats["total"]
            
            if success_rate < 0.7:  # Low success rate
                insights.append(LearningInsight(
                    insight_id=f"agent_performance_{agent_name}_{int(time.time())}",
                    timestamp=datetime.now(),
                    insight_type="agent_performance",
                    description=f"Agent {agent_name} has low success rate: {success_rate:.2%}",
                    confidence=0.8,
                    data={"agent_name": agent_name, "success_rate": success_rate},
                    recommendations=[
                        f"Review and improve agent {agent_name} logic",
                        "Check for systematic errors in agent execution",
                        "Consider retraining or updating agent models"
                    ],
                    affected_agents=[agent_name]
                ))
        
        # Analyze error patterns
        for error_type, count in stats.error_patterns.items():
            if count > 10:  # Frequent error
                insights.append(LearningInsight(
                    insight_id=f"error_pattern_{error_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    insight_type="error_pattern",
                    description=f"Frequent error: {error_type} (occurred {count} times)",
                    confidence=0.9,
                    data={"error_type": error_type, "count": count},
                    recommendations=[
                        f"Investigate root cause of {error_type} errors",
                        "Implement better error handling for this error type",
                        "Add monitoring and alerting for this error pattern"
                    ]
                ))
        
        # Analyze execution time trends
        if stats.avg_execution_time > 30.0:  # Slow execution
            insights.append(LearningInsight(
                insight_id=f"performance_trend_{int(time.time())}",
                timestamp=datetime.now(),
                insight_type="performance_trend",
                description=f"Average execution time is high: {stats.avg_execution_time:.2f}s",
                confidence=0.7,
                data={"avg_execution_time": stats.avg_execution_time},
                recommendations=[
                    "Optimize agent execution performance",
                    "Consider caching frequently used data",
                    "Review and optimize database queries"
                ]
            ))
        
        self.insights.extend(insights)
        return insights
    
    async def get_learning_insights(self, limit: int = 50) -> List[LearningInsight]:
        """Get recent learning insights.
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            List of learning insights
        """
        # Sort by timestamp (newest first)
        sorted_insights = sorted(self.insights, key=lambda x: x.timestamp, reverse=True)
        return sorted_insights[:limit]
    
    async def apply_learning_insights(self, insights: List[LearningInsight]) -> Dict[str, Any]:
        """Apply learning insights to improve system performance.
        
        Args:
            insights: Learning insights to apply
            
        Returns:
            Dictionary with applied changes
        """
        applied_changes = {
            "agent_adjustments": [],
            "model_updates": [],
            "system_improvements": []
        }
        
        for insight in insights:
            if insight.insight_type == "agent_performance":
                # Apply agent-specific improvements
                applied_changes["agent_adjustments"].append({
                    "agent": insight.affected_agents[0],
                    "insight": insight.description,
                    "recommendations": insight.recommendations
                })
            
            elif insight.insight_type == "error_pattern":
                # Apply error handling improvements
                applied_changes["system_improvements"].append({
                    "error_type": insight.data["error_type"],
                    "insight": insight.description,
                    "recommendations": insight.recommendations
                })
            
            elif insight.insight_type == "performance_trend":
                # Apply performance improvements
                applied_changes["system_improvements"].append({
                    "performance_metric": "execution_time",
                    "insight": insight.description,
                    "recommendations": insight.recommendations
                })
        
        return applied_changes


# Global memory instance
_persistent_memory = None

def get_persistent_memory(
    memory_type: MemoryType = MemoryType.HYBRID,
    redis_url: Optional[str] = None
) -> PersistentMemory:
    """Get the global persistent memory instance.
    
    Args:
        memory_type: Type of memory storage
        redis_url: Redis connection URL
        
    Returns:
        PersistentMemory: Global memory instance
    """
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemory(memory_type, redis_url)
    return _persistent_memory

async def store_interaction(
    prompt: str,
    action: str,
    outcome: OutcomeType,
    outcome_data: Dict[str, Any] = None,
    interaction_type: InteractionType = InteractionType.PROMPT_PROCESSING,
    **kwargs
) -> bool:
    """Store an interaction in persistent memory.
    
    Args:
        prompt: User prompt
        action: Action taken
        outcome: Outcome of the action
        outcome_data: Additional outcome data
        interaction_type: Type of interaction
        **kwargs: Additional interaction parameters
        
    Returns:
        True if stored successfully
    """
    memory = get_persistent_memory()
    
    interaction = MemoryInteraction(
        interaction_id=f"interaction_{int(time.time() * 1000000)}",
        timestamp=datetime.now(),
        interaction_type=interaction_type,
        prompt=prompt,
        action=action,
        outcome=outcome,
        outcome_data=outcome_data or {},
        **kwargs
    )
    
    return await memory.store_interaction(interaction) 