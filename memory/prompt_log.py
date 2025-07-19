"""
Prompt Memory Module

This module provides a flexible prompt memory system that can store and retrieve
prompt interactions using either JSON file storage or Redis backend.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PromptEntry:
    """A single prompt interaction entry."""
    
    prompt: str
    result: Dict[str, Any]
    timestamp: str
    session_id: str
    user_id: str
    agent_type: str
    execution_time: float
    success: bool
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


from abc import ABC, abstractmethod

class PromptMemoryBackend(ABC):
    """Abstract base class for prompt memory backends."""
    
    @abstractmethod
    async def log_prompt(self, entry: PromptEntry) -> bool:
        """Log a prompt entry."""
        pass
    
    @abstractmethod
    async def get_last_prompt(self, user_id: str = "default") -> Optional[PromptEntry]:
        """Get the last prompt for a user."""
        pass
    
    @abstractmethod
    async def get_prompt_history(self, user_id: str = "default", n: int = 10) -> List[PromptEntry]:
        """Get prompt history for a user."""
        pass
    
    @abstractmethod
    async def clear_history(self, user_id: str = "default") -> bool:
        """Clear prompt history for a user."""
        pass


class JSONPromptMemory(PromptMemoryBackend):
    """JSON file-based prompt memory backend."""
    
    def __init__(self, file_path: str = "memory/prompt_history.json"):
        """Initialize JSON prompt memory.
        
        Args:
            file_path: Path to the JSON file for storage
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
        # Initialize file if it doesn't exist
        if not self.file_path.exists():
            self._save_data({})
        
        logger.info(f"JSON prompt memory initialized at {self.file_path}")
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to JSON file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving prompt memory: {e}")
    
    async def log_prompt(self, entry: PromptEntry) -> bool:
        """Log a prompt entry to JSON file."""
        async with self._lock:
            try:
                data = self._load_data()
                
                # Initialize user data if not exists
                if entry.user_id not in data:
                    data[entry.user_id] = []
                
                # Add entry
                data[entry.user_id].append(entry.to_dict())
                
                # Keep only last 1000 entries per user to prevent file bloat
                if len(data[entry.user_id]) > 1000:
                    data[entry.user_id] = data[entry.user_id][-1000:]
                
                self._save_data(data)
                logger.debug(f"Logged prompt for user {entry.user_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error logging prompt to JSON: {e}")
                return False
    
    async def get_last_prompt(self, user_id: str = "default") -> Optional[PromptEntry]:
        """Get the last prompt for a user."""
        try:
            data = self._load_data()
            user_data = data.get(user_id, [])
            
            if not user_data:
                return None
            
            last_entry = user_data[-1]
            return PromptEntry(**last_entry)
            
        except Exception as e:
            logger.error(f"Error getting last prompt: {e}")
            return None
    
    async def get_prompt_history(self, user_id: str = "default", n: int = 10) -> List[PromptEntry]:
        """Get prompt history for a user."""
        try:
            data = self._load_data()
            user_data = data.get(user_id, [])
            
            # Get last n entries
            recent_entries = user_data[-n:] if len(user_data) > n else user_data
            
            return [PromptEntry(**entry) for entry in recent_entries]
            
        except Exception as e:
            logger.error(f"Error getting prompt history: {e}")
            return []
    
    async def clear_history(self, user_id: str = "default") -> bool:
        """Clear prompt history for a user."""
        async with self._lock:
            try:
                data = self._load_data()
                if user_id in data:
                    del data[user_id]
                    self._save_data(data)
                    logger.info(f"Cleared prompt history for user {user_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error clearing prompt history: {e}")
                return False


class RedisPromptMemory(PromptMemoryBackend):
    """Redis-based prompt memory backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        """Initialize Redis prompt memory.
        
        Args:
            redis_url: Redis connection URL
            db: Redis database number
        """
        self.redis_url = redis_url
        self.db = db
        self._redis = None
        self._initialized = False
        
        logger.info(f"Redis prompt memory initialized with URL: {redis_url}")
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url, db=self.db)
                await self._redis.ping()
                self._initialized = True
                logger.info("Redis connection established")
            except ImportError:
                logger.error("Redis not available. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis
    
    async def log_prompt(self, entry: PromptEntry) -> bool:
        """Log a prompt entry to Redis."""
        try:
            redis_client = await self._get_redis()
            
            # Create key for user's prompt list
            key = f"prompt_history:{entry.user_id}"
            
            # Add entry to list (left push to maintain order)
            await redis_client.lpush(key, json.dumps(entry.to_dict()))
            
            # Trim list to keep only last 1000 entries
            await redis_client.ltrim(key, 0, 999)
            
            # Set expiration (30 days)
            await redis_client.expire(key, 30 * 24 * 60 * 60)
            
            logger.debug(f"Logged prompt to Redis for user {entry.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging prompt to Redis: {e}")
            return False
    
    async def get_last_prompt(self, user_id: str = "default") -> Optional[PromptEntry]:
        """Get the last prompt for a user."""
        try:
            redis_client = await self._get_redis()
            key = f"prompt_history:{user_id}"
            
            # Get first (most recent) entry
            result = await redis_client.lindex(key, 0)
            
            if result is None:
                return None
            
            entry_data = json.loads(result)
            return PromptEntry(**entry_data)
            
        except Exception as e:
            logger.error(f"Error getting last prompt from Redis: {e}")
            return None
    
    async def get_prompt_history(self, user_id: str = "default", n: int = 10) -> List[PromptEntry]:
        """Get prompt history for a user."""
        try:
            redis_client = await self._get_redis()
            key = f"prompt_history:{user_id}"
            
            # Get last n entries
            results = await redis_client.lrange(key, 0, n - 1)
            
            entries = []
            for result in results:
                entry_data = json.loads(result)
                entries.append(PromptEntry(**entry_data))
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting prompt history from Redis: {e}")
            return []
    
    async def clear_history(self, user_id: str = "default") -> bool:
        """Clear prompt history for a user."""
        try:
            redis_client = await self._get_redis()
            key = f"prompt_history:{user_id}"
            
            await redis_client.delete(key)
            logger.info(f"Cleared prompt history for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing prompt history from Redis: {e}")
            return False


class PromptMemory:
    """Main prompt memory interface."""
    
    def __init__(self, backend: str = "json", **kwargs):
        """Initialize prompt memory.
        
        Args:
            backend: Backend type ("json" or "redis")
            **kwargs: Backend-specific configuration
        """
        self.backend_type = backend.lower()
        
        if self.backend_type == "json":
            self.backend = JSONPromptMemory(**kwargs)
        elif self.backend_type == "redis":
            self.backend = RedisPromptMemory(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'json' or 'redis'")
        
        logger.info(f"Prompt memory initialized with {self.backend_type} backend")
    
    async def log_prompt(
        self,
        prompt: str,
        result: Dict[str, Any],
        session_id: str = "default",
        user_id: str = "default",
        agent_type: str = "unknown",
        execution_time: float = 0.0,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a prompt interaction.
        
        Args:
            prompt: The user prompt
            result: The result/response
            session_id: Session identifier
            user_id: User identifier
            agent_type: Type of agent that processed the prompt
            execution_time: Time taken to process the prompt
            success: Whether the prompt was processed successfully
            metadata: Additional metadata
            
        Returns:
            bool: True if logged successfully
        """
        try:
            entry = PromptEntry(
                prompt=prompt,
                result=result,
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                user_id=user_id,
                agent_type=agent_type,
                execution_time=execution_time,
                success=success,
                metadata=metadata or {}
            )
            
            return await self.backend.log_prompt(entry)
            
        except Exception as e:
            logger.error(f"Error logging prompt: {e}")
            return False
    
    async def get_last_prompt(self, user_id: str = "default") -> Optional[PromptEntry]:
        """Get the last prompt for a user."""
        return await self.backend.get_last_prompt(user_id)
    
    async def get_prompt_history(self, user_id: str = "default", n: int = 10) -> List[PromptEntry]:
        """Get prompt history for a user."""
        return await self.backend.get_prompt_history(user_id, n)
    
    async def clear_history(self, user_id: str = "default") -> bool:
        """Clear prompt history for a user."""
        return await self.backend.clear_history(user_id)
    
    async def get_statistics(self, user_id: str = "default") -> Dict[str, Any]:
        """Get prompt statistics for a user."""
        try:
            history = await self.get_prompt_history(user_id, n=1000)
            
            if not history:
                return {
                    "total_prompts": 0,
                    "successful_prompts": 0,
                    "average_execution_time": 0.0,
                    "most_common_agent": None
                }
            
            total_prompts = len(history)
            successful_prompts = sum(1 for entry in history if entry.success)
            avg_execution_time = sum(entry.execution_time for entry in history) / total_prompts
            
            # Count agent types
            agent_counts = {}
            for entry in history:
                agent_counts[entry.agent_type] = agent_counts.get(entry.agent_type, 0) + 1
            
            most_common_agent = max(agent_counts.items(), key=lambda x: x[1])[0] if agent_counts else None
            
            return {
                "total_prompts": total_prompts,
                "successful_prompts": successful_prompts,
                "success_rate": successful_prompts / total_prompts if total_prompts > 0 else 0.0,
                "average_execution_time": avg_execution_time,
                "most_common_agent": most_common_agent,
                "agent_distribution": agent_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt statistics: {e}")
            return {}


# Global prompt memory instance
_prompt_memory: Optional[PromptMemory] = None


def get_prompt_memory(backend: str = "json", **kwargs) -> PromptMemory:
    """
    Get the global prompt memory instance.
    
    Args:
        backend: Backend type ("json" or "redis")
        **kwargs: Backend-specific configuration
        
    Returns:
        PromptMemory: Global prompt memory instance
    """
    global _prompt_memory
    if _prompt_memory is None:
        _prompt_memory = PromptMemory(backend, **kwargs)
    return _prompt_memory


# Convenience functions
async def log_prompt(
    prompt: str,
    result: Dict[str, Any],
    **kwargs
) -> bool:
    """Convenience function to log a prompt."""
    memory = get_prompt_memory()
    return await memory.log_prompt(prompt, result, **kwargs)


async def get_last_prompt(user_id: str = "default") -> Optional[PromptEntry]:
    """Convenience function to get the last prompt."""
    memory = get_prompt_memory()
    return await memory.get_last_prompt(user_id)


async def get_prompt_history(user_id: str = "default", n: int = 10) -> List[PromptEntry]:
    """Convenience function to get prompt history."""
    memory = get_prompt_memory()
    return await memory.get_prompt_history(user_id, n) 