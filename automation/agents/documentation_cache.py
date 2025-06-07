import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import redis
import aioredis
from dataclasses import dataclass
import asyncio
from functools import lru_cache
import mmh3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict
    hash: str
    size: int
    hits: int
    last_accessed: datetime

class DocumentationCache:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.cache_config = config.get('documentation', {}).get('cache', {})
        self.setup_redis()
        self.setup_memory_cache()
        self.setup_search_index()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }

    def setup_logging(self):
        """Configure logging for the documentation cache system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "cache.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_redis(self):
        """Setup Redis connection for distributed caching."""
        redis_config = self.cache_config.get('redis', {})
        self.redis = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=True
        )
        self.redis_async = aioredis.from_url(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 0)}"
        )

    def setup_memory_cache(self):
        """Setup in-memory cache with LRU eviction."""
        self.memory_cache = {}
        self.max_memory_size = self.cache_config.get('memory', {}).get('max_size', 1000)
        self.memory_ttl = self.cache_config.get('memory', {}).get('ttl', 3600)

    def setup_search_index(self):
        """Setup search index for cached content."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.document_vectors = {}
        self.document_texts = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback strategy."""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.expires_at or entry.expires_at > datetime.now():
                    self.cache_stats['hits'] += 1
                    entry.hits += 1
                    entry.last_accessed = datetime.now()
                    return entry.value
                else:
                    del self.memory_cache[key]
                    self.cache_stats['evictions'] += 1

            # Try Redis cache
            value = await self.redis_async.get(key)
            if value:
                self.cache_stats['hits'] += 1
                # Update memory cache
                entry = CacheEntry(
                    key=key,
                    value=pickle.loads(value),
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=self.memory_ttl),
                    metadata={},
                    hash=hashlib.md5(value).hexdigest(),
                    size=len(value),
                    hits=1,
                    last_accessed=datetime.now()
                )
                self._update_memory_cache(key, entry)
                return entry.value

            self.cache_stats['misses'] += 1
            return None

        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """Set value in cache with metadata."""
        try:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None,
                metadata=metadata or {},
                hash=hashlib.md5(pickle.dumps(value)).hexdigest(),
                size=len(pickle.dumps(value)),
                hits=0,
                last_accessed=datetime.now()
            )

            # Update memory cache
            self._update_memory_cache(key, entry)

            # Update Redis cache
            await self.redis_async.set(
                key,
                pickle.dumps(value),
                ex=ttl
            )

            # Update search index if value is text
            if isinstance(value, str):
                self._update_search_index(key, value)

            self.cache_stats['size'] += entry.size

        except Exception as e:
            self.logger.error(f"Cache set error: {str(e)}")

    def _update_memory_cache(self, key: str, entry: CacheEntry):
        """Update memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_size:
            # Evict least recently used entry
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            del self.memory_cache[lru_key]
            self.cache_stats['evictions'] += 1

        self.memory_cache[key] = entry

    def _update_search_index(self, key: str, text: str):
        """Update search index with new document."""
        self.document_texts[key] = text
        if not self.document_vectors:
            self.document_vectors = self.vectorizer.fit_transform([text])
        else:
            self.document_vectors = self.vectorizer.fit_transform(
                list(self.document_texts.values())
            )

    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search cached content using TF-IDF and cosine similarity."""
        try:
            if not self.document_vectors:
                return []

            # Transform query
            query_vector = self.vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]

            # Get top results
            top_indices = np.argsort(similarities)[-limit:][::-1]
            results = []

            for idx in top_indices:
                if similarities[idx] > 0:
                    key = list(self.document_texts.keys())[idx]
                    results.append({
                        'key': key,
                        'score': float(similarities[idx]),
                        'text': self.document_texts[key]
                    })

            return results

        except Exception as e:
            self.logger.error(f"Cache search error: {str(e)}")
            return []

    async def delete(self, key: str):
        """Delete value from cache."""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                self.cache_stats['size'] -= self.memory_cache[key].size
                del self.memory_cache[key]

            # Remove from Redis cache
            await self.redis_async.delete(key)

            # Remove from search index
            if key in self.document_texts:
                del self.document_texts[key]
                if self.document_texts:
                    self.document_vectors = self.vectorizer.fit_transform(
                        list(self.document_texts.values())
                    )
                else:
                    self.document_vectors = {}

        except Exception as e:
            self.logger.error(f"Cache delete error: {str(e)}")

    async def clear(self):
        """Clear all caches."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.cache_stats['size'] = 0

            # Clear Redis cache
            await self.redis_async.flushdb()

            # Clear search index
            self.document_texts.clear()
            self.document_vectors = {}

        except Exception as e:
            self.logger.error(f"Cache clear error: {str(e)}")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'size': self.cache_stats['size'],
            'memory_entries': len(self.memory_cache),
            'search_entries': len(self.document_texts)
        }

    async def cleanup(self):
        """Cleanup expired entries."""
        try:
            # Cleanup memory cache
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.expires_at and entry.expires_at <= now
            ]
            for key in expired_keys:
                self.cache_stats['size'] -= self.memory_cache[key].size
                del self.memory_cache[key]
                self.cache_stats['evictions'] += 1

            # Redis handles expiration automatically

        except Exception as e:
            self.logger.error(f"Cache cleanup error: {str(e)}")

    @lru_cache(maxsize=1000)
    def get_cached_value(self, key: str) -> Optional[Any]:
        """Get value from LRU cache."""
        return self.memory_cache.get(key)

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return sum(entry.size for entry in self.memory_cache.values())

    def get_redis_usage(self) -> int:
        """Get current Redis usage in bytes."""
        return self.redis.info()['used_memory']

    async def prefetch(self, keys: List[str]):
        """Prefetch multiple keys into memory cache."""
        try:
            for key in keys:
                value = await self.get(key)
                if value:
                    await self.set(key, value)
        except Exception as e:
            self.logger.error(f"Cache prefetch error: {str(e)}")

    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            results = {}
            for key in keys:
                value = await self.get(key)
                if value:
                    results[key] = value
            return results
        except Exception as e:
            self.logger.error(f"Cache batch get error: {str(e)}")
            return {}

    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None):
        """Set multiple values in cache."""
        try:
            for key, value in items.items():
                await self.set(key, value, ttl)
        except Exception as e:
            self.logger.error(f"Cache batch set error: {str(e)}")

    def get_hot_keys(self, limit: int = 10) -> List[Dict]:
        """Get most frequently accessed keys."""
        try:
            hot_keys = sorted(
                self.memory_cache.values(),
                key=lambda x: x.hits,
                reverse=True
            )[:limit]
            return [
                {
                    'key': entry.key,
                    'hits': entry.hits,
                    'last_accessed': entry.last_accessed.isoformat(),
                    'size': entry.size
                }
                for entry in hot_keys
            ]
        except Exception as e:
            self.logger.error(f"Cache hot keys error: {str(e)}")
            return [] 