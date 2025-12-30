"""
AgentMemory: Persistent memory for agent decisions, outcomes, and history.
Enhanced with expiration logic and memory overflow prevention.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)


class MemoryChunk:
    """Represents a memory chunk with expiration and size tracking."""

    def __init__(
        self,
        data: Any,
        ttl_seconds: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        priority: int = 1,
    ):
        self.data = data
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.max_size_bytes = max_size_bytes
        self.priority = priority  # Higher priority = less likely to be evicted
        self.access_count = 0
        self._size_bytes = None

    def is_expired(self) -> bool:
        """Check if the memory chunk has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def update_access(self):
        """Update last access time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_size_bytes(self) -> int:
        """Get the size of the memory chunk in bytes."""
        if self._size_bytes is None:
            try:
                self._size_bytes = len(json.dumps(self.data).encode("utf-8"))
            except (TypeError, ValueError):
                self._size_bytes = 0
        return self._size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "max_size_bytes": self.max_size_bytes,
            "priority": self.priority,
            "access_count": self.access_count,
            "size_bytes": self.get_size_bytes(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryChunk":
        """Create MemoryChunk from dictionary."""
        chunk = cls(
            data=data["data"],
            ttl_seconds=data.get("ttl_seconds"),
            max_size_bytes=data.get("max_size_bytes"),
            priority=data.get("priority", 1),
        )
        chunk.created_at = datetime.fromisoformat(data["created_at"])
        chunk.last_accessed = datetime.fromisoformat(data["last_accessed"])
        chunk.access_count = data.get("access_count", 0)
        chunk._size_bytes = data.get("size_bytes")
        return chunk


class MemoryManager:
    """Manages memory chunks with expiration and overflow prevention."""

    def __init__(
        self,
        max_memory_mb: int = 100,
        max_entries: int = 10000,
        cleanup_interval_seconds: int = 300,
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.chunks: Dict[str, MemoryChunk] = {}
        self.lock = threading.RLock()
        self.last_cleanup = time.time()
        self.total_memory_usage = 0
        self.entry_count = 0

        # Statistics
        self.stats = {
            "total_evictions": 0,
            "expired_evictions": 0,
            "size_evictions": 0,
            "manual_evictions": 0,
            "total_accesses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def add(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        priority: int = 1,
    ) -> bool:
        """Add a memory chunk."""
        with self.lock:
            # Check if cleanup is needed
            self._maybe_cleanup()

            # Create chunk
            chunk = MemoryChunk(data, ttl_seconds, max_size_bytes, priority)
            chunk_size = chunk.get_size_bytes()

            # Check size limit
            if max_size_bytes and chunk_size > max_size_bytes:
                logger.warning(
                    f"Chunk size {chunk_size} exceeds max_size_bytes {max_size_bytes}"
                )
                return False

            # Check if we need to evict to make space
            if not self._can_fit_chunk(chunk_size):
                if not self._evict_to_make_space(chunk_size):
                    logger.warning(f"Cannot make space for chunk of size {chunk_size}")
                    return False

            # Add chunk
            self.chunks[key] = chunk
            self.total_memory_usage += chunk_size
            self.entry_count += 1

            return True

    def get(self, key: str) -> Optional[Any]:
        """Get a memory chunk."""
        with self.lock:
            self.stats["total_accesses"] += 1

            if key in self.chunks:
                chunk = self.chunks[key]

                # Check if expired
                if chunk.is_expired():
                    self._remove_chunk(key, "expired")
                    self.stats["cache_misses"] += 1
                    return None

                # Update access
                chunk.update_access()
                self.stats["cache_hits"] += 1
                return chunk.data

            self.stats["cache_misses"] += 1
            return None

    def remove(self, key: str) -> bool:
        """Remove a memory chunk."""
        with self.lock:
            return self._remove_chunk(key, "manual")

    def _remove_chunk(self, key: str, reason: str) -> bool:
        """Remove a chunk and update statistics."""
        if key in self.chunks:
            chunk = self.chunks[key]
            self.total_memory_usage -= chunk.get_size_bytes()
            self.entry_count -= 1
            del self.chunks[key]

            if reason == "expired":
                self.stats["expired_evictions"] += 1
            elif reason == "size":
                self.stats["size_evictions"] += 1
            elif reason == "manual":
                self.stats["manual_evictions"] += 1

            self.stats["total_evictions"] += 1
            return True
        return False

    def _can_fit_chunk(self, chunk_size: int) -> bool:
        """Check if a chunk can fit in memory."""
        return (
            self.total_memory_usage + chunk_size <= self.max_memory_bytes
            and self.entry_count < self.max_entries
        )

    def _evict_to_make_space(self, required_size: int) -> bool:
        """Evict chunks to make space for new chunk."""
        if required_size > self.max_memory_bytes:
            return False

        # Sort chunks by priority and access pattern (LRU-like)
        chunks_to_evict = []
        for key, chunk in self.chunks.items():
            if chunk.is_expired():
                chunks_to_evict.append((key, chunk, 0))  # Expired chunks first
            else:
                # Calculate eviction score (lower = more likely to evict)
                time_factor = (
                    datetime.now() - chunk.last_accessed
                ).total_seconds() / 3600  # hours
                access_factor = max(1, chunk.access_count)
                eviction_score = time_factor / (access_factor * chunk.priority)
                chunks_to_evict.append((key, chunk, eviction_score))

        # Sort by eviction score (ascending)
        chunks_to_evict.sort(key=lambda x: x[2])

        # Evict chunks until we have enough space
        freed_space = 0
        for key, chunk, _ in chunks_to_evict:
            if freed_space >= required_size:
                break

            freed_space += chunk.get_size_bytes()
            self._remove_chunk(key, "size")

        return freed_space >= required_size

    def _maybe_cleanup(self):
        """Perform cleanup if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval_seconds:
            self.cleanup()
            self.last_cleanup = current_time

    def cleanup(self) -> Dict[str, int]:
        """Clean up expired chunks and return statistics."""
        with self.lock:
            expired_keys = [
                key for key, chunk in self.chunks.items() if chunk.is_expired()
            ]

            for key in expired_keys:
                self._remove_chunk(key, "expired")

            return {
                "expired_removed": len(expired_keys),
                "total_chunks": len(self.chunks),
                "total_memory_mb": self.total_memory_usage / (1024 * 1024),
                "entry_count": self.entry_count,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        with self.lock:
            hit_rate = (
                self.stats["cache_hits"] / max(1, self.stats["total_accesses"])
            ) * 100

            return {
                **self.stats,
                "hit_rate_percent": hit_rate,
                "total_chunks": len(self.chunks),
                "total_memory_mb": self.total_memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_usage_percent": (
                    self.total_memory_usage / self.max_memory_bytes
                )
                * 100,
                "entry_count": self.entry_count,
                "max_entries": self.max_entries,
            }


class AgentMemory:
    """
    Persistent memory for all agents using agent_memory.json.
    Each agent has its own section for decisions, model scores, trade outcomes, and tuning history.
    Thread-safe and robust with expiration logic and memory overflow prevention.
    """

    def __init__(
        self,
        path: str = "agent_memory.json",
        max_memory_mb: int = 100,
        max_entries: int = 10000,
        cleanup_interval_seconds: int = 300,
    ):
        self.path = Path(path)
        self.lock_path = Path(f"{path}.lock")
        self.lock = FileLock(str(self.lock_path))

        # Memory manager for short-term chunks
        self.memory_manager = MemoryManager(
            max_memory_mb=max_memory_mb,
            max_entries=max_entries,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )

        # Default TTL settings
        self.default_ttls = {
            "short_term": 3600,  # 1 hour
            "medium_term": 86400,  # 1 day
            "long_term": 604800,  # 1 week
            "permanent": None,  # No expiration
        }

        # Initialize file if it doesn't exist
        if not self.path.exists():
            self.path.write_text(json.dumps({}))

    def _load(self) -> Dict[str, Any]:
        with self.lock:
            with open(self.path, "r") as f:
                return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        """Save memory data to file with robust error handling."""
        with self.lock:
            try:
                # Ensure directory exists
                self.path.parent.mkdir(parents=True, exist_ok=True)

                # Create backup if file exists
                if self.path.exists():
                    backup_path = self.path.with_suffix(".backup")
                    try:
                        import shutil

                        shutil.copy2(self.path, backup_path)
                        logger.debug(f"Created backup: {backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create backup: {e}")

                # Save with atomic write
                temp_path = self.path.with_suffix(".tmp")
                try:
                    with open(temp_path, "w") as f:
                        json.dump(data, f, indent=2)

                    # Atomic move
                    temp_path.replace(self.path)
                    logger.debug(f"Successfully saved memory to: {self.path}")

                except Exception as e:
                    # Clean up temp file
                    try:
                        temp_path.unlink()
                    except BaseException:
                        pass
                    raise e

            except PermissionError as e:
                logger.error(f"Permission error saving memory to {self.path}: {e}")
                raise
            except OSError as e:
                logger.error(f"OS error saving memory to {self.path}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error saving memory to {self.path}: {e}")
                raise

    def log_outcome(
        self,
        agent: str,
        run_type: str,
        outcome: Dict[str, Any],
        memory_type: str = "medium_term",
    ) -> Dict[str, Any]:
        """
        Log an outcome for an agent with memory type classification.
        Args:
            agent: Name of the agent (e.g., 'ModelBuilderAgent')
            run_type: Type of run (e.g., 'build', 'evaluate', 'tune', 'trade')
            outcome: Dict with details (must include 'model_id' or similar)
            memory_type: Type of memory ('short_term', 'medium_term', 'long_term', 'permanent')
        Returns:
            Dictionary with logging status and details
        """
        try:
            # Get TTL for memory type
            ttl_seconds = self.default_ttls.get(memory_type)

            # Create memory key
            memory_key = f"{agent}:{run_type}:{outcome.get('model_id', 'unknown')}"

            # Add to memory manager for short-term storage
            if memory_type in ["short_term", "medium_term"]:
                success = self.memory_manager.add(
                    key=memory_key,
                    data=outcome,
                    ttl_seconds=ttl_seconds,
                    priority=2 if memory_type == "short_term" else 1,
                )
                if not success:
                    logger.warning(f"Failed to add to memory manager: {memory_key}")

            # Also save to persistent storage for long-term/permanent
            if memory_type in ["long_term", "permanent"]:
                data = self._load()
                now = datetime.now().isoformat()
                agent_section = data.setdefault(agent, {})
                run_section = agent_section.setdefault(run_type, [])
                entry = {"timestamp": now, "memory_type": memory_type, **outcome}
                run_section.append(entry)

                # Keep only last 1000 entries per run_type for permanent storage
                if len(run_section) > 1000:
                    run_section[:] = run_section[-1000:]

                self._save(data)

            return {
                "success": True,
                "message": f"Outcome logged successfully for {agent}",
                "agent": agent,
                "run_type": run_type,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat(),
                "memory_key": memory_key,
            }

        except Exception as e:
            logger.error(f"Error logging outcome: {e}")
            return {
                "success": False,
                "message": f"Error logging outcome: {str(e)}",
                "agent": agent,
                "run_type": run_type,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat(),
            }

    def get_history(
        self,
        agent: str,
        run_type: Optional[str] = None,
        model_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve past outcomes for an agent, optionally filtered by run_type, model_id, and memory_type.
        Args:
            agent: Name of the agent
            run_type: Type of run (optional)
            model_id: Filter by model_id (optional)
            memory_type: Filter by memory type (optional)
        Returns:
            Dictionary with history data and status
        """
        try:
            results = []

            # Get from memory manager (short-term and medium-term)
            if memory_type is None or memory_type in ["short_term", "medium_term"]:
                for key, chunk in self.memory_manager.chunks.items():
                    if key.startswith(f"{agent}:"):
                        if run_type and f":{run_type}:" not in key:
                            continue
                        if model_id and f":{model_id}" not in key:
                            continue

                        chunk.update_access()
                        results.append(chunk.data)

            # Get from persistent storage (long-term and permanent)
            if memory_type is None or memory_type in ["long_term", "permanent"]:
                data = self._load().get(agent, {})
                if run_type:
                    runs = data.get(run_type, [])
                else:
                    # All run types
                    runs = []
                    for v in data.values():
                        if isinstance(v, list):
                            runs.extend(v)

                if model_id:
                    runs = [r for r in runs if r.get("model_id") == model_id]

                if memory_type:
                    runs = [r for r in runs if r.get("memory_type") == memory_type]

                results.extend(runs)

            return {
                "success": True,
                "message": f"History retrieved for {agent}",
                "agent": agent,
                "run_type": run_type,
                "model_id": model_id,
                "memory_type": memory_type,
                "history": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return {
                "success": False,
                "message": f"Error getting history: {str(e)}",
                "agent": agent,
                "run_type": run_type,
                "model_id": model_id,
                "memory_type": memory_type,
                "history": [],
                "count": 0,
                "timestamp": datetime.now().isoformat(),
            }

    def get_recent_performance(
        self,
        agent: str,
        run_type: str,
        metric: str,
        window: int = 10,
        memory_type: Optional[str] = None,
    ) -> dict:
        """
        Get recent values of a performance metric for trend analysis.
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key (e.g., 'sharpe_ratio')
            window: Number of most recent entries to consider
            memory_type: Filter by memory type (optional)
        Returns:
            Dictionary with performance data and status
        """
        try:
            history_result = self.get_history(agent, run_type, memory_type=memory_type)
            if not history_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to get history",
                    "agent": agent,
                    "run_type": run_type,
                    "metric": metric,
                    "memory_type": memory_type,
                    "timestamp": datetime.now().isoformat(),
                }

            history = history_result.get("history", [])
            values = [r.get(metric) for r in history if metric in r]
            recent_values = values[-window:]

            return {
                "success": True,
                "message": f"Recent performance retrieved for {agent}",
                "agent": agent,
                "run_type": run_type,
                "metric": metric,
                "memory_type": memory_type,
                "values": recent_values,
                "count": len(recent_values),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": agent,
                "run_type": run_type,
                "metric": metric,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat(),
            }

    def is_improving(
        self,
        agent: str,
        run_type: str,
        metric: str,
        window: int = 10,
        memory_type: Optional[str] = None,
    ) -> dict:
        """
        Detect if a metric is improving (increasing or decreasing, depending on metric).
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key
            window: Number of recent entries to consider
            memory_type: Filter by memory type (optional)
        Returns:
            Dictionary with improvement analysis and status
        """
        try:
            performance_result = self.get_recent_performance(
                agent, run_type, metric, window, memory_type
            )
            if not performance_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to get performance data",
                    "agent": agent,
                    "run_type": run_type,
                    "metric": metric,
                    "memory_type": memory_type,
                    "timestamp": datetime.now().isoformat(),
                }

            values = performance_result.get("values", [])
            if len(values) < 2:
                return {
                    "success": True,
                    "message": "Insufficient data for improvement analysis",
                    "agent": agent,
                    "run_type": run_type,
                    "metric": metric,
                    "memory_type": memory_type,
                    "is_improving": False,
                    "trend": "insufficient_data",
                    "values": values,
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate trend
            first_half = values[: len(values) // 2]
            second_half = values[len(values) // 2 :]

            if not first_half or not second_half:
                return {
                    "success": True,
                    "message": "Insufficient data for trend analysis",
                    "agent": agent,
                    "run_type": run_type,
                    "metric": metric,
                    "memory_type": memory_type,
                    "is_improving": False,
                    "trend": "insufficient_data",
                    "values": values,
                    "timestamp": datetime.now().isoformat(),
                }

            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            # Determine if higher or lower is better based on metric name
            higher_is_better = any(
                keyword in metric.lower()
                for keyword in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "sharpe",
                    "return",
                    "profit",
                ]
            )

            if higher_is_better:
                is_improving = second_avg > first_avg
                trend = "improving" if is_improving else "declining"
            else:
                is_improving = second_avg < first_avg
                trend = "improving" if is_improving else "declining"

            return {
                "success": True,
                "message": f"Improvement analysis completed for {agent}",
                "agent": agent,
                "run_type": run_type,
                "metric": metric,
                "memory_type": memory_type,
                "is_improving": is_improving,
                "trend": trend,
                "first_half_avg": first_avg,
                "second_half_avg": second_avg,
                "improvement_pct": (
                    ((second_avg - first_avg) / abs(first_avg)) * 100
                    if first_avg != 0
                    else 0
                ),
                "values": values,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": agent,
                "run_type": run_type,
                "metric": metric,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat(),
            }

    def clear(self, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear memory, optionally by type.
        Args:
            memory_type: Type of memory to clear ('short_term', 'medium_term', 'long_term', 'permanent', or None for all)
        Returns:
            Dictionary with clearing status and details
        """
        try:
            cleared_count = 0

            # Clear memory manager
            if memory_type is None or memory_type in ["short_term", "medium_term"]:
                # Clear all chunks from memory manager
                with self.memory_manager.lock:
                    cleared_count += len(self.memory_manager.chunks)
                    self.memory_manager.chunks.clear()
                    self.memory_manager.total_memory_usage = 0
                    self.memory_manager.entry_count = 0

            # Clear persistent storage
            if memory_type is None or memory_type in ["long_term", "permanent"]:
                if memory_type is None:
                    # Clear all persistent data
                    self._save({})
                    cleared_count += 1000  # Estimate
                else:
                    # Clear specific memory type from persistent storage
                    data = self._load()
                    for agent in data:
                        for run_type in data[agent]:
                            if isinstance(data[agent][run_type], list):
                                original_count = len(data[agent][run_type])
                                data[agent][run_type] = [
                                    entry
                                    for entry in data[agent][run_type]
                                    if entry.get("memory_type") != memory_type
                                ]
                                cleared_count += original_count - len(
                                    data[agent][run_type]
                                )
                    self._save(data)

            return {
                "success": True,
                "message": f"Memory cleared successfully",
                "memory_type": memory_type,
                "cleared_count": cleared_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return {
                "success": False,
                "message": f"Error clearing memory: {str(e)}",
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat(),
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get memory manager stats
            mm_stats = self.memory_manager.get_stats()

            # Get persistent storage stats
            data = self._load()
            persistent_stats = {
                "total_agents": len(data),
                "total_entries": sum(
                    len(entries)
                    for entries in data.values()
                    if isinstance(entries, list)
                ),
                "file_size_mb": (
                    self.path.stat().st_size / (1024 * 1024)
                    if self.path.exists()
                    else 0
                ),
            }

            return {
                "success": True,
                "memory_manager": mm_stats,
                "persistent_storage": persistent_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def cleanup_expired(self) -> Dict[str, Any]:
        """Clean up expired memory chunks."""
        try:
            # Clean memory manager
            mm_cleanup = self.memory_manager.cleanup()

            return {
                "success": True,
                "memory_manager_cleanup": mm_cleanup,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error cleaning up expired memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
