"""
State Manager

Thread-safe state management with version control, concurrent write protection,
and memory optimization capabilities.
"""

import asyncio
import gzip
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import filelock
import psutil

logger = logging.getLogger(__name__)


class StateVersion:
    """State version information."""
    
    CURRENT_VERSION = "1.0.0"
    COMPATIBLE_VERSIONS = ["1.0.0"]
    
    def __init__(self, version: str = CURRENT_VERSION):
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateVersion':
        """Create from dictionary."""
        version = cls(data.get("version", cls.CURRENT_VERSION))
        if "created_at" in data:
            version.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            version.updated_at = datetime.fromisoformat(data["updated_at"])
        return version


class StateManager:
    """
    Thread-safe state manager with version control and memory optimization.
    
    Features:
    - Version header validation on load
    - Thread-safe concurrent writes using filelock
    - Memory compression and cleanup
    - Automatic backup creation
    - State validation and recovery
    """
    
    def __init__(self, 
                 state_file: str = "data/state.pkl",
                 max_memory_mb: int = 100,
                 compression_threshold_mb: int = 10,
                 backup_count: int = 5):
        """Initialize the state manager."""
        self.state_file = Path(state_file)
        self.max_memory_mb = max_memory_mb
        self.compression_threshold_mb = compression_threshold_mb
        self.backup_count = backup_count
        
        # Thread safety
        self._lock = threading.RLock()
        self._file_lock = None
        
        # State storage
        self._state: Dict[str, Any] = {}
        self._version = StateVersion()
        self._dirty = False
        self._last_save = None
        
        # Memory tracking
        self._memory_usage = 0
        self._last_cleanup = datetime.now()
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file lock
        self._init_file_lock()
        
        # Load existing state
        self._load_state()
        
        logger.info(f"StateManager initialized: {self.state_file}")
    
    def _init_file_lock(self):
        """Initialize file lock for concurrent access protection."""
        lock_file = self.state_file.with_suffix('.lock')
        self._file_lock = filelock.FileLock(str(lock_file), timeout=30)
    
    def _load_state(self):
        """Load state from file with version validation."""
        if not self.state_file.exists():
            logger.info("No existing state file found, starting fresh")
            return
        
        try:
            with self._file_lock:
                with open(self.state_file, 'rb') as f:
                    # Check if file is gzipped
                    magic = f.read(2)
                    f.seek(0)
                    
                    if magic.startswith(b'\x1f\x8b'):
                        # Gzipped file
                        with gzip.open(f, 'rb') as gz:
                            data = pickle.load(gz)
                    else:
                        # Regular pickle file
                        data = pickle.load(f)
                
                # Validate version
                if isinstance(data, dict) and 'version' in data:
                    version_data = data['version']
                    if isinstance(version_data, dict):
                        self._version = StateVersion.from_dict(version_data)
                    else:
                        # Legacy version format
                        self._version = StateVersion(str(version_data))
                    
                    # Check compatibility
                    if self._version.version not in StateVersion.COMPATIBLE_VERSIONS:
                        logger.warning(f"State version {self._version.version} may not be compatible")
                    
                    # Load state data
                    self._state = data.get('state', {})
                    self._last_save = datetime.fromisoformat(data.get('last_save', datetime.now().isoformat()))
                    
                    logger.info(f"State loaded successfully: version {self._version.version}")
                else:
                    # Legacy format without version
                    self._state = data
                    logger.warning("Loaded legacy state format without version")
                    
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            # Try to load backup
            self._load_backup()
    
    def _load_backup(self):
        """Load state from backup file."""
        backup_file = self.state_file.with_suffix('.backup')
        if backup_file.exists():
            try:
                with open(backup_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and 'state' in data:
                    self._state = data['state']
                    logger.info("State loaded from backup")
                else:
                    self._state = data
                    logger.info("Legacy state loaded from backup")
                    
            except Exception as e:
                logger.error(f"Failed to load backup: {e}")
                self._state = {}
        else:
            logger.warning("No backup available, starting with empty state")
            self._state = {}
    
    def _save_state(self, force: bool = False):
        """Save state to file with version header."""
        if not self._dirty and not force:
            return
        
        try:
            with self._file_lock:
                # Create backup before saving
                if self.state_file.exists():
                    self._create_backup()
                
                # Prepare data with version header
                data = {
                    'version': self._version.to_dict(),
                    'state': self._state,
                    'last_save': datetime.now().isoformat()
                }
                
                # Update version timestamp
                self._version.updated_at = datetime.now()
                
                # Determine if compression is needed
                temp_data = pickle.dumps(data)
                data_size_mb = len(temp_data) / (1024 * 1024)
                
                if data_size_mb > self.compression_threshold_mb:
                    # Use compression
                    with gzip.open(self.state_file, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"State saved with compression: {data_size_mb:.2f}MB")
                else:
                    # No compression
                    with open(self.state_file, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"State saved: {data_size_mb:.2f}MB")
                
                self._dirty = False
                self._last_save = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    def _create_backup(self):
        """Create backup of current state file."""
        try:
            backup_file = self.state_file.with_suffix('.backup')
            if backup_file.exists():
                # Rotate backups
                for i in range(self.backup_count - 1, 0, -1):
                    old_backup = backup_file.with_suffix(f'.backup.{i}')
                    new_backup = backup_file.with_suffix(f'.backup.{i + 1}')
                    if old_backup.exists():
                        old_backup.rename(new_backup)
                
                # Move current backup
                backup_file.rename(backup_file.with_suffix('.backup.1'))
            
            # Create new backup
            import shutil
            shutil.copy2(self.state_file, backup_file)
            logger.debug("Backup created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        with self._lock:
            self._state[key] = value
            self._dirty = True
            self._update_memory_usage()
    
    def delete(self, key: str) -> bool:
        """Delete key from state."""
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._dirty = True
                self._update_memory_usage()
                return True
            return False
    
    def has_key(self, key: str) -> bool:
        """Check if key exists in state."""
        with self._lock:
            return key in self._state
    
    def keys(self) -> list:
        """Get all keys in state."""
        with self._lock:
            return list(self._state.keys())
    
    def values(self) -> list:
        """Get all values in state."""
        with self._lock:
            return list(self._state.values())
    
    def items(self) -> list:
        """Get all key-value pairs in state."""
        with self._lock:
            return list(self._state.items())
    
    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._state.clear()
            self._dirty = True
            self._memory_usage = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update state with dictionary."""
        with self._lock:
            self._state.update(data)
            self._dirty = True
            self._update_memory_usage()
    
    def _update_memory_usage(self):
        """Update memory usage tracking."""
        try:
            # Estimate memory usage
            import sys
            self._memory_usage = sys.getsizeof(self._state)
            
            # Check if cleanup is needed
            if self._memory_usage > self.max_memory_mb * 1024 * 1024:
                self._cleanup_memory()
                
        except Exception as e:
            logger.warning(f"Failed to update memory usage: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory by compressing or removing old data."""
        try:
            current_time = datetime.now()
            
            # Only cleanup if enough time has passed
            if (current_time - self._last_cleanup).total_seconds() < 300:  # 5 minutes
                return
            
            logger.info("Starting memory cleanup...")
            
            # Get system memory info
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < 100:  # Less than 100MB available
                # Aggressive cleanup
                self._aggressive_cleanup()
            else:
                # Normal cleanup
                self._normal_cleanup()
            
            self._last_cleanup = current_time
            self._update_memory_usage()
            
            logger.info(f"Memory cleanup completed. Available: {available_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def _normal_cleanup(self):
        """Normal memory cleanup - compress old data."""
        with self._lock:
            # Find old data (older than 24 hours)
            cutoff_time = datetime.now().timestamp() - 86400
            
            for key, value in list(self._state.items()):
                if isinstance(value, dict) and 'timestamp' in value:
                    if value['timestamp'] < cutoff_time:
                        # Compress old data
                        if 'data' in value and not value.get('compressed', False):
                            try:
                                compressed_data = gzip.compress(pickle.dumps(value['data']))
                                value['data'] = compressed_data
                                value['compressed'] = True
                                logger.debug(f"Compressed old data for key: {key}")
                            except Exception as e:
                                logger.warning(f"Failed to compress data for {key}: {e}")
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup - remove old data."""
        with self._lock:
            # Find old data (older than 1 hour)
            cutoff_time = datetime.now().timestamp() - 3600
            
            keys_to_remove = []
            
            for key, value in list(self._state.items()):
                if isinstance(value, dict) and 'timestamp' in value:
                    if value['timestamp'] < cutoff_time:
                        keys_to_remove.append(key)
            
            # Remove old data
            for key in keys_to_remove:
                del self._state[key]
                logger.info(f"Removed old data: {key}")
    
    def save(self, force: bool = False) -> None:
        """Save state to file."""
        self._save_state(force=force)
    
    def load(self) -> None:
        """Reload state from file."""
        self._load_state()
    
    def get_version(self) -> StateVersion:
        """Get current state version."""
        return self._version
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        with self._lock:
            return {
                'estimated_bytes': self._memory_usage,
                'estimated_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_mb,
                'last_cleanup': self._last_cleanup.isoformat() if self._last_cleanup else None,
                'dirty': self._dirty,
                'last_save': self._last_save.isoformat() if self._last_save else None
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        with self._lock:
            return {
                'total_keys': len(self._state),
                'memory_usage': self.get_memory_usage(),
                'version': self._version.to_dict(),
                'file_size_mb': self.state_file.stat().st_size / (1024 * 1024) if self.state_file.exists() else 0
            }
    
    def compress_state(self) -> None:
        """Manually compress state data."""
        with self._lock:
            logger.info("Starting manual state compression...")
            
            for key, value in list(self._state.items()):
                if isinstance(value, dict) and 'data' in value and not value.get('compressed', False):
                    try:
                        compressed_data = gzip.compress(pickle.dumps(value['data']))
                        value['data'] = compressed_data
                        value['compressed'] = True
                        logger.debug(f"Compressed data for key: {key}")
                    except Exception as e:
                        logger.warning(f"Failed to compress data for {key}: {e}")
            
            self._dirty = True
            logger.info("Manual state compression completed")
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Clean up data older than specified hours. Returns number of items removed."""
        with self._lock:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            keys_to_remove = []
            
            for key, value in list(self._state.items()):
                if isinstance(value, dict) and 'timestamp' in value:
                    if value['timestamp'] < cutoff_time:
                        keys_to_remove.append(key)
            
            # Remove old data
            for key in keys_to_remove:
                del self._state[key]
            
            self._dirty = True
            logger.info(f"Cleaned up {len(keys_to_remove)} old data items")
            return len(keys_to_remove)


# Global state manager instance
_global_state_manager: Optional[StateManager] = None


def get_state_manager(state_file: str = "data/state.pkl") -> StateManager:
    """Get global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager(state_file)
    return _global_state_manager


# Convenience functions
def get_state(key: str, default: Any = None) -> Any:
    """Get value from global state."""
    return get_state_manager().get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set value in global state."""
    get_state_manager().set(key, value)


def save_state(force: bool = False) -> None:
    """Save global state."""
    get_state_manager().save(force)


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("🔧 State Manager Demo")
        print("=" * 50)
        
        # Create state manager
        state_manager = StateManager("demo_state.pkl")
        
        # Set some data
        print("\n📝 Setting test data...")
        state_manager.set("test_key", "test_value")
        state_manager.set("timestamp", datetime.now().isoformat())
        state_manager.set("large_data", {"data": "x" * 1000, "timestamp": time.time()})
        
        # Get data
        print(f"Test key: {state_manager.get('test_key')}")
        print(f"Has timestamp: {state_manager.has_key('timestamp')}")
        
        # Get stats
        stats = state_manager.get_stats()
        print(f"\n📊 Stats: {stats}")
        
        # Save state
        print("\n💾 Saving state...")
        state_manager.save()
        
        # Clean up demo file
        if os.path.exists("demo_state.pkl"):
            os.remove("demo_state.pkl")
        
        print("✅ Demo completed!")
    
    asyncio.run(demo()) 