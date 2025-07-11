"""
History Logger with automatic cleanup functionality.

This module provides a history logger that automatically manages log files
to prevent disk space overflow and maintain system performance.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import shutil
import glob
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetentionPolicy(Enum):
    """Retention policy types."""
    BY_AGE = "by_age"
    BY_SIZE = "by_size"
    BY_COUNT = "by_count"
    HYBRID = "hybrid"

@dataclass
class RetentionConfig:
    """Configuration for log retention."""
    policy: RetentionPolicy = RetentionPolicy.HYBRID
    max_age_days: int = 30
    max_size_mb: int = 1000
    max_file_count: int = 1000
    cleanup_interval_hours: int = 24
    compress_old_logs: bool = True
    archive_old_logs: bool = False
    archive_dir: Optional[str] = None

class HistoryLogger:
    """History logger with automatic cleanup and rotation."""
    
    def __init__(self, 
                 log_dir: str = "logs/history",
                 retention_config: Optional[RetentionConfig] = None,
                 enable_auto_cleanup: bool = True):
        """Initialize the history logger.
        
        Args:
            log_dir: Directory to store log files
            retention_config: Retention configuration
            enable_auto_cleanup: Whether to enable automatic cleanup
        """
        self.log_dir = Path(log_dir)
        self.retention_config = retention_config or RetentionConfig()
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.db_path = self.log_dir / "history.db"
        self._init_database()
        
        # Initialize cleanup
        self._cleanup_old_logs()
        self._cleanup_by_size()
        
        # Start auto-cleanup thread if enabled
        self._cleanup_thread = None
        if self.enable_auto_cleanup:
            self._start_auto_cleanup()
    
    def log_event(self, event_type: str, data: Dict[str, Any], 
                  timestamp: Optional[datetime] = None) -> str:
        """Log an event to history.
        
        Args:
            event_type: Type of event (e.g., 'trade', 'model_update', 'error')
            data: Event data
            timestamp: Optional timestamp (uses current time if not provided)
            
        Returns:
            Log file path
        """
        try:
            timestamp = timestamp or datetime.now()
            filename = f"{event_type}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.json"
            filepath = self.log_dir / filename
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'event_type': event_type,
                'data': data
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
            
            logger.debug(f"Logged event {event_type} to {filepath}")
            
            # Periodic cleanup
            if self._should_cleanup():
                self._cleanup_old_logs()
                self._cleanup_by_size()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return ""
    
    def get_recent_events(self, event_type: Optional[str] = None, 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events from history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        try:
            events = []
            pattern = f"{event_type}_*.json" if event_type else "*.json"
            
            for filepath in sorted(self.log_dir.glob(pattern), reverse=True):
                if len(events) >= limit:
                    break
                    
                try:
                    with open(filepath, 'r') as f:
                        event = json.load(f)
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Error reading log file {filepath}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    def get_events_by_date_range(self, start_date: datetime, 
                                end_date: datetime,
                                event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get events within a date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            event_type: Filter by event type (optional)
            
        Returns:
            List of event dictionaries
        """
        try:
            events = []
            pattern = f"{event_type}_*.json" if event_type else "*.json"
            
            for filepath in self.log_dir.glob(pattern):
                try:
                    # Extract timestamp from filename
                    filename = filepath.stem
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        
                        if start_date <= file_timestamp <= end_date:
                            with open(filepath, 'r') as f:
                                event = json.load(f)
                                events.append(event)
                except Exception as e:
                    logger.warning(f"Error processing log file {filepath}: {e}")
                    continue
            
            return sorted(events, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Error getting events by date range: {e}")
            return []
    
    def log_to_db(self, data: Dict[str, Any]) -> bool:
        """Add ability to log to database (SQLite or DuckDB).
        
        Args:
            data: Data to log to database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert data into events table
            cursor.execute("""
                INSERT INTO events (timestamp, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                data.get('timestamp', datetime.now().isoformat()),
                data.get('event_type', 'unknown'),
                json.dumps(data.get('data', {})),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Logged data to database: {data.get('event_type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging to database: {e}")
            return False
    
    def cleanup_logs_by_retention_policy(self) -> Dict[str, int]:
        """Clean up logs based on retention policy.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'files_compressed': 0,
            'files_archived': 0,
            'bytes_freed': 0
        }
        
        try:
            if self.retention_config.policy == RetentionPolicy.BY_AGE:
                stats.update(self._cleanup_by_age())
            elif self.retention_config.policy == RetentionPolicy.BY_SIZE:
                stats.update(self._cleanup_by_size())
            elif self.retention_config.policy == RetentionPolicy.BY_COUNT:
                stats.update(self._cleanup_by_count())
            elif self.retention_config.policy == RetentionPolicy.HYBRID:
                # Apply all policies in order
                age_stats = self._cleanup_by_age()
                size_stats = self._cleanup_by_size()
                count_stats = self._cleanup_by_count()
                
                # Combine statistics
                for key in stats:
                    stats[key] = age_stats.get(key, 0) + size_stats.get(key, 0) + count_stats.get(key, 0)
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return stats
    
    def _cleanup_by_age(self) -> Dict[str, int]:
        """Clean up logs older than retention period.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0}
        cutoff_date = datetime.now() - timedelta(days=self.retention_config.max_age_days)
        
        try:
            for filepath in self.log_dir.glob("*.json"):
                try:
                    # Extract timestamp from filename
                    filename = filepath.stem
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        
                        if file_timestamp < cutoff_date:
                            # Archive if enabled
                            if self.retention_config.archive_old_logs and self.retention_config.archive_dir:
                                self._archive_file(filepath)
                                stats['files_archived'] = stats.get('files_archived', 0) + 1
                            else:
                                # Compress if enabled
                                if self.retention_config.compress_old_logs:
                                    self._compress_file(filepath)
                                    stats['files_compressed'] = stats.get('files_compressed', 0) + 1
                                else:
                                    # Remove file
                                    file_size = filepath.stat().st_size
                                    filepath.unlink()
                                    stats['files_removed'] += 1
                                    stats['bytes_freed'] += file_size
                                    
                except Exception as e:
                    logger.warning(f"Error processing file {filepath}: {e}")
                    continue
            
            logger.info(f"Age-based cleanup: removed {stats['files_removed']} files, "
                       f"freed {stats['bytes_freed']} bytes")
            return stats
            
        except Exception as e:
            logger.error(f"Error in age-based cleanup: {e}")
            return stats
    
    def _cleanup_by_size(self) -> Dict[str, int]:
        """Clean up logs based on total size limit.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0}
        max_size_bytes = self.retention_config.max_size_mb * 1024 * 1024
        
        try:
            # Get all log files with their sizes and timestamps
            files_info = []
            total_size = 0
            
            for filepath in self.log_dir.glob("*.json"):
                try:
                    file_size = filepath.stat().st_size
                    filename = filepath.stem
                    
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        
                        files_info.append((filepath, file_size, file_timestamp))
                        total_size += file_size
                        
                except Exception as e:
                    logger.warning(f"Error processing file {filepath}: {e}")
                    continue
            
            # Sort by timestamp (oldest first)
            files_info.sort(key=lambda x: x[2])
            
            # Remove oldest files until under size limit
            for filepath, file_size, _ in files_info:
                if total_size <= max_size_bytes:
                    break
                    
                try:
                    filepath.unlink()
                    total_size -= file_size
                    stats['files_removed'] += 1
                    stats['bytes_freed'] += file_size
                except Exception as e:
                    logger.warning(f"Error removing file {filepath}: {e}")
            
            logger.info(f"Size-based cleanup: removed {stats['files_removed']} files, "
                       f"freed {stats['bytes_freed']} bytes")
            return stats
            
        except Exception as e:
            logger.error(f"Error in size-based cleanup: {e}")
            return stats
    
    def _cleanup_by_count(self) -> Dict[str, int]:
        """Clean up logs based on file count limit.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {'files_removed': 0, 'bytes_freed': 0}
        
        try:
            # Get all log files with their timestamps
            files_info = []
            
            for filepath in self.log_dir.glob("*.json"):
                try:
                    filename = filepath.stem
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        files_info.append((filepath, file_timestamp))
                        
                except Exception as e:
                    logger.warning(f"Error processing file {filepath}: {e}")
                    continue
            
            # Sort by timestamp (oldest first)
            files_info.sort(key=lambda x: x[1])
            
            # Remove oldest files if over limit
            if len(files_info) > self.retention_config.max_file_count:
                files_to_remove = len(files_info) - self.retention_config.max_file_count
                
                for filepath, _ in files_info[:files_to_remove]:
                    try:
                        file_size = filepath.stat().st_size
                        filepath.unlink()
                        stats['files_removed'] += 1
                        stats['bytes_freed'] += file_size
                    except Exception as e:
                        logger.warning(f"Error removing file {filepath}: {e}")
            
            logger.info(f"Count-based cleanup: removed {stats['files_removed']} files, "
                       f"freed {stats['bytes_freed']} bytes")
            return stats
            
        except Exception as e:
            logger.error(f"Error in count-based cleanup: {e}")
            return stats
    
    def _compress_file(self, filepath: Path) -> bool:
        """Compress a log file.
        
        Args:
            filepath: Path to the file to compress
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import gzip
            
            compressed_path = filepath.with_suffix('.json.gz')
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            filepath.unlink()  # Remove original file
            logger.debug(f"Compressed {filepath} to {compressed_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error compressing file {filepath}: {e}")
            return False
    
    def _archive_file(self, filepath: Path) -> bool:
        """Archive a log file.
        
        Args:
            filepath: Path to the file to archive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.retention_config.archive_dir:
                return False
            
            archive_dir = Path(self.retention_config.archive_dir)
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create year/month subdirectories
            file_timestamp = datetime.fromtimestamp(filepath.stat().st_mtime)
            archive_subdir = archive_dir / str(file_timestamp.year) / f"{file_timestamp.month:02d}"
            archive_subdir.mkdir(parents=True, exist_ok=True)
            
            # Move file to archive
            archive_path = archive_subdir / filepath.name
            shutil.move(str(filepath), str(archive_path))
            
            logger.debug(f"Archived {filepath} to {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving file {filepath}: {e}")
            return False
    
    def _start_auto_cleanup(self):
        """Start automatic cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.retention_config.cleanup_interval_hours * 3600)
                    self.cleanup_logs_by_retention_policy()
                except Exception as e:
                    logger.error(f"Error in auto-cleanup worker: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started automatic cleanup thread")
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            # The thread will stop when the main program exits
            logger.info("Auto-cleanup thread will stop on program exit")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type)
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup should be performed."""
        # Simple heuristic: cleanup every 100 events
        try:
            event_count = len(list(self.log_dir.glob("*.json")))
            return event_count % 100 == 0
        except Exception:
            return False
    
    def _cleanup_old_logs(self):
        """Legacy method - now calls the new retention policy cleanup."""
        self._cleanup_by_age()
    
    def _cleanup_by_size(self):
        """Legacy method - now calls the new retention policy cleanup."""
        self._cleanup_by_size()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about the log files.
        
        Returns:
            Dictionary with log statistics
        """
        try:
            stats = {
                'total_files': 0,
                'total_size_bytes': 0,
                'oldest_file': None,
                'newest_file': None,
                'files_by_type': {},
                'compressed_files': 0,
                'archived_files': 0
            }
            
            for filepath in self.log_dir.glob("*"):
                if filepath.is_file():
                    stats['total_files'] += 1
                    stats['total_size_bytes'] += filepath.stat().st_size
                    
                    # Check if compressed
                    if filepath.suffix == '.gz':
                        stats['compressed_files'] += 1
                    
                    # Extract event type and timestamp
                    filename = filepath.stem
                    if '_' in filename:
                        event_type = filename.split('_')[0]
                        stats['files_by_type'][event_type] = stats['files_by_type'].get(event_type, 0) + 1
                        
                        timestamp_str = filename.split('_', 1)[1]
                        try:
                            file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                            
                            if not stats['oldest_file'] or file_timestamp < stats['oldest_file']:
                                stats['oldest_file'] = file_timestamp
                            if not stats['newest_file'] or file_timestamp > stats['newest_file']:
                                stats['newest_file'] = file_timestamp
                        except ValueError:
                            pass
            
            # Convert timestamps to strings
            if stats['oldest_file']:
                stats['oldest_file'] = stats['oldest_file'].isoformat()
            if stats['newest_file']:
                stats['newest_file'] = stats['newest_file'].isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {}
    
    def clear_all_logs(self):
        """Clear all log files."""
        try:
            for filepath in self.log_dir.glob("*.json"):
                filepath.unlink()
            logger.info("Cleared all log files")
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")

def get_history_logger() -> HistoryLogger:
    """Get a singleton instance of the history logger."""
    if not hasattr(get_history_logger, '_instance'):
        get_history_logger._instance = HistoryLogger()
    return get_history_logger._instance

def log_event(event_type: str, data: Dict[str, Any], 
              timestamp: Optional[datetime] = None) -> str:
    """Convenience function to log an event."""
    return get_history_logger().log_event(event_type, data, timestamp)

def get_recent_events(event_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience function to get recent events."""
    return get_history_logger().get_recent_events(event_type, limit) 