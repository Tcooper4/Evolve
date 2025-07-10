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

logger = logging.getLogger(__name__)

class HistoryLogger:
    """History logger with automatic cleanup and rotation."""
    
    def __init__(self, 
                 log_dir: str = "logs/history",
                 max_logs: int = 100,
                 max_size_mb: int = 100,
                 retention_days: int = 30):
        """Initialize the history logger.
        
        Args:
            log_dir: Directory to store log files
            max_logs: Maximum number of log files to keep
            max_size_mb: Maximum total size of logs in MB
            retention_days: Number of days to retain logs
        """
        self.log_dir = Path(log_dir)
        self.max_logs = max_logs
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.retention_days = retention_days
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cleanup
        self._cleanup_old_logs()
        self._cleanup_by_size()
    
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
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup should be performed."""
        # Cleanup every 10 log entries or when directory is large
        log_count = len(list(self.log_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.log_dir.glob("*.json") if f.is_file())
        
        return (log_count > self.max_logs or 
                total_size > self.max_size_bytes or
                log_count % 10 == 0)  # Periodic cleanup
    
    def _cleanup_old_logs(self):
        """Remove logs older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            removed_count = 0
            
            for filepath in self.log_dir.glob("*.json"):
                try:
                    # Extract timestamp from filename
                    filename = filepath.stem
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        
                        if file_timestamp < cutoff_date:
                            filepath.unlink()
                            removed_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing file {filepath} for cleanup: {e}")
                    continue
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old log files")
                
        except Exception as e:
            logger.error(f"Error during old log cleanup: {e}")
    
    def _cleanup_by_size(self):
        """Remove oldest logs when total size exceeds limit."""
        try:
            # Get all log files with their sizes and timestamps
            log_files = []
            for filepath in self.log_dir.glob("*.json"):
                try:
                    filename = filepath.stem
                    if '_' in filename:
                        timestamp_str = filename.split('_', 1)[1]
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
                        
                        log_files.append({
                            'path': filepath,
                            'size': filepath.stat().st_size,
                            'timestamp': file_timestamp
                        })
                except Exception as e:
                    logger.warning(f"Error processing file {filepath}: {e}")
                    continue
            
            # Sort by timestamp (oldest first)
            log_files.sort(key=lambda x: x['timestamp'])
            
            # Calculate total size
            total_size = sum(f['size'] for f in log_files)
            
            # Remove oldest files until under size limit
            removed_count = 0
            for log_file in log_files:
                if total_size <= self.max_size_bytes:
                    break
                    
                try:
                    log_file['path'].unlink()
                    total_size -= log_file['size']
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Error removing file {log_file['path']}: {e}")
                    continue
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} log files due to size limit")
                
        except Exception as e:
            logger.error(f"Error during size-based cleanup: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about the log directory.
        
        Returns:
            Dictionary with log statistics
        """
        try:
            log_files = list(self.log_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in log_files if f.is_file())
            
            # Count by event type
            event_counts = {}
            for filepath in log_files:
                try:
                    filename = filepath.stem
                    if '_' in filename:
                        event_type = filename.split('_')[0]
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                except Exception:
                    continue
            
            return {
                'total_files': len(log_files),
                'total_size_mb': total_size / (1024 * 1024),
                'event_counts': event_counts,
                'max_logs': self.max_logs,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'retention_days': self.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {}
    
    def clear_all_logs(self):
        """Clear all log files (use with caution)."""
        try:
            for filepath in self.log_dir.glob("*.json"):
                filepath.unlink()
            logger.info("All log files cleared")
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")

# Global instance
_history_logger = None

def get_history_logger() -> HistoryLogger:
    """Get the global history logger instance."""
    global _history_logger
    if _history_logger is None:
        _history_logger = HistoryLogger()
    return _history_logger

def log_event(event_type: str, data: Dict[str, Any], 
              timestamp: Optional[datetime] = None) -> str:
    """Convenience function to log an event."""
    return get_history_logger().log_event(event_type, data, timestamp)

def get_recent_events(event_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience function to get recent events."""
    return get_history_logger().get_recent_events(event_type, limit) 