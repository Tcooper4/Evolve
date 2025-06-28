"""
Memory Logger Utility

This module provides logging functionality for memory-related operations
in the trading system, including performance tracking and debugging.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from trading.config.settings import LOG_DIR

class MemoryLogger:
    """Logger for memory-related operations and performance tracking."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the memory logger.
        
        Args:
            log_file: Optional custom log file path
        """
        self.log_file = log_file or LOG_DIR / "memory.log"
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Configure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
    
    def log_memory_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log a memory operation.
        
        Args:
            operation: Type of operation (e.g., 'read', 'write', 'update')
            details: Operation details
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'details': details
        }
        
        self.logger.info(f"Memory operation: {json.dumps(log_entry)}")
    
    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'performance_metrics',
            'metrics': metrics
        }
        
        self.logger.info(f"Performance metrics: {json.dumps(log_entry)}")
    
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error.
        
        Args:
            error: Error message
            context: Optional context information
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'error',
            'error': error,
            'context': context or {}
        }
        
        self.logger.error(f"Memory error: {json.dumps(log_entry)}")
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent log entries
        """
        logs = []
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        # Extract JSON from log line
                        if 'Memory operation:' in line or 'Performance metrics:' in line or 'Memory error:' in line:
                            json_start = line.find('{')
                            if json_start != -1:
                                json_str = line[json_start:]
                                log_entry = json.loads(json_str)
                                logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return logs
    
    def clear_logs(self) -> None:
        """Clear all log entries."""
        try:
            with open(self.log_file, 'w') as f:
                f.write('')
            self.logger.info("Logs cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear logs: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about the log file.
        
        Returns:
            Dictionary with log statistics
        """
        try:
            if not os.path.exists(self.log_file):
                return {'total_entries': 0, 'file_size': 0}
            
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            return {
                'total_entries': len(lines),
                'file_size': os.path.getsize(self.log_file),
                'last_modified': datetime.fromtimestamp(
                    os.path.getmtime(self.log_file)
                ).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get log stats: {e}")
            return {'error': str(e)} 