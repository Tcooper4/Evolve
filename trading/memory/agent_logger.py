"""
Agent Logger for Evolve Trading Platform

Comprehensive logging system for all agent actions with:
- Timestamped logs with source context
- Strategy switch logging
- Model decision logging
- Performance tracking
- Error logging and recovery
- UI-visible log sections
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import deque
import sqlite3

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for agent actions."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AgentAction(Enum):
    """Types of agent actions."""
    MODEL_SYNTHESIS = "model_synthesis"
    STRATEGY_SWITCH = "strategy_switch"
    FORECAST_GENERATION = "forecast_generation"
    TRADE_EXECUTION = "trade_execution"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_MONITORING = "system_monitoring"

@dataclass
class AgentLogEntry:
    """Individual agent log entry."""
    timestamp: datetime
    agent_name: str
    action: AgentAction
    level: LogLevel
    message: str
    data: Dict[str, Any]
    context: Dict[str, Any]
    session_id: str
    user_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'action': self.action.value,
            'level': self.level.value,
            'message': self.message,
            'data': self.data,
            'context': self.context,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'performance_metrics': self.performance_metrics,
            'error_details': self.error_details
        }

class AgentLogger:
    """Comprehensive agent logging system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent logger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.log_dir = Path(self.config.get('log_dir', 'logs/agents'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory log storage for UI display
        self.recent_logs = deque(maxlen=1000)
        self.log_lock = threading.Lock()
        
        # Database for persistent storage
        self.db_path = self.log_dir / 'agent_logs.db'
        self._init_database()
        
        # Log file handlers
        self._setup_file_handlers()
        
        logger.info("Agent Logger initialized successfully")
    
    def _init_database(self):
        """Initialize SQLite database for log storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data TEXT,
                        context TEXT,
                        session_id TEXT NOT NULL,
                        user_id TEXT,
                        performance_metrics TEXT,
                        error_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_logs(agent_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON agent_logs(action)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON agent_logs(level)")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log levels."""
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / 'agent_actions.log')
        main_handler.setLevel(logging.INFO)
        
        # Error log file
        error_handler = logging.FileHandler(self.log_dir / 'agent_errors.log')
        error_handler.setLevel(logging.ERROR)
        
        # Performance log file
        perf_handler = logging.FileHandler(self.log_dir / 'agent_performance.log')
        perf_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        for handler in [main_handler, error_handler, perf_handler]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def log_action(self, agent_name: str, action: AgentAction, message: str,
                  data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,
                  level: LogLevel = LogLevel.INFO, session_id: str = "default",
                  user_id: Optional[str] = None, performance_metrics: Optional[Dict[str, float]] = None,
                  error_details: Optional[str] = None) -> AgentLogEntry:
        """Log an agent action.
        
        Args:
            agent_name: Name of the agent
            action: Type of action performed
            message: Log message
            data: Additional data
            context: Context information
            level: Log level
            session_id: Session identifier
            user_id: User identifier
            performance_metrics: Performance metrics
            error_details: Error details if applicable
            
        Returns:
            AgentLogEntry object
        """
        try:
            # Create log entry
            log_entry = AgentLogEntry(
                timestamp=datetime.now(),
                agent_name=agent_name,
                action=action,
                level=level,
                message=message,
                data=data or {},
                context=context or {},
                session_id=session_id,
                user_id=user_id,
                performance_metrics=performance_metrics,
                error_details=error_details
            )
            
            # Add to in-memory storage
            with self.log_lock:
                self.recent_logs.append(log_entry)
            
            # Store in database
            self._store_in_database(log_entry)
            
            # Log to file system
            self._log_to_file(log_entry)
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
            return None
    
    def _store_in_database(self, log_entry: AgentLogEntry):
        """Store log entry in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO agent_logs 
                    (timestamp, agent_name, action, level, message, data, context, 
                     session_id, user_id, performance_metrics, error_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_entry.timestamp.isoformat(),
                    log_entry.agent_name,
                    log_entry.action.value,
                    log_entry.level.value,
                    log_entry.message,
                    json.dumps(log_entry.data),
                    json.dumps(log_entry.context),
                    log_entry.session_id,
                    log_entry.user_id,
                    json.dumps(log_entry.performance_metrics) if log_entry.performance_metrics else None,
                    log_entry.error_details
                ))
        except Exception as e:
            logger.error(f"Failed to store log in database: {e}")
    
    def _log_to_file(self, log_entry: AgentLogEntry):
        """Log entry to appropriate file."""
        try:
            # Main log
            if log_entry.level in [LogLevel.INFO, LogLevel.DEBUG]:
                logger.info(f"[{log_entry.agent_name}] {log_entry.message}")
            
            # Error log
            if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                logger.error(f"[{log_entry.agent_name}] {log_entry.message}")
                if log_entry.error_details:
                    logger.error(f"Error details: {log_entry.error_details}")
            
            # Performance log
            if log_entry.performance_metrics:
                logger.info(f"[{log_entry.agent_name}] Performance: {log_entry.performance_metrics}")
                
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")
    
    def get_recent_logs(self, limit: int = 50, agent_name: Optional[str] = None,
                       action: Optional[AgentAction] = None, level: Optional[LogLevel] = None) -> List[AgentLogEntry]:
        """Get recent logs with optional filtering.
        
        Args:
            limit: Maximum number of logs to return
            agent_name: Filter by agent name
            action: Filter by action type
            level: Filter by log level
            
        Returns:
            List of recent log entries
        """
        with self.log_lock:
            logs = list(self.recent_logs)
        
        # Apply filters
        if agent_name:
            logs = [log for log in logs if log.agent_name == agent_name]
        
        if action:
            logs = [log for log in logs if log.action == action]
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        # Return most recent logs
        return logs[-limit:]
    
    def get_logs_by_session(self, session_id: str, limit: int = 100) -> List[AgentLogEntry]:
        """Get logs for a specific session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM agent_logs 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))
                
                logs = []
                for row in cursor.fetchall():
                    log_entry = AgentLogEntry(
                        timestamp=datetime.fromisoformat(row[1]),
                        agent_name=row[2],
                        action=AgentAction(row[3]),
                        level=LogLevel(row[4]),
                        message=row[5],
                        data=json.loads(row[6]) if row[6] else {},
                        context=json.loads(row[7]) if row[7] else {},
                        session_id=row[8],
                        user_id=row[9],
                        performance_metrics=json.loads(row[10]) if row[10] else None,
                        error_details=row[11]
                    )
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Failed to get logs by session: {e}")
            return []
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get action counts
                cursor = conn.execute("""
                    SELECT action, COUNT(*) as count 
                    FROM agent_logs 
                    WHERE timestamp >= ? 
                    GROUP BY action
                """, (since.isoformat(),))
                
                action_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get error counts
                cursor = conn.execute("""
                    SELECT COUNT(*) as error_count 
                    FROM agent_logs 
                    WHERE timestamp >= ? AND level IN ('error', 'critical')
                """, (since.isoformat(),))
                
                error_count = cursor.fetchone()[0]
                
                # Get performance metrics
                cursor = conn.execute("""
                    SELECT performance_metrics 
                    FROM agent_logs 
                    WHERE timestamp >= ? AND performance_metrics IS NOT NULL
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, (since.isoformat(),))
                
                performance_data = []
                for row in cursor.fetchall():
                    if row[0]:
                        performance_data.append(json.loads(row[0]))
                
                return {
                    'action_counts': action_counts,
                    'error_count': error_count,
                    'performance_data': performance_data,
                    'time_period': f"{hours}h"
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def clear_old_logs(self, days: int = 30):
        """Clear logs older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM agent_logs WHERE timestamp < ?", (cutoff_date.isoformat(),))
            
            logger.info(f"Cleared logs older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to clear old logs: {e}")
    
    def export_logs(self, filepath: str, format: str = 'json', 
                   start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """Export logs to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv')
            start_date: Start date filter
            end_date: End date filter
        """
        try:
            # Build query
            query = "SELECT * FROM agent_logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            if format == 'json':
                logs = []
                for row in rows:
                    log_dict = {
                        'timestamp': row[1],
                        'agent_name': row[2],
                        'action': row[3],
                        'level': row[4],
                        'message': row[5],
                        'data': json.loads(row[6]) if row[6] else {},
                        'context': json.loads(row[7]) if row[7] else {},
                        'session_id': row[8],
                        'user_id': row[9],
                        'performance_metrics': json.loads(row[10]) if row[10] else None,
                        'error_details': row[11]
                    }
                    logs.append(log_dict)
                
                with open(filepath, 'w') as f:
                    json.dump(logs, f, indent=2)
            
            elif format == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'agent_name', 'action', 'level', 'message', 'session_id'])
                    for row in rows:
                        writer.writerow([row[1], row[2], row[3], row[4], row[5], row[8]])
            
            logger.info(f"Exported {len(rows)} logs to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")

# Global logger instance
_agent_logger = None

def get_agent_logger() -> AgentLogger:
    """Get global agent logger instance."""
    global _agent_logger
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    return _agent_logger

def log_agent_action(agent_name: str, action: AgentAction, message: str, **kwargs) -> AgentLogEntry:
    """Convenience function to log agent action."""
    logger = get_agent_logger()
    return logger.log_action(agent_name, action, message, **kwargs) 