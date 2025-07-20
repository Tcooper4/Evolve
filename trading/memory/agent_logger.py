"""
Agent Logger for Evolve Trading Platform

Comprehensive logging system for all agent actions with:
- Timestamped logs with source context
- Strategy switch logging
- Model decision logging
- Performance tracking
- Error logging and recovery
- UI-visible log sections
- Log level filtering and agent context tracking
"""

import json
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class AgentType(Enum):
    """Types of agents in the system."""

    MODEL_SELECTOR = "model_selector"
    MODEL_SYNTHESIZER = "model_synthesizer"
    MULTIMODAL = "multimodal"
    NLP = "nlp"
    OPTIMIZER = "optimizer"
    PERFORMANCE_CRITIC = "performance_critic"
    DATA_PROCESSOR = "data_processor"
    STRATEGY_MANAGER = "strategy_manager"
    RISK_MANAGER = "risk_manager"
    EXECUTION_ENGINE = "execution_engine"


@dataclass
class AgentContext:
    """Context information for agent operations."""

    agent_type: AgentType
    agent_id: str
    version: str = "1.0"
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    task_priority: int = 1
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    last_heartbeat: Optional[datetime] = None
    status: str = "active"
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLogEntry:
    """Individual agent log entry."""

    timestamp: datetime
    agent_name: str
    agent_type: AgentType
    action: AgentAction
    level: LogLevel
    message: str
    data: Dict[str, Any]
    context: AgentContext
    session_id: str
    user_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[str] = None
    task_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value,
            "action": self.action.value,
            "level": self.level.value,
            "message": self.message,
            "data": self.data,
            "context": asdict(self.context),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "performance_metrics": self.performance_metrics,
            "error_details": self.error_details,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
        }


@dataclass
class LogFilter:
    """Filter configuration for log queries."""

    min_level: LogLevel = LogLevel.DEBUG
    agent_names: Optional[List[str]] = None
    agent_types: Optional[List[AgentType]] = None
    actions: Optional[List[AgentAction]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    session_ids: Optional[List[str]] = None
    task_ids: Optional[List[str]] = None
    include_errors: bool = True
    include_performance: bool = True
    max_results: int = 1000


class AgentLogger:
    """Comprehensive agent logging system with filtering and context tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.log_dir = Path(self.config.get("log_dir", "logs/agents"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log level configuration
        min_level_str = self.config.get("min_log_level", "INFO").lower()
        try:
            self.min_log_level = LogLevel(min_level_str)
        except ValueError:
            logger.warning(f"Invalid log level '{min_level_str}', defaulting to INFO")
            self.min_log_level = LogLevel.INFO
        self.enable_debug_logs = self.config.get("enable_debug_logs", False)
        self.enable_performance_logs = self.config.get("enable_performance_logs", True)

        # In-memory log storage for UI display
        self.recent_logs = deque(maxlen=1000)
        self.log_lock = threading.Lock()

        # Database for persistent storage
        self.db_path = self.log_dir / "agent_logs.db"
        self._init_database()

        # Log file handlers
        self._setup_file_handlers()

        # Agent context registry
        self.agent_contexts: Dict[str, AgentContext] = {}
        self.context_lock = threading.Lock()

        logger.info("Agent Logger initialized successfully")

    def _init_database(self):
        """Initialize SQLite database for log storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if table exists and has the correct schema
                cursor = conn.execute("PRAGMA table_info(agent_logs)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if not columns:
                    # Table doesn't exist, create it
                    conn.execute(
                        """
                        CREATE TABLE agent_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            agent_name TEXT NOT NULL,
                            agent_type TEXT NOT NULL,
                            action TEXT NOT NULL,
                            level TEXT NOT NULL,
                            message TEXT NOT NULL,
                            data TEXT,
                            context TEXT,
                            session_id TEXT NOT NULL,
                            user_id TEXT,
                            performance_metrics TEXT,
                            error_details TEXT,
                            task_id TEXT,
                            correlation_id TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    )
                elif 'agent_type' not in columns:
                    # Table exists but missing agent_type column, recreate it
                    logger.info("Database schema outdated, recreating table...")
                    conn.execute("DROP TABLE IF EXISTS agent_logs")
                    conn.execute(
                        """
                        CREATE TABLE agent_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            agent_name TEXT NOT NULL,
                            agent_type TEXT NOT NULL,
                            action TEXT NOT NULL,
                            level TEXT NOT NULL,
                            message TEXT NOT NULL,
                            data TEXT,
                            context TEXT,
                            session_id TEXT NOT NULL,
                            user_id TEXT,
                            performance_metrics TEXT,
                            error_details TEXT,
                            task_id TEXT,
                            correlation_id TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    )

                # Create indexes for better performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_logs(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_logs(agent_name)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_type ON agent_logs(agent_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_action ON agent_logs(action)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_level ON agent_logs(level)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task_id ON agent_logs(task_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_id ON agent_logs(session_id)"
                )

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _setup_file_handlers(self):
        """Setup file handlers for different log levels."""
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / "agent_actions.log")
        main_handler.setLevel(logging.INFO)

        # Debug log file (if enabled)
        if self.enable_debug_logs:
            debug_handler = logging.FileHandler(self.log_dir / "agent_debug.log")
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(debug_handler)

        # Error log file
        error_handler = logging.FileHandler(self.log_dir / "agent_errors.log")
        error_handler.setLevel(logging.ERROR)

        # Performance log file
        if self.enable_performance_logs:
            perf_handler = logging.FileHandler(self.log_dir / "agent_performance.log")
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(perf_handler)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        for handler in [main_handler, error_handler]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def register_agent_context(
        self,
        agent_name: str,
        agent_type: AgentType,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentContext:
        """Register context for an agent.

        Args:
            agent_name: Name of the agent
            agent_type: Type of the agent
            context: Additional context information

        Returns:
            AgentContext object
        """
        with self.context_lock:
            agent_context = AgentContext(
                agent_type=agent_type,
                agent_id=agent_name,
                version=context.get("version", "1.0") if context else "1.0",
                capabilities=context.get("capabilities", []) if context else [],
                configuration=context.get("configuration", {}) if context else {},
            )
            self.agent_contexts[agent_name] = agent_context
            logger.info(
                f"Registered context for agent: {agent_name} ({agent_type.value})"
            )
            return agent_context

    def update_agent_context(self, agent_name: str, updates: Dict[str, Any]):
        """Update context for an agent.

        Args:
            agent_name: Name of the agent
            updates: Context updates
        """
        with self.context_lock:
            if agent_name in self.agent_contexts:
                context = self.agent_contexts[agent_name]
                for key, value in updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                context.last_heartbeat = datetime.now()
                logger.debug(f"Updated context for agent: {agent_name}")

    def log_action(
        self,
        agent_name: str,
        action: AgentAction,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        level: LogLevel = LogLevel.INFO,
        session_id: str = "default",
        user_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        error_details: Optional[str] = None,
        task_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[AgentLogEntry]:
        """Log an agent action with filtering and context.

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
            task_id: Task identifier
            correlation_id: Correlation identifier

        Returns:
            AgentLogEntry object or None if filtered out
        """
        try:
            # Check log level filtering
            if level.value < self.min_log_level.value:
                return None

            # Get agent context
            agent_context = self.agent_contexts.get(agent_name)
            if agent_context is None:
                # Create default context if not registered
                agent_context = self.register_agent_context(
                    agent_name, AgentType.MODEL_SELECTOR, context  # Default type
                )

            # Create log entry
            log_entry = AgentLogEntry(
                timestamp=datetime.now(),
                agent_name=agent_name,
                agent_type=agent_context.agent_type,
                action=action,
                level=level,
                message=message,
                data=data or {},
                context=agent_context,
                session_id=session_id,
                user_id=user_id,
                performance_metrics=performance_metrics,
                error_details=error_details,
                task_id=task_id,
                correlation_id=correlation_id,
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
            logger.error(f"Error logging agent action: {e}")
            return None

    def _store_in_database(self, log_entry: AgentLogEntry):
        """Store log entry in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_logs
                    (timestamp, agent_name, agent_type, action, level, message, data, context,
                     session_id, user_id, performance_metrics, error_details, task_id, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        log_entry.timestamp.isoformat(),
                        log_entry.agent_name,
                        log_entry.agent_type.value,
                        log_entry.action.value,
                        log_entry.level.value,
                        log_entry.message,
                        json.dumps(log_entry.data),
                        json.dumps(enum_to_value(log_entry.context)),
                        log_entry.session_id,
                        log_entry.user_id,
                        json.dumps(log_entry.performance_metrics)
                        if log_entry.performance_metrics
                        else None,
                        log_entry.error_details,
                        log_entry.task_id,
                        log_entry.correlation_id,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store log in database: {e}")

    def _log_to_file(self, log_entry: AgentLogEntry):
        """Log entry to appropriate file based on level."""
        try:
            log_message = f"[{log_entry.agent_name}:{log_entry.agent_type.value}] {log_entry.message}"

            if (
                log_entry.level == LogLevel.ERROR
                or log_entry.level == LogLevel.CRITICAL
            ):
                logger.error(log_message)
            elif log_entry.level == LogLevel.WARNING:
                logger.warning(log_message)
            elif log_entry.level == LogLevel.INFO:
                logger.info(log_message)
            elif log_entry.level == LogLevel.DEBUG and self.enable_debug_logs:
                logger.debug(log_message)

        except Exception as e:
            logger.error(f"Failed to log to file: {e}")

    def get_logs_with_filter(self, log_filter: LogFilter) -> List[AgentLogEntry]:
        """Get logs with advanced filtering.

        Args:
            log_filter: Filter configuration

        Returns:
            List of filtered log entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query
                query = "SELECT * FROM agent_logs WHERE 1=1"
                params = []

                # Level filter
                query += f" AND level >= ?"
                params.append(log_filter.min_level.value)

                # Agent name filter
                if log_filter.agent_names:
                    placeholders = ",".join(["?" for _ in log_filter.agent_names])
                    query += f" AND agent_name IN ({placeholders})"
                    params.extend(log_filter.agent_names)

                # Agent type filter
                if log_filter.agent_types:
                    placeholders = ",".join(["?" for _ in log_filter.agent_types])
                    query += f" AND agent_type IN ({placeholders})"
                    params.extend([t.value for t in log_filter.agent_types])

                # Action filter
                if log_filter.actions:
                    placeholders = ",".join(["?" for _ in log_filter.actions])
                    query += f" AND action IN ({placeholders})"
                    params.extend([a.value for a in log_filter.actions])

                # Time range filter
                if log_filter.start_time:
                    query += " AND timestamp >= ?"
                    params.append(log_filter.start_time.isoformat())

                if log_filter.end_time:
                    query += " AND timestamp <= ?"
                    params.append(log_filter.end_time.isoformat())

                # Session filter
                if log_filter.session_ids:
                    placeholders = ",".join(["?" for _ in log_filter.session_ids])
                    query += f" AND session_id IN ({placeholders})"
                    params.extend(log_filter.session_ids)

                # Task filter
                if log_filter.task_ids:
                    placeholders = ",".join(["?" for _ in log_filter.task_ids])
                    query += f" AND task_id IN ({placeholders})"
                    params.extend(log_filter.task_ids)

                # Error filter
                if not log_filter.include_errors:
                    query += " AND level NOT IN ('error', 'critical')"

                # Performance filter
                if not log_filter.include_performance:
                    query += " AND performance_metrics IS NULL"

                # Order and limit
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(log_filter.max_results)

                # Execute query
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                # Convert to AgentLogEntry objects
                entries = []
                for row in rows:
                    try:
                        entry = AgentLogEntry(
                            timestamp=datetime.fromisoformat(row[1]),
                            agent_name=row[2],
                            agent_type=AgentType(row[3]),
                            action=AgentAction(row[4]),
                            level=LogLevel(row[5]),
                            message=row[6],
                            data=json.loads(row[7]) if row[7] else {},
                            context=AgentContext(**json.loads(row[8]))
                            if row[8]
                            else None,
                            session_id=row[9],
                            user_id=row[10],
                            performance_metrics=json.loads(row[11])
                            if row[11]
                            else None,
                            error_details=row[12],
                            task_id=row[13],
                            correlation_id=row[14],
                        )
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing log entry: {e}")
                        continue

                return entries

        except Exception as e:
            logger.error(f"Error getting logs with filter: {e}")
            return []

    def get_recent_logs(
        self,
        limit: int = 50,
        agent_name: Optional[str] = None,
        action: Optional[AgentAction] = None,
        level: Optional[LogLevel] = None,
    ) -> List[AgentLogEntry]:
        """Get recent logs with basic filtering.

        Args:
            limit: Maximum number of logs to return
            agent_name: Filter by agent name
            action: Filter by action type
            level: Filter by log level

        Returns:
            List of log entries
        """
        # Create filter
        log_filter = LogFilter(
            min_level=level or LogLevel.DEBUG,
            agent_names=[agent_name] if agent_name else None,
            actions=[action] if action else None,
            max_results=limit,
        )

        return self.get_logs_with_filter(log_filter)

    def get_logs_by_session(
        self, session_id: str, limit: int = 100
    ) -> List[AgentLogEntry]:
        """Get logs for a specific session.

        Args:
            session_id: Session identifier
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        log_filter = LogFilter(session_ids=[session_id], max_results=limit)

        return self.get_logs_with_filter(log_filter)

    def get_logs_by_task(self, task_id: str, limit: int = 100) -> List[AgentLogEntry]:
        """Get logs for a specific task.

        Args:
            task_id: Task identifier
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        log_filter = LogFilter(task_ids=[task_id], max_results=limit)

        return self.get_logs_with_filter(log_filter)

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Performance summary dictionary
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            log_filter = LogFilter(
                start_time=start_time,
                end_time=end_time,
                include_performance=True,
                max_results=10000,
            )

            logs = self.get_logs_with_filter(log_filter)

            # Calculate summary statistics
            summary = {
                "total_logs": len(logs),
                "error_count": len(
                    [l for l in logs if l.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
                ),
                "warning_count": len([l for l in logs if l.level == LogLevel.WARNING]),
                "agent_activity": {},
                "action_distribution": {},
                "performance_metrics": {},
            }

            # Agent activity
            for log in logs:
                agent_name = log.agent_name
                if agent_name not in summary["agent_activity"]:
                    summary["agent_activity"][agent_name] = {
                        "total_actions": 0,
                        "errors": 0,
                        "warnings": 0,
                        "last_action": None,
                    }

                summary["agent_activity"][agent_name]["total_actions"] += 1
                if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    summary["agent_activity"][agent_name]["errors"] += 1
                elif log.level == LogLevel.WARNING:
                    summary["agent_activity"][agent_name]["warnings"] += 1

                if (
                    summary["agent_activity"][agent_name]["last_action"] is None
                    or log.timestamp
                    > summary["agent_activity"][agent_name]["last_action"]
                ):
                    summary["agent_activity"][agent_name]["last_action"] = log.timestamp

            # Action distribution
            for log in logs:
                action = log.action.value
                summary["action_distribution"][action] = (
                    summary["action_distribution"].get(action, 0) + 1
                )

            # Performance metrics
            perf_logs = [l for l in logs if l.performance_metrics]
            if perf_logs:
                all_metrics = {}
                for log in perf_logs:
                    for metric, value in log.performance_metrics.items():
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)

                for metric, values in all_metrics.items():
                    summary["performance_metrics"][metric] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            return summary

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def clear_old_logs(self, days: int = 30):
        """Clear logs older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM agent_logs WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )
                deleted_count = conn.total_changes

            logger.info(f"Cleared {deleted_count} old log entries")

        except Exception as e:
            logger.error(f"Error clearing old logs: {e}")

    def export_logs(
        self,
        filepath: str,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """Export logs to file.

        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            start_date: Start date for export
            end_date: End date for export
        """
        try:
            log_filter = LogFilter(
                start_time=start_date, end_time=end_date, max_results=100000
            )

            logs = self.get_logs_with_filter(log_filter)

            if format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump([log.to_dict() for log in logs], f, indent=2)
            elif format.lower() == "csv":
                import csv

                with open(filepath, "w", newline="") as f:
                    if logs:
                        writer = csv.DictWriter(f, fieldnames=logs[0].to_dict().keys())
                        writer.writeheader()
                        for log in logs:
                            writer.writerow(log.to_dict())

            logger.info(f"Exported {len(logs)} logs to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting logs: {e}")

    def get_agent_contexts(self) -> Dict[str, AgentContext]:
        """Get all registered agent contexts.

        Returns:
            Dictionary of agent contexts
        """
        with self.context_lock:
            return self.agent_contexts.copy()

    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Status dictionary or None if agent not found
        """
        with self.context_lock:
            if agent_name not in self.agent_contexts:
                return None

            context = self.agent_contexts[agent_name]
            return {
                "agent_name": agent_name,
                "agent_type": context.agent_type.value,
                "status": context.status,
                "version": context.version,
                "capabilities": context.capabilities,
                "current_task": context.current_task,
                "task_priority": context.task_priority,
                "memory_usage": context.memory_usage,
                "cpu_usage": context.cpu_usage,
                "last_heartbeat": context.last_heartbeat.isoformat()
                if context.last_heartbeat
                else None,
                "dependencies": context.dependencies,
            }


def enum_to_value(obj):
    """Recursively convert enums in a dict (or dataclass) to their .value for JSON serialization."""
    if isinstance(obj, dict):
        return {k: enum_to_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [enum_to_value(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(enum_to_value(i) for i in obj)
    elif hasattr(obj, "__dataclass_fields__"):
        return enum_to_value(asdict(obj))
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj


# --- Global Agent Logger Instance ---
_agent_logger = None


def get_agent_logger() -> AgentLogger:
    """Get the global agent logger instance."""
    global _agent_logger
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    return _agent_logger


def log_agent_action(
    agent_name: str, action: AgentAction, message: str, **kwargs
) -> Optional[AgentLogEntry]:
    """Convenience function to log an agent action."""
    return get_agent_logger().log_action(agent_name, action, message, **kwargs)
