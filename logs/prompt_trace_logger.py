"""
Prompt Trace Logger

This module provides logging functionality for tracking prompt processing
and action execution in the trading system.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of prompt actions."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class PromptTrace:
    """Trace record for prompt processing."""
    trace_id: str
    prompt: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionTrace:
    """Trace record for action execution."""
    action_id: str
    action_type: str
    status: ActionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTraceLogger:
    """
    Logger for tracking prompt processing and action execution.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize prompt trace logger.
        
        Args:
            log_file: Path to log file (optional)
        """
        self.log_file = log_file
        self.traces: Dict[str, PromptTrace] = {}
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("PromptTraceLogger initialized")
    
    def start_trace(self, prompt: str, user_id: Optional[str] = None,
                   session_id: Optional[str] = None) -> str:
        """
        Start a new prompt trace.
        
        Args:
            prompt: User prompt
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Trace ID
        """
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        trace = PromptTrace(
            trace_id=trace_id,
            prompt=prompt,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id
        )
        
        self.traces[trace_id] = trace
        logger.debug(f"Started prompt trace: {trace_id}")
        
        return trace_id
    
    def add_action(self, trace_id: str, action_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an action to a trace.
        
        Args:
            trace_id: Trace identifier
            action_type: Type of action
            metadata: Additional metadata
            
        Returns:
            Action ID
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace not found: {trace_id}")
        
        action_id = f"action_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        action_trace = ActionTrace(
            action_id=action_id,
            action_type=action_type,
            status=ActionStatus.PENDING,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        self.traces[trace_id].actions.append(action_trace.__dict__)
        logger.debug(f"Added action to trace {trace_id}: {action_id}")
        
        return action_id
    
    def update_action_status(self, trace_id: str, action_id: str, status: ActionStatus,
                           result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """
        Update action status.
        
        Args:
            trace_id: Trace identifier
            action_id: Action identifier
            status: New status
            result: Action result
            error: Error message
        """
        if trace_id not in self.traces:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        trace = self.traces[trace_id]
        
        for action in trace.actions:
            if action["action_id"] == action_id:
                action["status"] = status.value
                action["end_time"] = datetime.now().isoformat()
                
                if action["start_time"]:
                    start_time = datetime.fromisoformat(action["start_time"])
                    end_time = datetime.now()
                    action["duration"] = (end_time - start_time).total_seconds()
                
                if result:
                    action["result"] = result
                
                if error:
                    action["error"] = error
                
                logger.debug(f"Updated action {action_id} status to {status.value}")
                break
    
    def complete_trace(self, trace_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Complete a prompt trace.
        
        Args:
            trace_id: Trace identifier
            metadata: Additional metadata
        """
        if trace_id not in self.traces:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        trace = self.traces[trace_id]
        
        if metadata:
            trace.metadata.update(metadata)
        
        # Save to file if configured
        if self.log_file:
            self._save_trace(trace)
        
        logger.info(f"Completed prompt trace: {trace_id}")
    
    def _save_trace(self, trace: PromptTrace):
        """Save trace to log file."""
        try:
            trace_data = {
                "trace_id": trace.trace_id,
                "prompt": trace.prompt,
                "timestamp": trace.timestamp.isoformat(),
                "user_id": trace.user_id,
                "session_id": trace.session_id,
                "actions": trace.actions,
                "metadata": trace.metadata
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(trace_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving trace: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[PromptTrace]:
        """
        Get a trace by ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            PromptTrace or None
        """
        return self.traces.get(trace_id)
    
    def get_traces_by_user(self, user_id: str) -> List[PromptTrace]:
        """
        Get all traces for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of traces
        """
        return [trace for trace in self.traces.values() if trace.user_id == user_id]
    
    def get_traces_by_session(self, session_id: str) -> List[PromptTrace]:
        """
        Get all traces for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of traces
        """
        return [trace for trace in self.traces.values() if trace.session_id == session_id]
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about traces.
        
        Returns:
            Dictionary with trace statistics
        """
        if not self.traces:
            return {}
        
        total_traces = len(self.traces)
        total_actions = sum(len(trace.actions) for trace in self.traces.values())
        
        # Count actions by status
        status_counts = {}
        for trace in self.traces.values():
            for action in trace.actions:
                status = action.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average duration
        durations = []
        for trace in self.traces.values():
            for action in trace.actions:
                if "duration" in action:
                    durations.append(action["duration"])
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_traces": total_traces,
            "total_actions": total_actions,
            "avg_actions_per_trace": total_actions / total_traces if total_traces > 0 else 0,
            "action_status_distribution": status_counts,
            "avg_action_duration": avg_duration
        }
    
    def clear_traces(self, older_than_days: Optional[int] = None):
        """
        Clear old traces.
        
        Args:
            older_than_days: Clear traces older than this many days
        """
        if older_than_days is None:
            self.traces.clear()
            logger.info("Cleared all traces")
            return
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        to_remove = []
        
        for trace_id, trace in self.traces.items():
            if trace.timestamp < cutoff_time:
                to_remove.append(trace_id)
        
        for trace_id in to_remove:
            del self.traces[trace_id]
        
        logger.info(f"Cleared {len(to_remove)} old traces")


def create_prompt_trace_logger(log_file: Optional[str] = None) -> PromptTraceLogger:
    """
    Create a prompt trace logger instance.
    
    Args:
        log_file: Path to log file
        
    Returns:
        PromptTraceLogger instance
    """
    return PromptTraceLogger(log_file)
