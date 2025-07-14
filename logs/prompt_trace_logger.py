"""
Prompt Trace Logger

This module provides comprehensive logging of prompt processing traces including:
- User input → interpreted intent → executed action
- Performance metrics and timing
- Error tracking and fallback usage
- Session management and context tracking
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Trace logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ActionStatus(Enum):
    """Status of executed actions."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"


@dataclass
class PromptTrace:
    """Complete trace of a prompt processing session."""
    trace_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    timestamp: datetime
    
    # Input
    original_prompt: str
    normalized_prompt: str
    
    # Intent processing
    intent_detection_method: str  # 'regex', 'huggingface', 'openai', 'fallback'
    detected_intent: str
    intent_confidence: float
    extracted_parameters: Dict[str, Any]
    
    # Action execution
    executed_action: str
    action_status: ActionStatus
    action_result: Optional[Dict[str, Any]]
    action_duration: float
    
    # Performance metrics
    total_processing_time: float
    provider_usage: Dict[str, int]
    
    # Error handling
    errors: List[str]
    fallbacks_used: List[str]
    
    # Context
    context: Dict[str, Any]
    metadata: Dict[str, Any]


class PromptTraceLogger:
    """Comprehensive prompt trace logging system."""
    
    def __init__(
        self,
        log_dir: str = "logs/prompt_traces",
        max_traces_per_file: int = 1000,
        enable_compression: bool = True,
        retention_days: int = 30
    ):
        """Initialize the prompt trace logger.
        
        Args:
            log_dir: Directory to store trace logs
            max_traces_per_file: Maximum traces per log file
            enable_compression: Enable log file compression
            retention_days: Number of days to retain logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_traces_per_file = max_traces_per_file
        self.enable_compression = enable_compression
        self.retention_days = retention_days
        
        # Current trace file
        self.current_file = None
        self.traces_in_current_file = 0
        
        # Performance tracking
        self.total_traces = 0
        self.successful_traces = 0
        self.failed_traces = 0
        
        # Initialize current file
        self._initialize_current_file()
        
        # Cleanup old files
        self._cleanup_old_files()
        
        logger.info(f"PromptTraceLogger initialized with log_dir: {self.log_dir}")

    def _initialize_current_file(self):
        """Initialize the current trace log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.log_dir / f"prompt_traces_{timestamp}.jsonl"
        self.traces_in_current_file = 0
        
        # Create file with header
        with open(self.current_file, 'w') as f:
            f.write(f"# Prompt Trace Log - Started at {datetime.now().isoformat()}\n")

    def _cleanup_old_files(self):
        """Clean up old trace log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for log_file in self.log_dir.glob("prompt_traces_*.jsonl*"):
                file_date_str = log_file.stem.split('_')[2:4]  # Extract date part
                if len(file_date_str) >= 2:
                    try:
                        file_date = datetime.strptime(f"{file_date_str[0]}_{file_date_str[1]}", 
                                                    "%Y%m%d_%H%M%S")
                        if file_date < cutoff_date:
                            log_file.unlink()
                            logger.info(f"Deleted old trace file: {log_file}")
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")

    def start_trace(
        self,
        trace_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        original_prompt: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> PromptTrace:
        """Start a new prompt trace.
        
        Args:
            trace_id: Unique trace identifier
            session_id: Session identifier
            user_id: User identifier
            original_prompt: Original user prompt
            context: Additional context
            
        Returns:
            PromptTrace object
        """
        trace = PromptTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            original_prompt=original_prompt,
            normalized_prompt="",
            intent_detection_method="",
            detected_intent="",
            intent_confidence=0.0,
            extracted_parameters={},
            executed_action="",
            action_status=ActionStatus.FAILED,
            action_result=None,
            action_duration=0.0,
            total_processing_time=0.0,
            provider_usage={},
            errors=[],
            fallbacks_used=[],
            context=context or {},
            metadata={}
        )
        
        return trace

    def update_intent(
        self,
        trace: PromptTrace,
        detection_method: str,
        intent: str,
        confidence: float,
        parameters: Dict[str, Any],
        normalized_prompt: str = ""
    ):
        """Update trace with intent detection results.
        
        Args:
            trace: PromptTrace object
            detection_method: Method used for intent detection
            intent: Detected intent
            confidence: Confidence score
            parameters: Extracted parameters
            normalized_prompt: Normalized prompt text
        """
        trace.intent_detection_method = detection_method
        trace.detected_intent = intent
        trace.intent_confidence = confidence
        trace.extracted_parameters = parameters
        trace.normalized_prompt = normalized_prompt

    def update_action(
        self,
        trace: PromptTrace,
        action: str,
        status: ActionStatus,
        result: Optional[Dict[str, Any]] = None,
        duration: float = 0.0
    ):
        """Update trace with action execution results.
        
        Args:
            trace: PromptTrace object
            action: Executed action
            status: Action status
            result: Action result
            duration: Action duration
        """
        trace.executed_action = action
        trace.action_status = status
        trace.action_result = result
        trace.action_duration = duration

    def add_error(self, trace: PromptTrace, error: str):
        """Add an error to the trace.
        
        Args:
            trace: PromptTrace object
            error: Error message
        """
        trace.errors.append(error)

    def add_fallback(self, trace: PromptTrace, fallback_method: str):
        """Add a fallback method to the trace.
        
        Args:
            trace: PromptTrace object
            fallback_method: Fallback method used
        """
        trace.fallbacks_used.append(fallback_method)

    def update_provider_usage(self, trace: PromptTrace, provider: str, count: int = 1):
        """Update provider usage statistics.
        
        Args:
            trace: PromptTrace object
            provider: Provider name
            count: Usage count
        """
        trace.provider_usage[provider] = trace.provider_usage.get(provider, 0) + count

    def complete_trace(self, trace: PromptTrace, total_duration: float):
        """Complete a trace and log it.
        
        Args:
            trace: PromptTrace object
            total_duration: Total processing duration
        """
        trace.total_processing_time = total_duration
        
        # Update statistics
        self.total_traces += 1
        if trace.action_status == ActionStatus.SUCCESS:
            self.successful_traces += 1
        else:
            self.failed_traces += 1
        
        # Log the trace
        self._log_trace(trace)
        
        logger.debug(f"Completed trace {trace.trace_id} in {total_duration:.3f}s")

    def _log_trace(self, trace: PromptTrace):
        """Log a trace to the current file.
        
        Args:
            trace: PromptTrace object to log
        """
        try:
            # Check if we need to rotate files
            if self.traces_in_current_file >= self.max_traces_per_file:
                self._rotate_file()
            
            # Convert trace to JSON
            trace_dict = asdict(trace)
            trace_dict['timestamp'] = trace.timestamp.isoformat()
            trace_dict['action_status'] = trace.action_status.value
            
            # Write to file
            with open(self.current_file, 'a') as f:
                f.write(json.dumps(trace_dict) + '\n')
            
            self.traces_in_current_file += 1
            
        except Exception as e:
            logger.error(f"Error logging trace: {e}")

    def _rotate_file(self):
        """Rotate to a new trace log file."""
        try:
            # Compress current file if enabled
            if self.enable_compression and self.current_file.exists():
                import gzip
                compressed_file = self.current_file.with_suffix('.jsonl.gz')
                with open(self.current_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                self.current_file.unlink()
                logger.info(f"Compressed trace file: {compressed_file}")
            
            # Initialize new file
            self._initialize_current_file()
            
        except Exception as e:
            logger.error(f"Error rotating trace file: {e}")

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get trace logging statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_traces': self.total_traces,
            'successful_traces': self.successful_traces,
            'failed_traces': self.failed_traces,
            'success_rate': self.successful_traces / max(self.total_traces, 1),
            'current_file': str(self.current_file),
            'traces_in_current_file': self.traces_in_current_file
        }

    def search_traces(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        intent: Optional[str] = None,
        status: Optional[ActionStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PromptTrace]:
        """Search for traces based on criteria.
        
        Args:
            session_id: Session ID filter
            user_id: User ID filter
            intent: Intent filter
            status: Action status filter
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of results
            
        Returns:
            List of matching traces
        """
        traces = []
        
        try:
            # Search through all trace files
            for log_file in self.log_dir.glob("prompt_traces_*.jsonl*"):
                if log_file.suffix == '.gz':
                    import gzip
                    with gzip.open(log_file, 'rt') as f:
                        lines = f.readlines()
                else:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                
                # Skip header lines
                lines = [line for line in lines if not line.startswith('#')]
                
                for line in lines:
                    try:
                        trace_dict = json.loads(line.strip())
                        
                        # Apply filters
                        if session_id and trace_dict.get('session_id') != session_id:
                            continue
                        if user_id and trace_dict.get('user_id') != user_id:
                            continue
                        if intent and trace_dict.get('detected_intent') != intent:
                            continue
                        if status and trace_dict.get('action_status') != status.value:
                            continue
                        
                        # Date filters
                        trace_date = datetime.fromisoformat(trace_dict['timestamp'])
                        if start_date and trace_date < start_date:
                            continue
                        if end_date and trace_date > end_date:
                            continue
                        
                        # Convert back to PromptTrace
                        trace = PromptTrace(**trace_dict)
                        trace.timestamp = trace_date
                        trace.action_status = ActionStatus(trace_dict['action_status'])
                        
                        traces.append(trace)
                        
                        if len(traces) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error parsing trace line: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error searching traces: {e}")
        
        return traces[:limit]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        session_traces = self.search_traces(session_id=session_id)
        
        if not session_traces:
            return {}
        
        # Calculate statistics
        total_traces = len(session_traces)
        successful_traces = sum(1 for t in session_traces if t.action_status == ActionStatus.SUCCESS)
        avg_processing_time = sum(t.total_processing_time for t in session_traces) / total_traces
        
        # Most common intents
        intent_counts = {}
        for trace in session_traces:
            intent = trace.detected_intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        most_common_intent = max(intent_counts.items(), key=lambda x: x[1]) if intent_counts else None
        
        return {
            'session_id': session_id,
            'total_traces': total_traces,
            'successful_traces': successful_traces,
            'success_rate': successful_traces / total_traces,
            'avg_processing_time': avg_processing_time,
            'most_common_intent': most_common_intent,
            'first_trace': min(t.timestamp for t in session_traces),
            'last_trace': max(t.timestamp for t in session_traces)
        } 