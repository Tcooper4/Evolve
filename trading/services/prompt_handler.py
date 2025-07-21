"""
Prompt Handler - Batch 16
Enhanced error logging for prompt routing and GPT fallback failures
"""

import hashlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptStatus(Enum):
    """Status of prompt processing."""

    PENDING = "pending"
    ROUTING = "routing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK_FAILED = "fallback_failed"


@dataclass
class PromptError:
    """Error information for prompt processing."""

    error_type: str
    error_message: str
    prompt_hash: str
    timestamp: datetime
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRecord:
    """Record of prompt processing."""

    prompt_id: str
    prompt_content: str
    prompt_hash: str
    status: PromptStatus
    timestamp: datetime
    routing_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    errors: List[PromptError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptHandler:
    """
    Enhanced prompt handler with comprehensive error logging.

    Features:
    - Prompt content hashing for tracking
    - Detailed error logging with tracebacks
    - Routing failure detection
    - GPT fallback failure handling
    - Prompt processing history
    """

    def __init__(
        self,
        enable_error_logging: bool = True,
        log_prompt_content: bool = False,
        max_history_size: int = 1000,
    ):
        """
        Initialize prompt handler.

        Args:
            enable_error_logging: Enable detailed error logging
            log_prompt_content: Whether to log full prompt content
            max_history_size: Maximum number of records to keep
        """
        self.enable_error_logging = enable_error_logging
        self.log_prompt_content = log_prompt_content
        self.max_history_size = max_history_size

        # Storage
        self.prompt_history: Dict[str, PromptRecord] = {}
        self.error_history: List[PromptError] = []

        # Statistics
        self.stats = {
            "total_prompts": 0,
            "successful_routes": 0,
            "routing_failures": 0,
            "processing_failures": 0,
            "fallback_failures": 0,
            "gpt_failures": 0,
        }

        logger.info("PromptHandler initialized with enhanced error logging")

    def _generate_prompt_hash(self, prompt_content: str) -> str:
        """
        Generate hash for prompt content.

        Args:
            prompt_content: The prompt content

        Returns:
            Hash string (first 8 characters)
        """
        return hashlib.md5(prompt_content.encode()).hexdigest()[:8]

    def _generate_prompt_id(self, prompt_content: str) -> str:
        """
        Generate unique ID for prompt.

        Args:
            prompt_content: The prompt content

        Returns:
            Unique prompt ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prompt_hash = self._generate_prompt_hash(prompt_content)
        return f"prompt_{timestamp}_{prompt_hash}"

    def handle_prompt(
        self, prompt_content: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle a prompt with comprehensive error logging.

        Args:
            prompt_content: The prompt to process
            context: Additional context information

        Returns:
            Prompt ID for tracking
        """
        if context is None:
            context = {}

        # Generate prompt ID and hash
        prompt_id = self._generate_prompt_id(prompt_content)
        prompt_hash = self._generate_prompt_hash(prompt_content)

        # Create prompt record
        record = PromptRecord(
            prompt_id=prompt_id,
            prompt_content=prompt_content,
            prompt_hash=prompt_hash,
            status=PromptStatus.PENDING,
            timestamp=datetime.now(),
            metadata=context,
        )

        # Store record
        self.prompt_history[prompt_id] = record
        self.stats["total_prompts"] += 1

        # Log prompt if enabled
        if self.log_prompt_content:
            logger.info(
                f"Processing prompt {prompt_id} (hash: {prompt_hash}): {prompt_content[:100]}..."
            )
        else:
            logger.info(f"Processing prompt {prompt_id} (hash: {prompt_hash})")

        return prompt_id

    def log_routing_failure(
        self,
        prompt_id: str,
        error: Exception,
        routing_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a routing failure with detailed information.

        Args:
            prompt_id: ID of the failed prompt
            error: The exception that occurred
            routing_context: Additional routing context
        """
        if not self.enable_error_logging:
            return

        if prompt_id not in self.prompt_history:
            logger.warning(
                f"Attempted to log routing failure for unknown prompt: {prompt_id}"
            )
            return

        record = self.prompt_history[prompt_id]
        record.status = PromptStatus.FAILED

        # Create error record
        error_record = PromptError(
            error_type="routing_failure",
            error_message=str(error),
            prompt_hash=record.prompt_hash,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            context=routing_context or {},
        )

        record.errors.append(error_record)
        self.error_history.append(error_record)
        self.stats["routing_failures"] += 1

        # Log detailed error information
        logger.error(
            f"Routing failure for prompt {prompt_id} (hash: {record.prompt_hash}): {error}",
            extra={
                "prompt_id": prompt_id,
                "prompt_hash": record.prompt_hash,
                "error_type": "routing_failure",
                "traceback": traceback.format_exc(),
                "context": routing_context,
            },
        )

        # Log prompt content if enabled
        if self.log_prompt_content:
            logger.error(f"Failed prompt content: {record.prompt_content}")

    def log_processing_failure(
        self,
        prompt_id: str,
        error: Exception,
        processing_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a processing failure with detailed information.

        Args:
            prompt_id: ID of the failed prompt
            error: The exception that occurred
            processing_context: Additional processing context
        """
        if not self.enable_error_logging:
            return

        if prompt_id not in self.prompt_history:
            logger.warning(
                f"Attempted to log processing failure for unknown prompt: {prompt_id}"
            )
            return

        record = self.prompt_history[prompt_id]
        record.status = PromptStatus.FAILED

        # Create error record
        error_record = PromptError(
            error_type="processing_failure",
            error_message=str(error),
            prompt_hash=record.prompt_hash,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            context=processing_context or {},
        )

        record.errors.append(error_record)
        self.error_history.append(error_record)
        self.stats["processing_failures"] += 1

        # Log detailed error information
        logger.error(
            f"Processing failure for prompt {prompt_id} (hash: {record.prompt_hash}): {error}",
            extra={
                "prompt_id": prompt_id,
                "prompt_hash": record.prompt_hash,
                "error_type": "processing_failure",
                "traceback": traceback.format_exc(),
                "context": processing_context,
            },
        )

    def log_gpt_fallback_failure(
        self,
        prompt_id: str,
        error: Exception,
        fallback_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a GPT fallback failure with detailed information.

        Args:
            prompt_id: ID of the failed prompt
            error: The exception that occurred
            fallback_context: Additional fallback context
        """
        if not self.enable_error_logging:
            return

        if prompt_id not in self.prompt_history:
            logger.warning(
                f"Attempted to log GPT fallback failure for unknown prompt: {prompt_id}"
            )
            return

        record = self.prompt_history[prompt_id]
        record.status = PromptStatus.FALLBACK_FAILED

        # Create error record
        error_record = PromptError(
            error_type="gpt_fallback_failure",
            error_message=str(error),
            prompt_hash=record.prompt_hash,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            context=fallback_context or {},
        )

        record.errors.append(error_record)
        self.error_history.append(error_record)
        self.stats["fallback_failures"] += 1
        self.stats["gpt_failures"] += 1

        # Log detailed error information
        logger.error(
            f"GPT fallback failure for prompt {prompt_id} (hash: {record.prompt_hash}): {error}",
            extra={
                "prompt_id": prompt_id,
                "prompt_hash": record.prompt_hash,
                "error_type": "gpt_fallback_failure",
                "traceback": traceback.format_exc(),
                "context": fallback_context,
            },
        )

        # Log prompt content if enabled
        if self.log_prompt_content:
            logger.error(f"Failed GPT fallback prompt content: {record.prompt_content}")

    def log_successful_routing(self, prompt_id: str, routing_result: Dict[str, Any]):
        """
        Log successful routing.

        Args:
            prompt_id: ID of the successful prompt
            routing_result: Routing result information
        """
        if prompt_id not in self.prompt_history:
            logger.warning(
                f"Attempted to log successful routing for unknown prompt: {prompt_id}"
            )
            return

        record = self.prompt_history[prompt_id]
        record.status = PromptStatus.ROUTING
        record.routing_result = routing_result
        self.stats["successful_routes"] += 1

        logger.info(
            f"Successful routing for prompt {prompt_id} (hash: {record.prompt_hash})"
        )

    def log_successful_processing(
        self, prompt_id: str, processing_result: Dict[str, Any]
    ):
        """
        Log successful processing.

        Args:
            prompt_id: ID of the successful prompt
            processing_result: Processing result information
        """
        if prompt_id not in self.prompt_history:
            logger.warning(
                f"Attempted to log successful processing for unknown prompt: {prompt_id}"
            )
            return

        record = self.prompt_history[prompt_id]
        record.status = PromptStatus.COMPLETED
        record.processing_result = processing_result

        logger.info(
            f"Successful processing for prompt {prompt_id} (hash: {record.prompt_hash})"
        )

    def get_prompt_record(self, prompt_id: str) -> Optional[PromptRecord]:
        """
        Get prompt record by ID.

        Args:
            prompt_id: ID of the prompt

        Returns:
            PromptRecord or None if not found
        """
        return self.prompt_history.get(prompt_id)

    def get_prompts_by_hash(self, prompt_hash: str) -> List[PromptRecord]:
        """
        Get all prompts with a specific hash.

        Args:
            prompt_hash: Hash to search for

        Returns:
            List of matching PromptRecord objects
        """
        return [
            record
            for record in self.prompt_history.values()
            if record.prompt_hash == prompt_hash
        ]

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of errors.

        Returns:
            Dictionary with error statistics
        """
        error_types = {}
        for error in self.error_history:
            error_type = error.error_type
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1

        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "stats": self.stats.copy(),
        }

    def get_failed_prompts(
        self, error_type: Optional[str] = None
    ) -> List[PromptRecord]:
        """
        Get failed prompts.

        Args:
            error_type: Filter by specific error type

        Returns:
            List of failed PromptRecord objects
        """
        failed = []
        for record in self.prompt_history.values():
            if record.status in [PromptStatus.FAILED, PromptStatus.FALLBACK_FAILED]:
                if error_type is None or any(
                    e.error_type == error_type for e in record.errors
                ):
                    failed.append(record)
        return failed

    def clear_history(self, keep_recent: int = 100):
        """
        Clear old history while keeping recent records.

        Args:
            keep_recent: Number of recent records to keep
        """
        if len(self.prompt_history) <= keep_recent:
            return

        # Sort by timestamp and keep recent
        sorted_records = sorted(
            self.prompt_history.items(), key=lambda x: x[1].timestamp, reverse=True
        )

        # Keep only recent records
        self.prompt_history = dict(sorted_records[:keep_recent])

        # Clear old errors
        if len(self.error_history) > keep_recent:
            self.error_history = self.error_history[-keep_recent:]

        logger.info(f"Cleared history, kept {keep_recent} recent records")

    def export_error_log(self, format: str = "json") -> str:
        """
        Export error log.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported error log as string
        """
        if format == "json":
            data = []
            for error in self.error_history:
                data.append(
                    {
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                        "prompt_hash": error.prompt_hash,
                        "timestamp": error.timestamp.isoformat(),
                        "traceback": error.traceback,
                        "context": error.context,
                    }
                )
            return json.dumps(data, indent=2)

        elif format == "csv":
            import pandas as pd

            data = []
            for error in self.error_history:
                data.append(
                    {
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                        "prompt_hash": error.prompt_hash,
                        "timestamp": error.timestamp.isoformat(),
                        "traceback": error.traceback,
                    }
                )
            df = pd.DataFrame(data)
            return df.to_csv(index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")


def create_prompt_handler(enable_error_logging: bool = True) -> PromptHandler:
    """Factory function to create a prompt handler."""
    return PromptHandler(enable_error_logging=enable_error_logging)
