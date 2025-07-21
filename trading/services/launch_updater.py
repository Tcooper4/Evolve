#!/usr/bin/env python3
"""
Updater Service Launcher

Launches the UpdaterService as a standalone process.
Enhanced with logging for failed model re-training attempts.
"""

import logging
import os
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from services.updater_service import UpdaterService

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/updater_service.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class ModelRetrainingLogger:
    """Handles logging for model re-training attempts and failures."""

    def __init__(self, log_file: str = "logs/model_retraining_failures.log"):
        self.log_file = log_file
        self.failure_count = 0
        self.success_count = 0

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create dedicated logger for retraining failures
        self.retraining_logger = logging.getLogger("model_retraining")
        self.retraining_logger.setLevel(logging.ERROR)

        # File handler for retraining failures
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.retraining_logger.addHandler(file_handler)

    def log_retraining_attempt(
        self,
        model_id: str,
        model_type: str,
        attempt_number: int,
        context: Dict[str, Any] = None,
    ):
        """Log a model re-training attempt."""
        datetime.now().isoformat()
        logger.info(
            f"🔄 Model re-training attempt | Model: {model_id} | Type: {model_type} | Attempt: {attempt_number}"
        )

        if context:
            logger.debug(f"📋 Retraining context: {context}")

    def log_retraining_success(
        self,
        model_id: str,
        model_type: str,
        execution_time: float,
        metrics: Dict[str, Any] = None,
    ):
        """Log a successful model re-training."""
        timestamp = datetime.now().isoformat()
        self.success_count += 1

        logger.info(
            f"✅ Model re-training successful | Model: {model_id} | Type: {model_type} | Time: {execution_time:.2f}s"
        )

        if metrics:
            logger.info(f"📊 Retraining metrics: {metrics}")

        # Log to dedicated file
        self.retraining_logger.info(
            f"SUCCESS | Model: {model_id} | Type: {model_type} | "
            f"Execution Time: {execution_time:.2f}s | Timestamp: {timestamp}"
        )

    def log_retraining_failure(
        self,
        model_id: str,
        model_type: str,
        error: Exception,
        attempt_number: int = 1,
        context: Dict[str, Any] = None,
        traceback_str: str = None,
    ):
        """Log a failed model re-training attempt with full details."""
        timestamp = datetime.now().isoformat()
        self.failure_count += 1

        # Get traceback if not provided
        if not traceback_str:
            traceback_str = traceback.format_exc()

        # Log to main logger
        logger.error(
            f"❌ Model re-training failed | Model: {model_id} | Type: {model_type} | Attempt: {attempt_number}"
        )
        logger.error(f"💥 Error: {str(error)}")

        # Log detailed failure to dedicated file
        failure_details = {
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "attempt_number": attempt_number,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "traceback": traceback_str,
        }

        # Format detailed log entry
        log_entry = (
            f"FAILURE | Model: {model_id} | Type: {model_type} | "
            f"Attempt: {attempt_number} | Error: {type(error).__name__}: {str(error)} | "
            f"Timestamp: {timestamp}\n"
            f"Context: {context}\n"
            f"Traceback:\n{traceback_str}\n"
            f"{'=' * 80}\n"
        )

        self.retraining_logger.error(log_entry)

        # Also log to main logger with reduced detail
        logger.error(f"📝 Full failure details logged to: {self.log_file}")

    def log_retraining_timeout(
        self,
        model_id: str,
        model_type: str,
        timeout_duration: float,
        attempt_number: int = 1,
    ):
        """Log a model re-training timeout."""
        timestamp = datetime.now().isoformat()
        self.failure_count += 1

        logger.warning(
            f"⏰ Model re-training timeout | Model: {model_id} | Type: {model_type} | "
            f"Timeout: {timeout_duration}s | Attempt: {attempt_number}"
        )

        # Log to dedicated file
        self.retraining_logger.error(
            f"TIMEOUT | Model: {model_id} | Type: {model_type} | "
            f"Timeout Duration: {timeout_duration}s | Attempt: {attempt_number} | "
            f"Timestamp: {timestamp}"
        )

    def log_retraining_abandoned(
        self, model_id: str, model_type: str, max_attempts: int, total_failures: int
    ):
        """Log when model re-training is abandoned after max attempts."""
        timestamp = datetime.now().isoformat()

        logger.error(
            f"🚫 Model re-training abandoned | Model: {model_id} | Type: {model_type} | "
            f"Max Attempts: {max_attempts} | Total Failures: {total_failures}"
        )

        # Log to dedicated file
        self.retraining_logger.error(
            f"ABANDONED | Model: {model_id} | Type: {model_type} | "
            f"Max Attempts: {max_attempts} | Total Failures: {total_failures} | "
            f"Timestamp: {timestamp}"
        )

    def get_retraining_stats(self) -> Dict[str, Any]:
        """Get statistics about retraining attempts."""
        return {
            "total_attempts": self.success_count + self.failure_count,
            "successful_attempts": self.success_count,
            "failed_attempts": self.failure_count,
            "success_rate": (
                self.success_count / (self.success_count + self.failure_count)
                if (self.success_count + self.failure_count) > 0
                else 0
            ),
            "failure_log_file": self.log_file,
        }

    def log_stats_summary(self):
        """Log a summary of retraining statistics."""
        stats = self.get_retraining_stats()

        logger.info("📊 Model Re-training Statistics Summary")
        logger.info("=" * 50)
        logger.info(f"Total Attempts: {stats['total_attempts']}")
        logger.info(f"Successful: {stats['successful_attempts']}")
        logger.info(f"Failed: {stats['failed_attempts']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"Failure Log: {stats['failure_log_file']}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, "service"):
        signal_handler.service.stop()

    # Log final statistics if retraining logger exists
    if hasattr(signal_handler, "retraining_logger"):
        signal_handler.retraining_logger.log_stats_summary()

    sys.exit(0)


def main():
    """Main function to launch the UpdaterService."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger.info("Starting UpdaterService...")

        # Initialize retraining logger
        retraining_logger = ModelRetrainingLogger()

        # Initialize the service with retraining logger
        service = UpdaterService(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            retraining_logger=retraining_logger,
        )

        # Store references for signal handler
        signal_handler.service = service
        signal_handler.retraining_logger = retraining_logger

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the service
        service.start()

        logger.info("UpdaterService started successfully")
        logger.info(
            f"Listening on channels: {service.input_channel}, {service.control_channel}"
        )
        logger.info("Model re-training failure logging enabled")

        # Keep the service running
        try:
            while service.is_running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            service.stop()

    except Exception as e:
        logger.error(f"Error starting UpdaterService: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
