import asyncio
import json
import logging
import logging.handlers
import socket
import sys
import threading
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import sentry_sdk
import structlog
from cachetools import TTLCache
from pydantic import BaseModel, Field, validator
from ratelimit import limits, sleep_and_retry
from sentry_sdk.integrations.logging import LoggingIntegration
from structlog import get_logger

logger = logging.getLogger(__name__)


class LogConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    log_dir: str = Field(default="automation/logs")
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)
    rotation: str = Field(default="size")  # size or time
    rotation_interval: str = Field(default="1 day")
    sentry_dsn: Optional[str] = None
    sentry_environment: Optional[str] = None
    sentry_traces_sample_rate: float = Field(default=1.0)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)

    @validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    @validator("rotation")
    def validate_rotation(cls, v):
        valid_rotations = ["size", "time"]
        if v not in valid_rotations:
            raise ValueError(f"Invalid rotation type. Must be one of: {valid_rotations}")
        return v


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: datetime
    level: str
    logger: str
    message: str
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    host: str
    thread: str
    process: int
    metadata: Dict[str, Any] = {}


class AutomationLogging:
    """Logging functionality."""

    def __init__(self, config_path: str = "automation/config/logging.json"):
        """Initialize logging system."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_structlog()
        self.setup_sentry()
        self.setup_cache()
        self.log_entries: List[LogEntry] = []
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> LogConfig:
        """Load logging configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return LogConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load logging config: {str(e)}")
            return None

    def setup_logging(self):
        """Configure logging."""
        try:
            # Create log directory
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Setup root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.config.level)

            # Clear existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.config.format, datefmt=self.config.date_format))
            root_logger.addHandler(console_handler)

            # Add file handler
            if self.config.rotation == "size":
                file_handler = RotatingFileHandler(
                    log_path / "automation.log", maxBytes=self.config.max_bytes, backupCount=self.config.backup_count
                )
            else:
                file_handler = TimedRotatingFileHandler(
                    log_path / "automation.log",
                    when=self.config.rotation_interval,
                    backupCount=self.config.backup_count,
                )

            file_handler.setFormatter(logging.Formatter(self.config.format, datefmt=self.config.date_format))
            root_logger.addHandler(file_handler)

        except Exception as e:
            logger.error(f"Failed to setup logging: {str(e)}")
            return None

    def setup_structlog(self):
        """Setup structured logging."""
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        except Exception as e:
            logger.error(f"Failed to setup structlog: {str(e)}")
            return None

    def setup_sentry(self):
        """Setup Sentry integration."""
        try:
            if self.config.sentry_dsn:
                sentry_sdk.init(
                    dsn=self.config.sentry_dsn,
                    environment=self.config.sentry_environment,
                    traces_sample_rate=self.config.sentry_traces_sample_rate,
                    integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
                )

        except Exception as e:
            logger.error(f"Failed to setup Sentry: {str(e)}")
            return None

    def setup_cache(self):
        """Setup log caching."""
        self.cache = TTLCache(maxsize=self.config.cache_size, ttl=self.config.cache_ttl)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def log(
        self,
        level: str,
        message: str,
        logger_name: str = __name__,
        exception: Optional[Exception] = None,
        metadata: Dict[str, Any] = {},
    ):
        """Log a message."""
        try:
            # Create log entry
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level.upper(),
                logger=logger_name,
                message=message,
                exception=str(exception) if exception else None,
                stack_trace=traceback.format_exc() if exception else None,
                host=socket.gethostname(),
                thread=threading.current_thread().name,
                process=os.getpid(),
                metadata=metadata,
            )

            # Add to entries
            async with self.lock:
                self.log_entries.append(entry)

            # Log using structlog
            logger = get_logger(logger_name)
            log_func = getattr(logger, level.lower())
            log_func(message, exc_info=exception, **metadata)

            # Send to Sentry if error
            if level.upper() in ["ERROR", "CRITICAL"] and self.config.sentry_dsn:
                sentry_sdk.capture_exception(exception)

        except Exception as e:
            logger.error(f"Failed to log message: {str(e)}")
            return None

    async def get_logs(
        self,
        level: Optional[str] = None,
        logger: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[LogEntry]:
        """Get log entries with filtering."""
        try:
            async with self.lock:
                entries = self.log_entries

                if level:
                    entries = [e for e in entries if e.level == level.upper()]

                if logger:
                    entries = [e for e in entries if e.logger == logger]

                if start_time:
                    entries = [e for e in entries if e.timestamp >= start_time]

                if end_time:
                    entries = [e for e in entries if e.timestamp <= end_time]

                return entries[-limit:]

        except Exception as e:
            logger.error(f"Failed to get logs: {str(e)}")
            return None

    async def clear_logs(self):
        """Clear log entries."""
        try:
            async with self.lock:
                self.log_entries.clear()

        except Exception as e:
            logger.error(f"Failed to clear logs: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.log_entries.clear()

            # Flush Sentry
            if self.config.sentry_dsn:
                sentry_sdk.flush()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return None
