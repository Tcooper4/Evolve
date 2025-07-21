import logging
import logging.handlers
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator

from utils.launch_utils import setup_logging

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
            raise ValueError(
                f"Invalid rotation type. Must be one of: {valid_rotations}"
            )
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


class AutomationLoggingService:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger("automation")

    def setup_logging(self):
        return setup_logging(service_name="service")

    def setup_structlog(self):
        """Set up structlog for automation logging service."""
        # Structlog setup logic here
