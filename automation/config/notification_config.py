from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import os
import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator, root_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Supported notification channels."""
    WEB = "web"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    TEAMS = "teams"

class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationType(Enum):
    """Types of notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class SecurityConfig(BaseModel):
    """Security configuration for notification channels."""
    verify_ssl: bool = True
    allowed_ips: List[str] = Field(default_factory=list)
    blocked_ips: List[str] = Field(default_factory=list)
    rate_limit: int = 100  # per minute
    max_payload_size: int = 1048576  # 1MB
    require_tls: bool = True
    allowed_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    max_recipients: int = 100

class EmailConfig(BaseModel):
    """Email notification configuration."""
    smtp_server: str
    smtp_port: int
    use_tls: bool = True
    use_ssl: bool = False
    sender_email: str
    sender_password: str
    timeout: int = 30
    max_connections: int = 5
    retry_attempts: int = 3
    retry_delay: int = 5
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    templates: Dict[str, str] = Field(default_factory=dict)

    @validator('smtp_port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

class SlackConfig(BaseModel):
    """Slack notification configuration."""
    webhook_url: str
    channel: str = "#notifications"
    username: str = "Automation Bot"
    timeout: int = 10
    retry_attempts: int = 3
    retry_delay: int = 5
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    templates: Dict[str, str] = Field(default_factory=dict)

class WebhookConfig(BaseModel):
    """Webhook notification configuration."""
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    templates: Dict[str, str] = Field(default_factory=dict)

class ChannelConfig(BaseModel):
    """Configuration for a notification channel."""
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 30
    security: SecurityConfig = Field(default_factory=SecurityConfig)

class PriorityConfig(BaseModel):
    """Configuration for notification priorities."""
    retry_attempts: int
    retry_delay: int
    timeout: int
    rate_limit: int

class MonitoringConfig(BaseModel):
    """Monitoring configuration for notifications."""
    enabled: bool = True
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)

class LoggingConfig(BaseModel):
    """Logging configuration for notifications."""
    level: str = "INFO"
    format: str = "json"
    handlers: List[Dict[str, Any]] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)

class NotificationConfig(BaseModel):
    """Main notification configuration."""
    version: str = "1.0.0"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    
    # Channel configurations
    email: Optional[EmailConfig] = None
    slack: Optional[SlackConfig] = None
    webhook: Optional[WebhookConfig] = None
    channels: Dict[str, ChannelConfig] = Field(default_factory=dict)
    
    # Priority configurations
    priorities: Dict[str, PriorityConfig] = Field(default_factory=dict)
    
    # Monitoring and logging
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Settings
    retention_days: int = 30
    max_notifications_per_user: int = 1000
    batch_size: int = 50
    cleanup_interval_hours: int = 24
    
    @root_validator
    def validate_config(cls, values):
        """Validate the entire configuration."""
        # Ensure at least one channel is enabled
        enabled_channels = [
            channel for channel, config in values.get('channels', {}).items()
            if config.enabled
        ]
        if not enabled_channels:
            raise ValueError("At least one notification channel must be enabled")
        
        # Validate priority configurations
        required_priorities = ["low", "medium", "high", "critical"]
        for priority in required_priorities:
            if priority not in values.get('priorities', {}):
                raise ValueError(f"Missing configuration for {priority} priority")
        
        return values

def load_notification_config(config_path: str = "config/notification_config.yaml") -> NotificationConfig:
    """Load notification configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        return NotificationConfig(**config_data)
    except Exception as e:
        logger.error(f"Error loading notification config: {e}")
        return NotificationConfig()

def save_notification_config(config: NotificationConfig, config_path: str = "config/notification_config.yaml") -> None:
    """Save notification configuration to YAML file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Convert config to dictionary
        config_dict = config.dict()
        
        # Save to YAML file
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Notification configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving notification config: {e}")
        raise

# Default configuration
DEFAULT_CONFIG = NotificationConfig(
    channels={
        "web": ChannelConfig(),
        "email": ChannelConfig(),
        "slack": ChannelConfig(),
        "discord": ChannelConfig(),
        "sms": ChannelConfig(enabled=False),
        "teams": ChannelConfig(enabled=False)
    },
    priorities={
        "low": PriorityConfig(retry_attempts=1, retry_delay=600, timeout=60, rate_limit=100),
        "medium": PriorityConfig(retry_attempts=2, retry_delay=300, timeout=30, rate_limit=50),
        "high": PriorityConfig(retry_attempts=3, retry_delay=60, timeout=15, rate_limit=25),
        "critical": PriorityConfig(retry_attempts=5, retry_delay=30, timeout=10, rate_limit=10)
    }
)

# Example configuration
EXAMPLE_CONFIG = NotificationConfig(
    email=EmailConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email="notifications@example.com",
        sender_password="your-app-specific-password",
        security=SecurityConfig(
            verify_ssl=True,
            rate_limit=100,
            require_tls=True
        )
    ),
    slack=SlackConfig(
        webhook_url="https://hooks.slack.com/services/your-webhook-url",
        channel="#automation-notifications",
        security=SecurityConfig(
            verify_webhook=True,
            rate_limit=50
        )
    ),
    channels={
        "web": ChannelConfig(enabled=True),
        "email": ChannelConfig(enabled=True),
        "slack": ChannelConfig(enabled=True),
        "discord": ChannelConfig(enabled=False),
        "sms": ChannelConfig(enabled=False),
        "teams": ChannelConfig(enabled=False)
    },
    priorities={
        "low": PriorityConfig(retry_attempts=1, retry_delay=600, timeout=60, rate_limit=100),
        "medium": PriorityConfig(retry_attempts=2, retry_delay=300, timeout=30, rate_limit=50),
        "high": PriorityConfig(retry_attempts=3, retry_delay=60, timeout=15, rate_limit=25),
        "critical": PriorityConfig(retry_attempts=5, retry_delay=30, timeout=10, rate_limit=10)
    },
    retention_days=60,
    max_notifications_per_user=2000,
    batch_size=100,
    cleanup_interval_hours=12
) 