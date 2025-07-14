import asyncio
import json
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from cachetools import TTLCache
from pydantic import BaseModel, Field, validator
from ratelimit import limits, sleep_and_retry
from twilio.rest import Client

from trading.automation_core import AutomationCore
from trading.automation_monitoring import AutomationMonitoring

logger = logging.getLogger(__name__)


class NotificationConfig(BaseModel):
    """Configuration for notifications."""

    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_username: str
    smtp_password: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    slack_webhook_url: Optional[str] = None
    teams_webhook_url: Optional[str] = None
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=60)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)
    jwt_secret: str
    jwt_algorithm: str = Field(default="HS256")

    @validator(
        "smtp_username",
        "smtp_password",
        "twilio_account_sid",
        "twilio_auth_token",
        "twilio_phone_number",
        "jwt_secret",
    )
    def validate_required_fields(cls, v, field):
        if not v:
            raise ValueError(f"{field.name} is required")
        return v


class NotificationTemplate(BaseModel):
    """Template for notifications."""

    id: str
    name: str
    type: str
    subject: str
    body: str
    variables: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Notification(BaseModel):
    """Notification model."""

    id: str
    template_id: str
    recipient: str
    subject: str
    body: str
    type: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AutomationNotification:
    """Notification and alert handling functionality."""

    def __init__(
        self,
        core: AutomationCore,
        monitoring: AutomationMonitoring,
        config_path: str = "automation/config/notification.json",
    ):
        """Initialize notification system."""
        self.core = core
        self.monitoring = monitoring
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_cache()
        self.setup_clients()
        self.templates: Dict[str, NotificationTemplate] = {}
        self.notifications: Dict[str, Notification] = {}
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> NotificationConfig:
        """Load notification configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return NotificationConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load notification config: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path / "notification.log"),
                logging.StreamHandler(),
            ],
        )

    def setup_cache(self):
        """Setup notification caching."""
        self.cache = TTLCache(maxsize=self.config.cache_size, ttl=self.config.cache_ttl)

    def setup_clients(self):
        """Setup notification clients."""
        try:
            # Setup Twilio client
            self.twilio_client = Client(
                self.config.twilio_account_sid, self.config.twilio_auth_token
            )

            # Setup SMTP connection
            # self.smtp_server = smtplib.SMTP(
            #     self.config.smtp_host, self.config.smtp_port
            # )
            # self.smtp_server.starttls()
            # self.smtp_server.login(self.config.smtp_username, self.config.smtp_password)

        except Exception as e:
            logger.error(f"Failed to setup notification clients: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_template(
        self, name: str, type: str, subject: str, body: str, variables: List[str] = []
    ) -> NotificationTemplate:
        """Create a notification template."""
        try:
            template = NotificationTemplate(
                id=str(len(self.templates) + 1),
                name=name,
                type=type,
                subject=subject,
                body=body,
                variables=variables,
            )

            self.templates[template.id] = template
            return template

        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def send_notification(
        self,
        template_id: str,
        recipient: str,
        variables: Dict[str, Any] = {},
        metadata: Dict[str, Any] = {},
    ) -> Notification:
        """Send a notification."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")

            # Create notification
            notification = Notification(
                id=str(len(self.notifications) + 1),
                template_id=template_id,
                recipient=recipient,
                subject=self._format_template(template.subject, variables),
                body=self._format_template(template.body, variables),
                type=template.type,
                metadata=metadata,
            )

            # Send based on type
            if template.type == "email":
                await self._send_email(notification)
            elif template.type == "sms":
                await self._send_sms(notification)
            elif template.type == "slack":
                await self._send_slack(notification)
            elif template.type == "teams":
                await self._send_teams(notification)
            else:
                raise ValueError(f"Unsupported notification type: {template.type}")

            # Update status
            notification.status = "sent"
            notification.sent_at = datetime.now()
            self.notifications[notification.id] = notification

            return notification

        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
            if notification:
                notification.status = "failed"
                notification.error = str(e)
                self.notifications[notification.id] = notification
            raise

    def _format_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Format template with variables."""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {str(e)}")

    async def _send_email(self, notification: Notification):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.smtp_username
            msg["To"] = notification.recipient
            msg["Subject"] = notification.subject

            msg.attach(MIMEText(notification.body, "html"))

            # self.smtp_server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise

    async def _send_sms(self, notification: Notification):
        """Send SMS notification."""
        try:
            self.twilio_client.messages.create(
                body=notification.body,
                from_=self.config.twilio_phone_number,
                to=notification.recipient,
            )

        except Exception as e:
            logger.error(f"Failed to send SMS: {str(e)}")
            raise

    async def _send_slack(self, notification: Notification):
        """Send Slack notification."""
        if not self.config.slack_webhook_url:
            raise ValueError("Slack webhook URL not configured")

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": notification.body,
                    "attachments": [
                        {"title": notification.subject, "color": "#36a64f"}
                    ],
                }

                async with session.post(
                    self.config.slack_webhook_url, json=payload
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Slack API error: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
            raise

    async def _send_teams(self, notification: Notification):
        """Send Microsoft Teams notification."""
        if not self.config.teams_webhook_url:
            raise ValueError("Teams webhook URL not configured")

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "title": notification.subject,
                    "text": notification.body,
                    "themeColor": "0076D7",
                }

                async with session.post(
                    self.config.teams_webhook_url, json=payload
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Teams API error: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Teams notification: {str(e)}")
            raise

    async def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID."""
        return self.notifications.get(notification_id)

    async def get_notifications(
        self, status: Optional[str] = None, type: Optional[str] = None
    ) -> List[Notification]:
        """Get notifications with optional filtering."""
        notifications = list(self.notifications.values())

        if status:
            notifications = [n for n in notifications if n.status == status]

        if type:
            notifications = [n for n in notifications if n.type == type]

        return notifications

    async def retry_notification(self, notification_id: str) -> Notification:
        """Retry failed notification."""
        notification = self.notifications.get(notification_id)
        if not notification:
            raise ValueError(f"Notification {notification_id} not found")

        if notification.status != "failed":
            raise ValueError("Can only retry failed notifications")

        return await self.send_notification(
            notification.template_id, notification.recipient, notification.metadata
        )

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Close SMTP connection
            # self.smtp_server.quit()

            # Clear caches
            self.cache.clear()
            self.templates.clear()
            self.notifications.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
