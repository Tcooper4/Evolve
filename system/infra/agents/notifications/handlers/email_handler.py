import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from system.infra.agents.notifications.notification_service import Notification

logger = logging.getLogger(__name__)


class EmailHandler:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls

    async def __call__(self, notification: Notification) -> None:
        """Send an email notification."""
        try:
            message = self._create_message(notification)
            await self._send_email(message)
            logger.info(f"Email notification sent to {notification.recipient}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            raise

    def _create_message(self, notification: Notification) -> MIMEMultipart:
        """Create an email message from the notification."""
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = notification.recipient
        message["Subject"] = notification.title

        # Add priority header
        priority_map = {"low": "5", "medium": "3", "high": "1", "critical": "1"}
        message["X-Priority"] = priority_map.get(notification.priority, "3")

        # Create HTML content
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .notification {{ padding: 20px; border-radius: 5px; }}
                    .info {{ background-color: #e3f2fd; }}
                    .warning {{ background-color: #fff3e0; }}
                    .error {{ background-color: #ffebee; }}
                    .success {{ background-color: #e8f5e9; }}
                    .alert {{ background-color: #fce4ec; }}
                </style>
            </head>
            <body>
                <div class="notification {notification.type}">
                    <h2>{notification.title}</h2>
                    <p>{notification.message}</p>
                    <p><small>Priority: {notification.priority}</small></p>
                </div>
            </body>
        </html>
        """

        message.attach(MIMEText(html_content, "html"))
        return message

    async def _send_email(self, message: MIMEMultipart) -> None:
        """Send the email message."""
        try:
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host, port=self.smtp_port, use_tls=self.use_tls
            ) as smtp:
                await smtp.login(self.username, self.password)
                await smtp.send_message(message)
        except Exception as e:
            logger.error(f"SMTP error: {str(e)}")
            raise
