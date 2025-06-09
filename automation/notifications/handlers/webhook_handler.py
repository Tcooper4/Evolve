import logging
import aiohttp
from typing import Optional, Dict, Any

from ..notification_service import Notification

logger = logging.getLogger(__name__)

class WebhookHandler:
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout = timeout

    async def __call__(self, notification: Notification) -> None:
        """Send a webhook notification."""
        try:
            payload = self._create_payload(notification)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Webhook error: {error_text}")
            logger.info(f"Webhook notification sent to {self.webhook_url}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")
            raise

    def _create_payload(self, notification: Notification) -> Dict[str, Any]:
        """Create a webhook payload from the notification."""
        payload = {
            "id": notification.id,
            "title": notification.title,
            "message": notification.message,
            "type": notification.type,
            "priority": notification.priority,
            "recipient": notification.recipient,
            "created_at": notification.created_at.isoformat(),
            "metadata": notification.metadata
        }

        if notification.sent_at:
            payload["sent_at"] = notification.sent_at.isoformat()

        return payload 