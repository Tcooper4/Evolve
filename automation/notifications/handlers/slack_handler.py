import logging
import aiohttp
from typing import Optional, Dict, Any

from ..notification_service import Notification

logger = logging.getLogger(__name__)

class SlackHandler:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def __call__(self, notification: Notification) -> None:
        """Send a Slack notification."""
        try:
            payload = self._create_payload(notification)
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Slack API error: {error_text}")
            logger.info(f"Slack notification sent to {notification.recipient}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
            raise

    def _create_payload(self, notification: Notification) -> Dict[str, Any]:
        """Create a Slack message payload from the notification."""
        # Map notification types to Slack colors
        color_map = {
            "info": "#2196F3",      # Blue
            "warning": "#FFA000",   # Orange
            "error": "#F44336",     # Red
            "success": "#4CAF50",   # Green
            "alert": "#E91E63"      # Pink
        }

        # Create attachment
        attachment = {
            "color": color_map.get(notification.type, "#808080"),
            "title": notification.title,
            "text": notification.message,
            "fields": [
                {
                    "title": "Priority",
                    "value": notification.priority,
                    "short": True
                },
                {
                    "title": "Type",
                    "value": notification.type,
                    "short": True
                }
            ],
            "ts": int(notification.created_at.timestamp())
        }

        # Add metadata fields if present
        if notification.metadata:
            for key, value in notification.metadata.items():
                attachment["fields"].append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })

        # Create payload
        payload = {
            "channel": notification.recipient,
            "attachments": [attachment]
        }

        return payload 