import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .notification_service import NotificationService, Notification

logger = logging.getLogger(__name__)

class NotificationCleanupService:
    def __init__(
        self,
        notification_service: NotificationService,
        retention_days: int = 30,
        cleanup_interval: int = 3600
    ):
        self._notification_service = notification_service
        self._retention_days = retention_days
        self._cleanup_interval = cleanup_interval
        self._running: bool = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the cleanup service."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._run_cleanup())
        logger.info("Notification cleanup service started")

    async def stop(self):
        """Stop the cleanup service."""
        if not self._running:
            return

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Notification cleanup service stopped")

    async def _run_cleanup(self):
        """Run the cleanup process periodically."""
        while self._running:
            try:
                await self._cleanup_notifications()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup process: {str(e)}")
                await asyncio.sleep(60)

    async def _cleanup_notifications(self):
        """Clean up old notifications."""
        cutoff_date = datetime.utcnow() - timedelta(days=self._retention_days)
        
        # Get all notifications
        notifications = await self._notification_service.get_notifications()
        
        # Filter and delete old notifications
        for notification in notifications:
            if notification.created_at < cutoff_date:
                await self._notification_service.delete_notification(notification.id)
                logger.info(f"Cleaned up old notification: {notification.id}")

    async def cleanup_now(self):
        """Trigger an immediate cleanup."""
        await self._cleanup_notifications()
        logger.info("Manual cleanup completed")

    def set_retention_days(self, days: int):
        """Set the retention period for notifications."""
        if days < 1:
            raise ValueError("Retention period must be at least 1 day")
        self._retention_days = days
        logger.info(f"Set notification retention period to {days} days")

    def set_cleanup_interval(self, seconds: int):
        """Set the cleanup interval."""
        if seconds < 60:
            raise ValueError("Cleanup interval must be at least 60 seconds")
        self._cleanup_interval = seconds
        logger.info(f"Set cleanup interval to {seconds} seconds") 