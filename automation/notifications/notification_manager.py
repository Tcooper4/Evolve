import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import redis
from enum import Enum
from automation.config.notification_config import NotificationConfig, NotificationChannel

class NotificationType(Enum):
    TASK = "task"
    SYSTEM = "system"
    ALERT = "alert"
    USER = "user"

class NotificationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationManager:
    def __init__(self, redis_client: redis.Redis, config: Optional[NotificationConfig] = None):
        self.redis = redis_client
        self.config = config or NotificationConfig()
        self.logger = logging.getLogger(__name__)
        self._notification_channels = {}
        
    async def create_notification(
        self,
        title: str,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        data: Optional[Dict] = None,
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict:
        """Create a new notification."""
        try:
            notification = {
                "id": f"{notification_type.value}_{datetime.now().timestamp()}",
                "title": title,
                "message": message,
                "type": notification_type.value,
                "priority": priority.value,
                "data": data or {},
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "read": False,
                "channels": [channel.value for channel in (channels or self.config.enabled_channels)]
            }
            
            # Store notification
            notification_key = f"notification:{notification['id']}"
            self.redis.set(notification_key, json.dumps(notification))
            
            # Add to user's notifications if user_id is provided
            if user_id:
                user_notifications_key = f"user_notifications:{user_id}"
                self.redis.lpush(user_notifications_key, notification['id'])
                self.redis.ltrim(user_notifications_key, 0, self.config.max_notifications_per_user - 1)
            
            # Add to type-specific list
            type_key = f"notifications:{notification_type.value}"
            self.redis.lpush(type_key, notification['id'])
            self.redis.ltrim(type_key, 0, self.config.max_notifications_per_user - 1)
            
            # Publish to Redis channel for real-time updates
            channel = f"notifications:{user_id}" if user_id else "notifications:all"
            await self._publish_notification(channel, notification)
            
            # Send through enabled channels
            await self._send_through_channels(notification)
            
            self.logger.info(f"Created notification: {notification['id']}")
            return notification
            
        except Exception as e:
            self.logger.error(f"Error creating notification: {str(e)}")
            raise

    async def _send_through_channels(self, notification: Dict) -> None:
        """Send notification through enabled channels."""
        for channel in notification['channels']:
            try:
                if channel == NotificationChannel.EMAIL.value:
                    await self._send_email(notification)
                elif channel == NotificationChannel.SLACK.value:
                    await self._send_slack(notification)
                elif channel == NotificationChannel.DISCORD.value:
                    await self._send_discord(notification)
            except Exception as e:
                self.logger.error(f"Error sending notification through {channel}: {str(e)}")

    async def _send_email(self, notification: Dict) -> None:
        """Send notification via email."""
        if not self.config.email_config.get("sender_email"):
            return
            
        # Email sending implementation
        pass

    async def _send_slack(self, notification: Dict) -> None:
        """Send notification via Slack."""
        if not self.config.slack_config.get("webhook_url"):
            return
            
        # Slack sending implementation
        pass

    async def _send_discord(self, notification: Dict) -> None:
        """Send notification via Discord."""
        if not self.config.discord_config.get("webhook_url"):
            return
            
        # Discord sending implementation
        pass

    async def get_notification(self, notification_id: str) -> Optional[Dict]:
        """Get a specific notification."""
        try:
            notification_key = f"notification:{notification_id}"
            notification_data = self.redis.get(notification_key)
            
            if not notification_data:
                return None
            
            return json.loads(notification_data)
            
        except Exception as e:
            self.logger.error(f"Error getting notification: {str(e)}")
            return None
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False
    ) -> List[Dict]:
        """Get notifications for a specific user."""
        try:
            user_notifications_key = f"user_notifications:{user_id}"
            notification_ids = self.redis.lrange(user_notifications_key, offset, offset + limit - 1)
            
            notifications = []
            for notification_id in notification_ids:
                notification = await self.get_notification(notification_id.decode())
                if notification and (not unread_only or not notification['read']):
                    notifications.append(notification)
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"Error getting user notifications: {str(e)}")
            return []
    
    async def mark_as_read(self, notification_id: str, user_id: Optional[str] = None) -> bool:
        """Mark a notification as read."""
        try:
            notification = await self.get_notification(notification_id)
            if not notification:
                return False
            
            notification['read'] = True
            notification_key = f"notification:{notification_id}"
            self.redis.set(notification_key, json.dumps(notification))
            
            # Publish update
            channel = f"notifications:{user_id}" if user_id else "notifications:all"
            await self._publish_notification(channel, notification, "update")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error marking notification as read: {str(e)}")
            return False
    
    async def delete_notification(self, notification_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a notification."""
        try:
            notification = await self.get_notification(notification_id)
            if not notification:
                return False
            
            # Remove from Redis
            notification_key = f"notification:{notification_id}"
            self.redis.delete(notification_key)
            
            # Remove from user's notifications
            if user_id:
                user_notifications_key = f"user_notifications:{user_id}"
                self.redis.lrem(user_notifications_key, 0, notification_id)
            
            # Remove from type-specific list
            type_key = f"notifications:{notification['type']}"
            self.redis.lrem(type_key, 0, notification_id)
            
            # Publish deletion
            channel = f"notifications:{user_id}" if user_id else "notifications:all"
            await self._publish_notification(channel, {"id": notification_id}, "delete")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting notification: {str(e)}")
            return False
    
    async def subscribe(self, user_id: str, callback) -> None:
        """Subscribe to notifications for a specific user."""
        try:
            channel = f"notifications:{user_id}"
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            
            self._notification_channels[user_id] = {
                "pubsub": pubsub,
                "callback": callback
            }
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages(user_id))
            
        except Exception as e:
            self.logger.error(f"Error subscribing to notifications: {str(e)}")
            raise
    
    async def unsubscribe(self, user_id: str) -> None:
        """Unsubscribe from notifications."""
        try:
            if user_id in self._notification_channels:
                channel_data = self._notification_channels[user_id]
                await channel_data["pubsub"].unsubscribe()
                del self._notification_channels[user_id]
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from notifications: {str(e)}")
            raise
    
    async def _publish_notification(self, channel: str, data: Dict, action: str = "create") -> None:
        """Publish notification to Redis channel."""
        try:
            message = {
                "action": action,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            await self.redis.publish(channel, json.dumps(message))
            
        except Exception as e:
            self.logger.error(f"Error publishing notification: {str(e)}")
            raise
    
    async def _listen_for_messages(self, user_id: str) -> None:
        """Listen for messages on the notification channel."""
        try:
            channel_data = self._notification_channels[user_id]
            pubsub = channel_data["pubsub"]
            callback = channel_data["callback"]
            
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = json.loads(message["data"])
                    await callback(data)
                    
        except Exception as e:
            self.logger.error(f"Error listening for messages: {str(e)}")
            raise 