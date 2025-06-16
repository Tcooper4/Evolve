"""
Notification Service

This module implements a service for managing and sending notifications through
various channels.

Note: This module was adapted from the legacy automation/notifications/notification_service.py file.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from .notification_handlers import NotificationHandler, SlackHandler, WebhookHandler

class NotificationService:
    """Service for managing and sending notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the notification service."""
        self.config = config
        self.handlers: Dict[str, NotificationHandler] = {}
        self.setup_logging()
        self.initialize_handlers()
    
    def setup_logging(self):
        """Configure logging for notifications."""
        log_path = Path("logs/notifications")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "notification_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_handlers(self):
        """Initialize notification handlers."""
        try:
            # Initialize Slack handler if configured
            if "slack_webhook_url" in self.config:
                self.handlers["slack"] = SlackHandler(self.config)
                self.logger.info("Initialized Slack notification handler")
            
            # Initialize webhook handler if configured
            if "webhook_url" in self.config:
                self.handlers["webhook"] = WebhookHandler(self.config)
                self.logger.info("Initialized webhook notification handler")
            
            if not self.handlers:
                self.logger.warning("No notification handlers configured")
                
        except Exception as e:
            self.logger.error(f"Error initializing notification handlers: {str(e)}")
            raise
    
    async def send_notification(
        self,
        message: Dict[str, Any],
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send a notification through specified channels."""
        try:
            if not channels:
                channels = list(self.handlers.keys())
            
            results = {}
            for channel in channels:
                if channel in self.handlers:
                    success = await self.handlers[channel].send(message)
                    results[channel] = success
                else:
                    self.logger.warning(f"Unknown notification channel: {channel}")
                    results[channel] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            raise
    
    async def send_alert(
        self,
        alert: Dict[str, Any],
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send an alert notification."""
        try:
            # Format alert message
            message = {
                "text": f"🚨 Alert: {alert.get('title', 'Unknown Alert')}",
                "attachments": [{
                    "color": self._get_severity_color(alert.get("severity", "info")),
                    "title": alert.get("title", ""),
                    "text": alert.get("message", ""),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.get("severity", "info").upper(),
                            "short": True
                        },
                        {
                            "title": "Source",
                            "value": alert.get("source", "Unknown"),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.get("timestamp", datetime.utcnow().isoformat()),
                            "short": True
                        }
                    ]
                }]
            }
            
            # Add additional fields if present
            if "details" in alert:
                message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": str(alert["details"]),
                    "short": False
                })
            
            return await self.send_notification(message, channels)
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            raise
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color code for alert severity."""
        colors = {
            "critical": "#ff0000",  # Red
            "error": "#ff4d4d",     # Light Red
            "warning": "#ffa500",   # Orange
            "info": "#36a64f",      # Green
            "debug": "#808080"      # Gray
        }
        return colors.get(severity.lower(), colors["info"])
    
    async def register_handler(
        self,
        name: str,
        handler: NotificationHandler
    ) -> None:
        """Register a new notification handler."""
        try:
            self.handlers[name] = handler
            self.logger.info(f"Registered notification handler: {name}")
        except Exception as e:
            self.logger.error(f"Error registering notification handler: {str(e)}")
            raise
    
    async def unregister_handler(self, name: str) -> None:
        """Unregister a notification handler."""
        try:
            if name in self.handlers:
                del self.handlers[name]
                self.logger.info(f"Unregistered notification handler: {name}")
            else:
                self.logger.warning(f"Unknown notification handler: {name}")
        except Exception as e:
            self.logger.error(f"Error unregistering notification handler: {str(e)}")
            raise 