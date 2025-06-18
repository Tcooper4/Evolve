"""
Notification Manager

This module implements notification management functionality.

Note: This module was adapted from the legacy automation/notifications/notification_manager.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from trading.notification_service import NotificationService
from trading.notification_handlers import NotificationHandlerFactory

class NotificationManager:
    """Manages notifications and their delivery."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager."""
        self.config = config
        self.notification_service = NotificationService(config)
        self.handlers = {}
        self.setup_logging()
        self.initialize_handlers()
    
    def setup_logging(self):
        """Configure logging for notification manager."""
        log_path = Path("logs/notifications")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "notification_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_handlers(self):
        """Initialize notification handlers."""
        try:
            handler_configs = self.config.get('notification_handlers', {})
            
            for handler_type, handler_config in handler_configs.items():
                handler = NotificationHandlerFactory.create_handler(
                    handler_type,
                    handler_config
                )
                self.handlers[handler_type] = handler
            
            self.logger.info("Initialized notification handlers")
        except Exception as e:
            self.logger.error(f"Error initializing handlers: {str(e)}")
            raise
    
    async def send_notification(
        self,
        message: Dict[str, Any],
        channels: Optional[List[str]] = None
    ) -> None:
        """Send notification through specified channels."""
        try:
            if not channels:
                channels = list(self.handlers.keys())
            
            for channel in channels:
                if channel not in self.handlers:
                    self.logger.warning(f"Unsupported channel: {channel}")
                    continue
                
                handler = self.handlers[channel]
                success = await handler.send(message)
                
                if success:
                    self.logger.info(f"Sent notification via {channel}")
                else:
                    self.logger.error(f"Failed to send notification via {channel}")
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            raise
    
    async def broadcast_notification(
        self,
        message: Dict[str, Any]
    ) -> None:
        """Broadcast notification to all channels."""
        try:
            await self.send_notification(message)
            self.logger.info("Broadcast notification to all channels")
        except Exception as e:
            self.logger.error(f"Error broadcasting notification: {str(e)}")
            raise
    
    def get_available_channels(self) -> List[str]:
        """Get list of available notification channels."""
        return list(self.handlers.keys())
    
    def add_handler(
        self,
        handler_type: str,
        handler_config: Dict[str, Any]
    ) -> None:
        """Add a new notification handler."""
        try:
            handler = NotificationHandlerFactory.create_handler(
                handler_type,
                handler_config
            )
            self.handlers[handler_type] = handler
            self.logger.info(f"Added notification handler: {handler_type}")
        except Exception as e:
            self.logger.error(f"Error adding handler: {str(e)}")
            raise
    
    def remove_handler(self, handler_type: str) -> None:
        """Remove a notification handler."""
        try:
            if handler_type in self.handlers:
                del self.handlers[handler_type]
                self.logger.info(f"Removed notification handler: {handler_type}")
            else:
                self.logger.warning(f"Handler not found: {handler_type}")
        except Exception as e:
            self.logger.error(f"Error removing handler: {str(e)}")
            raise
    
    def get_handler_status(self, handler_type: str) -> Dict[str, Any]:
        """Get status of a notification handler."""
        try:
            if handler_type not in self.handlers:
                raise ValueError(f"Handler not found: {handler_type}")
            
            # TODO: Implement handler status check
            return {
                'status': 'active',
                'last_used': None,
                'success_rate': 1.0
            }
        except Exception as e:
            self.logger.error(f"Error getting handler status: {str(e)}")
            raise 