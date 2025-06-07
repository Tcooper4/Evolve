import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import asyncio
from datetime import datetime
from dataclasses import dataclass
import aiohttp
import websockets
import ssl
import certifi

@dataclass
class Notification:
    id: str
    type: str
    title: str
    message: str
    level: str
    timestamp: str
    source: str
    metadata: Dict
    read: bool = False

class NotificationSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.notifications: List[Notification] = []
        self.websocket_clients: List[websockets.WebSocketServerProtocol] = []
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.discord_config = config.get('discord', {})
        self.websocket_config = config.get('websocket', {})
        self.notification_levels = {
            'debug': 0,
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }

    def setup_logging(self):
        """Configure logging for the notification system."""
        log_path = Path("automation/logs/notifications")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "notifications.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def send_notification(
        self,
        type: str,
        title: str,
        message: str,
        level: str = 'info',
        source: str = 'system',
        metadata: Optional[Dict] = None
    ):
        """
        Send a notification through all configured channels.
        
        Args:
            type: Notification type (e.g., 'task', 'system', 'error')
            title: Notification title
            message: Notification message
            level: Notification level (debug, info, warning, error, critical)
            source: Source of the notification
            metadata: Additional metadata
        """
        # Create notification object
        notification = Notification(
            id=str(len(self.notifications) + 1),
            type=type,
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now().isoformat(),
            source=source,
            metadata=metadata or {}
        )
        
        # Add to notifications list
        self.notifications.append(notification)
        
        # Send through all channels
        await asyncio.gather(
            self._send_email(notification),
            self._send_slack(notification),
            self._send_discord(notification),
            self._send_websocket(notification)
        )
        
        # Log notification
        self.logger.info(f"Sent notification: {notification.id} - {title}")
        
        return notification.id

    async def _send_email(self, notification: Notification):
        """Send notification via email."""
        if not self.email_config.get('enabled'):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"[{notification.level.upper()}] {notification.title}"
            
            body = f"""
            Type: {notification.type}
            Source: {notification.source}
            Time: {notification.timestamp}
            
            {notification.message}
            
            Metadata:
            {json.dumps(notification.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls'):
                    server.starttls()
                if self.email_config.get('username'):
                    server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Sent email notification: {notification.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")

    async def _send_slack(self, notification: Notification):
        """Send notification via Slack."""
        if not self.slack_config.get('enabled'):
            return
        
        try:
            # Prepare message
            message = {
                'text': f"*[{notification.level.upper()}] {notification.title}*\n{notification.message}",
                'attachments': [{
                    'color': self._get_slack_color(notification.level),
                    'fields': [
                        {
                            'title': 'Type',
                            'value': notification.type,
                            'short': True
                        },
                        {
                            'title': 'Source',
                            'value': notification.source,
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': notification.timestamp,
                            'short': True
                        }
                    ]
                }]
            }
            
            # Add metadata if present
            if notification.metadata:
                message['attachments'][0]['fields'].append({
                    'title': 'Metadata',
                    'value': json.dumps(notification.metadata, indent=2),
                    'short': False
                })
            
            # Send message
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config['webhook_url'],
                    json=message
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API returned status {response.status}")
            
            self.logger.info(f"Sent Slack notification: {notification.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")

    async def _send_discord(self, notification: Notification):
        """Send notification via Discord."""
        if not self.discord_config.get('enabled'):
            return
        
        try:
            # Prepare message
            message = {
                'embeds': [{
                    'title': f"[{notification.level.upper()}] {notification.title}",
                    'description': notification.message,
                    'color': self._get_discord_color(notification.level),
                    'fields': [
                        {
                            'name': 'Type',
                            'value': notification.type,
                            'inline': True
                        },
                        {
                            'name': 'Source',
                            'value': notification.source,
                            'inline': True
                        },
                        {
                            'name': 'Time',
                            'value': notification.timestamp,
                            'inline': True
                        }
                    ]
                }]
            }
            
            # Add metadata if present
            if notification.metadata:
                message['embeds'][0]['fields'].append({
                    'name': 'Metadata',
                    'value': f"```json\n{json.dumps(notification.metadata, indent=2)}\n```",
                    'inline': False
                })
            
            # Send message
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_config['webhook_url'],
                    json=message
                ) as response:
                    if response.status != 204:
                        raise Exception(f"Discord API returned status {response.status}")
            
            self.logger.info(f"Sent Discord notification: {notification.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord notification: {str(e)}")

    async def _send_websocket(self, notification: Notification):
        """Send notification via WebSocket."""
        if not self.websocket_config.get('enabled'):
            return
        
        # Prepare message
        message = {
            'id': notification.id,
            'type': notification.type,
            'title': notification.title,
            'message': notification.message,
            'level': notification.level,
            'timestamp': notification.timestamp,
            'source': notification.source,
            'metadata': notification.metadata
        }
        
        # Send to all connected clients
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send WebSocket notification: {str(e)}")
                self.websocket_clients.remove(client)

    def _get_slack_color(self, level: str) -> str:
        """Get Slack color for notification level."""
        colors = {
            'debug': '#808080',  # Gray
            'info': '#36a64f',   # Green
            'warning': '#ffcc00', # Yellow
            'error': '#ff0000',  # Red
            'critical': '#800000' # Dark Red
        }
        return colors.get(level, colors['info'])

    def _get_discord_color(self, level: str) -> int:
        """Get Discord color for notification level."""
        colors = {
            'debug': 0x808080,  # Gray
            'info': 0x36a64f,   # Green
            'warning': 0xffcc00, # Yellow
            'error': 0xff0000,  # Red
            'critical': 0x800000 # Dark Red
        }
        return colors.get(level, colors['info'])

    async def start_websocket_server(self):
        """Start WebSocket server for real-time notifications."""
        if not self.websocket_config.get('enabled'):
            return
        
        async def handle_client(websocket: websockets.WebSocketServerProtocol, path: str):
            """Handle WebSocket client connection."""
            self.websocket_clients.append(websocket)
            try:
                async for message in websocket:
                    # Handle client messages if needed
                    pass
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.remove(websocket)
        
        # Start server
        ssl_context = None
        if self.websocket_config.get('use_ssl'):
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.load_cert_chain(
                self.websocket_config['ssl_cert'],
                self.websocket_config['ssl_key']
            )
        
        server = await websockets.serve(
            handle_client,
            self.websocket_config['host'],
            self.websocket_config['port'],
            ssl=ssl_context
        )
        
        self.logger.info(f"Started WebSocket server on {self.websocket_config['host']}:{self.websocket_config['port']}")
        await server.wait_closed()

    def get_notifications(
        self,
        level: Optional[str] = None,
        type: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Notification]:
        """Get notifications with optional filtering."""
        notifications = self.notifications
        
        if level:
            notifications = [n for n in notifications if n.level == level]
        if type:
            notifications = [n for n in notifications if n.type == type]
        if source:
            notifications = [n for n in notifications if n.source == source]
        
        # Sort by timestamp (newest first)
        notifications.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            notifications = notifications[:limit]
        
        return notifications

    def mark_as_read(self, notification_id: str):
        """Mark a notification as read."""
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.read = True
                break

    def clear_notifications(self, before: Optional[datetime] = None):
        """Clear old notifications."""
        if before:
            self.notifications = [
                n for n in self.notifications
                if datetime.fromisoformat(n.timestamp) > before
            ]
        else:
            self.notifications = []
        
        self.logger.info(f"Cleared notifications before {before}") 