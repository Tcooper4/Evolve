"""
Notification Service

Implements notification functionality for sending alerts and messages through various channels.
Adapted from legacy automation/services/automation_notification.py.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import jinja2
from pydantic import BaseModel
import sqlite3

class NotificationChannel(BaseModel):
    """Notification channel configuration."""
    type: str
    enabled: bool = True
    config: Dict[str, Any]

class NotificationTemplate(BaseModel):
    """Notification template configuration."""
    name: str
    subject: str
    body: str
    channel: str

class Notification(BaseModel):
    """Notification model."""
    id: Optional[str]
    channel: str
    template: str
    recipient: str
    data: Dict[str, Any]
    priority: str = "normal"
    created_at: datetime = datetime.now()

class NotificationService:
    """Manages notifications and alerts."""
    
    def __init__(self, config_path: str = "config/notification.json"):
        """Initialize notification service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.setup_templates()
        self.notification_queue = asyncio.Queue()
        self.running = False
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/notification")
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
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def setup_database(self) -> None:
        """Set up notification database."""
        try:
            db_path = Path(self.config['database']['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Create notifications table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    template TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    data TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    sent_at DATETIME
                )
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_templates(self) -> None:
        """Set up notification templates."""
        try:
            template_path = Path(self.config['templates']['path'])
            template_path.mkdir(parents=True, exist_ok=True)
            
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(template_path))
            )
            
            self.logger.info("Templates setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up templates: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def send_email(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            channel_config = self.config['channels']['email']
            
            msg = MIMEMultipart()
            msg['From'] = channel_config['from']
            msg['To'] = notification.recipient
            
            template = self.template_env.get_template(f"{notification.template}.html")
            body = template.render(**notification.data)
            
            msg['Subject'] = template.render(**notification.data)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(channel_config['smtp_host'], channel_config['smtp_port']) as server:
                server.starttls()
                server.login(channel_config['username'], channel_config['password'])
                server.send_message(msg)
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return False
    
    async def send_slack(self, notification: Notification) -> bool:
        """Send Slack notification."""
        try:
            channel_config = self.config['channels']['slack']
            
            template = self.template_env.get_template(f"{notification.template}.txt")
            message = template.render(**notification.data)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    channel_config['webhook_url'],
                    json={"text": message}
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Error sending Slack message: {str(e)}")
            return False
    
    async def send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            channel_config = self.config['channels']['webhook']
            
            template = self.template_env.get_template(f"{notification.template}.json")
            data = json.loads(template.render(**notification.data))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    channel_config['url'],
                    json=data,
                    headers=channel_config.get('headers', {})
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Error sending webhook: {str(e)}")
            return False
    
    async def send_notification(self, notification: Notification) -> bool:
        """Send notification through specified channel."""
        try:
            # Save to database
            self.cursor.execute('''
                INSERT INTO notifications (
                    id, channel, template, recipient, data,
                    priority, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.id,
                notification.channel,
                notification.template,
                notification.recipient,
                json.dumps(notification.data),
                notification.priority,
                "pending",
                notification.created_at
            ))
            
            # Send notification
            success = False
            if notification.channel == "email":
                success = await self.send_email(notification)
            elif notification.channel == "slack":
                success = await self.send_slack(notification)
            elif notification.channel == "webhook":
                success = await self.send_webhook(notification)
            
            # Update database
            self.cursor.execute('''
                UPDATE notifications
                SET status = ?, sent_at = ?
                WHERE id = ?
            ''', (
                "sent" if success else "failed",
                datetime.now() if success else None,
                notification.id
            ))
            
            self.conn.commit()
            return success
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return False
    
    async def process_notification_queue(self) -> None:
        """Process notification queue."""
        while self.running:
            try:
                notification = await self.notification_queue.get()
                await self.send_notification(notification)
                self.notification_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing notification: {str(e)}")
    
    async def start(self) -> None:
        """Start notification service."""
        try:
            self.running = True
            
            # Start queue processor
            processor = asyncio.create_task(self.process_notification_queue())
            
            # Keep service running
            while self.running:
                await asyncio.sleep(1)
            
            # Cleanup
            processor.cancel()
            try:
                await processor
            except asyncio.CancelledError:
                pass
        except Exception as e:
            self.logger.error(f"Error in notification service: {str(e)}")
            raise
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop notification service."""
        self.running = False

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Notification service')
    parser.add_argument('--config', default="config/notification.json", help='Path to config file')
    args = parser.parse_args()
    
    try:
        service = NotificationService(args.config)
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Notification service interrupted")
    except Exception as e:
        logging.error(f"Error in notification service: {str(e)}")
        raise

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == '__main__':
    main() 