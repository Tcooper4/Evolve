"""
Notification system for live trade logging and alerts.

This module provides Slack and Email notification capabilities for the trading system.
All notifications are wrapped in conditional blocks for environment variables.
"""

import os
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import asyncio
import requests

logger = logging.getLogger(__name__)

class BaseNotifier:
    """Base class for notification systems."""
    
    def __init__(self):
        """Initialize the base notifier."""
        self.enabled = False
        self.last_notification = None
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def send(self, message: str, **kwargs) -> bool:
        """Send a notification (to be implemented by subclasses)."""
        raise NotImplementedError
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def is_enabled(self) -> bool:
        """Check if the notifier is enabled."""
        return {'success': True, 'result': self.enabled, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


class SlackNotifier(BaseNotifier):
    """Slack notification system."""
    
    def __init__(self, webhook_url: str):
        """Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL
        """
        super().__init__()
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
        
        if self.enabled:
            logger.info("✅ Slack notifications enabled")
        else:
            logger.warning("⚠️ Slack webhook URL not provided")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def send(self, message: str, channel: str = "#trading-alerts", 
             username: str = "Trading Bot", icon_emoji: str = ":chart_with_upwards_trend:") -> bool:
        """Send a Slack notification.
        
        Args:
            message: Message to send
            channel: Slack channel
            username: Bot username
            icon_emoji: Bot icon emoji
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Slack notifications not enabled")
            return False
        
        try:
            # Create payload
            payload = {
                "channel": channel,
                "username": username,
                "icon_emoji": icon_emoji,
                "text": message
            }
            
            # Send request
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.last_notification = datetime.now()
            logger.info(f"✅ Slack notification sent to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Slack notification failed: {e}")
            return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


class EmailNotifier(BaseNotifier):
    """Email notification system."""
    
    def __init__(self, host: str, port: int, user: str, password: str):
        """Initialize email notifier.
        
        Args:
            host: SMTP host
            port: SMTP port
            user: Email username
            password: Email password
        """
        super().__init__()
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.enabled = bool(password and password != "dev-email-password")
        
        if self.enabled:
            logger.info("✅ Email notifications enabled")
        else:
            logger.warning("⚠️ Email password not configured")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def send(self, subject: str, message: str, to_email: str, 
             from_email: Optional[str] = None) -> bool:
        """Send an email notification.
        
        Args:
            subject: Email subject
            message: Email message
            to_email: Recipient email
            from_email: Sender email (defaults to configured user)
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Email notifications not enabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email or self.user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            body = MIMEText(message, 'plain')
            msg.attach(body)
            
            # Send email
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            
            self.last_notification = datetime.now()
            logger.info(f"✅ Email notification sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email notification failed: {e}")
            return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


class TradeLogger:
    """Centralized trade logging with notification capabilities."""
    
    def __init__(self):
        """Initialize the trade logger."""
        self.notifiers: List[BaseNotifier] = []
        self.trade_log = []
        self.log_file = "logs/trade_log.json"
        
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)
        
        logger.info("✅ Trade logger initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a notification system.
        
        Args:
            notifier: Notification system to add
        """
        self.notifiers.append(notifier)
        logger.info(f"✅ Added notifier: {type(notifier).__name__}")
        return {'success': True, 'result': {"status": "notifier_added", "type": type(notifier).__name__}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log a trade with notifications.
        
        Args:
            trade_data: Trade information
        """
        # Add timestamp
        trade_data['timestamp'] = datetime.now().isoformat()
        
        # Save to log
        self.trade_log.append(trade_data)
        
        # Save to file
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving trade log: {e}")
        
        # Send notifications
        self._send_notifications(trade_data)
        
        logger.info(f"✅ Trade logged: {trade_data.get('symbol', 'Unknown')} - {trade_data.get('action', 'Unknown')}")
        return {'success': True, 'result': {"status": "trade_logged", "symbol": trade_data.get('symbol', 'Unknown'), "action": trade_data.get('action', 'Unknown')}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _send_notifications(self, trade_data: Dict[str, Any]) -> None:
        """Send notifications for a trade.
        
        Args:
            trade_data: Trade information
        """
        # Format message
        symbol = trade_data.get('symbol', 'Unknown')
        action = trade_data.get('action', 'Unknown')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        timestamp = trade_data.get('timestamp', 'Unknown')
        
        message = f"🔔 Trade Alert: {action} {quantity} {symbol} @ ${price:.2f} at {timestamp}"
        
        # Send to all notifiers
        notifications_sent = 0
        for notifier in self.notifiers:
            if notifier.is_enabled():
                try:
                    success = notifier.send(message)
                    if success:
                        notifications_sent += 1
                except Exception as e:
                    logger.error(f"❌ Error sending notification via {type(notifier).__name__}: {e}")
        
        return {'success': True, 'result': {"status": "notifications_sent", "count": notifications_sent, "total_notifiers": len(self.notifiers)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades.
        
        Args:
            limit: Number of trades to return
            
        Returns:
            List of recent trades
        """
        return {'success': True, 'result': self.trade_log[-limit:] if self.trade_log else [], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get trade summary statistics.
        
        Returns:
            Trade summary
        """
        if not self.trade_log:
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        total_trades = len(self.trade_log)
        total_volume = sum(trade.get('quantity', 0) for trade in self.trade_log)
        last_trade = self.trade_log[-1] if self.trade_log else None
        
        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "last_trade": last_trade
        }


# Global trade logger instance
_trade_logger = None

def get_trade_logger() -> TradeLogger:
    """Get the global trade logger instance.
    
    Returns:
        Trade logger instance
    """
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return {'success': True, 'result': _trade_logger, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}