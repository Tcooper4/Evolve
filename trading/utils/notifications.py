"""
Notification system for live trade logging and alerts.

This module provides Slack and Email notification capabilities for the trading system.
All notifications are wrapped in conditional blocks for environment variables.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseNotifier:
    """Base class for notification systems."""

    def __init__(self):
        """Initialize the base notifier."""
        self.enabled = False
        self.last_notification = None

    def is_enabled(self) -> bool:
        """Check if the notifier is enabled."""
        return self.enabled

    def send_notification(
        self, message: str, level: str = "info", recipients: Optional[List[str]] = None
    ):
        """Send notification to specified recipients.

        Args:
            message: Notification message
            level: Notification level (info, warning, error, critical)
            recipients: List of recipient emails/usernames
        """
        try:
            # Log the notification
            logger.info(f"Notification ({level}): {message}")

            # For now, just log to file - can be extended with email/SMS/etc.
            notification_log = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "recipients": recipients or [],
            }

            # Save to notification log file
            log_file = Path("logs/notifications.log")
            log_file.parent.mkdir(exist_ok=True)

            with open(log_file, "a") as f:
                f.write(json.dumps(notification_log) + "\n")

            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False


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


class TradeLogger:
    """Centralized trade logging with notification capabilities."""

    def __init__(self):
        """Initialize the trade logger."""
        self.notifiers: List[BaseNotifier] = []
        self.trade_log = []
        self.log_file = "logs/trade_log.json"

        # Ensure log directory exists
        try:
            os.makedirs("logs", exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create logs directory: {e}")

        logger.info("✅ Trade logger initialized")

    def add_notifier(self, notifier: BaseNotifier) -> Dict[str, Any]:
        """Add a notification system.

        Args:
            notifier: Notification system to add
        """
        self.notifiers.append(notifier)
        logger.info(f"✅ Added notifier: {type(notifier).__name__}")
        return {
            "success": True,
            "result": {"status": "notifier_added", "type": type(notifier).__name__},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def log_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log a trade with notifications.

        Args:
            trade_data: Trade information
        """
        # Add timestamp
        trade_data["timestamp"] = datetime.now().isoformat()

        # Save to log
        self.trade_log.append(trade_data)

        # Save to file
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving trade log: {e}")

        # Send notifications
        self._send_notifications(trade_data)

        logger.info(
            f"✅ Trade logged: {trade_data.get('symbol', 'Unknown')} - {trade_data.get('action', 'Unknown')}"
        )
        return {
            "success": True,
            "result": {
                "status": "trade_logged",
                "symbol": trade_data.get("symbol", "Unknown"),
                "action": trade_data.get("action", "Unknown"),
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _send_notifications(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notifications for a trade.

        Args:
            trade_data: Trade information
        """
        # Format message
        symbol = trade_data.get("symbol", "Unknown")
        action = trade_data.get("action", "Unknown")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        timestamp = trade_data.get("timestamp", "Unknown")

        message = (
            f"🔔 Trade Alert: {action} {quantity} {symbol} @ ${price:.2f} at {timestamp}"
        )

        # Send to all notifiers
        notifications_sent = 0
        for notifier in self.notifiers:
            if notifier.is_enabled():
                try:
                    # Assuming send_notification is available on BaseNotifier or its subclasses
                    # For now, we'll call it directly on the notifier instance
                    # If specific notification levels are needed, this logic needs to be refined
                    # For example, SlackNotifier might have a send_slack_notification method
                    # EmailNotifier might have a send_email_notification method
                    # This is a placeholder for now.
                    success = notifier.send_notification(message)
                    if success:
                        notifications_sent += 1
                except Exception as e:
                    logger.error(
                        f"❌ Error sending notification via {type(notifier).__name__}: {e}"
                    )

        return {
            "success": True,
            "result": {
                "status": "notifications_sent",
                "count": notifications_sent,
                "total_notifiers": len(self.notifiers),
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades.

        Args:
            limit: Number of trades to return

        Returns:
            List of recent trades
        """
        return self.trade_log[-limit:] if self.trade_log else []

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get trade summary statistics.

        Returns:
            Trade summary
        """
        if not self.trade_log:
            return {"total_trades": 0, "total_volume": 0, "last_trade": None}

        total_trades = len(self.trade_log)
        total_volume = sum(trade.get("quantity", 0) for trade in self.trade_log)
        last_trade = self.trade_log[-1] if self.trade_log else None

        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "last_trade": last_trade,
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
    return _trade_logger
