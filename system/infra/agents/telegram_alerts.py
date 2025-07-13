"""
Telegram Alerts Module

This module provides Telegram notification functionality for the trading system.
It uses environment variables for secure API key management.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TelegramAlerts:
    """Telegram alerts manager for trading notifications."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Telegram alerts manager.

        Args:
            config: Configuration dictionary with telegram settings
        """
        self.config = config or {}
        self.bot_token = self._get_bot_token()
        self.chat_id = self._get_chat_id()
        self.enabled = self._is_enabled()

        if self.enabled and not self.bot_token:
            logger.warning("Telegram bot token not found. Telegram alerts will be disabled.")
            self.enabled = False

        if self.enabled and not self.chat_id:
            logger.warning("Telegram chat ID not found. Telegram alerts will be disabled.")
            self.enabled = False

    def _get_bot_token(self) -> Optional[str]:
        """Get bot token from environment variables."""
        # Try multiple environment variable names for flexibility
        token = os.getenv("TELEGRAM_BOT_KEY") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_API_KEY")

        if not token:
            logger.warning("No Telegram bot token found in environment variables")
            return None

        return token

    def _get_chat_id(self) -> Optional[str]:
        """Get chat ID from environment variables."""
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_USER_ID")

        if not chat_id:
            logger.warning("No Telegram chat ID found in environment variables")
            return None

        return chat_id

    def _is_enabled(self) -> bool:
        """Check if Telegram alerts are enabled."""
        enabled = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
        return enabled and bool(self.bot_token) and bool(self.chat_id)

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram.

        Args:
            message: Message to send
            parse_mode: Message parsing mode (HTML, Markdown, or plain text)

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Telegram alerts are disabled")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Telegram message sent successfully")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False

    def send_alert(self, subject: str, message: str, alert_type: str = "info", include_timestamp: bool = True) -> bool:
        """Send a formatted alert via Telegram.

        Args:
            subject: Alert subject/title
            message: Alert message content
            alert_type: Type of alert (info, warning, error, success)
            include_timestamp: Whether to include timestamp in message

        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Format message with HTML
        emoji_map = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}

        emoji = emoji_map.get(alert_type.lower(), "📢")

        formatted_message = f"{emoji} <b>{subject}</b>\n\n"

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message += f"<i>Time: {timestamp}</i>\n\n"

        formatted_message += message

        return self.send_message(formatted_message, parse_mode="HTML")

    def send_trading_alert(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Send a trading-specific alert.

        Args:
            symbol: Trading symbol
            action: Trading action (BUY, SELL, HOLD)
            price: Current price
            quantity: Trade quantity (optional)
            confidence: Model confidence (optional)

        Returns:
            True if alert sent successfully, False otherwise
        """
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action.upper(), "📊")

        message = f"{action_emoji} <b>{action.upper()}</b> {symbol}\n"
        message += f"Price: ${price:.2f}\n"

        if quantity:
            message += f"Quantity: {quantity}\n"

        if confidence:
            message += f"Confidence: {confidence:.1%}\n"

        return self.send_alert(f"Trading Signal: {symbol}", message, "info")

    def send_performance_alert(self, metric: str, value: float, threshold: float, status: str) -> bool:
        """Send a performance monitoring alert.

        Args:
            metric: Performance metric name
            value: Current value
            threshold: Threshold value
            status: Status (above/below threshold)

        Returns:
            True if alert sent successfully, False otherwise
        """
        status_emoji = "🟢" if status == "above" else "🔴"

        message = f"{status_emoji} <b>{metric}</b>\n"
        message += f"Current: {value:.2f}\n"
        message += f"Threshold: {threshold:.2f}\n"
        message += f"Status: {status.title()} threshold"

        alert_type = "warning" if status == "below" else "success"

        return self.send_alert(f"Performance Alert: {metric}", message, alert_type)

    def send_system_alert(self, component: str, message: str, alert_type: str = "error") -> bool:
        """Send a system-level alert.

        Args:
            component: System component name
            message: Alert message
            alert_type: Type of alert

        Returns:
            True if alert sent successfully, False otherwise
        """
        return self.send_alert(f"System Alert: {component}", message, alert_type)

    def test_connection(self) -> bool:
        """Test Telegram bot connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Telegram alerts are disabled")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            bot_info = response.json()
            if bot_info.get("ok"):
                logger.info(f"Telegram bot connection successful: {bot_info['result']['username']}")
                return True
            else:
                logger.error(f"Telegram bot connection failed: {bot_info}")
                return False

        except Exception as e:
            logger.error(f"Failed to test Telegram connection: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get Telegram alerts status.

        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "bot_token_configured": bool(self.bot_token),
            "chat_id_configured": bool(self.chat_id),
            "connection_test": self.test_connection() if self.enabled else False,
        }


# Example usage
if __name__ == "__main__":
    # Test the telegram alerts
    telegram = TelegramAlerts()

    # Test connection
    if telegram.test_connection():
        # Send test message
        telegram.send_alert("Test Alert", "This is a test message from the trading system.", "info")

        # Send trading alert
        telegram.send_trading_alert("AAPL", "BUY", 150.25, quantity=100, confidence=0.85)
    else:
        print("Telegram connection test failed")
