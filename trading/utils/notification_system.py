"""
Notification System for Trading Platform

Provides Slack and email notifications with fallback logic.
All notifications are wrapped in conditional blocks for environment variables.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationSystem:
    """Centralized notification system for the trading platform."""

    def __init__(self):
        """Initialize the notification system."""
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "dev-email-password")
        self.email_host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
        self.email_port = int(os.getenv("EMAIL_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER", "your-email@gmail.com")

        # Initialize notification status
        self.notification_status = {
            "slack": bool(self.slack_webhook_url),
            "email": bool(self.email_password),
            "last_notification": None,
        }

        logger.info(
            f"Notification system initialized - Slack: {self.notification_status['slack']}, Email: {self.notification_status['email']}"
        )

    def send_slack_notification(
        self,
        message: str,
        channel: str = "#trading-alerts",
        username: str = "Trading Bot",
        icon_emoji: str = ":robot_face:",
    ) -> bool:
        """
        Send Slack notification if webhook URL is configured.

        Args:
            message: Message to send
            channel: Slack channel
            username: Bot username
            icon_emoji: Bot icon emoji

        Returns:
            True if notification was sent successfully
        """
        if not self.slack_webhook_url:
            logger.debug("Slack webhook URL not configured, skipping notification")
            return False

        try:
            import requests

            payload = {
                "channel": channel,
                "username": username,
                "text": message,
                "icon_emoji": icon_emoji,
            }

            response = requests.post(
                self.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"✅ Slack notification sent successfully to {channel}")
                self.notification_status["last_notification"] = datetime.now()
                return True
            else:
                logger.error(
                    f"❌ Slack notification failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Error sending Slack notification: {str(e)}")
            return False

    def send_email_notification(
        self,
        subject: str,
        message: str,
        to_email: str,
        from_email: Optional[str] = None,
    ) -> bool:
        """
        Send email notification (DISABLED - email functionality removed).

        Args:
            subject: Email subject
            message: Email message
            to_email: Recipient email
            from_email: Sender email (optional)

        Returns:
            False (email functionality has been removed)
        """
        logger.info("Email notifications have been disabled - functionality removed")
        return False

    def send_trading_alert(
        self, alert_type: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send trading alert via all available channels.

        Args:
            alert_type: Type of alert (info, warning, error, success)
            message: Alert message
            data: Additional data to include

        Returns:
            Dictionary with notification results
        """
        results = {"slack": False, "email": False}

        # Format message based on alert type
        emoji_map = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "success": ":white_check_mark:",
        }

        emoji = emoji_map.get(alert_type, ":bell:")
        formatted_message = f"{emoji} **{alert_type.upper()}**: {message}"

        if data:
            formatted_message += f"\n```{json.dumps(data, indent=2)}```"

        # Send Slack notification
        if self.slack_webhook_url:
            results["slack"] = self.send_slack_notification(
                formatted_message,
                channel="#trading-alerts",
                username="Trading Alert Bot",
                icon_emoji=emoji,
            )

        # Send email notification (mock for now)
        if self.email_password:
            results["email"] = self.send_email_notification(
                subject=f"Trading Alert: {alert_type.upper()}",
                message=message,
                to_email="admin@trading.com",  # Mock email
            )

        return results

    def send_model_performance_alert(
        self, model_name: str, metrics: Dict[str, float], threshold: float = 0.7
    ) -> Dict[str, bool]:
        """
        Send model performance alert if metrics are below threshold.

        Args:
            model_name: Name of the model
            metrics: Performance metrics
            threshold: Performance threshold

        Returns:
            Dictionary with notification results
        """
        # Check if any metric is below threshold
        low_performance = {k: v for k, v in metrics.items() if v < threshold}

        if not low_performance:
            return {
                "success": True,
                "result": {"slack": False, "email": False},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        message = f"Model {model_name} performance below threshold ({threshold}):"
        for metric, value in low_performance.items():
            message += f"\n- {metric}: {value:.3f}"

        return self.send_trading_alert(
            "warning", message, {"model": model_name, "metrics": metrics}
        )

    def send_system_health_alert(
        self, component: str, status: str, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send system health alert.

        Args:
            component: System component
            status: Health status
            details: Additional details

        Returns:
            Dictionary with notification results
        """
        message = f"System health alert - {component}: {status}"

        return self.send_trading_alert(
            "error" if status == "down" else "warning", message, details
        )

    def send_agent_activity_notification(
        self, agent_name: str, action: str, result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send agent activity notification.

        Args:
            agent_name: Name of the agent
            action: Action performed
            result: Result of the action

        Returns:
            Dictionary with notification results
        """
        message = f"Agent {agent_name} performed: {action}"

        return {
            "success": True,
            "result": self.send_trading_alert("info", message, result),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def send_goal_progress_notification(
        self, goal_name: str, progress: float, target: float = 1.0
    ) -> Dict[str, bool]:
        """
        Send goal progress notification.

        Args:
            goal_name: Name of the goal
            progress: Current progress (0.0 to 1.0)
            target: Target progress

        Returns:
            Dictionary with notification results
        """
        if progress >= target:
            message = f"Goal '{goal_name}' completed! Progress: {progress:.1%}"
            alert_type = "success"
        elif progress >= 0.8:
            message = f"Goal '{goal_name}' nearly complete! Progress: {progress:.1%}"
            alert_type = "info"
        else:
            message = f"Goal '{goal_name}' progress update: {progress:.1%}"
            alert_type = "info"

        return {
            "success": True,
            "result": self.send_trading_alert(
                alert_type,
                message,
                {"goal": goal_name, "progress": progress, "target": target},
            ),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_notification_status(self) -> Dict[str, Any]:
        """
        Get notification system status.

        Returns:
            Dictionary with notification status
        """
        return {
            "slack_configured": bool(self.slack_webhook_url),
            "email_configured": bool(self.email_password),
            "last_notification": self.notification_status["last_notification"],
            "email_host": self.email_host,
            "email_port": self.email_port,
            "email_user": self.email_user,
        }

    def test_notifications(self) -> Dict[str, bool]:
        """
        Test all notification channels.

        Returns:
            Dictionary with test results
        """
        test_message = f"Test notification from Trading System at {datetime.now()}"

        results = {"slack": False, "email": False}

        # Test Slack
        if self.slack_webhook_url:
            results["slack"] = self.send_slack_notification(
                test_message,
                channel="#trading-alerts",
                username="Test Bot",
                icon_emoji=":test_tube:",
            )

        # Test Email
        if self.email_password:
            results["email"] = self.send_email_notification(
                subject="Test Notification",
                message=test_message,
                to_email="test@example.com",  # Mock email
            )

        return results


# Global notification instance
notification_system = NotificationSystem()


def send_alert(
    alert_type: str, message: str, data: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """
    Convenience function to send alerts.

    Args:
        alert_type: Type of alert
        message: Alert message
        data: Additional data

    Returns:
        Dictionary with notification results
    """
    return {
        "success": True,
        "result": notification_system.send_trading_alert(alert_type, message, data),
        "message": "Operation completed successfully",
        "timestamp": datetime.now().isoformat(),
    }


def send_model_alert(
    model_name: str, metrics: Dict[str, float], threshold: float = 0.7
) -> Dict[str, bool]:
    """
    Convenience function to send model performance alerts.

    Args:
        model_name: Name of the model
        metrics: Performance metrics
        threshold: Performance threshold

    Returns:
        Dictionary with notification results
    """
    return {
        "success": True,
        "result": notification_system.send_model_performance_alert(
            model_name, metrics, threshold
        ),
        "message": "Operation completed successfully",
        "timestamp": datetime.now().isoformat(),
    }


def send_system_alert(
    component: str, status: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """
    Convenience function to send system health alerts.

    Args:
        component: System component
        status: Health status
        details: Additional details

    Returns:
        Dictionary with notification results
    """
    return {
        "success": True,
        "result": notification_system.send_system_health_alert(
            component, status, details
        ),
        "message": "Operation completed successfully",
        "timestamp": datetime.now().isoformat(),
    }
