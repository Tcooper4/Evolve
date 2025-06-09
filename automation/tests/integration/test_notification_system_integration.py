import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import os
import yaml

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.notification_cleanup import NotificationCleanupService
from automation.notifications.handlers.email_handler import EmailHandler
from automation.notifications.handlers.slack_handler import SlackHandler
from automation.notifications.handlers.webhook_handler import WebhookHandler

@pytest.fixture
def config():
    """Load the notification configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "notification_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
async def notification_system(config):
    """Create a complete notification system with all handlers."""
    # Create services
    notification_service = NotificationService()
    cleanup_service = NotificationCleanupService(
        notification_service,
        retention_days=config["settings"]["retention_days"],
        cleanup_interval=config["settings"]["cleanup_interval"]
    )
    
    # Create handlers
    email_handler = EmailHandler(
        smtp_host=config["email"]["smtp_host"],
        smtp_port=config["email"]["smtp_port"],
        username=os.getenv("EMAIL_USERNAME", "test@example.com"),
        password=os.getenv("EMAIL_PASSWORD", "test_password"),
        from_email=os.getenv("EMAIL_FROM", "test@example.com")
    )
    
    slack_handler = SlackHandler(
        webhook_url=os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    )
    
    webhook_handler = WebhookHandler(
        webhook_url="https://example.com/webhook",
        headers={"Authorization": "Bearer test_token"}
    )
    
    # Register handlers
    notification_service.register_handler(NotificationChannel.EMAIL, email_handler)
    notification_service.register_handler(NotificationChannel.SLACK, slack_handler)
    notification_service.register_handler(NotificationChannel.WEBHOOK, webhook_handler)
    
    # Start services
    await notification_service.start()
    await cleanup_service.start()
    
    yield {
        "notification_service": notification_service,
        "cleanup_service": cleanup_service,
        "email_handler": email_handler,
        "slack_handler": slack_handler,
        "webhook_handler": webhook_handler
    }
    
    # Stop services
    await cleanup_service.stop()
    await notification_service.stop()

@pytest.mark.asyncio
async def test_notification_lifecycle(notification_system):
    """Test the complete notification lifecycle."""
    system = notification_system
    notification_service = system["notification_service"]
    
    # Send notifications through different channels
    email_notification = await notification_service.send_notification(
        title="Email Test",
        message="This is an email test",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    slack_notification = await notification_service.send_notification(
        title="Slack Test",
        message="This is a slack test",
        type=NotificationType.ERROR,
        priority=NotificationPriority.HIGH,
        channel=NotificationChannel.SLACK,
        recipient="#test-channel"
    )
    
    webhook_notification = await notification_service.send_notification(
        title="Webhook Test",
        message="This is a webhook test",
        type=NotificationType.SUCCESS,
        priority=NotificationPriority.LOW,
        channel=NotificationChannel.WEBHOOK,
        recipient="https://example.com/webhook"
    )
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify notifications were sent
    email_notification = await notification_service.get_notification(email_notification.id)
    slack_notification = await notification_service.get_notification(slack_notification.id)
    webhook_notification = await notification_service.get_notification(webhook_notification.id)
    
    assert email_notification.status == "sent"
    assert slack_notification.status == "sent"
    assert webhook_notification.status == "sent"
    
    # Update notifications
    await notification_service.update_notification(
        email_notification.id,
        status="read"
    )
    
    # Verify update
    email_notification = await notification_service.get_notification(email_notification.id)
    assert email_notification.status == "read"
    
    # Delete notification
    await notification_service.delete_notification(slack_notification.id)
    deleted = await notification_service.get_notification(slack_notification.id)
    assert deleted is None

@pytest.mark.asyncio
async def test_notification_cleanup(notification_system):
    """Test the notification cleanup process."""
    system = notification_system
    notification_service = system["notification_service"]
    cleanup_service = system["cleanup_service"]
    
    # Create old notification
    old_date = datetime.utcnow() - timedelta(days=2)
    with patch('automation.notifications.notification_service.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = old_date
        old_notification = await notification_service.send_notification(
            title="Old Notification",
            message="This is an old notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.LOW,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Create new notification
    new_notification = await notification_service.send_notification(
        title="New Notification",
        message="This is a new notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.LOW,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Run cleanup
    await cleanup_service.cleanup_now()
    
    # Verify cleanup
    old_notification = await notification_service.get_notification(old_notification.id)
    new_notification = await notification_service.get_notification(new_notification.id)
    
    assert old_notification is None
    assert new_notification is not None

@pytest.mark.asyncio
async def test_notification_retry(notification_system):
    """Test the notification retry mechanism."""
    system = notification_system
    notification_service = system["notification_service"]
    
    # Create handler that fails twice then succeeds
    handler = Mock()
    handler.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        None
    ]
    
    # Register handler
    notification_service.register_handler(NotificationChannel.EMAIL, handler)
    
    # Send notification
    notification = await notification_service.send_notification(
        title="Retry Test",
        message="This is a retry test",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Wait for processing and retries
    await asyncio.sleep(0.5)
    
    # Verify retry behavior
    assert handler.call_count == 3
    notification = await notification_service.get_notification(notification.id)
    assert notification.retry_count == 2
    assert notification.status == "sent"

@pytest.mark.asyncio
async def test_notification_filtering(notification_system):
    """Test notification filtering functionality."""
    system = notification_system
    notification_service = system["notification_service"]
    
    # Create notifications with different types and channels
    await notification_service.send_notification(
        title="Info Email",
        message="Info email message",
        type=NotificationType.INFO,
        priority=NotificationPriority.LOW,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    await notification_service.send_notification(
        title="Error Slack",
        message="Error slack message",
        type=NotificationType.ERROR,
        priority=NotificationPriority.HIGH,
        channel=NotificationChannel.SLACK,
        recipient="#test-channel"
    )
    
    # Test filtering
    info_notifications = await notification_service.get_notifications(
        type=NotificationType.INFO
    )
    error_notifications = await notification_service.get_notifications(
        type=NotificationType.ERROR
    )
    email_notifications = await notification_service.get_notifications(
        channel=NotificationChannel.EMAIL
    )
    slack_notifications = await notification_service.get_notifications(
        channel=NotificationChannel.SLACK
    )
    
    # Verify filters
    assert len(info_notifications) == 1
    assert len(error_notifications) == 1
    assert len(email_notifications) == 1
    assert len(slack_notifications) == 1
    
    assert info_notifications[0].type == NotificationType.INFO
    assert error_notifications[0].type == NotificationType.ERROR
    assert email_notifications[0].channel == NotificationChannel.EMAIL
    assert slack_notifications[0].channel == NotificationChannel.SLACK 