import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.notification_cleanup import NotificationCleanupService

@pytest.fixture
async def notification_system():
    # Create services
    notification_service = NotificationService()
    cleanup_service = NotificationCleanupService(
        notification_service,
        retention_days=1,
        cleanup_interval=1
    )
    
    # Start services
    await notification_service.start()
    await cleanup_service.start()
    
    yield notification_service, cleanup_service
    
    # Stop services
    await cleanup_service.stop()
    await notification_service.stop()

@pytest.mark.asyncio
async def test_notification_lifecycle(notification_system):
    notification_service, cleanup_service = notification_system
    
    # Create mock handlers
    email_handler = Mock()
    slack_handler = Mock()
    
    # Register handlers
    notification_service.register_handler(NotificationChannel.EMAIL, email_handler)
    notification_service.register_handler(NotificationChannel.SLACK, slack_handler)
    
    # Send notifications
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
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify notifications were sent
    assert email_handler.call_count == 1
    assert slack_handler.call_count == 1
    
    # Verify notification states
    email_notification = await notification_service.get_notification(email_notification.id)
    slack_notification = await notification_service.get_notification(slack_notification.id)
    
    assert email_notification.status == "sent"
    assert slack_notification.status == "sent"
    
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
async def test_notification_cleanup_integration(notification_system):
    notification_service, cleanup_service = notification_system
    
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
    
    # Wait for cleanup
    await asyncio.sleep(2)
    
    # Verify cleanup
    old_notification = await notification_service.get_notification(old_notification.id)
    new_notification = await notification_service.get_notification(new_notification.id)
    
    assert old_notification is None
    assert new_notification is not None

@pytest.mark.asyncio
async def test_notification_retry_integration(notification_system):
    notification_service, _ = notification_system
    
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
async def test_notification_filtering_integration(notification_system):
    notification_service, _ = notification_system
    
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