import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from automation.notifications.notification_service import (
    NotificationService,
    Notification,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)

@pytest.fixture
async def notification_service():
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
def mock_handler():
    return Mock()

@pytest.mark.asyncio
async def test_send_notification(notification_service, mock_handler):
    # Register mock handler
    notification_service.register_handler(NotificationChannel.EMAIL, mock_handler)
    
    # Send notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify notification was created
    assert notification.id in notification_service._notifications
    assert notification.status == "pending"
    
    # Verify handler was called
    mock_handler.assert_called_once()
    called_notification = mock_handler.call_args[0][0]
    assert called_notification.id == notification.id

@pytest.mark.asyncio
async def test_get_notification(notification_service):
    # Create notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Get notification
    retrieved = await notification_service.get_notification(notification.id)
    
    # Verify
    assert retrieved is not None
    assert retrieved.id == notification.id
    assert retrieved.title == "Test Notification"

@pytest.mark.asyncio
async def test_get_notifications_with_filters(notification_service):
    # Create notifications with different types
    await notification_service.send_notification(
        title="Info Notification",
        message="Info message",
        type=NotificationType.INFO,
        priority=NotificationPriority.LOW,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    await notification_service.send_notification(
        title="Error Notification",
        message="Error message",
        type=NotificationType.ERROR,
        priority=NotificationPriority.HIGH,
        channel=NotificationChannel.SLACK,
        recipient="test@example.com"
    )
    
    # Get filtered notifications
    info_notifications = await notification_service.get_notifications(
        type=NotificationType.INFO
    )
    error_notifications = await notification_service.get_notifications(
        type=NotificationType.ERROR
    )
    
    # Verify
    assert len(info_notifications) == 1
    assert len(error_notifications) == 1
    assert info_notifications[0].type == NotificationType.INFO
    assert error_notifications[0].type == NotificationType.ERROR

@pytest.mark.asyncio
async def test_update_notification(notification_service):
    # Create notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Update notification
    updated = await notification_service.update_notification(
        notification.id,
        title="Updated Title",
        status="sent"
    )
    
    # Verify
    assert updated is not None
    assert updated.title == "Updated Title"
    assert updated.status == "sent"

@pytest.mark.asyncio
async def test_delete_notification(notification_service):
    # Create notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Delete notification
    success = await notification_service.delete_notification(notification.id)
    
    # Verify
    assert success is True
    assert notification.id not in notification_service._notifications

@pytest.mark.asyncio
async def test_notification_retry(notification_service, mock_handler):
    # Configure mock to fail twice then succeed
    mock_handler.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        None
    ]
    
    # Register mock handler
    notification_service.register_handler(NotificationChannel.EMAIL, mock_handler)
    
    # Send notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Wait for processing and retries
    await asyncio.sleep(0.5)
    
    # Verify
    assert mock_handler.call_count == 3
    assert notification.retry_count == 2
    assert notification.status == "sent"

@pytest.mark.asyncio
async def test_notification_max_retries(notification_service, mock_handler):
    # Configure mock to always fail
    mock_handler.side_effect = Exception("Persistent failure")
    
    # Register mock handler
    notification_service.register_handler(NotificationChannel.EMAIL, mock_handler)
    
    # Send notification
    notification = await notification_service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Wait for processing and retries
    await asyncio.sleep(0.5)
    
    # Verify
    assert mock_handler.call_count == notification.max_retries + 1
    assert notification.retry_count == notification.max_retries
    assert notification.status == "failed" 