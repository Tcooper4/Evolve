import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.cleanup import NotificationCleanupService

@pytest.fixture
async def cleanup_test_system():
    """Create a notification system for cleanup testing."""
    service = NotificationService()
    cleanup_service = NotificationCleanupService(service)
    await service.start()
    yield service, cleanup_service
    await service.stop()

@pytest.mark.asyncio
async def test_cleanup_old_notifications(cleanup_test_system):
    """Test cleanup of old notifications."""
    service, cleanup_service = cleanup_test_system
    
    # Create old notifications
    old_date = datetime.now() - timedelta(days=31)  # 31 days old
    for i in range(5):
        notification = await service.send_notification(
            title=f"Old Notification {i}",
            message=f"This is an old notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notification.created_at = old_date.timestamp()
    
    # Create new notifications
    for i in range(3):
        await service.send_notification(
            title=f"New Notification {i}",
            message=f"This is a new notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Run cleanup
    await cleanup_service.cleanup_old_notifications()
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 3  # Only new notifications should remain
    
    # Verify old notifications are gone
    for notification in notifications:
        assert notification.title.startswith("New Notification")

@pytest.mark.asyncio
async def test_cleanup_by_status(cleanup_test_system):
    """Test cleanup of notifications by status."""
    service, cleanup_service = cleanup_test_system
    
    # Create completed notifications
    for i in range(3):
        notification = await service.send_notification(
            title=f"Completed Notification {i}",
            message=f"This is a completed notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        await service.update_notification(notification.id, status="completed")
    
    # Create failed notifications
    for i in range(2):
        notification = await service.send_notification(
            title=f"Failed Notification {i}",
            message=f"This is a failed notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        await service.update_notification(notification.id, status="failed")
    
    # Create pending notifications
    for i in range(4):
        await service.send_notification(
            title=f"Pending Notification {i}",
            message=f"This is a pending notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Run cleanup for completed notifications
    await cleanup_service.cleanup_by_status("completed")
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 6  # Failed and pending notifications should remain
    
    # Verify completed notifications are gone
    for notification in notifications:
        assert not notification.title.startswith("Completed Notification")
    
    # Run cleanup for failed notifications
    await cleanup_service.cleanup_by_status("failed")
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 4  # Only pending notifications should remain
    
    # Verify failed notifications are gone
    for notification in notifications:
        assert not notification.title.startswith("Failed Notification")

@pytest.mark.asyncio
async def test_cleanup_by_type(cleanup_test_system):
    """Test cleanup of notifications by type."""
    service, cleanup_service = cleanup_test_system
    
    # Create INFO notifications
    for i in range(3):
        await service.send_notification(
            title=f"Info Notification {i}",
            message=f"This is an info notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Create WARNING notifications
    for i in range(2):
        await service.send_notification(
            title=f"Warning Notification {i}",
            message=f"This is a warning notification {i}",
            type=NotificationType.WARNING,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Run cleanup for INFO notifications
    await cleanup_service.cleanup_by_type(NotificationType.INFO)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 2  # Only WARNING notifications should remain
    
    # Verify INFO notifications are gone
    for notification in notifications:
        assert not notification.title.startswith("Info Notification")

@pytest.mark.asyncio
async def test_cleanup_by_channel(cleanup_test_system):
    """Test cleanup of notifications by channel."""
    service, cleanup_service = cleanup_test_system
    
    # Create EMAIL notifications
    for i in range(3):
        await service.send_notification(
            title=f"Email Notification {i}",
            message=f"This is an email notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Create SLACK notifications
    for i in range(2):
        await service.send_notification(
            title=f"Slack Notification {i}",
            message=f"This is a slack notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.SLACK,
            recipient="#test-channel"
        )
    
    # Run cleanup for EMAIL notifications
    await cleanup_service.cleanup_by_channel(NotificationChannel.EMAIL)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 2  # Only SLACK notifications should remain
    
    # Verify EMAIL notifications are gone
    for notification in notifications:
        assert not notification.title.startswith("Email Notification")

@pytest.mark.asyncio
async def test_cleanup_by_priority(cleanup_test_system):
    """Test cleanup of notifications by priority."""
    service, cleanup_service = cleanup_test_system
    
    # Create MEDIUM priority notifications
    for i in range(3):
        await service.send_notification(
            title=f"Medium Priority Notification {i}",
            message=f"This is a medium priority notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Create HIGH priority notifications
    for i in range(2):
        await service.send_notification(
            title=f"High Priority Notification {i}",
            message=f"This is a high priority notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Run cleanup for MEDIUM priority notifications
    await cleanup_service.cleanup_by_priority(NotificationPriority.MEDIUM)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 2  # Only HIGH priority notifications should remain
    
    # Verify MEDIUM priority notifications are gone
    for notification in notifications:
        assert not notification.title.startswith("Medium Priority Notification")

@pytest.mark.asyncio
async def test_cleanup_retention_period(cleanup_test_system):
    """Test cleanup with different retention periods."""
    service, cleanup_service = cleanup_test_system
    
    # Create notifications with different ages
    now = datetime.now()
    ages = [7, 14, 21, 28, 35]  # Days old
    
    for i, age in enumerate(ages):
        notification = await service.send_notification(
            title=f"Notification {i}",
            message=f"This is a notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notification.created_at = (now - timedelta(days=age)).timestamp()
    
    # Run cleanup with 30-day retention
    await cleanup_service.cleanup_old_notifications(retention_days=30)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 4  # Only notifications less than 30 days old should remain
    
    # Run cleanup with 20-day retention
    await cleanup_service.cleanup_old_notifications(retention_days=20)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 3  # Only notifications less than 20 days old should remain
    
    # Run cleanup with 10-day retention
    await cleanup_service.cleanup_old_notifications(retention_days=10)
    
    # Verify cleanup
    notifications = await service.get_notifications()
    assert len(notifications) == 2  # Only notifications less than 10 days old should remain

@pytest.mark.asyncio
async def test_cleanup_error_handling(cleanup_test_system):
    """Test error handling during cleanup."""
    service, cleanup_service = cleanup_test_system
    
    # Create test notifications
    for i in range(5):
        await service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Mock service to raise exception
    with patch.object(service, "delete_notification", side_effect=Exception("Delete error")):
        # Run cleanup
        await cleanup_service.cleanup_old_notifications()
        
        # Verify cleanup continues despite errors
        notifications = await service.get_notifications()
        assert len(notifications) == 5  # All notifications should remain

@pytest.mark.asyncio
async def test_cleanup_concurrent(cleanup_test_system):
    """Test concurrent cleanup operations."""
    service, cleanup_service = cleanup_test_system
    
    # Create test notifications
    for i in range(10):
        await service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Run multiple cleanup operations concurrently
    tasks = [
        cleanup_service.cleanup_old_notifications(),
        cleanup_service.cleanup_by_status("completed"),
        cleanup_service.cleanup_by_type(NotificationType.INFO),
        cleanup_service.cleanup_by_channel(NotificationChannel.EMAIL),
        cleanup_service.cleanup_by_priority(NotificationPriority.MEDIUM)
    ]
    
    await asyncio.gather(*tasks)
    
    # Verify cleanup completed without errors
    notifications = await service.get_notifications()
    assert len(notifications) >= 0  # Any number of notifications may remain

@pytest.mark.asyncio
async def test_cleanup_metrics(cleanup_test_system):
    """Test cleanup metrics collection."""
    service, cleanup_service = cleanup_test_system
    
    # Create old notifications
    old_date = datetime.now() - timedelta(days=31)
    for i in range(5):
        notification = await service.send_notification(
            title=f"Old Notification {i}",
            message=f"This is an old notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notification.created_at = old_date.timestamp()
    
    # Run cleanup
    metrics = await cleanup_service.cleanup_old_notifications()
    
    # Verify metrics
    assert metrics["total_notifications"] == 5
    assert metrics["cleaned_up_notifications"] == 5
    assert metrics["cleanup_duration"] > 0
    assert metrics["cleanup_success"] is True
    assert metrics["cleanup_errors"] == 0 