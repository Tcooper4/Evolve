import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import time

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.retry import NotificationRetryService

@pytest.fixture
async def retry_test_system():
    """Create a notification system for retry testing."""
    service = NotificationService()
    retry_service = NotificationRetryService(
        max_retries=3,
        retry_delay=1,
        backoff_factor=2
    )
    await service.start()
    yield service, retry_service
    await service.stop()

@pytest.mark.asyncio
async def test_retry_failed_notification(retry_test_system):
    """Test retrying a failed notification."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark notification as failed
    await service.update_notification(notification.id, status="failed")
    
    # Retry notification
    await retry_service.retry_notification(notification.id)
    
    # Verify retry
    updated_notification = await service.get_notification(notification.id)
    assert updated_notification.retry_count == 1
    assert updated_notification.status == "pending"
    assert updated_notification.last_retry_at is not None

@pytest.mark.asyncio
async def test_retry_max_retries(retry_test_system):
    """Test retry limit."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark notification as failed and retry multiple times
    for i in range(4):  # Try to retry 4 times (should fail on 4th)
        await service.update_notification(notification.id, status="failed")
        await retry_service.retry_notification(notification.id)
    
    # Verify retry limit
    updated_notification = await service.get_notification(notification.id)
    assert updated_notification.retry_count == 3  # Should stop at max retries
    assert updated_notification.status == "failed"
    assert updated_notification.last_retry_at is not None

@pytest.mark.asyncio
async def test_retry_backoff(retry_test_system):
    """Test retry backoff delay."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark notification as failed and retry
    await service.update_notification(notification.id, status="failed")
    
    # Measure retry delay
    start_time = time.time()
    await retry_service.retry_notification(notification.id)
    first_retry_delay = time.time() - start_time
    
    # Mark as failed again and retry
    await service.update_notification(notification.id, status="failed")
    
    start_time = time.time()
    await retry_service.retry_notification(notification.id)
    second_retry_delay = time.time() - start_time
    
    # Verify backoff
    assert second_retry_delay > first_retry_delay
    assert second_retry_delay >= first_retry_delay * retry_service.backoff_factor

@pytest.mark.asyncio
async def test_retry_successful_notification(retry_test_system):
    """Test retrying a successful notification."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark notification as successful
    await service.update_notification(notification.id, status="completed")
    
    # Try to retry notification
    with pytest.raises(ValueError):
        await retry_service.retry_notification(notification.id)
    
    # Verify notification unchanged
    updated_notification = await service.get_notification(notification.id)
    assert updated_notification.retry_count == 0
    assert updated_notification.status == "completed"
    assert updated_notification.last_retry_at is None

@pytest.mark.asyncio
async def test_retry_pending_notification(retry_test_system):
    """Test retrying a pending notification."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Try to retry notification
    with pytest.raises(ValueError):
        await retry_service.retry_notification(notification.id)
    
    # Verify notification unchanged
    updated_notification = await service.get_notification(notification.id)
    assert updated_notification.retry_count == 0
    assert updated_notification.status == "pending"
    assert updated_notification.last_retry_at is None

@pytest.mark.asyncio
async def test_retry_priority(retry_test_system):
    """Test retry with different priorities."""
    service, retry_service = retry_test_system
    
    # Create high priority notification
    high_priority = await service.send_notification(
        title="High Priority Notification",
        message="This is a high priority notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.HIGH,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Create medium priority notification
    medium_priority = await service.send_notification(
        title="Medium Priority Notification",
        message="This is a medium priority notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark both as failed
    await service.update_notification(high_priority.id, status="failed")
    await service.update_notification(medium_priority.id, status="failed")
    
    # Retry both
    await retry_service.retry_notification(high_priority.id)
    await retry_service.retry_notification(medium_priority.id)
    
    # Verify retries
    high_updated = await service.get_notification(high_priority.id)
    medium_updated = await service.get_notification(medium_priority.id)
    
    assert high_updated.retry_count == 1
    assert medium_updated.retry_count == 1
    assert high_updated.status == "pending"
    assert medium_updated.status == "pending"

@pytest.mark.asyncio
async def test_retry_channel(retry_test_system):
    """Test retry for different channels."""
    service, retry_service = retry_test_system
    
    # Create email notification
    email_notification = await service.send_notification(
        title="Email Notification",
        message="This is an email notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Create Slack notification
    slack_notification = await service.send_notification(
        title="Slack Notification",
        message="This is a slack notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.SLACK,
        recipient="#test-channel"
    )
    
    # Mark both as failed
    await service.update_notification(email_notification.id, status="failed")
    await service.update_notification(slack_notification.id, status="failed")
    
    # Retry both
    await retry_service.retry_notification(email_notification.id)
    await retry_service.retry_notification(slack_notification.id)
    
    # Verify retries
    email_updated = await service.get_notification(email_notification.id)
    slack_updated = await service.get_notification(slack_notification.id)
    
    assert email_updated.retry_count == 1
    assert slack_updated.retry_count == 1
    assert email_updated.status == "pending"
    assert slack_updated.status == "pending"

@pytest.mark.asyncio
async def test_retry_concurrent(retry_test_system):
    """Test concurrent retry operations."""
    service, retry_service = retry_test_system
    
    # Create multiple notifications
    notifications = []
    for i in range(5):
        notification = await service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notifications.append(notification)
    
    # Mark all as failed
    for notification in notifications:
        await service.update_notification(notification.id, status="failed")
    
    # Retry all concurrently
    tasks = [
        retry_service.retry_notification(notification.id)
        for notification in notifications
    ]
    await asyncio.gather(*tasks)
    
    # Verify retries
    for notification in notifications:
        updated = await service.get_notification(notification.id)
        assert updated.retry_count == 1
        assert updated.status == "pending"
        assert updated.last_retry_at is not None

@pytest.mark.asyncio
async def test_retry_metrics(retry_test_system):
    """Test retry metrics collection."""
    service, retry_service = retry_test_system
    
    # Create notification
    notification = await service.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mark as failed and retry
    await service.update_notification(notification.id, status="failed")
    await retry_service.retry_notification(notification.id)
    
    # Get metrics
    metrics = retry_service.get_metrics()
    
    # Verify metrics
    assert metrics["total_retries"] == 1
    assert metrics["successful_retries"] == 0
    assert metrics["failed_retries"] == 1
    assert metrics["average_retry_delay"] > 0
    assert metrics["max_retry_delay"] > 0
    assert metrics["min_retry_delay"] > 0
    assert metrics["retries_by_priority"]["MEDIUM"] == 1
    assert metrics["retries_by_channel"]["EMAIL"] == 1 