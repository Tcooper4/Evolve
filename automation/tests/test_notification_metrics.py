import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
import json
import time

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)

@pytest.fixture
async def metrics_test_system():
    """Create a notification system for metrics testing."""
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_notification_count_metrics(metrics_test_system):
    """Test counting metrics for notifications."""
    # Create notifications of different types
    for i in range(3):
        await metrics_test_system.send_notification(
            title=f"Info Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    for i in range(2):
        await metrics_test_system.send_notification(
            title=f"Warning Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.WARNING,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.SLACK,
            recipient="#test-channel"
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify total count
    assert metrics["total_notifications"] == 5
    
    # Verify type counts
    assert metrics["notifications_by_type"]["INFO"] == 3
    assert metrics["notifications_by_type"]["WARNING"] == 2
    
    # Verify channel counts
    assert metrics["notifications_by_channel"]["EMAIL"] == 3
    assert metrics["notifications_by_channel"]["SLACK"] == 2
    
    # Verify priority counts
    assert metrics["notifications_by_priority"]["MEDIUM"] == 3
    assert metrics["notifications_by_priority"]["HIGH"] == 2

@pytest.mark.asyncio
async def test_notification_status_metrics(metrics_test_system):
    """Test status metrics for notifications."""
    # Create notifications with different statuses
    for i in range(3):
        notification = await metrics_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        await metrics_test_system.update_notification(
            notification.id,
            status="completed"
        )
    
    for i in range(2):
        notification = await metrics_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        await metrics_test_system.update_notification(
            notification.id,
            status="failed"
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify status counts
    assert metrics["notifications_by_status"]["completed"] == 3
    assert metrics["notifications_by_status"]["failed"] == 2

@pytest.mark.asyncio
async def test_notification_latency_metrics(metrics_test_system):
    """Test latency metrics for notifications."""
    # Create notifications and measure latency
    latencies = []
    for i in range(5):
        start_time = time.time()
        await metrics_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        latency = time.time() - start_time
        latencies.append(latency)
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify latency metrics
    assert "average_latency" in metrics
    assert "min_latency" in metrics
    assert "max_latency" in metrics
    assert "p95_latency" in metrics
    assert "p99_latency" in metrics
    
    # Verify latency values
    assert metrics["average_latency"] == sum(latencies) / len(latencies)
    assert metrics["min_latency"] == min(latencies)
    assert metrics["max_latency"] == max(latencies)

@pytest.mark.asyncio
async def test_notification_error_metrics(metrics_test_system):
    """Test error metrics for notifications."""
    # Create notifications with errors
    for i in range(3):
        try:
            await metrics_test_system.send_notification(
                title=f"Test Notification {i}",
                message=f"Test message {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel="INVALID_CHANNEL",  # Invalid channel
                recipient="test@example.com"
            )
        except ValueError:
            pass
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify error metrics
    assert "error_count" in metrics
    assert "error_rate" in metrics
    assert "errors_by_type" in metrics
    
    # Verify error values
    assert metrics["error_count"] == 3
    assert metrics["error_rate"] == 1.0  # All notifications failed
    assert "ValueError" in metrics["errors_by_type"]

@pytest.mark.asyncio
async def test_notification_retry_metrics(metrics_test_system):
    """Test retry metrics for notifications."""
    # Create notifications with retries
    for i in range(3):
        notification = await metrics_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        await metrics_test_system.update_notification(
            notification.id,
            retry_count=i + 1
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify retry metrics
    assert "total_retries" in metrics
    assert "average_retries" in metrics
    assert "max_retries" in metrics
    
    # Verify retry values
    assert metrics["total_retries"] == 6  # 1 + 2 + 3
    assert metrics["average_retries"] == 2.0  # (1 + 2 + 3) / 3
    assert metrics["max_retries"] == 3

@pytest.mark.asyncio
async def test_notification_cleanup_metrics(metrics_test_system):
    """Test cleanup metrics for notifications."""
    # Create old notifications
    old_date = datetime.now().timestamp() - 86400  # 1 day old
    for i in range(5):
        notification = await metrics_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notification.created_at = old_date
    
    # Run cleanup
    await metrics_test_system.cleanup_old_notifications()
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify cleanup metrics
    assert "cleaned_up_notifications" in metrics
    assert "cleanup_duration" in metrics
    
    # Verify cleanup values
    assert metrics["cleaned_up_notifications"] == 5
    assert metrics["cleanup_duration"] > 0

@pytest.mark.asyncio
async def test_notification_channel_metrics(metrics_test_system):
    """Test channel-specific metrics for notifications."""
    # Create notifications for different channels
    for i in range(3):
        await metrics_test_system.send_notification(
            title=f"Email Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    for i in range(2):
        await metrics_test_system.send_notification(
            title=f"Slack Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.SLACK,
            recipient="#test-channel"
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify channel metrics
    assert "channel_metrics" in metrics
    assert "EMAIL" in metrics["channel_metrics"]
    assert "SLACK" in metrics["channel_metrics"]
    
    # Verify email metrics
    assert metrics["channel_metrics"]["EMAIL"]["total"] == 3
    assert metrics["channel_metrics"]["EMAIL"]["success"] == 3
    assert metrics["channel_metrics"]["EMAIL"]["failed"] == 0
    
    # Verify Slack metrics
    assert metrics["channel_metrics"]["SLACK"]["total"] == 2
    assert metrics["channel_metrics"]["SLACK"]["success"] == 2
    assert metrics["channel_metrics"]["SLACK"]["failed"] == 0

@pytest.mark.asyncio
async def test_notification_type_metrics(metrics_test_system):
    """Test type-specific metrics for notifications."""
    # Create notifications of different types
    for i in range(3):
        await metrics_test_system.send_notification(
            title=f"Info Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    for i in range(2):
        await metrics_test_system.send_notification(
            title=f"Warning Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.WARNING,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify type metrics
    assert "type_metrics" in metrics
    assert "INFO" in metrics["type_metrics"]
    assert "WARNING" in metrics["type_metrics"]
    
    # Verify info metrics
    assert metrics["type_metrics"]["INFO"]["total"] == 3
    assert metrics["type_metrics"]["INFO"]["success"] == 3
    assert metrics["type_metrics"]["INFO"]["failed"] == 0
    
    # Verify warning metrics
    assert metrics["type_metrics"]["WARNING"]["total"] == 2
    assert metrics["type_metrics"]["WARNING"]["success"] == 2
    assert metrics["type_metrics"]["WARNING"]["failed"] == 0

@pytest.mark.asyncio
async def test_notification_priority_metrics(metrics_test_system):
    """Test priority-specific metrics for notifications."""
    # Create notifications of different priorities
    for i in range(3):
        await metrics_test_system.send_notification(
            title=f"Medium Priority Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    for i in range(2):
        await metrics_test_system.send_notification(
            title=f"High Priority Notification {i}",
            message=f"Test message {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Get metrics
    metrics = await metrics_test_system.get_metrics()
    
    # Verify priority metrics
    assert "priority_metrics" in metrics
    assert "MEDIUM" in metrics["priority_metrics"]
    assert "HIGH" in metrics["priority_metrics"]
    
    # Verify medium priority metrics
    assert metrics["priority_metrics"]["MEDIUM"]["total"] == 3
    assert metrics["priority_metrics"]["MEDIUM"]["success"] == 3
    assert metrics["priority_metrics"]["MEDIUM"]["failed"] == 0
    
    # Verify high priority metrics
    assert metrics["priority_metrics"]["HIGH"]["total"] == 2
    assert metrics["priority_metrics"]["HIGH"]["success"] == 2
    assert metrics["priority_metrics"]["HIGH"]["failed"] == 0 