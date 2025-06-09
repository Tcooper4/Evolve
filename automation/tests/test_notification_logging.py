import pytest
import asyncio
import logging
from datetime import datetime
from unittest.mock import Mock, patch
import json
import os

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)

@pytest.fixture
async def logging_test_system():
    """Create a notification system for logging testing."""
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("notification_service")
    
    # Create log directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "notification.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_notification_creation_logging(logging_test_system, caplog):
    """Test logging of notification creation."""
    with caplog.at_level(logging.INFO):
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Verify creation logs
        assert "Creating notification" in caplog.text
        assert notification.id in caplog.text
        assert "Test Notification" in caplog.text
        assert "EMAIL" in caplog.text

@pytest.mark.asyncio
async def test_notification_sending_logging(logging_test_system, caplog):
    """Test logging of notification sending."""
    with caplog.at_level(logging.INFO):
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Verify sending logs
        assert "Sending notification" in caplog.text
        assert notification.id in caplog.text
        assert "EMAIL" in caplog.text
        assert "Test Notification" in caplog.text

@pytest.mark.asyncio
async def test_notification_error_logging(logging_test_system, caplog):
    """Test logging of notification errors."""
    with caplog.at_level(logging.ERROR):
        # Create notification with invalid channel
        with pytest.raises(ValueError):
            await logging_test_system.send_notification(
                title="Test Notification",
                message="Test message",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel="INVALID_CHANNEL",  # Invalid channel
                recipient="test@example.com"
            )
        
        # Verify error logs
        assert "Error creating notification" in caplog.text
        assert "INVALID_CHANNEL" in caplog.text

@pytest.mark.asyncio
async def test_notification_retry_logging(logging_test_system, caplog):
    """Test logging of notification retries."""
    with caplog.at_level(logging.WARNING):
        # Create notification with failing handler
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Simulate handler failure
        notification.status = "failed"
        notification.retry_count = 1
        
        # Verify retry logs
        assert "Retrying notification" in caplog.text
        assert notification.id in caplog.text
        assert "1" in caplog.text  # Retry count

@pytest.mark.asyncio
async def test_notification_cleanup_logging(logging_test_system, caplog):
    """Test logging of notification cleanup."""
    with caplog.at_level(logging.INFO):
        # Create old notification
        old_date = datetime.now().timestamp() - 86400  # 1 day old
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notification.created_at = old_date
        
        # Run cleanup
        await logging_test_system.cleanup_old_notifications()
        
        # Verify cleanup logs
        assert "Cleaning up old notifications" in caplog.text
        assert notification.id in caplog.text

@pytest.mark.asyncio
async def test_notification_query_logging(logging_test_system, caplog):
    """Test logging of notification queries."""
    with caplog.at_level(logging.DEBUG):
        # Create test notifications
        for i in range(3):
            await logging_test_system.send_notification(
                title=f"Test Notification {i}",
                message=f"Test message {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
        
        # Query notifications
        notifications = await logging_test_system.get_notifications(
            type=NotificationType.INFO,
            channel=NotificationChannel.EMAIL
        )
        
        # Verify query logs
        assert "Querying notifications" in caplog.text
        assert "INFO" in caplog.text
        assert "EMAIL" in caplog.text
        assert str(len(notifications)) in caplog.text

@pytest.mark.asyncio
async def test_notification_update_logging(logging_test_system, caplog):
    """Test logging of notification updates."""
    with caplog.at_level(logging.INFO):
        # Create notification
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Update notification
        await logging_test_system.update_notification(
            notification.id,
            status="completed"
        )
        
        # Verify update logs
        assert "Updating notification" in caplog.text
        assert notification.id in caplog.text
        assert "completed" in caplog.text

@pytest.mark.asyncio
async def test_notification_delete_logging(logging_test_system, caplog):
    """Test logging of notification deletion."""
    with caplog.at_level(logging.INFO):
        # Create notification
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Delete notification
        await logging_test_system.delete_notification(notification.id)
        
        # Verify deletion logs
        assert "Deleting notification" in caplog.text
        assert notification.id in caplog.text

@pytest.mark.asyncio
async def test_notification_handler_logging(logging_test_system, caplog):
    """Test logging of notification handler operations."""
    with caplog.at_level(logging.DEBUG):
        # Create notification
        notification = await logging_test_system.send_notification(
            title="Test Notification",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Verify handler logs
        assert "Handler processing notification" in caplog.text
        assert notification.id in caplog.text
        assert "EMAIL" in caplog.text
        assert "Test Notification" in caplog.text

@pytest.mark.asyncio
async def test_notification_metrics_logging(logging_test_system, caplog):
    """Test logging of notification metrics."""
    with caplog.at_level(logging.INFO):
        # Create multiple notifications
        for i in range(5):
            await logging_test_system.send_notification(
                title=f"Test Notification {i}",
                message=f"Test message {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
        
        # Verify metrics logs
        assert "Notification metrics" in caplog.text
        assert "total" in caplog.text
        assert "success" in caplog.text
        assert "failed" in caplog.text
        assert "pending" in caplog.text 