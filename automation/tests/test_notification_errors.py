import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.errors import (
    NotificationError,
    ValidationError,
    HandlerError,
    RetryError,
    RateLimitError,
    CleanupError
)

@pytest.fixture
async def error_test_system():
    """Create a notification system for error testing."""
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_validation_error(error_test_system):
    """Test handling of validation errors."""
    # Test invalid notification type
    with pytest.raises(ValidationError):
        await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type="INVALID_TYPE",
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Test invalid notification priority
    with pytest.raises(ValidationError):
        await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority="INVALID_PRIORITY",
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Test invalid notification channel
    with pytest.raises(ValidationError):
        await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel="INVALID_CHANNEL",
            recipient="test@example.com"
        )
    
    # Test invalid recipient
    with pytest.raises(ValidationError):
        await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="invalid-recipient"
        )

@pytest.mark.asyncio
async def test_handler_error(error_test_system):
    """Test handling of handler errors."""
    # Mock handler to raise error
    with patch("automation.notifications.handlers.EmailHandler.send", side_effect=HandlerError("Handler error")):
        # Send notification
        notification = await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Verify notification status
        assert notification.status == "failed"
        assert notification.error_message == "Handler error"

@pytest.mark.asyncio
async def test_retry_error(error_test_system):
    """Test handling of retry errors."""
    # Create notification
    notification = await error_test_system.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mock retry to raise error
    with patch("automation.notifications.retry.NotificationRetryService.retry_notification", side_effect=RetryError("Retry error")):
        # Try to retry notification
        with pytest.raises(RetryError):
            await error_test_system.retry_notification(notification.id)
        
        # Verify notification status
        updated_notification = await error_test_system.get_notification(notification.id)
        assert updated_notification.status == "failed"
        assert updated_notification.error_message == "Retry error"

@pytest.mark.asyncio
async def test_rate_limit_error(error_test_system):
    """Test handling of rate limit errors."""
    # Mock rate limiter to raise error
    with patch("automation.notifications.rate_limiting.NotificationRateLimiter.check_rate_limit", side_effect=RateLimitError("Rate limit exceeded")):
        # Try to send notification
        with pytest.raises(RateLimitError):
            await error_test_system.send_notification(
                title="Test Notification",
                message="This is a test notification",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )

@pytest.mark.asyncio
async def test_cleanup_error(error_test_system):
    """Test handling of cleanup errors."""
    # Mock cleanup to raise error
    with patch("automation.notifications.cleanup.NotificationCleanupService.cleanup_old_notifications", side_effect=CleanupError("Cleanup error")):
        # Try to run cleanup
        with pytest.raises(CleanupError):
            await error_test_system.cleanup_old_notifications()

@pytest.mark.asyncio
async def test_database_error(error_test_system):
    """Test handling of database errors."""
    # Mock database to raise error
    with patch("automation.notifications.database.NotificationDatabase.save_notification", side_effect=Exception("Database error")):
        # Try to send notification
        with pytest.raises(NotificationError):
            await error_test_system.send_notification(
                title="Test Notification",
                message="This is a test notification",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )

@pytest.mark.asyncio
async def test_network_error(error_test_system):
    """Test handling of network errors."""
    # Mock network request to raise error
    with patch("aiohttp.ClientSession.post", side_effect=Exception("Network error")):
        # Try to send notification
        notification = await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.WEBHOOK,
            recipient="https://api.example.com/webhook"
        )
        
        # Verify notification status
        assert notification.status == "failed"
        assert "Network error" in notification.error_message

@pytest.mark.asyncio
async def test_timeout_error(error_test_system):
    """Test handling of timeout errors."""
    # Mock request to timeout
    with patch("aiohttp.ClientSession.post", side_effect=asyncio.TimeoutError("Request timeout")):
        # Try to send notification
        notification = await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.WEBHOOK,
            recipient="https://api.example.com/webhook"
        )
        
        # Verify notification status
        assert notification.status == "failed"
        assert "Request timeout" in notification.error_message

@pytest.mark.asyncio
async def test_concurrent_error(error_test_system):
    """Test handling of concurrent errors."""
    # Create multiple notifications
    notifications = []
    for i in range(5):
        notification = await error_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        notifications.append(notification)
    
    # Mock concurrent operations to raise errors
    with patch("automation.notifications.database.NotificationDatabase.update_notification", side_effect=Exception("Concurrent error")):
        # Try to update notifications concurrently
        tasks = [
            error_test_system.update_notification(notification.id, status="completed")
            for notification in notifications
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify error handling
        assert all(isinstance(r, Exception) for r in results)
        
        # Verify notification statuses
        for notification in notifications:
            updated = await error_test_system.get_notification(notification.id)
            assert updated.status == "pending"  # Should not be updated

@pytest.mark.asyncio
async def test_error_recovery(error_test_system):
    """Test error recovery mechanism."""
    # Create notification
    notification = await error_test_system.send_notification(
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com"
    )
    
    # Mock handler to fail first, then succeed
    mock_handler = AsyncMock(side_effect=[HandlerError("Handler error"), True])
    with patch("automation.notifications.handlers.EmailHandler.send", mock_handler):
        # Try to send notification
        notification = await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        
        # Verify notification status
        assert notification.status == "completed"
        assert notification.error_message is None

@pytest.mark.asyncio
async def test_error_logging(error_test_system):
    """Test error logging."""
    # Mock logger
    mock_logger = Mock()
    with patch("logging.Logger.error", mock_logger):
        # Try to send invalid notification
        with pytest.raises(ValidationError):
            await error_test_system.send_notification(
                title="Test Notification",
                message="This is a test notification",
                type="INVALID_TYPE",
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
        
        # Verify error logging
        mock_logger.assert_called_once()
        assert "ValidationError" in mock_logger.call_args[0][0]

@pytest.mark.asyncio
async def test_error_metrics(error_test_system):
    """Test error metrics collection."""
    # Try to send invalid notification
    with pytest.raises(ValidationError):
        await error_test_system.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type="INVALID_TYPE",
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Get metrics
    metrics = await error_test_system.get_metrics()
    
    # Verify error metrics
    assert metrics["error_count"] > 0
    assert metrics["error_rate"] > 0
    assert "ValidationError" in metrics["errors_by_type"] 