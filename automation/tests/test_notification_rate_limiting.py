import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.rate_limiting import NotificationRateLimiter

@pytest.fixture
async def rate_limit_test_system():
    """Create a notification system for rate limiting testing."""
    service = NotificationService()
    rate_limiter = NotificationRateLimiter(
        max_notifications_per_minute=10,
        max_notifications_per_hour=100
    )
    await service.start()
    yield service, rate_limiter
    await service.stop()

@pytest.mark.asyncio
async def test_rate_limiting_per_minute(rate_limit_test_system):
    """Test rate limiting per minute."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications rapidly
    tasks = []
    for i in range(15):  # Try to send 15 notifications
        task = service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify rate limiting
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count <= 10  # Should be rate limited to 10 per minute
    
    # Verify rate limiter state
    assert rate_limiter.get_minute_count() <= 10
    assert rate_limiter.get_hour_count() <= 100

@pytest.mark.asyncio
async def test_rate_limiting_per_hour(rate_limit_test_system):
    """Test rate limiting per hour."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications over time
    for i in range(120):  # Try to send 120 notifications
        try:
            await service.send_notification(
                title=f"Test Notification {i}",
                message=f"This is a test notification {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
        except Exception:
            pass
        
        # Wait a bit between notifications
        await asyncio.sleep(0.1)
    
    # Verify rate limiter state
    assert rate_limiter.get_hour_count() <= 100  # Should be rate limited to 100 per hour

@pytest.mark.asyncio
async def test_rate_limiting_reset(rate_limit_test_system):
    """Test rate limiter reset."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications up to limit
    for i in range(10):
        await service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Verify rate limiter is at limit
    assert rate_limiter.get_minute_count() == 10
    
    # Wait for minute to reset
    await asyncio.sleep(60)
    
    # Send more notifications
    for i in range(5):
        await service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Verify rate limiter has reset
    assert rate_limiter.get_minute_count() == 5

@pytest.mark.asyncio
async def test_rate_limiting_priority(rate_limit_test_system):
    """Test rate limiting with different priorities."""
    service, rate_limiter = rate_limit_test_system
    
    # Send high priority notifications
    for i in range(5):
        await service.send_notification(
            title=f"High Priority Notification {i}",
            message=f"This is a high priority notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Send medium priority notifications
    for i in range(5):
        await service.send_notification(
            title=f"Medium Priority Notification {i}",
            message=f"This is a medium priority notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Verify rate limiter state
    assert rate_limiter.get_minute_count() == 10
    
    # Try to send more notifications
    with pytest.raises(Exception):
        await service.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )

@pytest.mark.asyncio
async def test_rate_limiting_channel(rate_limit_test_system):
    """Test rate limiting for different channels."""
    service, rate_limiter = rate_limit_test_system
    
    # Send email notifications
    for i in range(5):
        await service.send_notification(
            title=f"Email Notification {i}",
            message=f"This is an email notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Send Slack notifications
    for i in range(5):
        await service.send_notification(
            title=f"Slack Notification {i}",
            message=f"This is a slack notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.SLACK,
            recipient="#test-channel"
        )
    
    # Verify rate limiter state
    assert rate_limiter.get_minute_count() == 10
    
    # Try to send more notifications
    with pytest.raises(Exception):
        await service.send_notification(
            title="Test Notification",
            message="This is a test notification",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )

@pytest.mark.asyncio
async def test_rate_limiting_concurrent(rate_limit_test_system):
    """Test rate limiting with concurrent requests."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications concurrently
    tasks = []
    for i in range(20):  # Try to send 20 notifications
        task = service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is a test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify rate limiting
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count <= 10  # Should be rate limited to 10 per minute
    
    # Verify rate limiter state
    assert rate_limiter.get_minute_count() <= 10
    assert rate_limiter.get_hour_count() <= 100

@pytest.mark.asyncio
async def test_rate_limiting_burst(rate_limit_test_system):
    """Test rate limiting with burst traffic."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications in bursts
    for burst in range(3):
        tasks = []
        for i in range(10):  # Try to send 10 notifications per burst
            task = service.send_notification(
                title=f"Burst {burst} Notification {i}",
                message=f"This is a test notification {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
            tasks.append(task)
        
        # Wait for burst to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify rate limiting
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count <= 10  # Should be rate limited to 10 per minute
        
        # Wait between bursts
        await asyncio.sleep(60)
    
    # Verify rate limiter state
    assert rate_limiter.get_minute_count() <= 10
    assert rate_limiter.get_hour_count() <= 100

@pytest.mark.asyncio
async def test_rate_limiting_metrics(rate_limit_test_system):
    """Test rate limiting metrics collection."""
    service, rate_limiter = rate_limit_test_system
    
    # Send notifications
    for i in range(15):  # Try to send 15 notifications
        try:
            await service.send_notification(
                title=f"Test Notification {i}",
                message=f"This is a test notification {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.MEDIUM,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
        except Exception:
            pass
    
    # Get metrics
    metrics = rate_limiter.get_metrics()
    
    # Verify metrics
    assert metrics["minute_limit"] == 10
    assert metrics["hour_limit"] == 100
    assert metrics["minute_count"] <= 10
    assert metrics["hour_count"] <= 100
    assert metrics["minute_reset_time"] > 0
    assert metrics["hour_reset_time"] > 0
    assert metrics["total_limited"] >= 5  # At least 5 notifications should be limited 