import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.notification_cleanup import NotificationCleanupService

@pytest.fixture
async def performance_system():
    """Create a notification system for performance testing."""
    notification_service = NotificationService()
    cleanup_service = NotificationCleanupService(
        notification_service,
        retention_days=1,
        cleanup_interval=1
    )
    
    # Create mock handlers
    email_handler = Mock()
    slack_handler = Mock()
    webhook_handler = Mock()
    
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
async def test_notification_send_performance(performance_system):
    """Test the performance of sending notifications."""
    system = performance_system
    notification_service = system["notification_service"]
    
    # Number of notifications to send
    num_notifications = 1000
    
    # Start timing
    start_time = time.time()
    
    # Send notifications
    tasks = []
    for i in range(num_notifications):
        task = notification_service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        tasks.append(task)
    
    # Wait for all notifications to be sent
    await asyncio.gather(*tasks)
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    notifications_per_second = num_notifications / total_time
    
    # Verify performance
    assert notifications_per_second >= 100  # Minimum 100 notifications per second
    assert total_time < 10  # Should complete within 10 seconds

@pytest.mark.asyncio
async def test_notification_cleanup_performance(performance_system):
    """Test the performance of notification cleanup."""
    system = performance_system
    notification_service = system["notification_service"]
    cleanup_service = system["cleanup_service"]
    
    # Number of notifications to create
    num_notifications = 10000
    
    # Create old notifications
    old_date = datetime.utcnow() - timedelta(days=2)
    with patch('automation.notifications.notification_service.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = old_date
        for i in range(num_notifications):
            await notification_service.send_notification(
                title=f"Old Notification {i}",
                message=f"This is old notification {i}",
                type=NotificationType.INFO,
                priority=NotificationPriority.LOW,
                channel=NotificationChannel.EMAIL,
                recipient="test@example.com"
            )
    
    # Start timing
    start_time = time.time()
    
    # Run cleanup
    await cleanup_service.cleanup_now()
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    notifications_per_second = num_notifications / total_time
    
    # Verify performance
    assert notifications_per_second >= 1000  # Minimum 1000 notifications per second
    assert total_time < 10  # Should complete within 10 seconds

@pytest.mark.asyncio
async def test_notification_query_performance(performance_system):
    """Test the performance of notification queries."""
    system = performance_system
    notification_service = system["notification_service"]
    
    # Number of notifications to create
    num_notifications = 10000
    
    # Create notifications with different types and channels
    for i in range(num_notifications):
        channel = NotificationChannel.EMAIL if i % 2 == 0 else NotificationChannel.SLACK
        type_ = NotificationType.INFO if i % 2 == 0 else NotificationType.ERROR
        await notification_service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is test notification {i}",
            type=type_,
            priority=NotificationPriority.MEDIUM,
            channel=channel,
            recipient="test@example.com"
        )
    
    # Test query performance
    start_time = time.time()
    
    # Query notifications
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
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify performance
    assert total_time < 1  # Should complete within 1 second
    assert len(info_notifications) == num_notifications // 2
    assert len(error_notifications) == num_notifications // 2
    assert len(email_notifications) == num_notifications // 2
    assert len(slack_notifications) == num_notifications // 2

@pytest.mark.asyncio
async def test_notification_concurrent_performance(performance_system):
    """Test the performance of concurrent notification operations."""
    system = performance_system
    notification_service = system["notification_service"]
    
    # Number of concurrent operations
    num_operations = 1000
    
    # Start timing
    start_time = time.time()
    
    # Perform concurrent operations
    tasks = []
    for i in range(num_operations):
        # Send notification
        send_task = notification_service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
        tasks.append(send_task)
        
        # Query notifications
        query_task = notification_service.get_notifications(
            type=NotificationType.INFO
        )
        tasks.append(query_task)
    
    # Wait for all operations to complete
    await asyncio.gather(*tasks)
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    operations_per_second = num_operations * 2 / total_time
    
    # Verify performance
    assert operations_per_second >= 100  # Minimum 100 operations per second
    assert total_time < 20  # Should complete within 20 seconds

@pytest.mark.asyncio
async def test_notification_memory_usage(performance_system):
    """Test the memory usage of the notification system."""
    import psutil
    import os
    
    system = performance_system
    notification_service = system["notification_service"]
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Number of notifications to create
    num_notifications = 10000
    
    # Create notifications
    for i in range(num_notifications):
        await notification_service.send_notification(
            title=f"Test Notification {i}",
            message=f"This is test notification {i}",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Get final memory usage
    final_memory = process.memory_info().rss
    memory_per_notification = (final_memory - initial_memory) / num_notifications
    
    # Verify memory usage
    assert memory_per_notification < 1024  # Less than 1KB per notification
    assert final_memory < 1024 * 1024 * 100  # Less than 100MB total 