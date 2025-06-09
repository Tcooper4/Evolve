import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import psutil
import os

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.notification_cleanup import NotificationCleanupService

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def notification_service():
    """Create a notification service instance for testing."""
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
async def cleanup_service(notification_service):
    """Create a cleanup service instance for testing."""
    service = NotificationCleanupService(
        notification_service,
        retention_days=1,
        cleanup_interval=1
    )
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
def mock_email_handler():
    """Create a mock email handler."""
    return Mock()

@pytest.fixture
def mock_slack_handler():
    """Create a mock Slack handler."""
    return Mock()

@pytest.fixture
def mock_webhook_handler():
    """Create a mock webhook handler."""
    return Mock()

@pytest.fixture
def sample_notification():
    """Create a sample notification for testing."""
    return {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "channel": NotificationChannel.EMAIL,
        "recipient": "test@example.com",
        "metadata": {
            "source": "test",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@pytest.fixture
def old_notification():
    """Create an old notification for testing cleanup."""
    return {
        "title": "Old Notification",
        "message": "This is an old notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.LOW,
        "channel": NotificationChannel.EMAIL,
        "recipient": "test@example.com",
        "created_at": datetime.utcnow() - timedelta(days=2)
    }

@pytest.fixture
def new_notification():
    """Create a new notification for testing cleanup."""
    return {
        "title": "New Notification",
        "message": "This is a new notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.LOW,
        "channel": NotificationChannel.EMAIL,
        "recipient": "test@example.com",
        "created_at": datetime.utcnow()
    }

@pytest.fixture
def notification_system(notification_service, cleanup_service):
    """Create a complete notification system for testing."""
    return {
        "notification_service": notification_service,
        "cleanup_service": cleanup_service
    }

@pytest.fixture
def process_info():
    """Get information about the current process."""
    return psutil.Process(os.getpid())

@pytest.fixture
def memory_tracker(process_info):
    """Create a memory usage tracker."""
    class MemoryTracker:
        def __init__(self, process):
            self.process = process
            self.initial_memory = process.memory_info().rss
        
        def get_current_usage(self):
            return self.process.memory_info().rss
        
        def get_usage_increase(self):
            return self.get_current_usage() - self.initial_memory
        
        def get_usage_per_item(self, num_items):
            return self.get_usage_increase() / num_items if num_items > 0 else 0
    
    return MemoryTracker(process_info)

@pytest.fixture
def performance_timer():
    """Create a performance timer."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.utcnow()
            return self
        
        def stop(self):
            self.end_time = datetime.utcnow()
            return self
        
        def get_duration(self):
            if not self.start_time or not self.end_time:
                return None
            return (self.end_time - self.start_time).total_seconds()
        
        def get_operations_per_second(self, num_operations):
            duration = self.get_duration()
            if not duration:
                return None
            return num_operations / duration
    
    return PerformanceTimer() 