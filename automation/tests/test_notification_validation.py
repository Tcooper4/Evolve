import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
import json
import re

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.validation import NotificationValidator

@pytest.fixture
async def validation_test_system():
    """Create a notification system for validation testing."""
    service = NotificationService()
    validator = NotificationValidator()
    await service.start()
    yield service, validator
    await service.stop()

def test_validate_notification_type(validation_test_system):
    """Test validation of notification types."""
    _, validator = validation_test_system
    
    # Test valid types
    assert validator.validate_type(NotificationType.INFO) is True
    assert validator.validate_type(NotificationType.WARNING) is True
    assert validator.validate_type(NotificationType.ERROR) is True
    assert validator.validate_type(NotificationType.SUCCESS) is True
    
    # Test invalid types
    assert validator.validate_type("INVALID_TYPE") is False
    assert validator.validate_type("") is False
    assert validator.validate_type(None) is False

def test_validate_notification_priority(validation_test_system):
    """Test validation of notification priorities."""
    _, validator = validation_test_system
    
    # Test valid priorities
    assert validator.validate_priority(NotificationPriority.LOW) is True
    assert validator.validate_priority(NotificationPriority.MEDIUM) is True
    assert validator.validate_priority(NotificationPriority.HIGH) is True
    assert validator.validate_priority(NotificationPriority.CRITICAL) is True
    
    # Test invalid priorities
    assert validator.validate_priority("INVALID_PRIORITY") is False
    assert validator.validate_priority("") is False
    assert validator.validate_priority(None) is False

def test_validate_notification_channel(validation_test_system):
    """Test validation of notification channels."""
    _, validator = validation_test_system
    
    # Test valid channels
    assert validator.validate_channel(NotificationChannel.EMAIL) is True
    assert validator.validate_channel(NotificationChannel.SLACK) is True
    assert validator.validate_channel(NotificationChannel.WEBHOOK) is True
    
    # Test invalid channels
    assert validator.validate_channel("INVALID_CHANNEL") is False
    assert validator.validate_channel("") is False
    assert validator.validate_channel(None) is False

def test_validate_notification_title(validation_test_system):
    """Test validation of notification titles."""
    _, validator = validation_test_system
    
    # Test valid titles
    assert validator.validate_title("Valid Title") is True
    assert validator.validate_title("Title with 123") is True
    assert validator.validate_title("Title with !@#$%^&*()") is True
    
    # Test invalid titles
    assert validator.validate_title("") is False
    assert validator.validate_title(None) is False
    assert validator.validate_title("x" * 256) is False  # Too long

def test_validate_notification_message(validation_test_system):
    """Test validation of notification messages."""
    _, validator = validation_test_system
    
    # Test valid messages
    assert validator.validate_message("Valid message") is True
    assert validator.validate_message("Message with 123") is True
    assert validator.validate_message("Message with !@#$%^&*()") is True
    
    # Test invalid messages
    assert validator.validate_message("") is False
    assert validator.validate_message(None) is False
    assert validator.validate_message("x" * 4096) is False  # Too long

def test_validate_notification_recipient(validation_test_system):
    """Test validation of notification recipients."""
    _, validator = validation_test_system
    
    # Test valid email recipients
    assert validator.validate_recipient("test@example.com", NotificationChannel.EMAIL) is True
    assert validator.validate_recipient("user.name@domain.com", NotificationChannel.EMAIL) is True
    
    # Test valid Slack recipients
    assert validator.validate_recipient("#channel", NotificationChannel.SLACK) is True
    assert validator.validate_recipient("@user", NotificationChannel.SLACK) is True
    
    # Test valid webhook recipients
    assert validator.validate_recipient("https://api.example.com/webhook", NotificationChannel.WEBHOOK) is True
    assert validator.validate_recipient("http://localhost:8080/webhook", NotificationChannel.WEBHOOK) is True
    
    # Test invalid recipients
    assert validator.validate_recipient("invalid-email", NotificationChannel.EMAIL) is False
    assert validator.validate_recipient("invalid-channel", NotificationChannel.SLACK) is False
    assert validator.validate_recipient("invalid-url", NotificationChannel.WEBHOOK) is False
    assert validator.validate_recipient("", NotificationChannel.EMAIL) is False
    assert validator.validate_recipient(None, NotificationChannel.EMAIL) is False

def test_validate_notification_metadata(validation_test_system):
    """Test validation of notification metadata."""
    _, validator = validation_test_system
    
    # Test valid metadata
    assert validator.validate_metadata({"key": "value"}) is True
    assert validator.validate_metadata({"number": 123}) is True
    assert validator.validate_metadata({"list": [1, 2, 3]}) is True
    assert validator.validate_metadata({"nested": {"key": "value"}}) is True
    
    # Test invalid metadata
    assert validator.validate_metadata({"key": object()}) is False  # Invalid type
    assert validator.validate_metadata({"x" * 1000: "value"}) is False  # Key too long
    assert validator.validate_metadata({"key": "x" * 1000}) is False  # Value too long
    assert validator.validate_metadata(None) is False

def test_validate_notification_id(validation_test_system):
    """Test validation of notification IDs."""
    _, validator = validation_test_system
    
    # Test valid IDs
    assert validator.validate_id("test-123") is True
    assert validator.validate_id("notification-456") is True
    assert validator.validate_id("1234567890") is True
    
    # Test invalid IDs
    assert validator.validate_id("") is False
    assert validator.validate_id(None) is False
    assert validator.validate_id("x" * 256) is False  # Too long

def test_validate_notification_status(validation_test_system):
    """Test validation of notification statuses."""
    _, validator = validation_test_system
    
    # Test valid statuses
    assert validator.validate_status("pending") is True
    assert validator.validate_status("completed") is True
    assert validator.validate_status("failed") is True
    
    # Test invalid statuses
    assert validator.validate_status("INVALID_STATUS") is False
    assert validator.validate_status("") is False
    assert validator.validate_status(None) is False

def test_validate_notification_retry_count(validation_test_system):
    """Test validation of notification retry counts."""
    _, validator = validation_test_system
    
    # Test valid retry counts
    assert validator.validate_retry_count(0) is True
    assert validator.validate_retry_count(1) is True
    assert validator.validate_retry_count(5) is True
    
    # Test invalid retry counts
    assert validator.validate_retry_count(-1) is False
    assert validator.validate_retry_count(100) is False  # Too many retries
    assert validator.validate_retry_count(None) is False

def test_validate_notification_timestamp(validation_test_system):
    """Test validation of notification timestamps."""
    _, validator = validation_test_system
    
    # Test valid timestamps
    assert validator.validate_timestamp(datetime.now().timestamp()) is True
    assert validator.validate_timestamp(0) is True
    assert validator.validate_timestamp(1609459200) is True  # 2021-01-01 00:00:00
    
    # Test invalid timestamps
    assert validator.validate_timestamp(-1) is False
    assert validator.validate_timestamp(None) is False
    assert validator.validate_timestamp("invalid") is False

def test_validate_notification_complete(validation_test_system):
    """Test complete notification validation."""
    _, validator = validation_test_system
    
    # Test valid notification
    valid_notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "channel": NotificationChannel.EMAIL,
        "recipient": "test@example.com",
        "metadata": {
            "key": "value"
        },
        "status": "pending",
        "retry_count": 0,
        "created_at": datetime.now().timestamp()
    }
    
    assert validator.validate_notification(valid_notification) is True
    
    # Test invalid notification
    invalid_notification = {
        "id": "",
        "title": "",
        "message": "",
        "type": "INVALID_TYPE",
        "priority": "INVALID_PRIORITY",
        "channel": "INVALID_CHANNEL",
        "recipient": "invalid-recipient",
        "metadata": None,
        "status": "INVALID_STATUS",
        "retry_count": -1,
        "created_at": -1
    }
    
    assert validator.validate_notification(invalid_notification) is False

def test_validate_notification_partial(validation_test_system):
    """Test partial notification validation."""
    _, validator = validation_test_system
    
    # Test valid partial notification
    valid_partial = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "channel": NotificationChannel.EMAIL,
        "recipient": "test@example.com"
    }
    
    assert validator.validate_partial_notification(valid_partial) is True
    
    # Test invalid partial notification
    invalid_partial = {
        "title": "",
        "message": "",
        "type": "INVALID_TYPE",
        "priority": "INVALID_PRIORITY",
        "channel": "INVALID_CHANNEL",
        "recipient": "invalid-recipient"
    }
    
    assert validator.validate_partial_notification(invalid_partial) is False

def test_validate_notification_update(validation_test_system):
    """Test notification update validation."""
    _, validator = validation_test_system
    
    # Test valid update
    valid_update = {
        "status": "completed",
        "retry_count": 1,
        "metadata": {
            "key": "value"
        }
    }
    
    assert validator.validate_update(valid_update) is True
    
    # Test invalid update
    invalid_update = {
        "status": "INVALID_STATUS",
        "retry_count": -1,
        "metadata": None
    }
    
    assert validator.validate_update(invalid_update) is False

def test_validate_notification_query(validation_test_system):
    """Test notification query validation."""
    _, validator = validation_test_system
    
    # Test valid query
    valid_query = {
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "channel": NotificationChannel.EMAIL,
        "status": "pending",
        "start_date": datetime.now().timestamp(),
        "end_date": datetime.now().timestamp()
    }
    
    assert validator.validate_query(valid_query) is True
    
    # Test invalid query
    invalid_query = {
        "type": "INVALID_TYPE",
        "priority": "INVALID_PRIORITY",
        "channel": "INVALID_CHANNEL",
        "status": "INVALID_STATUS",
        "start_date": -1,
        "end_date": -1
    }
    
    assert validator.validate_query(invalid_query) is False 