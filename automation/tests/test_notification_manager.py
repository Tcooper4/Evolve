import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from automation.notifications.notification_manager import (
    NotificationManager,
    NotificationType,
    NotificationPriority
)

@pytest.fixture
def mock_redis():
    redis_mock = Mock()
    redis_mock.set = Mock()
    redis_mock.get = Mock()
    redis_mock.delete = Mock()
    redis_mock.lpush = Mock()
    redis_mock.ltrim = Mock()
    redis_mock.lrange = Mock()
    redis_mock.lrem = Mock()
    redis_mock.publish = AsyncMock()
    redis_mock.pubsub = Mock()
    return redis_mock

@pytest.fixture
def notification_manager(mock_redis):
    return NotificationManager(mock_redis)

@pytest.mark.asyncio
async def test_create_notification(notification_manager, mock_redis):
    # Test data
    title = "Test Notification"
    message = "This is a test notification"
    notification_type = NotificationType.TASK
    priority = NotificationPriority.HIGH
    data = {"task_id": "123"}
    user_id = "user1"
    
    # Create notification
    notification = await notification_manager.create_notification(
        title=title,
        message=message,
        notification_type=notification_type,
        priority=priority,
        data=data,
        user_id=user_id
    )
    
    # Verify notification structure
    assert notification["title"] == title
    assert notification["message"] == message
    assert notification["type"] == notification_type.value
    assert notification["priority"] == priority.value
    assert notification["data"] == data
    assert notification["user_id"] == user_id
    assert notification["read"] is False
    assert "id" in notification
    assert "created_at" in notification
    
    # Verify Redis operations
    mock_redis.set.assert_called_once()
    mock_redis.lpush.assert_called()
    mock_redis.ltrim.assert_called()
    mock_redis.publish.assert_called_once()

@pytest.mark.asyncio
async def test_get_notification(notification_manager, mock_redis):
    # Test data
    notification_id = "task_123"
    notification_data = {
        "id": notification_id,
        "title": "Test",
        "message": "Test message",
        "type": NotificationType.TASK.value,
        "priority": NotificationPriority.MEDIUM.value,
        "data": {},
        "user_id": "user1",
        "created_at": datetime.now().isoformat(),
        "read": False
    }
    
    # Mock Redis get
    mock_redis.get.return_value = json.dumps(notification_data).encode()
    
    # Get notification
    notification = await notification_manager.get_notification(notification_id)
    
    # Verify result
    assert notification == notification_data
    mock_redis.get.assert_called_once_with(f"notification:{notification_id}")

@pytest.mark.asyncio
async def test_get_user_notifications(notification_manager, mock_redis):
    # Test data
    user_id = "user1"
    notification_ids = [b"task_1", b"task_2"]
    notification_data = {
        "id": "task_1",
        "title": "Test",
        "message": "Test message",
        "type": NotificationType.TASK.value,
        "priority": NotificationPriority.MEDIUM.value,
        "data": {},
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "read": False
    }
    
    # Mock Redis operations
    mock_redis.lrange.return_value = notification_ids
    mock_redis.get.return_value = json.dumps(notification_data).encode()
    
    # Get notifications
    notifications = await notification_manager.get_user_notifications(user_id)
    
    # Verify result
    assert len(notifications) == 2
    mock_redis.lrange.assert_called_once_with(f"user_notifications:{user_id}", 0, 49)

@pytest.mark.asyncio
async def test_mark_as_read(notification_manager, mock_redis):
    # Test data
    notification_id = "task_123"
    notification_data = {
        "id": notification_id,
        "title": "Test",
        "message": "Test message",
        "type": NotificationType.TASK.value,
        "priority": NotificationPriority.MEDIUM.value,
        "data": {},
        "user_id": "user1",
        "created_at": datetime.now().isoformat(),
        "read": False
    }
    
    # Mock Redis operations
    mock_redis.get.return_value = json.dumps(notification_data).encode()
    
    # Mark as read
    result = await notification_manager.mark_as_read(notification_id)
    
    # Verify result
    assert result is True
    mock_redis.set.assert_called_once()
    mock_redis.publish.assert_called_once()

@pytest.mark.asyncio
async def test_delete_notification(notification_manager, mock_redis):
    # Test data
    notification_id = "task_123"
    notification_data = {
        "id": notification_id,
        "title": "Test",
        "message": "Test message",
        "type": NotificationType.TASK.value,
        "priority": NotificationPriority.MEDIUM.value,
        "data": {},
        "user_id": "user1",
        "created_at": datetime.now().isoformat(),
        "read": False
    }
    
    # Mock Redis operations
    mock_redis.get.return_value = json.dumps(notification_data).encode()
    
    # Delete notification
    result = await notification_manager.delete_notification(notification_id)
    
    # Verify result
    assert result is True
    mock_redis.delete.assert_called_once()
    mock_redis.lrem.assert_called()
    mock_redis.publish.assert_called_once()

@pytest.mark.asyncio
async def test_subscribe_unsubscribe(notification_manager, mock_redis):
    # Test data
    user_id = "user1"
    callback = AsyncMock()
    
    # Mock pubsub
    pubsub_mock = Mock()
    pubsub_mock.subscribe = AsyncMock()
    pubsub_mock.unsubscribe = AsyncMock()
    pubsub_mock.get_message = AsyncMock(return_value=None)
    mock_redis.pubsub.return_value = pubsub_mock
    
    # Subscribe
    await notification_manager.subscribe(user_id, callback)
    
    # Verify subscription
    assert user_id in notification_manager._notification_channels
    pubsub_mock.subscribe.assert_called_once()
    
    # Unsubscribe
    await notification_manager.unsubscribe(user_id)
    
    # Verify unsubscription
    assert user_id not in notification_manager._notification_channels
    pubsub_mock.unsubscribe.assert_called_once()

@pytest.mark.asyncio
async def test_notification_expiration(notification_manager, mock_redis):
    # Test data
    title = "Expiring Notification"
    message = "This notification will expire"
    notification_type = NotificationType.TASK
    expires_at = datetime.now() + timedelta(hours=1)
    
    # Create notification with expiration
    notification = await notification_manager.create_notification(
        title=title,
        message=message,
        notification_type=notification_type,
        expires_at=expires_at
    )
    
    # Verify expiration
    assert notification["expires_at"] == expires_at.isoformat()

@pytest.mark.asyncio
async def test_notification_priority_levels(notification_manager, mock_redis):
    # Test different priority levels
    priorities = [
        NotificationPriority.LOW,
        NotificationPriority.MEDIUM,
        NotificationPriority.HIGH,
        NotificationPriority.CRITICAL
    ]
    
    for priority in priorities:
        notification = await notification_manager.create_notification(
            title=f"Priority {priority.value}",
            message="Test message",
            notification_type=NotificationType.TASK,
            priority=priority
        )
        
        assert notification["priority"] == priority.value

@pytest.mark.asyncio
async def test_notification_types(notification_manager, mock_redis):
    # Test different notification types
    types = [
        NotificationType.TASK,
        NotificationType.SYSTEM,
        NotificationType.ALERT,
        NotificationType.USER
    ]
    
    for notification_type in types:
        notification = await notification_manager.create_notification(
            title=f"Type {notification_type.value}",
            message="Test message",
            notification_type=notification_type
        )
        
        assert notification["type"] == notification_type.value

@pytest.mark.asyncio
async def test_error_handling(notification_manager, mock_redis):
    # Test Redis error
    mock_redis.get.side_effect = Exception("Redis error")
    
    # Attempt to get notification
    notification = await notification_manager.get_notification("task_123")
    
    # Verify error handling
    assert notification is None
    
    # Test invalid notification data
    mock_redis.get.return_value = b"invalid json"
    
    # Attempt to get notification
    notification = await notification_manager.get_notification("task_123")
    
    # Verify error handling
    assert notification is None 