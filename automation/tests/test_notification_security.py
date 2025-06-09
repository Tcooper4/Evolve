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

@pytest.fixture
async def security_test_system():
    """Create a notification system for security testing."""
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_xss_prevention(security_test_system):
    """Test prevention of XSS attacks in notifications."""
    # Create notification with potential XSS payload
    xss_payload = "<script>alert('xss')</script>"
    notification = await security_test_system.send_notification(
        title=xss_payload,
        message=xss_payload,
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com",
        metadata={"key": xss_payload}
    )
    
    # Verify XSS prevention
    assert "<script>" not in notification.title
    assert "<script>" not in notification.message
    assert "<script>" not in str(notification.metadata)
    
    # Verify HTML encoding
    assert "&lt;script&gt;" in notification.title
    assert "&lt;script&gt;" in notification.message
    assert "&lt;script&gt;" in str(notification.metadata)

@pytest.mark.asyncio
async def test_sql_injection_prevention(security_test_system):
    """Test prevention of SQL injection attacks."""
    # Create notification with potential SQL injection payload
    sql_payload = "'; DROP TABLE notifications; --"
    notification = await security_test_system.send_notification(
        title=sql_payload,
        message=sql_payload,
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient=sql_payload,
        metadata={"key": sql_payload}
    )
    
    # Verify SQL injection prevention
    assert "DROP TABLE" not in notification.title
    assert "DROP TABLE" not in notification.message
    assert "DROP TABLE" not in notification.recipient
    assert "DROP TABLE" not in str(notification.metadata)
    
    # Verify proper escaping
    assert "';" in notification.title
    assert "';" in notification.message
    assert "';" in notification.recipient
    assert "';" in str(notification.metadata)

@pytest.mark.asyncio
async def test_command_injection_prevention(security_test_system):
    """Test prevention of command injection attacks."""
    # Create notification with potential command injection payload
    cmd_payload = "& rm -rf /"
    notification = await security_test_system.send_notification(
        title=cmd_payload,
        message=cmd_payload,
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient=cmd_payload,
        metadata={"key": cmd_payload}
    )
    
    # Verify command injection prevention
    assert "rm -rf" not in notification.title
    assert "rm -rf" not in notification.message
    assert "rm -rf" not in notification.recipient
    assert "rm -rf" not in str(notification.metadata)
    
    # Verify proper escaping
    assert "&" in notification.title
    assert "&" in notification.message
    assert "&" in notification.recipient
    assert "&" in str(notification.metadata)

@pytest.mark.asyncio
async def test_input_validation(security_test_system):
    """Test input validation for notification fields."""
    # Test title validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="",  # Empty title
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Test message validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="",  # Empty message
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com"
        )
    
    # Test recipient validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient=""  # Empty recipient
        )

@pytest.mark.asyncio
async def test_metadata_validation(security_test_system):
    """Test validation of notification metadata."""
    # Test metadata size limit
    large_metadata = {"key": "x" * 10000}  # 10KB metadata
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com",
            metadata=large_metadata
        )
    
    # Test metadata type validation
    invalid_metadata = {"key": object()}  # Invalid type
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com",
            metadata=invalid_metadata
        )

@pytest.mark.asyncio
async def test_rate_limiting(security_test_system):
    """Test rate limiting for notification sending."""
    # Send multiple notifications rapidly
    tasks = []
    for i in range(100):  # Try to send 100 notifications
        task = security_test_system.send_notification(
            title=f"Test Notification {i}",
            message=f"Test message {i}",
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
    assert success_count <= 50  # Should be rate limited

@pytest.mark.asyncio
async def test_channel_security(security_test_system):
    """Test security of notification channels."""
    # Test email validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.EMAIL,
            recipient="invalid-email"  # Invalid email
        )
    
    # Test webhook URL validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.WEBHOOK,
            recipient="invalid-url"  # Invalid URL
        )
    
    # Test Slack channel validation
    with pytest.raises(ValueError):
        await security_test_system.send_notification(
            title="Test title",
            message="Test message",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.SLACK,
            recipient="invalid-channel"  # Invalid channel
        )

@pytest.mark.asyncio
async def test_notification_encryption(security_test_system):
    """Test encryption of sensitive notification data."""
    # Create notification with sensitive data
    sensitive_data = {
        "password": "secret123",
        "token": "sensitive-token",
        "api_key": "secret-key"
    }
    
    notification = await security_test_system.send_notification(
        title="Test title",
        message="Test message",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com",
        metadata=sensitive_data
    )
    
    # Verify sensitive data is encrypted
    assert "secret123" not in str(notification.metadata)
    assert "sensitive-token" not in str(notification.metadata)
    assert "secret-key" not in str(notification.metadata)
    
    # Verify encryption pattern
    assert re.search(r'[A-Za-z0-9+/]{32,}={0,2}', str(notification.metadata)) 