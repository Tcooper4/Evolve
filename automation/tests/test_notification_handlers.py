import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import ssl

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.handlers import (
    EmailHandler,
    SlackHandler,
    WebhookHandler
)

@pytest.fixture
async def handler_test_system():
    """Create a notification system for handler testing."""
    service = NotificationService()
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
def email_config():
    """Create email handler configuration."""
    return {
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "test@example.com",
        "password": "test-password",
        "from_address": "notifications@example.com",
        "use_tls": True
    }

@pytest.fixture
def slack_config():
    """Create Slack handler configuration."""
    return {
        "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        "default_channel": "#notifications"
    }

@pytest.fixture
def webhook_config():
    """Create webhook handler configuration."""
    return {
        "endpoints": [
            {
                "url": "https://api.example.com/webhook",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer test-token"
                }
            }
        ]
    }

@pytest.mark.asyncio
async def test_email_handler_send(email_config):
    """Test sending email notifications."""
    handler = EmailHandler(email_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "recipient@example.com",
        "metadata": {
            "key": "value"
        }
    }
    
    # Mock SMTP client
    mock_smtp = AsyncMock()
    with patch("aiosmtplib.SMTP", return_value=mock_smtp):
        # Send notification
        result = await handler.send(notification)
        
        # Verify SMTP connection
        mock_smtp.connect.assert_called_once_with(
            hostname=email_config["smtp_server"],
            port=email_config["smtp_port"],
            use_tls=email_config["use_tls"]
        )
        
        # Verify SMTP login
        mock_smtp.login.assert_called_once_with(
            email_config["username"],
            email_config["password"]
        )
        
        # Verify email sending
        mock_smtp.send_message.assert_called_once()
        
        # Verify result
        assert result is True

@pytest.mark.asyncio
async def test_email_handler_validation(email_config):
    """Test email handler validation."""
    handler = EmailHandler(email_config)
    
    # Test valid email
    assert handler.validate_email("test@example.com") is True
    
    # Test invalid email
    assert handler.validate_email("invalid-email") is False
    
    # Test empty email
    assert handler.validate_email("") is False
    
    # Test None email
    assert handler.validate_email(None) is False

@pytest.mark.asyncio
async def test_email_handler_template(email_config):
    """Test email handler template rendering."""
    handler = EmailHandler(email_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "recipient@example.com",
        "metadata": {
            "key": "value"
        }
    }
    
    # Render template
    template = handler.render_template(notification)
    
    # Verify template content
    assert notification["title"] in template
    assert notification["message"] in template
    assert notification["type"] in template
    assert notification["priority"] in template
    assert notification["metadata"]["key"] in template

@pytest.mark.asyncio
async def test_slack_handler_send(slack_config):
    """Test sending Slack notifications."""
    handler = SlackHandler(slack_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "#test-channel",
        "metadata": {
            "key": "value"
        }
    }
    
    # Mock HTTP client
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="ok")
    
    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Send notification
        result = await handler.send(notification)
        
        # Verify HTTP request
        mock_session.post.assert_called_once_with(
            slack_config["webhook_url"],
            json={
                "channel": notification["recipient"],
                "text": f"*{notification['title']}*\n{notification['message']}",
                "attachments": [
                    {
                        "color": handler.get_color(notification["type"]),
                        "fields": [
                            {
                                "title": "Type",
                                "value": notification["type"],
                                "short": True
                            },
                            {
                                "title": "Priority",
                                "value": notification["priority"],
                                "short": True
                            }
                        ]
                    }
                ]
            }
        )
        
        # Verify result
        assert result is True

@pytest.mark.asyncio
async def test_slack_handler_validation(slack_config):
    """Test Slack handler validation."""
    handler = SlackHandler(slack_config)
    
    # Test valid channel
    assert handler.validate_channel("#test-channel") is True
    
    # Test invalid channel
    assert handler.validate_channel("invalid-channel") is False
    
    # Test empty channel
    assert handler.validate_channel("") is False
    
    # Test None channel
    assert handler.validate_channel(None) is False

@pytest.mark.asyncio
async def test_slack_handler_color(slack_config):
    """Test Slack handler color mapping."""
    handler = SlackHandler(slack_config)
    
    # Test color mapping
    assert handler.get_color(NotificationType.INFO) == "#36a64f"  # Green
    assert handler.get_color(NotificationType.WARNING) == "#ffcc00"  # Yellow
    assert handler.get_color(NotificationType.ERROR) == "#ff0000"  # Red
    assert handler.get_color(NotificationType.SUCCESS) == "#36a64f"  # Green

@pytest.mark.asyncio
async def test_webhook_handler_send(webhook_config):
    """Test sending webhook notifications."""
    handler = WebhookHandler(webhook_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "https://api.example.com/webhook",
        "metadata": {
            "key": "value"
        }
    }
    
    # Mock HTTP client
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="ok")
    
    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Send notification
        result = await handler.send(notification)
        
        # Verify HTTP request
        mock_session.post.assert_called_once_with(
            webhook_config["endpoints"][0]["url"],
            headers=webhook_config["endpoints"][0]["headers"],
            json={
                "id": notification["id"],
                "title": notification["title"],
                "message": notification["message"],
                "type": notification["type"],
                "priority": notification["priority"],
                "metadata": notification["metadata"]
            }
        )
        
        # Verify result
        assert result is True

@pytest.mark.asyncio
async def test_webhook_handler_validation(webhook_config):
    """Test webhook handler validation."""
    handler = WebhookHandler(webhook_config)
    
    # Test valid URL
    assert handler.validate_url("https://api.example.com/webhook") is True
    
    # Test invalid URL
    assert handler.validate_url("invalid-url") is False
    
    # Test empty URL
    assert handler.validate_url("") is False
    
    # Test None URL
    assert handler.validate_url(None) is False

@pytest.mark.asyncio
async def test_webhook_handler_retry(webhook_config):
    """Test webhook handler retry mechanism."""
    handler = WebhookHandler(webhook_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "https://api.example.com/webhook",
        "metadata": {
            "key": "value"
        }
    }
    
    # Mock HTTP client with retry
    mock_response = AsyncMock()
    mock_response.status = 500  # Server error
    mock_response.text = AsyncMock(return_value="error")
    
    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Send notification
        result = await handler.send(notification)
        
        # Verify retry attempts
        assert mock_session.post.call_count == 3  # Default max retries
        
        # Verify result
        assert result is False

@pytest.mark.asyncio
async def test_handler_error_handling(handler_test_system, email_config, slack_config, webhook_config):
    """Test error handling in notification handlers."""
    # Create handlers
    email_handler = EmailHandler(email_config)
    slack_handler = SlackHandler(slack_config)
    webhook_handler = WebhookHandler(webhook_config)
    
    # Create test notification
    notification = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "recipient": "test@example.com",
        "metadata": {
            "key": "value"
        }
    }
    
    # Test email handler error
    with patch("aiosmtplib.SMTP", side_effect=Exception("SMTP error")):
        result = await email_handler.send(notification)
        assert result is False
    
    # Test Slack handler error
    with patch("aiohttp.ClientSession", side_effect=Exception("HTTP error")):
        result = await slack_handler.send(notification)
        assert result is False
    
    # Test webhook handler error
    with patch("aiohttp.ClientSession", side_effect=Exception("HTTP error")):
        result = await webhook_handler.send(notification)
        assert result is False

@pytest.mark.asyncio
async def test_handler_concurrent_sending(handler_test_system, email_config, slack_config, webhook_config):
    """Test concurrent sending of notifications."""
    # Create handlers
    email_handler = EmailHandler(email_config)
    slack_handler = SlackHandler(slack_config)
    webhook_handler = WebhookHandler(webhook_config)
    
    # Create test notifications
    notifications = [
        {
            "id": f"test-{i}",
            "title": f"Test Notification {i}",
            "message": f"This is test notification {i}",
            "type": NotificationType.INFO,
            "priority": NotificationPriority.MEDIUM,
            "recipient": "test@example.com",
            "metadata": {
                "key": f"value-{i}"
            }
        }
        for i in range(10)
    ]
    
    # Mock handlers
    with patch.object(email_handler, "send", return_value=True), \
         patch.object(slack_handler, "send", return_value=True), \
         patch.object(webhook_handler, "send", return_value=True):
        
        # Send notifications concurrently
        tasks = []
        for notification in notifications:
            tasks.append(email_handler.send(notification))
            tasks.append(slack_handler.send(notification))
            tasks.append(webhook_handler.send(notification))
        
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert all(results) is True
        assert len(results) == len(notifications) * 3 