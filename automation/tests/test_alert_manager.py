import pytest
import asyncio
from datetime import datetime
import json
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import redis

from ..monitoring.alert_manager import AlertManager

@pytest.fixture
def test_config():
    return {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "monitoring": {
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "task_success_rate": 90,
                "min_active_agents": 3,
                "model_accuracy": 85
            },
            "alert_check_interval": 1
        },
        "notifications": {
            "email": {
                "from": "test@example.com",
                "to": "admin@example.com",
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "test",
                "password": "password"
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/test"
            }
        }
    }

@pytest.fixture
def mock_redis():
    redis_mock = AsyncMock(spec=redis.Redis)
    redis_mock.hset = AsyncMock()
    redis_mock.hgetall = AsyncMock()
    redis_mock.close = AsyncMock()
    return redis_mock

@pytest.fixture
def mock_http_session():
    session_mock = AsyncMock(spec=aiohttp.ClientSession)
    session_mock.post = AsyncMock()
    session_mock.close = AsyncMock()
    return session_mock

@pytest.fixture
def alert_manager(test_config, mock_redis, mock_http_session):
    with patch("automation.monitoring.alert_manager.load_config", return_value=test_config), \
         patch("redis.Redis", return_value=mock_redis), \
         patch("aiohttp.ClientSession", return_value=mock_http_session):
        manager = AlertManager()
        return manager

@pytest.mark.asyncio
async def test_alert_manager_initialization(alert_manager):
    """Test alert manager initialization."""
    await alert_manager.initialize()
    assert alert_manager.redis_client is not None
    assert alert_manager.http_session is not None

@pytest.mark.asyncio
async def test_alert_manager_cleanup(alert_manager):
    """Test alert manager cleanup."""
    await alert_manager.initialize()
    await alert_manager.cleanup()
    alert_manager.redis_client.close.assert_called_once()
    alert_manager.http_session.close.assert_called_once()

@pytest.mark.asyncio
async def test_check_metrics(alert_manager):
    """Test metrics checking and alert generation."""
    metrics = {
        "system": {
            "cpu_usage": 90,
            "memory_usage": 90
        },
        "tasks": {
            "success_rate": 85
        },
        "agents": {
            "active_agents": 2
        },
        "redis": {
            "connected": False
        },
        "models": {
            "accuracy": 80
        }
    }
    
    alerts = await alert_manager.check_metrics(metrics)
    assert len(alerts) == 5
    
    # Verify alert types and levels
    alert_types = {alert["type"] for alert in alerts}
    assert alert_types == {"system", "task", "agent", "redis", "model"}
    
    # Verify error alerts
    error_alerts = [alert for alert in alerts if alert["level"] == "error"]
    assert len(error_alerts) == 3  # task, agent, redis

@pytest.mark.asyncio
async def test_send_email_alert(alert_manager):
    """Test email alert sending."""
    alert = {
        "type": "system",
        "level": "error",
        "message": "Test alert",
        "timestamp": datetime.now().isoformat()
    }
    
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        await alert_manager.send_email_alert(alert)
        
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()

@pytest.mark.asyncio
async def test_send_slack_alert(alert_manager):
    """Test Slack alert sending."""
    alert = {
        "type": "system",
        "level": "error",
        "message": "Test alert",
        "timestamp": datetime.now().isoformat()
    }
    
    mock_response = AsyncMock()
    mock_response.status = 200
    alert_manager.http_session.post.return_value.__aenter__.return_value = mock_response
    
    await alert_manager.send_slack_alert(alert)
    
    alert_manager.http_session.post.assert_called_once()
    call_args = alert_manager.http_session.post.call_args
    assert "json" in call_args[1]
    assert "text" in call_args[1]["json"]

@pytest.mark.asyncio
async def test_process_alerts(alert_manager):
    """Test alert processing and storage."""
    alerts = [
        {
            "type": "system",
            "level": "error",
            "message": "Test error",
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "task",
            "level": "warning",
            "message": "Test warning",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    await alert_manager.process_alerts(alerts)
    
    # Verify Redis storage
    assert alert_manager.redis_client.hset.call_count == 2
    
    # Verify notification sending
    assert alert_manager.send_email_alert.call_count == 1  # Only for error
    assert alert_manager.send_slack_alert.call_count == 2  # For both error and warning

@pytest.mark.asyncio
async def test_alert_manager_start_stop(alert_manager):
    """Test alert manager start and stop."""
    # Mock metrics data
    alert_manager.redis_client.hgetall.return_value = {
        "20240101_120000": json.dumps({
            "system": {"cpu_usage": 90},
            "tasks": {"success_rate": 85}
        })
    }
    
    # Start the manager
    start_task = asyncio.create_task(alert_manager.start())
    
    # Wait for one iteration
    await asyncio.sleep(0.1)
    
    # Stop the manager
    await alert_manager.stop()
    await start_task
    
    # Verify metrics were checked
    assert alert_manager.redis_client.hgetall.called
    assert alert_manager.check_metrics.called

@pytest.mark.asyncio
async def test_error_handling(alert_manager):
    """Test error handling in alert manager."""
    # Test Redis connection error
    alert_manager.redis_client.hgetall.side_effect = redis.ConnectionError
    
    with pytest.raises(redis.ConnectionError):
        await alert_manager.start()
    
    # Test SMTP error
    alert = {
        "type": "system",
        "level": "error",
        "message": "Test alert",
        "timestamp": datetime.now().isoformat()
    }
    
    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.side_effect = Exception("SMTP error")
        
        with pytest.raises(Exception) as exc_info:
            await alert_manager.send_email_alert(alert)
        assert "SMTP error" in str(exc_info.value)
    
    # Test Slack API error
    mock_response = AsyncMock()
    mock_response.status = 500
    alert_manager.http_session.post.return_value.__aenter__.return_value = mock_response
    
    with pytest.raises(Exception) as exc_info:
        await alert_manager.send_slack_alert(alert)
    assert "Failed to send Slack message" in str(exc_info.value) 