import pytest
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

from automation.notifications.notification_service import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)

@pytest.fixture
def config_path():
    """Get the path to the notification configuration file."""
    return Path(__file__).parent.parent / "config" / "notification_config.yaml"

@pytest.fixture
def config(config_path):
    """Load the notification configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def test_email_config(config):
    """Test the email configuration."""
    email_config = config["email"]
    assert "smtp_host" in email_config
    assert "smtp_port" in email_config
    assert "username" in email_config
    assert "password" in email_config
    assert "from_email" in email_config
    assert "use_tls" in email_config
    
    # Verify environment variable placeholders
    assert email_config["username"].startswith("${")
    assert email_config["password"].startswith("${")
    assert email_config["from_email"].startswith("${")

def test_slack_config(config):
    """Test the Slack configuration."""
    slack_config = config["slack"]
    assert "webhook_url" in slack_config
    assert "default_channel" in slack_config
    
    # Verify environment variable placeholder
    assert slack_config["webhook_url"].startswith("${")

def test_webhook_config(config):
    """Test the webhook configuration."""
    webhook_config = config["webhook"]
    assert "timeout" in webhook_config
    assert "retry_attempts" in webhook_config
    assert "retry_delay" in webhook_config
    
    # Verify values
    assert isinstance(webhook_config["timeout"], int)
    assert isinstance(webhook_config["retry_attempts"], int)
    assert isinstance(webhook_config["retry_delay"], int)

def test_notification_settings(config):
    """Test the notification settings."""
    settings = config["settings"]
    assert "retention_days" in settings
    assert "cleanup_interval" in settings
    assert "max_retries" in settings
    assert "retry_delay" in settings
    
    # Verify values
    assert isinstance(settings["retention_days"], int)
    assert isinstance(settings["cleanup_interval"], int)
    assert isinstance(settings["max_retries"], int)
    assert isinstance(settings["retry_delay"], int)

def test_priority_settings(config):
    """Test the priority settings."""
    priorities = config["priorities"]
    assert "low" in priorities
    assert "medium" in priorities
    assert "high" in priorities
    assert "critical" in priorities
    
    # Verify each priority has required fields
    for priority in priorities.values():
        assert "retry_attempts" in priority
        assert "retry_delay" in priority
        assert isinstance(priority["retry_attempts"], int)
        assert isinstance(priority["retry_delay"], int)

def test_channel_settings(config):
    """Test the channel settings."""
    channels = config["channels"]
    assert "email" in channels
    assert "slack" in channels
    assert "webhook" in channels
    assert "sms" in channels
    assert "teams" in channels
    assert "discord" in channels
    
    # Verify each channel has required fields
    for channel in channels.values():
        assert "enabled" in channel
        assert "max_retries" in channel
        assert "timeout" in channel
        assert isinstance(channel["enabled"], bool)
        assert isinstance(channel["max_retries"], int)
        assert isinstance(channel["timeout"], int)

def test_template_settings(config):
    """Test the template settings."""
    templates = config["templates"]
    assert "email" in templates
    assert "slack" in templates
    assert "webhook" in templates
    
    # Verify email template settings
    email_templates = templates["email"]
    assert "subject_template" in email_templates
    assert "body_template" in email_templates
    
    # Verify Slack template settings
    slack_templates = templates["slack"]
    assert "message_template" in slack_templates
    
    # Verify webhook template settings
    webhook_templates = templates["webhook"]
    assert "payload_template" in webhook_templates

def test_monitoring_settings(config):
    """Test the monitoring settings."""
    monitoring = config["monitoring"]
    assert "enabled" in monitoring
    assert "metrics" in monitoring
    assert "alerts" in monitoring
    
    # Verify metrics configuration
    metrics = monitoring["metrics"]
    assert len(metrics) > 0
    for metric in metrics:
        assert "name" in metric
        assert "type" in metric
        assert "labels" in metric
    
    # Verify alerts configuration
    alerts = monitoring["alerts"]
    assert len(alerts) > 0
    for alert in alerts:
        assert "name" in alert
        assert "condition" in alert
        assert "severity" in alert

def test_config_file_exists(config_path):
    """Test that the configuration file exists."""
    assert config_path.exists()
    assert config_path.is_file()

def test_config_file_permissions(config_path):
    """Test the configuration file permissions."""
    # Verify file is readable
    assert os.access(config_path, os.R_OK)
    
    # Verify file is not world-writable
    assert not os.access(config_path, os.W_OK) or oct(config_path.stat().st_mode)[-3:] != "666"

@pytest.fixture
def config_test_dir(tmp_path):
    """Create a temporary directory for configuration testing."""
    return tmp_path

@pytest.fixture
def valid_config(config_test_dir):
    """Create a valid notification configuration."""
    config = {
        "email": {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "test-password",
            "from_address": "notifications@example.com",
            "use_tls": True
        },
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            "default_channel": "#notifications"
        },
        "webhook": {
            "endpoints": [
                {
                    "url": "https://api.example.com/webhook",
                    "method": "POST",
                    "headers": {
                        "Authorization": "Bearer test-token"
                    }
                }
            ]
        },
        "retry": {
            "max_retries": 3,
            "retry_delay": 60,
            "backoff_factor": 2
        },
        "cleanup": {
            "retention_days": 30,
            "cleanup_interval": 3600
        },
        "rate_limiting": {
            "max_notifications_per_minute": 100,
            "max_notifications_per_hour": 1000
        },
        "security": {
            "allowed_origins": ["https://example.com"],
            "allowed_methods": ["GET", "POST"],
            "allowed_headers": ["Content-Type", "Authorization"]
        }
    }
    
    config_path = config_test_dir / "notification_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    return config_path

@pytest.fixture
def invalid_config(config_test_dir):
    """Create an invalid notification configuration."""
    config = {
        "email": {
            "smtp_server": "",  # Invalid empty server
            "smtp_port": -1,  # Invalid port
            "username": "invalid-email",  # Invalid email
            "password": "",  # Invalid empty password
            "from_address": "invalid-email",  # Invalid email
            "use_tls": "invalid"  # Invalid boolean
        },
        "slack": {
            "webhook_url": "invalid-url",  # Invalid URL
            "default_channel": ""  # Invalid empty channel
        },
        "webhook": {
            "endpoints": [
                {
                    "url": "invalid-url",  # Invalid URL
                    "method": "INVALID",  # Invalid method
                    "headers": {}  # Invalid empty headers
                }
            ]
        },
        "retry": {
            "max_retries": -1,  # Invalid negative retries
            "retry_delay": -1,  # Invalid negative delay
            "backoff_factor": 0  # Invalid zero factor
        },
        "cleanup": {
            "retention_days": -1,  # Invalid negative days
            "cleanup_interval": -1  # Invalid negative interval
        },
        "rate_limiting": {
            "max_notifications_per_minute": -1,  # Invalid negative limit
            "max_notifications_per_hour": -1  # Invalid negative limit
        },
        "security": {
            "allowed_origins": [],  # Invalid empty origins
            "allowed_methods": [],  # Invalid empty methods
            "allowed_headers": []  # Invalid empty headers
        }
    }
    
    config_path = config_test_dir / "invalid_notification_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    return config_path

def test_valid_config_loading(valid_config):
    """Test loading of valid configuration."""
    service = NotificationService()
    config = service.load_config(valid_config)
    
    # Verify email configuration
    assert config["email"]["smtp_server"] == "smtp.example.com"
    assert config["email"]["smtp_port"] == 587
    assert config["email"]["username"] == "test@example.com"
    assert config["email"]["password"] == "test-password"
    assert config["email"]["from_address"] == "notifications@example.com"
    assert config["email"]["use_tls"] is True
    
    # Verify Slack configuration
    assert config["slack"]["webhook_url"] == "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
    assert config["slack"]["default_channel"] == "#notifications"
    
    # Verify webhook configuration
    assert len(config["webhook"]["endpoints"]) == 1
    assert config["webhook"]["endpoints"][0]["url"] == "https://api.example.com/webhook"
    assert config["webhook"]["endpoints"][0]["method"] == "POST"
    assert config["webhook"]["endpoints"][0]["headers"]["Authorization"] == "Bearer test-token"
    
    # Verify retry configuration
    assert config["retry"]["max_retries"] == 3
    assert config["retry"]["retry_delay"] == 60
    assert config["retry"]["backoff_factor"] == 2
    
    # Verify cleanup configuration
    assert config["cleanup"]["retention_days"] == 30
    assert config["cleanup"]["cleanup_interval"] == 3600
    
    # Verify rate limiting configuration
    assert config["rate_limiting"]["max_notifications_per_minute"] == 100
    assert config["rate_limiting"]["max_notifications_per_hour"] == 1000
    
    # Verify security configuration
    assert config["security"]["allowed_origins"] == ["https://example.com"]
    assert config["security"]["allowed_methods"] == ["GET", "POST"]
    assert config["security"]["allowed_headers"] == ["Content-Type", "Authorization"]

def test_invalid_config_loading(invalid_config):
    """Test loading of invalid configuration."""
    service = NotificationService()
    
    with pytest.raises(ValueError) as exc_info:
        service.load_config(invalid_config)
    
    # Verify error message contains validation failures
    error_message = str(exc_info.value)
    assert "email" in error_message
    assert "slack" in error_message
    assert "webhook" in error_message
    assert "retry" in error_message
    assert "cleanup" in error_message
    assert "rate_limiting" in error_message
    assert "security" in error_message

def test_missing_config_file(config_test_dir):
    """Test handling of missing configuration file."""
    service = NotificationService()
    missing_config = config_test_dir / "missing_config.json"
    
    with pytest.raises(FileNotFoundError):
        service.load_config(missing_config)

def test_invalid_json_config(config_test_dir):
    """Test handling of invalid JSON configuration."""
    service = NotificationService()
    invalid_json_config = config_test_dir / "invalid_json_config.json"
    
    # Create file with invalid JSON
    with open(invalid_json_config, "w") as f:
        f.write("invalid json content")
    
    with pytest.raises(json.JSONDecodeError):
        service.load_config(invalid_json_config)

def test_config_validation(valid_config):
    """Test configuration validation."""
    service = NotificationService()
    config = service.load_config(valid_config)
    
    # Verify email validation
    assert service.validate_email_config(config["email"]) is True
    
    # Verify Slack validation
    assert service.validate_slack_config(config["slack"]) is True
    
    # Verify webhook validation
    assert service.validate_webhook_config(config["webhook"]) is True
    
    # Verify retry validation
    assert service.validate_retry_config(config["retry"]) is True
    
    # Verify cleanup validation
    assert service.validate_cleanup_config(config["cleanup"]) is True
    
    # Verify rate limiting validation
    assert service.validate_rate_limiting_config(config["rate_limiting"]) is True
    
    # Verify security validation
    assert service.validate_security_config(config["security"]) is True

def test_config_defaults(valid_config):
    """Test configuration default values."""
    service = NotificationService()
    config = service.load_config(valid_config)
    
    # Verify default values are set
    assert "default_priority" in config
    assert "default_channel" in config
    assert "default_type" in config
    assert "default_retention_days" in config
    assert "default_cleanup_interval" in config
    assert "default_max_retries" in config
    assert "default_retry_delay" in config
    assert "default_backoff_factor" in config

def test_config_environment_variables(valid_config):
    """Test configuration loading from environment variables."""
    service = NotificationService()
    
    # Set environment variables
    os.environ["NOTIFICATION_SMTP_SERVER"] = "env-smtp.example.com"
    os.environ["NOTIFICATION_SMTP_PORT"] = "587"
    os.environ["NOTIFICATION_SMTP_USERNAME"] = "env-test@example.com"
    os.environ["NOTIFICATION_SMTP_PASSWORD"] = "env-test-password"
    os.environ["NOTIFICATION_SMTP_FROM"] = "env-notifications@example.com"
    os.environ["NOTIFICATION_SMTP_USE_TLS"] = "true"
    
    config = service.load_config(valid_config)
    
    # Verify environment variables override config
    assert config["email"]["smtp_server"] == "env-smtp.example.com"
    assert config["email"]["smtp_port"] == 587
    assert config["email"]["username"] == "env-test@example.com"
    assert config["email"]["password"] == "env-test-password"
    assert config["email"]["from_address"] == "env-notifications@example.com"
    assert config["email"]["use_tls"] is True
    
    # Clean up environment variables
    del os.environ["NOTIFICATION_SMTP_SERVER"]
    del os.environ["NOTIFICATION_SMTP_PORT"]
    del os.environ["NOTIFICATION_SMTP_USERNAME"]
    del os.environ["NOTIFICATION_SMTP_PASSWORD"]
    del os.environ["NOTIFICATION_SMTP_FROM"]
    del os.environ["NOTIFICATION_SMTP_USE_TLS"]

def test_config_secrets(valid_config):
    """Test configuration secrets handling."""
    service = NotificationService()
    config = service.load_config(valid_config)
    
    # Verify secrets are properly handled
    assert "password" in config["email"]
    assert config["email"]["password"] == "test-password"
    
    # Verify secrets are not logged
    with patch("logging.Logger.info") as mock_logger:
        service.log_config(config)
        log_message = mock_logger.call_args[0][0]
        assert "test-password" not in log_message
        assert "password" in log_message
        assert "****" in log_message

def test_config_reloading(valid_config):
    """Test configuration reloading."""
    service = NotificationService()
    config = service.load_config(valid_config)
    
    # Modify configuration file
    with open(valid_config, "r") as f:
        config_data = json.load(f)
    
    config_data["email"]["smtp_server"] = "new-smtp.example.com"
    
    with open(valid_config, "w") as f:
        json.dump(config_data, f)
    
    # Reload configuration
    new_config = service.load_config(valid_config)
    
    # Verify configuration is reloaded
    assert new_config["email"]["smtp_server"] == "new-smtp.example.com"
    assert new_config["email"]["smtp_server"] != config["email"]["smtp_server"] 