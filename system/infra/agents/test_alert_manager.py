import json
from unittest.mock import MagicMock, patch

import pytest

from system.infra.agents.alert_manager import AlertManager


@pytest.fixture
def test_config():
    return {
        "alerts": {
            "email": {
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "sender_email": "test@example.com",
                "recipient_email": "recipient@example.com",
                "password": "test_password",
            },
            "thresholds": {"model_performance": 0.8, "prediction_confidence": 0.7},
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "test_alerts.log",
        },
    }


@pytest.fixture
def alert_manager(test_config, tmp_path):
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f)
    return AlertManager(str(config_path))


def test_alert_manager_initialization(alert_manager, test_config):
    """Test alert manager initialization and config loading."""
    assert alert_manager.config == test_config
    assert alert_manager.logger is not None


@patch("smtplib.SMTP")
def test_send_alert(mock_smtp, alert_manager):
    """Test sending a basic alert."""
    mock_smtp_instance = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

    success = alert_manager.send_alert(subject="Test Alert", message="Test message", alert_type="info")

    assert success is True
    mock_smtp_instance.starttls.assert_called_once()
    mock_smtp_instance.login.assert_called_once()
    mock_smtp_instance.send_message.assert_called_once()


def test_check_model_performance(alert_manager):
    """Test model performance threshold checking."""
    # Test below threshold
    metrics = {"accuracy": 0.75}
    with patch.object(alert_manager, "send_alert") as mock_send:
        result = alert_manager.check_model_performance("test_model", metrics)
        assert result is False
        mock_send.assert_called_once()

    # Test above threshold
    metrics = {"accuracy": 0.85}
    with patch.object(alert_manager, "send_alert") as mock_send:
        result = alert_manager.check_model_performance("test_model", metrics)
        assert result is True
        mock_send.assert_not_called()


def test_check_prediction_confidence(alert_manager):
    """Test prediction confidence threshold checking."""
    # Test below threshold
    with patch.object(alert_manager, "send_alert") as mock_send:
        result = alert_manager.check_prediction_confidence("test_model", 0.6, 0.5)
        assert result is False
        mock_send.assert_called_once()

    # Test above threshold
    with patch.object(alert_manager, "send_alert") as mock_send:
        result = alert_manager.check_prediction_confidence("test_model", 0.8, 0.5)
        assert result is True
        mock_send.assert_not_called()


def test_send_system_alert(alert_manager):
    """Test sending system-level alerts."""
    with patch.object(alert_manager, "send_alert") as mock_send:
        alert_manager.send_system_alert(component="TestComponent", message="Test system message", alert_type="error")
        mock_send.assert_called_once_with(
            subject="System Alert: TestComponent", message="Test system message", alert_type="error"
        )


def test_send_backup_alert(alert_manager):
    """Test sending backup-related alerts."""
    # Test successful backup
    with patch.object(alert_manager, "send_alert") as mock_send:
        alert_manager.send_backup_alert(backup_path="/test/path", success=True, message="Backup completed")
        mock_send.assert_called_once_with(
            subject="Backup Success", message="Backup path: /test/path\nBackup completed", alert_type="info"
        )

    # Test failed backup
    with patch.object(alert_manager, "send_alert") as mock_send:
        alert_manager.send_backup_alert(backup_path="/test/path", success=False, message="Backup failed")
        mock_send.assert_called_once_with(
            subject="Backup Failure", message="Backup path: /test/path\nBackup failed", alert_type="error"
        )


def test_alert_with_custom_recipients(alert_manager):
    """Test sending alert to custom recipients."""
    custom_recipients = ["user1@example.com", "user2@example.com"]
    with patch.object(alert_manager, "send_alert") as mock_send:
        alert_manager.send_alert(subject="Test Alert", message="Test message", recipients=custom_recipients)
        mock_send.assert_called_once()
        # Verify recipients in the email message
        msg = mock_send.call_args[1]["msg"]
        assert msg["To"] == ", ".join(custom_recipients)
