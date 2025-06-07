# Alert System

The alert system provides comprehensive monitoring and notification capabilities for the trading system. It handles various types of alerts including model performance monitoring, prediction confidence checks, system-level alerts, and backup status notifications.

## Features

- Email-based alert system with configurable settings
- Logging integration for alert tracking
- Multiple alert types:
  - Model performance monitoring
  - Prediction confidence checks
  - System-level alerts
  - Backup status notifications
- Customizable alert thresholds
- Support for multiple recipients

## Configuration

The alert system is configured through the main configuration file (`config/config.json`). The following settings are available:

```json
{
    "alerts": {
        "email": {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "sender_email": "alerts@example.com",
            "recipient_email": "admin@example.com",
            "password": "your_password"
        },
        "thresholds": {
            "model_performance": 0.8,
            "prediction_confidence": 0.7
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "alerts.log"
    }
}
```

## Usage

### Basic Alert

```python
from alerts.alert_manager import AlertManager

alert_manager = AlertManager()

# Send a basic alert
alert_manager.send_alert(
    subject="Test Alert",
    message="This is a test alert message.",
    alert_type="info"
)
```

### Model Performance Monitoring

```python
# Check model performance
metrics = {"accuracy": 0.75}
alert_manager.check_model_performance("lstm_model", metrics)
```

### Prediction Confidence Check

```python
# Check prediction confidence
alert_manager.check_prediction_confidence(
    model_name="lstm_model",
    confidence=0.8,
    prediction=0.5
)
```

### System Alert

```python
# Send a system-level alert
alert_manager.send_system_alert(
    component="DataProcessor",
    message="Data processing failed",
    alert_type="error"
)
```

### Backup Status

```python
# Send backup status alert
alert_manager.send_backup_alert(
    backup_path="/backup/path",
    success=True,
    message="Backup completed successfully"
)
```

## Testing

The alert system includes a comprehensive test suite. To run the tests:

```bash
pytest alerts/test_alert_manager.py -v
```

## Alert Types

- `info`: General information alerts
- `warning`: Warning alerts for potential issues
- `error`: Error alerts for critical issues

## Best Practices

1. Always include relevant context in alert messages
2. Use appropriate alert types based on severity
3. Configure thresholds based on system requirements
4. Monitor alert logs for system health
5. Regularly review and update alert configurations

## Security Considerations

1. Store email credentials securely
2. Use TLS for email communication
3. Implement rate limiting for alerts
4. Monitor alert patterns for potential abuse
5. Regularly rotate credentials

## Maintenance

1. Regularly check alert logs
2. Update alert thresholds as needed
3. Monitor alert delivery success rates
4. Review and update recipient lists
5. Test alert system periodically 