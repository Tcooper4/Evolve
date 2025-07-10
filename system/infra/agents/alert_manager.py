import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path
from typing import Dict, List, Optional
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlertManager:
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize notification channels
        self.telegram_alerts = None
        self._init_telegram_alerts()
        
        # Strategy-specific alert settings
        self.strategy_alerts = self._load_strategy_alerts()
        
    def _load_config(self) -> Dict:
        """Load alert configuration with environment variable fallbacks."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to default config if file doesn't exist or is invalid
            config = self._get_default_config()
        
        # Override with environment variables
        config = self._override_with_env_vars(config)
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "alerts": {
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "recipient_email": "",
                    "password": "",
                    "use_tls": True
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
    
    def _override_with_env_vars(self, config: Dict) -> Dict:
        """Override configuration with environment variables."""
        # Email configuration from environment variables
        email_config = config.get("alerts", {}).get("email", {})
        
        # SMTP settings
        email_config["smtp_server"] = os.getenv("SMTP_HOST", email_config.get("smtp_server", "smtp.gmail.com"))
        email_config["smtp_port"] = int(os.getenv("SMTP_PORT", email_config.get("smtp_port", 587)))
        email_config["use_tls"] = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        
        # Email credentials
        email_config["sender_email"] = os.getenv("EMAIL_SENDER", email_config.get("sender_email", ""))
        email_config["recipient_email"] = os.getenv("EMAIL_RECIPIENT", email_config.get("recipient_email", ""))
        email_config["password"] = os.getenv("EMAIL_PASSWORD", email_config.get("password", ""))
        
        # Update config with environment overrides
        if "alerts" not in config:
            config["alerts"] = {}
        config["alerts"]["email"] = email_config
        
        return config
    
    def setup_logging(self):
        """Setup logging for alert system."""
        log_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            filename=log_config["file"]
        )
        self.logger = logging.getLogger("AlertManager")
    
    def _validate_email_config(self) -> bool:
        """Validate email configuration."""
        email_config = self.config.get("alerts", {}).get("email", {})
        
        required_fields = ["smtp_server", "smtp_port", "sender_email", "password"]
        missing_fields = [field for field in required_fields if not email_config.get(field)]
        
        if missing_fields:
            self.logger.warning(f"Missing email configuration fields: {missing_fields}")
            return False
        
        return True
    
    def send_alert(self, 
                  subject: str, 
                  message: str, 
                  alert_type: str = "info",
                  recipients: Optional[List[str]] = None) -> bool:
        """Send an alert via email."""
        try:
            # Validate email configuration
            if not self._validate_email_config():
                self.logger.error("Email configuration is incomplete. Cannot send alert.")
                return False
            
            email_config = self.config["alerts"]["email"]
            
            # Create message
            msg = MIMEMultipart()
            msg["Subject"] = f"[{alert_type.upper()}] {subject}"
            msg["From"] = email_config["sender_email"]
            msg["To"] = ", ".join(recipients or [email_config["recipient_email"]])
            
            # Add timestamp and message
            body = f"""
            Time: {datetime.datetime.now().isoformat()}
            Type: {alert_type}
            
            {message}
            """
            msg.attach(MIMEText(body, "plain"))
            
            # Send email with proper error handling
            try:
                with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                    if email_config.get("use_tls", True):
                        server.starttls()
                    
                    # Login with credentials
                    server.login(email_config["sender_email"], email_config["password"])
                    server.send_message(msg)
                
                self.logger.info(f"Alert sent successfully: {subject}")
                return True
                
            except smtplib.SMTPAuthenticationError:
                self.logger.error("SMTP authentication failed. Check email credentials.")
                return False
            except smtplib.SMTPConnectError:
                self.logger.error(f"Failed to connect to SMTP server: {email_config['smtp_server']}:{email_config['smtp_port']}")
                return False
            except smtplib.SMTPException as e:
                self.logger.error(f"SMTP error: {str(e)}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {str(e)}")
            return False
    
    def check_model_performance(self, 
                              model_name: str, 
                              metrics: Dict[str, float]) -> bool:
        """Check if model performance meets thresholds."""
        thresholds = self.config["alerts"]["thresholds"]
        
        if metrics.get("accuracy", 0) < thresholds["model_performance"]:
            self.send_alert(
                subject=f"Model Performance Alert: {model_name}",
                message=f"Model performance below threshold:\n{json.dumps(metrics, indent=2)}",
                alert_type="warning"
            )
            return False
        return True
    
    def check_prediction_confidence(self, 
                                  model_name: str, 
                                  confidence: float,
                                  prediction: float) -> bool:
        """Check if prediction confidence meets threshold."""
        thresholds = self.config["alerts"]["thresholds"]
        
        if confidence < thresholds["prediction_confidence"]:
            self.send_alert(
                subject=f"Low Confidence Alert: {model_name}",
                message=f"Prediction confidence ({confidence:.2f}) below threshold for prediction: {prediction:.2f}",
                alert_type="warning"
            )
            return False
        return True
    
    def send_system_alert(self, 
                         component: str, 
                         message: str, 
                         alert_type: str = "error") -> None:
        """Send a system-level alert."""
        self.send_alert(
            subject=f"System Alert: {component}",
            message=message,
            alert_type=alert_type
        )
    
    def send_backup_alert(self, 
                         backup_path: str, 
                         success: bool, 
                         message: str) -> None:
        """Send a backup-related alert."""
        alert_type = "info" if success else "error"
        self.send_alert(
            subject=f"Backup {'Success' if success else 'Failure'}",
            message=f"Backup path: {backup_path}\n{message}",
            alert_type=alert_type
        )

    def _init_telegram_alerts(self):
        """Initialize Telegram alerts if configured."""
        try:
            from .telegram_alerts import TelegramAlerts
            self.telegram_alerts = TelegramAlerts()
            if self.telegram_alerts.enabled:
                self.logger.info("Telegram alerts initialized successfully")
            else:
                self.logger.info("Telegram alerts disabled or not configured")
        except ImportError:
            self.logger.warning("Telegram alerts module not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram alerts: {e}")
    
    def _load_strategy_alerts(self) -> Dict[str, Dict[str, bool]]:
        """Load strategy-specific alert settings."""
        try:
            settings_path = Path("config/settings.json")
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    return settings.get('strategy_alerts', {})
            else:
                # Create default settings
                default_settings = {
                    'strategy_alerts': {
                        'default': {
                            'email_alerts': True,
                            'telegram_alerts': True,
                            'slack_alerts': False
                        }
                    }
                }
                with open(settings_path, 'w') as f:
                    json.dump(default_settings, f, indent=2)
                return default_settings['strategy_alerts']
        except Exception as e:
            self.logger.error(f"Failed to load strategy alerts: {e}")
            return {'default': {'email_alerts': True, 'telegram_alerts': True, 'slack_alerts': False}}
    
    def _save_strategy_alerts(self):
        """Save strategy-specific alert settings."""
        try:
            settings_path = Path("config/settings.json")
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {}
            
            settings['strategy_alerts'] = self.strategy_alerts
            
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save strategy alerts: {e}")
    
    def update_strategy_alerts(self, strategy_name: str, alert_settings: Dict[str, bool]):
        """Update alert settings for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            alert_settings: Dictionary with alert type settings
        """
        self.strategy_alerts[strategy_name] = alert_settings
        self._save_strategy_alerts()
        self.logger.info(f"Updated alert settings for strategy: {strategy_name}")
    
    def is_alert_enabled(self, strategy_name: str, alert_type: str) -> bool:
        """Check if a specific alert type is enabled for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            alert_type: Type of alert (email_alerts, telegram_alerts, slack_alerts)
            
        Returns:
            True if alert is enabled, False otherwise
        """
        # Get strategy settings, fallback to default
        strategy_settings = self.strategy_alerts.get(strategy_name, self.strategy_alerts.get('default', {}))
        return strategy_settings.get(alert_type, True)
    
    def send_multi_channel_alert(self, 
                                subject: str, 
                                message: str, 
                                alert_type: str = "info",
                                strategy_name: str = "default",
                                recipients: Optional[List[str]] = None) -> Dict[str, bool]:
        """Send alert through multiple channels based on strategy settings.
        
        Args:
            subject: Alert subject
            message: Alert message
            alert_type: Type of alert
            strategy_name: Name of the strategy
            recipients: Email recipients (optional)
            
        Returns:
            Dictionary with success status for each channel
        """
        results = {}
        
        # Email alerts
        if self.is_alert_enabled(strategy_name, 'email_alerts'):
            results['email'] = self.send_alert(subject, message, alert_type, recipients)
        else:
            results['email'] = False
            self.logger.debug(f"Email alerts disabled for strategy: {strategy_name}")
        
        # Telegram alerts
        if (self.is_alert_enabled(strategy_name, 'telegram_alerts') and 
            self.telegram_alerts and self.telegram_alerts.enabled):
            results['telegram'] = self.telegram_alerts.send_alert(subject, message, alert_type)
        else:
            results['telegram'] = False
            self.logger.debug(f"Telegram alerts disabled for strategy: {strategy_name}")
        
        # Slack alerts (placeholder for future implementation)
        if self.is_alert_enabled(strategy_name, 'slack_alerts'):
            results['slack'] = False  # Not implemented yet
            self.logger.debug(f"Slack alerts not implemented yet")
        else:
            results['slack'] = False
        
        return results

if __name__ == "__main__":
    # Example usage
    alert_manager = AlertManager()
    
    # Send a test alert
    alert_manager.send_alert(
        subject="Test Alert",
        message="This is a test alert message.",
        alert_type="info"
    ) 