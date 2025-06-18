import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path
from typing import Dict, List, Optional
import datetime

class AlertManager:
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
    def _load_config(self) -> Dict:
        """Load alert configuration."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def setup_logging(self):
        """Setup logging for alert system."""
        log_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            filename=log_config["file"]
        )
        self.logger = logging.getLogger("AlertManager")
    
    def send_alert(self, 
                  subject: str, 
                  message: str, 
                  alert_type: str = "info",
                  recipients: Optional[List[str]] = None) -> bool:
        """Send an alert via email."""
        try:
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
            
            # Send email
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["sender_email"], email_config.get("password", ""))
                server.send_message(msg)
            
            self.logger.info(f"Alert sent: {subject}")
            return True
            
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

if __name__ == "__main__":
    # Example usage
    alert_manager = AlertManager()
    
    # Send a test alert
    alert_manager.send_alert(
        subject="Test Alert",
        message="This is a test alert message.",
        alert_type="info"
    ) 