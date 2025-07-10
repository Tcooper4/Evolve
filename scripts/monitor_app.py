#!/usr/bin/env python3
"""
Application monitoring script.
Provides commands for monitoring the running application and reporting status.

This script supports:
- Monitoring application status
- Reporting application health
- Exporting monitoring data

Usage:
    python monitor_app.py <command> [options]

Commands:
    status      Show application status
    health      Report application health
    export      Export monitoring data

Examples:
    # Show application status
    python monitor_app.py status

    # Report application health
    python monitor_app.py health

    # Export monitoring data
    python monitor_app.py export --output monitoring.json
"""

import os
import sys
import time
import psutil
import logging
import logging.config
import yaml
import requests
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    status_code: int
    timestamp: str

@dataclass
class Alert:
    """Container for alert information."""
    metric: str
    value: float
    threshold: float
    timestamp: str
    severity: str = "warning"

class ApplicationMonitor:
    """Application monitoring system with comprehensive health checks and alerting."""
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the application monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("performance")
        self.metrics: Dict[str, float] = {}
        self.alert_thresholds = self.config.get("monitoring", {}).get("metrics", [])
        self.alert_channels = self.config.get("monitoring", {}).get("alert", {})
        self.monitoring_history: List[MonitoringMetrics] = []
        
        logger.info("ApplicationMonitor initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load application configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            # Return default configuration
            return {
                "monitoring": {
                    "interval": 30,
                    "metrics": [
                        {"name": "cpu_usage", "threshold": 80.0},
                        {"name": "memory_usage", "threshold": 1000.0},
                        {"name": "response_time", "threshold": 5.0}
                    ],
                    "alert": {
                        "email": None,
                        "slack": None
                    }
                },
                "server": {
                    "host": "localhost",
                    "port": 8501
                }
            }
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def setup_logging(self) -> None:
        """Initialize logging configuration."""
        try:
            log_config_path = Path("config/logging_config.yaml")
            if log_config_path.exists():
                with open(log_config_path, 'r') as f:
                    log_config = yaml.safe_load(f)
                logging.config.dictConfig(log_config)
                logger.info("Logging configuration loaded from file")
            else:
                # Setup basic logging if config file doesn't exist
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('logs/monitor.log')
                    ]
                )
                logger.info("Basic logging configuration initialized")
        except Exception as e:
            logger.error(f"Error setting up logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)

    def collect_metrics(self) -> MonitoringMetrics:
        """
        Collect system and application metrics.
        
        Returns:
            MonitoringMetrics object with current metrics
        """
        try:
            process = psutil.Process(os.getpid())
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            disk_usage = psutil.disk_usage("/").percent
            
            # Application metrics
            response_time = float("inf")
            status_code = 500
            
            try:
                server_config = self.config.get("server", {})
                host = server_config.get("host", "localhost")
                port = server_config.get("port", 8501)
                
                response = requests.get(
                    f"http://{host}:{port}/health",
                    timeout=10
                )
                response_time = response.elapsed.total_seconds()
                status_code = response.status_code
            except requests.RequestException as e:
                logger.warning(f"Error checking application health: {e}")
            
            metrics = MonitoringMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                response_time=response_time,
                status_code=status_code,
                timestamp=datetime.now().isoformat()
            )
            
            # Store in metrics dict for compatibility
            self.metrics = asdict(metrics)
            self.monitoring_history.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.monitoring_history) > 1000:
                self.monitoring_history = self.monitoring_history[-1000:]
            
            logger.debug(f"Collected metrics: CPU={cpu_usage}%, Memory={memory_usage:.2f}MB")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return MonitoringMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                response_time=float("inf"),
                status_code=500,
                timestamp=datetime.now().isoformat()
            )

    def check_thresholds(self) -> List[Alert]:
        """
        Check if any metrics exceed their thresholds.
        
        Returns:
            List of Alert objects for threshold violations
        """
        try:
            alerts = []
            
            for metric_config in self.alert_thresholds:
                name = metric_config.get("name")
                threshold = metric_config.get("threshold")
                severity = metric_config.get("severity", "warning")
                
                if name and threshold is not None:
                    value = self.metrics.get(name)
                    
                    if value is not None and value > threshold:
                        alerts.append(Alert(
                            metric=name,
                            value=value,
                            threshold=threshold,
                            severity=severity,
                            timestamp=datetime.now().isoformat()
                        ))
            
            if alerts:
                logger.warning(f"Found {len(alerts)} threshold violations")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
            return []

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through configured channels.
        
        Args:
            alert: Alert object to send
            
        Returns:
            True if alert was sent successfully
        """
        try:
            message = (
                f"ALERT: {alert.metric} exceeded threshold\n"
                f"Value: {alert.value}\n"
                f"Threshold: {alert.threshold}\n"
                f"Severity: {alert.severity}\n"
                f"Time: {alert.timestamp}"
            )
            
            # Log alert
            self.logger.warning(message)
            
            # Send email alert
            if self.alert_channels.get("email"):
                self._send_email_alert(message)
            
            # Send Slack alert
            if self.alert_channels.get("slack"):
                self._send_slack_alert(message)
            
            logger.info(f"Alert sent for {alert.metric}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False

    def _send_email_alert(self, message: str) -> None:
        """
        Send email alert.
        
        Args:
            message: Alert message to send
        """
        # Implement actual email sending logic
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Email configuration
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            sender_email = os.environ.get('SENDER_EMAIL', '')
            sender_password = os.environ.get('SENDER_PASSWORD', '')
            
            if not sender_email or not sender_password:
                logger.warning("Email credentials not configured, skipping email notification")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _send_slack_alert(self, message: str) -> None:
        """
        Send Slack alert.
        
        Args:
            message: Alert message to send
        """
        try:
            slack_webhook = self.alert_channels.get("slack")
            if slack_webhook:
                response = requests.post(
                    slack_webhook,
                    json={"text": message},
                    timeout=10
                )
                response.raise_for_status()
                logger.info("Slack alert sent successfully")
        except requests.RequestException as e:
            logger.error(f"Error sending Slack alert: {e}")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def log_metrics(self) -> None:
        """Log collected metrics."""
        try:
            metrics = self.metrics
            self.logger.info(
                f"Metrics - CPU: {metrics.get('cpu_usage', 0):.1f}%, "
                f"Memory: {metrics.get('memory_usage', 0):.2f}MB, "
                f"Response Time: {metrics.get('response_time', 0):.3f}s"
            )
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current application status.
        
        Returns:
            Dictionary with current status information
        """
        try:
            metrics = self.collect_metrics()
            
            # Check thresholds
            alerts = self.check_thresholds()
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(metrics),
                "alerts": [asdict(alert) for alert in alerts],
                "health": "healthy" if not alerts else "unhealthy",
                "uptime": self._get_uptime()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "health": "error"
            }

    def _get_uptime(self) -> str:
        """Get application uptime."""
        try:
            process = psutil.Process(os.getpid())
            uptime_seconds = time.time() - process.create_time()
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        except Exception:
            return "unknown"

    def export_monitoring_data(self, output_path: str = "monitoring_data.json") -> bool:
        """
        Export monitoring data to file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if export was successful
        """
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "monitoring_history": [asdict(metric) for metric in self.monitoring_history],
                "current_status": self.get_status(),
                "config": self.config
            }
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Monitoring data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            return False

    def execute_monitoring_loop(self, duration: Optional[int] = None) -> None:
        """
        Execute the monitoring loop.
        
        Args:
            duration: Duration to run monitoring in seconds (None for infinite)
        """
        try:
            self.logger.info("Starting application monitoring")
            
            start_time = time.time()
            loop_count = 0
            total_alerts = 0
            
            while True:
                try:
                    # Check if we should stop
                    if duration and (time.time() - start_time) > duration:
                        break
                    
                    # Collect metrics
                    metrics = self.collect_metrics()
                    
                    # Log metrics
                    self.log_metrics()
                    
                    # Check thresholds and send alerts
                    alerts = self.check_thresholds()
                    
                    for alert in alerts:
                        if self.send_alert(alert):
                            total_alerts += 1
                    
                    loop_count += 1
                    
                    # Sleep for the configured interval
                    interval = self.config.get("monitoring", {}).get("interval", 30)
                    time.sleep(interval)
                
                except KeyboardInterrupt:
                    self.logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Wait before retrying
            
            self.logger.info(f"Monitoring completed: {loop_count} loops, {total_alerts} alerts")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")

def main() -> None:
    """Main function for the application monitor."""
    parser = argparse.ArgumentParser(description="Application monitoring script")
    parser.add_argument("command", choices=["status", "health", "export", "monitor"],
                       help="Command to execute")
    parser.add_argument("--output", "-o", default="monitoring_data.json",
                       help="Output file for export command")
    parser.add_argument("--duration", "-d", type=int,
                       help="Duration to run monitoring in seconds")
    parser.add_argument("--config", "-c", default="config/app_config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        monitor = ApplicationMonitor(args.config)
        
        if args.command == "status":
            status = monitor.get_status()
            print(json.dumps(status, indent=2))
            
        elif args.command == "health":
            status = monitor.get_status()
            health = status.get("health", "unknown")
            print(f"Application health: {health}")
            
        elif args.command == "export":
            success = monitor.export_monitoring_data(args.output)
            if success:
                print(f"Monitoring data exported to {args.output}")
            else:
                print("Failed to export monitoring data")
                sys.exit(1)
                
        elif args.command == "monitor":
            monitor.execute_monitoring_loop(args.duration)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 