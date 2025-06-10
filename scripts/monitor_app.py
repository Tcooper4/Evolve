#!/usr/bin/env python3
"""
Application monitoring script.
Monitors system resources, application performance, and sends alerts if needed.
"""

import os
import sys
import time
import psutil
import logging
import logging.config
import yaml
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ApplicationMonitor:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the application monitor."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("performance")
        self.metrics = {}
        self.alert_thresholds = self.config["monitoring"]["metrics"]
        self.alert_channels = self.config["monitoring"]["alert"]

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    def collect_metrics(self):
        """Collect system and application metrics."""
        process = psutil.Process(os.getpid())
        
        # System metrics
        self.metrics["cpu_usage"] = psutil.cpu_percent()
        self.metrics["memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
        self.metrics["disk_usage"] = psutil.disk_usage("/").percent
        
        # Application metrics
        try:
            response = requests.get(f"http://{self.config['server']['host']}:{self.config['server']['port']}/health")
            self.metrics["response_time"] = response.elapsed.total_seconds()
            self.metrics["status_code"] = response.status_code
        except requests.RequestException as e:
            self.metrics["response_time"] = float("inf")
            self.metrics["status_code"] = 500
            self.logger.error(f"Error checking application health: {e}")

    def check_thresholds(self):
        """Check if any metrics exceed their thresholds."""
        alerts = []
        
        for metric in self.alert_thresholds:
            name = metric["name"]
            threshold = metric["threshold"]
            value = self.metrics.get(name)
            
            if value is not None and value > threshold:
                alerts.append({
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts

    def send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels."""
        message = (
            f"ALERT: {alert['metric']} exceeded threshold\n"
            f"Value: {alert['value']}\n"
            f"Threshold: {alert['threshold']}\n"
            f"Time: {alert['timestamp']}"
        )
        
        # Log alert
        self.logger.warning(message)
        
        # Send email alert
        if "email" in self.alert_channels:
            self._send_email_alert(message)
        
        # Send Slack alert
        if "slack" in self.alert_channels:
            self._send_slack_alert(message)

    def _send_email_alert(self, message: str):
        """Send email alert."""
        # Implement email sending logic here
        pass

    def _send_slack_alert(self, message: str):
        """Send Slack alert."""
        try:
            response = requests.post(
                self.alert_channels["slack"],
                json={"text": message}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error sending Slack alert: {e}")

    def log_metrics(self):
        """Log collected metrics."""
        self.logger.info(
            f"Metrics - CPU: {self.metrics['cpu_usage']}%, "
            f"Memory: {self.metrics['memory_usage']:.2f}MB, "
            f"Response Time: {self.metrics['response_time']:.3f}s"
        )

    def run(self):
        """Run the monitoring loop."""
        self.logger.info("Starting application monitoring")
        
        while True:
            try:
                # Collect metrics
                self.collect_metrics()
                
                # Log metrics
                self.log_metrics()
                
                # Check thresholds and send alerts
                alerts = self.check_thresholds()
                for alert in alerts:
                    self.send_alert(alert)
                
                # Sleep for the configured interval
                time.sleep(self.config["monitoring"]["interval"])
            
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying

def main():
    """Main function."""
    monitor = ApplicationMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 