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
        return {
            'success': True,
            'message': 'ApplicationMonitor initialized successfully',
            'timestamp': datetime.now().isoformat()
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return {
                'success': True,
                'result': config,
                'message': 'Configuration loaded successfully',
                'timestamp': datetime.now().isoformat()
            }

    def setup_logging(self) -> Dict[str, Any]:
        """Initialize logging configuration."""
        try:
            log_config_path = Path("config/logging_config.yaml")
            if not log_config_path.exists():
                print("Error: logging_config.yaml not found")
                sys.exit(1)
            
            with open(log_config_path) as f:
                log_config = yaml.safe_load(f)
            
            logging.config.dictConfig(log_config)
            
            return {
                'success': True,
                'message': 'Logging configuration initialized',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system and application metrics."""
        try:
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
            
            return {
                'success': True,
                'message': f'Collected {len(self.metrics)} metrics',
                'timestamp': datetime.now().isoformat(),
                'metrics_count': len(self.metrics)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def check_thresholds(self) -> Dict[str, Any]:
        """Check if any metrics exceed their thresholds."""
        try:
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
            
            return {
                'success': True,
                'result': alerts,
                'message': f'Found {len(alerts)} threshold violations',
                'timestamp': datetime.now().isoformat(),
                'alerts_count': len(alerts)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def send_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert through configured channels."""
        try:
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
            
            return {
                'success': True,
                'message': f'Alert sent for {alert["metric"]}',
                'timestamp': datetime.now().isoformat(),
                'alert_metric': alert["metric"]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _send_email_alert(self, message: str) -> Dict[str, Any]:
        """Send email alert."""
        try:
            # Implement email sending logic here
            # For now, just log the message
            self.logger.info(f"Email alert would be sent: {message}")
            
            return {
                'success': True,
                'message': 'Email alert logged',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _send_slack_alert(self, message: str) -> Dict[str, Any]:
        """Send Slack alert."""
        try:
            response = requests.post(
                self.alert_channels["slack"],
                json={"text": message}
            )
            response.raise_for_status()
            
            return {
                'success': True,
                'message': 'Slack alert sent successfully',
                'timestamp': datetime.now().isoformat()
            }
        except requests.RequestException as e:
            self.logger.error(f"Error sending Slack alert: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def log_metrics(self) -> Dict[str, Any]:
        """Log collected metrics."""
        try:
            self.logger.info(
                f"Metrics - CPU: {self.metrics['cpu_usage']}%, "
                f"Memory: {self.metrics['memory_usage']:.2f}MB, "
                f"Response Time: {self.metrics['response_time']:.3f}s"
            )
            
            return {
                'success': True,
                'message': 'Metrics logged successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def execute_monitoring_loop(self) -> Dict[str, Any]:
        """Execute the monitoring loop.
        
        Continuously monitors application health and system metrics,
        sending alerts when thresholds are exceeded.
        
        Returns:
            Dictionary containing monitoring results and metadata
        """
        try:
            self.logger.info("Starting application monitoring")
            
            loop_count = 0
            total_alerts = 0
            
            while True:
                try:
                    # Collect metrics
                    collect_result = self.collect_metrics()
                    
                    # Log metrics
                    log_result = self.log_metrics()
                    
                    # Check thresholds and send alerts
                    threshold_result = self.check_thresholds()
                    alerts = threshold_result.get('result', [])
                    
                    for alert in alerts:
                        alert_result = self.send_alert(alert)
                        if alert_result['success']:
                            total_alerts += 1
                    
                    loop_count += 1
                    
                    # Sleep for the configured interval
                    time.sleep(self.config["monitoring"]["interval"])
                
                except KeyboardInterrupt:
                    self.logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Wait before retrying
            
            return {
                'success': True,
                'message': 'Monitoring loop completed',
                'timestamp': datetime.now().isoformat(),
                'loop_count': loop_count,
                'total_alerts': total_alerts
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main() -> Dict[str, Any]:
    """Main function for the application monitor."""
    try:
        monitor = ApplicationMonitor()
        result = monitor.execute_monitoring_loop()
        
        return {
            'success': True,
            'message': 'Application monitoring completed',
            'timestamp': datetime.now().isoformat(),
            'monitoring_result': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    main() 