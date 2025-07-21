#!/usr/bin/env python3
"""
Monitoring management script.
Provides commands for monitoring application health, metrics, and alerts.

This script supports:
- Monitoring application health
- Viewing and exporting metrics
- Managing alerts

Usage:
    python manage_monitor.py <command> [options]

Commands:
    health      Check application health
    metrics     View or export metrics
    alerts      Manage alerts

Examples:
    # Check application health
    python manage_monitor.py health

    # View metrics
    python manage_monitor.py metrics --output metrics.json

    # Manage alerts
    python manage_monitor.py alerts --action list
"""

import argparse
import json
import logging
import logging.config
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import psutil
import requests
import yaml
from prometheus_client import start_http_server

from utils.launch_utils import setup_logging


class MonitorManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the monitor manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.monitor_dir = Path("monitor")
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging for the service."""
        return setup_logging(service_name="service")

    def start_monitoring(self, port: int = 9090):
        """Start monitoring service on the specified port."""
        self.logger.info(f"Starting monitoring service on port {port}")

        try:
            # Start monitoring server
            self._start_monitoring_server(port)

            self.logger.info("Monitoring service started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring service: {e}")
            return False

    def _start_monitoring_server(self, port: int):
        """Start Prometheus server and monitoring loop."""
        self.logger.info(f"Starting Prometheus server on port {port}...")

        try:
            # Start Prometheus server
            start_http_server(port)

            # Start monitoring loop
            while True:
                self.collect_metrics()
                time.sleep(15)  # Collect metrics every 15 seconds
        except Exception as e:
            self.logger.error(f"Failed to start monitoring server: {e}")
            return False

    def collect_metrics(self):
        """Collect system and application metrics."""
        try:
            # System metrics
            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.virtual_memory().percent)
            self.disk_usage.set(psutil.disk_usage("/").percent)

            # Application metrics
            self._collect_app_metrics()

            # Save metrics to file
            self._save_metrics()
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")

    def _collect_app_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Get application health
            response = requests.get("http://localhost:8501/health")
            if response.status_code == 200:
                health_data = response.json()

                # Update metrics based on health data
                if "errors" in health_data:
                    self.error_count.inc(len(health_data["errors"]))

                if "latency" in health_data:
                    self.request_latency.observe(health_data["latency"])

                self.request_count.inc()
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
            self.error_count.inc()

    def _save_metrics(self):
        """Save metrics to file."""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent,
                    "disk": psutil.disk_usage("/").percent,
                },
                "application": {
                    "requests": self.request_count._value.get(),
                    "errors": self.error_count._value.get(),
                    "latency": self.request_latency._sum.get()
                    / max(self.request_latency._count.get(), 1),
                },
            }

            # Save to file
            metrics_file = (
                self.monitor_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
            )
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def analyze_metrics(self, days: int = 7):
        """Analyze collected metrics."""
        self.logger.info(f"Analyzing metrics for the last {days} days...")

        try:
            # Load metrics
            metrics = []
            cutoff_date = datetime.now() - timedelta(days=days)

            for metrics_file in self.monitor_dir.glob("metrics_*.json"):
                with open(metrics_file) as f:
                    for line in f:
                        metric = json.loads(line)
                        metric_date = datetime.fromisoformat(metric["timestamp"])
                        if metric_date >= cutoff_date:
                            metrics.append(metric)

            if not metrics:
                self.logger.warning("No metrics found for analysis")
                return False

            # Calculate statistics
            stats = {
                "system": {
                    "cpu": self._calculate_stats([m["system"]["cpu"] for m in metrics]),
                    "memory": self._calculate_stats(
                        [m["system"]["memory"] for m in metrics]
                    ),
                    "disk": self._calculate_stats(
                        [m["system"]["disk"] for m in metrics]
                    ),
                },
                "application": {
                    "requests": sum(m["application"]["requests"] for m in metrics),
                    "errors": sum(m["application"]["errors"] for m in metrics),
                    "latency": self._calculate_stats(
                        [m["application"]["latency"] for m in metrics]
                    ),
                },
            }

            # Print results
            print("\nMetrics Analysis:")
            print("\nSystem Metrics:")
            for metric, values in stats["system"].items():
                print(f"\n{metric.upper()}:")
                print(f"  Average: {values['avg']:.2f}%")
                print(f"  Maximum: {values['max']:.2f}%")
                print(f"  Minimum: {values['min']:.2f}%")

            print("\nApplication Metrics:")
            print(f"\nTotal Requests: {stats['application']['requests']}")
            print(f"Total Errors: {stats['application']['errors']}")
            print(
                f"Error Rate: {(stats['application']['errors'] / max(stats['application']['requests'], 1)) * 100:.2f}%"
            )
            print(f"\nLatency:")
            print(f"  Average: {stats['application']['latency']['avg']:.2f}s")
            print(f"  Maximum: {stats['application']['latency']['max']:.2f}s")
            print(f"  Minimum: {stats['application']['latency']['min']:.2f}s")

            return True
        except Exception as e:
            self.logger.error(f"Failed to analyze metrics: {e}")
            return False

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        return {
            "avg": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
        }

    def clean_metrics(self, days: int = 30):
        """Clean old metrics files."""
        self.logger.info(f"Cleaning metrics older than {days} days...")

        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0

            for metrics_file in self.monitor_dir.glob("metrics_*.json"):
                if metrics_file.stat().st_mtime < cutoff_date.timestamp():
                    metrics_file.unlink()
                    cleaned_count += 1

            self.logger.info(f"Cleaned {cleaned_count} metrics files")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean metrics: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitoring Manager")
    parser.add_argument(
        "command", choices=["start", "analyze", "clean"], help="Command to execute"
    )
    parser.add_argument(
        "--port", type=int, default=9090, help="Port for monitoring server"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days for analysis/cleaning"
    )

    args = parser.parse_args()
    manager = MonitorManager()

    commands = {
        "start": lambda: manager.start_monitoring(args.port),
        "analyze": lambda: manager.analyze_metrics(args.days),
        "clean": lambda: manager.clean_metrics(args.days),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
