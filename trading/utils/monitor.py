# -*- coding: utf-8 -*-
"""Asynchronous system monitoring and alerting."""

import asyncio
import json
import logging
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil


@dataclass
class Metric:
    """Represents a collected system metric."""

    name: str
    value: float
    timestamp: str
    status: str
    tags: Dict[str, str]


@dataclass
class Alert:
    """Represents a triggered alert."""

    id: str
    metric: str
    value: float
    threshold: float
    severity: str
    message: str
    timestamp: str
    agent_id: str


class SystemMonitor:
    """Asynchronous system resource monitor."""

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict] = None,
        log_dir: Optional[Union[str, Path]] = None,
        test_mode: bool = False,
    ):
        """Initialize the system monitor.

        Args:
            agent_id: Unique identifier for this monitor instance
            config: Configuration dictionary
            log_dir: Optional custom path for log directory
            test_mode: Whether to run in test mode with mock values
        """
        self.agent_id = agent_id
        self.test_mode = test_mode

        # Default configuration
        self.default_config = {
            "metrics": {
                "cpu": {"enabled": True, "interval": 60, "threshold": 90.0, "warning_threshold": 80.0},
                "memory": {"enabled": True, "interval": 60, "threshold": 85.0, "warning_threshold": 75.0},
                "disk": {"enabled": True, "interval": 300, "threshold": 90.0, "warning_threshold": 80.0},
                "network": {"enabled": True, "interval": 60, "threshold": 1000, "warning_threshold": 800},  # KB/s
                "api": {"enabled": True, "interval": 30, "threshold": 1000, "warning_threshold": 500},  # ms
            },
            "alert_callbacks": {},
            "log_retention_days": 7,
        }

        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}

        # Setup logging
        self._setup_logging(log_dir)

        # Initialize state
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self.hostname = socket.gethostname()
        self.platform = platform.system()

        self.logger.info(f"System monitor initialized on {self.hostname}")

    def _setup_logging(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        """Setup logging with rotating file handlers.

        Args:
            log_dir: Optional custom path for log directory
        """
        if log_dir:
            log_dir = Path(log_dir)
        else:
            log_dir = Path("logs/monitoring") / datetime.now().strftime("%Y-%m-%d")

        log_dir.mkdir(parents=True, exist_ok=True)

        # Main logger
        self.logger = logging.getLogger(f"SystemMonitor.{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Metric log handler
        metric_handler = RotatingFileHandler(log_dir / "metrics.log", maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
        metric_handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON
        self.logger.addHandler(metric_handler)

        # Alert log handler
        alert_handler = RotatingFileHandler(log_dir / "alerts.log", maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
        alert_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(alert_handler)

        return {"success": True, "message": "Initialization completed", "timestamp": datetime.now().isoformat()}

    async def start(self) -> None:
        """Start monitoring all enabled metrics."""
        if self.running:
            self.logger.warning("Monitor is already running")
            return

        self.running = True
        for metric_name, metric_config in self.config["metrics"].items():
            if metric_config.get("enabled", True):
                self.monitoring_tasks[metric_name] = asyncio.create_task(
                    self._monitor_metric(metric_name, metric_config)
                )
        self.logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop all monitoring tasks."""
        if not self.running:
            return

        self.running = False
        for task in self.monitoring_tasks.values():
            task.cancel()
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        self.monitoring_tasks.clear()
        self.logger.info("System monitoring stopped")

    async def run_once(self) -> Dict[str, Any]:
        """Run a single monitoring cycle for all metrics.

        Returns:
            Dict containing all metric values
        """
        results = {}
        for metric_name, metric_config in self.config["metrics"].items():
            if metric_config.get("enabled", True):
                value = await self._collect_metric(metric_name)
                results[metric_name] = value
        return results

    async def _monitor_metric(self, metric_name: str, config: Dict) -> None:
        """Monitor a single metric continuously.

        Args:
            metric_name: Name of metric to monitor
            config: Metric configuration
        """
        while self.running:
            try:
                value = await self._collect_metric(metric_name)
                await self._evaluate_alert(metric_name, value, config)
                await asyncio.sleep(config["interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring {metric_name}: {str(e)}")
                await asyncio.sleep(5)

    async def _collect_metric(self, metric_name: str) -> float:
        """Collect a single metric value.

        Args:
            metric_name: Name of metric to collect

        Returns:
            Metric value
        """
        if self.test_mode:
            return self._get_mock_value(metric_name)

        if metric_name == "cpu":
            return await self._monitor_cpu()
        elif metric_name == "memory":
            return await self._monitor_memory()
        elif metric_name == "disk":
            return await self._monitor_disk()
        elif metric_name == "network":
            return await self._monitor_network()
        elif metric_name == "api":
            return await self._monitor_api()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    async def _monitor_cpu(self) -> float:
        """Monitor CPU usage.

        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=1)

    async def _monitor_memory(self) -> float:
        """Monitor memory usage.

        Returns:
            Memory usage percentage
        """
        return psutil.virtual_memory().percent

    async def _monitor_disk(self) -> float:
        """Monitor disk usage.

        Returns:
            Disk usage percentage
        """
        return psutil.disk_usage("/").percent

    async def _monitor_network(self) -> float:
        """Monitor network throughput.

        Returns:
            Network throughput in KB/s
        """
        io_counters = psutil.net_io_counters()
        return (io_counters.bytes_sent + io_counters.bytes_recv) / 1024

    async def _monitor_api(self) -> float:
        """Monitor API latency.

        Returns:
            API latency in milliseconds
        """
        if self.platform == "Windows":
            return 0.0  # Skip on Windows

        # Simulate API latency measurement
        await asyncio.sleep(0.1)
        return 100.0

    def _get_mock_value(self, metric_name: str) -> float:
        """Get a mock value for testing.

        Args:
            metric_name: Name of metric to mock

        Returns:
            Mock metric value
        """
        mock_values = {"cpu": 75.0, "memory": 65.0, "disk": 70.0, "network": 500.0, "api": 200.0}
        return {
            "success": True,
            "result": mock_values.get(metric_name, 0.0),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    async def _evaluate_alert(self, metric_name: str, value: float, config: Dict) -> None:
        """Evaluate a metric value against thresholds.

        Args:
            metric_name: Name of metric
            value: Current metric value
            config: Metric configuration
        """
        # Create metric record
        metric = Metric(
            name=metric_name,
            value=value,
            timestamp=datetime.utcnow().isoformat(),
            status="ok",
            tags={"agent_id": self.agent_id, "hostname": self.hostname, "platform": self.platform},
        )

        # Log metric
        self.logger.info(json.dumps(asdict(metric)))

        # Check thresholds
        if value >= config["threshold"]:
            severity = "critical"
        elif value >= config["warning_threshold"]:
            severity = "warning"
        else:
            return

        # Create alert
        alert = Alert(
            id=f"{metric_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            metric=metric_name,
            value=value,
            threshold=config["threshold"],
            severity=severity,
            message=f"{metric_name} is {severity}: {value} (threshold: {config['threshold']})",
            timestamp=datetime.utcnow().isoformat(),
            agent_id=self.agent_id,
        )

        # Log alert
        self.logger.warning(f"Alert: {json.dumps(asdict(alert))}")

        # Trigger callback if configured
        if severity in self.config["alert_callbacks"]:
            try:
                await self.config["alert_callbacks"][severity](alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status.

        Returns:
            Dict containing monitor status
        """
        return {
            "agent_id": self.agent_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "running": self.running,
            "active_metrics": list(self.monitoring_tasks.keys()),
            "config": self.config,
        }
