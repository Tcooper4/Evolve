"""Performance monitoring and alerting system."""

import json
import logging
import platform
import socket
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


@dataclass
class Metric:
    """Represents a collected system metric."""

    name: str
    value: float
    timestamp: str
    tags: Dict[str, str]
    unit: str
    threshold: Optional[float] = None


@dataclass
class Alert:
    """Represents a triggered alert."""

    id: str
    metric_name: str
    severity: str
    message: str
    timestamp: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[str] = None


class PerformanceMonitor:
    """Monitors system performance metrics and triggers alerts."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        log_file_path: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ):
        """Initialize the performance monitor.

        Args:
            config: Configuration dictionary
            log_file_path: Optional custom path for log file
            debug: Whether to enable debug logging
        """
        # Setup logging
        self.logger = logging.getLogger("PerformanceMonitor")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        if log_file_path:
            log_file = Path(log_file_path)
        else:
            log_dir = Path("logs/performance")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "performance.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Default configuration
        self.default_config = {
            "collection_interval": 60,  # seconds
            "alert_thresholds": {
                "cpu_percent": 90.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0,
                "cpu_temperature": 80.0,  # Celsius
                "disk_io_percent": 90.0,
                "net_io_percent": 90.0,
            },
            "alert_severities": {
                "warning": 0.8,
                "critical": 0.9,
            },  # 80% of threshold  # 90% of threshold
            "metric_history_window": 3600,  # 1 hour
            "alert_callbacks": {},  # Map of severity to callback functions
        }

        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}

        # Initialize state
        self.active = False
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: Dict[str, Alert] = {}
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Create directories
        self.metrics_dir = Path("data/metrics")
        self.alerts_dir = Path("data/alerts")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # System info
        self.hostname = socket.gethostname()
        self.platform = platform.platform()

        self.logger.info(f"Performance monitor initialized on {self.hostname}")

    def start_monitoring(self) -> None:
        """Start the monitoring process."""
        if self.active:
            self.logger.warning("Monitor is already active")

        self.active = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        if not self.active:
            return

        self.logger.info("Stopping performance monitoring...")
        self.active = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")

    def restart_monitoring(self) -> None:
        """Restart the monitoring process with fresh configuration."""
        self.stop_monitoring()
        time.sleep(1)  # Brief pause to ensure clean shutdown
        self.start_monitoring()
        self.logger.info("Performance monitoring restarted")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                for metric in metrics:
                    self._store_metric(metric)
                    self.evaluate_alerts(metric)
                time.sleep(self.config["collection_interval"])
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(5)  # Brief pause before retry

    def collect_metrics(self) -> List[Metric]:
        """Collect all system metrics.

        Returns:
            List of collected metrics
        """
        timestamp = datetime.utcnow().isoformat()
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(
            Metric(
                name="cpu_percent",
                value=cpu_percent,
                timestamp=timestamp,
                tags={"hostname": self.hostname, "platform": self.platform},
                unit="percent",
                threshold=self.config["alert_thresholds"]["cpu_percent"],
            )
        )

        # CPU temperature if available
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                cpu_temp = max(temp.current for temp in temps.values())
                metrics.append(
                    Metric(
                        name="cpu_temperature",
                        value=cpu_temp,
                        timestamp=timestamp,
                        tags={"hostname": self.hostname, "platform": self.platform},
                        unit="celsius",
                        threshold=self.config["alert_thresholds"]["cpu_temperature"],
                    )
                )
        except Exception as e:
            self.logger.debug(f"CPU temperature not available: {str(e)}")

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(
            Metric(
                name="memory_percent",
                value=memory.percent,
                timestamp=timestamp,
                tags={"hostname": self.hostname, "platform": self.platform},
                unit="percent",
                threshold=self.config["alert_thresholds"]["memory_percent"],
            )
        )

        # Disk metrics
        disk = psutil.disk_usage("/")
        metrics.append(
            Metric(
                name="disk_percent",
                value=disk.percent,
                timestamp=timestamp,
                tags={"hostname": self.hostname, "platform": self.platform},
                unit="percent",
                threshold=self.config["alert_thresholds"]["disk_percent"],
            )
        )

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.append(
                Metric(
                    name="disk_io_percent",
                    value=(disk_io.read_bytes + disk_io.write_bytes)
                    / 1024
                    / 1024,  # MB
                    timestamp=timestamp,
                    tags={"hostname": self.hostname, "platform": self.platform},
                    unit="MB",
                    threshold=self.config["alert_thresholds"]["disk_io_percent"],
                )
            )

        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            metrics.append(
                Metric(
                    name="net_io_percent",
                    value=(net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024,  # MB
                    timestamp=timestamp,
                    tags={"hostname": self.hostname, "platform": self.platform},
                    unit="MB",
                    threshold=self.config["alert_thresholds"]["net_io_percent"],
                )
            )

        return metrics

    def _store_metric(self, metric: Metric) -> None:
        """Store a metric in memory and on disk.

        Args:
            metric: Metric to store
        """
        # Store in memory
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        self.metrics[metric.name].append(metric)

        # Trim old metrics
        window = timedelta(seconds=self.config["metric_history_window"])
        cutoff = datetime.utcnow() - window
        self.metrics[metric.name] = [
            m
            for m in self.metrics[metric.name]
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]

        # Store on disk
        metric_file = self.metrics_dir / f"{metric.name}.jsonl"
        with open(metric_file, "a") as f:
            f.write(json.dumps(asdict(metric)) + "\n")

        self.logger.debug(f"Stored metric: {metric.name}={metric.value}{metric.unit}")

    def evaluate_alerts(self, metric: Metric) -> None:
        """Evaluate a metric against thresholds and trigger alerts.

        Args:
            metric: Metric to evaluate
        """
        if not metric.threshold:
            return

        # Calculate severity thresholds
        warning_threshold = (
            metric.threshold * self.config["alert_severities"]["warning"]
        )
        critical_threshold = (
            metric.threshold * self.config["alert_severities"]["critical"]
        )

        # Determine severity
        severity = None
        if metric.value >= critical_threshold:
            severity = "critical"
        elif metric.value >= warning_threshold:
            severity = "warning"

        if severity:
            # Create alert
            alert_id = f"{metric.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            alert = Alert(
                id=alert_id,
                metric_name=metric.name,
                severity=severity,
                message=f"{
                    metric.name} is {severity}: {
                    metric.value}{
                    metric.unit} (threshold: {
                    metric.threshold}{
                        metric.unit})",
                timestamp=datetime.utcnow().isoformat(),
                value=metric.value,
                threshold=metric.threshold,
            )

            # Store alert
            self.alerts[alert_id] = alert
            self._store_alert(alert)

            # Trigger callback if configured
            if severity in self.config["alert_callbacks"]:
                try:
                    self.config["alert_callbacks"][severity](alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")

            self.logger.warning(f"Alert triggered: {alert.message}")

    def _store_alert(self, alert: Alert) -> None:
        """Store an alert on disk.

        Args:
            alert: Alert to store
        """
        alert_file = (
            self.alerts_dir / f"alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        )
        with open(alert_file, "a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")

    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert.

        Args:
            alert_id: ID of alert to resolve
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow().isoformat()
            self._store_alert(alert)
            self.logger.info(f"Alert resolved: {alert_id}")

    def get_metric_history(self, metric_name: str) -> List[Metric]:
        """Get historical data for a metric.

        Args:
            metric_name: Name of metric to retrieve

        Returns:
            List of historical metrics
        """
        return {
            "success": True,
            "result": self.metrics.get(metric_name, []),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def report_summary(self) -> Dict[str, Any]:
        """Generate a summary report of current metrics and alerts.

        Returns:
            Dict containing summary information
        """
        # Get latest metrics
        latest_metrics = {}
        for name, metrics in self.metrics.items():
            if metrics:
                latest_metrics[name] = metrics[-1]

        # Count active alerts by severity
        alert_counts = {"warning": 0, "critical": 0}
        for alert in self.alerts.values():
            if not alert.resolved:
                alert_counts[alert.severity] += 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "hostname": self.hostname,
            "platform": self.platform,
            "metrics": {
                name: asdict(metric) for name, metric in latest_metrics.items()
            },
            "alerts": alert_counts,
            "monitoring_active": self.active,
        }

    def load_synthetic_data(self, data_file: Union[str, Path]) -> None:
        """Load synthetic metric data for testing.

        Args:
            data_file: Path to JSON file containing synthetic data
        """
        try:
            with open(data_file) as f:
                data = json.load(f)

            for metric_data in data:
                metric = Metric(**metric_data)
                self._store_metric(metric)
                self.evaluate_alerts(metric)

            self.logger.info(f"Loaded synthetic data from {data_file}")
        except Exception as e:
            self.logger.error(f"Error loading synthetic data: {str(e)}")
            raise
