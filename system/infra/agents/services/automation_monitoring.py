import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import prometheus_client as prom
from cachetools import TTLCache
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field
from ratelimit import limits, sleep_and_retry

from trading.automation_core import AutomationCore
from trading.automation_tasks import AutomationTasks
from trading.automation_workflows import AutomationWorkflows

logger = logging.getLogger(__name__)


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""

    metrics_port: int = Field(default=9090)
    scrape_interval: int = Field(default=15)
    retention_days: int = Field(default=30)
    alert_threshold: float = Field(default=0.9)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)


class AutomationMonitoring:
    """Monitoring and metrics collection functionality."""

    def __init__(
        self,
        core: AutomationCore,
        tasks: AutomationTasks,
        workflows: AutomationWorkflows,
        config_path: str = "automation/config/monitoring.json",
    ):
        """Initialize monitoring."""
        self.core = core
        self.tasks = tasks
        self.workflows = workflows
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_metrics()
        self.setup_cache()
        self.metrics_data: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> MonitoringConfig:
        """Load monitoring configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return MonitoringConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load monitoring config: {str(e)}")
            raise

    from utils.launch_utils import setup_logging

def setup_logging():
    """Set up logging for the service."""
    return setup_logging(service_name="execution_agent")def setup_metrics(self):
        """Setup Prometheus metrics."""
        # Task metrics
        self.task_counter = Counter(
            "automation_tasks_total", "Total number of tasks", ["type", "status"]
        )
        self.task_duration = Histogram(
            "automation_task_duration_seconds", "Task execution duration", ["type"]
        )
        self.task_queue_size = Gauge(
            "automation_task_queue_size", "Number of tasks in queue"
        )

        # Workflow metrics
        self.workflow_counter = Counter(
            "automation_workflows_total", "Total number of workflows", ["status"]
        )
        self.workflow_duration = Histogram(
            "automation_workflow_duration_seconds", "Workflow execution duration"
        )
        self.workflow_queue_size = Gauge(
            "automation_workflow_queue_size", "Number of workflows in queue"
        )

        # System metrics
        self.cpu_usage = Gauge("automation_cpu_usage", "CPU usage percentage")
        self.memory_usage = Gauge("automation_memory_usage", "Memory usage percentage")
        self.disk_usage = Gauge("automation_disk_usage", "Disk usage percentage")

        # Error metrics
        self.error_counter = Counter(
            "automation_errors_total", "Total number of errors", ["type"]
        )

    def setup_cache(self):
        """Setup metrics caching."""
        self.cache = TTLCache(maxsize=self.config.cache_size, ttl=self.config.cache_ttl)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def collect_metrics(self):
        """Collect system metrics."""
        try:
            async with self.lock:
                # Collect task metrics
                running_tasks = await self.tasks.get_running_tasks()
                self.task_queue_size.set(len(running_tasks))

                for task_id in running_tasks:
                    task = await self.core.get_task(task_id)
                    if task:
                        self.task_counter.labels(
                            type=task.type.value, status=task.status.value
                        ).inc()

                # Collect workflow metrics
                running_workflows = await self.workflows.get_running_workflows()
                self.workflow_queue_size.set(len(running_workflows))

                for workflow_id in running_workflows:
                    workflow = self.workflows.workflows.get(workflow_id)
                    if workflow:
                        self.workflow_counter.labels(status=workflow.status).inc()

                # Collect system metrics
                self._collect_system_metrics()

                # Check for alerts
                await self._check_alerts()

        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            self.error_counter.labels(type="metrics_collection").inc()

    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.disk_usage.set(disk.percent)

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            self.error_counter.labels(type="system_metrics").inc()

    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            # Check CPU usage
            if self.cpu_usage._value.get() > self.config.alert_threshold * 100:
                await self._create_alert(
                    "high_cpu_usage", f"CPU usage is {self.cpu_usage._value.get()}%"
                )

            # Check memory usage
            if self.memory_usage._value.get() > self.config.alert_threshold * 100:
                await self._create_alert(
                    "high_memory_usage",
                    f"Memory usage is {self.memory_usage._value.get()}%",
                )

            # Check disk usage
            if self.disk_usage._value.get() > self.config.alert_threshold * 100:
                await self._create_alert(
                    "high_disk_usage", f"Disk usage is {self.disk_usage._value.get()}%"
                )

            # Check error rate
            error_rate = (
                self.error_counter._value.get() / self.task_counter._value.get()
            )
            if error_rate > self.config.alert_threshold:
                await self._create_alert(
                    "high_error_rate", f"Error rate is {error_rate:.2%}"
                )

        except Exception as e:
            logger.error(f"Failed to check alerts: {str(e)}")
            self.error_counter.labels(type="alert_check").inc()

    async def _create_alert(self, alert_type: str, message: str):
        """Create a new alert."""
        try:
            alert = {
                "type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "status": "active",
            }

            self.alerts.append(alert)
            logger.warning(f"Alert created: {message}")

        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
            self.error_counter.labels(type="alert_creation").inc()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with caching."""
        try:
            # Try cache first
            if "metrics" in self.cache:
                return self.cache["metrics"]

            # Collect fresh metrics
            await self.collect_metrics()

            metrics = {
                "tasks": {
                    "total": self.task_counter._value.get(),
                    "queue_size": self.task_queue_size._value.get(),
                },
                "workflows": {
                    "total": self.workflow_counter._value.get(),
                    "queue_size": self.workflow_queue_size._value.get(),
                },
                "system": {
                    "cpu_usage": self.cpu_usage._value.get(),
                    "memory_usage": self.memory_usage._value.get(),
                    "disk_usage": self.disk_usage._value.get(),
                },
                "errors": self.error_counter._value.get(),
                "alerts": self.alerts,
            }

            self.cache["metrics"] = metrics
            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {}

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        return self.alerts

    async def clear_alerts(self):
        """Clear resolved alerts."""
        self.alerts = [a for a in self.alerts if a["status"] == "active"]

    async def start_metrics_server(self):
        """Start Prometheus metrics server."""
        try:
            prom.start_http_server(self.config.metrics_port)
            logger.info(f"Started metrics server on port {self.config.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.metrics_data.clear()
            self.alerts.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
