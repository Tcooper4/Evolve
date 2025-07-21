import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from cachetools import TTLCache
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator
from ratelimit import limits, sleep_and_retry

from utils.launch_utils import setup_logging

logger = logging.getLogger(__name__)


class MetricsConfig(BaseModel):
    """Configuration for metrics."""

    metrics_port: int = Field(default=9090)
    scrape_interval: int = Field(default=15)
    retention_days: int = Field(default=30)
    alert_threshold: float = Field(default=0.9)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)
    export_path: str = Field(default="automation/metrics")
    export_format: str = Field(default="json")

    @validator("export_format")
    def validate_export_format(cls, v):
        valid_formats = ["json", "csv", "excel", "html"]
        if v not in valid_formats:
            raise ValueError(f"Invalid export format. Must be one of: {valid_formats}")
        return v


class Metric(BaseModel):
    """Metric model."""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}


class MetricSeries(BaseModel):
    """Metric series model."""

    name: str
    values: List[float]
    timestamps: List[datetime]
    labels: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}


class AutomationMetricsService:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger("execution_agent")
        self.metrics = {}
        self.lock = asyncio.Lock()

    def setup_logging(self):
        return setup_logging(service_name="execution_agent")

    def setup_metrics(self):
        """Set up metrics for automation metrics service."""
        # Initialize Prometheus metrics
        self.cpu_usage = Gauge("system_cpu_usage", "CPU usage percentage")
        self.memory_usage = Gauge("system_memory_usage", "Memory usage percentage")
        self.disk_usage = Gauge("system_disk_usage", "Disk usage percentage")
        self.task_counter = Counter(
            "task_total", "Total number of tasks", ["type", "status"]
        )
        self.task_duration = Histogram(
            "task_duration_seconds", "Task execution duration", ["type"]
        )
        self.task_queue_size = Gauge("task_queue_size", "Number of tasks in queue")
        self.workflow_counter = Counter(
            "workflow_total", "Total number of workflows", ["status"]
        )
        self.workflow_duration = Histogram(
            "workflow_duration_seconds", "Workflow execution duration"
        )
        self.workflow_queue_size = Gauge(
            "workflow_queue_size", "Number of workflows in queue"
        )
        self.error_counter = Counter("error_total", "Total number of errors", ["type"])

    def setup_cache(self):
        """Setup metrics caching."""
        self.cache = TTLCache(maxsize=1000, ttl=3600)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = {},
        metadata: Dict[str, Any] = {},
    ):
        """Record a metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels,
                metadata=metadata,
            )

            async with self.lock:
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(metric)

            # Update Prometheus metric
            if name == "system_cpu_usage":
                self.cpu_usage.set(value)
            elif name == "system_memory_usage":
                self.memory_usage.set(value)
            elif name == "system_disk_usage":
                self.disk_usage.set(value)
            elif name == "task_total":
                self.task_counter.labels(**labels).inc()
            elif name == "task_duration":
                self.task_duration.labels(**labels).observe(value)
            elif name == "task_queue_size":
                self.task_queue_size.set(value)
            elif name == "workflow_total":
                self.workflow_counter.labels(**labels).inc()
            elif name == "workflow_duration":
                self.workflow_duration.observe(value)
            elif name == "workflow_queue_size":
                self.workflow_queue_size.set(value)
            elif name == "error_total":
                self.error_counter.labels(**labels).inc()

        except Exception as e:
            logger.error(f"Failed to record metric: {str(e)}")
            return None

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def get_metric_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> MetricSeries:
        """Get metric series with filtering."""
        try:
            async with self.lock:
                metrics = self.metrics.get(name, [])

                if start_time:
                    metrics = [m for m in metrics if m.timestamp >= start_time]

                if end_time:
                    metrics = [m for m in metrics if m.timestamp <= end_time]

                if labels:
                    metrics = [
                        m
                        for m in metrics
                        if all(m.labels.get(k) == v for k, v in labels.items())
                    ]

                return MetricSeries(
                    name=name,
                    values=[m.value for m in metrics],
                    timestamps=[m.timestamp for m in metrics],
                    labels=labels or {},
                    metadata=metrics[0].metadata if metrics else {},
                )

        except Exception as e:
            logger.error(f"Failed to get metric series: {str(e)}")
            return None

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def analyze_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Analyze metric series."""
        try:
            series = await self.get_metric_series(name, start_time, end_time, labels)

            if not series.values:
                return {}

            values = np.array(series.values)

            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "percentiles": {
                    "25": float(np.percentile(values, 25)),
                    "75": float(np.percentile(values, 75)),
                    "90": float(np.percentile(values, 90)),
                    "95": float(np.percentile(values, 95)),
                    "99": float(np.percentile(values, 99)),
                },
            }

        except Exception as e:
            logger.error(f"Failed to analyze metric: {str(e)}")
            return {}

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def plot_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
    ) -> str:
        """Plot metric series."""
        try:
            series = await self.get_metric_series(name, start_time, end_time, labels)

            if not series.values:
                return ""

            # Create figure
            fig = go.Figure()

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=series.timestamps, y=series.values, mode="lines", name=name
                )
            )

            # Update layout
            fig.update_layout(
                title=title or name,
                xaxis_title="Time",
                y_axis_title=y_axis_title or "Value",
                showlegend=True,
            )

            # Convert to JSON
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        except Exception as e:
            logger.error(f"Failed to plot metric: {str(e)}")
            return ""

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def export_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[List[str]] = None,
    ):
        """Export metrics to file."""
        try:
            # Create export directory
            export_path = Path("automation/metrics")
            export_path.mkdir(parents=True, exist_ok=True)

            # Get metrics to export
            metrics_to_export = metrics or list(self.metrics.keys())

            # Prepare data
            data = {}
            for name in metrics_to_export:
                series = await self.get_metric_series(name, start_time, end_time)

                if series.values:
                    data[name] = {
                        "values": series.values,
                        "timestamps": [t.isoformat() for t in series.timestamps],
                        "labels": series.labels,
                        "metadata": series.metadata,
                    }

            # Export based on format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = export_path / f"metrics_{timestamp}.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Metrics exported to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.metrics.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return None
