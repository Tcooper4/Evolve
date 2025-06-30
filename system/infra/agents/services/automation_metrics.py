import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, Summary
import psutil
import numpy as np
from scipy import stats
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.utils
import json

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
    
    @validator('export_format')
    def validate_export_format(cls, v):
        valid_formats = ["json", "csv", "excel", "html"]
        if v not in valid_formats:
            raise ValueError(f"Invalid export format. Must be one of: {valid_formats}")
        return {'success': True, 'result': v, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

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

class AutomationMetrics:
    """Metrics collection and analysis functionality."""
    
    def __init__(
        self,
        config_path: str = "automation/config/metrics.json"
    ):
        """Initialize metrics system."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_metrics()
        self.setup_cache()
        self.metrics: Dict[str, List[Metric]] = {}
        self.lock = asyncio.Lock()
        
    def _load_config(self, config_path: str) -> MetricsConfig:
        """Load metrics configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return MetricsConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load metrics config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "metrics.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_metrics(self):
        """Setup Prometheus metrics."""
        try:
            # System metrics
            self.cpu_usage = Gauge(
                'system_cpu_usage',
                'CPU usage percentage'
            )
            self.memory_usage = Gauge(
                'system_memory_usage',
                'Memory usage percentage'
            )
            self.disk_usage = Gauge(
                'system_disk_usage',
                'Disk usage percentage'
            )
            
            # Task metrics
            self.task_counter = Counter(
                'task_total',
                'Total number of tasks',
                ['type', 'status']
            )
            self.task_duration = Histogram(
                'task_duration_seconds',
                'Task execution duration',
                ['type']
            )
            self.task_queue_size = Gauge(
                'task_queue_size',
                'Number of tasks in queue'
            )
            
            # Workflow metrics
            self.workflow_counter = Counter(
                'workflow_total',
                'Total number of workflows',
                ['status']
            )
            self.workflow_duration = Histogram(
                'workflow_duration_seconds',
                'Workflow execution duration'
            )
            self.workflow_queue_size = Gauge(
                'workflow_queue_size',
                'Number of workflows in queue'
            )
            
            # Error metrics
            self.error_counter = Counter(
                'error_total',
                'Total number of errors',
                ['type']
            )
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")
            raise
            
    def setup_cache(self):
        """Setup metrics caching."""
        self.cache = TTLCache(
            maxsize=self.config.cache_size,
            ttl=self.config.cache_ttl
        )
        
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = {},
        metadata: Dict[str, Any] = {}
    ):
        """Record a metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels,
                metadata=metadata
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
            raise
            
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def get_metric_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
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
                        m for m in metrics
                        if all(
                            m.labels.get(k) == v
                            for k, v in labels.items()
                        )
                    ]
                    
                return MetricSeries(
                    name=name,
                    values=[m.value for m in metrics],
                    timestamps=[m.timestamp for m in metrics],
                    labels=labels or {},
                    metadata=metrics[0].metadata if metrics else {}
                )
                
        except Exception as e:
            logger.error(f"Failed to get metric series: {str(e)}")
            raise
            
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def analyze_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Analyze metric series."""
        try:
            series = await self.get_metric_series(
                name,
                start_time,
                end_time,
                labels
            )
            
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
                    "99": float(np.percentile(values, 99))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze metric: {str(e)}")
            raise
            
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def plot_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        y_axis_title: Optional[str] = None
    ) -> str:
        """Plot metric series."""
        try:
            series = await self.get_metric_series(
                name,
                start_time,
                end_time,
                labels
            )
            
            if not series.values:
                return ""
                
            # Create figure
            fig = go.Figure()
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=series.timestamps,
                    y=series.values,
                    mode='lines',
                    name=name
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or name,
                xaxis_title="Time",
                yaxis_title=y_axis_title or "Value",
                showlegend=True
            )
            
            # Convert to JSON
            return json.dumps(
                fig,
                cls=plotly.utils.PlotlyJSONEncoder
            )
            
        except Exception as e:
            logger.error(f"Failed to plot metric: {str(e)}")
            raise
            
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def export_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[List[str]] = None
    ):
        """Export metrics to file."""
        try:
            # Create export directory
            export_path = Path(self.config.export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Get metrics to export
            metrics_to_export = metrics or list(self.metrics.keys())
            
            # Prepare data
            data = {}
            for name in metrics_to_export:
                series = await self.get_metric_series(
                    name,
                    start_time,
                    end_time
                )
                
                if series.values:
                    data[name] = {
                        "values": series.values,
                        "timestamps": [
                            t.isoformat()
                            for t in series.timestamps
                        ],
                        "labels": series.labels,
                        "metadata": series.metadata
                    }
                    
            # Export based on format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.config.export_format == "json":
                file_path = export_path / f"metrics_{timestamp}.json"
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
                    
            elif self.config.export_format == "csv":
                file_path = export_path / f"metrics_{timestamp}.csv"
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                
            elif self.config.export_format == "excel":
                file_path = export_path / f"metrics_{timestamp}.xlsx"
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)
                
            elif self.config.export_format == "html":
                file_path = export_path / f"metrics_{timestamp}.html"
                fig = make_subplots(
                    rows=len(data),
                    cols=1,
                    subplot_titles=list(data.keys())
                )
                
                for i, (name, series) in enumerate(data.items(), 1):
                    fig.add_trace(
                        go.Scatter(
                            x=series["timestamps"],
                            y=series["values"],
                            name=name
                        ),
                        row=i,
                        col=1
                    )
                    
                fig.update_layout(
                    height=300 * len(data),
                    showlegend=True
                )
                
                fig.write_html(file_path)
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.metrics.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise 