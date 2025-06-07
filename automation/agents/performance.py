import psutil
import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import aiohttp
import platform
import time
from collections import deque
import statistics
from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str]

@dataclass
class Alert:
    id: str
    metric: str
    threshold: float
    value: float
    timestamp: str
    severity: str
    message: str
    resolved: bool = False

class PerformanceMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.metric_history: Dict[str, deque] = {}
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.monitoring_interval = config.get('monitoring_interval', 60)
        self.history_size = config.get('history_size', 1000)
        self.monitoring_task = None

    def setup_logging(self):
        """Configure logging for the performance monitor."""
        log_path = Path("automation/logs/performance")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "performance.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        """Start the performance monitoring loop."""
        self.logger.info("Starting performance monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop the performance monitoring loop."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped performance monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self._collect_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)

    async def _collect_metrics(self):
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        self._record_metric('cpu.percent', cpu_percent, {'type': 'system'})
        self._record_metric('cpu.count', cpu_count, {'type': 'system'})
        if cpu_freq:
            self._record_metric('cpu.frequency', cpu_freq.current, {'type': 'system'})
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._record_metric('memory.total', memory.total, {'type': 'system'})
        self._record_metric('memory.available', memory.available, {'type': 'system'})
        self._record_metric('memory.used', memory.used, {'type': 'system'})
        self._record_metric('memory.percent', memory.percent, {'type': 'system'})
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self._record_metric('disk.total', disk.total, {'type': 'system'})
        self._record_metric('disk.used', disk.used, {'type': 'system'})
        self._record_metric('disk.free', disk.free, {'type': 'system'})
        self._record_metric('disk.percent', disk.percent, {'type': 'system'})
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self._record_metric('network.bytes_sent', net_io.bytes_sent, {'type': 'system'})
        self._record_metric('network.bytes_recv', net_io.bytes_recv, {'type': 'system'})
        self._record_metric('network.packets_sent', net_io.packets_sent, {'type': 'system'})
        self._record_metric('network.packets_recv', net_io.packets_recv, {'type': 'system'})
        
        # Process metrics
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                self._record_metric(
                    'process.cpu_percent',
                    proc.info['cpu_percent'],
                    {'pid': str(proc.info['pid']), 'name': proc.info['name']}
                )
                self._record_metric(
                    'process.memory_percent',
                    proc.info['memory_percent'],
                    {'pid': str(proc.info['pid']), 'name': proc.info['name']}
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _record_metric(self, name: str, value: float, tags: Dict[str, str]):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            tags=tags
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
            self.metric_history[name] = deque(maxlen=self.history_size)
        
        self.metrics[name].append(metric)
        self.metric_history[name].append(value)

    async def _check_thresholds(self):
        """Check metrics against alert thresholds."""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name not in self.metrics:
                continue
            
            current_value = self.metrics[metric_name][-1].value
            if current_value > threshold['value']:
                await self._create_alert(
                    metric_name,
                    threshold['value'],
                    current_value,
                    threshold.get('severity', 'warning'),
                    threshold.get('message', f"{metric_name} exceeded threshold")
                )

    async def _create_alert(
        self,
        metric: str,
        threshold: float,
        value: float,
        severity: str,
        message: str
    ):
        """Create a new alert."""
        alert = Alert(
            id=str(len(self.alerts) + 1),
            metric=metric,
            threshold=threshold,
            value=value,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            message=message
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {alert.id} - {message}")

    def get_metrics(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Metric]:
        """Get metrics with optional filtering."""
        metrics = []
        
        for metric_name, metric_list in self.metrics.items():
            if name and metric_name != name:
                continue
            
            for metric in metric_list:
                if tags and not all(metric.tags.get(k) == v for k, v in tags.items()):
                    continue
                
                if start_time and metric.timestamp < start_time:
                    continue
                
                if end_time and metric.timestamp > end_time:
                    continue
                
                metrics.append(metric)
        
        # Sort by timestamp (newest first)
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            metrics = metrics[:limit]
        
        return metrics

    def get_metric_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metric_history:
            return {}
        
        values = list(self.metric_history[name])
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0
        }

    def get_alerts(
        self,
        metric: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if metric:
            alerts = [a for a in alerts if a.metric == metric]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            alerts = alerts[:limit]
        
        return alerts

    def mark_alert_resolved(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                break

    def clear_metrics(self, before: Optional[datetime] = None):
        """Clear old metrics."""
        if before:
            for name in self.metrics:
                self.metrics[name] = [
                    m for m in self.metrics[name]
                    if datetime.fromisoformat(m.timestamp) > before
                ]
        else:
            self.metrics.clear()
            self.metric_history.clear()
        
        self.logger.info(f"Cleared metrics before {before}")

    def clear_alerts(self, before: Optional[datetime] = None):
        """Clear old alerts."""
        if before:
            self.alerts = [
                a for a in self.alerts
                if datetime.fromisoformat(a.timestamp) > before
            ]
        else:
            self.alerts = []
        
        self.logger.info(f"Cleared alerts before {before}")

    async def get_system_summary(self) -> Dict:
        """Get a summary of system performance."""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv
            },
            'alerts': {
                'total': len(self.alerts),
                'active': len([a for a in self.alerts if not a.resolved]),
                'critical': len([a for a in self.alerts if a.severity == 'critical' and not a.resolved])
            }
        } 