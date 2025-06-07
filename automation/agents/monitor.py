import logging
import psutil
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import aiohttp
import platform

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, config: Dict):
        """Initialize the system monitor."""
        self.config = config
        self.monitor_log_path = Path("automation/logs/monitoring")
        self.monitor_log_path.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict = {}
        self.alerts: List[Dict] = []
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Set up monitoring configuration."""
        self.metrics = {
            "cpu": {
                "threshold": 80,  # CPU usage threshold in percentage
                "check_interval": 60  # Check interval in seconds
            },
            "memory": {
                "threshold": 85,  # Memory usage threshold in percentage
                "check_interval": 60
            },
            "disk": {
                "threshold": 90,  # Disk usage threshold in percentage
                "check_interval": 300
            },
            "network": {
                "threshold": 1000,  # Network latency threshold in milliseconds
                "check_interval": 30
            },
            "api": {
                "threshold": 500,  # API response time threshold in milliseconds
                "check_interval": 30
            }
        }

    async def start_monitoring(self):
        """Start the monitoring system."""
        try:
            while True:
                await self._collect_metrics()
                await self._check_thresholds()
                await self._log_metrics()
                await asyncio.sleep(60)  # Main monitoring loop interval
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            raise

    async def _collect_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            # Network metrics
            net_io = psutil.net_io_counters()
            net_connections = psutil.net_connections()

            # System metrics
            boot_time = psutil.boot_time()
            users = psutil.users()

            # Store metrics
            self.metrics.update({
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": {
                        "current": cpu_freq.current if cpu_freq else None,
                        "min": cpu_freq.min if cpu_freq else None,
                        "max": cpu_freq.max if cpu_freq else None
                    }
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "swap": {
                        "total": swap.total,
                        "used": swap.used,
                        "free": swap.free,
                        "percent": swap.percent
                    }
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                    "io": {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count
                    }
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "connections": len(net_connections)
                },
                "system": {
                    "boot_time": boot_time,
                    "users": len(users),
                    "platform": platform.platform(),
                    "python_version": platform.python_version()
                }
            })

        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            raise

    async def _check_thresholds(self):
        """Check metrics against thresholds."""
        try:
            # Check CPU usage
            if self.metrics["cpu"]["usage_percent"] > self.metrics["cpu"]["threshold"]:
                await self._create_alert("cpu", "High CPU usage detected")

            # Check memory usage
            if self.metrics["memory"]["percent"] > self.metrics["memory"]["threshold"]:
                await self._create_alert("memory", "High memory usage detected")

            # Check disk usage
            if self.metrics["disk"]["percent"] > self.metrics["disk"]["threshold"]:
                await self._create_alert("disk", "High disk usage detected")

        except Exception as e:
            logger.error(f"Failed to check thresholds: {str(e)}")
            raise

    async def _create_alert(self, metric_type: str, message: str):
        """Create an alert."""
        alert = {
            "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "type": metric_type,
            "message": message,
            "metrics": self.metrics[metric_type]
        }
        self.alerts.append(alert)
        logger.warning(f"Alert created: {message}")

    async def _log_metrics(self):
        """Log metrics to file."""
        try:
            log_file = self.monitor_log_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            raise

    async def check_api_health(self, url: str) -> Dict:
        """Check API health."""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = datetime.now()
                async with session.get(url) as response:
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000

                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "healthy": response.status == 200 and response_time < self.metrics["api"]["threshold"]
                    }
        except Exception as e:
            logger.error(f"Failed to check API health: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "healthy": False
            }

    def get_metrics_summary(self) -> Dict:
        """Get a summary of current metrics."""
        try:
            return {
                "timestamp": self.metrics.get("timestamp"),
                "cpu_usage": self.metrics.get("cpu", {}).get("usage_percent"),
                "memory_usage": self.metrics.get("memory", {}).get("percent"),
                "disk_usage": self.metrics.get("disk", {}).get("percent"),
                "active_alerts": len(self.alerts)
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {}

    def get_alerts(self) -> List[Dict]:
        """Get all alerts."""
        return self.alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = [] 