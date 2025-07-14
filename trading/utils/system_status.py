"""System status monitoring utility.

This module provides functions to check the health of various system components
and return an overall system status.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class HealthProbe:
    """Health probe for monitoring system metrics over time."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.uptime_start = time.time()
        self.error_history = deque(maxlen=history_size)
        self.queue_sizes = defaultdict(lambda: deque(maxlen=history_size))
        self.response_times = deque(maxlen=history_size)
        self.health_checks = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def record_error(
        self, error_type: str, error_message: str, component: str = "unknown"
    ):
        """Record an error occurrence."""
        with self._lock:
            self.error_history.append(
                {
                    "timestamp": datetime.now(),
                    "type": error_type,
                    "message": error_message,
                    "component": component,
                }
            )

    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size for a specific queue."""
        with self._lock:
            self.queue_sizes[queue_name].append(
                {"timestamp": datetime.now(), "size": size}
            )

    def record_response_time(
        self, endpoint: str, response_time: float, status_code: int = 200
    ):
        """Record response time for an endpoint."""
        with self._lock:
            self.response_times.append(
                {
                    "timestamp": datetime.now(),
                    "endpoint": endpoint,
                    "response_time": response_time,
                    "status_code": status_code,
                }
            )

    def record_health_check(
        self, component: str, status: str, details: Dict[str, Any] = None
    ):
        """Record health check result."""
        with self._lock:
            self.health_checks.append(
                {
                    "timestamp": datetime.now(),
                    "component": component,
                    "status": status,
                    "details": details or {},
                }
            )

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.uptime_start

    def get_error_rate(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Calculate error rate over a time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_errors = [
                e for e in self.error_history if e["timestamp"] > cutoff_time
            ]

            if not recent_errors:
                return {
                    "error_rate": 0.0,
                    "total_errors": 0,
                    "error_types": {},
                    "window_minutes": window_minutes,
                }

            # Count errors by type
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error["type"]] += 1

            # Calculate rate (errors per minute)
            error_rate = len(recent_errors) / window_minutes

            return {
                "error_rate": error_rate,
                "total_errors": len(recent_errors),
                "error_types": dict(error_types),
                "window_minutes": window_minutes,
                "recent_errors": recent_errors[-10:],  # Last 10 errors
            }

    def get_queue_metrics(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """Get queue size metrics."""
        with self._lock:
            if queue_name:
                queue_data = self.queue_sizes.get(queue_name, [])
                if not queue_data:
                    return {"queue_name": queue_name, "status": "no_data"}

                sizes = [entry["size"] for entry in queue_data]
                return {
                    "queue_name": queue_name,
                    "current_size": sizes[-1] if sizes else 0,
                    "average_size": sum(sizes) / len(sizes),
                    "max_size": max(sizes),
                    "min_size": min(sizes),
                    "data_points": len(sizes),
                }
            else:
                # Return metrics for all queues
                all_metrics = {}
                for name in self.queue_sizes:
                    all_metrics[name] = self.get_queue_metrics(name)
                return all_metrics

    def get_response_time_metrics(
        self, endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get response time metrics."""
        with self._lock:
            if endpoint:
                endpoint_data = [
                    r for r in self.response_times if r["endpoint"] == endpoint
                ]
                if not endpoint_data:
                    return {"endpoint": endpoint, "status": "no_data"}

                response_times = [r["response_time"] for r in endpoint_data]
                status_codes = [r["status_code"] for r in endpoint_data]

                return {
                    "endpoint": endpoint,
                    "average_response_time": sum(response_times) / len(response_times),
                    "max_response_time": max(response_times),
                    "min_response_time": min(response_times),
                    "success_rate": sum(1 for code in status_codes if code < 400)
                    / len(status_codes),
                    "total_requests": len(response_times),
                    "recent_requests": endpoint_data[-10:],  # Last 10 requests
                }
            else:
                # Return metrics for all endpoints
                endpoints = set(r["endpoint"] for r in self.response_times)
                all_metrics = {}
                for ep in endpoints:
                    all_metrics[ep] = self.get_response_time_metrics(ep)
                return all_metrics

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._lock:
            return {
                "uptime_seconds": self.get_uptime(),
                "uptime_formatted": str(timedelta(seconds=int(self.get_uptime()))),
                "error_metrics": self.get_error_rate(),
                "queue_metrics": self.get_queue_metrics(),
                "response_time_metrics": self.get_response_time_metrics(),
                "health_check_history": list(self.health_checks)[
                    -20:
                ],  # Last 20 health checks
                "system_load": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                },
            }


# Global health probe instance
health_probe = HealthProbe()


def check_disk_space() -> Dict[str, Any]:
    """Check available disk space.

    Returns:
        Dictionary with disk space status
    """
    try:
        disk = psutil.disk_usage("/")
        status = "operational" if disk.percent < 90 else "degraded"

        result = {
            "status": status,
            "percent_used": disk.percent,
            "free_gb": disk.free / (1024**3),
        }

        health_probe.record_health_check("disk", status, result)
        return result

    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        health_probe.record_error("disk_check", str(e), "disk")
        return {"status": "down", "error": str(e)}


def check_memory_usage() -> Dict[str, Any]:
    """Check memory usage.

    Returns:
        Dictionary with memory status
    """
    try:
        memory = psutil.virtual_memory()
        status = "operational" if memory.percent < 90 else "degraded"

        result = {
            "status": status,
            "percent_used": memory.percent,
            "free_gb": memory.available / (1024**3),
        }

        health_probe.record_health_check("memory", status, result)
        return result

    except Exception as e:
        logger.error(f"Error checking memory: {str(e)}")
        health_probe.record_error("memory_check", str(e), "memory")
        return {"status": "down", "error": str(e)}


def check_model_health() -> Dict[str, Any]:
    """Check health of model files and directories.

    Returns:
        Dictionary with model health status
    """
    try:
        model_dir = Path("models")
        if not model_dir.exists():
            health_probe.record_error(
                "model_check", "Model directory not found", "models"
            )
            return {"status": "down", "error": "Model directory not found"}

        # Check if models directory has any Python files (more flexible than requiring specific files)
        model_files = list(model_dir.glob("*.py"))
        if not model_files:
            result = {"status": "degraded", "message": "No Python model files found"}
            health_probe.record_health_check("models", "degraded", result)
            return result

        result = {"status": "operational", "model_count": len(model_files)}
        health_probe.record_health_check("models", "operational", result)
        return result

    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        health_probe.record_error("model_check", str(e), "models")
        return {"status": "down", "error": str(e)}


def check_data_health() -> Dict[str, Any]:
    """Check health of data files and directories.

    Returns:
        Dictionary with data health status
    """
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            health_probe.record_error("data_check", "Data directory not found", "data")
            return {"status": "down", "error": "Data directory not found"}

        # Check if data directory has any files (more flexible than requiring specific files)
        data_files = list(data_dir.glob("*"))
        if not data_files:
            result = {"status": "degraded", "message": "Data directory is empty"}
            health_probe.record_health_check("data", "degraded", result)
            return result

        result = {"status": "operational", "file_count": len(data_files)}
        health_probe.record_health_check("data", "operational", result)
        return result

    except Exception as e:
        logger.error(f"Error checking data health: {str(e)}")
        health_probe.record_error("data_check", str(e), "data")
        return {"status": "down", "error": str(e)}


def get_system_status() -> Dict[str, Any]:
    """Get overall system status.

    Returns:
        Dictionary with system status and component details
    """
    try:
        # Check individual components
        disk_status = check_disk_space()
        memory_status = check_memory_usage()
        model_status = check_model_health()
        data_status = check_data_health()

        # Determine overall status
        statuses = [
            disk_status["status"],
            memory_status["status"],
            model_status["status"],
            data_status["status"],
        ]

        if "down" in statuses:
            overall_status = "down"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "operational"

        # Record overall health check
        health_probe.record_health_check(
            "system",
            overall_status,
            {"component_statuses": statuses, "timestamp": datetime.now().isoformat()},
        )

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "disk": disk_status,
                "memory": memory_status,
                "models": model_status,
                "data": data_status,
            },
            "health_probe": health_probe.get_health_summary(),
        }

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        health_probe.record_error("system_status", str(e), "system")
        return {
            "status": "down",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def build_health_probe_endpoint() -> Dict[str, Any]:
    """Build health probe endpoint data for external monitoring.

    Returns:
        Dictionary containing health probe endpoint configuration
    """
    return {
        "endpoint": "/health",
        "method": "GET",
        "response_format": "json",
        "metrics": {
            "uptime": health_probe.get_uptime(),
            "error_rate": health_probe.get_error_rate(),
            "queue_sizes": health_probe.get_queue_metrics(),
            "response_times": health_probe.get_response_time_metrics(),
            "system_status": get_system_status(),
        },
        "alerts": {
            "high_error_rate": health_probe.get_error_rate()["error_rate"]
            > 1.0,  # > 1 error per minute
            "high_queue_size": any(
                metrics.get("current_size", 0) > 1000
                for metrics in health_probe.get_queue_metrics().values()
                if isinstance(metrics, dict)
            ),
            "high_response_time": any(
                metrics.get("average_response_time", 0) > 5.0  # > 5 seconds
                for metrics in health_probe.get_response_time_metrics().values()
                if isinstance(metrics, dict)
            ),
        },
    }


def get_health_probe_data() -> Dict[str, Any]:
    """Get current health probe data for external consumption.

    Returns:
        Dictionary containing current health metrics
    """
    return health_probe.get_health_summary()


def record_endpoint_call(endpoint: str, response_time: float, status_code: int = 200):
    """Record an endpoint call for monitoring.

    Args:
        endpoint: The endpoint that was called
        response_time: Response time in seconds
        status_code: HTTP status code
    """
    health_probe.record_response_time(endpoint, response_time, status_code)

    # Record errors for non-2xx status codes
    if status_code >= 400:
        health_probe.record_error("http_error", f"HTTP {status_code}", endpoint)


def record_queue_metric(queue_name: str, size: int):
    """Record queue size metric.

    Args:
        queue_name: Name of the queue
        size: Current queue size
    """
    health_probe.record_queue_size(queue_name, size)
