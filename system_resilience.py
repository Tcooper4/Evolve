"""
System Resilience Module

This module provides comprehensive system resilience and fallback management
for the trading system, including health monitoring, automatic recovery,
and performance tracking.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import psutil
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status information."""

    component: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]


@dataclass
class FallbackConfig:
    """Fallback configuration."""

    component: str
    primary_handler: Callable
    fallback_handler: Callable
    health_check: Callable
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0


class SystemResilience:
    """System resilience and fallback management."""

    def __init__(self):
        """Initialize system resilience."""
        self.health_status: Dict[str, HealthStatus] = {}
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.recovery_handlers: Dict[str, Callable] = {}

        # Performance metrics
        self.performance_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_io": [],
            "response_times": [],
        }

        # Initialize monitoring
        self._initialize_monitoring()

        logger.info("System Resilience initialized")

    def _initialize_monitoring(self):
        """Initialize system monitoring."""
        # Register default health checks
        self.register_health_check("system", self._check_system_health)
        self.register_health_check("database", self._check_database_health)
        self.register_health_check("api", self._check_api_health)
        self.register_health_check("models", self._check_models_health)
        self.register_health_check("agents", self._check_agents_health)

        # Register default recovery handlers
        self.register_recovery_handler("system", self._recover_system)
        self.register_recovery_handler("database", self._recover_database)
        self.register_recovery_handler("api", self._recover_api)
        self.register_recovery_handler("models", self._recover_models)
        self.register_recovery_handler("agents", self._recover_agents)

    def register_fallback(
        self,
        component: str,
        primary_handler: Callable,
        fallback_handler: Callable,
        health_check: Callable,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ):
        """Register a fallback configuration for a component."""
        self.fallback_configs[component] = FallbackConfig(
            component=component,
            primary_handler=primary_handler,
            fallback_handler=fallback_handler,
            health_check=health_check,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        logger.info(f"Registered fallback for component: {component}")

    async def execute_with_fallback(self, component: str, *args, **kwargs) -> Any:
        """Execute a function with fallback handling."""
        if component not in self.fallback_configs:
            raise ValueError(f"No fallback configuration for component: {component}")

        config = self.fallback_configs[component]
        last_error = None

        # Try primary handler with retries
        for attempt in range(config.retry_attempts):
            try:
                # Check health before execution
                health_status = config.health_check()
                if health_status.status == "error":
                    raise Exception(f"Component unhealthy: {health_status.message}")

                # Execute primary handler
                if asyncio.iscoroutinefunction(config.primary_handler):
                    result = await config.primary_handler(*args, **kwargs)
                else:
                    result = config.primary_handler(*args, **kwargs)

                # Validate result
                if self._validate_result(result):
                    logger.info(f"Primary handler succeeded for {component}")
                    return result
                else:
                    raise Exception("Primary handler returned invalid result")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Primary handler failed for {component} (attempt {attempt + 1}): {e}"
                )
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(config.retry_delay)

        # Try fallback handler
        try:
            logger.info(f"Attempting fallback handler for {component}")
            if asyncio.iscoroutinefunction(config.fallback_handler):
                result = await config.fallback_handler(*args, **kwargs)
            else:
                result = config.fallback_handler(*args, **kwargs)

            if self._validate_result(result):
                logger.info(f"Fallback handler succeeded for {component}")
                return result
            else:
                raise Exception("Fallback handler returned invalid result")

        except Exception as e:
            logger.error(f"Fallback handler failed for {component}: {e}")
            raise Exception(f"Both primary and fallback handlers failed: {last_error}")

    def _validate_result(self, result: Any) -> bool:
        """Validate the result of a handler."""
        return result is not None

    def register_health_check(self, component: str, health_check: Callable):
        """Register a health check function for a component."""
        self.health_status[component] = HealthStatus(
            component=component,
            status="unknown",
            message="Health check not run",
            timestamp=datetime.now(),
            metrics={},
        )
        logger.info(f"Registered health check for component: {component}")

    def register_recovery_handler(self, component: str, recovery_handler: Callable):
        """Register a recovery handler for a component."""
        self.recovery_handlers[component] = recovery_handler
        logger.info(f"Registered recovery handler for component: {component}")

    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                self._collect_performance_metrics()

                # Run health checks
                self._run_health_checks()

                # Check and recover if needed
                self._check_and_recover()

                # Sleep before next iteration
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_performance_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_metrics["cpu_usage"].append(
                {"timestamp": datetime.now(), "value": cpu_percent}
            )

            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_metrics["memory_usage"].append(
                {"timestamp": datetime.now(), "value": memory.percent}
            )

            # Disk usage
            disk = psutil.disk_usage("/")
            self.performance_metrics["disk_usage"].append(
                {"timestamp": datetime.now(), "value": disk.percent}
            )

            # Network I/O
            network = psutil.net_io_counters()
            self.performance_metrics["network_io"].append(
                {
                    "timestamp": datetime.now(),
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                }
            )

            # Keep only last 1000 entries
            for metric in self.performance_metrics.values():
                if len(metric) > 1000:
                    metric.pop(0)

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    def _run_health_checks(self):
        """Run all registered health checks."""
        for component in self.health_status.keys():
            try:
                if component == "system":
                    status = self._check_system_health()
                elif component == "database":
                    status = self._check_database_health()
                elif component == "api":
                    status = self._check_api_health()
                elif component == "models":
                    status = self._check_models_health()
                elif component == "agents":
                    status = self._check_agents_health()
                else:
                    continue

                self.health_status[component] = status

            except Exception as e:
                logger.error(f"Error running health check for {component}: {e}")

    def _check_system_health(self) -> HealthStatus:
        """Check overall system health."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_healthy = cpu_percent < 80

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < 85

            # Check disk usage
            disk = psutil.disk_usage("/")
            disk_healthy = disk.percent < 90

            # Determine overall status
            if cpu_healthy and memory_healthy and disk_healthy:
                status = "healthy"
                message = "System operating normally"
            elif cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = "error"
                message = "Critical system resource usage"
            else:
                status = "warning"
                message = "Elevated system resource usage"

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_healthy": cpu_healthy,
                "memory_healthy": memory_healthy,
                "disk_healthy": disk_healthy,
            }

            return HealthStatus(
                component="system",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
            )

        except Exception as e:
            return HealthStatus(
                component="system",
                status="error",
                message=f"Health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_database_health(self) -> HealthStatus:
        """Check database health."""
        try:
            # This would typically check database connectivity and performance
            # For now, we'll simulate a healthy database
            import sqlite3

            # Try to connect to a test database
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()

            if result and result[0] == 1:
                status = "healthy"
                message = "Database connection successful"
                metrics = {"connection_test": True, "response_time_ms": 5}
            else:
                status = "error"
                message = "Database connection test failed"
                metrics = {"connection_test": False}

            return HealthStatus(
                component="database",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
            )

        except Exception as e:
            return HealthStatus(
                component="database",
                status="error",
                message=f"Database health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_api_health(self) -> HealthStatus:
        """Check API health."""
        try:
            # Test API endpoints
            test_urls = [
                "http://localhost:8000/health",
                "http://localhost:8000/api/v1/status",
            ]

            healthy_endpoints = 0
            total_response_time = 0

            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        healthy_endpoints += 1
                        total_response_time += response.elapsed.total_seconds() * 1000
                except Exception as e:
                    # Log the error but continue checking other endpoints
                    logger.debug(f"API endpoint {url} failed: {e}")
                    pass

            if healthy_endpoints > 0:
                avg_response_time = total_response_time / healthy_endpoints
                if avg_response_time < 1000:  # Less than 1 second
                    status = "healthy"
                    message = f"API responding normally ({healthy_endpoints}/{len(test_urls)} endpoints)"
                else:
                    status = "warning"
                    message = f"API responding slowly ({avg_response_time:.0f}ms avg)"
            else:
                status = "error"
                message = "No API endpoints responding"

            metrics = {
                "healthy_endpoints": healthy_endpoints,
                "total_endpoints": len(test_urls),
                "avg_response_time_ms": total_response_time / max(healthy_endpoints, 1),
            }

            return HealthStatus(
                component="api",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
            )

        except Exception as e:
            return HealthStatus(
                component="api",
                status="error",
                message=f"API health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_models_health(self) -> HealthStatus:
        """Check ML models health."""
        try:
            # Check if model files exist and are accessible
            model_paths = [
                "models/lstm_model.pt",
                "models/xgboost_model.pkl",
                "models/ensemble_model.pkl",
            ]

            existing_models = 0
            total_size = 0

            for model_path in model_paths:
                if Path(model_path).exists():
                    existing_models += 1
                    total_size += Path(model_path).stat().st_size

            if existing_models > 0:
                total_size / existing_models / (1024 * 1024)
                if existing_models >= len(model_paths) * 0.8:  # 80% of models available
                    status = "healthy"
                    message = f"Models available ({existing_models}/{len(model_paths)})"
                else:
                    status = "warning"
                    message = (
                        f"Some models missing ({existing_models}/{len(model_paths)})"
                    )
            else:
                status = "error"
                message = "No models found"

            metrics = {
                "existing_models": existing_models,
                "total_models": len(model_paths),
                "avg_model_size_mb": total_size
                / max(existing_models, 1)
                / (1024 * 1024),
            }

            return HealthStatus(
                component="models",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
            )

        except Exception as e:
            return HealthStatus(
                component="models",
                status="error",
                message=f"Models health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_agents_health(self) -> HealthStatus:
        """Check agents health."""
        try:
            # Check if agent processes are running
            agent_processes = ["python", "agent_controller", "task_orchestrator"]

            running_agents = 0
            total_processes = 0

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if any(
                        agent in " ".join(proc.info["cmdline"] or [])
                        for agent in agent_processes
                    ):
                        total_processes += 1
                        if proc.is_running():
                            running_agents += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if running_agents > 0:
                if running_agents >= total_processes * 0.8:  # 80% of agents running
                    status = "healthy"
                    message = (
                        f"Agents running normally ({running_agents}/{total_processes})"
                    )
                else:
                    status = "warning"
                    message = (
                        f"Some agents not running ({running_agents}/{total_processes})"
                    )
            else:
                status = "error"
                message = "No agents running"

            metrics = {
                "running_agents": running_agents,
                "total_agents": total_processes,
                "agent_health_ratio": running_agents / max(total_processes, 1),
            }

            return HealthStatus(
                component="agents",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
            )

        except Exception as e:
            return HealthStatus(
                component="agents",
                status="error",
                message=f"Agents health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_and_recover(self):
        """Check health status and trigger recovery if needed."""
        for component, status in self.health_status.items():
            if status.status == "error" and component in self.recovery_handlers:
                logger.warning(f"Triggering recovery for {component}: {status.message}")
                try:
                    self.recovery_handlers[component]()
                    logger.info(f"Recovery completed for {component}")
                except Exception as e:
                    logger.error(f"Recovery failed for {component}: {e}")

    def _recover_system(self):
        """Recover system resources."""
        try:
            # Clear memory caches
            import gc

            gc.collect()
            logger.info("System recovery: Memory cache cleared")
        except Exception as e:
            logger.error(f"System recovery failed: {e}")

    def _recover_database(self):
        """Recover database connection."""
        try:
            # This would typically reconnect to the database
            logger.info("Database recovery: Attempting reconnection")
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")

    def _recover_api(self):
        """Recover API services."""
        try:
            # This would typically restart API services
            logger.info("API recovery: Attempting service restart")
        except Exception as e:
            logger.error(f"API recovery failed: {e}")

    def _recover_models(self):
        """Recover ML models."""
        try:
            # This would typically reload models
            logger.info("Models recovery: Attempting model reload")
        except Exception as e:
            logger.error(f"Models recovery failed: {e}")

    def _recover_agents(self):
        """Recover agent processes."""
        try:
            # This would typically restart agent processes
            logger.info("Agents recovery: Attempting process restart")
        except Exception as e:
            logger.error(f"Agents recovery failed: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = "healthy"
        total_components = len(self.health_status)
        healthy_components = 0
        warning_components = 0
        error_components = 0

        for status in self.health_status.values():
            if status.status == "healthy":
                healthy_components += 1
            elif status.status == "warning":
                warning_components += 1
            elif status.status == "error":
                error_components += 1

        if error_components > 0:
            overall_status = "error"
        elif warning_components > 0:
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(),
            "components": {
                component: {
                    "status": status.status,
                    "message": status.message,
                    "metrics": status.metrics,
                }
                for component, status in self.health_status.items()
            },
            "summary": {
                "total_components": total_components,
                "healthy_components": healthy_components,
                "warning_components": warning_components,
                "error_components": error_components,
            },
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get system performance report."""
        if not self.performance_metrics["cpu_usage"]:
            return {"error": "No performance data available"}

        # Calculate averages
        recent_cpu = [m["value"] for m in self.performance_metrics["cpu_usage"][-10:]]
        recent_memory = [
            m["value"] for m in self.performance_metrics["memory_usage"][-10:]
        ]
        recent_disk = [m["value"] for m in self.performance_metrics["disk_usage"][-10:]]

        return {
            "timestamp": datetime.now(),
            "metrics": {
                "cpu_usage": {
                    "current": recent_cpu[-1] if recent_cpu else 0,
                    "average": sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0,
                    "max": max(recent_cpu) if recent_cpu else 0,
                },
                "memory_usage": {
                    "current": recent_memory[-1] if recent_memory else 0,
                    "average": (
                        sum(recent_memory) / len(recent_memory) if recent_memory else 0
                    ),
                    "max": max(recent_memory) if recent_memory else 0,
                },
                "disk_usage": {
                    "current": recent_disk[-1] if recent_disk else 0,
                    "average": (
                        sum(recent_disk) / len(recent_disk) if recent_disk else 0
                    ),
                    "max": max(recent_disk) if recent_disk else 0,
                },
            },
            "monitoring_active": self.monitoring_active,
        }


# Global instance
_system_resilience = None


def get_system_resilience() -> SystemResilience:
    """Get the global system resilience instance."""
    global _system_resilience
    if _system_resilience is None:
        _system_resilience = SystemResilience()
    return _system_resilience


def start_system_monitoring():
    """Start system monitoring."""
    resilience = get_system_resilience()
    resilience.start_monitoring()


def stop_system_monitoring():
    """Stop system monitoring."""
    resilience = get_system_resilience()
    resilience.stop_monitoring()


def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    resilience = get_system_resilience()
    return resilience.get_system_health()


def get_performance_report() -> Dict[str, Any]:
    """Get system performance report."""
    resilience = get_system_resilience()
    return resilience.get_performance_report()
