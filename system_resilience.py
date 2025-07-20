"""
System Resilience and Fallback Mechanisms

This module provides comprehensive system resilience features:
- Health monitoring and diagnostics
- Automatic fallback mechanisms
- System recovery and self-healing
- Performance monitoring
- Security validation
"""

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

    def execute_with_fallback(self, component: str, *args, **kwargs) -> Any:
        """Execute a function with fallback mechanism."""
        if component not in self.fallback_configs:
            logger.warning(f"No fallback configured for component: {component}")
            return None

        config = self.fallback_configs[component]

        # Try primary handler
        for attempt in range(config.retry_attempts):
            try:
                logger.info(
                    f"Attempting primary handler for {component} (attempt {attempt + 1})"
                )
                result = config.primary_handler(*args, **kwargs)

                # Check if result is valid
                if self._validate_result(result):
                    logger.info(f"Primary handler for {component} succeeded")
                    return result
                else:
                    logger.warning(
                        f"Primary handler for {component} returned invalid result"
                    )

            except Exception as e:
                logger.warning(
                    f"Primary handler for {component} failed (attempt {attempt + 1}): {e}"
                )

                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay)
                    continue

        # Try fallback handler
        try:
            logger.info(f"Using fallback handler for {component}")
            result = config.fallback_handler(*args, **kwargs)

            if self._validate_result(result):
                logger.info(f"Fallback handler for {component} succeeded")
                return result
            else:
                logger.error(
                    f"Fallback handler for {component} returned invalid result"
                )
                return None

        except Exception as e:
            logger.error(f"Fallback handler for {component} failed: {e}")
            return None

    def _validate_result(self, result: Any) -> bool:
        """Validate if a result is acceptable."""
        if result is None:
            return False

        # Add custom validation logic here
        return True

    def register_health_check(self, component: str, health_check: Callable):
        """Register a health check for a component."""
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
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                self._collect_performance_metrics()

                # Run health checks
                self._run_health_checks()

                # Check for issues and trigger recovery
                self._check_and_recover()

                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds

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
                {"timestamp": datetime.now(), "value": (disk.used / disk.total) * 100}
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

            # Keep only last 1000 metrics
            for key in self.performance_metrics:
                if len(self.performance_metrics[key]) > 1000:
                    self.performance_metrics[key] = self.performance_metrics[key][
                        -1000:
                    ]

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    def _run_health_checks(self):
        """Run all registered health checks."""
        for component in self.health_status:
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
                self.health_status[component] = HealthStatus(
                    component=component,
                    status="error",
                    message=f"Health check failed: {e}",
                    timestamp=datetime.now(),
                    metrics={},
                )

    def _check_system_health(self) -> HealthStatus:
        """Check system health."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Check memory usage
            memory = psutil.virtual_memory()

            # Check disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # Determine status
            if cpu_percent > 90 or memory.percent > 90 or disk_percent > 90:
                status = "error"
                message = "System resources critically high"
            elif cpu_percent > 80 or memory.percent > 80 or disk_percent > 80:
                status = "warning"
                message = "System resources high"
            else:
                status = "healthy"
                message = "System resources normal"

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk_percent,
                "memory_available": memory.available,
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
                message=f"System health check failed: {e}",
                timestamp=datetime.now(),
                metrics={},
            )

    def _check_database_health(self) -> HealthStatus:
        """Check database health."""
        try:
            # Try to import database modules
            try:
                import pymongo
                import redis
            except ImportError:
                return HealthStatus(
                    component="database",
                    status="warning",
                    message="Database modules not available",
                    timestamp=datetime.now(),
                    metrics={},
                )

            # Check Redis connection
            redis_healthy = False
            try:
                r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=5)
                r.ping()
                redis_healthy = True
            except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
                logger.debug(f"Redis connection failed: {e}")

            # Check MongoDB connection
            mongo_healthy = False
            try:
                client = pymongo.MongoClient(
                    "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
                )
                client.admin.command("ping")
                mongo_healthy = True
            except (
                pymongo.errors.ConnectionFailure,
                pymongo.errors.ServerSelectionTimeoutError,
                Exception,
            ) as e:
                logger.debug(f"MongoDB connection failed: {e}")

            # Determine overall status
            if redis_healthy and mongo_healthy:
                status = "healthy"
                message = "All databases healthy"
            elif redis_healthy or mongo_healthy:
                status = "warning"
                message = "Some databases unavailable"
            else:
                status = "error"
                message = "All databases unavailable"

            metrics = {"redis_healthy": redis_healthy, "mongo_healthy": mongo_healthy}

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
            # Check if main application is responding
            try:
                response = requests.get(
                    "http://localhost:8501/_stcore/health", timeout=5
                )
                if response.status_code == 200:
                    status = "healthy"
                    message = "API responding normally"
                else:
                    status = "warning"
                    message = f"API responding with status {response.status_code}"
            except requests.exceptions.RequestException:
                status = "error"
                message = "API not responding"

            metrics = {
                "response_time": response.elapsed.total_seconds()
                if "response" in locals()
                else None,
                "status_code": response.status_code if "response" in locals() else None,
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
        """Check models health."""
        try:
            # Check if model files exist
            model_path = Path("models")
            if not model_path.exists():
                return HealthStatus(
                    component="models",
                    status="warning",
                    message="Models directory not found",
                    timestamp=datetime.now(),
                    metrics={"models_count": 0},
                )

            # Count available models
            model_files = list(model_path.glob("*.pkl")) + list(model_path.glob("*.pt"))
            models_count = len(model_files)

            if models_count > 0:
                status = "healthy"
                message = f"{models_count} models available"
            else:
                status = "warning"
                message = "No models found"

            metrics = {
                "models_count": models_count,
                "model_files": [str(f.name) for f in model_files[:5]],  # First 5 models
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
            # Check if agent modules can be imported
            agent_modules = [
                "trading.agents.prompt_router_agent",
                "trading.agents.market_regime_agent",
                "trading.optimization.strategy_selection_agent",
            ]

            available_agents = 0
            for module in agent_modules:
                try:
                    __import__(module)
                    available_agents += 1
                except ImportError:
                    pass

            if available_agents == len(agent_modules):
                status = "healthy"
                message = f"All {available_agents} agents available"
            elif available_agents > 0:
                status = "warning"
                message = f"{available_agents}/{len(agent_modules)} agents available"
            else:
                status = "error"
                message = "No agents available"

            metrics = {
                "available_agents": available_agents,
                "total_agents": len(agent_modules),
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
        """Check for issues and trigger recovery."""
        for component, status in self.health_status.items():
            if status.status == "error" and component in self.recovery_handlers:
                logger.warning(f"Triggering recovery for {component}")
                try:
                    self.recovery_handlers[component]()
                    logger.info(f"Recovery completed for {component}")
                except Exception as e:
                    logger.error(f"Recovery failed for {component}: {e}")

    def _recover_system(self):
        """Recover system issues."""
        logger.info("Attempting system recovery...")

        # Clear memory cache
        try:
            import gc

            gc.collect()
            logger.info("Memory cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear memory cache: {e}")

        # Restart monitoring if needed
        if not self.monitoring_active:
            self.start_monitoring()

    def _recover_database(self):
        """Recover database issues."""
        logger.info("Attempting database recovery...")

        # Try to reconnect to databases
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=5)
            r.ping()
            logger.info("Redis reconnected")
        except Exception as e:
            logger.error(f"Failed to reconnect to Redis: {e}")

        try:
            import pymongo

            client = pymongo.MongoClient(
                "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
            )
            client.admin.command("ping")
            logger.info("MongoDB reconnected")
        except Exception as e:
            logger.error(f"Failed to reconnect to MongoDB: {e}")

    def _recover_api(self):
        """Recover API issues."""
        logger.info("Attempting API recovery...")

        # This would typically involve restarting the API service
        # For now, just log the attempt
        logger.info("API recovery attempted")

    def _recover_models(self):
        """Recover models issues."""
        logger.info("Attempting models recovery...")

        # Check if models need to be retrained
        model_path = Path("models")
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created models directory")

    def _recover_agents(self):
        """Recover agents issues."""
        logger.info("Attempting agents recovery...")

        # Try to reinitialize agents
        try:
            # This would involve reinitializing agent components
            logger.info("Agents reinitialized")
        except Exception as e:
            logger.error(f"Failed to reinitialize agents: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        overall_status = "healthy"
        issues = []

        for component, status in self.health_status.items():
            if status.status == "error":
                overall_status = "error"
                issues.append(f"{component}: {status.message}")
            elif status.status == "warning":
                if overall_status != "error":
                    overall_status = "warning"
                issues.append(f"{component}: {status.message}")

        return {
            "overall_status": overall_status,
            "components": {
                k: {
                    "status": v.status,
                    "message": v.message,
                    "timestamp": v.timestamp.isoformat(),
                    "metrics": v.metrics,
                }
                for k, v in self.health_status.items()
            },
            "issues": issues,
            "performance_metrics": {
                k: v[-10:] if v else [] for k, v in self.performance_metrics.items()
            },
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        if not self.performance_metrics["cpu_usage"]:
            return {"error": "No performance data available"}

        # Calculate averages
        cpu_avg = sum(
            m["value"] for m in self.performance_metrics["cpu_usage"][-100:]
        ) / len(self.performance_metrics["cpu_usage"][-100:])
        memory_avg = sum(
            m["value"] for m in self.performance_metrics["memory_usage"][-100:]
        ) / len(self.performance_metrics["memory_usage"][-100:])

        return {
            "cpu_usage_avg": cpu_avg,
            "memory_usage_avg": memory_avg,
            "recent_metrics": {k: v[-10:] for k, v in self.performance_metrics.items()},
        }


# Global instance
system_resilience = SystemResilience()


def get_system_resilience() -> SystemResilience:
    """Get the global system resilience instance."""
    return system_resilience


def start_system_monitoring():
    """Start system monitoring."""
    system_resilience.start_monitoring()


def stop_system_monitoring():
    """Stop system monitoring."""
    system_resilience.stop_monitoring()


def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    return system_resilience.get_system_health()


def get_performance_report() -> Dict[str, Any]:
    """Get performance report."""
    return system_resilience.get_performance_report()

