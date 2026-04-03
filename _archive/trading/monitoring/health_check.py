"""Thin wrapper so pages can import HealthMonitor/SystemHealthMonitor from trading.monitoring.health_check."""

try:
    from monitoring.health_check import HealthChecker

    HealthMonitor = HealthChecker
    SystemHealthMonitor = HealthChecker
except ImportError:
    HealthChecker = None  # type: ignore[misc, assignment]
    HealthMonitor = None  # type: ignore[misc, assignment]
    SystemHealthMonitor = None  # type: ignore[misc, assignment]
