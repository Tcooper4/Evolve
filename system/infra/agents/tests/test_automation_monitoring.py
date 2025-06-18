"""
Test suite for automation monitoring service.

This module contains test cases for:
- Service health checks
- Performance monitoring
- Resource usage tracking
- Alert generation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import os

from services.automation_monitoring import AutomationMonitor
from services.automation_workflows import WorkflowManager
from logs.automation_logging import AutomationLogger

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock(spec=AutomationLogger)

@pytest.fixture
def mock_workflow_manager():
    """Create a mock workflow manager."""
    return Mock(spec=WorkflowManager)

@pytest.fixture
def monitor(mock_logger, mock_workflow_manager):
    """Create an AutomationMonitor instance with mocks."""
    return AutomationMonitor(
        logger=mock_logger,
        workflow_manager=mock_workflow_manager,
        check_interval=1,
        alert_threshold=0.8
    )

@pytest.mark.asyncio
async def test_service_health_check(monitor, mock_logger):
    """Test service health check functionality."""
    # Mock service status
    monitor.services = {
        "api": {"status": "healthy", "last_check": datetime.now()},
        "worker": {"status": "healthy", "last_check": datetime.now()},
        "monitor": {"status": "healthy", "last_check": datetime.now()}
    }
    
    # Run health check
    await monitor.check_service_health()
    
    # Verify logging
    mock_logger.info.assert_called_with(
        "Service health check completed",
        services=monitor.services
    )
    
    # Verify all services are healthy
    assert all(s["status"] == "healthy" for s in monitor.services.values())

@pytest.mark.asyncio
async def test_service_health_check_with_failure(monitor, mock_logger):
    """Test service health check with a failed service."""
    # Mock service status with one failed service
    monitor.services = {
        "api": {"status": "healthy", "last_check": datetime.now()},
        "worker": {"status": "unhealthy", "last_check": datetime.now()},
        "monitor": {"status": "healthy", "last_check": datetime.now()}
    }
    
    # Run health check
    await monitor.check_service_health()
    
    # Verify alert was generated
    mock_logger.error.assert_called_with(
        "Service health check failed",
        services=monitor.services
    )
    
    # Verify alert was sent
    assert monitor.alerts["worker"] is not None

@pytest.mark.asyncio
async def test_performance_monitoring(monitor, mock_logger):
    """Test performance monitoring functionality."""
    # Mock performance metrics
    metrics = {
        "cpu_usage": 0.5,
        "memory_usage": 0.6,
        "disk_usage": 0.4,
        "response_time": 100
    }
    
    # Run performance check
    await monitor.check_performance()
    
    # Verify metrics were logged
    mock_logger.info.assert_called_with(
        "Performance check completed",
        metrics=metrics
    )
    
    # Verify no alerts were generated
    assert not monitor.alerts

@pytest.mark.asyncio
async def test_performance_monitoring_with_threshold(monitor, mock_logger):
    """Test performance monitoring with threshold exceeded."""
    # Mock performance metrics exceeding threshold
    metrics = {
        "cpu_usage": 0.9,
        "memory_usage": 0.95,
        "disk_usage": 0.4,
        "response_time": 1000
    }
    
    # Run performance check
    await monitor.check_performance()
    
    # Verify alert was generated
    mock_logger.warning.assert_called_with(
        "Performance threshold exceeded",
        metrics=metrics
    )
    
    # Verify alert was sent
    assert monitor.alerts["performance"] is not None

@pytest.mark.asyncio
async def test_resource_usage_tracking(monitor, mock_logger):
    """Test resource usage tracking functionality."""
    # Mock resource usage
    resources = {
        "cpu": {"usage": 0.5, "limit": 1.0},
        "memory": {"usage": 0.6, "limit": 1.0},
        "disk": {"usage": 0.4, "limit": 1.0}
    }
    
    # Run resource check
    await monitor.check_resources()
    
    # Verify resources were logged
    mock_logger.info.assert_called_with(
        "Resource check completed",
        resources=resources
    )
    
    # Verify no alerts were generated
    assert not monitor.alerts

@pytest.mark.asyncio
async def test_resource_usage_tracking_with_limit(monitor, mock_logger):
    """Test resource usage tracking with limit exceeded."""
    # Mock resource usage exceeding limits
    resources = {
        "cpu": {"usage": 0.95, "limit": 1.0},
        "memory": {"usage": 0.98, "limit": 1.0},
        "disk": {"usage": 0.4, "limit": 1.0}
    }
    
    # Run resource check
    await monitor.check_resources()
    
    # Verify alert was generated
    mock_logger.warning.assert_called_with(
        "Resource limit exceeded",
        resources=resources
    )
    
    # Verify alert was sent
    assert monitor.alerts["resources"] is not None

@pytest.mark.asyncio
async def test_alert_generation(monitor, mock_logger):
    """Test alert generation functionality."""
    # Create test alert
    alert = {
        "type": "test",
        "message": "Test alert",
        "severity": "high",
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate alert
    await monitor.generate_alert(alert)
    
    # Verify alert was logged
    mock_logger.warning.assert_called_with(
        "Alert generated",
        alert=alert
    )
    
    # Verify alert was stored
    assert alert in monitor.alerts.values()

@pytest.mark.asyncio
async def test_alert_cleanup(monitor, mock_logger):
    """Test alert cleanup functionality."""
    # Create test alerts
    old_alert = {
        "type": "old",
        "message": "Old alert",
        "severity": "low",
        "timestamp": (datetime.now() - timedelta(days=2)).isoformat()
    }
    new_alert = {
        "type": "new",
        "message": "New alert",
        "severity": "high",
        "timestamp": datetime.now().isoformat()
    }
    
    # Add alerts
    monitor.alerts = {
        "old": old_alert,
        "new": new_alert
    }
    
    # Run cleanup
    await monitor.cleanup_alerts()
    
    # Verify old alert was removed
    assert "old" not in monitor.alerts
    assert "new" in monitor.alerts
    
    # Verify cleanup was logged
    mock_logger.info.assert_called_with(
        "Alerts cleaned up",
        removed_count=1
    )

@pytest.mark.asyncio
async def test_monitoring_loop(monitor, mock_logger):
    """Test the main monitoring loop."""
    # Mock the check methods
    monitor.check_service_health = AsyncMock()
    monitor.check_performance = AsyncMock()
    monitor.check_resources = AsyncMock()
    monitor.cleanup_alerts = AsyncMock()
    
    # Run monitoring loop for a short duration
    try:
        await asyncio.wait_for(monitor.start_monitoring(), timeout=0.1)
    except asyncio.TimeoutError:
        pass
    
    # Verify all checks were called
    monitor.check_service_health.assert_called()
    monitor.check_performance.assert_called()
    monitor.check_resources.assert_called()
    monitor.cleanup_alerts.assert_called()
    
    # Verify monitoring was logged
    mock_logger.info.assert_called_with(
        "Monitoring started",
        check_interval=monitor.check_interval
    ) 