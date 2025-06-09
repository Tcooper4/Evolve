"""
Tests for the service manager.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
from aiohttp import ClientSession, ClientResponse
from automation.core.service_manager import (
    ServiceManager,
    ServiceInfo,
    ServiceHealth,
    ServiceStatus
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "health_check_interval": 1,
        "health_check_timeout": 1,
        "max_retries": 2,
        "retry_delay": 1
    }

@pytest.fixture
def service_info():
    """Test service information."""
    return ServiceInfo(
        name="test-service",
        version="1.0.0",
        description="Test service",
        endpoints={
            "health": "http://localhost:8080/health",
            "metrics": "http://localhost:8080/metrics"
        },
        dependencies=["dependency-service"]
    )

@pytest.fixture
def dependency_service():
    """Test dependency service."""
    return ServiceInfo(
        name="dependency-service",
        version="1.0.0",
        description="Dependency service",
        endpoints={
            "health": "http://localhost:8081/health",
            "metrics": "http://localhost:8081/metrics"
        }
    )

@pytest.mark.asyncio
async def test_service_manager_initialization(config):
    """Test service manager initialization."""
    manager = ServiceManager(config)
    assert manager.health_check_interval == 1
    assert manager.health_check_timeout == 1
    assert manager.max_retries == 2
    assert manager.retry_delay == 1
    assert len(manager.services) == 0

@pytest.mark.asyncio
async def test_service_registration(config, service_info):
    """Test service registration."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Register service
    success = await manager.register_service(service_info)
    assert success
    assert service_info.name in manager.services
    assert manager.services[service_info.name] == service_info
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_unregistration(config, service_info):
    """Test service unregistration."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Register and then unregister service
    await manager.register_service(service_info)
    success = await manager.unregister_service(service_info.name)
    assert success
    assert service_info.name not in manager.services
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_health_check(config, service_info):
    """Test service health check."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock health check response
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = Mock(return_value={"healthy": True})
    
    with patch.object(ClientSession, "get", return_value=mock_response):
        await manager.register_service(service_info)
        health = await manager.check_service_health(service_info.name)
        
        assert health is not None
        assert health.status == ServiceStatus.RUNNING
        assert health.error_count == 0
        assert len(health.warnings) == 0
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_dependency_check(config, service_info, dependency_service):
    """Test service dependency check."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Register dependency first
    await manager.register_service(dependency_service)
    
    # Mock health check responses
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = Mock(return_value={"healthy": True})
    
    with patch.object(ClientSession, "get", return_value=mock_response):
        # Register service with dependency
        success = await manager.register_service(service_info)
        assert success
        
        # Get dependencies
        dependencies = await manager.get_service_dependencies(service_info.name)
        assert len(dependencies) == 1
        assert dependencies[0].name == dependency_service.name
        
        # Get dependent services
        dependents = await manager.get_dependent_services(dependency_service.name)
        assert len(dependents) == 1
        assert dependents[0].name == service_info.name
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_metrics(config, service_info):
    """Test service metrics."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock metrics response
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = Mock(return_value={
        "cpu_usage": 0.5,
        "memory_usage": 0.7
    })
    
    with patch.object(ClientSession, "get", return_value=mock_response):
        await manager.register_service(service_info)
        metrics = await manager.get_service_metrics(service_info.name)
        
        assert metrics is not None
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert metrics["cpu_usage"] == 0.5
        assert metrics["memory_usage"] == 0.7
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_warnings(config, service_info):
    """Test service warnings."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock health check response with warnings
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = Mock(return_value={
        "healthy": False,
        "warnings": ["High CPU usage", "Low memory"]
    })
    
    with patch.object(ClientSession, "get", return_value=mock_response):
        await manager.register_service(service_info)
        warnings = await manager.get_service_warnings(service_info.name)
        
        assert len(warnings) == 2
        assert "High CPU usage" in warnings
        assert "Low memory" in warnings
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_validation(config):
    """Test service validation."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Test invalid service (missing required fields)
    invalid_service = ServiceInfo(
        name="",
        version="",
        description="",
        endpoints={}
    )
    success = await manager.register_service(invalid_service)
    assert not success
    
    # Test invalid service (invalid endpoint URL)
    invalid_service = ServiceInfo(
        name="test",
        version="1.0.0",
        description="Test",
        endpoints={"health": "invalid-url"}
    )
    success = await manager.register_service(invalid_service)
    assert not success
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_health_status_transitions(config, service_info):
    """Test service health status transitions."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock health check responses
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = Mock(return_value={"healthy": True})
    
    with patch.object(ClientSession, "get", return_value=mock_response):
        await manager.register_service(service_info)
        
        # Initial state
        health = await manager.check_service_health(service_info.name)
        assert health.status == ServiceStatus.RUNNING
        
        # Simulate degraded state
        mock_response.json = Mock(return_value={"healthy": False, "warnings": ["Warning"]})
        health = await manager.check_service_health(service_info.name)
        assert health.status == ServiceStatus.DEGRADED
        
        # Simulate failed state
        mock_response.status = 500
        health = await manager.check_service_health(service_info.name)
        assert health.status == ServiceStatus.FAILED
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_health_check_timeout(config, service_info):
    """Test service health check timeout."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock health check timeout
    async def mock_get(*args, **kwargs):
        await asyncio.sleep(2)  # Longer than timeout
        return Mock(spec=ClientResponse)
    
    with patch.object(ClientSession, "get", side_effect=mock_get):
        await manager.register_service(service_info)
        health = await manager.check_service_health(service_info.name)
        
        assert health is not None
        assert health.status == ServiceStatus.FAILED
        assert health.error_count > 0
    
    await manager.stop()

@pytest.mark.asyncio
async def test_service_health_check_retry(config, service_info):
    """Test service health check retry mechanism."""
    manager = ServiceManager(config)
    await manager.start()
    
    # Mock health check with temporary failure
    call_count = 0
    async def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("Temporary failure")
        response = Mock(spec=ClientResponse)
        response.status = 200
        response.json = Mock(return_value={"healthy": True})
        return response
    
    with patch.object(ClientSession, "get", side_effect=mock_get):
        await manager.register_service(service_info)
        health = await manager.check_service_health(service_info.name)
        
        assert health is not None
        assert health.status == ServiceStatus.RUNNING
        assert call_count == 2
    
    await manager.stop() 