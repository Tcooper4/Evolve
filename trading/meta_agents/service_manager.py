"""
# Adapted from automation/core/service_manager.py — legacy service management logic

Service Manager

This module implements a comprehensive service management system that handles:
- Service discovery and registration
- Health checks and monitoring
- Service lifecycle management
- Dependency management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class ServiceHealth:
    """Service health information."""
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    version: str
    description: str
    endpoints: Dict[str, str]
    dependencies: List[str] = field(default_factory=list)
    health: ServiceHealth = field(default_factory=lambda: ServiceHealth(
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.now(),
        response_time=0.0
    ))
    metadata: Dict[str, Any] = field(default_factory=dict)

class ServiceManager:
    """Service management system."""
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service manager."""
        self.config = config
        self.services: Dict[str, ServiceInfo] = {}
        self.health_check_interval = config.get("health_check_interval", 30)
        self.health_check_timeout = config.get("health_check_timeout", 5)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self._health_check_task = None
        self._session = None

    async def start(self):
        """Start the service manager."""
        self._session = aiohttp.ClientSession()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service manager started")

    async def stop(self):
        """Stop the service manager."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("Service manager stopped")

    async def register_service(self, service: ServiceInfo) -> bool:
        """Register a new service."""
        try:
            # Validate service information
            if not self._validate_service(service):
                return False

            # Check dependencies
            if not await self._check_dependencies(service):
                return False

            # Register service
            self.services[service.name] = service
            logger.info(f"Service registered: {service.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register service {service.name}: {str(e)}")
            return False

    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        try:
            if service_name in self.services:
                del self.services[service_name]
                logger.info(f"Service unregistered: {service_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}: {str(e)}")
            return False

    async def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get service information."""
        return self.services.get(service_name)

    async def get_services(self, status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """Get all services, optionally filtered by status."""
        if status is None:
            return list(self.services.values())
        return [s for s in self.services.values() if s.health.status == status]

    async def check_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Check the health of a specific service."""
        service = await self.get_service(service_name)
        if not service:
            return None

        try:
            health = await self._check_service_health(service)
            service.health = health
            return health
        except Exception as e:
            logger.error(f"Failed to check health for service {service_name}: {str(e)}")
            return None

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                for service in self.services.values():
                    await self.check_service_health(service.name)
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def _check_service_health(self, service: ServiceInfo) -> ServiceHealth:
        """Check the health of a service."""
        start_time = datetime.now()
        error_count = 0
        warnings = []

        try:
            # Check health endpoint
            if "health" in service.endpoints:
                async with self._session.get(
                    service.endpoints["health"],
                    timeout=self.health_check_timeout
                ) as response:
                    if response.status != 200:
                        error_count += 1
                        warnings.append(f"Health endpoint returned status {response.status}")
                    else:
                        health_data = await response.json()
                        if not health_data.get("healthy", False):
                            error_count += 1
                            warnings.extend(health_data.get("warnings", []))

            # Check metrics endpoint
            if "metrics" in service.endpoints:
                async with self._session.get(
                    service.endpoints["metrics"],
                    timeout=self.health_check_timeout
                ) as response:
                    if response.status == 200:
                        metrics = await response.json()
                        service.health.metrics.update(metrics)

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Determine status
            if error_count >= self.max_retries:
                status = ServiceStatus.FAILED
            elif error_count > 0:
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.RUNNING

            return ServiceHealth(
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=error_count,
                warnings=warnings,
                metrics=service.health.metrics
            )

        except Exception as e:
            logger.error(f"Health check failed for service {service.name}: {str(e)}")
            return ServiceHealth(
                status=ServiceStatus.FAILED,
                last_check=datetime.now(),
                response_time=0.0,
                error_count=error_count + 1,
                warnings=[str(e)]
            )

    def _validate_service(self, service: ServiceInfo) -> bool:
        """Validate service information."""
        try:
            # Check required fields
            if not service.name or not service.version or not service.endpoints:
                return False

            # Check endpoint URLs
            for endpoint, url in service.endpoints.items():
                if not url.startswith(("http://", "https://")):
                    return False

            return True
        except Exception as e:
            logger.error(f"Service validation failed: {str(e)}")
            return False

    async def _check_dependencies(self, service: ServiceInfo) -> bool:
        """Check if all service dependencies are available."""
        try:
            for dep_name in service.dependencies:
                dep = await self.get_service(dep_name)
                if not dep or dep.health.status != ServiceStatus.RUNNING:
                    logger.warning(f"Dependency {dep_name} not available for service {service.name}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Dependency check failed for service {service.name}: {str(e)}")
            return False

    async def get_service_dependencies(self, service_name: str) -> List[ServiceInfo]:
        """Get all dependencies for a service."""
        service = await self.get_service(service_name)
        if not service:
            return []
        return [await self.get_service(dep) for dep in service.dependencies if await self.get_service(dep)]

    async def get_dependent_services(self, service_name: str) -> List[ServiceInfo]:
        """Get all services that depend on the specified service."""
        return [s for s in self.services.values() if service_name in s.dependencies]

    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get metrics for a service."""
        service = await self.get_service(service_name)
        if not service:
            return {}
        return service.health.metrics

    async def get_service_warnings(self, service_name: str) -> List[str]:
        """Get warnings for a service."""
        service = await self.get_service(service_name)
        if not service:
            return []
        return service.health.warnings 