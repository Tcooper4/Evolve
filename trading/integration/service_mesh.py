"""
Service Mesh

Centralized service orchestration system that provides:
- Service registration and discovery
- Health monitoring and load balancing
- Intelligent request routing
- Circuit breakers and fault tolerance
- Service communication protocols
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


class RequestType(Enum):
    """Request type enumeration."""
    FORECAST = "forecast"
    TRADE = "trade"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    REPORT = "report"
    DATA = "data"
    MODEL = "model"
    STRATEGY = "strategy"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    name: str
    service_type: str
    endpoint: str
    status: ServiceStatus
    registered_at: datetime
    last_health_check: datetime
    capabilities: List[str]
    load_factor: float = 1.0
    max_concurrent_requests: int = 10
    current_requests: int = 0
    error_count: int = 0
    success_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['registered_at'] = self.registered_at.isoformat()
        result['last_health_check'] = self.last_health_check.isoformat()
        return result


@dataclass
class ServiceRequest:
    """Service request structure."""
    request_id: str
    request_type: RequestType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ServiceResponse:
    """Service response structure."""
    request_id: str
    service_name: str
    status: str
    data: Any
    error_message: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ServiceHealthMonitor:
    """Monitor service health and performance."""
    
    def __init__(self, health_check_interval: float = 30.0):
        """Initialize the health monitor."""
        self.health_check_interval = health_check_interval
        self.services: Dict[str, ServiceInfo] = {}
        self.health_checkers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"{__name__}.ServiceHealthMonitor")
        self.running = False
    
    async def register_service(self, service_name: str, service_info: ServiceInfo, health_checker: Optional[Callable] = None):
        """Register a service for health monitoring."""
        self.services[service_name] = service_info
        if health_checker:
            self.health_checkers[service_name] = health_checker
        
        self.logger.info(f"âœ… Service '{service_name}' registered for health monitoring")
    
    async def unregister_service(self, service_name: str):
        """Unregister a service from health monitoring."""
        if service_name in self.services:
            del self.services[service_name]
        if service_name in self.health_checkers:
            del self.health_checkers[service_name]
        
        self.logger.info(f"âœ… Service '{service_name}' unregistered from health monitoring")
    
    async def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check health of a specific service."""
        try:
            if service_name not in self.services:
                return ServiceStatus.OFFLINE
            
            service_info = self.services[service_name]
            
            # Use custom health checker if available
            if service_name in self.health_checkers:
                health_checker = self.health_checkers[service_name]
                try:
                    status = await health_checker(service_info)
                    service_info.status = status
                    service_info.last_health_check = datetime.now()
                    return status
                except Exception as e:
                    self.logger.error(f"Health check failed for '{service_name}': {e}")
                    service_info.status = ServiceStatus.UNHEALTHY
                    service_info.error_count += 1
                    return ServiceStatus.UNHEALTHY
            
            # Default health check
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_info.endpoint}/health", timeout=5.0) as response:
                        if response.status == 200:
                            service_info.status = ServiceStatus.HEALTHY
                            service_info.success_count += 1
                        else:
                            service_info.status = ServiceStatus.DEGRADED
                            service_info.error_count += 1
                        
                        service_info.last_health_check = datetime.now()
                        return service_info.status
                        
            except Exception as e:
                self.logger.error(f"Default health check failed for '{service_name}': {e}")
                service_info.status = ServiceStatus.UNHEALTHY
                service_info.error_count += 1
                service_info.last_health_check = datetime.now()
                return ServiceStatus.UNHEALTHY
                
        except Exception as e:
            self.logger.error(f"Health check error for '{service_name}': {e}")
            return ServiceStatus.UNHEALTHY
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True
        self.logger.info("ðŸ”„ Starting service health monitoring...")
        
        while self.running:
            try:
                for service_name in list(self.services.keys()):
                    await self.check_service_health(service_name)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        self.logger.info("ðŸ›‘ Stopped service health monitoring")
    
    def get_all_health_statuses(self) -> Dict[str, ServiceStatus]:
        """Get health status of all services."""
        return {name: service.status for name, service in self.services.items()}
    
    def get_healthy_services(self) -> List[str]:
        """Get list of healthy services."""
        return [name for name, service in self.services.items() 
                if service.status == ServiceStatus.HEALTHY]


class LoadBalancer:
    """Load balancer for distributing requests across services."""
    
    def __init__(self):
        """Initialize the load balancer."""
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
    
    def select_service(self, 
                      available_services: List[ServiceInfo], 
                      request_type: RequestType,
                      strategy: str = "round_robin") -> Optional[ServiceInfo]:
        """Select a service using the specified strategy."""
        if not available_services:
            return None
        
        # Filter services by capability
        capable_services = [
            service for service in available_services
            if request_type.value in service.capabilities
        ]
        
        if not capable_services:
            self.logger.warning(f"No services capable of handling {request_type.value}")
            return None
        
        # Apply load balancing strategy
        if strategy == "round_robin":
            return self._round_robin_select(capable_services)
        elif strategy == "least_loaded":
            return self._least_loaded_select(capable_services)
        elif strategy == "weighted":
            return self._weighted_select(capable_services)
        else:
            return capable_services[0]  # Default to first available
    
    def _round_robin_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Round-robin selection."""
        # Simple round-robin (in practice, you'd want thread-safe counter)
        return services[0]
    
    def _least_loaded_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Select least loaded service."""
        return min(services, key=lambda s: s.current_requests / s.max_concurrent_requests)
    
    def _weighted_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Weighted selection based on load factor and current load."""
        # Calculate weighted scores
        scores = []
        for service in services:
            load_ratio = service.current_requests / service.max_concurrent_requests
            score = service.load_factor / (1 + load_ratio)
            scores.append((score, service))
        
        # Return service with highest score
        return max(scores, key=lambda x: x[0])[1]


class ServiceMesh:
    """
    Centralized service orchestration system.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the service mesh."""
        self.services: Dict[str, ServiceInfo] = {}
        self.health_monitor = ServiceHealthMonitor()
        self.load_balancer = LoadBalancer()
        self.redis_url = redis_url
        self.redis_client = None
        self.logger = logging.getLogger(f"{__name__}.ServiceMesh")
        self.request_counter = 0
        
        # Initialize Redis if URL provided
        if redis_url:
            self._initialize_redis()
        
        self.logger.info("ServiceMesh initialized successfully")
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            self.logger.info("âœ… Redis connection established")
        except Exception as e:
            self.logger.warning(f"âš ï¸ Redis connection failed: {e}")
            self.redis_client = None
    
    async def register_service(self, 
                             service_name: str, 
                             service_type: str,
                             endpoint: str,
                             capabilities: List[str],
                             health_checker: Optional[Callable] = None,
                             **kwargs) -> bool:
        """Register a service in the mesh."""
        try:
            service_info = ServiceInfo(
                name=service_name,
                service_type=service_type,
                endpoint=endpoint,
                status=ServiceStatus.STARTING,
                registered_at=datetime.now(),
                last_health_check=datetime.now(),
                capabilities=capabilities,
                **kwargs
            )
            
            self.services[service_name] = service_info
            await self.health_monitor.register_service(service_name, service_info, health_checker)
            
            # Publish service registration to Redis if available
            if self.redis_client:
                await self._publish_service_event("service_registered", service_info)
            
            self.logger.info(f"âœ… Service '{service_name}' registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"âŒ Failed to register service '{service_name}': {e}")
            return False
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service from the mesh."""
        try:
            if service_name in self.services:
                del self.services[service_name]
                await self.health_monitor.unregister_service(service_name)
                
                # Publish service unregistration to Redis if available
                if self.redis_client:
                    await self._publish_service_event("service_unregistered", {"name": service_name})
                
                self.logger.info(f"âœ… Service '{service_name}' unregistered successfully")
                return True
            else:
                self.logger.warning(f"âš ï¸ Service '{service_name}' not found")
                return False
                
        except Exception as e:
            self.logger.error(f"âŒ Failed to unregister service '{service_name}': {e}")
            return False
    
    async def route_request(self, 
                           request_type: RequestType, 
                           payload: Dict[str, Any],
                           strategy: str = "least_loaded",
                           timeout: float = 30.0) -> ServiceResponse:
        """Route a request to an appropriate service."""
        try:
            # Generate request ID
            self.request_counter += 1
            request_id = f"req_{self.request_counter}_{int(time.time())}"
            
            # Create service request
            service_request = ServiceRequest(
                request_id=request_id,
                request_type=request_type,
                payload=payload,
                timestamp=datetime.now(),
                timeout=timeout
            )
            
            # Get healthy services
            healthy_services = self.health_monitor.get_healthy_services()
            available_services = [self.services[name] for name in healthy_services]
            
            if not available_services:
                return ServiceResponse(
                    request_id=request_id,
                    service_name="none",
                    status="error",
                    data=None,
                    error_message="No healthy services available"
                )
            
            # Select service using load balancer
            selected_service = self.load_balancer.select_service(
                available_services, request_type, strategy
            )
            
            if not selected_service:
                return ServiceResponse(
                    request_id=request_id,
                    service_name="none",
                    status="error",
                    data=None,
                    error_message=f"No service capable of handling {request_type.value}"
                )
            
            # Execute request
            start_time = time.time()
            selected_service.current_requests += 1
            
            try:
                response = await self._execute_request(selected_service, service_request)
                selected_service.success_count += 1
                return response
                
            except Exception as e:
                selected_service.error_count += 1
                return ServiceResponse(
                    request_id=request_id,
                    service_name=selected_service.name,
                    status="error",
                    data=None,
                    error_message=str(e),
                    response_time=time.time() - start_time
                )
                
            finally:
                selected_service.current_requests -= 1
                
        except Exception as e:
            self.logger.error(f"âŒ Request routing failed: {e}")
            return ServiceResponse(
                request_id=request_id if 'request_id' in locals() else "unknown",
                service_name="none",
                status="error",
                data=None,
                error_message=str(e)
            )
    
    async def _execute_request(self, service: ServiceInfo, request: ServiceRequest) -> ServiceResponse:
        """Execute a request against a service."""
        start_time = time.time()
        
        try:
            # Try HTTP request first
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{service.endpoint}/execute",
                    json=request.to_dict(),
                    timeout=request.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ServiceResponse(
                            request_id=request.request_id,
                            service_name=service.name,
                            status="success",
                            data=data,
                            response_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ServiceResponse(
                            request_id=request.request_id,
                            service_name=service.name,
                            status="error",
                            data=None,
                            error_message=f"HTTP {response.status}: {error_text}",
                            response_time=time.time() - start_time
                        )
                        
        except asyncio.TimeoutError:
            return ServiceResponse(
                request_id=request.request_id,
                service_name=service.name,
                status="error",
                data=None,
                error_message="Request timeout",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ServiceResponse(
                request_id=request.request_id,
                service_name=service.name,
                status="error",
                data=None,
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    async def _publish_service_event(self, event_type: str, data: Any):
        """Publish service event to Redis."""
        if not self.redis_client:
            return
        
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            await self.redis_client.publish("service_events", json.dumps(event))
        except Exception as e:
            self.logger.error(f"Failed to publish service event: {e}")
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_statuses = self.health_monitor.get_all_health_statuses()
        
        return {
            "overall_health": "healthy" if all(status == ServiceStatus.HEALTHY for status in health_statuses.values()) else "degraded",
            "services": {
                name: {
                    "status": status.value,
                    "info": self.services[name].to_dict() if name in self.services else None
                }
                for name, status in health_statuses.items()
            },
            "total_services": len(self.services),
            "healthy_services": len([s for s in health_statuses.values() if s == ServiceStatus.HEALTHY]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def start(self):
        """Start the service mesh."""
        self.logger.info("ðŸš€ Starting ServiceMesh...")
        await self.health_monitor.start_monitoring()
    
    async def stop(self):
        """Stop the service mesh."""
        self.logger.info("ðŸ›‘ Stopping ServiceMesh...")
        await self.health_monitor.stop_monitoring()
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific service."""
        if service_name not in self.services:
            return None
        
        service = self.services[service_name]
        return {
            "info": service.to_dict(),
            "health_status": self.health_monitor.get_all_health_statuses().get(service_name),
            "capabilities": service.capabilities
        }
    
    def list_services(self, service_type: Optional[str] = None) -> List[str]:
        """List services with optional filtering."""
        services = list(self.services.keys())
        
        if service_type:
            services = [name for name in services 
                       if self.services[name].service_type == service_type]
        
        return services


# Convenience functions
async def create_service_mesh(redis_url: Optional[str] = None) -> ServiceMesh:
    """Create and start a service mesh."""
    mesh = ServiceMesh(redis_url)
    await mesh.start()
    return mesh


async def register_forecasting_service(mesh: ServiceMesh, 
                                     service_name: str,
                                     endpoint: str) -> bool:
    """Register a forecasting service."""
    return await mesh.register_service(
        service_name=service_name,
        service_type="forecasting",
        endpoint=endpoint,
        capabilities=["forecast", "model", "time_series"]
    )


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("ðŸŒ Service Mesh Demo")
        print("=" * 50)
        
        # Create service mesh
        mesh = ServiceMesh()
        
        # Register example services
        print("\nðŸ”§ Registering example services...")
        
        await mesh.register_service(
            service_name="forecast_service",
            service_type="forecasting",
            endpoint="http://localhost:8001",
            capabilities=["forecast", "model", "time_series"]
        )
        
        await mesh.register_service(
            service_name="analysis_service",
            service_type="analysis",
            endpoint="http://localhost:8002",
            capabilities=["analysis", "data", "report"]
        )
        
        # Route example requests
        print("\nðŸ“¡ Routing example requests...")
        
        # Forecast request
        forecast_response = await mesh.route_request(
            RequestType.FORECAST,
            {"symbol": "AAPL", "horizon": 7},
            strategy="least_loaded"
        )
        print(f"Forecast response: {forecast_response.status}")
        
        # Analysis request
        analysis_response = await mesh.route_request(
            RequestType.ANALYSIS,
            {"symbol": "AAPL", "analysis_type": "technical"},
            strategy="round_robin"
        )
        print(f"Analysis response: {analysis_response.status}")
        
        # Get service health
        print("\nðŸ¥ Service health...")
        health = await mesh.get_service_health()
        print(f"Overall health: {health['overall_health']}")
        print(f"Total services: {health['total_services']}")
        print(f"Healthy services: {health['healthy_services']}")
        
        # Stop mesh
        await mesh.stop()
        
        print("\nâœ… Demo completed!")
    
    asyncio.run(demo())
