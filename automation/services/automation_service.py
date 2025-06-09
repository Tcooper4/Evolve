import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import json
import yaml
from pathlib import Path
import random

from automation.models.automation import (
    AutomationTask,
    AutomationWorkflow,
    TaskStatus,
    WorkflowStatus
)
from automation.services.notification_service import NotificationService
from automation.services.monitoring_service import MonitoringService
from automation.services.rate_limiting_service import RateLimitingService
from automation.services.retry_service import RetryService
from automation.services.cleanup_service import CleanupService
from automation.services.metrics_service import MetricsService
from automation.services.security_service import SecurityService
from automation.services.logging_service import LoggingService
from automation.services.error_handling_service import ErrorHandlingService
from automation.services.validation_service import ValidationService
from automation.services.audit_service import AuditService
from automation.services.config_service import ConfigService
from automation.services.database_service import DatabaseService
from automation.services.cache_service import CacheService
from automation.services.queue_service import QueueService
from automation.services.scheduler_service import SchedulerService
from automation.services.health_service import HealthService
from automation.services.transaction_service import TransactionService
from automation.services.persistence_service import PersistenceService
from automation.services.dependency_service import DependencyService
from automation.services.recovery_service import RecoveryService
from automation.services.state_service import StateService
from automation.services.validation_service import ValidationService
from automation.services.security_service import SecurityService
from automation.services.metrics_service import MetricsService
from automation.services.audit_service import AuditService
from automation.services.health_service import HealthService
from automation.services.logging_service import LoggingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceState(str, Enum):
    """State of a service."""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"
    PAUSED = "paused"
    RESUMING = "resuming"
    VALIDATING = "validating"
    SECURITY_CHECK = "security_check"
    HEALTH_CHECK = "health_check"
    METRICS_COLLECTION = "metrics_collection"
    AUDIT = "audit"
    LOGGING = "logging"

class ServicePriority(int, Enum):
    """Priority of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceSecurityLevel(int, Enum):
    """Security level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceValidationLevel(int, Enum):
    """Validation level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceMetricsLevel(int, Enum):
    """Metrics level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceAuditLevel(int, Enum):
    """Audit level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceHealthLevel(int, Enum):
    """Health level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ServiceLoggingLevel(int, Enum):
    """Logging level of a service."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class CircuitBreakerState(str, Enum):
    """State of circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service is recovered

class CircuitBreaker(BaseModel):
    """Circuit breaker for service protection."""
    service_id: str = Field(..., min_length=1, max_length=255)
    state: CircuitBreakerState = Field(default=CircuitBreakerState.CLOSED)
    failure_threshold: int = Field(default=5)
    success_threshold: int = Field(default=2)
    reset_timeout: int = Field(default=60)  # seconds
    failure_count: int = Field(default=0)
    success_count: int = Field(default=0)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = datetime.utcnow()

    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        self.success_count += 1
        self.last_success_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        if self.state == CircuitBreakerState.HALF_OPEN and self.success_count >= self.success_threshold:
            self.state = CircuitBreakerState.CLOSED
            self.reset()

    def reset(self) -> None:
        """Reset the circuit breaker state."""
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.updated_at = datetime.utcnow()

    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.reset_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False

        return True  # HALF_OPEN state allows execution

class RetryStrategy(BaseModel):
    """Retry strategy for service operations."""
    max_attempts: int = Field(default=3)
    initial_delay: float = Field(default=1.0)  # seconds
    max_delay: float = Field(default=30.0)     # seconds
    backoff_factor: float = Field(default=2.0)
    jitter: bool = Field(default=True)
    jitter_factor: float = Field(default=0.1)
    timeout: Optional[float] = None
    retry_on_exceptions: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.initial_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            jitter_amount = delay * self.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        return max(0, delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_attempts:
            return False
            
        if self.retry_on_exceptions:
            return exception.__class__.__name__ in self.retry_on_exceptions
            
        return True

@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    enabled: bool = True
    priority: ServicePriority = ServicePriority.MEDIUM
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5
    health_check_interval: int = 60
    cleanup_interval: int = 3600
    max_instances: int = 1
    dependencies: Set[str] = field(default_factory=set)
    security_level: ServiceSecurityLevel = ServiceSecurityLevel.MEDIUM
    validation_level: ServiceValidationLevel = ServiceValidationLevel.MEDIUM
    metrics_level: ServiceMetricsLevel = ServiceMetricsLevel.MEDIUM
    audit_level: ServiceAuditLevel = ServiceAuditLevel.MEDIUM
    health_level: ServiceHealthLevel = ServiceHealthLevel.MEDIUM
    logging_level: ServiceLoggingLevel = ServiceLoggingLevel.MEDIUM
    persistence_enabled: bool = True
    recovery_enabled: bool = True
    state_persistence: bool = True
    rate_limit: Optional[Dict[str, int]] = None
    circuit_breaker: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    audit: Optional[Dict[str, Any]] = None
    health: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout < 0:
            raise ValueError("Timeout must be non-negative")
        if self.retry_count < 0:
            raise ValueError("Retry count must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")
        if self.health_check_interval < 0:
            raise ValueError("Health check interval must be non-negative")
        if self.cleanup_interval < 0:
            raise ValueError("Cleanup interval must be non-negative")
        if self.max_instances < 1:
            raise ValueError("Max instances must be positive")
        if self.security_level < 0 or self.security_level > 3:
            raise ValueError("Security level must be between 0 and 3")
        if self.validation_level < 0 or self.validation_level > 3:
            raise ValueError("Validation level must be between 0 and 3")
        if self.metrics_level < 0 or self.metrics_level > 3:
            raise ValueError("Metrics level must be between 0 and 3")
        if self.audit_level < 0 or self.audit_level > 3:
            raise ValueError("Audit level must be between 0 and 3")
        if self.health_level < 0 or self.health_level > 3:
            raise ValueError("Health level must be between 0 and 3")
        if self.logging_level < 0 or self.logging_level > 3:
            raise ValueError("Logging level must be between 0 and 3")
        if self.expires_at and self.expires_at < datetime.utcnow():
            raise ValueError("Expiration time must be in the future")

    def is_expired(self) -> bool:
        """Check if configuration is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at

    def update(self, **kwargs) -> None:
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

class ServiceMetrics(BaseModel):
    """Metrics for a service."""
    service_id: str = Field(..., min_length=1, max_length=255)
    service_name: str = Field(..., min_length=1, max_length=255)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    total_errors: int = Field(default=0)
    total_warnings: int = Field(default=0)
    total_retries: int = Field(default=0)
    total_timeouts: int = Field(default=0)
    total_recoveries: int = Field(default=0)
    cpu_usage: float = Field(default=0.0)
    memory_usage: float = Field(default=0.0)
    disk_usage: float = Field(default=0.0)
    network_usage: float = Field(default=0.0)
    security_violations: int = Field(default=0)
    validation_failures: int = Field(default=0)
    audit_events: int = Field(default=0)
    health_checks: int = Field(default=0)
    metrics_updates: int = Field(default=0)
    logging_events: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('cpu_usage', 'memory_usage', 'disk_usage', 'network_usage')
    def validate_usage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Usage must be between 0 and 100")
        return v

    def add_metric(self, name: str, value: Any) -> None:
        """Add metric."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.metadata[name] = value
        self.updated_at = datetime.utcnow()

    def increment_metric(self, name: str, amount: int = 1) -> None:
        """Increment metric."""
        if hasattr(self, name):
            current = getattr(self, name)
            if isinstance(current, int):
                setattr(self, name, current + amount)
                self.updated_at = datetime.utcnow()

class ServiceHealth(BaseModel):
    """Health status of a service."""
    service_id: str = Field(..., min_length=1, max_length=255)
    service_name: str = Field(..., min_length=1, max_length=255)
    status: ServiceState = Field(default=ServiceState.INITIALIZED)
    healthy: bool = Field(default=False)
    last_check: datetime = Field(default_factory=datetime.utcnow)
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    recovery_count: int = Field(default=0)
    security_violations: int = Field(default=0)
    validation_failures: int = Field(default=0)
    audit_failures: int = Field(default=0)
    metrics_failures: int = Field(default=0)
    logging_failures: int = Field(default=0)
    uptime: timedelta = Field(default=timedelta())
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('error_count', 'warning_count', 'recovery_count', 'security_violations',
              'validation_failures', 'audit_failures', 'metrics_failures', 'logging_failures')
    def validate_counts(cls, v):
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v

    def add_detail(self, key: str, value: Any) -> None:
        """Add health detail."""
        self.details[key] = value
        self.updated_at = datetime.utcnow()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add health metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()

class ServiceAudit(BaseModel):
    """Audit record for a service."""
    id: str = Field(..., min_length=1, max_length=255)
    service_id: str = Field(..., min_length=1, max_length=255)
    service_name: str = Field(..., min_length=1, max_length=255)
    action: str = Field(..., min_length=1, max_length=255)
    status: str = Field(..., min_length=1, max_length=255)
    user_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    security_level: ServiceSecurityLevel = Field(default=ServiceSecurityLevel.MEDIUM)
    validation_level: ServiceValidationLevel = Field(default=ServiceValidationLevel.MEDIUM)
    metrics_level: ServiceMetricsLevel = Field(default=ServiceMetricsLevel.MEDIUM)
    audit_level: ServiceAuditLevel = Field(default=ServiceAuditLevel.MEDIUM)
    health_level: ServiceHealthLevel = Field(default=ServiceHealthLevel.MEDIUM)
    logging_level: ServiceLoggingLevel = Field(default=ServiceLoggingLevel.MEDIUM)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_detail(self, key: str, value: Any) -> None:
        """Add audit detail."""
        self.details[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add audit metadata."""
        self.metadata[key] = value

class AutomationService:
    """Core automation service that manages automation tasks and workflows."""
    
    def __init__(
        self,
        config: Dict[str, ServiceConfig],
        persistence_service: PersistenceService,
        dependency_service: DependencyService,
        recovery_service: RecoveryService,
        state_service: StateService,
        security_service: SecurityService,
        audit_service: AuditService,
        metrics_service: MetricsService,
        health_service: HealthService,
        logging_service: LoggingService
    ):
        """Initialize the automation service.
        
        Args:
            config: Service configuration dictionary
            persistence_service: Service for persisting service state
            dependency_service: Service for managing service dependencies
            recovery_service: Service for handling service recovery
            state_service: Service for managing service state
            security_service: Service for handling security
            audit_service: Service for handling auditing
            metrics_service: Service for handling metrics
            health_service: Service for handling health checks
            logging_service: Service for handling logging
        """
        self.config = config
        self.persistence_service = persistence_service
        self.dependency_service = dependency_service
        self.recovery_service = recovery_service
        self.state_service = state_service
        self.security_service = security_service
        self.audit_service = audit_service
        self.metrics_service = metrics_service
        self.health_service = health_service
        self.logging_service = logging_service
        
        self._services: Dict[str, Any] = {}
        self._states: Dict[str, ServiceState] = {}
        self._health: Dict[str, ServiceHealth] = {}
        self._metrics: Dict[str, ServiceMetrics] = {}
        self._health_checks: Dict[str, asyncio.Task] = {}
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        
        # Initialize core services
        self._init_core_services()
        
        # Load persisted state
        self._load_persisted_state()
        
        # Initialize security
        self._init_security()
        
        # Initialize logging
        self._init_logging()
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, RetryStrategy] = {}
    
    def _init_core_services(self):
        """Initialize core services."""
        core_services = {
            'notification': NotificationService,
            'monitoring': MonitoringService,
            'rate_limiting': RateLimitingService,
            'retry': RetryService,
            'cleanup': CleanupService,
            'metrics': MetricsService,
            'security': SecurityService,
            'logging': LoggingService,
            'error_handling': ErrorHandlingService,
            'validation': ValidationService,
            'audit': AuditService,
            'config': ConfigService,
            'database': DatabaseService,
            'cache': CacheService,
            'queue': QueueService,
            'scheduler': SchedulerService,
            'health': HealthService,
            'transaction': TransactionService
        }
        
        for name, service_class in core_services.items():
            if name in self.config and self.config[name].enabled:
                try:
                    self._services[name] = service_class()
                    self._states[name] = ServiceState.INITIALIZED
                    self._health[name] = ServiceHealth(
                        service_id=str(uuid4()),
                        service_name=name
                    )
                    self._metrics[name] = ServiceMetrics(
                        service_id=str(uuid4()),
                        service_name=name
                    )
                except Exception as e:
                    logger.error(f"Error initializing service {name}: {str(e)}")
                    self._states[name] = ServiceState.ERROR
                    self._health[name].healthy = False
                    self._health[name].details['error'] = str(e)
    
    def _init_security(self):
        """Initialize security settings."""
        try:
            # Initialize security service
            self.security_service.initialize()
            
            # Set up security policies
            for name, service in self._services.items():
                if hasattr(service, 'set_security_policy'):
                    service.set_security_policy(
                        self.config[name].security_level
                    )
            
            # Set up audit policies
            for name, service in self._services.items():
                if hasattr(service, 'set_audit_policy'):
                    service.set_audit_policy(
                        self.config[name].audit_level
                    )
            
        except Exception as e:
            logger.error(f"Error initializing security: {str(e)}")
            raise
    
    def _init_logging(self):
        """Initialize logging settings."""
        try:
            # Initialize logging service
            self.logging_service.initialize()
            
            # Set up logging for each service
            for name, service in self._services.items():
                if hasattr(service, 'set_logging'):
                    service.set_logging(
                        self.config[name].logging
                    )
            
        except Exception as e:
            logger.error(f"Error initializing logging: {str(e)}")
            raise
    
    def _load_persisted_state(self):
        """Load persisted service state."""
        try:
            # Load states
            states = self.persistence_service.load_service_states()
            for name, state in states.items():
                if name in self._states:
                    self._states[name] = state
            
            # Load health
            health = self.persistence_service.load_service_health()
            for name, health_data in health.items():
                if name in self._health:
                    self._health[name] = ServiceHealth(**health_data)
            
            # Load metrics
            metrics = self.persistence_service.load_service_metrics()
            for name, metrics_data in metrics.items():
                if name in self._metrics:
                    self._metrics[name] = ServiceMetrics(**metrics_data)
            
        except Exception as e:
            logger.error(f"Error loading persisted state: {str(e)}")
            raise
    
    def _save_persisted_state(self):
        """Save service state to persistence."""
        try:
            # Save states
            self.persistence_service.save_service_states(self._states)
            
            # Save health
            health_data = {
                name: health.dict()
                for name, health in self._health.items()
            }
            self.persistence_service.save_service_health(health_data)
            
            # Save metrics
            metrics_data = {
                name: metrics.dict()
                for name, metrics in self._metrics.items()
            }
            self.persistence_service.save_service_metrics(metrics_data)
            
        except Exception as e:
            logger.error(f"Error saving persisted state: {str(e)}")
            raise
    
    def _resolve_dependencies(self) -> List[Tuple[str, Any]]:
        """Resolve service dependencies and return ordered list of services to start."""
        try:
            # Get dependency graph
            graph = {
                name: set(service.dependencies)
                for name, service in self._services.items()
            }
            
            # Check for circular dependencies
            if self.dependency_service.has_circular_dependencies(graph):
                raise ValueError("Circular dependencies detected")
            
            # Get ordered list of services
            ordered_services = self.dependency_service.get_ordered_services(graph)
            
            return [
                (name, self._services[name])
                for name in ordered_services
                if name in self._services
            ]
            
        except Exception as e:
            logger.error(f"Error resolving dependencies: {str(e)}")
            raise
    
    async def start(self):
        """Start the automation service."""
        async with self._startup_lock:
            if self._running:
                return
            
            try:
                # Start services in dependency order
                ordered_services = self._resolve_dependencies()
                
                for name, service in ordered_services:
                    await self._start_service(name, service)
                
                self._running = True
                
                # Start health checks
                for name in self._services:
                    self._health_checks[name] = asyncio.create_task(
                        self._health_check_loop(name)
                    )
                
                # Start cleanup tasks
                for name in self._services:
                    self._cleanup_tasks[name] = asyncio.create_task(
                        self._cleanup_loop(name)
                    )
                
                logger.info("Automation service started successfully")
                
            except Exception as e:
                logger.error(f"Error starting automation service: {str(e)}")
                await self.stop()
                raise
    
    async def stop(self):
        """Stop the automation service."""
        async with self._shutdown_lock:
            if not self._running:
                return
            
            try:
                # Stop health checks
                for task in self._health_checks.values():
                    task.cancel()
                self._health_checks.clear()
                
                # Stop cleanup tasks
                for task in self._cleanup_tasks.values():
                    task.cancel()
                self._cleanup_tasks.clear()
                
                # Stop services in reverse dependency order
                ordered_services = self._resolve_dependencies()
                ordered_services.reverse()
                
                for name, service in ordered_services:
                    await self._stop_service(name, service)
                
                self._running = False
                
                # Save final state
                self._save_persisted_state()
                
                logger.info("Automation service stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping automation service: {str(e)}")
                raise
    
    async def _start_service(self, name: str, service: Any):
        """Start a service."""
        try:
            # Update state
            self._states[name] = ServiceState.STARTING
            
            # Start service
            if hasattr(service, 'start'):
                await service.start()
            
            # Update state
            self._states[name] = ServiceState.RUNNING
            self._health[name].healthy = True
            self._health[name].last_check = datetime.utcnow()
            
            # Record metrics
            self._record_service_metrics('start', True)
            
            # Audit
            await self._audit_service_action('start', 'success')
            
            logger.info(f"Service {name} started successfully")
            
        except Exception as e:
            # Update state
            self._states[name] = ServiceState.ERROR
            self._health[name].healthy = False
            self._health[name].error_count += 1
            self._health[name].details['error'] = str(e)
            
            # Record metrics
            self._record_service_metrics('start', False)
            
            # Audit
            await self._audit_service_action('start', 'error', str(e))
            
            logger.error(f"Error starting service {name}: {str(e)}")
            raise
    
    async def _stop_service(self, name: str, service: Any):
        """Stop a service."""
        try:
            # Update state
            self._states[name] = ServiceState.STOPPING
            
            # Stop service
            if hasattr(service, 'stop'):
                await service.stop()
            
            # Update state
            self._states[name] = ServiceState.STOPPED
            
            # Record metrics
            self._record_service_metrics('stop', True)
            
            # Audit
            await self._audit_service_action('stop', 'success')
            
            logger.info(f"Service {name} stopped successfully")
            
        except Exception as e:
            # Update state
            self._states[name] = ServiceState.ERROR
            
            # Record metrics
            self._record_service_metrics('stop', False)
            
            # Audit
            await self._audit_service_action('stop', 'error', str(e))
            
            logger.error(f"Error stopping service {name}: {str(e)}")
            raise
    
    async def _health_check_loop(self, name: str):
        """Health check loop for a service."""
        while True:
            try:
                # Get service
                service = self._services[name]
                
                # Check health
                if hasattr(service, 'check_health'):
                    healthy = await service.check_health()
                else:
                    healthy = self._states[name] == ServiceState.RUNNING
                
                # Update health
                self._health[name].healthy = healthy
                self._health[name].last_check = datetime.utcnow()
                
                if not healthy:
                    self._health[name].error_count += 1
                    
                    # Try recovery
                    if self.config[name].recovery_enabled:
                        await self._recover_service(name)
                
                # Record metrics
                self._record_service_metrics('health_check', healthy)
                
                # Audit
                await self._audit_service_action(
                    'health_check',
                    'success' if healthy else 'error'
                )
                
                # Wait for next check
                await asyncio.sleep(self.config[name].health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {name}: {str(e)}")
                await asyncio.sleep(self.config[name].health_check_interval)
    
    async def _cleanup_loop(self, name: str):
        """Cleanup loop for a service."""
        while True:
            try:
                # Get service
                service = self._services[name]
                
                # Run cleanup
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                
                # Record metrics
                self._record_service_metrics('cleanup', True)
                
                # Audit
                await self._audit_service_action('cleanup', 'success')
                
                # Wait for next cleanup
                await asyncio.sleep(self.config[name].cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop for {name}: {str(e)}")
                await asyncio.sleep(self.config[name].cleanup_interval)
    
    async def _recover_service(self, name: str):
        """Recover a failed service with circuit breaker and retry strategy."""
        if name not in self._services:
            logger.error(f"Service {name} not found")
            return False

        service = self._services[name]
        config = self.config.get(name)
        
        if not config:
            logger.error(f"Configuration not found for service {name}")
            return False

        # Initialize circuit breaker if not exists
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                service_id=name,
                failure_threshold=config.circuit_breaker.get('failure_threshold', 5) if config.circuit_breaker else 5,
                success_threshold=config.circuit_breaker.get('success_threshold', 2) if config.circuit_breaker else 2,
                reset_timeout=config.circuit_breaker.get('reset_timeout', 60) if config.circuit_breaker else 60
            )

        # Initialize retry strategy if not exists
        if name not in self.retry_strategies:
            self.retry_strategies[name] = RetryStrategy(
                max_attempts=config.retry_count,
                initial_delay=config.retry_delay,
                retry_on_exceptions=set(config.validation.get('retry_on_exceptions', [])) if config.validation else set()
            )

        circuit_breaker = self.circuit_breakers[name]
        retry_strategy = self.retry_strategies[name]

        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker is open for service {name}")
            return False

        attempt = 0
        while attempt < retry_strategy.max_attempts:
            try:
                attempt += 1
                logger.info(f"Attempting to recover service {name} (attempt {attempt}/{retry_strategy.max_attempts})")

                # Perform recovery
                await self._start_service(name, service)
                
                # Record success
                circuit_breaker.record_success()
                self._record_service_metrics('recovery', True)
                await self._audit_service_action('recovery', 'success', f"Service {name} recovered successfully")
                
                return True

            except Exception as e:
                logger.error(f"Failed to recover service {name}: {str(e)}")
                
                # Record failure
                circuit_breaker.record_failure()
                self._record_service_metrics('recovery', False)
                await self._audit_service_action('recovery', 'failure', str(e))

                if not retry_strategy.should_retry(attempt, e):
                    break

                # Calculate delay with jitter
                delay = retry_strategy.get_delay(attempt)
                logger.info(f"Waiting {delay:.2f} seconds before next recovery attempt")
                await asyncio.sleep(delay)

        logger.error(f"Failed to recover service {name} after {attempt} attempts")
        return False
    
    def _record_service_metrics(self, action: str, success: bool):
        """Record service metrics."""
        try:
            # Update metrics
            for name, metrics in self._metrics.items():
                metrics.total_requests += 1
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1
                metrics.updated_at = datetime.utcnow()
            
            # Save metrics
            self._save_persisted_state()
            
        except Exception as e:
            logger.error(f"Error recording service metrics: {str(e)}")
    
    async def _audit_service_action(
        self,
        action: str,
        status: str,
        details: Optional[str] = None
    ):
        """Audit a service action."""
        try:
            # Create audit record
            audit = ServiceAudit(
                id=str(uuid4()),
                service_id=str(uuid4()),
                service_name="automation",
                action=action,
                status=status,
                details={'details': details} if details else {}
            )
            
            # Record audit
            await self.audit_service.record_audit(audit)
            
        except Exception as e:
            logger.error(f"Error auditing service action: {str(e)}")
    
    async def create_task(self, task: AutomationTask) -> str:
        """Create a new automation task."""
        try:
            # Validate task
            if not self.validation_service.validate_task(task):
                raise ValueError("Invalid task")
            
            # Check security
            if not self.security_service.can_create_task(task):
                raise PermissionError("Not authorized to create task")
            
            # Create task
            task_id = await self.persistence_service.create_task(task)
            
            # Record metrics
            self._record_service_metrics('create_task', True)
            
            # Audit
            await self._audit_service_action(
                'create_task',
                'success',
                f"Created task {task_id}"
            )
            
            return task_id
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('create_task', False)
            
            # Audit
            await self._audit_service_action(
                'create_task',
                'error',
                str(e)
            )
            
            logger.error(f"Error creating task: {str(e)}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[AutomationTask]:
        """Get an automation task by ID."""
        try:
            # Check security
            if not self.security_service.can_view_task(task_id):
                raise PermissionError("Not authorized to view task")
            
            # Get task
            task = await self.persistence_service.get_task(task_id)
            
            # Record metrics
            self._record_service_metrics('get_task', True)
            
            # Audit
            await self._audit_service_action(
                'get_task',
                'success',
                f"Retrieved task {task_id}"
            )
            
            return task
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('get_task', False)
            
            # Audit
            await self._audit_service_action(
                'get_task',
                'error',
                str(e)
            )
            
            logger.error(f"Error getting task {task_id}: {str(e)}")
            raise
    
    async def update_task(self, task_id: str, task: AutomationTask) -> bool:
        """Update an automation task."""
        try:
            # Validate task
            if not self.validation_service.validate_task(task):
                raise ValueError("Invalid task")
            
            # Check security
            if not self.security_service.can_update_task(task_id):
                raise PermissionError("Not authorized to update task")
            
            # Update task
            success = await self.persistence_service.update_task(task_id, task)
            
            # Record metrics
            self._record_service_metrics('update_task', success)
            
            # Audit
            await self._audit_service_action(
                'update_task',
                'success' if success else 'error',
                f"Updated task {task_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('update_task', False)
            
            # Audit
            await self._audit_service_action(
                'update_task',
                'error',
                str(e)
            )
            
            logger.error(f"Error updating task {task_id}: {str(e)}")
            raise
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete an automation task."""
        try:
            # Check security
            if not self.security_service.can_delete_task(task_id):
                raise PermissionError("Not authorized to delete task")
            
            # Delete task
            success = await self.persistence_service.delete_task(task_id)
            
            # Record metrics
            self._record_service_metrics('delete_task', success)
            
            # Audit
            await self._audit_service_action(
                'delete_task',
                'success' if success else 'error',
                f"Deleted task {task_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('delete_task', False)
            
            # Audit
            await self._audit_service_action(
                'delete_task',
                'error',
                str(e)
            )
            
            logger.error(f"Error deleting task {task_id}: {str(e)}")
            raise
    
    async def list_tasks(self) -> List[AutomationTask]:
        """List all automation tasks."""
        try:
            # Check security
            if not self.security_service.can_list_tasks():
                raise PermissionError("Not authorized to list tasks")
            
            # List tasks
            tasks = await self.persistence_service.list_tasks()
            
            # Record metrics
            self._record_service_metrics('list_tasks', True)
            
            # Audit
            await self._audit_service_action(
                'list_tasks',
                'success',
                f"Listed {len(tasks)} tasks"
            )
            
            return tasks
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('list_tasks', False)
            
            # Audit
            await self._audit_service_action(
                'list_tasks',
                'error',
                str(e)
            )
            
            logger.error(f"Error listing tasks: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute an automation task."""
        try:
            # Get task
            task = await self.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            # Check security
            if not self.security_service.can_execute_task(task_id):
                raise PermissionError("Not authorized to execute task")
            
            # Check dependencies
            if not await self._check_task_dependencies(task):
                raise ValueError("Task dependencies not satisfied")
            
            # Execute task
            success = await self.persistence_service.execute_task(task_id)
            
            # Record metrics
            self._record_service_metrics('execute_task', success)
            
            # Audit
            await self._audit_service_action(
                'execute_task',
                'success' if success else 'error',
                f"Executed task {task_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('execute_task', False)
            
            # Audit
            await self._audit_service_action(
                'execute_task',
                'error',
                str(e)
            )
            
            logger.error(f"Error executing task {task_id}: {str(e)}")
            raise
    
    async def _check_task_dependencies(self, task: AutomationTask) -> bool:
        """Check if task dependencies are satisfied."""
        try:
            # Get dependencies
            dependencies = task.dependencies
            
            # Check each dependency
            for dep_id in dependencies:
                dep_task = await self.get_task(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task dependencies: {str(e)}")
            return False
    
    async def create_workflow(self, workflow: AutomationWorkflow) -> str:
        """Create a new automation workflow."""
        try:
            # Validate workflow
            if not self.validation_service.validate_workflow(workflow):
                raise ValueError("Invalid workflow")
            
            # Check security
            if not self.security_service.can_create_workflow(workflow):
                raise PermissionError("Not authorized to create workflow")
            
            # Create workflow
            workflow_id = await self.persistence_service.create_workflow(workflow)
            
            # Record metrics
            self._record_service_metrics('create_workflow', True)
            
            # Audit
            await self._audit_service_action(
                'create_workflow',
                'success',
                f"Created workflow {workflow_id}"
            )
            
            return workflow_id
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('create_workflow', False)
            
            # Audit
            await self._audit_service_action(
                'create_workflow',
                'error',
                str(e)
            )
            
            logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    async def get_workflow(self, workflow_id: str) -> Optional[AutomationWorkflow]:
        """Get an automation workflow by ID."""
        try:
            # Check security
            if not self.security_service.can_view_workflow(workflow_id):
                raise PermissionError("Not authorized to view workflow")
            
            # Get workflow
            workflow = await self.persistence_service.get_workflow(workflow_id)
            
            # Record metrics
            self._record_service_metrics('get_workflow', True)
            
            # Audit
            await self._audit_service_action(
                'get_workflow',
                'success',
                f"Retrieved workflow {workflow_id}"
            )
            
            return workflow
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('get_workflow', False)
            
            # Audit
            await self._audit_service_action(
                'get_workflow',
                'error',
                str(e)
            )
            
            logger.error(f"Error getting workflow {workflow_id}: {str(e)}")
            raise
    
    async def update_workflow(self, workflow_id: str, workflow: AutomationWorkflow) -> bool:
        """Update an automation workflow."""
        try:
            # Validate workflow
            if not self.validation_service.validate_workflow(workflow):
                raise ValueError("Invalid workflow")
            
            # Check security
            if not self.security_service.can_update_workflow(workflow_id):
                raise PermissionError("Not authorized to update workflow")
            
            # Update workflow
            success = await self.persistence_service.update_workflow(workflow_id, workflow)
            
            # Record metrics
            self._record_service_metrics('update_workflow', success)
            
            # Audit
            await self._audit_service_action(
                'update_workflow',
                'success' if success else 'error',
                f"Updated workflow {workflow_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('update_workflow', False)
            
            # Audit
            await self._audit_service_action(
                'update_workflow',
                'error',
                str(e)
            )
            
            logger.error(f"Error updating workflow {workflow_id}: {str(e)}")
            raise
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete an automation workflow."""
        try:
            # Check security
            if not self.security_service.can_delete_workflow(workflow_id):
                raise PermissionError("Not authorized to delete workflow")
            
            # Delete workflow
            success = await self.persistence_service.delete_workflow(workflow_id)
            
            # Record metrics
            self._record_service_metrics('delete_workflow', success)
            
            # Audit
            await self._audit_service_action(
                'delete_workflow',
                'success' if success else 'error',
                f"Deleted workflow {workflow_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('delete_workflow', False)
            
            # Audit
            await self._audit_service_action(
                'delete_workflow',
                'error',
                str(e)
            )
            
            logger.error(f"Error deleting workflow {workflow_id}: {str(e)}")
            raise
    
    async def list_workflows(self) -> List[AutomationWorkflow]:
        """List all automation workflows."""
        try:
            # Check security
            if not self.security_service.can_list_workflows():
                raise PermissionError("Not authorized to list workflows")
            
            # List workflows
            workflows = await self.persistence_service.list_workflows()
            
            # Record metrics
            self._record_service_metrics('list_workflows', True)
            
            # Audit
            await self._audit_service_action(
                'list_workflows',
                'success',
                f"Listed {len(workflows)} workflows"
            )
            
            return workflows
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('list_workflows', False)
            
            # Audit
            await self._audit_service_action(
                'list_workflows',
                'error',
                str(e)
            )
            
            logger.error(f"Error listing workflows: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """Execute an automation workflow."""
        try:
            # Get workflow
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Check security
            if not self.security_service.can_execute_workflow(workflow_id):
                raise PermissionError("Not authorized to execute workflow")
            
            # Check dependencies
            if not await self._check_workflow_dependencies(workflow):
                raise ValueError("Workflow dependencies not satisfied")
            
            # Execute workflow
            success = await self.persistence_service.execute_workflow(workflow_id)
            
            # Record metrics
            self._record_service_metrics('execute_workflow', success)
            
            # Audit
            await self._audit_service_action(
                'execute_workflow',
                'success' if success else 'error',
                f"Executed workflow {workflow_id}"
            )
            
            return success
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('execute_workflow', False)
            
            # Audit
            await self._audit_service_action(
                'execute_workflow',
                'error',
                str(e)
            )
            
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            raise
    
    async def _check_workflow_dependencies(self, workflow: AutomationWorkflow) -> bool:
        """Check if workflow dependencies are satisfied."""
        try:
            # Get dependencies
            dependencies = workflow.dependencies
            
            # Check each dependency
            for dep_id in dependencies:
                dep_workflow = await self.get_workflow(dep_id)
                if not dep_workflow or dep_workflow.status != WorkflowStatus.COMPLETED:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking workflow dependencies: {str(e)}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        try:
            # Check security
            if not self.security_service.can_view_metrics():
                raise PermissionError("Not authorized to view metrics")
            
            # Get metrics
            metrics = {
                name: metrics.dict()
                for name, metrics in self._metrics.items()
            }
            
            # Record metrics
            self._record_service_metrics('get_metrics', True)
            
            # Audit
            await self._audit_service_action(
                'get_metrics',
                'success',
                "Retrieved service metrics"
            )
            
            return metrics
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('get_metrics', False)
            
            # Audit
            await self._audit_service_action(
                'get_metrics',
                'error',
                str(e)
            )
            
            logger.error(f"Error getting metrics: {str(e)}")
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Check security
            if not self.security_service.can_view_health():
                raise PermissionError("Not authorized to view health status")
            
            # Get health
            health = {
                name: health.dict()
                for name, health in self._health.items()
            }
            
            # Record metrics
            self._record_service_metrics('get_health', True)
            
            # Audit
            await self._audit_service_action(
                'get_health',
                'success',
                "Retrieved service health"
            )
            
            return health
            
        except Exception as e:
            # Record metrics
            self._record_service_metrics('get_health', False)
            
            # Audit
            await self._audit_service_action(
                'get_health',
                'error',
                str(e)
            )
            
            logger.error(f"Error getting health status: {str(e)}")
            raise 