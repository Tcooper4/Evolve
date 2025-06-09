import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field, validator, root_validator
import json
import yaml
from pathlib import Path

from automation.notifications.notification_manager import (
    NotificationManager,
    NotificationType,
    NotificationPriority
)
from automation.services.rate_limiting_service import RateLimitingService
from automation.services.template_service import TemplateService
from automation.services.cache_service import CacheService
from automation.services.metrics_service import MetricsService
from automation.services.retry_service import RetryService
from automation.services.error_handling_service import ErrorHandlingService
from automation.services.config_service import ConfigService
from automation.services.audit_service import AuditService
from automation.services.health_service import HealthService
from automation.services.transaction_service import TransactionService
from automation.services.validation_service import ValidationService
from automation.services.security_service import SecurityService
from automation.services.logging_service import LoggingService
from automation.services.monitoring_service import MonitoringService
from automation.services.scheduler_service import SchedulerService
from automation.services.queue_service import QueueService
from automation.services.worker_service import WorkerService
from automation.services.persistence_service import PersistenceService
from automation.services.dependency_service import DependencyService
from automation.services.recovery_service import RecoveryService
from automation.services.state_service import StateService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemplateVersion(BaseModel):
    """Version of a notification template."""
    version: int = Field(..., ge=1)
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=1000)
    notification_type: NotificationType
    priority: NotificationPriority
    template_vars: Dict[str, Any] = Field(default_factory=dict)
    rate_limit: Optional[Dict[str, int]] = None
    retry_config: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    cache_ttl: Optional[int] = None
    validation_rules: Optional[Dict[str, Any]] = None
    security_level: int = Field(default=1, ge=1, le=3)
    audit_level: int = Field(default=1, ge=1, le=3)
    metrics_level: int = Field(default=1, ge=1, le=3)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        if v:
            if 'max_requests' not in v or 'window_seconds' not in v:
                raise ValueError("Rate limit must contain max_requests and window_seconds")
            if v['max_requests'] < 1 or v['window_seconds'] < 1:
                raise ValueError("Rate limit values must be positive")
        return v
    
    @validator('retry_config')
    def validate_retry_config(cls, v):
        if v:
            if 'max_retries' not in v or 'delay_seconds' not in v:
                raise ValueError("Retry config must contain max_retries and delay_seconds")
            if v['max_retries'] < 0 or v['delay_seconds'] < 0:
                raise ValueError("Retry config values must be non-negative")
        return v
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v < 0:
            raise ValueError("Timeout must be non-negative")
        return v
    
    @validator('cache_ttl')
    def validate_cache_ttl(cls, v):
        if v is not None and v < 0:
            raise ValueError("Cache TTL must be non-negative")
        return v
    
    @validator('validation_rules')
    def validate_rules(cls, v):
        if v:
            if 'required_fields' not in v or 'field_types' not in v:
                raise ValueError("Validation rules must contain required_fields and field_types")
        return v
    
    @validator('security_level', 'audit_level', 'metrics_level')
    def validate_levels(cls, v):
        if not 1 <= v <= 3:
            raise ValueError("Level must be between 1 and 3")
        return v

class NotificationTemplate(BaseModel):
    """Template for notification messages."""
    id: str = Field(..., min_length=1, max_length=255)
    versions: Dict[int, TemplateVersion] = Field(default_factory=dict)
    parent_template: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_latest_version(self) -> Optional[TemplateVersion]:
        """Get the latest version of the template."""
        if not self.versions:
            return None
        return self.versions[max(self.versions.keys())]
    
    def get_version(self, version: int) -> Optional[TemplateVersion]:
        """Get a specific version of the template."""
        return self.versions.get(version)
    
    def add_version(self, version: TemplateVersion) -> None:
        """Add a new version of the template."""
        self.versions[version.version] = version
        self.updated_at = datetime.utcnow()
    
    def remove_version(self, version: int) -> None:
        """Remove a version of the template."""
        if version in self.versions:
            del self.versions[version]
            self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the template."""
        self.tags.add(tag)
        self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the template."""
        self.tags.discard(tag)
        self.updated_at = datetime.utcnow()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update template metadata."""
        self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()

class RateLimit(BaseModel):
    """Rate limit configuration."""
    id: str = Field(..., min_length=1, max_length=255)
    user_id: Optional[str] = None
    template_id: Optional[str] = None
    max_requests: int = Field(..., gt=0)
    window_seconds: int = Field(..., gt=0)
    current_requests: int = Field(default=0)
    window_start: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        now = datetime.utcnow()
        if (now - self.window_start).total_seconds() >= self.window_seconds:
            self.current_requests = 0
            self.window_start = now
            return False
        return self.current_requests >= self.max_requests
    
    def increment(self) -> None:
        """Increment request count."""
        self.current_requests += 1
        self.updated_at = datetime.utcnow()
    
    def reset(self) -> None:
        """Reset rate limit."""
        self.current_requests = 0
        self.window_start = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_remaining_requests(self) -> int:
        """Get remaining requests in current window."""
        now = datetime.utcnow()
        if (now - self.window_start).total_seconds() >= self.window_seconds:
            return self.max_requests
        return max(0, self.max_requests - self.current_requests)
    
    def get_window_end(self) -> datetime:
        """Get end of current window."""
        return self.window_start + timedelta(seconds=self.window_seconds)

class NotificationDeliveryStatus(str, Enum):
    """Status of notification delivery."""
    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"
    VALIDATION_FAILED = "validation_failed"
    SECURITY_FAILED = "security_failed"

class NotificationDelivery(BaseModel):
    """Notification delivery record."""
    id: str = Field(..., min_length=1, max_length=255)
    notification_id: str = Field(..., min_length=1, max_length=255)
    template_id: str = Field(..., min_length=1, max_length=255)
    template_version: int = Field(..., ge=1)
    user_id: Optional[str] = None
    status: NotificationDeliveryStatus = Field(default=NotificationDeliveryStatus.PENDING)
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    last_attempt: Optional[datetime] = None
    next_attempt: Optional[datetime] = None
    error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    security_level: int = Field(default=1, ge=1, le=3)
    validation_level: int = Field(default=1, ge=1, le=3)
    audit_level: int = Field(default=1, ge=1, le=3)
    metrics_level: int = Field(default=1, ge=1, le=3)
    health_level: int = Field(default=1, ge=1, le=3)
    logging_level: int = Field(default=1, ge=1, le=3)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_id: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    priority: NotificationPriority = Field(default=NotificationPriority.MEDIUM)
    channel: str = Field(default="email")
    retry_strategy: Optional[Dict[str, Any]] = None
    rate_limit: Optional[Dict[str, int]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    security_rules: Optional[Dict[str, Any]] = None
    audit_rules: Optional[Dict[str, Any]] = None
    metrics_rules: Optional[Dict[str, Any]] = None
    health_rules: Optional[Dict[str, Any]] = None
    logging_rules: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str, attempt: int, details: Optional[Dict[str, Any]] = None) -> None:
        """Add error to history with additional details."""
        error_entry = {
            'error': error,
            'attempt': attempt,
            'timestamp': datetime.utcnow(),
            'details': details or {}
        }
        self.error_history.append(error_entry)
        self.error = error
        self.updated_at = datetime.utcnow()
        
        # Record metrics
        self.metrics_service.record_delivery_error(self.id, error, attempt)
        
        # Audit
        self.audit_service.record_audit(
            'delivery_error',
            'error',
            f"Delivery {self.id} failed: {error}",
            details=error_entry
        )
    
    def update_status(self, status: NotificationDeliveryStatus, details: Optional[Dict[str, Any]] = None) -> None:
        """Update delivery status with additional details."""
        old_status = self.status
        self.status = status
        self.updated_at = datetime.utcnow()
        
        # Record metrics
        self.metrics_service.record_delivery_status_change(self.id, old_status, status)
        
        # Audit
        self.audit_service.record_audit(
            'delivery_status_change',
            'info',
            f"Delivery {self.id} status changed from {old_status} to {status}",
            details=details
        )
    
    def increment_attempts(self) -> None:
        """Increment attempt count with validation."""
        if self.attempts >= self.max_attempts:
            self.update_status(NotificationDeliveryStatus.FAILED)
            self.add_error("Max attempts exceeded", self.attempts)
            return
            
        self.attempts += 1
        self.last_attempt = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Record metrics
        self.metrics_service.record_delivery_attempt(self.id, self.attempts)
        
        # Audit
        self.audit_service.record_audit(
            'delivery_attempt',
            'info',
            f"Delivery {self.id} attempt {self.attempts} of {self.max_attempts}"
        )
    
    def set_next_attempt(self, next_attempt: datetime) -> None:
        """Set next attempt time with validation."""
        if next_attempt < datetime.utcnow():
            raise ValueError("Next attempt time must be in the future")
            
        self.next_attempt = next_attempt
        self.updated_at = datetime.utcnow()
        
        # Record metrics
        self.metrics_service.record_delivery_scheduled(self.id, next_attempt)
        
        # Audit
        self.audit_service.record_audit(
            'delivery_scheduled',
            'info',
            f"Delivery {self.id} scheduled for {next_attempt}"
        )
    
    def is_expired(self) -> bool:
        """Check if delivery is expired with validation."""
        if not self.expires_at:
            return False
            
        is_expired = datetime.utcnow() >= self.expires_at
        if is_expired:
            self.update_status(NotificationDeliveryStatus.EXPIRED)
            
        return is_expired
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata with validation."""
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
            
        self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()
        
        # Audit
        self.audit_service.record_audit(
            'delivery_metadata_update',
            'info',
            f"Delivery {self.id} metadata updated"
        )
    
    def add_tag(self, tag: str) -> None:
        """Add tag with validation."""
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("Tag must be a non-empty string")
            
        self.tags.add(tag.strip())
        self.updated_at = datetime.utcnow()
        
        # Audit
        self.audit_service.record_audit(
            'delivery_tag_add',
            'info',
            f"Delivery {self.id} tag added: {tag}"
        )
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag with validation."""
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("Tag must be a non-empty string")
            
        self.tags.discard(tag.strip())
        self.updated_at = datetime.utcnow()
        
        # Audit
        self.audit_service.record_audit(
            'delivery_tag_remove',
            'info',
            f"Delivery {self.id} tag removed: {tag}"
        )
    
    def validate(self) -> bool:
        """Validate delivery record."""
        try:
            # Validate required fields
            if not self.id or not self.notification_id or not self.template_id:
                return False
                
            # Validate version
            if self.template_version < 1:
                return False
                
            # Validate status
            if self.status not in NotificationDeliveryStatus:
                return False
                
            # Validate attempts
            if self.attempts < 0 or self.attempts > self.max_attempts:
                return False
                
            # Validate timestamps
            if self.last_attempt and self.last_attempt > datetime.utcnow():
                return False
            if self.next_attempt and self.next_attempt < datetime.utcnow():
                return False
            if self.expires_at and self.expires_at < datetime.utcnow():
                return False
                
            # Validate levels
            for level in [self.security_level, self.validation_level, self.audit_level,
                         self.metrics_level, self.health_level, self.logging_level]:
                if not 1 <= level <= 3:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating delivery {self.id}: {str(e)}")
            return False

class NotificationService:
    """Service for managing notifications."""
    
    def __init__(
        self,
        notification_manager: NotificationManager,
        rate_limiting_service: RateLimitingService,
        template_service: TemplateService,
        cache_service: CacheService,
        metrics_service: MetricsService,
        retry_service: RetryService,
        error_handling_service: ErrorHandlingService,
        config_service: ConfigService,
        audit_service: AuditService,
        health_service: HealthService,
        transaction_service: TransactionService,
        validation_service: ValidationService,
        security_service: SecurityService,
        logging_service: LoggingService,
        monitoring_service: MonitoringService,
        scheduler_service: SchedulerService,
        queue_service: QueueService,
        worker_service: WorkerService,
        persistence_service: PersistenceService,
        dependency_service: DependencyService,
        recovery_service: RecoveryService,
        state_service: StateService
    ):
        """Initialize notification service."""
        self.notification_manager = notification_manager
        self.rate_limiting_service = rate_limiting_service
        self.template_service = template_service
        self.cache_service = cache_service
        self.metrics_service = metrics_service
        self.retry_service = retry_service
        self.error_handling_service = error_handling_service
        self.config_service = config_service
        self.audit_service = audit_service
        self.health_service = health_service
        self.transaction_service = transaction_service
        self.validation_service = validation_service
        self.security_service = security_service
        self.logging_service = logging_service
        self.monitoring_service = monitoring_service
        self.scheduler_service = scheduler_service
        self.queue_service = queue_service
        self.worker_service = worker_service
        self.persistence_service = persistence_service
        self.dependency_service = dependency_service
        self.recovery_service = recovery_service
        self.state_service = state_service
        
        # Initialize internal state
        self._templates: Dict[str, NotificationTemplate] = {}
        self._rate_limits: Dict[str, RateLimit] = {}
        self._deliveries: Dict[str, NotificationDelivery] = {}
        self._running = False
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._template_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._delivery_lock = asyncio.Lock()
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize security
        self._init_security()
        
        # Initialize logging
        self._init_logging()
        
        # Load templates
        self._load_templates()
        
        # Load rate limits
        self._load_rate_limits()
        
        # Schedule tasks
        self._template_reload_task = self.scheduler_service.schedule_task(
            self._reload_templates(),
            interval=300  # 5 minutes
        )
        self._rate_limit_cleanup_task = self.scheduler_service.schedule_task(
            self._cleanup_rate_limits(),
            interval=60  # 1 minute
        )
        self._delivery_cleanup_task = self.scheduler_service.schedule_task(
            self._cleanup_deliveries(),
            interval=300  # 5 minutes
        )
        self._health_check_task = self.scheduler_service.schedule_task(
            self._check_health(),
            interval=60  # 1 minute
        )
        
        # Record metrics
        self.metrics_service.record_service_init()
        
        # Audit
        self.audit_service.record_audit(
            'service_init',
            'info',
            "Notification service initialized"
        )
        
        logger.info("Notification service initialized")
    
    def _init_metrics(self):
        """Initialize metrics tracking."""
        try:
            # Register metrics
            self.metrics_service.register_metric(
                'notification_sent',
                'counter',
                'Number of notifications sent'
            )
            self.metrics_service.register_metric(
                'notification_failed',
                'counter',
                'Number of failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_delivered',
                'counter',
                'Number of delivered notifications'
            )
            self.metrics_service.register_metric(
                'notification_retried',
                'counter',
                'Number of retried notifications'
            )
            self.metrics_service.register_metric(
                'notification_rate_limited',
                'counter',
                'Number of rate limited notifications'
            )
            self.metrics_service.register_metric(
                'notification_expired',
                'counter',
                'Number of expired notifications'
            )
            self.metrics_service.register_metric(
                'notification_validation_failed',
                'counter',
                'Number of validation failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_security_failed',
                'counter',
                'Number of security failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_audit_failed',
                'counter',
                'Number of audit failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_metrics_failed',
                'counter',
                'Number of metrics failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_health_failed',
                'counter',
                'Number of health failed notifications'
            )
            self.metrics_service.register_metric(
                'notification_logging_failed',
                'counter',
                'Number of logging failed notifications'
            )
            
            # Record metrics
            self.metrics_service.record_metrics_init()
            
            # Audit
            self.audit_service.record_audit(
                'metrics_init',
                'info',
                "Metrics initialized"
            )
            
        except Exception as e:
            logger.error(f"Error initializing metrics: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_metrics_init_error()
            
            # Audit
            self.audit_service.record_audit(
                'metrics_init',
                'error',
                str(e)
            )
            
            raise
    
    def _init_security(self):
        """Initialize security settings."""
        try:
            # Register security rules
            self.security_service.register_rule(
                'notification_send',
                {
                    'required_permissions': ['notification.send'],
                    'rate_limit': {
                        'max_requests': 100,
                        'window_seconds': 60
                    },
                    'validation': {
                        'required_fields': ['template_id', 'template_vars'],
                        'field_types': {
                            'template_id': 'string',
                            'template_vars': 'object'
                        }
                    }
                }
            )
            self.security_service.register_rule(
                'notification_read',
                {
                    'required_permissions': ['notification.read'],
                    'rate_limit': {
                        'max_requests': 1000,
                        'window_seconds': 60
                    }
                }
            )
            self.security_service.register_rule(
                'notification_manage',
                {
                    'required_permissions': ['notification.manage'],
                    'rate_limit': {
                        'max_requests': 50,
                        'window_seconds': 60
                    }
                }
            )
            
            # Record metrics
            self.metrics_service.record_security_init()
            
            # Audit
            self.audit_service.record_audit(
                'security_init',
                'info',
                "Security initialized"
            )
            
        except Exception as e:
            logger.error(f"Error initializing security: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_security_init_error()
            
            # Audit
            self.audit_service.record_audit(
                'security_init',
                'error',
                str(e)
            )
            
            raise
    
    def _init_logging(self):
        """Initialize logging settings."""
        try:
            # Configure logging
            self.logging_service.configure(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('notification.log')
                ]
            )
            
            # Record metrics
            self.metrics_service.record_logging_init()
            
            # Audit
            self.audit_service.record_audit(
                'logging_init',
                'info',
                "Logging initialized"
            )
            
        except Exception as e:
            logger.error(f"Error initializing logging: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_logging_init_error()
            
            # Audit
            self.audit_service.record_audit(
                'logging_init',
                'error',
                str(e)
            )
            
            raise
    
    def _load_templates(self):
        """Load notification templates from config."""
        templates_config = self.config_service.get('notification.templates', {})
        for template_id, template_data in templates_config.items():
            try:
                # Validate template data
                self.validation_service.validate_template(template_data)
                
                # Create template
                template = NotificationTemplate(id=template_id)
                
                # Add versions
                for version_data in template_data.get('versions', []):
                    version = TemplateVersion(**version_data)
                    template.add_version(version)
                
                # Add tags
                for tag in template_data.get('tags', []):
                    template.add_tag(tag)
                
                # Add metadata
                template.update_metadata(template_data.get('metadata', {}))
                
                # Store template
                self._templates[template_id] = template
                
                # Initialize metrics
                self.metrics_service.register_template_metrics(template_id)
                
                # Set up security
                version = template.get_latest_version()
                if version:
                    self.security_service.set_template_policy(
                        template_id,
                        version.security_level
                    )
                
                # Set up logging
                if version:
                    self.logging_service.set_template_logging(
                        template_id,
                        version.audit_level
                    )
                
            except Exception as e:
                logger.error(f"Error loading template {template_id}: {str(e)}")
                continue
    
    def _load_rate_limits(self):
        """Load rate limits from config."""
        rate_limits_config = self.config_service.get('notification.rate_limits', {})
        for limit_id, limit_data in rate_limits_config.items():
            try:
                # Validate rate limit data
                self.validation_service.validate_rate_limit(limit_data)
                
                # Create rate limit
                rate_limit = RateLimit(
                    id=limit_id,
                    **limit_data
                )
                
                # Store rate limit
                self._rate_limits[limit_id] = rate_limit
                
            except Exception as e:
                logger.error(f"Error loading rate limit {limit_id}: {str(e)}")
                continue
    
    async def _reload_templates(self):
        """Reload notification templates."""
        try:
            # Load templates
            self._load_templates()
            
            # Record metrics
            self.metrics_service.record_template_reload()
            
            # Audit
            await self.audit_service.record_audit(
                'template_reload',
                'success',
                "Reloaded notification templates"
            )
            
        except Exception as e:
            logger.error(f"Error reloading templates: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_template_reload_error()
            
            # Audit
            await self.audit_service.record_audit(
                'template_reload',
                'error',
                str(e)
            )
    
    async def _cleanup_rate_limits(self):
        """Clean up expired rate limits with enhanced error handling."""
        try:
            # Get current time
            now = datetime.utcnow()
            
            # Find expired rate limits
            async with self._rate_limit_lock:
                expired = [
                    (id_, limit) for id_, limit in self._rate_limits.items()
                    if (now - limit.window_start).total_seconds() >= limit.window_seconds
                ]
                
                # Remove expired limits
                for id_, _ in expired:
                    del self._rate_limits[id_]
            
            # Record metrics
            self.metrics_service.record_rate_limit_cleanup(len(expired))
            
            # Audit
            await self.audit_service.record_audit(
                'rate_limit_cleanup',
                'info',
                f"Cleaned up {len(expired)} expired rate limits"
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up rate limits: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_rate_limit_cleanup_error()
            
            # Audit
            await self.audit_service.record_audit(
                'rate_limit_cleanup_error',
                'error',
                str(e)
            )
            
            raise
    
    async def _cleanup_deliveries(self):
        """Clean up expired deliveries with enhanced error handling."""
        try:
            # Get current time
            now = datetime.utcnow()
            
            # Find expired deliveries
            async with self._delivery_lock:
                expired = [
                    (id_, delivery) for id_, delivery in self._deliveries.items()
                    if delivery.is_expired()
                ]
                
                # Remove expired deliveries
                for id_, _ in expired:
                    del self._deliveries[id_]
            
            # Record metrics
            self.metrics_service.record_delivery_cleanup(len(expired))
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_cleanup',
                'info',
                f"Cleaned up {len(expired)} expired deliveries"
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up deliveries: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_cleanup_error()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_cleanup_error',
                'error',
                str(e)
            )
            
            raise
    
    async def _check_health(self):
        """Check service health with enhanced monitoring."""
        try:
            # Get health status
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow(),
                'metrics': {},
                'errors': [],
                'warnings': []
            }
            
            # Check templates
            async with self._template_lock:
                template_count = len(self._templates)
                if template_count == 0:
                    health_status['warnings'].append('No templates loaded')
                health_status['metrics']['template_count'] = template_count
            
            # Check rate limits
            async with self._rate_limit_lock:
                rate_limit_count = len(self._rate_limits)
                health_status['metrics']['rate_limit_count'] = rate_limit_count
            
            # Check deliveries
            async with self._delivery_lock:
                delivery_count = len(self._deliveries)
                failed_deliveries = sum(
                    1 for d in self._deliveries.values()
                    if d.status == NotificationDeliveryStatus.FAILED
                )
                retrying_deliveries = sum(
                    1 for d in self._deliveries.values()
                    if d.status == NotificationDeliveryStatus.RETRYING
                )
                health_status['metrics'].update({
                    'delivery_count': delivery_count,
                    'failed_deliveries': failed_deliveries,
                    'retrying_deliveries': retrying_deliveries
                })
                
                if failed_deliveries > 0:
                    health_status['warnings'].append(
                        f'{failed_deliveries} failed deliveries'
                    )
            
            # Check service dependencies
            for service in [
                self.notification_manager,
                self.rate_limiting_service,
                self.template_service,
                self.cache_service,
                self.metrics_service,
                self.retry_service,
                self.error_handling_service,
                self.config_service,
                self.audit_service,
                self.health_service,
                self.transaction_service,
                self.validation_service,
                self.security_service,
                self.logging_service,
                self.monitoring_service,
                self.scheduler_service,
                self.queue_service,
                self.worker_service,
                self.persistence_service,
                self.dependency_service,
                self.recovery_service,
                self.state_service
            ]:
                try:
                    service_health = await service.get_health()
                    if service_health['status'] != 'healthy':
                        health_status['warnings'].append(
                            f'Service {service.__class__.__name__} unhealthy'
                        )
                except Exception as e:
                    health_status['errors'].append(
                        f'Error checking {service.__class__.__name__}: {str(e)}'
                    )
            
            # Update overall status
            if health_status['errors']:
                health_status['status'] = 'unhealthy'
            elif health_status['warnings']:
                health_status['status'] = 'degraded'
            
            # Record metrics
            self.metrics_service.record_health_check(health_status)
            
            # Audit
            await self.audit_service.record_audit(
                'health_check',
                health_status['status'],
                f"Health check completed: {health_status['status']}",
                details=health_status
            )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking health: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_health_check_error()
            
            # Audit
            await self.audit_service.record_audit(
                'health_check_error',
                'error',
                str(e)
            )
            
            raise
    
    async def _delivery_worker(self, batch: List[NotificationDelivery]):
        """Process a batch of notification deliveries."""
        try:
            # Process each delivery
            for delivery in batch:
                try:
                    # Get template
                    template = await self._get_template(delivery.template_id)
                    if not template:
                        raise ValueError(f"Template {delivery.template_id} not found")
                    
                    # Get version
                    version = template.get_version(delivery.template_version)
                    if not version:
                        raise ValueError(f"Version {delivery.template_version} not found")
                    
                    # Check rate limit
                    if not await self._check_rate_limit(
                        delivery.user_id,
                        delivery.template_id,
                        version.rate_limit
                    ):
                        delivery.update_status(NotificationDeliveryStatus.RATE_LIMITED)
                        await self._update_delivery(delivery)
                        continue
                    
                    # Send notification
                    await self._send_notification_with_retry(delivery, version)
                    
                except Exception as e:
                    # Handle delivery failure
                    await self._handle_delivery_failure(delivery, str(e))
            
            # Record metrics
            self.metrics_service.record_delivery_batch(len(batch))
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_batch',
                'success',
                f"Processed {len(batch)} deliveries"
            )
            
        except Exception as e:
            logger.error(f"Error processing delivery batch: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_batch_error()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_batch',
                'error',
                str(e)
            )
    
    async def _get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get a notification template."""
        try:
            # Check cache
            template = self.cache_service.get_template(template_id)
            if template:
                return template
            
            # Get template
            template = self._templates.get(template_id)
            if template:
                # Cache template
                self.cache_service.set_template(template_id, template)
            
            return template
            
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {str(e)}")
            return None
    
    async def _check_rate_limit(
        self,
        user_id: str,
        template_id: str,
        rate_limit: Optional[Dict[str, int]]
    ) -> bool:
        """Check if rate limit is exceeded."""
        try:
            if not rate_limit:
                return True
            
            # Get rate limit
            limit_id = f"{user_id}:{template_id}"
            rate_limit_obj = self._rate_limits.get(limit_id)
            
            if not rate_limit_obj:
                # Create rate limit
                rate_limit_obj = RateLimit(
                    id=limit_id,
                    user_id=user_id,
                    template_id=template_id,
                    max_requests=rate_limit['max_requests'],
                    window_seconds=rate_limit['window_seconds']
                )
                self._rate_limits[limit_id] = rate_limit_obj
            
            # Check rate limit
            if rate_limit_obj.is_exceeded():
                return False
            
            # Increment rate limit
            rate_limit_obj.increment()
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return False
    
    async def _send_notification(
        self,
        template_id: str,
        template_vars: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Send a notification with enhanced error handling and security."""
        try:
            # Validate inputs
            if not template_id or not template_vars:
                raise ValueError("Template ID and variables are required")
            
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Get latest version
            version = template.get_latest_version()
            if not version:
                raise ValueError(f"No version found for template {template_id}")
            
            # Check rate limit
            if not await self._check_rate_limit(user_id, template_id, version.rate_limit):
                raise ValueError("Rate limit exceeded")
            
            # Create delivery record
            delivery = NotificationDelivery(
                id=str(uuid4()),
                notification_id=str(uuid4()),
                template_id=template_id,
                template_version=version.version,
                user_id=user_id,
                status=NotificationDeliveryStatus.PENDING,
                security_level=version.security_level,
                validation_level=version.validation_level,
                audit_level=version.audit_level,
                metrics_level=version.metrics_level,
                health_level=version.health_level,
                logging_level=version.logging_level,
                retry_strategy=version.retry_config,
                rate_limit=version.rate_limit,
                validation_rules=version.validation_rules,
                priority=version.priority,
                channel=version.notification_type
            )
            
            # Validate delivery
            if not delivery.validate():
                raise ValueError("Invalid delivery record")
            
            # Store delivery
            async with self._delivery_lock:
                self._deliveries[delivery.id] = delivery
            
            # Send notification
            await self._send_notification_with_retry(delivery, version)
            
            # Record metrics
            self.metrics_service.record_notification_sent(delivery.id)
            
            # Audit
            await self.audit_service.record_audit(
                'notification_sent',
                'info',
                f"Notification sent: {delivery.id}",
                details={
                    'template_id': template_id,
                    'user_id': user_id,
                    'version': version.version
                }
            )
            
            return delivery.id
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_notification_failed(template_id)
            
            # Audit
            await self.audit_service.record_audit(
                'notification_send_error',
                'error',
                str(e),
                details={
                    'template_id': template_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def _send_notification_with_retry(
        self,
        delivery: NotificationDelivery,
        version: TemplateVersion
    ) -> None:
        """Send notification with retry logic."""
        try:
            # Get retry config
            retry_config = delivery.retry_strategy or {
                'max_retries': 3,
                'delay_seconds': 5
            }
            
            # Define send function
            async def send():
                # Get notification
                notification = await self.notification_manager.create_notification(
                    template_id=delivery.template_id,
                    template_vars=version.template_vars,
                    user_id=delivery.user_id,
                    priority=version.priority,
                    notification_type=version.notification_type
                )
                
                # Send notification
                await self.notification_manager.send_notification(notification)
                
                # Update delivery
                delivery.update_status(NotificationDeliveryStatus.DELIVERED)
                
                # Record metrics
                self.metrics_service.record_notification_delivered(delivery.id)
                
                # Audit
                await self.audit_service.record_audit(
                    'notification_delivered',
                    'info',
                    f"Notification delivered: {delivery.id}"
                )
            
            # Execute with retry
            await self.retry_service.execute_with_retry(
                send,
                max_retries=retry_config['max_retries'],
                delay_seconds=retry_config['delay_seconds'],
                on_retry=lambda attempt: delivery.increment_attempts()
            )
            
        except Exception as e:
            logger.error(f"Error sending notification with retry: {str(e)}")
            
            # Handle failure
            await self._handle_delivery_failure(delivery, str(e))
            
            raise
    
    async def _handle_delivery_failure(
        self,
        delivery: NotificationDelivery,
        error: str
    ) -> None:
        """Handle delivery failure with enhanced error handling."""
        try:
            # Update delivery status
            delivery.update_status(NotificationDeliveryStatus.FAILED)
            
            # Add error
            delivery.add_error(error, delivery.attempts)
            
            # Record metrics
            self.metrics_service.record_notification_failed(delivery.id)
            
            # Audit
            await self.audit_service.record_audit(
                'notification_failed',
                'error',
                f"Notification failed: {delivery.id}",
                details={'error': error}
            )
            
            # Check if should retry
            if delivery.attempts < delivery.max_attempts:
                # Calculate next attempt
                next_attempt = datetime.utcnow() + timedelta(
                    seconds=delivery.retry_strategy['delay_seconds']
                )
                
                # Update delivery
                delivery.set_next_attempt(next_attempt)
                delivery.update_status(NotificationDeliveryStatus.RETRYING)
                
                # Schedule retry
                self.scheduler_service.schedule_task(
                    self._send_notification_with_retry(
                        delivery,
                        await self._get_template(delivery.template_id)
                    ),
                    run_at=next_attempt
                )
                
                # Record metrics
                self.metrics_service.record_notification_retried(delivery.id)
                
                # Audit
                await self.audit_service.record_audit(
                    'notification_retry_scheduled',
                    'info',
                    f"Notification retry scheduled: {delivery.id}",
                    details={'next_attempt': next_attempt}
                )
            
        except Exception as e:
            logger.error(f"Error handling delivery failure: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_failure_handling_error(delivery.id)
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_failure_handling_error',
                'error',
                str(e),
                details={'delivery_id': delivery.id}
            )
            
            raise
    
    async def _update_delivery(self, delivery: NotificationDelivery) -> None:
        """Update delivery record."""
        try:
            # Update delivery
            await self.persistence_service.update_delivery(delivery)
            
            # Record metrics
            self.metrics_service.record_delivery_update()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_update',
                'success',
                f"Updated delivery {delivery.id}"
            )
            
        except Exception as e:
            logger.error(f"Error updating delivery {delivery.id}: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_update_error()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_update',
                'error',
                str(e)
            )
    
    async def notify_task_created(self, task_id: str, task_name: str, user_id: str) -> str:
        """Notify task creation with enhanced error handling."""
        try:
            # Validate inputs
            if not task_id or not task_name or not user_id:
                raise ValueError("Task ID, name, and user ID are required")
            
            # Send notification
            return await self._send_notification(
                template_id='task_created',
                template_vars={
                    'task_id': task_id,
                    'task_name': task_name,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying task creation: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_task_notification_error('created')
            
            # Audit
            await self.audit_service.record_audit(
                'task_notification_error',
                'error',
                str(e),
                details={
                    'action': 'created',
                    'task_id': task_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def notify_task_updated(self, task_id: str, task_name: str, user_id: str) -> str:
        """Notify task update with enhanced error handling."""
        try:
            # Validate inputs
            if not task_id or not task_name or not user_id:
                raise ValueError("Task ID, name, and user ID are required")
            
            # Send notification
            return await self._send_notification(
                template_id='task_updated',
                template_vars={
                    'task_id': task_id,
                    'task_name': task_name,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying task update: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_task_notification_error('updated')
            
            # Audit
            await self.audit_service.record_audit(
                'task_notification_error',
                'error',
                str(e),
                details={
                    'action': 'updated',
                    'task_id': task_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def notify_task_deleted(self, task_id: str, task_name: str, user_id: str) -> str:
        """Notify task deletion with enhanced error handling."""
        try:
            # Validate inputs
            if not task_id or not task_name or not user_id:
                raise ValueError("Task ID, name, and user ID are required")
            
            # Send notification
            return await self._send_notification(
                template_id='task_deleted',
                template_vars={
                    'task_id': task_id,
                    'task_name': task_name,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying task deletion: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_task_notification_error('deleted')
            
            # Audit
            await self.audit_service.record_audit(
                'task_notification_error',
                'error',
                str(e),
                details={
                    'action': 'deleted',
                    'task_id': task_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def notify_task_execution_started(self, task_id: str, task_name: str, user_id: str) -> str:
        """Notify task execution start with enhanced error handling."""
        try:
            # Validate inputs
            if not task_id or not task_name or not user_id:
                raise ValueError("Task ID, name, and user ID are required")
            
            # Send notification
            return await self._send_notification(
                template_id='task_execution_started',
                template_vars={
                    'task_id': task_id,
                    'task_name': task_name,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying task execution start: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_task_notification_error('execution_started')
            
            # Audit
            await self.audit_service.record_audit(
                'task_notification_error',
                'error',
                str(e),
                details={
                    'action': 'execution_started',
                    'task_id': task_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def notify_task_execution_stopped(self, task_id: str, task_name: str, user_id: str) -> str:
        """Notify task execution stop with enhanced error handling."""
        try:
            # Validate inputs
            if not task_id or not task_name or not user_id:
                raise ValueError("Task ID, name, and user ID are required")
            
            # Send notification
            return await self._send_notification(
                template_id='task_execution_stopped',
                template_vars={
                    'task_id': task_id,
                    'task_name': task_name,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying task execution stop: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_task_notification_error('execution_stopped')
            
            # Audit
            await self.audit_service.record_audit(
                'task_notification_error',
                'error',
                str(e),
                details={
                    'action': 'execution_stopped',
                    'task_id': task_id,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def notify_system_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Notify system metrics with enhanced error handling."""
        try:
            # Validate inputs
            if not metrics:
                raise ValueError("Metrics are required")
            
            # Validate metrics
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid metric value for {key}")
                if value < 0 or value > 100:
                    raise ValueError(f"Metric value for {key} must be between 0 and 100")
            
            # Get threshold config
            thresholds = {
                'cpu_usage': 80,
                'memory_usage': 80,
                'disk_usage': 80,
                'network_usage': 80
            }
            
            # Check thresholds
            notifications = []
            for metric, value in metrics.items():
                if value >= thresholds.get(metric, 80):
                    # Send notification
                    notification_id = await self._send_notification(
                        template_id='system_metric_alert',
                        template_vars={
                            'metric': metric,
                            'value': value,
                            'threshold': thresholds.get(metric, 80),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    )
                    notifications.append(notification_id)
            
            # Record metrics
            self.metrics_service.record_system_metrics_notification(len(notifications))
            
            # Audit
            await self.audit_service.record_audit(
                'system_metrics_notification',
                'info',
                f"Sent {len(notifications)} system metric notifications",
                details={'metrics': metrics}
            )
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error notifying system metrics: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_system_metrics_notification_error()
            
            # Audit
            await self.audit_service.record_audit(
                'system_metrics_notification_error',
                'error',
                str(e),
                details={'metrics': metrics}
            )
            
            raise
    
    async def notify_agent_status(self, agent_id: str, agent_name: str, status: str) -> str:
        """Notify agent status with enhanced error handling."""
        try:
            # Validate inputs
            if not agent_id or not agent_name or not status:
                raise ValueError("Agent ID, name, and status are required")
            
            # Validate status
            valid_statuses = ['online', 'offline', 'busy', 'error']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
            
            # Send notification
            return await self._send_notification(
                template_id='agent_status',
                template_vars={
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'status': status,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error notifying agent status: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_agent_notification_error()
            
            # Audit
            await self.audit_service.record_audit(
                'agent_notification_error',
                'error',
                str(e),
                details={
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'status': status
                }
            )
            
            raise
    
    async def notify_model_training(self, model_id: str, model_name: str, status: str) -> str:
        """Notify model training status with enhanced error handling."""
        try:
            # Validate inputs
            if not model_id or not model_name or not status:
                raise ValueError("Model ID, name, and status are required")
            
            # Validate status
            valid_statuses = ['started', 'completed', 'failed', 'cancelled']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
            
            # Send notification
            return await self._send_notification(
                template_id='model_training',
                template_vars={
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': status,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error notifying model training: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_model_notification_error()
            
            # Audit
            await self.audit_service.record_audit(
                'model_notification_error',
                'error',
                str(e),
                details={
                    'model_id': model_id,
                    'model_name': model_name,
                    'status': status
                }
            )
            
            raise
    
    async def notify_user_activity(self, user_id: str, activity: str) -> str:
        """Notify user activity with enhanced error handling."""
        try:
            # Validate inputs
            if not user_id or not activity:
                raise ValueError("User ID and activity are required")
            
            # Validate activity
            valid_activities = ['login', 'logout', 'password_change', 'profile_update']
            if activity not in valid_activities:
                raise ValueError(f"Invalid activity. Must be one of: {valid_activities}")
            
            # Send notification
            return await self._send_notification(
                template_id='user_activity',
                template_vars={
                    'user_id': user_id,
                    'activity': activity,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying user activity: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_user_notification_error()
            
            # Audit
            await self.audit_service.record_audit(
                'user_notification_error',
                'error',
                str(e),
                details={
                    'user_id': user_id,
                    'activity': activity
                }
            )
            
            raise
    
    async def notify_error(self, error_message: str, user_id: Optional[str] = None) -> str:
        """Notify error with enhanced error handling."""
        try:
            # Validate inputs
            if not error_message:
                raise ValueError("Error message is required")
            
            # Send notification
            return await self._send_notification(
                template_id='error',
                template_vars={
                    'error_message': error_message,
                    'timestamp': datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error notifying error: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_error_notification_error()
            
            # Audit
            await self.audit_service.record_audit(
                'error_notification_error',
                'error',
                str(e),
                details={
                    'error_message': error_message,
                    'user_id': user_id
                }
            )
            
            raise
    
    async def get_delivery_status(self, notification_id: str) -> Optional[NotificationDelivery]:
        """Get delivery status with enhanced error handling."""
        try:
            # Validate inputs
            if not notification_id:
                raise ValueError("Notification ID is required")
            
            # Get delivery
            async with self._delivery_lock:
                delivery = self._deliveries.get(notification_id)
            
            if not delivery:
                # Record metrics
                self.metrics_service.record_delivery_status_not_found(notification_id)
                
                # Audit
                await self.audit_service.record_audit(
                    'delivery_status_not_found',
                    'warning',
                    f"Delivery not found: {notification_id}"
                )
                
                return None
            
            # Validate delivery
            if not delivery.validate():
                # Record metrics
                self.metrics_service.record_delivery_status_invalid(notification_id)
                
                # Audit
                await self.audit_service.record_audit(
                    'delivery_status_invalid',
                    'error',
                    f"Invalid delivery: {notification_id}"
                )
                
                return None
            
            # Record metrics
            self.metrics_service.record_delivery_status_retrieved(notification_id)
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_status_retrieved',
                'info',
                f"Retrieved delivery status: {notification_id}",
                details={'status': delivery.status}
            )
            
            return delivery
            
        except Exception as e:
            logger.error(f"Error getting delivery status: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_status_error(notification_id)
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_status_error',
                'error',
                str(e),
                details={'notification_id': notification_id}
            )
            
            raise
    
    async def get_delivery_metrics(self) -> Dict[str, Any]:
        """Get delivery metrics with enhanced error handling."""
        try:
            # Initialize metrics
            metrics = {
                'total_deliveries': 0,
                'delivered': 0,
                'failed': 0,
                'retrying': 0,
                'pending': 0,
                'expired': 0,
                'rate_limited': 0,
                'validation_failed': 0,
                'security_failed': 0,
                'audit_failed': 0,
                'metrics_failed': 0,
                'health_failed': 0,
                'logging_failed': 0,
                'average_attempts': 0,
                'success_rate': 0,
                'error_rate': 0,
                'retry_rate': 0,
                'expiration_rate': 0,
                'rate_limit_rate': 0,
                'validation_failure_rate': 0,
                'security_failure_rate': 0,
                'audit_failure_rate': 0,
                'metrics_failure_rate': 0,
                'health_failure_rate': 0,
                'logging_failure_rate': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Get deliveries
            async with self._delivery_lock:
                deliveries = list(self._deliveries.values())
            
            # Calculate metrics
            if deliveries:
                metrics['total_deliveries'] = len(deliveries)
                metrics['delivered'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.DELIVERED
                )
                metrics['failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.FAILED
                )
                metrics['retrying'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.RETRYING
                )
                metrics['pending'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.PENDING
                )
                metrics['expired'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.EXPIRED
                )
                metrics['rate_limited'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.RATE_LIMITED
                )
                metrics['validation_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.VALIDATION_FAILED
                )
                metrics['security_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.SECURITY_FAILED
                )
                metrics['audit_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.AUDIT_FAILED
                )
                metrics['metrics_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.METRICS_FAILED
                )
                metrics['health_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.HEALTH_FAILED
                )
                metrics['logging_failed'] = sum(
                    1 for d in deliveries
                    if d.status == NotificationDeliveryStatus.LOGGING_FAILED
                )
                
                # Calculate rates
                total = metrics['total_deliveries']
                metrics['average_attempts'] = sum(d.attempts for d in deliveries) / total
                metrics['success_rate'] = metrics['delivered'] / total
                metrics['error_rate'] = metrics['failed'] / total
                metrics['retry_rate'] = metrics['retrying'] / total
                metrics['expiration_rate'] = metrics['expired'] / total
                metrics['rate_limit_rate'] = metrics['rate_limited'] / total
                metrics['validation_failure_rate'] = metrics['validation_failed'] / total
                metrics['security_failure_rate'] = metrics['security_failed'] / total
                metrics['audit_failure_rate'] = metrics['audit_failed'] / total
                metrics['metrics_failure_rate'] = metrics['metrics_failed'] / total
                metrics['health_failure_rate'] = metrics['health_failed'] / total
                metrics['logging_failure_rate'] = metrics['logging_failed'] / total
            
            # Record metrics
            self.metrics_service.record_delivery_metrics_retrieved()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_metrics_retrieved',
                'info',
                "Retrieved delivery metrics",
                details=metrics
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting delivery metrics: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_delivery_metrics_error()
            
            # Audit
            await self.audit_service.record_audit(
                'delivery_metrics_error',
                'error',
                str(e)
            )
            
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health with enhanced error handling."""
        try:
            # Get health status
            health_status = await self._check_health()
            
            # Add service info
            health_status.update({
                'service': 'notification',
                'version': '1.0.0',
                'uptime': (datetime.utcnow() - self.created_at).total_seconds(),
                'dependencies': {
                    'notification_manager': await self.notification_manager.get_health(),
                    'rate_limiting_service': await self.rate_limiting_service.get_health(),
                    'template_service': await self.template_service.get_health(),
                    'cache_service': await self.cache_service.get_health(),
                    'metrics_service': await self.metrics_service.get_health(),
                    'retry_service': await self.retry_service.get_health(),
                    'error_handling_service': await self.error_handling_service.get_health(),
                    'config_service': await self.config_service.get_health(),
                    'audit_service': await self.audit_service.get_health(),
                    'health_service': await self.health_service.get_health(),
                    'transaction_service': await self.transaction_service.get_health(),
                    'validation_service': await self.validation_service.get_health(),
                    'security_service': await self.security_service.get_health(),
                    'logging_service': await self.logging_service.get_health(),
                    'monitoring_service': await self.monitoring_service.get_health(),
                    'scheduler_service': await self.scheduler_service.get_health(),
                    'queue_service': await self.queue_service.get_health(),
                    'worker_service': await self.worker_service.get_health(),
                    'persistence_service': await self.persistence_service.get_health(),
                    'dependency_service': await self.dependency_service.get_health(),
                    'recovery_service': await self.recovery_service.get_health(),
                    'state_service': await self.state_service.get_health()
                }
            })
            
            # Record metrics
            self.metrics_service.record_health_retrieved()
            
            # Audit
            await self.audit_service.record_audit(
                'health_retrieved',
                'info',
                "Retrieved service health",
                details=health_status
            )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting service health: {str(e)}")
            
            # Record metrics
            self.metrics_service.record_health_error()
            
            # Audit
            await self.audit_service.record_audit(
                'health_error',
                'error',
                str(e)
            )
            
            raise 