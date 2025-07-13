import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from automation.config.notification_config import NotificationConfig
from automation.notifications.notification_manager import NotificationManager
from automation.services.audit_service import AuditService
from automation.services.cache_service import CacheService
from automation.services.database_service import DatabaseService
from automation.services.dependency_service import DependencyService
from automation.services.error_handling_service import ErrorHandlingService
from automation.services.health_service import HealthService
from automation.services.logging_service import LoggingService
from automation.services.metrics_service import MetricsService
from automation.services.monitoring_service import MonitoringService
from automation.services.persistence_service import PersistenceService
from automation.services.queue_service import QueueService
from automation.services.recovery_service import RecoveryService
from automation.services.retry_service import RetryService
from automation.services.scheduler_service import SchedulerService
from automation.services.security_service import SecurityService
from automation.services.state_service import StateService
from automation.services.transaction_service import TransactionService
from automation.services.validation_service import ValidationService
from automation.services.worker_service import WorkerService
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleanupStatus(str, Enum):
    """Status of cleanup operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMING = "resuming"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    SECURITY_FAILED = "security_failed"


class CleanupPriority(int, Enum):
    """Priority of cleanup operation."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class NotificationTTL(BaseModel):
    """Time-to-live configuration for notifications."""

    default_ttl_hours: int = Field(default=24, description="Default TTL in hours")
    priority_ttl_hours: Dict[str, int] = Field(default_factory=dict, description="TTL by priority")
    type_ttl_hours: Dict[str, int] = Field(default_factory=dict, description="TTL by notification type")
    user_ttl_hours: Dict[str, int] = Field(default_factory=dict, description="TTL by user preference")
    max_ttl_hours: int = Field(default=168, description="Maximum TTL in hours (1 week)")
    min_ttl_hours: int = Field(default=1, description="Minimum TTL in hours")

    @validator("default_ttl_hours", "max_ttl_hours", "min_ttl_hours")
    def validate_ttl_hours(cls, v):
        if v <= 0:
            raise ValueError("TTL hours must be positive")
        return v

    @validator("max_ttl_hours")
    def validate_max_ttl(cls, v, values):
        if "min_ttl_hours" in values and v < values["min_ttl_hours"]:
            raise ValueError("max_ttl_hours must be greater than min_ttl_hours")
        return v

    def get_ttl_for_notification(self, priority: str = None, notification_type: str = None, user_id: str = None) -> int:
        """Get TTL in hours for a specific notification."""
        # Check user-specific TTL first
        if user_id and user_id in self.user_ttl_hours:
            ttl = self.user_ttl_hours[user_id]
        # Check type-specific TTL
        elif notification_type and notification_type in self.type_ttl_hours:
            ttl = self.type_ttl_hours[notification_type]
        # Check priority-specific TTL
        elif priority and priority in self.priority_ttl_hours:
            ttl = self.priority_ttl_hours[priority]
        else:
            ttl = self.default_ttl_hours

        # Ensure TTL is within bounds
        return max(self.min_ttl_hours, min(self.max_ttl_hours, ttl))


class CleanupOperation(BaseModel):
    """Cleanup operation record."""

    id: str = Field(..., min_length=1, max_length=255)
    status: CleanupStatus = Field(default=CleanupStatus.PENDING)
    priority: CleanupPriority = Field(default=CleanupPriority.MEDIUM)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_users: int = Field(default=0)
    processed_users: int = Field(default=0)
    total_notifications: int = Field(default=0)
    deleted_notifications: int = Field(default=0)
    error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    ttl_config: Optional[NotificationTTL] = None

    def add_error(self, error: str) -> None:
        """Add error to history."""
        self.error_history.append({"error": error, "timestamp": datetime.utcnow()})
        self.error = error
        self.updated_at = datetime.utcnow()

    def add_metric(self, name: str, value: Any) -> None:
        """Add metric."""
        self.metrics[name] = value
        self.updated_at = datetime.utcnow()

    def update_status(self, status: CleanupStatus) -> None:
        """Update operation status."""
        self.status = status
        self.updated_at = datetime.utcnow()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to operation."""
        self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if operation is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at


class UserBatch(BaseModel):
    """User batch for processing."""

    id: str = Field(..., min_length=1, max_length=255)
    user_ids: List[str] = Field(..., min_items=1)
    priority: CleanupPriority = Field(default=CleanupPriority.MEDIUM)
    status: CleanupStatus = Field(default=CleanupStatus.PENDING)
    attempts: int = Field(default=0)
    last_attempt: Optional[datetime] = None
    next_attempt: Optional[datetime] = None
    error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def add_error(self, error: str) -> None:
        """Add error to history."""
        self.error_history.append({"error": error, "timestamp": datetime.utcnow()})
        self.error = error
        self.updated_at = datetime.utcnow()

    def add_metric(self, name: str, value: Any) -> None:
        """Add metric."""
        self.metrics[name] = value
        self.updated_at = datetime.utcnow()

    def update_status(self, status: CleanupStatus) -> None:
        """Update batch status."""
        self.status = status
        self.updated_at = datetime.utcnow()

    def increment_attempts(self) -> None:
        """Increment attempt count."""
        self.attempts += 1
        self.last_attempt = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def set_next_attempt(self, next_attempt: datetime) -> None:
        """Set next attempt time."""
        self.next_attempt = next_attempt
        self.updated_at = datetime.utcnow()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to batch."""
        self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if batch is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at


class UserMetrics(BaseModel):
    """User metrics."""

    user_id: str = Field(..., min_length=1, max_length=255)
    total_notifications: int = Field(default=0)
    deleted_notifications: int = Field(default=0)
    last_cleanup: Optional[datetime] = None
    cleanup_count: int = Field(default=0)
    error_count: int = Field(default=0)
    success_count: int = Field(default=0)
    average_cleanup_time: float = Field(default=0.0)
    last_error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_error(self, error: str) -> None:
        """Add error to history."""
        self.error_history.append({"error": error, "timestamp": datetime.utcnow()})
        self.last_error = error
        self.error_count += 1
        self.updated_at = datetime.utcnow()

    def add_success(self, cleanup_time: float) -> None:
        """Add success record."""
        self.success_count += 1
        self.cleanup_count += 1
        self.last_cleanup = datetime.utcnow()

        # Update average cleanup time
        if self.average_cleanup_time == 0:
            self.average_cleanup_time = cleanup_time
        else:
            self.average_cleanup_time = (
                self.average_cleanup_time * (self.success_count - 1) + cleanup_time
            ) / self.success_count

        self.updated_at = datetime.utcnow()

    def add_metric(self, name: str, value: Any) -> None:
        """Add metric."""
        self.metrics[name] = value
        self.updated_at = datetime.utcnow()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to metrics."""
        self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()


class NotificationCleanupService:
    def __init__(
        self,
        notification_manager: NotificationManager,
        config: NotificationConfig,
        database_service: DatabaseService,
        transaction_service: TransactionService,
        metrics_service: MetricsService,
        retry_service: RetryService,
        error_handling_service: ErrorHandlingService,
        audit_service: AuditService,
        health_service: HealthService,
        cache_service: CacheService,
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
        state_service: StateService,
    ):
        self.notification_manager = notification_manager
        self.config = config
        self.database_service = database_service
        self.transaction_service = transaction_service
        self.metrics_service = metrics_service
        self.retry_service = retry_service
        self.error_handling_service = error_handling_service
        self.audit_service = audit_service
        self.health_service = health_service
        self.cache_service = cache_service
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
        self.logger = logging.getLogger(__name__)

        # Initialize service state
        self._cleanup_task: Optional[asyncio.Task] = None
        self._batch_size = self.config.get("cleanup.batch_size", 1000)
        self._max_retries = self.config.get("cleanup.max_retries", 3)
        self._retry_delay = self.config.get("cleanup.retry_delay", 5)
        self._transaction_timeout = self.config.get("cleanup.transaction_timeout", 30)
        self._user_batch_queue = self.queue_service.create_queue(
            "user_batch", max_size=self.config.get("cleanup.queue_size", 10000)
        )
        self._active_operation: Optional[CleanupOperation] = None
        self._workers = self.worker_service.create_workers(
            "user_batch",
            self._batch_worker,
            count=self.config.get("cleanup.worker_count", 5),
            batch_size=self.config.get("cleanup.worker_batch_size", 100),
            timeout=self.config.get("cleanup.worker_timeout", 30),
        )

        # Initialize user metrics
        self._user_metrics: Dict[str, UserMetrics] = {}
        self._load_user_metrics()

        # Start metrics collection task
        self._metrics_task = self.scheduler_service.schedule_task(
            self._collect_metrics, interval=self.config.get("cleanup.metrics_interval", 60)
        )

        # Start health check task
        self._health_task = self.scheduler_service.schedule_task(
            self._check_health, interval=self.config.get("cleanup.health_check_interval", 60)
        )

        # Start cleanup task
        self._cleanup_task = self.scheduler_service.schedule_task(
            self._cleanup_loop, interval=self.config.get("cleanup.cleanup_interval", 3600)
        )

        # Initialize metrics
        self._init_metrics()

        # Initialize security
        self._init_security()

        # Initialize logging
        self._init_logging()

    def _init_metrics(self):
        """Initialize service metrics."""
        try:
            # Initialize metrics
            self.metrics_service.initialize()

            # Set up cleanup metrics
            self.metrics_service.register_cleanup_metrics()

            # Set up user metrics
            for user_id, metrics in self._user_metrics.items():
                self.metrics_service.register_user_metrics(user_id)

        except Exception as e:
            logger.error(f"Error initializing metrics: {str(e)}")
            raise

    def _init_security(self):
        """Initialize security settings."""
        try:
            # Initialize security service
            self.security_service.initialize()

            # Set up security policies
            self.security_service.set_cleanup_policy(self.config.get("cleanup.security_level", 1))

        except Exception as e:
            logger.error(f"Error initializing security: {str(e)}")
            raise

    def _init_logging(self):
        """Initialize logging settings."""
        try:
            # Initialize logging service
            self.logging_service.initialize()

            # Set up logging
            self.logging_service.set_cleanup_logging(self.config.get("cleanup.audit_level", 1))

        except Exception as e:
            logger.error(f"Error initializing logging: {str(e)}")
            raise

    async def _load_user_metrics(self):
        """Load user metrics from storage."""
        try:
            # Get metrics from persistence
            metrics_data = await self.persistence_service.get_user_metrics()

            # Create metrics objects
            for user_id, data in metrics_data.items():
                self._user_metrics[user_id] = UserMetrics(user_id=user_id, **data)

            # Record metrics
            self.metrics_service.record_metrics_load(len(metrics_data))

            # Audit
            await self.audit_service.record_audit("metrics_load", "success", f"Loaded {len(metrics_data)} user metrics")

        except Exception as e:
            logger.error(f"Error loading user metrics: {str(e)}")

            # Record metrics
            self.metrics_service.record_metrics_load_error()

            # Audit
            await self.audit_service.record_audit("metrics_load", "error", str(e))

    async def _collect_metrics(self):
        """Collect service metrics."""
        try:
            # Collect operation metrics
            if self._active_operation:
                self._active_operation.add_metric("queue_size", self._user_batch_queue.qsize())
                self._active_operation.add_metric("worker_count", len(self._workers))
                self._active_operation.add_metric(
                    "user_metrics", {user_id: metrics.dict() for user_id, metrics in self._user_metrics.items()}
                )

            # Record metrics
            self.metrics_service.record_cleanup_metrics(
                self._user_batch_queue.qsize(), len(self._workers), len(self._user_metrics)
            )

            # Audit
            await self.audit_service.record_audit("metrics_collection", "success", "Collected cleanup metrics")

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")

            # Record metrics
            self.metrics_service.record_metrics_collection_error()

            # Audit
            await self.audit_service.record_audit("metrics_collection", "error", str(e))

    async def _check_health(self):
        """Check service health."""
        try:
            # Check queue health
            queue_health = await self._user_batch_queue.check_health()

            # Check worker health
            worker_health = await self.worker_service.check_health()

            # Update health status
            self.health_service.update_health({"queue": queue_health, "workers": worker_health})

            # Record metrics
            self.metrics_service.record_health_check()

            # Audit
            await self.audit_service.record_audit("health_check", "success", "Checked service health")

        except Exception as e:
            logger.error(f"Error checking health: {str(e)}")

            # Record metrics
            self.metrics_service.record_health_check_error()

            # Audit
            await self.audit_service.record_audit("health_check", "error", str(e))

    async def start(self) -> None:
        """Start the cleanup service."""
        try:
            if self._cleanup_task is not None:
                return

            # Start cleanup task
            self._cleanup_task = self.scheduler_service.schedule_task(
                self._cleanup_loop, interval=self.config.get("cleanup.cleanup_interval", 3600)
            )

            # Record metrics
            self.metrics_service.record_service_start()

            # Audit
            await self.audit_service.record_audit("service_start", "success", "Started cleanup service")

            logger.info("Notification cleanup service started")

        except Exception as e:
            logger.error(f"Error starting service: {str(e)}")

            # Record metrics
            self.metrics_service.record_service_start_error()

            # Audit
            await self.audit_service.record_audit("service_start", "error", str(e))

    async def stop(self) -> None:
        """Stop the cleanup service."""
        try:
            if self._cleanup_task is None:
                return

            # Stop cleanup task
            self.scheduler_service.cancel_task(self._cleanup_task)
            self._cleanup_task = None

            # Record metrics
            self.metrics_service.record_service_stop()

            # Audit
            await self.audit_service.record_audit("service_stop", "success", "Stopped cleanup service")

            logger.info("Notification cleanup service stopped")

        except Exception as e:
            logger.error(f"Error stopping service: {str(e)}")

            # Record metrics
            self.metrics_service.record_service_stop_error()

            # Audit
            await self.audit_service.record_audit("service_stop", "error", str(e))

    async def _cleanup_loop(self) -> None:
        """Main cleanup loop."""
        try:
            # Check if already running
            if self._active_operation and self._active_operation.status == CleanupStatus.RUNNING:
                return

            # Start cleanup
            await self._start_cleanup()

            # Record metrics
            self.metrics_service.record_cleanup_loop()

            # Audit
            await self.audit_service.record_audit("cleanup_loop", "success", "Completed cleanup loop")

        except Exception as e:
            logger.error(f"Error in cleanup loop: {str(e)}")

            # Record metrics
            self.metrics_service.record_cleanup_loop_error()

            # Audit
            await self.audit_service.record_audit("cleanup_loop", "error", str(e))

    async def _start_cleanup(self) -> None:
        """Start a new cleanup operation."""
        try:
            # Create operation
            operation = CleanupOperation(
                id=str(uuid4()),
                status=CleanupStatus.RUNNING,
                start_time=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.config.get("cleanup.operation_timeout", 3600)),
            )

            # Store operation
            await self.persistence_service.store_operation(operation)

            # Set active operation
            self._active_operation = operation

            # Get users
            user_ids = await self._get_all_users()

            # Sort users by priority
            sorted_users = await self._sort_users_by_priority(user_ids)

            # Create batches
            await self._create_user_batches(sorted_users)

            # Update operation
            operation.total_users = len(user_ids)
            operation.add_metric("total_users", len(user_ids))
            await self._update_operation(operation)

            # Record metrics
            self.metrics_service.record_cleanup_start()

            # Audit
            await self.audit_service.record_audit(
                "cleanup_start", "success", f"Started cleanup operation {operation.id}"
            )

        except Exception as e:
            logger.error(f"Error starting cleanup: {str(e)}")

            # Record metrics
            self.metrics_service.record_cleanup_start_error()

            # Audit
            await self.audit_service.record_audit("cleanup_start", "error", str(e))

            raise

    async def _resume_cleanup(self) -> None:
        """Resume a paused cleanup operation."""
        try:
            # Get active operation
            if not self._active_operation:
                return

            # Check operation status
            if self._active_operation.status != CleanupStatus.PAUSED:
                return

            # Update operation
            self._active_operation.update_status(CleanupStatus.RESUMING)
            await self._update_operation(self._active_operation)

            # Get remaining users
            user_ids = await self._get_remaining_users()

            # Sort users by priority
            sorted_users = await self._sort_users_by_priority(user_ids)

            # Create batches
            await self._create_user_batches(sorted_users)

            # Update operation
            self._active_operation.update_status(CleanupStatus.RUNNING)
            await self._update_operation(self._active_operation)

            # Record metrics
            self.metrics_service.record_cleanup_resume()

            # Audit
            await self.audit_service.record_audit(
                "cleanup_resume", "success", f"Resumed cleanup operation {self._active_operation.id}"
            )

        except Exception as e:
            logger.error(f"Error resuming cleanup: {str(e)}")

            # Record metrics
            self.metrics_service.record_cleanup_resume_error()

            # Audit
            await self.audit_service.record_audit("cleanup_resume", "error", str(e))

            raise

    async def _create_user_batches(self, user_ids: Optional[List[str]] = None) -> None:
        """Create user batches for processing."""
        try:
            # Get users if not provided
            if user_ids is None:
                user_ids = await self._get_all_users()

            # Create batches
            for i in range(0, len(user_ids), self._batch_size):
                batch = UserBatch(
                    id=str(uuid4()),
                    user_ids=user_ids[i : i + self._batch_size],
                    expires_at=datetime.utcnow() + timedelta(seconds=self.config.get("cleanup.batch_timeout", 300)),
                )

                # Store batch
                await self.persistence_service.store_batch(batch)

                # Enqueue batch
                await self._user_batch_queue.enqueue(batch)

            # Record metrics
            self.metrics_service.record_batch_creation(len(user_ids) // self._batch_size + 1)

            # Audit
            await self.audit_service.record_audit(
                "batch_creation", "success", f"Created {len(user_ids) // self._batch_size + 1} batches"
            )

        except Exception as e:
            logger.error(f"Error creating batches: {str(e)}")

            # Record metrics
            self.metrics_service.record_batch_creation_error()

            # Audit
            await self.audit_service.record_audit("batch_creation", "error", str(e))

            raise

    async def _sort_users_by_priority(self, user_ids: List[str]) -> List[str]:
        """Sort users by cleanup priority."""
        try:
            # Get user metrics
            metrics = {user_id: self._user_metrics.get(user_id) for user_id in user_ids}

            # Sort users
            sorted_users = sorted(
                user_ids,
                key=lambda x: (
                    metrics[x].error_count if metrics[x] else 0,
                    metrics[x].total_notifications if metrics[x] else 0,
                    metrics[x].last_cleanup.timestamp() if metrics[x] and metrics[x].last_cleanup else 0,
                ),
                reverse=True,
            )

            # Record metrics
            self.metrics_service.record_user_sorting(len(user_ids))

            # Audit
            await self.audit_service.record_audit(
                "user_sorting", "success", f"Sorted {len(user_ids)} users by priority"
            )

            return sorted_users

        except Exception as e:
            logger.error(f"Error sorting users: {str(e)}")

            # Record metrics
            self.metrics_service.record_user_sorting_error()

            # Audit
            await self.audit_service.record_audit("user_sorting", "error", str(e))

            return user_ids

    async def _batch_worker(self, batch: UserBatch) -> None:
        """Process a batch of users."""
        try:
            # Process batch
            await self._process_user_batch(batch)

            # Record metrics
            self.metrics_service.record_batch_processing(len(batch.user_ids))

            # Audit
            await self.audit_service.record_audit(
                "batch_processing", "success", f"Processed batch {batch.id} with {len(batch.user_ids)} users"
            )

        except Exception as e:
            logger.error(f"Error processing batch {batch.id}: {str(e)}")

            # Record metrics
            self.metrics_service.record_batch_processing_error()

            # Audit
            await self.audit_service.record_audit("batch_processing", "error", str(e))

            raise

    async def _process_user_batch(self, batch: UserBatch) -> None:
        """Process a batch of users."""
        try:
            # Update batch status
            batch.update_status(CleanupStatus.RUNNING)
            await self._update_batch(batch)

            # Process each user
            for user_id in batch.user_ids:
                try:
                    # Get notifications
                    notifications = await self.notification_manager.get_user_notifications(user_id)

                    # Process notifications
                    await self._process_batch_with_transaction(user_id, notifications)

                    # Update user metrics
                    await self._update_user_metrics(user_id, len(notifications))

                except Exception as e:
                    # Handle user error
                    await self._update_user_error(user_id)

                    # Add error to batch
                    batch.add_error(f"Error processing user {user_id}: {str(e)}")

                    # Update batch
                    await self._update_batch(batch)

                    continue

            # Update batch status
            batch.update_status(CleanupStatus.COMPLETED)
            await self._update_batch(batch)

            # Update operation
            if self._active_operation:
                self._active_operation.processed_users += len(batch.user_ids)
                await self._update_operation(self._active_operation)

            # Record metrics
            self.metrics_service.record_user_batch_processing(len(batch.user_ids))

            # Audit
            await self.audit_service.record_audit(
                "user_batch_processing", "success", f"Processed {len(batch.user_ids)} users in batch {batch.id}"
            )

        except Exception as e:
            logger.error(f"Error processing user batch {batch.id}: {str(e)}")

            # Update batch status
            batch.update_status(CleanupStatus.FAILED)
            batch.add_error(str(e))
            await self._update_batch(batch)

            # Record metrics
            self.metrics_service.record_user_batch_processing_error()

            # Audit
            await self.audit_service.record_audit("user_batch_processing", "error", str(e))

            raise

    async def _process_batch_with_transaction(self, user_id: str, notifications: List[Dict[str, Any]]) -> None:
        """Process notifications in a transaction."""
        try:
            # Start transaction
            async with self.transaction_service.transaction(timeout=self._transaction_timeout) as transaction:
                # Delete notifications
                await self.notification_manager.delete_notifications([n["id"] for n in notifications])

                # Update metrics
                if self._active_operation:
                    self._active_operation.deleted_notifications += len(notifications)
                    self._active_operation.total_notifications += len(notifications)
                    await self._update_operation(self._active_operation)

            # Record metrics
            self.metrics_service.record_notification_deletion(len(notifications))

            # Audit
            await self.audit_service.record_audit(
                "notification_deletion", "success", f"Deleted {len(notifications)} notifications for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Error processing notifications for user {user_id}: {str(e)}")

            # Record metrics
            self.metrics_service.record_notification_deletion_error()

            # Audit
            await self.audit_service.record_audit("notification_deletion", "error", str(e))

            raise

    async def _update_user_metrics(self, user_id: str, notification_count: int) -> None:
        """Update user metrics."""
        try:
            # Get metrics
            metrics = self._user_metrics.get(user_id)
            if not metrics:
                metrics = UserMetrics(user_id=user_id)
                self._user_metrics[user_id] = metrics

            # Update metrics
            metrics.total_notifications += notification_count
            metrics.deleted_notifications += notification_count

            # Calculate cleanup time (simplified - in real implementation would track actual time)
            cleanup_time = 0.1  # Estimated cleanup time per notification
            metrics.add_success(cleanup_time)

            # Store metrics
            await self.persistence_service.store_user_metrics(user_id, metrics.dict())

            # Record metrics
            self.metrics_service.record_user_metrics_update()

            # Audit
            await self.audit_service.record_audit(
                "user_metrics_update", "success", f"Updated metrics for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Error updating metrics for user {user_id}: {str(e)}")

            # Record metrics
            self.metrics_service.record_user_metrics_update_error()

            # Audit
            await self.audit_service.record_audit("user_metrics_update", "error", str(e))

    async def _update_user_error(self, user_id: str) -> None:
        """Update user error count."""
        try:
            # Get metrics
            metrics = self._user_metrics.get(user_id)
            if not metrics:
                metrics = UserMetrics(user_id=user_id)
                self._user_metrics[user_id] = metrics

            # Update metrics
            metrics.add_error("Cleanup failed")

            # Store metrics
            await self.persistence_service.store_user_metrics(user_id, metrics.dict())

            # Record metrics
            self.metrics_service.record_user_error_update()

            # Audit
            await self.audit_service.record_audit(
                "user_error_update", "success", f"Updated error count for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Error updating error count for user {user_id}: {str(e)}")

            # Record metrics
            self.metrics_service.record_user_error_update_error()

            # Audit
            await self.audit_service.record_audit("user_error_update", "error", str(e))

    async def _get_total_users(self) -> int:
        """Get total number of users."""
        try:
            # Get users
            users = await self._get_all_users()

            return len(users)

        except Exception as e:
            logger.error(f"Error getting total users: {str(e)}")
            return 0

    async def _get_remaining_users(self) -> List[str]:
        """Get remaining users to process."""
        try:
            # Get all users
            all_users = await self._get_all_users()

            # Get processed users
            if self._active_operation:
                processed_users = await self.persistence_service.get_processed_users(self._active_operation.id)
            else:
                processed_users = []

            # Get remaining users
            remaining_users = list(set(all_users) - set(processed_users))

            return remaining_users

        except Exception as e:
            logger.error(f"Error getting remaining users: {str(e)}")
            return []

    async def _get_all_users(self) -> List[str]:
        """Get all users."""
        try:
            # Get users
            users = await self.notification_manager.get_all_users()

            return users

        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return []

    async def cleanup_now(self) -> None:
        """Start cleanup immediately."""
        try:
            # Start cleanup
            await self._start_cleanup()

            # Record metrics
            self.metrics_service.record_cleanup_now()

            # Audit
            await self.audit_service.record_audit("cleanup_now", "success", "Started immediate cleanup")

        except Exception as e:
            logger.error(f"Error starting immediate cleanup: {str(e)}")

            # Record metrics
            self.metrics_service.record_cleanup_now_error()

            # Audit
            await self.audit_service.record_audit("cleanup_now", "error", str(e))

            raise

    async def get_cleanup_status(self) -> Optional[CleanupOperation]:
        """Get cleanup operation status."""
        try:
            # Get operation
            if self._active_operation:
                return self._active_operation

            return None

        except Exception as e:
            logger.error(f"Error getting cleanup status: {str(e)}")
            return None

    async def get_cleanup_metrics(self) -> Dict[str, Any]:
        """Get cleanup metrics."""
        try:
            # Get metrics
            metrics = await self.metrics_service.get_cleanup_metrics()

            # Record metrics
            self.metrics_service.record_metrics_check()

            # Audit
            await self.audit_service.record_audit("metrics_check", "success", "Retrieved cleanup metrics")

            return metrics

        except Exception as e:
            logger.error(f"Error getting cleanup metrics: {str(e)}")

            # Record metrics
            self.metrics_service.record_metrics_check_error()

            # Audit
            await self.audit_service.record_audit("metrics_check", "error", str(e))

            return {}

    async def get_health(self) -> Dict[str, Any]:
        """Get service health."""
        try:
            # Get health
            health = await self.health_service.get_health()

            # Record metrics
            self.metrics_service.record_health_check()

            # Audit
            await self.audit_service.record_audit("health_check", "success", "Retrieved service health")

            return health

        except Exception as e:
            logger.error(f"Error getting health: {str(e)}")

            # Record metrics
            self.metrics_service.record_health_check_error()

            # Audit
            await self.audit_service.record_audit("health_check", "error", str(e))

            return {}

    async def cleanup_expired_notifications(self, ttl_config: Optional[NotificationTTL] = None) -> Dict[str, Any]:
        """Clean up notifications that have exceeded their TTL.

        Args:
            ttl_config: TTL configuration to use (uses default if None)

        Returns:
            Dictionary with cleanup results
        """
        try:
            if not ttl_config:
                ttl_config = NotificationTTL()

            logger.info("Starting TTL-based notification cleanup")

            # Get all users
            users = await self._get_all_users()
            total_deleted = 0
            total_processed = 0
            user_results = {}

            for user_id in users:
                try:
                    # Get user notifications
                    notifications = await self.notification_manager.get_user_notifications(user_id)

                    if not notifications:
                        continue

                    # Filter expired notifications
                    expired_notifications = []
                    for notification in notifications:
                        if self._is_notification_expired(notification, ttl_config, user_id):
                            expired_notifications.append(notification)

                    if expired_notifications:
                        # Delete expired notifications
                        deleted_count = await self._delete_expired_notifications(user_id, expired_notifications)

                        total_deleted += deleted_count
                        user_results[user_id] = {
                            "total_notifications": len(notifications),
                            "expired_notifications": len(expired_notifications),
                            "deleted_notifications": deleted_count,
                        }

                    total_processed += 1

                    # Update metrics
                    await self._update_user_metrics(user_id, len(expired_notifications))

                except Exception as e:
                    logger.error(f"Error processing user {user_id} for TTL cleanup: {e}")
                    await self._update_user_error(user_id)
                    user_results[user_id] = {"error": str(e)}

            # Record metrics
            self.metrics_service.record_ttl_cleanup(total_deleted, total_processed)

            # Audit
            await self.audit_service.record_audit(
                "ttl_cleanup",
                "success",
                f"TTL cleanup completed: {total_deleted} notifications deleted from {total_processed} users",
            )

            return {
                "success": True,
                "total_deleted": total_deleted,
                "total_processed": total_processed,
                "user_results": user_results,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in TTL cleanup: {e}")

            # Record metrics
            self.metrics_service.record_ttl_cleanup_error()

            # Audit
            await self.audit_service.record_audit("ttl_cleanup", "error", str(e))

            return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def _is_notification_expired(self, notification: Dict[str, Any], ttl_config: NotificationTTL, user_id: str) -> bool:
        """Check if a notification has expired based on TTL.

        Args:
            notification: Notification data
            ttl_config: TTL configuration
            user_id: User ID

        Returns:
            True if notification has expired
        """
        try:
            # Get notification creation time
            created_at = notification.get("created_at")
            if not created_at:
                # If no creation time, assume it's old
                return True

            # Parse creation time
            if isinstance(created_at, str):
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created_dt = created_at

            # Get TTL for this notification
            priority = notification.get("priority", "medium")
            notification_type = notification.get("type", "general")
            ttl_hours = ttl_config.get_ttl_for_notification(priority, notification_type, user_id)

            # Calculate expiration time
            expiration_time = created_dt + timedelta(hours=ttl_hours)

            # Check if expired
            return datetime.utcnow() >= expiration_time

        except Exception as e:
            logger.error(f"Error checking notification expiration: {e}")
            # If we can't determine, assume it's expired
            return True

    async def _delete_expired_notifications(self, user_id: str, expired_notifications: List[Dict[str, Any]]) -> int:
        """Delete expired notifications for a user.

        Args:
            user_id: User ID
            expired_notifications: List of expired notifications

        Returns:
            Number of notifications deleted
        """
        try:
            deleted_count = 0

            for notification in expired_notifications:
                try:
                    # Delete notification
                    await self.notification_manager.delete_notification(user_id, notification.get("id"))
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error deleting notification {notification.get('id')}: {e}")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting expired notifications for user {user_id}: {e}")
            return 0

    async def configure_ttl_settings(self, ttl_config: NotificationTTL) -> Dict[str, Any]:
        """Configure TTL settings for the cleanup service.

        Args:
            ttl_config: TTL configuration

        Returns:
            Dictionary with configuration status
        """
        try:
            # Validate TTL configuration
            ttl_config.validate()

            # Store TTL configuration
            await self.persistence_service.store_ttl_config(ttl_config.dict())

            # Update service configuration
            self._ttl_config = ttl_config

            # Record metrics
            self.metrics_service.record_ttl_config_update()

            # Audit
            await self.audit_service.record_audit(
                "ttl_config_update", "success", f"TTL configuration updated: default={ttl_config.default_ttl_hours}h"
            )

            return {
                "success": True,
                "message": "TTL configuration updated successfully",
                "ttl_config": ttl_config.dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error configuring TTL settings: {e}")

            # Record metrics
            self.metrics_service.record_ttl_config_update_error()

            # Audit
            await self.audit_service.record_audit("ttl_config_update", "error", str(e))

            return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def get_ttl_statistics(self) -> Dict[str, Any]:
        """Get TTL statistics and metrics.

        Returns:
            Dictionary with TTL statistics
        """
        try:
            # Get all users
            users = await self._get_all_users()

            total_notifications = 0
            expired_notifications = 0
            ttl_distribution = {}

            for user_id in users:
                try:
                    # Get user notifications
                    notifications = await self.notification_manager.get_user_notifications(user_id)

                    if not notifications:
                        continue

                    total_notifications += len(notifications)

                    # Count expired notifications
                    for notification in notifications:
                        if self._is_notification_expired(notification, self._ttl_config, user_id):
                            expired_notifications += 1

                        # Track TTL distribution
                        priority = notification.get("priority", "medium")
                        if priority not in ttl_distribution:
                            ttl_distribution[priority] = 0
                        ttl_distribution[priority] += 1

                except Exception as e:
                    logger.error(f"Error getting TTL statistics for user {user_id}: {e}")

            # Calculate statistics
            expiration_rate = (expired_notifications / total_notifications * 100) if total_notifications > 0 else 0

            return {
                "success": True,
                "total_notifications": total_notifications,
                "expired_notifications": expired_notifications,
                "expiration_rate_percent": round(expiration_rate, 2),
                "ttl_distribution": ttl_distribution,
                "ttl_config": self._ttl_config.dict() if self._ttl_config else None,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting TTL statistics: {e}")

            return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def schedule_ttl_cleanup(self, interval_hours: int = 6) -> Dict[str, Any]:
        """Schedule periodic TTL cleanup.

        Args:
            interval_hours: Cleanup interval in hours

        Returns:
            Dictionary with scheduling status
        """
        try:
            # Schedule TTL cleanup task
            await self.scheduler_service.schedule_periodic_task(
                "ttl_cleanup", self.cleanup_expired_notifications, interval_hours=interval_hours
            )

            # Record metrics
            self.metrics_service.record_ttl_schedule()

            # Audit
            await self.audit_service.record_audit(
                "ttl_schedule", "success", f"TTL cleanup scheduled with {interval_hours}h interval"
            )

            return {
                "success": True,
                "message": f"TTL cleanup scheduled with {interval_hours}h interval",
                "interval_hours": interval_hours,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error scheduling TTL cleanup: {e}")

            # Record metrics
            self.metrics_service.record_ttl_schedule_error()

            # Audit
            await self.audit_service.record_audit("ttl_schedule", "error", str(e))

            return {"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}
