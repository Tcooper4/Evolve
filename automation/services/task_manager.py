import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from automation.models.automation import AutomationTask, TaskStatus
from automation.services.security_service import SecurityService
from automation.services.validation_service import ValidationService
from automation.services.persistence_service import PersistenceService
from automation.services.audit_service import AuditService
from automation.services.metrics_service import MetricsService
from automation.services.logging_service import LoggingService

logger = logging.getLogger(__name__)

class TaskMetrics(BaseModel):
    """Metrics for task operations."""
    total_tasks: int = Field(default=0)
    active_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    average_execution_time: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TaskManager:
    """Manages automation tasks."""
    
    def __init__(
        self,
        security_service: SecurityService,
        validation_service: ValidationService,
        persistence_service: PersistenceService,
        audit_service: AuditService,
        metrics_service: MetricsService,
        logging_service: LoggingService
    ):
        """Initialize task manager.
        
        Args:
            security_service: Service for handling security
            validation_service: Service for handling validation
            persistence_service: Service for handling persistence
            audit_service: Service for handling auditing
            metrics_service: Service for handling metrics
            logging_service: Service for handling logging
        """
        self.security_service = security_service
        self.validation_service = validation_service
        self.persistence_service = persistence_service
        self.audit_service = audit_service
        self.metrics_service = metrics_service
        self.logging_service = logging_service
        
        self._metrics = TaskMetrics()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_locks: Dict[str, asyncio.Lock] = {}
        
    async def create_task(self, task: AutomationTask) -> str:
        """Create a new automation task.
        
        Args:
            task: Task to create
            
        Returns:
            str: Task ID
            
        Raises:
            ValueError: If task is invalid
            PermissionError: If not authorized
        """
        try:
            # Validate task
            if not self.validation_service.validate_task(task):
                raise ValueError("Invalid task")
            
            # Check security
            if not self.security_service.can_create_task(task):
                raise PermissionError("Not authorized to create task")
            
            # Create task
            task_id = await self.persistence_service.create_task(task)
            
            # Update metrics
            self._metrics.total_tasks += 1
            self._metrics.active_tasks += 1
            self._metrics.updated_at = datetime.utcnow()
            
            # Record metrics
            await self.metrics_service.record_task_creation(task_id)
            
            # Audit
            await self.audit_service.record_task_creation(task_id, task)
            
            logger.info(f"Created task {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[AutomationTask]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[AutomationTask]: Task if found
            
        Raises:
            PermissionError: If not authorized
        """
        try:
            # Check security
            if not self.security_service.can_view_task(task_id):
                raise PermissionError("Not authorized to view task")
            
            # Get task
            task = await self.persistence_service.get_task(task_id)
            
            # Record metrics
            await self.metrics_service.record_task_view(task_id)
            
            # Audit
            await self.audit_service.record_task_view(task_id)
            
            return task
            
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {str(e)}")
            raise
    
    async def update_task(self, task_id: str, task: AutomationTask) -> bool:
        """Update a task.
        
        Args:
            task_id: Task ID
            task: Updated task
            
        Returns:
            bool: True if updated
            
        Raises:
            ValueError: If task is invalid
            PermissionError: If not authorized
        """
        try:
            # Validate task
            if not self.validation_service.validate_task(task):
                raise ValueError("Invalid task")
            
            # Check security
            if not self.security_service.can_update_task(task_id):
                raise PermissionError("Not authorized to update task")
            
            # Get lock
            if task_id not in self._task_locks:
                self._task_locks[task_id] = asyncio.Lock()
            
            async with self._task_locks[task_id]:
                # Update task
                success = await self.persistence_service.update_task(task_id, task)
                
                if success:
                    # Record metrics
                    await self.metrics_service.record_task_update(task_id)
                    
                    # Audit
                    await self.audit_service.record_task_update(task_id, task)
                    
                    logger.info(f"Updated task {task_id}")
                
                return success
            
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {str(e)}")
            raise
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if deleted
            
        Raises:
            PermissionError: If not authorized
        """
        try:
            # Check security
            if not self.security_service.can_delete_task(task_id):
                raise PermissionError("Not authorized to delete task")
            
            # Get lock
            if task_id not in self._task_locks:
                self._task_locks[task_id] = asyncio.Lock()
            
            async with self._task_locks[task_id]:
                # Delete task
                success = await self.persistence_service.delete_task(task_id)
                
                if success:
                    # Update metrics
                    self._metrics.total_tasks -= 1
                    self._metrics.active_tasks -= 1
                    self._metrics.updated_at = datetime.utcnow()
                    
                    # Record metrics
                    await self.metrics_service.record_task_deletion(task_id)
                    
                    # Audit
                    await self.audit_service.record_task_deletion(task_id)
                    
                    # Cleanup
                    if task_id in self._task_locks:
                        del self._task_locks[task_id]
                    
                    logger.info(f"Deleted task {task_id}")
                
                return success
            
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {str(e)}")
            raise
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AutomationTask]:
        """List tasks.
        
        Args:
            status: Filter by status
            limit: Maximum number of tasks
            offset: Offset for pagination
            
        Returns:
            List[AutomationTask]: List of tasks
            
        Raises:
            PermissionError: If not authorized
        """
        try:
            # Check security
            if not self.security_service.can_list_tasks():
                raise PermissionError("Not authorized to list tasks")
            
            # List tasks
            tasks = await self.persistence_service.list_tasks(
                status=status,
                limit=limit,
                offset=offset
            )
            
            # Record metrics
            await self.metrics_service.record_task_list(len(tasks))
            
            # Audit
            await self.audit_service.record_task_list(len(tasks))
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error listing tasks: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if executed successfully
            
        Raises:
            PermissionError: If not authorized
            ValueError: If task not found or dependencies not met
        """
        try:
            # Check security
            if not self.security_service.can_execute_task(task_id):
                raise PermissionError("Not authorized to execute task")
            
            # Get task
            task = await self.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            # Check dependencies
            if not await self._check_task_dependencies(task):
                raise ValueError(f"Task {task_id} dependencies not met")
            
            # Get lock
            if task_id not in self._task_locks:
                self._task_locks[task_id] = asyncio.Lock()
            
            async with self._task_locks[task_id]:
                # Check if already running
                if task_id in self._running_tasks:
                    raise ValueError(f"Task {task_id} is already running")
                
                # Update task status
                task.status = TaskStatus.RUNNING
                await self.update_task(task_id, task)
                
                # Create execution task
                execution_task = asyncio.create_task(
                    self._execute_task(task_id, task)
                )
                self._running_tasks[task_id] = execution_task
                
                # Update metrics
                self._metrics.total_executions += 1
                self._metrics.updated_at = datetime.utcnow()
                
                # Record metrics
                await self.metrics_service.record_task_execution(task_id)
                
                # Audit
                await self.audit_service.record_task_execution(task_id)
                
                logger.info(f"Started execution of task {task_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            raise
    
    async def _execute_task(self, task_id: str, task: AutomationTask):
        """Execute a task.
        
        Args:
            task_id: Task ID
            task: Task to execute
        """
        start_time = datetime.utcnow()
        success = False
        
        try:
            # Execute task
            if hasattr(task, 'execute'):
                success = await task.execute()
            else:
                raise NotImplementedError("Task does not implement execute method")
            
            # Update task status
            task.status = (
                TaskStatus.COMPLETED if success
                else TaskStatus.FAILED
            )
            await self.update_task(task_id, task)
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            if success:
                self._metrics.successful_executions += 1
                self._metrics.completed_tasks += 1
            else:
                self._metrics.failed_executions += 1
                self._metrics.failed_tasks += 1
            
            self._metrics.active_tasks -= 1
            self._metrics.average_execution_time = (
                (self._metrics.average_execution_time * (self._metrics.total_executions - 1) +
                 execution_time) / self._metrics.total_executions
            )
            self._metrics.updated_at = datetime.utcnow()
            
            # Record metrics
            await self.metrics_service.record_task_completion(
                task_id,
                success,
                execution_time
            )
            
            # Audit
            await self.audit_service.record_task_completion(
                task_id,
                success,
                execution_time
            )
            
            logger.info(
                f"Completed execution of task {task_id} "
                f"({'success' if success else 'failure'})"
            )
            
        except Exception as e:
            # Update task status
            task.status = TaskStatus.FAILED
            await self.update_task(task_id, task)
            
            # Update metrics
            self._metrics.failed_executions += 1
            self._metrics.failed_tasks += 1
            self._metrics.active_tasks -= 1
            self._metrics.updated_at = datetime.utcnow()
            
            # Record metrics
            await self.metrics_service.record_task_failure(task_id, str(e))
            
            # Audit
            await self.audit_service.record_task_failure(task_id, str(e))
            
            logger.error(f"Error executing task {task_id}: {str(e)}")
            
        finally:
            # Cleanup
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
    
    async def _check_task_dependencies(self, task: AutomationTask) -> bool:
        """Check if task dependencies are met.
        
        Args:
            task: Task to check
            
        Returns:
            bool: True if dependencies are met
        """
        try:
            if not task.dependencies:
                return True
            
            for dep_id in task.dependencies:
                dep_task = await self.get_task(dep_id)
                if not dep_task:
                    return False
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task dependencies: {str(e)}")
            return False
    
    async def get_metrics(self) -> TaskMetrics:
        """Get task metrics.
        
        Returns:
            TaskMetrics: Current metrics
        """
        return self._metrics
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cancel running tasks
            for task_id, task in self._running_tasks.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self._running_tasks.clear()
            self._task_locks.clear()
            
        except Exception as e:
            logger.error(f"Error cleaning up task manager: {str(e)}")
            raise 