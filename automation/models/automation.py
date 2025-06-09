from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from uuid import UUID
from pydantic import BaseModel, Field, validator, root_validator
import logging
from automation.services.metrics_service import MetricsService
from automation.services.transaction_service import TransactionService

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    """Status of an automation task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    SCHEDULED = "scheduled"
    PAUSED = "paused"
    RESUMED = "resumed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    UNBLOCKED = "unblocked"
    UNKNOWN = "unknown"

    @classmethod
    def get_valid_transitions(cls, current_status: 'TaskStatus') -> Set['TaskStatus']:
        """Get valid status transitions from current status."""
        transitions = {
            cls.PENDING: {cls.RUNNING, cls.CANCELLED, cls.SKIPPED},
            cls.RUNNING: {cls.COMPLETED, cls.FAILED, cls.TIMEOUT, cls.CANCELLED, cls.PAUSED},
            cls.PAUSED: {cls.RESUMED, cls.CANCELLED},
            cls.RESUMED: {cls.RUNNING, cls.FAILED, cls.TIMEOUT},
            cls.FAILED: {cls.RETRYING, cls.CANCELLED},
            cls.RETRYING: {cls.RUNNING, cls.FAILED, cls.CANCELLED},
            cls.TIMEOUT: {cls.RETRYING, cls.CANCELLED},
            cls.BLOCKED: {cls.UNBLOCKED, cls.CANCELLED},
            cls.UNBLOCKED: {cls.PENDING},
            cls.SCHEDULED: {cls.PENDING, cls.CANCELLED},
            cls.SKIPPED: {cls.COMPLETED},
            cls.COMPLETED: set(),
            cls.CANCELLED: set(),
            cls.UNKNOWN: set()
        }
        return transitions.get(current_status, set())

class WorkflowStatus(str, Enum):
    """Status of an automation workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    SCHEDULED = "scheduled"
    PAUSED = "paused"
    RESUMED = "resumed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    UNBLOCKED = "unblocked"
    UNKNOWN = "unknown"

    @classmethod
    def get_valid_transitions(cls, current_status: 'WorkflowStatus') -> Set['WorkflowStatus']:
        """Get valid status transitions from current status."""
        transitions = {
            cls.PENDING: {cls.RUNNING, cls.CANCELLED, cls.SKIPPED},
            cls.RUNNING: {cls.COMPLETED, cls.FAILED, cls.TIMEOUT, cls.CANCELLED, cls.PAUSED},
            cls.PAUSED: {cls.RESUMED, cls.CANCELLED},
            cls.RESUMED: {cls.RUNNING, cls.FAILED, cls.TIMEOUT},
            cls.FAILED: {cls.RETRYING, cls.CANCELLED},
            cls.RETRYING: {cls.RUNNING, cls.FAILED, cls.CANCELLED},
            cls.TIMEOUT: {cls.RETRYING, cls.CANCELLED},
            cls.BLOCKED: {cls.UNBLOCKED, cls.CANCELLED},
            cls.UNBLOCKED: {cls.PENDING},
            cls.SCHEDULED: {cls.PENDING, cls.CANCELLED},
            cls.SKIPPED: {cls.COMPLETED},
            cls.COMPLETED: set(),
            cls.CANCELLED: set(),
            cls.UNKNOWN: set()
        }
        return transitions.get(current_status, set())

class TaskConfiguration(BaseModel):
    """Configuration for an automation task."""
    schedule: Dict[str, Any] = Field(default_factory=dict, description="Schedule configuration")
    trigger: Dict[str, Any] = Field(default_factory=dict, description="Trigger configuration")
    condition: Dict[str, Any] = Field(default_factory=dict, description="Condition configuration")
    action: Dict[str, Any] = Field(default_factory=dict, description="Action configuration")
    handler: Dict[str, Any] = Field(default_factory=dict, description="Handler configuration")
    callback: Dict[str, Any] = Field(default_factory=dict, description="Callback configuration")
    notification: Dict[str, Any] = Field(default_factory=dict, description="Notification configuration")
    monitoring: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
    logging: Dict[str, Any] = Field(default_factory=dict, description="Logging configuration")
    auditing: Dict[str, Any] = Field(default_factory=dict, description="Auditing configuration")
    security: Dict[str, Any] = Field(default_factory=dict, description="Security configuration")
    compliance: Dict[str, Any] = Field(default_factory=dict, description="Compliance configuration")
    governance: Dict[str, Any] = Field(default_factory=dict, description="Governance configuration")
    risk: Dict[str, Any] = Field(default_factory=dict, description="Risk configuration")
    cost: Dict[str, Any] = Field(default_factory=dict, description="Cost configuration")
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance configuration")
    reliability: Dict[str, Any] = Field(default_factory=dict, description="Reliability configuration")
    availability: Dict[str, Any] = Field(default_factory=dict, description="Availability configuration")
    scalability: Dict[str, Any] = Field(default_factory=dict, description="Scalability configuration")
    maintainability: Dict[str, Any] = Field(default_factory=dict, description="Maintainability configuration")
    testability: Dict[str, Any] = Field(default_factory=dict, description="Testability configuration")
    deployability: Dict[str, Any] = Field(default_factory=dict, description="Deployability configuration")
    observability: Dict[str, Any] = Field(default_factory=dict, description="Observability configuration")
    traceability: Dict[str, Any] = Field(default_factory=dict, description="Traceability configuration")
    recoverability: Dict[str, Any] = Field(default_factory=dict, description="Recoverability configuration")
    resilience: Dict[str, Any] = Field(default_factory=dict, description="Resilience configuration")
    fault_tolerance: Dict[str, Any] = Field(default_factory=dict, description="Fault tolerance configuration")
    disaster_recovery: Dict[str, Any] = Field(default_factory=dict, description="Disaster recovery configuration")
    backup: Dict[str, Any] = Field(default_factory=dict, description="Backup configuration")
    restore: Dict[str, Any] = Field(default_factory=dict, description="Restore configuration")
    migration: Dict[str, Any] = Field(default_factory=dict, description="Migration configuration")
    rollback: Dict[str, Any] = Field(default_factory=dict, description="Rollback configuration")
    versioning: Dict[str, Any] = Field(default_factory=dict, description="Versioning configuration")
    dependency: Dict[str, Any] = Field(default_factory=dict, description="Dependency configuration")
    resource: Dict[str, Any] = Field(default_factory=dict, description="Resource configuration")
    lock: Dict[str, Any] = Field(default_factory=dict, description="Lock configuration")
    event: Dict[str, Any] = Field(default_factory=dict, description="Event configuration")
    hook: Dict[str, Any] = Field(default_factory=dict, description="Hook configuration")
    plugin: Dict[str, Any] = Field(default_factory=dict, description="Plugin configuration")

class IntegrationConfiguration(BaseModel):
    """Configuration for external integrations."""
    api: Dict[str, Any] = Field(default_factory=dict, description="API configuration")
    webhook: Dict[str, Any] = Field(default_factory=dict, description="Webhook configuration")
    websocket: Dict[str, Any] = Field(default_factory=dict, description="WebSocket configuration")
    grpc: Dict[str, Any] = Field(default_factory=dict, description="gRPC configuration")
    graphql: Dict[str, Any] = Field(default_factory=dict, description="GraphQL configuration")
    rest: Dict[str, Any] = Field(default_factory=dict, description="REST configuration")
    soap: Dict[str, Any] = Field(default_factory=dict, description="SOAP configuration")
    ftp: Dict[str, Any] = Field(default_factory=dict, description="FTP configuration")
    sftp: Dict[str, Any] = Field(default_factory=dict, description="SFTP configuration")
    s3: Dict[str, Any] = Field(default_factory=dict, description="S3 configuration")
    azure: Dict[str, Any] = Field(default_factory=dict, description="Azure configuration")
    gcp: Dict[str, Any] = Field(default_factory=dict, description="GCP configuration")
    aws: Dict[str, Any] = Field(default_factory=dict, description="AWS configuration")
    kubernetes: Dict[str, Any] = Field(default_factory=dict, description="Kubernetes configuration")
    docker: Dict[str, Any] = Field(default_factory=dict, description="Docker configuration")
    vm: Dict[str, Any] = Field(default_factory=dict, description="VM configuration")
    container: Dict[str, Any] = Field(default_factory=dict, description="Container configuration")
    serverless: Dict[str, Any] = Field(default_factory=dict, description="Serverless configuration")
    faas: Dict[str, Any] = Field(default_factory=dict, description="FaaS configuration")
    paas: Dict[str, Any] = Field(default_factory=dict, description="PaaS configuration")
    iaas: Dict[str, Any] = Field(default_factory=dict, description="IaaS configuration")
    saas: Dict[str, Any] = Field(default_factory=dict, description="SaaS configuration")

class CommunicationConfiguration(BaseModel):
    """Configuration for communication channels."""
    email: Dict[str, Any] = Field(default_factory=dict, description="Email configuration")
    sms: Dict[str, Any] = Field(default_factory=dict, description="SMS configuration")
    push: Dict[str, Any] = Field(default_factory=dict, description="Push notification configuration")
    voice: Dict[str, Any] = Field(default_factory=dict, description="Voice configuration")
    fax: Dict[str, Any] = Field(default_factory=dict, description="Fax configuration")
    chat: Dict[str, Any] = Field(default_factory=dict, description="Chat configuration")
    bot: Dict[str, Any] = Field(default_factory=dict, description="Bot configuration")
    slack: Dict[str, Any] = Field(default_factory=dict, description="Slack configuration")
    teams: Dict[str, Any] = Field(default_factory=dict, description="Teams configuration")
    discord: Dict[str, Any] = Field(default_factory=dict, description="Discord configuration")

class DevelopmentConfiguration(BaseModel):
    """Configuration for development tools."""
    git: Dict[str, Any] = Field(default_factory=dict, description="Git configuration")
    jenkins: Dict[str, Any] = Field(default_factory=dict, description="Jenkins configuration")
    github: Dict[str, Any] = Field(default_factory=dict, description="GitHub configuration")
    gitlab: Dict[str, Any] = Field(default_factory=dict, description="GitLab configuration")
    bitbucket: Dict[str, Any] = Field(default_factory=dict, description="Bitbucket configuration")
    jira: Dict[str, Any] = Field(default_factory=dict, description="Jira configuration")
    confluence: Dict[str, Any] = Field(default_factory=dict, description="Confluence configuration")
    ci: Dict[str, Any] = Field(default_factory=dict, description="CI configuration")
    cd: Dict[str, Any] = Field(default_factory=dict, description="CD configuration")
    devops: Dict[str, Any] = Field(default_factory=dict, description="DevOps configuration")

class AutomationTask(BaseModel):
    """Represents an automation task with validation."""
    
    id: Optional[str] = Field(default=None, description="Unique identifier for the task")
    name: str = Field(..., min_length=1, max_length=255, description="Name of the task")
    description: str = Field(default="", max_length=1000, description="Description of the task")
    type: str = Field(..., min_length=1, max_length=50, description="Type of the task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    priority: int = Field(default=0, ge=0, le=10, description="Priority of the task (0-10)")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    timeout: Optional[int] = Field(default=None, ge=1, description="Timeout in seconds")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")
    retry_delay: int = Field(default=60, ge=1, description="Delay between retries in seconds")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    result: Optional[Any] = Field(default=None, description="Task result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    dependencies: List[str] = Field(default_factory=list, description="List of dependent task IDs")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    owner: Optional[str] = Field(default=None, description="Task owner")
    team: Optional[str] = Field(default=None, description="Team responsible for the task")
    project: Optional[str] = Field(default=None, description="Project name")
    environment: Optional[str] = Field(default=None, description="Environment name")
    version: Optional[str] = Field(default=None, description="Version number")
    source: Optional[str] = Field(default=None, description="Source system")
    destination: Optional[str] = Field(default=None, description="Destination system")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    configuration: TaskConfiguration = Field(default_factory=TaskConfiguration, description="Task configuration")
    integration: IntegrationConfiguration = Field(default_factory=IntegrationConfiguration, description="Integration configuration")
    communication: CommunicationConfiguration = Field(default_factory=CommunicationConfiguration, description="Communication configuration")
    development: DevelopmentConfiguration = Field(default_factory=DevelopmentConfiguration, description="Development configuration")

    @validator('retry_count')
    def validate_retry_count(cls, v, values):
        """Validate that retry count doesn't exceed max retries."""
        if 'max_retries' in values and v > values['max_retries']:
            raise ValueError('retry_count cannot exceed max_retries')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v is not None and v < 1:
            raise ValueError('timeout must be at least 1 second')
        return v

    @root_validator
    def validate_dependencies(cls, values):
        """Validate task dependencies."""
        dependencies = values.get('dependencies', [])
        task_id = values.get('id')
        
        if task_id in dependencies:
            raise ValueError('Task cannot depend on itself')
        
        # Check for circular dependencies
        visited = set()
        path = set()
        
        def check_circular_dep(dep_id):
            if dep_id in path:
                raise ValueError(f'Circular dependency detected: {" -> ".join(path)} -> {dep_id}')
            if dep_id in visited:
                return
            
            visited.add(dep_id)
            path.add(dep_id)
            
            # Get dependent task's dependencies
            dep_task = cls.get_task(dep_id)  # This would need to be implemented
            if dep_task:
                for sub_dep in dep_task.dependencies:
                    check_circular_dep(sub_dep)
            
            path.remove(dep_id)
        
        for dep_id in dependencies:
            check_circular_dep(dep_id)
        
        return values

    async def execute(self, metrics_service: MetricsService, transaction_service: TransactionService) -> Dict[str, Any]:
        """Execute the automation task.
        
        Args:
            metrics_service: Service for collecting metrics
            transaction_service: Service for managing transactions
            
        Returns:
            Dict[str, Any]: The execution result.
            
        Raises:
            ValueError: If the task is invalid.
            RuntimeError: If the task execution fails.
        """
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot execute task in {self.status} status")
        
        async with transaction_service.transaction() as transaction:
            try:
                # Update status and start time
                self.status = TaskStatus.RUNNING
                self.started_at = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                
                # Log task start
                logger.info(f"Starting task {self.id} ({self.name})")
                
                # Record metrics
                metrics_service.record_task_start(self.id, self.type)
                
                # Task execution logic here
                # This would be implemented by the specific task type
                
                # Update status and completion time
                self.status = TaskStatus.COMPLETED
                self.completed_at = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                
                # Log task completion
                logger.info(f"Completed task {self.id} ({self.name})")
                
                # Record metrics
                metrics_service.record_task_completion(self.id, self.type)
                
                return {"status": "success", "result": self.result}
                
            except Exception as e:
                # Update status and error
                self.status = TaskStatus.FAILED
                self.error = str(e)
                self.updated_at = datetime.utcnow()
                
                # Log task failure
                logger.error(f"Failed task {self.id} ({self.name}): {str(e)}")
                
                # Record metrics
                metrics_service.record_task_failure(self.id, self.type, str(e))
                
                # Handle retry if applicable
                if self.retry_count < self.max_retries:
                    self.status = TaskStatus.RETRYING
                    self.retry_count += 1
                    self.updated_at = datetime.utcnow()
                    
                    # Log retry
                    logger.info(f"Retrying task {self.id} ({self.name}) - attempt {self.retry_count}")
                    
                    # Record metrics
                    metrics_service.record_task_retry(self.id, self.type)
                    
                    # Schedule retry
                    await asyncio.sleep(self.retry_delay)
                    return await self.execute(metrics_service, transaction_service)
                
                raise RuntimeError(f"Task execution failed: {str(e)}")

class AutomationWorkflow(BaseModel):
    """Represents an automation workflow with validation."""
    
    id: Optional[str] = Field(default=None, description="Unique identifier for the workflow")
    name: str = Field(..., min_length=1, max_length=255, description="Name of the workflow")
    description: str = Field(default="", max_length=1000, description="Description of the workflow")
    type: str = Field(..., min_length=1, max_length=50, description="Type of the workflow")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current status of the workflow")
    priority: int = Field(default=0, ge=0, le=10, description="Priority of the workflow (0-10)")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    timeout: Optional[int] = Field(default=None, ge=1, description="Timeout in seconds")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")
    retry_delay: int = Field(default=60, ge=1, description="Delay between retries in seconds")
    error: Optional[str] = Field(default=None, description="Error message if workflow failed")
    result: Optional[Any] = Field(default=None, description="Workflow result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    dependencies: List[str] = Field(default_factory=list, description="List of dependent workflow IDs")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    owner: Optional[str] = Field(default=None, description="Workflow owner")
    team: Optional[str] = Field(default=None, description="Team responsible for the workflow")
    project: Optional[str] = Field(default=None, description="Project name")
    environment: Optional[str] = Field(default=None, description="Environment name")
    version: Optional[str] = Field(default=None, description="Version number")
    source: Optional[str] = Field(default=None, description="Source system")
    destination: Optional[str] = Field(default=None, description="Destination system")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    configuration: TaskConfiguration = Field(default_factory=TaskConfiguration, description="Workflow configuration")
    integration: IntegrationConfiguration = Field(default_factory=IntegrationConfiguration, description="Integration configuration")
    communication: CommunicationConfiguration = Field(default_factory=CommunicationConfiguration, description="Communication configuration")
    development: DevelopmentConfiguration = Field(default_factory=DevelopmentConfiguration, description="Development configuration")
    tasks: List[AutomationTask] = Field(default_factory=list, description="List of tasks in the workflow")

    @validator('retry_count')
    def validate_retry_count(cls, v, values):
        """Validate that retry count doesn't exceed max retries."""
        if 'max_retries' in values and v > values['max_retries']:
            raise ValueError('retry_count cannot exceed max_retries')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v is not None and v < 1:
            raise ValueError('timeout must be at least 1 second')
        return v

    @root_validator
    def validate_dependencies(cls, values):
        """Validate workflow dependencies."""
        dependencies = values.get('dependencies', [])
        workflow_id = values.get('id')
        
        if workflow_id in dependencies:
            raise ValueError('Workflow cannot depend on itself')
        
        # Check for circular dependencies
        visited = set()
        path = set()
        
        def check_circular_dep(dep_id):
            if dep_id in path:
                raise ValueError(f'Circular dependency detected: {" -> ".join(path)} -> {dep_id}')
            if dep_id in visited:
                return
            
            visited.add(dep_id)
            path.add(dep_id)
            
            # Get dependent workflow's dependencies
            dep_workflow = cls.get_workflow(dep_id)  # This would need to be implemented
            if dep_workflow:
                for sub_dep in dep_workflow.dependencies:
                    check_circular_dep(sub_dep)
            
            path.remove(dep_id)
        
        for dep_id in dependencies:
            check_circular_dep(dep_id)
        
        return values

    async def execute(self, metrics_service: MetricsService, transaction_service: TransactionService) -> Dict[str, Any]:
        """Execute the automation workflow.
        
        Args:
            metrics_service: Service for collecting metrics
            transaction_service: Service for managing transactions
            
        Returns:
            Dict[str, Any]: The execution result.
            
        Raises:
            ValueError: If the workflow is invalid.
            RuntimeError: If the workflow execution fails.
        """
        if self.status != WorkflowStatus.PENDING:
            raise ValueError(f"Cannot execute workflow in {self.status} status")
        
        async with transaction_service.transaction() as transaction:
            try:
                # Update status and start time
                self.status = WorkflowStatus.RUNNING
                self.started_at = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                
                # Log workflow start
                logger.info(f"Starting workflow {self.id} ({self.name})")
                
                # Record metrics
                metrics_service.record_workflow_start(self.id, self.type)
                
                # Execute tasks in order
                for task in self.tasks:
                    try:
                        result = await task.execute(metrics_service, transaction_service)
                        self.output[task.id] = result
                    except Exception as e:
                        logger.error(f"Task {task.id} failed in workflow {self.id}: {str(e)}")
                        raise
                
                # Update status and completion time
                self.status = WorkflowStatus.COMPLETED
                self.completed_at = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                
                # Log workflow completion
                logger.info(f"Completed workflow {self.id} ({self.name})")
                
                # Record metrics
                metrics_service.record_workflow_completion(self.id, self.type)
                
                return {"status": "success", "result": self.result}
                
            except Exception as e:
                # Update status and error
                self.status = WorkflowStatus.FAILED
                self.error = str(e)
                self.updated_at = datetime.utcnow()
                
                # Log workflow failure
                logger.error(f"Failed workflow {self.id} ({self.name}): {str(e)}")
                
                # Record metrics
                metrics_service.record_workflow_failure(self.id, self.type, str(e))
                
                # Handle retry if applicable
                if self.retry_count < self.max_retries:
                    self.status = WorkflowStatus.RETRYING
                    self.retry_count += 1
                    self.updated_at = datetime.utcnow()
                    
                    # Log retry
                    logger.info(f"Retrying workflow {self.id} ({self.name}) - attempt {self.retry_count}")
                    
                    # Record metrics
                    metrics_service.record_workflow_retry(self.id, self.type)
                    
                    # Schedule retry
                    await asyncio.sleep(self.retry_delay)
                    return await self.execute(metrics_service, transaction_service)
                
                raise RuntimeError(f"Workflow execution failed: {str(e)}") 