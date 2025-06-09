import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path
import json
from pydantic import BaseModel, Field
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry

from ..core.models.task import Task, TaskStatus, TaskPriority, TaskType
from .automation_core import AutomationCore
from .automation_tasks import AutomationTasks

logger = logging.getLogger(__name__)

class WorkflowConfig(BaseModel):
    """Configuration for workflow management."""
    max_concurrent_workflows: int = Field(default=5)
    workflow_timeout: int = Field(default=3600)
    retry_delay: int = Field(default=10)
    max_retries: int = Field(default=3)
    cleanup_interval: int = Field(default=3600)

class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    name: str
    task_type: TaskType
    parameters: Dict[str, Any]
    dependencies: List[str] = Field(default_factory=list)
    retry_on_failure: bool = True
    timeout: Optional[int] = None

class Workflow(BaseModel):
    """Workflow configuration."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AutomationWorkflows:
    """Workflow management functionality."""
    
    def __init__(self, core: AutomationCore, tasks: AutomationTasks, config_path: str = "automation/config/workflows.json"):
        """Initialize workflow management."""
        self.core = core
        self.tasks = tasks
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_cache()
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_results: Dict[str, Dict[str, Any]] = {}
        self.workflow_errors: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        
    def _load_config(self, config_path: str) -> WorkflowConfig:
        """Load workflow configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return WorkflowConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load workflow config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "workflows.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_cache(self):
        """Setup workflow result caching."""
        self.cache = TTLCache(
            maxsize=1000,
            ttl=3600
        )
        
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow."""
        try:
            async with self.lock:
                # Check concurrent workflow limit
                if len(self.running_workflows) >= self.config.max_concurrent_workflows:
                    raise ValueError("Maximum concurrent workflows reached")
                    
                # Create workflow
                workflow = Workflow(
                    id=f"workflow_{len(self.workflows) + 1}",
                    name=name,
                    description=description,
                    steps=steps,
                    metadata=metadata or {}
                )
                
                self.workflows[workflow.id] = workflow
                logger.info(f"Created workflow {workflow.id}: {name}")
                return workflow.id
                
        except Exception as e:
            logger.error(f"Failed to create workflow: {str(e)}")
            raise
            
    async def execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow."""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
                
            # Update status
            workflow.status = "running"
            workflow.updated_at = datetime.now()
            
            # Execute steps
            step_results = {}
            for step in workflow.steps:
                try:
                    # Check dependencies
                    if step.dependencies:
                        for dep in step.dependencies:
                            if dep not in step_results:
                                raise ValueError(f"Dependency {dep} not completed")
                                
                    # Create and execute task
                    task_id = await self.tasks.schedule_task(
                        name=step.name,
                        description=f"Step in workflow {workflow_id}",
                        task_type=step.task_type,
                        priority=TaskPriority.MEDIUM,
                        parameters=step.parameters
                    )
                    
                    # Wait for task completion
                    while True:
                        status = await self.core.get_task_status(task_id)
                        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            break
                        await asyncio.sleep(1)
                        
                    # Get result
                    result = await self.tasks.get_task_result(task_id)
                    if not result and status == TaskStatus.FAILED:
                        raise Exception(f"Task {task_id} failed")
                        
                    step_results[step.name] = result
                    
                except Exception as e:
                    if step.retry_on_failure:
                        logger.warning(f"Step {step.name} failed, retrying: {str(e)}")
                        continue
                    raise
                    
            # Update workflow status
            workflow.status = "completed"
            workflow.updated_at = datetime.now()
            self.workflow_results[workflow_id] = step_results
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            workflow.status = "failed"
            workflow.updated_at = datetime.now()
            self.workflow_errors[workflow_id] = str(e)
            
    async def get_workflow_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow result with caching."""
        try:
            # Try cache first
            if workflow_id in self.cache:
                return self.cache[workflow_id]
                
            # Get from results
            result = self.workflow_results.get(workflow_id)
            if result:
                self.cache[workflow_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to get workflow result {workflow_id}: {str(e)}")
            return None
            
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            async with self.lock:
                if workflow_id in self.running_workflows:
                    self.running_workflows[workflow_id].cancel()
                    del self.running_workflows[workflow_id]
                    
                    workflow = self.workflows.get(workflow_id)
                    if workflow:
                        workflow.status = "cancelled"
                        workflow.updated_at = datetime.now()
                        
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel workflow {workflow_id}: {str(e)}")
            return False
            
    async def cleanup_completed_workflows(self):
        """Cleanup completed workflows."""
        try:
            async with self.lock:
                for workflow_id, task in self.running_workflows.items():
                    if task.done():
                        del self.running_workflows[workflow_id]
                        if workflow_id in self.cache:
                            del self.cache[workflow_id]
                            
        except Exception as e:
            logger.error(f"Failed to cleanup workflows: {str(e)}")
            
    async def get_running_workflows(self) -> List[str]:
        """Get list of running workflow IDs."""
        return list(self.running_workflows.keys())
        
    async def get_workflow_error(self, workflow_id: str) -> Optional[str]:
        """Get workflow error message."""
        return self.workflow_errors.get(workflow_id)
        
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cancel all running workflows
            for workflow_id in list(self.running_workflows.keys()):
                await self.cancel_workflow(workflow_id)
                
            # Clear caches
            self.cache.clear()
            self.workflow_results.clear()
            self.workflow_errors.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise 