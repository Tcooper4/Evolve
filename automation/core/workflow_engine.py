"""
Workflow Execution Engine

This module implements the core workflow execution engine that manages the lifecycle
of workflows, including creation, execution, monitoring, and completion.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from .step_handlers import StepHandlerFactory

class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class WorkflowStep:
    """Represents a single step in a workflow."""
    def __init__(self, 
                 step_id: str,
                 name: str,
                 action: str,
                 parameters: Dict[str, Any],
                 dependencies: List[str] = None):
        self.step_id = step_id
        self.name = name
        self.action = action
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.result = None

class Workflow:
    """Represents a complete workflow with multiple steps."""
    def __init__(self,
                 workflow_id: str,
                 name: str,
                 description: str = "",
                 steps: List[WorkflowStep] = None):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps = steps or []
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.metadata = {}

class WorkflowEngine:
    """Core workflow execution engine."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workflows: Dict[str, Workflow] = {}
        self._running = False
        self._task_queue = asyncio.Queue()
        self._results = {}

    async def start(self):
        """Start the workflow engine."""
        self._running = True
        self.logger.info("Workflow engine started")
        await self._process_queue()

    async def stop(self):
        """Stop the workflow engine."""
        self._running = False
        self.logger.info("Workflow engine stopped")

    async def create_workflow(self, workflow: Workflow) -> str:
        """Create a new workflow."""
        if workflow.workflow_id in self.workflows:
            raise ValueError(f"Workflow with ID {workflow.workflow_id} already exists")
        
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Created workflow {workflow.workflow_id}")
        return workflow.workflow_id

    async def execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            # Execute steps in dependency order
            executed_steps = set()
            while len(executed_steps) < len(workflow.steps):
                for step in workflow.steps:
                    if step.step_id in executed_steps:
                        continue
                    
                    # Check if dependencies are met
                    if all(dep in executed_steps for dep in step.dependencies):
                        await self._execute_step(workflow, step)
                        executed_steps.add(step.step_id)

            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            self.logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            raise

    async def _execute_step(self, workflow: Workflow, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        step.status = WorkflowStatus.RUNNING
        step.start_time = datetime.utcnow()
        
        try:
            # Get the appropriate handler for the step
            handler = StepHandlerFactory.get_handler(step.action)
            if not handler:
                raise ValueError(f"No handler found for action type: {step.action}")

            # Validate step parameters
            if not await handler.validate(step.parameters):
                raise ValueError(f"Invalid parameters for step {step.step_id}")

            # Execute the step
            result = await handler.execute(step.parameters)
            
            step.status = WorkflowStatus.COMPLETED
            step.end_time = datetime.utcnow()
            step.result = result
            self.logger.info(f"Step {step.step_id} completed successfully")

        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            self.logger.error(f"Step {step.step_id} failed: {str(e)}")
            raise

    async def _process_queue(self) -> None:
        """Process the workflow queue."""
        while self._running:
            try:
                workflow_id = await self._task_queue.get()
                await self.execute_workflow(workflow_id)
                self._task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing workflow: {str(e)}")

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "error": workflow.error,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "start_time": step.start_time.isoformat() if step.start_time else None,
                    "end_time": step.end_time.isoformat() if step.end_time else None,
                    "error": step.error,
                    "result": step.result
                }
                for step in workflow.steps
            ]
        } 