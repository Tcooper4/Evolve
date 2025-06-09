"""
Tests for the workflow engine.
"""

import pytest
import asyncio
from datetime import datetime
from automation.core.workflow_engine import (
    WorkflowEngine,
    Workflow,
    WorkflowStep,
    WorkflowStatus
)

@pytest.fixture
async def workflow_engine():
    """Create a workflow engine for testing."""
    engine = WorkflowEngine()
    await engine.start()
    yield engine
    await engine.stop()

@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    steps = [
        WorkflowStep(
            step_id="step1",
            name="First Step",
            action="command",
            parameters={"command": "echo 'Hello'"}
        ),
        WorkflowStep(
            step_id="step2",
            name="Second Step",
            action="api",
            parameters={"url": "http://example.com"},
            dependencies=["step1"]
        ),
        WorkflowStep(
            step_id="step3",
            name="Third Step",
            action="notification",
            parameters={
                "channel": "email",
                "message": "Workflow completed",
                "recipients": ["test@example.com"]
            },
            dependencies=["step2"]
        )
    ]
    
    return Workflow(
        workflow_id="test_workflow",
        name="Test Workflow",
        description="A workflow for testing",
        steps=steps
    )

@pytest.mark.asyncio
async def test_create_workflow(workflow_engine, sample_workflow):
    """Test workflow creation."""
    workflow_id = await workflow_engine.create_workflow(sample_workflow)
    assert workflow_id == "test_workflow"
    assert workflow_id in workflow_engine.workflows

@pytest.mark.asyncio
async def test_execute_workflow(workflow_engine, sample_workflow):
    """Test workflow execution."""
    await workflow_engine.create_workflow(sample_workflow)
    await workflow_engine.execute_workflow("test_workflow")
    
    workflow = workflow_engine.workflows["test_workflow"]
    assert workflow.status == WorkflowStatus.COMPLETED
    assert workflow.started_at is not None
    assert workflow.completed_at is not None
    
    for step in workflow.steps:
        assert step.status == WorkflowStatus.COMPLETED
        assert step.start_time is not None
        assert step.end_time is not None
        assert step.result is not None

@pytest.mark.asyncio
async def test_workflow_with_invalid_step(workflow_engine):
    """Test workflow execution with an invalid step."""
    steps = [
        WorkflowStep(
            step_id="invalid_step",
            name="Invalid Step",
            action="invalid_action",
            parameters={}
        )
    ]
    
    workflow = Workflow(
        workflow_id="invalid_workflow",
        name="Invalid Workflow",
        steps=steps
    )
    
    await workflow_engine.create_workflow(workflow)
    
    with pytest.raises(ValueError):
        await workflow_engine.execute_workflow("invalid_workflow")
    
    workflow = workflow_engine.workflows["invalid_workflow"]
    assert workflow.status == WorkflowStatus.FAILED
    assert workflow.error is not None

@pytest.mark.asyncio
async def test_workflow_status(workflow_engine, sample_workflow):
    """Test workflow status retrieval."""
    await workflow_engine.create_workflow(sample_workflow)
    await workflow_engine.execute_workflow("test_workflow")
    
    status = workflow_engine.get_workflow_status("test_workflow")
    assert status["workflow_id"] == "test_workflow"
    assert status["status"] == WorkflowStatus.COMPLETED.value
    assert len(status["steps"]) == 3
    
    for step in status["steps"]:
        assert step["status"] == WorkflowStatus.COMPLETED.value
        assert step["start_time"] is not None
        assert step["end_time"] is not None
        assert step["result"] is not None

@pytest.mark.asyncio
async def test_workflow_dependencies(workflow_engine):
    """Test workflow step dependencies."""
    steps = [
        WorkflowStep(
            step_id="step1",
            name="First Step",
            action="command",
            parameters={"command": "echo 'Hello'"}
        ),
        WorkflowStep(
            step_id="step2",
            name="Second Step",
            action="command",
            parameters={"command": "echo 'World'"},
            dependencies=["step1"]
        ),
        WorkflowStep(
            step_id="step3",
            name="Third Step",
            action="command",
            parameters={"command": "echo '!'"},
            dependencies=["step2"]
        )
    ]
    
    workflow = Workflow(
        workflow_id="dependency_workflow",
        name="Dependency Workflow",
        steps=steps
    )
    
    await workflow_engine.create_workflow(workflow)
    await workflow_engine.execute_workflow("dependency_workflow")
    
    workflow = workflow_engine.workflows["dependency_workflow"]
    assert workflow.status == WorkflowStatus.COMPLETED
    
    # Verify step execution order
    step1 = next(step for step in workflow.steps if step.step_id == "step1")
    step2 = next(step for step in workflow.steps if step.step_id == "step2")
    step3 = next(step for step in workflow.steps if step.step_id == "step3")
    
    assert step1.end_time < step2.start_time
    assert step2.end_time < step3.start_time 