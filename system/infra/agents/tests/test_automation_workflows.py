"""
Test suite for automation workflows.

This module contains test cases for:
- Workflow creation and management
- Workflow execution
- Workflow state tracking
- Error handling and recovery
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import os

from services.automation_workflows import WorkflowManager, Workflow, WorkflowState
from logs.automation_logging import AutomationLogger

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock(spec=AutomationLogger)

@pytest.fixture
def workflow_manager(mock_logger):
    """Create a WorkflowManager instance with mock logger."""
    return WorkflowManager(logger=mock_logger)

@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    return {
        "id": "test_workflow",
        "name": "Test Workflow",
        "description": "A test workflow",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "action": "test_action",
                "params": {"param1": "value1"}
            },
            {
                "id": "step2",
                "name": "Step 2",
                "action": "test_action",
                "params": {"param2": "value2"}
            }
        ],
        "schedule": {
            "type": "interval",
            "value": 3600
        }
    }

def test_workflow_creation(workflow_manager, sample_workflow):
    """Test workflow creation."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Verify workflow was created
    assert workflow.id == sample_workflow["id"]
    assert workflow.name == sample_workflow["name"]
    assert workflow.description == sample_workflow["description"]
    assert len(workflow.steps) == len(sample_workflow["steps"])
    assert workflow.schedule == sample_workflow["schedule"]
    
    # Verify workflow was stored
    assert workflow.id in workflow_manager.workflows

def test_workflow_creation_validation(workflow_manager):
    """Test workflow creation validation."""
    # Create invalid workflow
    invalid_workflow = {
        "id": "test_workflow",
        "name": "Test Workflow"
        # Missing required fields
    }
    
    # Verify validation error
    with pytest.raises(ValueError):
        workflow_manager.create_workflow(invalid_workflow)

def test_workflow_update(workflow_manager, sample_workflow):
    """Test workflow update."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Update workflow
    updates = {
        "name": "Updated Workflow",
        "description": "Updated description"
    }
    updated_workflow = workflow_manager.update_workflow(workflow.id, updates)
    
    # Verify updates
    assert updated_workflow.name == updates["name"]
    assert updated_workflow.description == updates["description"]
    assert updated_workflow.id == workflow.id

def test_workflow_deletion(workflow_manager, sample_workflow):
    """Test workflow deletion."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Delete workflow
    workflow_manager.delete_workflow(workflow.id)
    
    # Verify workflow was deleted
    assert workflow.id not in workflow_manager.workflows

@pytest.mark.asyncio
async def test_workflow_execution(workflow_manager, sample_workflow, mock_logger):
    """Test workflow execution."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Mock step execution
    workflow_manager.execute_step = AsyncMock(return_value=True)
    
    # Execute workflow
    await workflow_manager.execute_workflow(workflow.id)
    
    # Verify steps were executed
    assert workflow_manager.execute_step.call_count == len(workflow.steps)
    
    # Verify execution was logged
    mock_logger.info.assert_called_with(
        "Workflow execution completed",
        workflow_id=workflow.id
    )

@pytest.mark.asyncio
async def test_workflow_execution_with_error(workflow_manager, sample_workflow, mock_logger):
    """Test workflow execution with error."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Mock step execution with error
    workflow_manager.execute_step = AsyncMock(side_effect=Exception("Test error"))
    
    # Execute workflow
    await workflow_manager.execute_workflow(workflow.id)
    
    # Verify error was logged
    mock_logger.error.assert_called_with(
        "Workflow execution failed",
        workflow_id=workflow.id,
        error="Test error"
    )
    
    # Verify workflow state
    assert workflow.state == WorkflowState.FAILED

@pytest.mark.asyncio
async def test_workflow_scheduling(workflow_manager, sample_workflow):
    """Test workflow scheduling."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Mock workflow execution
    workflow_manager.execute_workflow = AsyncMock()
    
    # Start scheduling
    await workflow_manager.start_scheduling()
    
    # Wait for scheduled execution
    await asyncio.sleep(0.1)
    
    # Verify workflow was executed
    workflow_manager.execute_workflow.assert_called_with(workflow.id)

def test_workflow_state_tracking(workflow_manager, sample_workflow):
    """Test workflow state tracking."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Verify initial state
    assert workflow.state == WorkflowState.CREATED
    
    # Update state
    workflow_manager.update_workflow_state(workflow.id, WorkflowState.RUNNING)
    assert workflow.state == WorkflowState.RUNNING
    
    # Update state again
    workflow_manager.update_workflow_state(workflow.id, WorkflowState.COMPLETED)
    assert workflow.state == WorkflowState.COMPLETED

def test_workflow_history(workflow_manager, sample_workflow):
    """Test workflow history tracking."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Add history entries
    workflow_manager.add_workflow_history(
        workflow.id,
        "Test event",
        {"data": "test"}
    )
    
    # Verify history
    assert len(workflow.history) == 1
    assert workflow.history[0]["event"] == "Test event"
    assert workflow.history[0]["data"] == {"data": "test"}

@pytest.mark.asyncio
async def test_workflow_recovery(workflow_manager, sample_workflow, mock_logger):
    """Test workflow recovery."""
    # Create workflow
    workflow = workflow_manager.create_workflow(sample_workflow)
    
    # Set workflow to failed state
    workflow_manager.update_workflow_state(workflow.id, WorkflowState.FAILED)
    
    # Mock step execution
    workflow_manager.execute_step = AsyncMock(return_value=True)
    
    # Recover workflow
    await workflow_manager.recover_workflow(workflow.id)
    
    # Verify workflow was recovered
    assert workflow.state == WorkflowState.COMPLETED
    
    # Verify recovery was logged
    mock_logger.info.assert_called_with(
        "Workflow recovered",
        workflow_id=workflow.id
    )

def test_workflow_validation(workflow_manager):
    """Test workflow validation."""
    # Test valid workflow
    valid_workflow = {
        "id": "test_workflow",
        "name": "Test Workflow",
        "description": "Test description",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "action": "test_action",
                "params": {}
            }
        ],
        "schedule": {
            "type": "interval",
            "value": 3600
        }
    }
    assert workflow_manager.validate_workflow(valid_workflow)
    
    # Test invalid workflow
    invalid_workflow = {
        "id": "test_workflow",
        "name": "Test Workflow"
        # Missing required fields
    }
    assert not workflow_manager.validate_workflow(invalid_workflow) 