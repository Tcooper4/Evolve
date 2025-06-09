import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
from unittest.mock import Mock, patch, AsyncMock

from ..api.task_api import app, TaskCreate, TaskResponse, TaskUpdate
from ..core.orchestrator import Task

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_orchestrator():
    with patch("automation.api.task_api.get_orchestrator") as mock:
        orchestrator = Mock()
        orchestrator.create_task = AsyncMock()
        orchestrator.get_task = AsyncMock()
        orchestrator.get_tasks = AsyncMock()
        orchestrator.update_task = AsyncMock()
        orchestrator.delete_task = AsyncMock()
        orchestrator.execute_task = AsyncMock()
        orchestrator.get_task_metrics = AsyncMock()
        orchestrator.get_task_dependencies = AsyncMock()
        mock.return_value = orchestrator
        yield orchestrator

@pytest.fixture
def sample_task():
    return Task(
        task_id="test_task_1",
        name="Test Task",
        task_type="data_collection",
        status="pending",
        priority=1,
        parameters={"param1": "value1"},
        dependencies=[],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def test_create_task(test_client, mock_orchestrator, sample_task):
    """Test task creation endpoint."""
    # Setup mock
    mock_orchestrator.create_task.return_value = sample_task.task_id
    mock_orchestrator.get_task.return_value = sample_task
    
    # Test data
    task_data = {
        "name": "Test Task",
        "task_type": "data_collection",
        "priority": 1,
        "parameters": {"param1": "value1"},
        "dependencies": []
    }
    
    # Make request
    response = test_client.post("/tasks", json=task_data)
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == sample_task.task_id
    assert data["name"] == sample_task.name
    assert data["task_type"] == sample_task.task_type
    
    # Verify orchestrator calls
    mock_orchestrator.create_task.assert_called_once()
    mock_orchestrator.get_task.assert_called_once()

def test_get_task(test_client, mock_orchestrator, sample_task):
    """Test get task endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    
    # Make request
    response = test_client.get(f"/tasks/{sample_task.task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == sample_task.task_id
    assert data["name"] == sample_task.name
    
    # Verify orchestrator call
    mock_orchestrator.get_task.assert_called_once_with(sample_task.task_id)

def test_get_task_not_found(test_client, mock_orchestrator):
    """Test get task endpoint with non-existent task."""
    # Setup mock
    mock_orchestrator.get_task.return_value = None
    
    # Make request
    response = test_client.get("/tasks/non_existent_task")
    
    # Verify response
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"

def test_list_tasks(test_client, mock_orchestrator, sample_task):
    """Test list tasks endpoint."""
    # Setup mock
    mock_orchestrator.get_tasks.return_value = [sample_task]
    
    # Make request
    response = test_client.get("/tasks")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["task_id"] == sample_task.task_id
    
    # Test with filters
    response = test_client.get("/tasks?status=pending&task_type=data_collection")
    assert response.status_code == 200

def test_update_task(test_client, mock_orchestrator, sample_task):
    """Test update task endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    
    # Test data
    update_data = {
        "status": "running",
        "priority": 2
    }
    
    # Make request
    response = test_client.patch(f"/tasks/{sample_task.task_id}", json=update_data)
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["priority"] == 2
    
    # Verify orchestrator calls
    mock_orchestrator.get_task.assert_called_once()
    mock_orchestrator.update_task.assert_called_once()

def test_delete_task(test_client, mock_orchestrator, sample_task):
    """Test delete task endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    
    # Make request
    response = test_client.delete(f"/tasks/{sample_task.task_id}")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["message"] == f"Task {sample_task.task_id} deleted successfully"
    
    # Verify orchestrator calls
    mock_orchestrator.get_task.assert_called_once()
    mock_orchestrator.delete_task.assert_called_once_with(sample_task.task_id)

def test_execute_task(test_client, mock_orchestrator, sample_task):
    """Test execute task endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    
    # Make request
    response = test_client.post(f"/tasks/{sample_task.task_id}/execute")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["message"] == f"Task {sample_task.task_id} execution started"
    
    # Verify orchestrator calls
    mock_orchestrator.get_task.assert_called_once()
    mock_orchestrator.execute_task.assert_called_once_with(sample_task)

def test_get_task_metrics(test_client, mock_orchestrator, sample_task):
    """Test get task metrics endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    mock_orchestrator.get_task_metrics.return_value = {
        "execution_time": 10.5,
        "memory_usage": 100,
        "status": "completed"
    }
    
    # Make request
    response = test_client.get(f"/tasks/{sample_task.task_id}/metrics")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "execution_time" in data
    assert "memory_usage" in data
    assert "status" in data
    
    # Verify orchestrator calls
    mock_orchestrator.get_task.assert_called_once()
    mock_orchestrator.get_task_metrics.assert_called_once_with(sample_task.task_id)

def test_get_task_dependencies(test_client, mock_orchestrator, sample_task):
    """Test get task dependencies endpoint."""
    # Setup mock
    mock_orchestrator.get_task.return_value = sample_task
    mock_orchestrator.get_task_dependencies.return_value = ["dep1", "dep2"]
    
    # Make request
    response = test_client.get(f"/tasks/{sample_task.task_id}/dependencies")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "dependencies" in data
    assert len(data["dependencies"]) == 2
    
    # Verify orchestrator calls
    mock_orchestrator.get_task.assert_called_once()
    mock_orchestrator.get_task_dependencies.assert_called_once_with(sample_task.task_id)

def test_error_handling(test_client, mock_orchestrator):
    """Test error handling in API endpoints."""
    # Test create task with invalid data
    response = test_client.post("/tasks", json={})
    assert response.status_code == 422  # Validation error
    
    # Test update task with invalid task ID
    mock_orchestrator.get_task.return_value = None
    response = test_client.patch("/tasks/invalid_id", json={"status": "running"})
    assert response.status_code == 404
    
    # Test execute task with error
    mock_orchestrator.get_task.return_value = sample_task
    mock_orchestrator.execute_task.side_effect = Exception("Execution failed")
    response = test_client.post(f"/tasks/{sample_task.task_id}/execute")
    assert response.status_code == 400
    assert "Execution failed" in response.json()["detail"] 