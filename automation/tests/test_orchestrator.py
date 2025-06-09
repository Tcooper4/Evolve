import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from typing import Dict, Any

from automation.core.orchestrator import Orchestrator, Task
from .base_test import BaseTest

@pytest.fixture
def test_config(tmp_path):
    config = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None
        },
        "ray": {
            "address": "auto",
            "namespace": "test",
            "runtime_env": {
                "working_dir": ".",
                "py_modules": ["trading", "automation"]
            }
        },
        "kubernetes": {
            "in_cluster": False,
            "namespace": "test",
            "config_path": "~/.kube/config"
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return str(config_path)

@pytest.fixture
def mock_redis():
    with patch('redis.Redis') as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_ray():
    with patch('ray.init') as mock_init, \
         patch('ray.serve.start') as mock_serve:
        yield mock_init, mock_serve

@pytest.fixture
def mock_kubernetes():
    with patch('kubernetes.config.load_kube_config') as mock_config, \
         patch('kubernetes.client.CoreV1Api') as mock_api:
        mock_instance = MagicMock()
        mock_api.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def orchestrator(test_config, mock_redis, mock_ray, mock_kubernetes):
    return Orchestrator(config_path=test_config)

@pytest.fixture
def sample_task():
    return Task(
        id="test_task_001",
        name="Test Task",
        type="model_training",
        parameters={
            "model_type": "lstm",
            "epochs": 10,
            "batch_size": 32
        }
    )

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization."""
    assert orchestrator.config is not None
    assert orchestrator.redis_client is not None
    assert orchestrator.tasks == {}
    assert orchestrator.running is False

@pytest.mark.asyncio
async def test_create_task(orchestrator, sample_task):
    """Test task creation."""
    task_id = await orchestrator.create_task(sample_task)
    assert task_id == sample_task.id
    assert sample_task.id in orchestrator.tasks
    orchestrator.redis_client.hset.assert_called_once()
    orchestrator.redis_client.zadd.assert_called_once()

@pytest.mark.asyncio
async def test_get_task(orchestrator, sample_task):
    """Test task retrieval."""
    # Setup mock response
    orchestrator.redis_client.hget.return_value = sample_task.json()
    
    # Get task
    task = await orchestrator.get_task(sample_task.id)
    assert task is not None
    assert task.id == sample_task.id
    assert task.name == sample_task.name
    orchestrator.redis_client.hget.assert_called_once()

@pytest.mark.asyncio
async def test_update_task_status(orchestrator, sample_task):
    """Test task status update."""
    # Setup mock response
    orchestrator.redis_client.hget.return_value = sample_task.json()
    
    # Update status
    await orchestrator.update_task_status(sample_task.id, "running")
    orchestrator.redis_client.hset.assert_called_once()

@pytest.mark.asyncio
async def test_execute_task(orchestrator, sample_task):
    """Test task execution."""
    # Setup mock handler
    mock_handler = AsyncMock()
    orchestrator._get_task_handler = MagicMock(return_value=mock_handler)
    
    # Execute task
    await orchestrator.execute_task(sample_task)
    mock_handler.assert_called_once_with(sample_task)
    orchestrator.redis_client.hset.assert_called()

@pytest.mark.asyncio
async def test_process_dependencies(orchestrator):
    """Test dependency processing."""
    # Create parent task
    parent_task = Task(
        id="parent_task",
        name="Parent Task",
        type="data_collection"
    )
    
    # Create dependent task
    child_task = Task(
        id="child_task",
        name="Child Task",
        type="model_training",
        dependencies=["parent_task"]
    )
    
    # Add tasks to orchestrator
    orchestrator.tasks = {
        parent_task.id: parent_task,
        child_task.id: child_task
    }
    
    # Process dependencies
    await orchestrator._process_dependencies(parent_task)
    assert child_task.status == "completed"

@pytest.mark.asyncio
async def test_start_stop(orchestrator):
    """Test orchestrator start and stop."""
    # Start orchestrator
    start_task = asyncio.create_task(orchestrator.start())
    
    # Wait a bit
    await asyncio.sleep(0.1)
    
    # Stop orchestrator
    await orchestrator.stop()
    await start_task
    
    assert not orchestrator.running

@pytest.mark.asyncio
async def test_task_handlers(orchestrator, sample_task):
    """Test task handlers."""
    # Test data collection handler
    result = await orchestrator._handle_data_collection(sample_task)
    assert result is None
    
    # Test model training handler
    result = await orchestrator._handle_model_training(sample_task)
    assert result is None
    
    # Test model evaluation handler
    result = await orchestrator._handle_model_evaluation(sample_task)
    assert result is None
    
    # Test model deployment handler
    result = await orchestrator._handle_model_deployment(sample_task)
    assert result is None
    
    # Test backtesting handler
    result = await orchestrator._handle_backtesting(sample_task)
    assert result is None
    
    # Test optimization handler
    result = await orchestrator._handle_optimization(sample_task)
    assert result is None

@pytest.mark.asyncio
async def test_error_handling(orchestrator, sample_task):
    """Test error handling."""
    # Setup mock to raise exception
    orchestrator.redis_client.hset.side_effect = Exception("Test error")
    
    # Test error in create_task
    with pytest.raises(Exception):
        await orchestrator.create_task(sample_task)
    
    # Test error in get_task
    orchestrator.redis_client.hget.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        await orchestrator.get_task(sample_task.id)
    
    # Test error in update_task_status
    with pytest.raises(Exception):
        await orchestrator.update_task_status(sample_task.id, "failed", "Test error")

@pytest.mark.asyncio
async def test_invalid_task_type(orchestrator):
    """Test handling of invalid task type."""
    invalid_task = Task(
        id="invalid_task",
        name="Invalid Task",
        type="invalid_type"
    )
    
    # Test execution of invalid task
    await orchestrator.execute_task(invalid_task)
    assert invalid_task.status == "failed"
    assert "No handler found" in invalid_task.error

class TestOrchestrator(BaseTest):
    """Test suite for the orchestrator component."""
    
    @pytest.mark.asyncio
    async def test_task_scheduling(self):
        """Test task scheduling functionality."""
        # Create test task
        task = {
            "type": "feature_implementation",
            "description": "Test feature implementation",
            "priority": 1
        }
        
        # Schedule task
        task_id = await self.orchestrator.schedule_task(task)
        
        # Assert task was scheduled
        self._assert_task_status(task_id, "scheduled")
        
        # Verify task in Redis
        task_data = self.redis_client.get(f"task:{task_id}")
        assert task_data is not None
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test task execution functionality."""
        # Create and schedule test task
        task = {
            "type": "model_training",
            "description": "Test model training",
            "priority": 2
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Execute task
        await self.orchestrator.coordinate_agents(task_id)
        
        # Assert task was completed
        self._assert_task_status(task_id, "completed")
        
        # Verify task progress
        status = self.orchestrator.get_task_status(task_id)
        assert status["progress"] == 100
        assert len(status["steps_completed"]) == 4  # All steps completed
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling functionality."""
        # Create task that will fail
        task = {
            "type": "invalid_type",
            "description": "Test error handling",
            "priority": 1
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Attempt to execute task
        with pytest.raises(ValueError):
            await self.orchestrator.coordinate_agents(task_id)
        
        # Assert task failed
        self._assert_task_status(task_id, "failed")
        
        # Verify error was recorded
        status = self.orchestrator.get_task_status(task_id)
        assert len(status["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_tasks(self):
        """Test handling of concurrent tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = {
                "type": "feature_implementation",
                "description": f"Test concurrent task {i}",
                "priority": 1
            }
            task_id = await self.orchestrator.schedule_task(task)
            tasks.append(task_id)
        
        # Execute tasks concurrently
        await asyncio.gather(*[
            self.orchestrator.coordinate_agents(task_id)
            for task_id in tasks
        ])
        
        # Verify all tasks completed
        for task_id in tasks:
            self._assert_task_status(task_id, "completed")
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        # Get initial health
        initial_health = self.orchestrator.get_system_health()
        
        # Schedule some tasks
        for i in range(5):
            task = {
                "type": "feature_implementation",
                "description": f"Test health monitoring {i}",
                "priority": 1
            }
            self.orchestrator.schedule_task(task)
        
        # Get updated health
        updated_health = self.orchestrator.get_system_health()
        
        # Verify metrics were updated
        assert updated_health["queued_tasks"] > initial_health["queued_tasks"]
    
    @pytest.mark.asyncio
    async def test_websocket_updates(self):
        """Test WebSocket real-time updates."""
        # Create test task
        task = {
            "type": "model_training",
            "description": "Test WebSocket updates",
            "priority": 1
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Connect to WebSocket
        with self.test_client.websocket_connect("/ws") as websocket:
            # Start task execution
            execution_task = asyncio.create_task(
                self.orchestrator.coordinate_agents(task_id)
            )
            
            # Receive updates
            updates = []
            for _ in range(3):  # Expect at least 3 updates
                data = websocket.receive_json()
                updates.append(data)
            
            # Verify updates
            assert any(update["task_id"] == task_id for update in updates)
            assert any(update["status"] == "completed" for update in updates)
            
            # Wait for execution to complete
            await execution_task
    
    def test_task_dependencies(self):
        """Test task dependency handling."""
        # Create dependent tasks
        task1 = {
            "type": "data_processing",
            "description": "Process data",
            "priority": 1
        }
        task2 = {
            "type": "model_training",
            "description": "Train model",
            "priority": 2,
            "dependencies": ["task1"]  # Depends on task1
        }
        
        # Schedule tasks
        task1_id = self.orchestrator.schedule_task(task1)
        task2_id = self.orchestrator.schedule_task(task2)
        
        # Verify dependency was recorded
        task2_data = self.redis_client.get(f"task:{task2_id}")
        assert "task1" in task2_data["dependencies"]
    
    def test_task_prioritization(self):
        """Test task prioritization."""
        # Create tasks with different priorities
        tasks = []
        for priority in [3, 1, 2]:
            task = {
                "type": "feature_implementation",
                "description": f"Test priority {priority}",
                "priority": priority
            }
            task_id = self.orchestrator.schedule_task(task)
            tasks.append(task_id)
        
        # Get all tasks
        all_tasks = self.orchestrator.get_all_tasks()
        
        # Verify tasks are ordered by priority
        priorities = [task["priority"] for task in all_tasks]
        assert priorities == sorted(priorities)
    
    def test_resource_management(self):
        """Test resource management."""
        # Get initial resource usage
        initial_health = self.orchestrator.get_system_health()
        
        # Create resource-intensive task
        task = {
            "type": "model_training",
            "description": "Test resource management",
            "priority": 1,
            "resource_requirements": {
                "cpu": 80,
                "memory": 1024,
                "gpu": 1
            }
        }
        task_id = self.orchestrator.schedule_task(task)
        
        # Execute task
        self.orchestrator.coordinate_agents(task_id)
        
        # Get updated resource usage
        updated_health = self.orchestrator.get_system_health()
        
        # Verify resource usage was monitored
        assert "cpu_usage" in updated_health
        assert "memory_usage" in updated_health
        assert "gpu_usage" in updated_health

if __name__ == "__main__":
    pytest.main([__file__]) 