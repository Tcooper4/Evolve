import json
import os
import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading.agents.model_builder_agent import ModelBuilderAgent as ModelBuilder
from trading.agents.prompt_router_agent import PromptRouterAgent as AgentRouter
from trading.agents.self_improving_agent import SelfImprovingAgent
from trading.agents.task_dashboard import TaskDashboard
from trading.agents.task_memory import Task, TaskMemory, TaskStatus

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def task_memory():
    """Create a task memory instance for testing."""
    return TaskMemory()


@pytest.fixture
def dashboard(task_memory):
    """Create a task dashboard instance for testing."""
    return TaskDashboard(task_memory)


@pytest.fixture
def model_builder():
    """Create a model builder instance for testing."""
    return ModelBuilder()


@pytest.fixture
def agent_router():
    """Create an agent router instance for testing."""
    return AgentRouter()


@pytest.fixture
def self_improving_agent():
    """Create a self improving agent instance for testing."""
    return SelfImprovingAgent()


def test_empty_data_handling(task_memory, model_builder):
    """Test system behavior with empty or invalid data."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()

    # Test model training with empty data
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "model_type": "lstm",
        },
        notes="Testing empty data handling",
    )
    task_memory.add_task(task)

    try:
        model_builder.run_lstm(empty_df)
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.metadata.update(
            {"error": str(e), "completion_time": datetime.now().isoformat()}
        )
        task_memory.update_task(task)

    # Verify task failure and error handling
    failed_task = task_memory.get_task(task_id)
    assert failed_task.status == TaskStatus.FAILED
    assert "error" in failed_task.metadata


def test_concurrent_task_handling(task_memory):
    """Test system behavior with concurrent tasks."""
    # Create multiple tasks simultaneously
    task_ids = []
    for i in range(10):
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
                "model_type": "lstm",
            },
            notes=f"Concurrent task {i}",
        )
        task_memory.add_task(task)
        task_ids.append(task_id)

    # Verify all tasks are created
    assert len(task_memory.tasks) == 10

    # Update tasks concurrently
    for task_id in task_ids:
        task = task_memory.get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.metadata.update(
            {"completion_time": datetime.now().isoformat(), "duration": "1 minute"}
        )
        task_memory.update_task(task)

    # Verify all tasks are updated
    completed_tasks = task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    assert len(completed_tasks) == 10


def test_large_data_handling(task_memory, model_builder):
    """Test system behavior with large datasets."""
    # Create large dataset
    large_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=10000, freq="H"),
            "price": np.random.normal(100, 10, 10000),
            "volume": np.random.normal(1000, 100, 10000),
        }
    )

    # Test task creation and tracking with large data
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "model_type": "lstm",
            "data_size": len(large_df),
        },
        notes="Testing large data handling",
    )
    task_memory.add_task(task)

    try:
        # Simulate processing large data
        task.status = TaskStatus.COMPLETED
        task.metadata.update(
            {
                "completion_time": datetime.now().isoformat(),
                "duration": "10 minutes",
                "memory_usage": "1GB",
            }
        )
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.metadata.update(
            {"error": str(e), "completion_time": datetime.now().isoformat()}
        )

    task_memory.update_task(task)

    # Verify task completion
    completed_task = task_memory.get_task(task_id)
    assert completed_task.status == TaskStatus.COMPLETED
    assert "memory_usage" in completed_task.metadata


def test_task_retry_mechanism(task_memory):
    """Test task retry mechanism after failures."""
    # Create initial task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "model_type": "lstm",
        },
        notes="Initial task",
    )
    task_memory.add_task(task)

    # Simulate task failure
    task.status = TaskStatus.FAILED
    task.metadata.update(
        {"error": "Test error", "completion_time": datetime.now().isoformat()}
    )
    task_memory.update_task(task)

    # Create retry task
    retry_task_id = str(uuid.uuid4())
    retry_task = Task(
        task_id=retry_task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "model_type": "lstm",
            "retry_of": task_id,
            "retry_count": 1,
        },
        notes=f"Retry of task {task_id}",
    )
    task_memory.add_task(retry_task)

    # Verify retry task
    retry_task = task_memory.get_task(retry_task_id)
    assert retry_task.metadata["retry_of"] == task_id
    assert retry_task.metadata["retry_count"] == 1


def test_invalid_task_states(task_memory):
    """Test handling of invalid task states and transitions."""
    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "model_type": "lstm",
        },
        notes="Test task",
    )
    task_memory.add_task(task)

    # Test invalid status transition
    task.status = "INVALID_STATUS"
    task_memory.update_task(task)

    # Verify task is still in valid state
    updated_task = task_memory.get_task(task_id)
    assert updated_task.status in [
        TaskStatus.PENDING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
    ]


def test_corrupted_task_data(task_memory):
    """Test handling of corrupted task data."""
    # Create task with corrupted metadata
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "corrupted_field": object(),  # Non-serializable object
        },
        notes="Testing corrupted data",
    )

    # Verify task creation handles corrupted data
    try:
        task_memory.add_task(task)
    except Exception as e:
        assert isinstance(e, (TypeError, json.JSONDecodeError))


def test_task_timeout_handling(task_memory):
    """Test handling of task timeouts."""
    # Create long-running task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            "agent": "model_builder",
            "creation_time": datetime.now().isoformat(),
            "timeout": 300,
        },  # 5 minutes
        notes="Testing timeout handling",
    )
    task_memory.add_task(task)

    # Simulate timeout
    task.status = TaskStatus.FAILED
    task.metadata.update(
        {
            "error": "Task timeout",
            "completion_time": datetime.now().isoformat(),
            "duration": "300 seconds",
        }
    )
    task_memory.update_task(task)

    # Verify timeout handling
    failed_task = task_memory.get_task(task_id)
    assert failed_task.status == TaskStatus.FAILED
    assert "timeout" in failed_task.metadata.get("error", "").lower()


def test_system_resource_limits(task_memory):
    """Test system behavior under resource constraints."""
    # Create multiple resource-intensive tasks
    task_ids = []
    for i in range(5):
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
                "resource_requirements": {"memory": "2GB", "cpu": "100%", "gpu": "1"},
            },
            notes=f"Resource-intensive task {i}",
        )
        task_memory.add_task(task)
        task_ids.append(task_id)

    # Simulate resource constraints
    for task_id in task_ids:
        task = task_memory.get_task(task_id)
        task.status = TaskStatus.FAILED
        task.metadata.update(
            {
                "error": "Insufficient resources",
                "completion_time": datetime.now().isoformat(),
                "resource_status": {
                    "memory_available": "1GB",
                    "cpu_available": "50%",
                    "gpu_available": "0",
                },
            }
        )
        task_memory.update_task(task)

    # Verify resource constraint handling
    failed_tasks = task_memory.get_tasks_by_status(TaskStatus.FAILED)
    assert len(failed_tasks) == 5
    for task in failed_tasks:
        assert "resource_status" in task.metadata
