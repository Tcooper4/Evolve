import os
import sys
from unittest.mock import patch

import pytest

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
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        Task(
            task_id="test_task_1",
            type="model_training",
            status=TaskStatus.COMPLETED,
            metadata={
                "agent": "model_builder",
                "creation_time": "2024-01-01T10:00:00",
                "completion_time": "2024-01-01T10:05:00",
                "duration": "5 minutes",
                "metrics": {"mse": 0.0012, "sharpe_ratio": 1.5, "max_drawdown": 0.15},
            },
            notes="Test task 1 completed successfully",
        ),
        Task(
            task_id="test_task_2",
            type="forecast",
            status=TaskStatus.FAILED,
            metadata={
                "agent": "forecast_agent",
                "creation_time": "2024-01-01T11:00:00",
                "error": "Invalid input data",
            },
            notes="Test task 2 failed due to invalid data",
        ),
        Task(
            task_id="test_task_3",
            type="strategy",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "strategy_agent",
                "creation_time": "2024-01-01T12:00:00",
            },
            notes="Test task 3 is pending",
        ),
    ]


@pytest.fixture
def populated_task_memory(task_memory, sample_tasks):
    """Create a task memory populated with sample tasks."""
    for task in sample_tasks:
        task_memory.add_task(task)
    return task_memory


def test_task_metrics_calculation(populated_task_memory):
    """Test the calculation of task metrics."""
    # Get metrics
    total_tasks = len(populated_task_memory.tasks)
    completed_tasks = len(
        populated_task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    )
    failed_tasks = len(populated_task_memory.get_tasks_by_status(TaskStatus.FAILED))
    pending_tasks = len(populated_task_memory.get_tasks_by_status(TaskStatus.PENDING))

    # Verify metrics
    assert total_tasks == 3
    assert completed_tasks == 1
    assert failed_tasks == 1
    assert pending_tasks == 1


def test_task_filtering(populated_task_memory):
    """Test task filtering functionality."""
    # Test status filter
    completed_tasks = populated_task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    assert len(completed_tasks) == 1
    assert completed_tasks[0].task_id == "test_task_1"

    # Test search filter
    model_tasks = [t for t in populated_task_memory.tasks if "model" in t.type.lower()]
    assert len(model_tasks) == 1
    assert model_tasks[0].task_id == "test_task_1"


def test_task_details(populated_task_memory):
    """Test task details retrieval and display."""
    task = populated_task_memory.get_task("test_task_1")

    # Verify task details
    assert task is not None
    assert task.type == "model_training"
    assert task.status == TaskStatus.COMPLETED
    assert task.metadata["agent"] == "model_builder"
    assert "metrics" in task.metadata
    assert task.notes == "Test task 1 completed successfully"


def test_status_distribution(populated_task_memory):
    """Test task status distribution calculation."""
    status_counts = {
        status: len(populated_task_memory.get_tasks_by_status(status))
        for status in TaskStatus
    }

    # Verify status distribution
    assert status_counts[TaskStatus.COMPLETED] == 1
    assert status_counts[TaskStatus.FAILED] == 1
    assert status_counts[TaskStatus.PENDING] == 1


def test_timeline_data(populated_task_memory):
    """Test timeline data generation."""
    completed_tasks = populated_task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    timeline_data = []

    for task in completed_tasks:
        if "completion_time" in task.metadata:
            timeline_data.append(
                {
                    "task_id": task.task_id,
                    "completion_time": task.metadata["completion_time"],
                    "type": task.type,
                }
            )

    # Verify timeline data
    assert len(timeline_data) == 1
    assert timeline_data[0]["task_id"] == "test_task_1"
    assert timeline_data[0]["type"] == "model_training"


@patch("streamlit.write")
def test_dashboard_initialization(mock_write, dashboard):
    """Test dashboard initialization and basic display."""
    # Verify dashboard setup
    assert dashboard is not None
    assert dashboard.task_memory is not None

    # Test page setup
    dashboard.setup_page()
    mock_write.assert_called()


def test_error_handling(populated_task_memory):
    """Test error handling for invalid task IDs."""
    # Test non-existent task
    task = populated_task_memory.get_task("non_existent_task")
    assert task is None

    # Test invalid task status
    with pytest.raises(ValueError):
        populated_task_memory.get_tasks_by_status("INVALID_STATUS")
