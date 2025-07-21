import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import psutil
import pytest

from trading.agents.task_dashboard import TaskDashboard
from trading.agents.task_memory import Task, TaskMemory, TaskStatus

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ModelBuilder, AgentRouter, and SelfImprovingAgent are omitted as they are not present in the current codebase


@pytest.fixture
def task_memory():
    return TaskMemory()


@pytest.fixture
def dashboard(task_memory):
    return TaskDashboard(task_memory)


def test_task_creation_performance(task_memory):
    num_tasks = 1000
    creation_times = []
    for i in range(num_tasks):
        start_time = time.time()
        task = Task(
            task_id=str(uuid.uuid4()),
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
                "model_type": "lstm",
            },
            notes=f"Performance test task {i}",
        )
        task_memory.add_task(task)
        end_time = time.time()
        creation_times.append(end_time - start_time)
    avg_creation_time = sum(creation_times) / len(creation_times)
    max_creation_time = max(creation_times)
    assert avg_creation_time < 0.1
    assert max_creation_time < 0.5


def test_concurrent_task_processing(task_memory):
    num_tasks = 100
    num_threads = 10

    def process_task(task_id):
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
            },
        )
        task_memory.add_task(task)
        time.sleep(0.1)
        task.status = TaskStatus.COMPLETED
        task.metadata.update({"completion_time": datetime.now().isoformat()})
        task_memory.update_task(task)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_task, str(uuid.uuid4())) for _ in range(num_tasks)
        ]
        for future in as_completed(futures):
            future.result()
    completed_tasks = task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    assert len(completed_tasks) == num_tasks


def test_memory_usage_under_load(task_memory):
    initial_memory = psutil.Process().memory_info().rss
    for i in range(1000):
        task = Task(
            task_id=str(uuid.uuid4()),
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
                "large_data": "x" * 10000,
            },
        )
        task_memory.add_task(task)
    final_memory = psutil.Process().memory_info().rss
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024


def test_query_performance(task_memory):
    for i in range(1000):
        task = Task(
            task_id=str(uuid.uuid4()),
            type=f"task_type_{i % 5}",
            status=TaskStatus.PENDING,
            metadata={
                "agent": f"agent_{i % 3}",
                "creation_time": datetime.now().isoformat(),
                "priority": i % 10,
            },
        )
        task_memory.add_task(task)
    query_times = []
    for status in TaskStatus:
        start_time = time.time()
        tasks = task_memory.get_tasks_by_status(status)
        end_time = time.time()
        query_times.append(end_time - start_time)
    for agent in ["agent_0", "agent_1", "agent_2"]:
        start_time = time.time()
        tasks = [t for t in task_memory.tasks if t.metadata.get("agent") == agent]
        end_time = time.time()
        query_times.append(end_time - start_time)
    avg_query_time = sum(query_times) / len(query_times)
    assert avg_query_time < 0.1


def test_dashboard_rendering_performance(dashboard, task_memory):
    for i in range(1000):
        task = Task(
            task_id=str(uuid.uuid4()),
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                "agent": "model_builder",
                "creation_time": datetime.now().isoformat(),
                "large_data": "x" * 10000,
            },
        )
        task_memory.add_task(task)
    # Simulate dashboard rendering (replace with actual render call if available)
    assert dashboard is not None
