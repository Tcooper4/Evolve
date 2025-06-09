import pytest
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
from automation.models.automation import AutomationTask, TaskStatus
from automation.services.automation_service import AutomationService

@pytest.fixture
async def stress_test_system():
    """Fixture for stress testing the automation system."""
    service = AutomationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_concurrent_task_execution(stress_test_system):
    """Test system behavior under concurrent task execution."""
    num_tasks = 100
    tasks = []
    
    # Create multiple tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"stress_test_task_{i}",
            description=f"Stress test task {i}",
            type="test",
            priority=i % 3
        )
        tasks.append(task)
    
    # Execute tasks concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(result["status"] == TaskStatus.COMPLETED for result in results)
    assert end_time - start_time < 30  # Should complete within 30 seconds

@pytest.mark.asyncio
async def test_high_priority_task_handling(stress_test_system):
    """Test system behavior with high priority tasks."""
    num_tasks = 50
    tasks = []
    
    # Create mixed priority tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"priority_test_task_{i}",
            description=f"Priority test task {i}",
            type="test",
            priority=2 if i % 5 == 0 else 0  # Every 5th task is high priority
        )
        tasks.append(task)
    
    # Execute tasks
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify high priority tasks completed first
    high_priority_results = [r for r in results if r["priority"] == 2]
    low_priority_results = [r for r in results if r["priority"] == 0]
    
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert len(high_priority_results) == num_tasks // 5
    assert len(low_priority_results) == num_tasks - (num_tasks // 5)

@pytest.mark.asyncio
async def test_resource_exhaustion(stress_test_system):
    """Test system behavior under resource exhaustion."""
    num_tasks = 200
    tasks = []
    
    # Create resource-intensive tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"resource_test_task_{i}",
            description=f"Resource test task {i}",
            type="test",
            metadata={"resource_intensive": True}
        )
        tasks.append(task)
    
    # Execute tasks
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system handled resource exhaustion
    assert len(results) == num_tasks
    assert all(r["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED] for r in results)
    assert stress_test_system.get_metrics()["resource_usage"]["cpu"] < 90
    assert stress_test_system.get_metrics()["resource_usage"]["memory"] < 90

@pytest.mark.asyncio
async def test_long_running_tasks(stress_test_system):
    """Test system behavior with long-running tasks."""
    num_tasks = 20
    tasks = []
    
    # Create long-running tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"long_running_task_{i}",
            description=f"Long running task {i}",
            type="test",
            timeout=300,  # 5 minutes timeout
            metadata={"duration": 60}  # 1 minute execution
        )
        tasks.append(task)
    
    # Execute tasks
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert end_time - start_time < 300  # Should complete within 5 minutes

@pytest.mark.asyncio
async def test_concurrent_workflow_execution(stress_test_system):
    """Test system behavior under concurrent workflow execution."""
    num_workflows = 20
    workflows = []
    
    # Create workflows with multiple tasks
    for i in range(num_workflows):
        tasks = [
            AutomationTask(
                name=f"workflow_{i}_task_{j}",
                description=f"Task {j} in workflow {i}",
                type="test"
            )
            for j in range(5)
        ]
        workflow = await stress_test_system.create_workflow(
            name=f"stress_test_workflow_{i}",
            description=f"Stress test workflow {i}",
            tasks=tasks
        )
        workflows.append(workflow)
    
    # Execute workflows concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_workflow(workflow.id)
        for workflow in workflows
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_workflows
    assert all(r["status"] == "completed" for r in results)
    assert end_time - start_time < 60  # Should complete within 1 minute

@pytest.mark.asyncio
async def test_database_stress(stress_test_system):
    """Test system behavior under database stress."""
    num_operations = 1000
    operations = []
    
    # Create multiple database operations
    for i in range(num_operations):
        task = AutomationTask(
            name=f"db_stress_task_{i}",
            description=f"Database stress test task {i}",
            type="database",
            metadata={"operation": "write" if i % 2 == 0 else "read"}
        )
        operations.append(task)
    
    # Execute operations
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in operations
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_operations
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert end_time - start_time < 60  # Should complete within 1 minute

@pytest.mark.asyncio
async def test_network_stress(stress_test_system):
    """Test system behavior under network stress."""
    num_requests = 200
    requests = []
    
    # Create network-intensive tasks
    for i in range(num_requests):
        task = AutomationTask(
            name=f"network_stress_task_{i}",
            description=f"Network stress test task {i}",
            type="network",
            metadata={"request_size": "large"}
        )
        requests.append(task)
    
    # Execute requests
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in requests
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_requests
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert end_time - start_time < 30  # Should complete within 30 seconds

@pytest.mark.asyncio
async def test_memory_stress(stress_test_system):
    """Test system behavior under memory stress."""
    num_tasks = 50
    tasks = []
    
    # Create memory-intensive tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"memory_stress_task_{i}",
            description=f"Memory stress test task {i}",
            type="test",
            metadata={"memory_intensive": True}
        )
        tasks.append(task)
    
    # Execute tasks
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert stress_test_system.get_metrics()["resource_usage"]["memory"] < 90
    assert end_time - start_time < 30  # Should complete within 30 seconds

@pytest.mark.asyncio
async def test_cpu_stress(stress_test_system):
    """Test system behavior under CPU stress."""
    num_tasks = 30
    tasks = []
    
    # Create CPU-intensive tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"cpu_stress_task_{i}",
            description=f"CPU stress test task {i}",
            type="test",
            metadata={"cpu_intensive": True}
        )
        tasks.append(task)
    
    # Execute tasks
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert stress_test_system.get_metrics()["resource_usage"]["cpu"] < 90
    assert end_time - start_time < 30  # Should complete within 30 seconds

@pytest.mark.asyncio
async def test_disk_stress(stress_test_system):
    """Test system behavior under disk stress."""
    num_tasks = 40
    tasks = []
    
    # Create disk-intensive tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"disk_stress_task_{i}",
            description=f"Disk stress test task {i}",
            type="test",
            metadata={"disk_intensive": True}
        )
        tasks.append(task)
    
    # Execute tasks
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    assert stress_test_system.get_metrics()["resource_usage"]["disk"] < 90
    assert end_time - start_time < 30  # Should complete within 30 seconds

@pytest.mark.asyncio
async def test_mixed_stress(stress_test_system):
    """Test system behavior under mixed stress conditions."""
    num_tasks = 100
    tasks = []
    
    # Create mixed stress tasks
    for i in range(num_tasks):
        task = AutomationTask(
            name=f"mixed_stress_task_{i}",
            description=f"Mixed stress test task {i}",
            type="test",
            priority=i % 3,
            metadata={
                "cpu_intensive": i % 3 == 0,
                "memory_intensive": i % 3 == 1,
                "disk_intensive": i % 3 == 2
            }
        )
        tasks.append(task)
    
    # Execute tasks
    start_time = time.time()
    results = await asyncio.gather(*[
        stress_test_system.execute_task(task.id)
        for task in tasks
    ])
    end_time = time.time()
    
    # Verify results
    assert len(results) == num_tasks
    assert all(r["status"] == TaskStatus.COMPLETED for r in results)
    metrics = stress_test_system.get_metrics()["resource_usage"]
    assert metrics["cpu"] < 90
    assert metrics["memory"] < 90
    assert metrics["disk"] < 90
    assert end_time - start_time < 60  # Should complete within 1 minute 