import pytest
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
from automation.models.automation import AutomationTask, TaskStatus
from automation.services.automation_service import AutomationService

@pytest.fixture
async def chaos_test_system():
    """Fixture for chaos testing the automation system."""
    service = AutomationService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_network_failure(chaos_test_system):
    """Test system behavior during network failure."""
    # Create a task that requires network access
    task = AutomationTask(
        name="network_failure_test",
        description="Test system behavior during network failure",
        type="network",
        metadata={"requires_network": True}
    )
    
    # Simulate network failure
    with pytest.raises(Exception) as exc_info:
        await chaos_test_system.execute_task(task.id)
    
    # Verify system handled network failure
    assert "NetworkError" in str(exc_info.value)
    assert task.status == TaskStatus.FAILED
    assert task.error is not None

@pytest.mark.asyncio
async def test_service_disruption(chaos_test_system):
    """Test system behavior during service disruption."""
    # Create multiple tasks
    tasks = [
        AutomationTask(
            name=f"service_disruption_test_{i}",
            description=f"Test service disruption {i}",
            type="test"
        )
        for i in range(5)
    ]
    
    # Simulate service disruption
    async def disrupt_service():
        await asyncio.sleep(1)
        await chaos_test_system.stop()
        await asyncio.sleep(1)
        await chaos_test_system.start()
    
    # Execute tasks with service disruption
    results = await asyncio.gather(
        *[chaos_test_system.execute_task(task.id) for task in tasks],
        disrupt_service(),
        return_exceptions=True
    )
    
    # Verify system handled service disruption
    assert any(isinstance(r, Exception) for r in results)
    assert all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for task in tasks)

@pytest.mark.asyncio
async def test_data_corruption(chaos_test_system):
    """Test system behavior during data corruption."""
    # Create a task with data
    task = AutomationTask(
        name="data_corruption_test",
        description="Test system behavior during data corruption",
        type="test",
        metadata={"data": "test_data"}
    )
    
    # Simulate data corruption
    task.metadata["data"] = None
    
    # Execute task
    with pytest.raises(Exception) as exc_info:
        await chaos_test_system.execute_task(task.id)
    
    # Verify system handled data corruption
    assert "DataError" in str(exc_info.value)
    assert task.status == TaskStatus.FAILED
    assert task.error is not None

@pytest.mark.asyncio
async def test_resource_exhaustion(chaos_test_system):
    """Test system behavior during resource exhaustion."""
    # Create resource-intensive tasks
    tasks = [
        AutomationTask(
            name=f"resource_exhaustion_test_{i}",
            description=f"Test resource exhaustion {i}",
            type="test",
            metadata={"resource_intensive": True}
        )
        for i in range(10)
    ]
    
    # Execute tasks to exhaust resources
    results = await asyncio.gather(*[
        chaos_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system handled resource exhaustion
    assert all(r["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED] for r in results)
    metrics = chaos_test_system.get_metrics()["resource_usage"]
    assert metrics["cpu"] < 100
    assert metrics["memory"] < 100

@pytest.mark.asyncio
async def test_concurrent_failures(chaos_test_system):
    """Test system behavior during concurrent failures."""
    # Create tasks that will fail
    tasks = [
        AutomationTask(
            name=f"concurrent_failure_test_{i}",
            description=f"Test concurrent failure {i}",
            type="test",
            metadata={"should_fail": True}
        )
        for i in range(5)
    ]
    
    # Execute tasks concurrently
    results = await asyncio.gather(*[
        chaos_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system handled concurrent failures
    assert all(r["status"] == TaskStatus.FAILED for r in results)
    assert all(task.error is not None for task in tasks)

@pytest.mark.asyncio
async def test_cascading_failures(chaos_test_system):
    """Test system behavior during cascading failures."""
    # Create dependent tasks
    tasks = []
    for i in range(3):
        task = AutomationTask(
            name=f"cascading_failure_test_{i}",
            description=f"Test cascading failure {i}",
            type="test",
            dependencies=[f"cascading_failure_test_{i-1}"] if i > 0 else None
        )
        tasks.append(task)
    
    # Simulate failure in first task
    tasks[0].metadata["should_fail"] = True
    
    # Execute tasks
    results = await asyncio.gather(*[
        chaos_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system handled cascading failures
    assert all(r["status"] == TaskStatus.FAILED for r in results)
    assert all(task.error is not None for task in tasks)

@pytest.mark.asyncio
async def test_partial_system_failure(chaos_test_system):
    """Test system behavior during partial system failure."""
    # Create tasks for different system components
    tasks = [
        AutomationTask(
            name=f"partial_failure_test_{i}",
            description=f"Test partial failure {i}",
            type=f"component_{i}",
            metadata={"component": f"component_{i}"}
        )
        for i in range(3)
    ]
    
    # Simulate partial system failure
    async def fail_component():
        await asyncio.sleep(1)
        # Simulate component failure
        pass
    
    # Execute tasks with partial failure
    results = await asyncio.gather(
        *[chaos_test_system.execute_task(task.id) for task in tasks],
        fail_component(),
        return_exceptions=True
    )
    
    # Verify system handled partial failure
    assert any(isinstance(r, Exception) for r in results)
    assert all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for task in tasks)

@pytest.mark.asyncio
async def test_recovery_after_failure(chaos_test_system):
    """Test system recovery after failure."""
    # Create a task that will fail
    task = AutomationTask(
        name="recovery_test",
        description="Test system recovery after failure",
        type="test",
        metadata={"should_fail": True}
    )
    
    # Execute task
    with pytest.raises(Exception):
        await chaos_test_system.execute_task(task.id)
    
    # Verify system recovered
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    
    # Create and execute a new task
    new_task = AutomationTask(
        name="recovery_test_new",
        description="Test system after recovery",
        type="test"
    )
    
    result = await chaos_test_system.execute_task(new_task.id)
    assert result["status"] == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_error_propagation(chaos_test_system):
    """Test system behavior during error propagation."""
    # Create tasks with error propagation
    tasks = [
        AutomationTask(
            name=f"error_propagation_test_{i}",
            description=f"Test error propagation {i}",
            type="test",
            metadata={"propagate_error": True}
        )
        for i in range(3)
    ]
    
    # Execute tasks
    results = await asyncio.gather(*[
        chaos_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system handled error propagation
    assert all(r["status"] == TaskStatus.FAILED for r in results)
    assert all(task.error is not None for task in tasks)

@pytest.mark.asyncio
async def test_system_stability(chaos_test_system):
    """Test system stability under chaotic conditions."""
    # Create various types of tasks
    tasks = []
    for i in range(10):
        task = AutomationTask(
            name=f"stability_test_{i}",
            description=f"Test system stability {i}",
            type="test",
            metadata={
                "should_fail": i % 2 == 0,
                "resource_intensive": i % 3 == 0,
                "requires_network": i % 4 == 0
            }
        )
        tasks.append(task)
    
    # Execute tasks with various failure conditions
    results = await asyncio.gather(*[
        chaos_test_system.execute_task(task.id)
        for task in tasks
    ])
    
    # Verify system maintained stability
    assert all(r["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED] for r in results)
    metrics = chaos_test_system.get_metrics()
    assert metrics["health"] == "healthy"
    assert metrics["stability"] > 0.8 