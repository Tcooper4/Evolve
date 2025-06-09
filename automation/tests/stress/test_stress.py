import asyncio
import logging
import random
import time
from typing import List, Dict
import pytest
from ..conftest import setup_test_environment
from ...core.task_manager import TaskManager
from ...core.orchestrator import Orchestrator
from ...monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_concurrent_task_execution():
    """Test system performance with concurrent task execution."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create tasks
        tasks = []
        for i in range(100):  # Create 100 tasks
            task = await task_manager.create_task(
                name=f"stress_test_task_{i}",
                description="Stress test task",
                type="test",
                priority="high",
                parameters={
                    "sleep_time": random.uniform(0.1, 1.0)  # Random sleep time
                }
            )
            tasks.append(task)
            
        # Execute tasks concurrently
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        tasks_per_second = len(tasks) / execution_time
        
        # Get system metrics
        metrics = await metrics_collector.get_metrics(
            "cpu_usage",
            start_time=start_time,
            end_time=end_time
        )
        avg_cpu = sum(m["value"] for m in metrics) / len(metrics)
        
        # Assertions
        assert tasks_per_second >= 10  # Minimum 10 tasks per second
        assert avg_cpu < 80  # CPU usage should not exceed 80%
        
    except Exception as e:
        logger.error(f"Error in concurrent task execution test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_memory_usage():
    """Test system memory usage under load."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create memory-intensive tasks
        tasks = []
        for i in range(50):  # Create 50 memory-intensive tasks
            task = await task_manager.create_task(
                name=f"memory_test_task_{i}",
                description="Memory test task",
                type="test",
                priority="high",
                parameters={
                    "memory_size": 100 * 1024 * 1024  # 100MB
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Get memory metrics
        metrics = await metrics_collector.get_metrics(
            "memory_usage",
            start_time=start_time,
            end_time=end_time
        )
        max_memory = max(m["value"] for m in metrics)
        
        # Assertions
        assert max_memory < 90  # Memory usage should not exceed 90%
        
    except Exception as e:
        logger.error(f"Error in memory usage test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_disk_io():
    """Test system disk I/O performance."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create disk I/O intensive tasks
        tasks = []
        for i in range(20):  # Create 20 disk I/O intensive tasks
            task = await task_manager.create_task(
                name=f"disk_test_task_{i}",
                description="Disk test task",
                type="test",
                priority="high",
                parameters={
                    "file_size": 10 * 1024 * 1024,  # 10MB
                    "operations": 100  # 100 read/write operations
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Get disk metrics
        metrics = await metrics_collector.get_metrics(
            "disk_usage",
            start_time=start_time,
            end_time=end_time
        )
        max_disk = max(m["value"] for m in metrics)
        
        # Assertions
        assert max_disk < 90  # Disk usage should not exceed 90%
        
    except Exception as e:
        logger.error(f"Error in disk I/O test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_network_io():
    """Test system network I/O performance."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create network I/O intensive tasks
        tasks = []
        for i in range(30):  # Create 30 network I/O intensive tasks
            task = await task_manager.create_task(
                name=f"network_test_task_{i}",
                description="Network test task",
                type="test",
                priority="high",
                parameters={
                    "data_size": 5 * 1024 * 1024,  # 5MB
                    "requests": 50  # 50 network requests
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Get network metrics
        metrics = await metrics_collector.get_metrics(
            "network_io",
            start_time=start_time,
            end_time=end_time
        )
        max_network = max(m["value"] for m in metrics)
        
        # Assertions
        assert max_network < 100 * 1024 * 1024  # Network I/O should not exceed 100MB/s
        
    except Exception as e:
        logger.error(f"Error in network I/O test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_database_performance():
    """Test system database performance."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create database-intensive tasks
        tasks = []
        for i in range(200):  # Create 200 database-intensive tasks
            task = await task_manager.create_task(
                name=f"db_test_task_{i}",
                description="Database test task",
                type="test",
                priority="high",
                parameters={
                    "operations": 1000,  # 1000 database operations
                    "batch_size": 100  # Batch size of 100
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        operations_per_second = (len(tasks) * 1000) / execution_time
        
        # Assertions
        assert operations_per_second >= 1000  # Minimum 1000 operations per second
        
    except Exception as e:
        logger.error(f"Error in database performance test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_api_performance():
    """Test system API performance."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create API-intensive tasks
        tasks = []
        for i in range(1000):  # Create 1000 API-intensive tasks
            task = await task_manager.create_task(
                name=f"api_test_task_{i}",
                description="API test task",
                type="test",
                priority="high",
                parameters={
                    "endpoints": [
                        "/api/tasks",
                        "/api/metrics",
                        "/api/alerts"
                    ],
                    "requests": 10  # 10 requests per endpoint
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        requests_per_second = (len(tasks) * 30) / execution_time  # 30 requests per task
        
        # Get API metrics
        metrics = await metrics_collector.get_metrics(
            "api_request_duration",
            start_time=start_time,
            end_time=end_time
        )
        avg_duration = sum(m["value"] for m in metrics) / len(metrics)
        
        # Assertions
        assert requests_per_second >= 100  # Minimum 100 requests per second
        assert avg_duration < 1.0  # Average request duration should be less than 1 second
        
    except Exception as e:
        logger.error(f"Error in API performance test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_notification_performance():
    """Test system notification performance."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create notification-intensive tasks
        tasks = []
        for i in range(500):  # Create 500 notification-intensive tasks
            task = await task_manager.create_task(
                name=f"notification_test_task_{i}",
                description="Notification test task",
                type="test",
                priority="high",
                parameters={
                    "notifications": 100,  # 100 notifications per task
                    "channels": ["email", "slack", "web"]
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        notifications_per_second = (len(tasks) * 100) / execution_time
        
        # Assertions
        assert notifications_per_second >= 1000  # Minimum 1000 notifications per second
        
    except Exception as e:
        logger.error(f"Error in notification performance test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_system_recovery():
    """Test system recovery after high load."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create high-load tasks
        tasks = []
        for i in range(100):  # Create 100 high-load tasks
            task = await task_manager.create_task(
                name=f"recovery_test_task_{i}",
                description="Recovery test task",
                type="test",
                priority="high",
                parameters={
                    "cpu_load": 0.8,  # 80% CPU load
                    "memory_load": 0.8,  # 80% memory load
                    "duration": 30  # 30 seconds
                }
            )
            tasks.append(task)
            
        # Execute tasks
        start_time = time.time()
        await asyncio.gather(*[
            orchestrator.execute_task(task.id)
            for task in tasks
        ])
        end_time = time.time()
        
        # Wait for system recovery
        await asyncio.sleep(60)  # Wait 60 seconds
        
        # Get recovery metrics
        metrics = await metrics_collector.get_metrics(
            "cpu_usage",
            start_time=end_time,
            end_time=end_time + 60
        )
        recovery_cpu = metrics[-1]["value"]
        
        # Assertions
        assert recovery_cpu < 50  # CPU usage should be below 50% after recovery
        
    except Exception as e:
        logger.error(f"Error in system recovery test: {str(e)}")
        raise 