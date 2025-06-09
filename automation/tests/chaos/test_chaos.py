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
from ...monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_network_failure():
    """Test system behavior during network failure."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create network-dependent tasks
        tasks = []
        for i in range(50):  # Create 50 network-dependent tasks
            task = await task_manager.create_task(
                name=f"network_chaos_task_{i}",
                description="Network chaos test task",
                type="test",
                priority="high",
                parameters={
                    "requires_network": True,
                    "retry_count": 3
                }
            )
            tasks.append(task)
            
        # Simulate network failure
        start_time = time.time()
        
        # Execute tasks with network failure
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected network failure: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "retrying"]  # Tasks should be failed or retrying
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "network_failure" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in network failure test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_database_failure():
    """Test system behavior during database failure."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create database-dependent tasks
        tasks = []
        for i in range(50):  # Create 50 database-dependent tasks
            task = await task_manager.create_task(
                name=f"db_chaos_task_{i}",
                description="Database chaos test task",
                type="test",
                priority="high",
                parameters={
                    "requires_db": True,
                    "retry_count": 3
                }
            )
            tasks.append(task)
            
        # Simulate database failure
        start_time = time.time()
        
        # Execute tasks with database failure
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected database failure: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "retrying"]  # Tasks should be failed or retrying
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "database_failure" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in database failure test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_memory_exhaustion():
    """Test system behavior during memory exhaustion."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create memory-intensive tasks
        tasks = []
        for i in range(20):  # Create 20 memory-intensive tasks
            task = await task_manager.create_task(
                name=f"memory_chaos_task_{i}",
                description="Memory chaos test task",
                type="test",
                priority="high",
                parameters={
                    "memory_size": 500 * 1024 * 1024  # 500MB
                }
            )
            tasks.append(task)
            
        # Execute tasks to exhaust memory
        start_time = time.time()
        
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected memory exhaustion: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "killed"]  # Tasks should be failed or killed
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "memory_exhaustion" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in memory exhaustion test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_cpu_exhaustion():
    """Test system behavior during CPU exhaustion."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create CPU-intensive tasks
        tasks = []
        for i in range(20):  # Create 20 CPU-intensive tasks
            task = await task_manager.create_task(
                name=f"cpu_chaos_task_{i}",
                description="CPU chaos test task",
                type="test",
                priority="high",
                parameters={
                    "cpu_load": 1.0,  # 100% CPU load
                    "duration": 30  # 30 seconds
                }
            )
            tasks.append(task)
            
        # Execute tasks to exhaust CPU
        start_time = time.time()
        
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected CPU exhaustion: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "killed"]  # Tasks should be failed or killed
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "cpu_exhaustion" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in CPU exhaustion test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_disk_exhaustion():
    """Test system behavior during disk exhaustion."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create disk-intensive tasks
        tasks = []
        for i in range(10):  # Create 10 disk-intensive tasks
            task = await task_manager.create_task(
                name=f"disk_chaos_task_{i}",
                description="Disk chaos test task",
                type="test",
                priority="high",
                parameters={
                    "file_size": 1 * 1024 * 1024 * 1024,  # 1GB
                    "operations": 1000  # 1000 write operations
                }
            )
            tasks.append(task)
            
        # Execute tasks to exhaust disk
        start_time = time.time()
        
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected disk exhaustion: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "killed"]  # Tasks should be failed or killed
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "disk_exhaustion" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in disk exhaustion test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_service_failure():
    """Test system behavior during service failure."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create service-dependent tasks
        tasks = []
        for i in range(50):  # Create 50 service-dependent tasks
            task = await task_manager.create_task(
                name=f"service_chaos_task_{i}",
                description="Service chaos test task",
                type="test",
                priority="high",
                parameters={
                    "service": random.choice(["auth", "notification", "monitoring"]),
                    "retry_count": 3
                }
            )
            tasks.append(task)
            
        # Simulate service failure
        start_time = time.time()
        
        # Execute tasks with service failure
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected service failure: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "retrying"]  # Tasks should be failed or retrying
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "service_failure" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in service failure test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_cascading_failure():
    """Test system behavior during cascading failure."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create interdependent tasks
        tasks = []
        for i in range(20):  # Create 20 interdependent tasks
            task = await task_manager.create_task(
                name=f"cascade_chaos_task_{i}",
                description="Cascading failure test task",
                type="test",
                priority="high",
                parameters={
                    "dependencies": [f"cascade_chaos_task_{j}" for j in range(i)],
                    "retry_count": 3
                }
            )
            tasks.append(task)
            
        # Simulate cascading failure
        start_time = time.time()
        
        # Execute tasks with cascading failure
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected cascading failure: {str(e)}")
                
        end_time = time.time()
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["failed", "retrying"]  # Tasks should be failed or retrying
            
        # Verify alert creation
        alerts = await alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        assert any(a.metadata.get("type") == "cascading_failure" for a in alerts)
        
    except Exception as e:
        logger.error(f"Error in cascading failure test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_system_recovery():
    """Test system recovery after chaos."""
    try:
        # Setup
        task_manager, orchestrator, metrics_collector = await setup_test_environment()
        
        # Create recovery test tasks
        tasks = []
        for i in range(50):  # Create 50 recovery test tasks
            task = await task_manager.create_task(
                name=f"recovery_chaos_task_{i}",
                description="Recovery test task",
                type="test",
                priority="high",
                parameters={
                    "failure_type": random.choice([
                        "network",
                        "database",
                        "memory",
                        "cpu",
                        "disk",
                        "service"
                    ]),
                    "recovery_timeout": 300  # 5 minutes
                }
            )
            tasks.append(task)
            
        # Execute tasks with failures
        start_time = time.time()
        
        for task in tasks:
            try:
                await orchestrator.execute_task(task.id)
            except Exception as e:
                logger.warning(f"Expected failure: {str(e)}")
                
        # Wait for recovery
        await asyncio.sleep(300)  # Wait 5 minutes
        
        end_time = time.time()
        
        # Verify system state
        metrics = await metrics_collector.get_metrics(
            "cpu_usage",
            start_time=end_time - 60,
            end_time=end_time
        )
        recovery_cpu = metrics[-1]["value"]
        
        # Verify task status
        for task in tasks:
            task = await task_manager.get_task(task.id)
            assert task.status in ["completed", "failed"]  # Tasks should be completed or failed
            
        # Verify system recovery
        assert recovery_cpu < 50  # CPU usage should be below 50% after recovery
        
    except Exception as e:
        logger.error(f"Error in system recovery test: {str(e)}")
        raise 