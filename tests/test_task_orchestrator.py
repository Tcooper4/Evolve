"""
Tests for Task Orchestrator (Modular)

Tests core/orchestrator/task_orchestrator.py functionality including:
- Task scheduling and execution
- Performance monitoring
- Health checks
- Configuration management
- Error handling
Updated for the new modular structure.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from core.orchestrator.task_orchestrator import TaskOrchestrator, create_task_orchestrator, start_orchestrator
from core.orchestrator.task_models import TaskConfig, TaskStatus, TaskPriority, TaskType
from core.orchestrator.task_scheduler import TaskScheduler
from core.orchestrator.task_executor import TaskExecutor
from core.orchestrator.task_monitor import TaskMonitor
from core.orchestrator.task_conditions import TaskConditions


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config(temp_dir):
    """Create sample configuration for testing."""
    config = {
        'orchestrator': {
            'enabled': True,
            'max_concurrent_tasks': 3,
            'default_timeout_minutes': 10,
            'health_check_interval_minutes': 2,
            'performance_monitoring': True,
            'error_alerting': True
        },
        'scheduler_config': {
            'max_workers': 5
        },
        'executor_config': {
            'max_workers': 5
        },
        'monitor_config': {
            'health_check_interval_minutes': 2,
            'performance_threshold': 0.8,
            'error_threshold': 3
        },
        'conditions_config': {},
        'tasks': {
            'test_task': {
                'enabled': True,
                'interval_minutes': 5,
                'priority': 'medium',
                'dependencies': [],
                'conditions': {'market_hours': True},
                'parameters': {'test_param': 'test_value'}
            },
            'system_health': {
                'enabled': True,
                'interval_minutes': 2,
                'priority': 'high',
                'dependencies': [],
                'conditions': {}
            }
        }
    }
    
    config_path = Path(temp_dir) / "test_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


@pytest.fixture
def task_orchestrator(sample_config):
    """Create TaskOrchestrator instance for testing."""
    return TaskOrchestrator(sample_config)


class TestTaskOrchestrator:
    """Test TaskOrchestrator functionality"""

    def test_task_orchestrator_initialization(self, task_orchestrator):
        """Test TaskOrchestrator initialization"""
        assert task_orchestrator is not None
        assert task_orchestrator.is_running is False
        assert task_orchestrator.config is not None

    def test_load_task_config(self, task_orchestrator):
        """Test loading task configuration"""
        assert task_orchestrator.config is not None
        assert 'tasks' in task_orchestrator.config
        assert 'test_task' in task_orchestrator.config['tasks']

    def test_initialize_components(self, task_orchestrator):
        """Test component initialization"""
        assert task_orchestrator.scheduler is not None
        assert task_orchestrator.executor is not None
        assert task_orchestrator.monitor is not None
        assert task_orchestrator.conditions is not None

    def test_initialize_task_providers(self, task_orchestrator):
        """Test task provider initialization"""
        assert len(task_orchestrator.task_providers) > 0
        assert "agent" in task_orchestrator.task_providers
        assert "system" in task_orchestrator.task_providers

    def test_load_tasks(self, task_orchestrator):
        """Test loading tasks into scheduler"""
        scheduled_tasks = task_orchestrator.scheduler.get_scheduled_tasks()
        assert len(scheduled_tasks) > 0

    async def test_start_stop_orchestrator(self, task_orchestrator):
        """Test orchestrator start and stop"""
        await task_orchestrator.start()
        assert task_orchestrator.is_running is True
        
        await task_orchestrator.stop()
        assert task_orchestrator.is_running is False

    async def test_execute_task_now(self, task_orchestrator):
        """Test immediate task execution"""
        with patch.object(task_orchestrator.executor, 'execute_task_now') as mock_execute:
            mock_execute.return_value = "task_123"
            
            task_id = await task_orchestrator.execute_task_now("test_task")
            assert task_id == "task_123"
            mock_execute.assert_called_once()

    def test_get_task_status(self, task_orchestrator):
        """Test getting task status"""
        with patch.object(task_orchestrator.executor, 'get_task_status') as mock_status:
            mock_status.return_value = {
                "task_name": "test_task",
                "status": "completed",
                "start_time": "2024-01-01T00:00:00Z"
            }
            
            status = task_orchestrator.get_task_status("test_task")
            assert status is not None
            assert status["task_name"] == "test_task"
            mock_status.assert_called_once_with("test_task")

    def test_get_system_status(self, task_orchestrator):
        """Test getting system status"""
        with patch.object(task_orchestrator.monitor, 'check_system_health') as mock_health:
            mock_health.return_value = {"overall_health": 0.9}
            
            status = task_orchestrator.get_system_status()
            assert status is not None
            assert "orchestrator_status" in status
            assert "scheduled_tasks" in status
            assert "current_executions" in status

    def test_update_task_config(self, task_orchestrator):
        """Test updating task configuration"""
        updates = {"interval_minutes": 10, "priority": "high"}
        
        with patch.object(task_orchestrator.scheduler, 'update_task') as mock_update:
            task_orchestrator.update_task_config("test_task", updates)
            mock_update.assert_called_once_with("test_task", updates)

    def test_export_status_report(self, task_orchestrator):
        """Test exporting status report"""
        with patch('core.orchestrator.task_orchestrator.safe_json_save') as mock_save:
            mock_save.return_value = None
            
            report_path = task_orchestrator.export_status_report()
            assert report_path is not None
            mock_save.assert_called_once()

    async def test_health_monitor_loop(self, task_orchestrator):
        """Test health monitoring loop"""
        with patch.object(task_orchestrator.monitor, 'check_system_health') as mock_health:
            mock_health.return_value = {"overall_health": 0.9}
            
            # Start the loop
            task_orchestrator.is_running = True
            task = asyncio.create_task(task_orchestrator._health_monitor_loop())
            
            # Let it run for a short time
            await asyncio.sleep(0.1)
            
            # Stop the loop
            task_orchestrator.is_running = False
            await asyncio.sleep(0.1)
            
            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            mock_health.assert_called()

    async def test_performance_monitor_loop(self, task_orchestrator):
        """Test performance monitoring loop"""
        with patch.object(task_orchestrator.monitor, 'update_performance_metrics') as mock_update:
            # Start the loop
            task_orchestrator.is_running = True
            task = asyncio.create_task(task_orchestrator._performance_monitor_loop())
            
            # Let it run for a short time
            await asyncio.sleep(0.1)
            
            # Stop the loop
            task_orchestrator.is_running = False
            await asyncio.sleep(0.1)
            
            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            mock_update.assert_called()


class TestTaskScheduler:
    """Test TaskScheduler functionality"""

    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = TaskScheduler()
        assert scheduler is not None
        assert scheduler.is_running is False

    def test_add_task(self):
        """Test adding task to scheduler"""
        scheduler = TaskScheduler()
        
        task_config = TaskConfig(
            name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            enabled=True,
            interval_minutes=5
        )
        
        scheduler.add_task(task_config)
        assert "test_task" in scheduler.tasks

    def test_remove_task(self):
        """Test removing task from scheduler"""
        scheduler = TaskScheduler()
        
        task_config = TaskConfig(
            name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            enabled=True
        )
        
        scheduler.add_task(task_config)
        assert "test_task" in scheduler.tasks
        
        scheduler.remove_task("test_task")
        assert "test_task" not in scheduler.tasks

    def test_update_task(self):
        """Test updating task configuration"""
        scheduler = TaskScheduler()
        
        task_config = TaskConfig(
            name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            enabled=True,
            interval_minutes=5
        )
        
        scheduler.add_task(task_config)
        
        updates = {"interval_minutes": 10, "priority": TaskPriority.HIGH}
        scheduler.update_task("test_task", updates)
        
        updated_task = scheduler.get_task_config("test_task")
        assert updated_task.interval_minutes == 10
        assert updated_task.priority == TaskPriority.HIGH


class TestTaskExecutor:
    """Test TaskExecutor functionality"""

    def test_executor_initialization(self):
        """Test executor initialization"""
        executor = TaskExecutor()
        assert executor is not None
        assert executor.executor is not None

    def test_register_task_provider(self):
        """Test registering task provider"""
        executor = TaskExecutor()
        
        def mock_provider(task_name, parameters):
            return {"success": True, "result": "test"}
        
        executor.register_task_provider("test_type", mock_provider)
        assert "test_type" in executor.task_providers

    async def test_execute_task(self):
        """Test task execution"""
        executor = TaskExecutor()
        
        # Mock task provider
        def mock_provider(task_name, parameters):
            return {"success": True, "result": "test"}
        
        executor.register_task_provider("test_type", mock_provider)
        
        task_config = TaskConfig(
            name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            enabled=True
        )
        
        with patch.object(executor, '_get_task_type') as mock_get_type:
            mock_get_type.return_value = "test_type"
            
            execution = await executor.execute_task("test_task", task_config)
            assert execution is not None
            assert execution.task_name == "test_task"

    def test_get_task_status(self):
        """Test getting task status"""
        executor = TaskExecutor()
        
        # Test with no current executions
        status = executor.get_task_status("nonexistent_task")
        assert status is None

    def test_get_execution_history(self):
        """Test getting execution history"""
        executor = TaskExecutor()
        
        history = executor.get_execution_history()
        assert isinstance(history, list)


class TestTaskMonitor:
    """Test TaskMonitor functionality"""

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = TaskMonitor()
        assert monitor is not None
        assert len(monitor.agent_status) == 0

    def test_update_task_performance(self):
        """Test updating task performance"""
        monitor = TaskMonitor()
        
        from core.orchestrator.task_models import TaskExecution, TaskStatus
        
        execution = TaskExecution(
            task_id="test_123",
            task_name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:01:00Z",
            status=TaskStatus.COMPLETED,
            duration_seconds=60.0
        )
        
        monitor.update_task_performance("test_task", execution)
        assert "test_task" in monitor.performance_metrics

    async def test_check_system_health(self):
        """Test system health check"""
        monitor = TaskMonitor()
        
        health_status = await monitor.check_system_health()
        assert health_status is not None
        assert "overall_health" in health_status
        assert "timestamp" in health_status

    def test_get_agent_status(self):
        """Test getting agent status"""
        monitor = TaskMonitor()
        
        status = monitor.get_agent_status()
        assert isinstance(status, dict)

    def test_get_performance_summary(self):
        """Test getting performance summary"""
        monitor = TaskMonitor()
        
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)
        assert "total_tasks" in summary


class TestTaskConditions:
    """Test TaskConditions functionality"""

    def test_conditions_initialization(self):
        """Test conditions initialization"""
        conditions = TaskConditions()
        assert conditions is not None

    async def test_check_condition(self):
        """Test condition checking"""
        conditions = TaskConditions()
        
        # Test market hours condition
        result = await conditions.check_condition("market_hours", True)
        assert isinstance(result, bool)

    async def test_market_hours_check(self):
        """Test market hours condition"""
        conditions = TaskConditions()
        
        result = await conditions._is_market_hours()
        assert isinstance(result, bool)

    async def test_system_health_check(self):
        """Test system health condition"""
        conditions = TaskConditions()
        
        health = await conditions._get_system_health()
        assert isinstance(health, float)
        assert 0 <= health <= 1

    def test_get_available_conditions(self):
        """Test getting available conditions"""
        conditions = TaskConditions()
        
        available = conditions.get_available_conditions()
        assert isinstance(available, dict)
        assert len(available) > 0

    def test_validate_condition(self):
        """Test condition validation"""
        conditions = TaskConditions()
        
        # Valid condition
        assert conditions.validate_condition("market_hours", True)
        
        # Invalid condition
        assert not conditions.validate_condition("nonexistent_condition", True)


def test_create_task_orchestrator():
    """Test create_task_orchestrator function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.yaml"
        
        # Create minimal config
        config = {'orchestrator': {'enabled': True}, 'tasks': {}}
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        orchestrator = create_task_orchestrator(str(config_path))
        assert orchestrator is not None
        assert isinstance(orchestrator, TaskOrchestrator)


async def test_start_orchestrator():
    """Test start_orchestrator function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.yaml"
        
        # Create minimal config
        config = {'orchestrator': {'enabled': True}, 'tasks': {}}
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        orchestrator = await start_orchestrator(str(config_path))
        assert orchestrator is not None
        assert orchestrator.is_running is True
        
        await orchestrator.stop()


class TestIntegration:
    """Integration tests for TaskOrchestrator"""

    async def test_full_orchestrator_workflow(self, temp_dir):
        """Test complete orchestrator workflow"""
        config_path = Path(temp_dir) / "test_config.yaml"
        
        # Create test configuration
        config = {
            'orchestrator': {
                'enabled': True,
                'max_concurrent_tasks': 2,
                'default_timeout_minutes': 5
            },
            'scheduler_config': {'max_workers': 2},
            'executor_config': {'max_workers': 2},
            'monitor_config': {'health_check_interval_minutes': 1},
            'conditions_config': {},
            'tasks': {
                'system_health': {
                    'enabled': True,
                    'interval_minutes': 1,
                    'priority': 'high'
                }
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create and start orchestrator
        orchestrator = TaskOrchestrator(str(config_path))
        await orchestrator.start()
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        # Check system status
        status = orchestrator.get_system_status()
        assert status is not None
        assert "orchestrator_status" in status
        
        # Stop orchestrator
        await orchestrator.stop()
        assert orchestrator.is_running is False


if __name__ == "__main__":
    pytest.main([__file__])
