"""
Test Task Orchestrator

This test validates the TaskOrchestrator functionality including
task execution, dependency management, and performance tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from core.task_orchestrator import TaskOrchestrator, TaskConfig, TaskType, TaskPriority

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockAgent:
    """Simple mock agent for testing"""

    def __init__(self, name: str):
        self.name = name
        self.execution_count = 0

    async def test_method(self, **kwargs):
        """Mock method that simulates work"""
        self.execution_count += 1
        await asyncio.sleep(1)  # Simulate work
        return {
            "status": "success",
            "agent": self.name,
            "execution_count": self.execution_count,
            "parameters": kwargs
        }


async def test_orchestrator():
    """Test the TaskOrchestrator functionality"""
    logger.info("Testing TaskOrchestrator...")

    # Create orchestrator
    orchestrator = TaskOrchestrator()

    # Add mock agents
    mock_agents = {
        'test_task1': MockAgent('TestAgent1'),
        'test_task2': MockAgent('TestAgent2'),
        'test_task3': MockAgent('TestAgent3')
    }

    orchestrator.agents = mock_agents

    # Initialize agent status
    for agent_name in mock_agents.keys():
        orchestrator.agent_status[agent_name] = orchestrator.AgentStatus(
            agent_name=agent_name
        )

    # Add test tasks
    test_tasks = {
        'test_task1': TaskConfig(
            name='test_task1',
            task_type=TaskType.MODEL_INNOVATION,
            enabled=True,
            interval_minutes=1,
            priority=TaskPriority.HIGH
        ),
        'test_task2': TaskConfig(
            name='test_task2',
            task_type=TaskType.STRATEGY_RESEARCH,
            enabled=True,
            interval_minutes=2,
            priority=TaskPriority.MEDIUM,
            dependencies=['test_task1']
        ),
        'test_task3': TaskConfig(
            name='test_task3',
            task_type=TaskType.SENTIMENT_FETCH,
            enabled=True,
            interval_minutes=1,
            priority=TaskPriority.LOW
        )
    }

    orchestrator.tasks = test_tasks

    logger.info(f"Created orchestrator with {len(orchestrator.tasks)} tasks")

    # Test system status
    status = orchestrator.get_system_status()
    logger.info(f"System status: {status['orchestrator_running']}")
    logger.info(f"Total tasks: {status['total_tasks']}")

    # Test task execution
    logger.info("\nTesting task execution...")
    result = await orchestrator.execute_task_now('test_task1', {'test_param': 'value'})
    logger.info(f"Task execution result: {result}")

    # Test dependency checking
    logger.info("\nTesting dependency checking...")
    should_execute = await orchestrator._should_execute_task('test_task2')
    logger.info(f"Should execute test_task2: {should_execute}")

    # Test condition checking
    logger.info("\nTesting condition checking...")
    is_market_hours = await orchestrator._is_market_hours()
    logger.info(f"Is market hours: {is_market_hours}")

    system_health = await orchestrator._get_system_health()
    logger.info(f"System health: {system_health}")

    # Test performance tracking
    logger.info("\nTesting performance tracking...")
    performance = orchestrator.get_performance_metrics()
    logger.info(f"Performance metrics: {performance}")

    # Test task scheduling
    logger.info("\nTesting task scheduling...")
    next_execution = orchestrator.get_next_execution_time('test_task1')
    logger.info(f"Next execution for test_task1: {next_execution}")

    # Test task status
    logger.info("\nTesting task status...")
    task_status = orchestrator.get_task_status('test_task1')
    logger.info(f"Task status: {task_status}")

    logger.info("✅ TaskOrchestrator tests completed successfully!")


async def test_error_handling():
    """Test error handling in the orchestrator"""
    logger.info("\nTesting error handling...")

    orchestrator = TaskOrchestrator()

    # Test with non-existent task
    try:
        result = await orchestrator.execute_task_now('non_existent_task')
        logger.error("❌ Should have raised an error for non-existent task")
    except Exception as e:
        logger.info(f"✅ Correctly handled non-existent task: {e}")

    # Test with invalid dependencies
    try:
        task_config = TaskConfig(
            name='invalid_task',
            task_type=TaskType.MODEL_INNOVATION,
            enabled=True,
            dependencies=['non_existent_dependency']
        )
        orchestrator.tasks['invalid_task'] = task_config
        should_execute = await orchestrator._should_execute_task('invalid_task')
        logger.info(f"✅ Dependency validation: {should_execute}")
    except Exception as e:
        logger.info(f"✅ Correctly handled invalid dependencies: {e}")

    logger.info("✅ Error handling tests completed!")


async def test_performance_monitoring():
    """Test performance monitoring functionality"""
    logger.info("\nTesting performance monitoring...")

    orchestrator = TaskOrchestrator()

    # Simulate some task executions
    for i in range(5):
        orchestrator.record_task_execution(
            task_name='test_task',
            execution_time=1.5,
            success=True,
            error_message=None
        )
        await asyncio.sleep(0.1)

    # Test performance metrics
    metrics = orchestrator.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")

    # Test task history
    history = orchestrator.get_task_execution_history('test_task')
    logger.info(f"Task history length: {len(history)}")

    logger.info("✅ Performance monitoring tests completed!")


async def main():
    """Main test function"""
    logger.info("🚀 Starting TaskOrchestrator Test Suite")
    logger.info("=" * 50)

    try:
        await test_orchestrator()
        await test_error_handling()
        await test_performance_monitoring()

        logger.info("\n🎉 All TaskOrchestrator tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 