#!/usr/bin/env python3
"""
Simple test script for TaskOrchestrator
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.task_orchestrator import TaskOrchestrator, TaskConfig, TaskType, TaskPriority


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
    print("Testing TaskOrchestrator...")
    
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
    
    print(f"Created orchestrator with {len(orchestrator.tasks)} tasks")
    
    # Test system status
    status = orchestrator.get_system_status()
    print(f"System status: {status['orchestrator_running']}")
    print(f"Total tasks: {status['total_tasks']}")
    
    # Test task execution
    print("\nTesting task execution...")
    result = await orchestrator.execute_task_now('test_task1', {'test_param': 'value'})
    print(f"Task execution result: {result}")
    
    # Test dependency checking
    print("\nTesting dependency checking...")
    should_execute = await orchestrator._should_execute_task('test_task2')
    print(f"Should execute test_task2: {should_execute}")
    
    # Test condition checking
    print("\nTesting condition checking...")
    is_market_hours = await orchestrator._is_market_hours()
    print(f"Is market hours: {is_market_hours}")
    
    system_health = await orchestrator._get_system_health()
    print(f"System health: {system_health}")
    
    # Test performance tracking
    print("\nTesting performance tracking...")
    execution = orchestrator.TaskExecution(
        task_id="test_exec",
        task_name="test_task1",
        task_type=TaskType.MODEL_INNOVATION,
        start_time="2023-01-01T10:00:00",
        end_time="2023-01-01T10:00:30",
        status=orchestrator.TaskStatus.COMPLETED,
        duration_seconds=30.0
    )
    
    orchestrator._update_task_performance("test_task1", execution)
    
    # Check updated status
    task_status = orchestrator.get_task_status("test_task1")
    print(f"Task status: {task_status['success_rate']}")
    
    # Export status report
    print("\nTesting status report export...")
    report_path = orchestrator.export_status_report()
    print(f"Status report exported to: {report_path}")
    
    print("\nTaskOrchestrator test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator()) 