Unit tests for Swarm Orchestrator.

Tests the SwarmOrchestrator class for coordinating multiple agents as async jobs.

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from trading.agents.swarm_orchestrator import (
    SwarmOrchestrator,
    SwarmConfig,
    AgentType,
    AgentStatus,
    SwarmTask
)
from trading.agents.base_agent_interface import AgentConfig, AgentResult


class TestSwarmOrchestrator:
   Test cases for Swarm Orchestrator."   @pytest.fixture
    def swarm_config(self):
      Create SwarmConfig for testing.
        return SwarmConfig(
            max_concurrent_agents=3,
            coordination_backend="sqlite",
            task_timeout=60
            retry_attempts=2,
            enable_logging=True,
            enable_monitoring=True
        )

    @pytest.fixture
    def orchestrator(self, swarm_config):
      Create SwarmOrchestrator instance for testing.
        return SwarmOrchestrator(swarm_config)

    @pytest.fixture
    def agent_config(self):
      Create AgentConfig for testing.      return AgentConfig(
            name="test_agent",
            enabled=True,
            priority=1           max_concurrent_runs=1,
            timeout_seconds=30
            retry_attempts=2,
            custom_config={}
        )

    def test_initialization(self, orchestrator, swarm_config):
      Test orchestrator initialization.   assert orchestrator.config == swarm_config
        assert orchestrator.tasks ==[object Object]   assert orchestrator.running_tasks == set()
        assert orchestrator.completed_tasks ==[object Object]   assert orchestrator.running is False
        assert len(orchestrator.agent_registry) > 0
        assert AgentType.ALPHA_GEN in orchestrator.agent_registry
        assert AgentType.SIGNAL_TESTER in orchestrator.agent_registry

    def test_agent_registry(self, orchestrator):
      Test agent registry contains expected agents.""
        expected_agents = [
            AgentType.ALPHA_GEN,
            AgentType.SIGNAL_TESTER,
            AgentType.RISK_VALIDATOR,
            AgentType.SENTIMENT,
            AgentType.ALPHA_REGISTRY,
            AgentType.WALK_FORWARD,
            AgentType.REGIME_DETECTION
        ]
        
        for agent_type in expected_agents:
            assert agent_type in orchestrator.agent_registry
            assert orchestrator.agent_registry[agent_type] is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
      Test orchestrator start and stop functionality.      # Test start
        await orchestrator.start()
        assert orchestrator.running is True
        assert orchestrator.loop is not None
        
        # Test stop
        await orchestrator.stop()
        assert orchestrator.running is false

    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator, agent_config):
      Test task submission."""
        input_data = {"test":data}
        
        task_id = await orchestrator.submit_task(
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data=input_data
        )
        
        assert task_id is not None
        assert task_id in orchestrator.tasks
        
        task = orchestrator.tasks[task_id]
        assert task.agent_type == AgentType.ALPHA_GEN
        assert task.agent_config == agent_config
        assert task.input_data == input_data
        assert task.status == AgentStatus.IDLE
        assert task.created_at is not None

    @pytest.mark.asyncio
    async def test_submit_task_with_dependencies(self, orchestrator, agent_config):
      Test task submission with dependencies.""    # Submit first task
        task1_id = await orchestrator.submit_task(
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data={task": 1}
        )
        
        # Submit second task with dependency
        task2_id = await orchestrator.submit_task(
            agent_type=AgentType.SIGNAL_TESTER,
            agent_config=agent_config,
            input_data={"task: 2},      dependencies=[task1_id]
        )
        
        task2orchestrator.tasks[task2_id]
        assert task1_id in task2.dependencies

    @pytest.mark.asyncio
    async def test_task_execution_success(self, orchestrator, agent_config):
      Test successful task execution.      # Mock agent execution
        mock_result = AgentResult(
            success=True,
            data={result": "success"},
            error_message=None,
            error_type=None
        )
        
        with patch.object(orchestrator, '_run_agent', return_value=mock_result):
            task_id = await orchestrator.submit_task(
                agent_type=AgentType.ALPHA_GEN,
                agent_config=agent_config,
                input_data={"test": "data"}
            )
            
            # Wait for task to complete
            await asyncio.sleep(0.1      
            task = orchestrator.get_task_status(task_id)
            assert task.status == AgentStatus.COMPLETED
            assert task.result == mock_result
            assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_execution_failure(self, orchestrator, agent_config):
      Test task execution failure.      # Mock agent execution to raise exception
        with patch.object(orchestrator, _run_agent', side_effect=Exception("Test error")):
            task_id = await orchestrator.submit_task(
                agent_type=AgentType.ALPHA_GEN,
                agent_config=agent_config,
                input_data={"test": "data"}
            )
            
            # Wait for task to complete
            await asyncio.sleep(0.1      
            task = orchestrator.get_task_status(task_id)
            assert task.status == AgentStatus.FAILED
            assert "Test error" in task.error_message
            assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, orchestrator, agent_config):
      Test concurrency limit enforcement."""
        # Set very low concurrency limit
        orchestrator.config.max_concurrent_agents = 1        
        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            task_id = await orchestrator.submit_task(
                agent_type=AgentType.ALPHA_GEN,
                agent_config=agent_config,
                input_data={"task": i}
            )
            task_ids.append(task_id)
        
        # Wait a bit for execution
        await asyncio.sleep(0.1)
        
        # Check that only one task is running
        assert len(orchestrator.running_tasks) <= 1

    @pytest.mark.asyncio
    async def test_dependency_checking(self, orchestrator, agent_config):
      Test dependency checking.""# Submit dependent task first
        task2_id = await orchestrator.submit_task(
            agent_type=AgentType.SIGNAL_TESTER,
            agent_config=agent_config,
            input_data={"task: 2},      dependencies=["nonexistent_task"]
        )
        
        # Wait for task to be processed
        await asyncio.sleep(0.1)
        
        # Task should not start due to missing dependency
        task2 = orchestrator.get_task_status(task2_id)
        assert task2.status == AgentStatus.IDLE

    def test_get_task_status(self, orchestrator, agent_config):
      Test getting task status.""        # Create a task manually
        task = SwarmTask(
            task_id="test_task",
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data=[object Object]ata"},
            status=AgentStatus.COMPLETED,
            created_at=datetime.now()
        )
        
        orchestrator.completed_tasks["test_task"] = task
        
        # Test getting status
        retrieved_task = orchestrator.get_task_status("test_task")
        assert retrieved_task == task
        
        # Test getting non-existent task
        assert orchestrator.get_task_status("nonexistent") is None

    def test_get_swarm_status(self, orchestrator, agent_config):
      Test getting swarm status."""
        # Add some tasks
        task1 = SwarmTask(
            task_id="task1",
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data={},
            status=AgentStatus.RUNNING,
            created_at=datetime.now()
        )
        
        task2 = SwarmTask(
            task_id="task2",
            agent_type=AgentType.SIGNAL_TESTER,
            agent_config=agent_config,
            input_data={},
            status=AgentStatus.FAILED,
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        orchestrator.tasks["task1] = task1      orchestrator.running_tasks.add("task1")
        orchestrator.completed_tasks["task2"] = task2   
        status = orchestrator.get_swarm_status()
        
        assert status["running] == orchestrator.running
        assert statustotal_tasks"] == 1
        assert status["running_tasks"] == 1
        assert status[completed_tasks"] == 1
        assert status[failed_tasks"] == 1
        assert status["max_concurrent_agents] == orchestrator.config.max_concurrent_agents

    @pytest.mark.asyncio
    async def test_cancel_task(self, orchestrator, agent_config):
      Test task cancellation."
        task_id = await orchestrator.submit_task(
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data={"test:data       )
        
        # Add task to running tasks
        orchestrator.running_tasks.add(task_id)
        
        # Cancel task
        await orchestrator._cancel_task(task_id)
        
        task = orchestrator.get_task_status(task_id)
        assert task.status == AgentStatus.CANCELLED
        assert task_id not in orchestrator.running_tasks

    @pytest.mark.asyncio
    async def test_run_agent_with_retries(self, orchestrator, agent_config):
      Test agent execution with retries.""        # Create a task
        task = SwarmTask(
            task_id="test_task",
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data=[object Object]ata"},
            status=AgentStatus.RUNNING,
            created_at=datetime.now()
        )
        
        # Mock agent class
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(side_effect=Exception("Test error"))
        
        with patch.object(orchestrator.agent_registry[AgentType.ALPHA_GEN], __call__', return_value=mock_agent):
            # Should raise exception after retries
            with pytest.raises(Exception):
                await orchestrator._run_agent(task)
            
            # Should have been called retry_attempts times
            assert mock_agent.execute.call_count == orchestrator.config.retry_attempts

    @pytest.mark.asyncio
    async def test_run_agent_timeout(self, orchestrator, agent_config):
      Test agent execution timeout.""        # Create a task
        task = SwarmTask(
            task_id="test_task",
            agent_type=AgentType.ALPHA_GEN,
            agent_config=agent_config,
            input_data=[object Object]ata"},
            status=AgentStatus.RUNNING,
            created_at=datetime.now()
        )
        
        # Mock agent class that takes too long
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        
        with patch.object(orchestrator.agent_registry[AgentType.ALPHA_GEN], __call__', return_value=mock_agent):
            # Should raise TimeoutError after retries
            with pytest.raises(asyncio.TimeoutError):
                await orchestrator._run_agent(task)

    def test_unknown_agent_type(self, orchestrator, agent_config):
      Test handling of unknown agent type.""        # Create a task with unknown agent type
        task = SwarmTask(
            task_id="test_task",
            agent_type=AgentType.STRATEGY,  # Not in registry
            agent_config=agent_config,
            input_data=[object Object]ata"},
            status=AgentStatus.RUNNING,
            created_at=datetime.now()
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            asyncio.run(orchestrator._run_agent(task)) 