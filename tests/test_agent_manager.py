"""
Tests for Enhanced Agent Manager with Batch 12 Features

This module tests the enhanced agent manager with comprehensive retry/backoff logic,
error handling, and graceful degradation capabilities.
"""

import asyncio
import logging
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from trading.agents.agent_manager import (
    EnhancedAgentManager,
    AgentManagerConfig,
    RetryConfig,
    AgentTask,
    AgentManagementRequest,
    AgentManagementResult
)
from trading.agents.base_agent_interface import (
    AgentConfig,
    AgentResult,
    AgentStatus,
    BaseAgent
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, fail_count: int = 0):
        """Initialize mock agent.
        
        Args:
            name: Agent name
            should_fail: Whether agent should fail
            fail_count: Number of times to fail before succeeding
        """
        config = AgentConfig(
            name=name,
            enabled=True,
            priority=1,
            max_concurrent_runs=1,
            timeout_seconds=30,
            retry_attempts=3
        )
        super().__init__(config)
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.execution_count = 0
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute mock agent."""
        self.execution_count += 1
        
        if self.should_fail and self.execution_count <= self.fail_count:
            raise Exception(f"Mock agent {self.config.name} failed on attempt {self.execution_count}")
        
        return AgentResult(
            success=True,
            data={"message": f"Mock agent {self.config.name} executed successfully"},
            metadata={"execution_count": self.execution_count}
        )


class TestEnhancedAgentManager:
    """Test cases for enhanced agent manager."""
    
    @pytest.fixture
    def agent_manager(self):
        """Create agent manager for testing."""
        config = AgentManagerConfig(
            max_concurrent_agents=3,
            execution_timeout=10,
            enable_logging=True,
            enable_metrics=True
        )
        return EnhancedAgentManager(config)
    
    @pytest.fixture
    def retry_config(self):
        """Create retry configuration for testing."""
        return RetryConfig(
            max_retries=3,
            base_delay=0.1,  # Short delays for testing
            max_delay=1.0,
            backoff_factor=2.0,
            jitter=True,
            exponential_backoff=True
        )
    
    def test_agent_manager_initialization(self, agent_manager):
        """Test agent manager initialization."""
        assert agent_manager.config is not None
        assert agent_manager.agent_registry == {}
        assert agent_manager.pending_tasks == {}
        assert agent_manager.completed_tasks == {}
        assert agent_manager.failed_tasks == {}
        assert agent_manager.global_error_count == 0
    
    def test_register_agent(self, agent_manager):
        """Test agent registration."""
        mock_agent = MockAgent("test_agent")
        agent_manager.register_agent("test_agent", MockAgent, mock_agent.config)
        
        assert "test_agent" in agent_manager.agent_registry
        assert agent_manager.agent_registry["test_agent"].agent_class == MockAgent
        assert agent_manager.agent_registry["test_agent"].config.name == "test_agent"
    
    def test_register_agent_with_retry_config(self, agent_manager, retry_config):
        """Test agent registration with retry configuration."""
        mock_agent = MockAgent("test_agent")
        agent_manager.register_agent("test_agent", MockAgent, mock_agent.config)
        
        # Update retry config
        agent_manager.update_agent_retry_config("test_agent", retry_config)
        
        assert agent_manager.agent_registry["test_agent"].retry_config == retry_config
    
    @pytest.mark.asyncio
    async def test_execute_agent_success(self, agent_manager):
        """Test successful agent execution."""
        # Register successful agent
        mock_agent = MockAgent("success_agent", should_fail=False)
        agent_manager.register_agent("success_agent", MockAgent, mock_agent.config)
        
        # Execute agent
        result = await agent_manager.execute_agent_with_retry("success_agent", test_param="value")
        
        assert result.success is True
        assert "message" in result.data
        assert result.data["message"] == "Mock agent success_agent executed successfully"
        assert len(agent_manager.completed_tasks) == 1
        assert len(agent_manager.failed_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_agent_failure_no_retry(self, agent_manager):
        """Test agent execution failure without retry."""
        # Register failing agent
        mock_agent = MockAgent("fail_agent", should_fail=True, fail_count=10)
        agent_manager.register_agent("fail_agent", MockAgent, mock_agent.config)
        
        # Execute agent with no retries
        retry_config = RetryConfig(max_retries=0)
        result = await agent_manager.execute_agent_with_retry("fail_agent", retry_config=retry_config)
        
        assert result.success is False
        assert "MaxRetriesExceeded" in result.error_type
        assert len(agent_manager.failed_tasks) == 1
        assert len(agent_manager.completed_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_agent_with_retry_success(self, agent_manager, retry_config):
        """Test agent execution with retry that eventually succeeds."""
        # Register agent that fails twice then succeeds
        mock_agent = MockAgent("retry_agent", should_fail=True, fail_count=2)
        agent_manager.register_agent("retry_agent", MockAgent, mock_agent.config)
        
        # Execute agent with retry
        start_time = time.time()
        result = await agent_manager.execute_agent_with_retry("retry_agent", retry_config=retry_config)
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert "message" in result.data
        assert len(agent_manager.completed_tasks) == 1
        assert len(agent_manager.failed_tasks) == 0
        
        # Check that retries took some time (due to backoff)
        assert execution_time > 0.1  # At least some delay
    
    @pytest.mark.asyncio
    async def test_execute_agent_with_retry_failure(self, agent_manager, retry_config):
        """Test agent execution with retry that eventually fails."""
        # Register agent that always fails
        mock_agent = MockAgent("always_fail_agent", should_fail=True, fail_count=10)
        agent_manager.register_agent("always_fail_agent", MockAgent, mock_agent.config)
        
        # Execute agent with retry
        start_time = time.time()
        result = await agent_manager.execute_agent_with_retry("always_fail_agent", retry_config=retry_config)
        execution_time = time.time() - start_time
        
        assert result.success is False
        assert "MaxRetriesExceeded" in result.error_type
        assert len(agent_manager.failed_tasks) == 1
        assert len(agent_manager.completed_tasks) == 0
        
        # Check that retries took some time
        assert execution_time > 0.1
    
    @pytest.mark.asyncio
    async def test_execute_agent_timeout(self, agent_manager):
        """Test agent execution timeout."""
        # Create slow agent
        class SlowAgent(MockAgent):
            async def execute(self, **kwargs) -> AgentResult:
                await asyncio.sleep(2.0)  # Sleep longer than timeout
                return AgentResult(success=True, data={"message": "Slow execution"})
        
        slow_agent = SlowAgent("slow_agent")
        agent_manager.register_agent("slow_agent", SlowAgent, slow_agent.config)
        
        # Execute with short timeout
        retry_config = RetryConfig(max_retries=1, base_delay=0.1)
        result = await agent_manager.execute_agent_with_retry("slow_agent", retry_config=retry_config)
        
        assert result.success is False
        assert "ExecutionTimeout" in result.error_type
    
    @pytest.mark.asyncio
    async def test_execute_agent_dependency_wait(self, agent_manager):
        """Test agent execution with dependency wait."""
        # Register two agents
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        
        agent_manager.register_agent("agent1", MockAgent, agent1.config)
        agent_manager.register_agent("agent2", MockAgent, agent2.config)
        
        # Create task with dependency
        task = AgentTask(
            agent_name="agent2",
            task_id="task2",
            kwargs={},
            retry_config=RetryConfig(),
            created_at=datetime.now()
        )
        task.dependencies = ["task1"]  # Depends on task1
        
        # This should wait for dependency
        assert not agent_manager._check_dependencies("task2", {"task2": ["task1"]}, {})
    
    def test_calculate_backoff_delay(self, agent_manager, retry_config):
        """Test backoff delay calculation."""
        # Test exponential backoff
        delay1 = agent_manager._calculate_backoff_delay(1, retry_config)
        delay2 = agent_manager._calculate_backoff_delay(2, retry_config)
        delay3 = agent_manager._calculate_backoff_delay(3, retry_config)
        
        assert delay1 > 0
        assert delay2 > delay1
        assert delay3 > delay2
        
        # Test with jitter disabled
        retry_config_no_jitter = RetryConfig(jitter=False, base_delay=0.1)
        delay_no_jitter = agent_manager._calculate_backoff_delay(1, retry_config_no_jitter)
        assert delay_no_jitter == 0.1
    
    def test_log_error_with_traceback(self, agent_manager):
        """Test error logging with traceback."""
        error = Exception("Test error")
        context = {"test_param": "value"}
        
        agent_manager._log_error_with_traceback("test_agent", error, "task1", 1, context)
        
        assert agent_manager.global_error_count == 1
        assert agent_manager.agent_error_counts["test_agent"] == 1
        assert len(agent_manager.error_log) == 1
        
        error_entry = agent_manager.error_log[0]
        assert error_entry["agent_name"] == "test_agent"
        assert error_entry["task_id"] == "task1"
        assert error_entry["attempt"] == 1
        assert error_entry["error_type"] == "Exception"
        assert error_entry["error_message"] == "Test error"
        assert "traceback" in error_entry
        assert error_entry["context"] == context
    
    def test_get_error_statistics(self, agent_manager):
        """Test error statistics retrieval."""
        # Add some errors
        error1 = Exception("Error 1")
        error2 = ValueError("Error 2")
        
        agent_manager._log_error_with_traceback("agent1", error1, "task1", 1)
        agent_manager._log_error_with_traceback("agent1", error2, "task2", 1)
        agent_manager._log_error_with_traceback("agent2", error1, "task3", 1)
        
        stats = agent_manager.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["agent_error_counts"]["agent1"] == 2
        assert stats["agent_error_counts"]["agent2"] == 1
        assert stats["error_types"]["Exception"] == 2
        assert stats["error_types"]["ValueError"] == 1
        assert len(stats["recent_errors"]) <= 100
    
    def test_clear_error_log(self, agent_manager):
        """Test error log clearing."""
        # Add some errors
        error = Exception("Test error")
        agent_manager._log_error_with_traceback("test_agent", error, "task1", 1)
        
        assert len(agent_manager.error_log) == 1
        
        # Clear errors older than 0 days (should clear all)
        cleared_count = agent_manager.clear_error_log(older_than_days=0)
        
        assert cleared_count == 1
        assert len(agent_manager.error_log) == 0
    
    def test_get_agent_retry_config(self, agent_manager, retry_config):
        """Test getting agent retry configuration."""
        # Register agent
        mock_agent = MockAgent("test_agent")
        agent_manager.register_agent("test_agent", MockAgent, mock_agent.config)
        
        # Get default retry config
        default_config = agent_manager.get_agent_retry_config("test_agent")
        assert default_config is not None
        
        # Update retry config
        agent_manager.update_agent_retry_config("test_agent", retry_config)
        
        # Get updated config
        updated_config = agent_manager.get_agent_retry_config("test_agent")
        assert updated_config == retry_config
    
    def test_update_agent_retry_config(self, agent_manager, retry_config):
        """Test updating agent retry configuration."""
        # Register agent
        mock_agent = MockAgent("test_agent")
        agent_manager.register_agent("test_agent", MockAgent, mock_agent.config)
        
        # Update retry config
        success = agent_manager.update_agent_retry_config("test_agent", retry_config)
        assert success is True
        
        # Try to update non-existent agent
        success = agent_manager.update_agent_retry_config("non_existent", retry_config)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, agent_manager):
        """Test concurrent agent execution."""
        # Register multiple agents
        agents = []
        for i in range(3):
            agent = MockAgent(f"agent_{i}")
            agent_manager.register_agent(f"agent_{i}", MockAgent, agent.config)
            agents.append(agent)
        
        # Execute all agents concurrently
        tasks = []
        for i in range(3):
            task = agent_manager.execute_agent_with_retry(f"agent_{i}", param=f"value_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Check all succeeded
        for result in results:
            assert result.success is True
        
        # Check all agents were executed
        for agent in agents:
            assert agent.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_agent_execution_with_shared_context(self, agent_manager):
        """Test agent execution with shared context."""
        # Register agent
        mock_agent = MockAgent("context_agent")
        agent_manager.register_agent("context_agent", MockAgent, mock_agent.config)
        
        # Execute with context
        context = {"shared_data": "test_value"}
        result = await agent_manager.execute_agent_with_retry("context_agent", context=context)
        
        assert result.success is True
        # Agent should have access to context through kwargs
    
    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid config
        valid_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0
        )
        assert valid_config.max_retries == 3
        assert valid_config.base_delay == 1.0
        
        # Test default values
        default_config = RetryConfig()
        assert default_config.max_retries == 3
        assert default_config.jitter is True
        assert default_config.exponential_backoff is True
    
    @pytest.mark.asyncio
    async def test_agent_execution_metrics(self, agent_manager):
        """Test agent execution metrics tracking."""
        # Register and execute agent
        mock_agent = MockAgent("metrics_agent")
        agent_manager.register_agent("metrics_agent", MockAgent, mock_agent.config)
        
        result = await agent_manager.execute_agent_with_retry("metrics_agent")
        
        # Check metrics were updated
        assert result.success is True
        # Additional metrics checks could be added here
    
    def test_task_creation_and_management(self, agent_manager):
        """Test task creation and management."""
        # Create task
        task = AgentTask(
            agent_name="test_agent",
            task_id="test_task",
            kwargs={"param": "value"},
            retry_config=RetryConfig(),
            created_at=datetime.now()
        )
        
        assert task.agent_name == "test_agent"
        assert task.task_id == "test_task"
        assert task.attempts == 0
        assert task.error_history == []
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, agent_manager):
        """Test graceful degradation when agents fail."""
        # Register multiple agents with different failure patterns
        agents = [
            MockAgent("always_fail", should_fail=True, fail_count=10),
            MockAgent("sometimes_fail", should_fail=True, fail_count=1),
            MockAgent("never_fail", should_fail=False)
        ]
        
        for i, agent in enumerate(agents):
            agent_manager.register_agent(f"agent_{i}", MockAgent, agent.config)
        
        # Execute all agents
        results = []
        for i in range(3):
            result = await agent_manager.execute_agent_with_retry(f"agent_{i}")
            results.append(result)
        
        # Check that some succeeded and some failed
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)
        
        assert success_count > 0  # At least one should succeed
        assert failure_count > 0  # At least one should fail
        assert success_count + failure_count == 3
    
    def test_error_recovery(self, agent_manager):
        """Test error recovery mechanisms."""
        # Add some errors
        error = Exception("Recovery test error")
        agent_manager._log_error_with_traceback("recovery_agent", error, "task1", 1)
        
        # Check error was logged
        assert agent_manager.global_error_count == 1
        
        # Clear errors
        agent_manager.clear_error_log(older_than_days=0)
        
        # Check error was cleared
        assert agent_manager.global_error_count == 0
        assert len(agent_manager.error_log) == 0


class TestRetryConfig:
    """Test cases for retry configuration."""
    
    def test_retry_config_defaults(self):
        """Test retry configuration default values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
        assert config.exponential_backoff is True
        assert Exception in config.retry_on_exceptions
    
    def test_retry_config_custom_values(self):
        """Test retry configuration with custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_factor=1.5,
            jitter=False,
            exponential_backoff=False
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
        assert config.exponential_backoff is False


class TestAgentTask:
    """Test cases for agent task."""
    
    def test_agent_task_creation(self):
        """Test agent task creation."""
        task = AgentTask(
            agent_name="test_agent",
            task_id="test_task",
            kwargs={"param": "value"},
            retry_config=RetryConfig(),
            created_at=datetime.now()
        )
        
        assert task.agent_name == "test_agent"
        assert task.task_id == "test_task"
        assert task.kwargs == {"param": "value"}
        assert task.attempts == 0
        assert task.retry_count == 0
        assert task.error_history == []
        assert task.last_attempt is None
        assert task.next_retry is None
    
    def test_agent_task_post_init(self):
        """Test agent task post-initialization."""
        task = AgentTask(
            agent_name="test_agent",
            task_id="test_task",
            kwargs={},
            retry_config=RetryConfig(),
            created_at=datetime.now()
        )
        
        # Check that error_history was initialized
        assert task.error_history == []


if __name__ == "__main__":
    pytest.main([__file__]) 