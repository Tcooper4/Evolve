"""
Tests for the Base Agent Interface

This module tests the base agent interface and ensures all agents
properly implement the required functionality.
"""

import sys
import os
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
from trading.agents.base_agent_interface import (
    BaseAgent, AgentConfig, AgentStatus, AgentResult
)


class TestAgentConfig:
    """Test the AgentConfig dataclass."""
    
    def test_agent_config_creation(self):
        """Test creating an AgentConfig instance."""
        config = AgentConfig(
            name="test_agent",
            enabled=True,
            priority=1,
            max_concurrent_runs=2,
            timeout_seconds=300,
            retry_attempts=3,
            custom_config={"param": "value"}
        )
        
        assert config.name == "test_agent"
        assert config.enabled is True
        assert config.priority == 1
        assert config.max_concurrent_runs == 2
        assert config.timeout_seconds == 300
        assert config.retry_attempts == 3
        assert config.custom_config == {"param": "value"}
    
    def test_agent_config_defaults(self):
        """Test AgentConfig with default values."""
        config = AgentConfig(name="test_agent")
        
        assert config.name == "test_agent"
        assert config.enabled is True
        assert config.priority == 1
        assert config.max_concurrent_runs == 1
        assert config.timeout_seconds == 300
        assert config.retry_attempts == 3
        assert config.custom_config is None


class TestAgentStatus:
    """Test the AgentStatus dataclass."""
    
    def test_agent_status_creation(self):
        """Test creating an AgentStatus instance."""
        status = AgentStatus(
            name="test_agent",
            enabled=True,
            is_running=False,
            last_run=datetime.now(),
            last_success=datetime.now(),
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            current_error="Test error"
        )
        
        assert status.name == "test_agent"
        assert status.enabled is True
        assert status.is_running is False
        assert status.total_runs == 10
        assert status.successful_runs == 8
        assert status.failed_runs == 2
        assert status.current_error == "Test error"
    
    def test_agent_status_defaults(self):
        """Test AgentStatus with default values."""
        status = AgentStatus(
            name="test_agent",
            enabled=True,
            is_running=False
        )
        
        assert status.name == "test_agent"
        assert status.enabled is True
        assert status.is_running is False
        assert status.last_run is None
        assert status.last_success is None
        assert status.total_runs == 0
        assert status.successful_runs == 0
        assert status.failed_runs == 0
        assert status.current_error is None


class TestAgentResult:
    """Test the AgentResult dataclass."""
    
    def test_agent_result_creation(self):
        """Test creating an AgentResult instance."""
        result = AgentResult(
            success=True,
            data={"key": "value"},
            error_message=None,
            execution_time=1.5,
            timestamp=datetime.now()
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error_message is None
        assert result.execution_time == 1.5
        assert isinstance(result.timestamp, datetime)
    
    def test_agent_result_defaults(self):
        """Test AgentResult with default values."""
        result = AgentResult(success=False)
        
        assert result.success is False
        assert result.data is None
        assert result.error_message is None
        assert result.execution_time == 0.0
        assert isinstance(result.timestamp, datetime)
    
    def test_agent_result_post_init(self):
        """Test AgentResult post_init sets timestamp if None."""
        result = AgentResult(success=True, timestamp=None)
        
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class MockAgent(BaseAgent):
    """Mock agent for testing the base interface."""
    
    version = "1.0.0"
    description = "Mock agent for testing"
    author = "Test Author"
    tags = ["test", "mock"]
    capabilities = ["test_execution"]
    dependencies = []
    
    def __init__(self, config: AgentConfig, should_fail: bool = False):
        """Initialize mock agent."""
        super().__init__(config)
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute mock agent logic."""
        self.execution_count += 1
        
        if self.should_fail:
            return AgentResult(
                success=False,
                error_message="Mock agent failure",
                data={"execution_count": self.execution_count}
            )
        
        return AgentResult(
            success=True,
            data={
                "message": kwargs.get("message", "Mock execution successful"),
                "execution_count": self.execution_count,
                "kwargs": kwargs
            }
        )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        return "message" in kwargs


class TestBaseAgent:
    """Test the BaseAgent abstract base class."""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        assert agent.config == config
        assert agent.status.name == "test_agent"
        assert agent.status.enabled is True
        assert agent.status.is_running is False
        assert agent.execution_count == 0
    
    def test_base_agent_enable(self):
        """Test enabling an agent."""
        config = AgentConfig(name="test_agent", enabled=False)
        agent = MockAgent(config)
        
        assert agent.status.enabled is False
        agent.enable()
        assert agent.status.enabled is True
        assert agent.config.enabled is True
    
    def test_base_agent_disable(self):
        """Test disabling an agent."""
        config = AgentConfig(name="test_agent", enabled=True)
        agent = MockAgent(config)
        
        assert agent.status.enabled is True
        agent.disable()
        assert agent.status.enabled is False
        assert agent.config.enabled is False
    
    def test_base_agent_is_enabled(self):
        """Test checking if agent is enabled."""
        config = AgentConfig(name="test_agent", enabled=True)
        agent = MockAgent(config)
        
        assert agent.is_enabled() is True
        
        agent.disable()
        assert agent.is_enabled() is False
    
    def test_base_agent_get_status(self):
        """Test getting agent status."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        status = agent.get_status()
        assert isinstance(status, AgentStatus)
        assert status.name == "test_agent"
        assert status.enabled is True
    
    def test_base_agent_get_config(self):
        """Test getting agent configuration."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        retrieved_config = agent.get_config()
        assert retrieved_config == config
    
    def test_base_agent_update_config(self):
        """Test updating agent configuration."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        new_config = {"priority": 5, "timeout_seconds": 600}
        agent.update_config(new_config)
        
        assert agent.config.priority == 5
        assert agent.config.timeout_seconds == 600
    
    def test_base_agent_update_config_custom(self):
        """Test updating agent custom configuration."""
        config = AgentConfig(name="test_agent", custom_config={"param": "old"})
        agent = MockAgent(config)
        
        new_config = {"custom_param": "new_value"}
        agent.update_config(new_config)
        
        assert agent.config.custom_config["custom_param"] == "new_value"
    
    def test_base_agent_validate_input_default(self):
        """Test default input validation."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Default implementation should return True
        assert agent.validate_input() is True
        assert agent.validate_input(key = os.getenv('KEY', '')) is True
    
    def test_base_agent_handle_error(self):
        """Test error handling."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        error = Exception("Test error")
        result = agent.handle_error(error)
        
        assert result.success is False
        assert result.error_message == "Test error"
        assert agent.status.failed_runs == 1
        assert agent.status.current_error == "Test error"
        assert agent.status.is_running is False
    
    def test_base_agent_get_metadata(self):
        """Test getting agent metadata."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        metadata = agent.get_metadata()
        
        assert metadata["name"] == "test_agent"
        assert metadata["version"] == "1.0.0"
        assert metadata["description"] == "Mock agent for testing"
        assert metadata["author"] == "Test Author"
        assert metadata["tags"] == ["test", "mock"]
        assert metadata["capabilities"] == ["test_execution"]
        assert metadata["dependencies"] == []


class TestBaseAgentExecution:
    """Test BaseAgent execution methods."""
    
    @pytest.mark.asyncio
    async def test_base_agent_run_success(self):
        """Test successful agent execution."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        result = await agent.run(message="Test message")
        
        assert result.success is True
        assert result.data["message"] == "Test message"
        assert result.data["execution_count"] == 1
        assert result.execution_time > 0
        assert agent.status.successful_runs == 1
        assert agent.status.total_runs == 1
        assert agent.status.last_success is not None
    
    @pytest.mark.asyncio
    async def test_base_agent_run_failure(self):
        """Test failed agent execution."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config, should_fail=True)
        
        result = await agent.run(message="Test message")
        
        assert result.success is False
        assert result.error_message == "Mock agent failure"
        assert result.execution_time > 0
        assert agent.status.failed_runs == 1
        assert agent.status.total_runs == 1
        assert agent.status.last_success is None
    
    @pytest.mark.asyncio
    async def test_base_agent_run_disabled(self):
        """Test running a disabled agent."""
        config = AgentConfig(name="test_agent", enabled=False)
        agent = MockAgent(config)
        
        result = await agent.run(message="Test message")
        
        assert result.success is False
        assert "disabled" in result.error_message.lower()
        assert agent.status.total_runs == 0
    
    @pytest.mark.asyncio
    async def test_base_agent_run_invalid_input(self):
        """Test running agent with invalid input."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Mock validate_input to return False
        with patch.object(agent, 'validate_input', return_value=False):
            result = await agent.run(invalid_param="value")
        
        assert result.success is False
        assert "invalid input" in result.error_message.lower()
        assert agent.status.total_runs == 0
    
    @pytest.mark.asyncio
    async def test_base_agent_run_exception(self):
        """Test running agent that raises an exception."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Mock execute to raise an exception
        async def mock_execute(**kwargs):
            raise RuntimeError("Test exception")
        
        with patch.object(agent, 'execute', side_effect=mock_execute):
            result = await agent.run(message="Test message")
        
        assert result.success is False
        assert result.error_message == "Test exception"
        assert agent.status.failed_runs == 1
        assert agent.status.current_error == "Test exception"
    
    @pytest.mark.asyncio
    async def test_base_agent_run_status_updates(self):
        """Test that agent status is properly updated during execution."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Check initial status
        assert agent.status.is_running is False
        assert agent.status.total_runs == 0
        assert agent.status.last_run is None
        
        # Run agent
        result = await agent.run(message="Test message")
        
        # Check updated status
        assert result.success is True
        assert agent.status.is_running is False  # Should be False after completion
        assert agent.status.total_runs == 1
        assert agent.status.last_run is not None
        assert agent.status.last_success is not None
        assert agent.status.current_error is None


class TestBaseAgentCustomValidation:
    """Test custom input validation in agents."""
    
    def test_mock_agent_validate_input(self):
        """Test MockAgent's custom input validation."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Valid input
        assert agent.validate_input(message="test") is True
        
        # Invalid input
        assert agent.validate_input(no_message="test") is False
        assert agent.validate_input() is False


class TestBaseAgentConcurrency:
    """Test agent concurrency handling."""
    
    @pytest.mark.asyncio
    async def test_agent_concurrent_execution(self):
        """Test multiple concurrent agent executions."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # Run multiple executions concurrently
        tasks = [
            agent.run(message=f"Message {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.success is True
        
        # Should have 5 total runs
        assert agent.status.total_runs == 5
        assert agent.status.successful_runs == 5
        assert agent.execution_count == 5
    
    @pytest.mark.asyncio
    async def test_agent_max_concurrent_runs(self):
        """Test agent respects max_concurrent_runs setting."""
        config = AgentConfig(name="test_agent", max_concurrent_runs=2)
        agent = MockAgent(config)
        
        # This should work even with max_concurrent_runs=2
        # because the limit is enforced by the manager, not the agent itself
        tasks = [
            agent.run(message=f"Message {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.success is True
        
        assert agent.status.total_runs == 5


class TestBaseAgentErrorRecovery:
    """Test agent error recovery and retry logic."""
    
    @pytest.mark.asyncio
    async def test_agent_error_recovery(self):
        """Test that agent can recover from errors."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # First run fails
        agent.should_fail = True
        result1 = await agent.run(message="Test 1")
        assert result1.success is False
        assert agent.status.failed_runs == 1
        
        # Second run succeeds
        agent.should_fail = False
        result2 = await agent.run(message="Test 2")
        assert result2.success is True
        assert agent.status.successful_runs == 1
        assert agent.status.failed_runs == 1
        assert agent.status.total_runs == 2
    
    @pytest.mark.asyncio
    async def test_agent_error_clearing(self):
        """Test that agent errors are cleared on successful execution."""
        config = AgentConfig(name="test_agent")
        agent = MockAgent(config)
        
        # First run fails
        agent.should_fail = True
        result1 = await agent.run(message="Test 1")
        assert result1.success is False
        assert agent.status.current_error is not None
        
        # Second run succeeds
        agent.should_fail = False
        result2 = await agent.run(message="Test 2")
        assert result2.success is True
        assert agent.status.current_error is None 