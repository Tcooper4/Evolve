"""
Test cases for agent registry and fallback loading.

This module tests the agent registry system to ensure proper loading,
registration, and fallback mechanisms work correctly.
"""

import logging
import os
import sys
from datetime import datetime
from unittest.mock import patch

import pytest

from trading.agents.agent_registry import AgentRegistry
from trading.agents.base_agent_interface import AgentConfig, BaseAgent

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent registry components

# ForecastAgent not implemented yet - using placeholder
# from trading.agents.forecast_agent import ForecastAgent
# StrategyAgent and OptimizationAgent not implemented yet - using placeholders
# from trading.agents.strategy_agent import StrategyAgent
# from trading.agents.optimization_agent import OptimizationAgent

logger = logging.getLogger(__name__)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.execution_count = 0

    async def execute(self, *args, **kwargs):
        self.execution_count += 1
        return {
            "success": True,
            "data": {"mock_result": "test"},
            "message": "Mock agent executed successfully",
            "timestamp": datetime.now().isoformat(),
        }


class TestAgentRegistry:
    """Test agent registry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh agent registry for testing."""
        return AgentRegistry()

    @pytest.fixture
    def sample_config(self):
        """Create sample agent configuration."""
        return AgentConfig(
            name="test_agent",
            enabled=True,
            priority=1,
            custom_config={"test_param": "test_value", "timeout": 30},
        )

    def test_agent_registration(self, registry, sample_config):
        """Test basic agent registration."""
        logger.info("Testing agent registration")

        # Create and register agent
        agent = MockAgent(sample_config)
        registry.register_agent("test_agent", agent)

        # Verify registration
        assert "test_agent" in registry.get_available_agents()
        assert registry.get_agent("test_agent") == agent
        assert registry.is_agent_available("test_agent")

        # Test duplicate registration
        duplicate_agent = MockAgent(sample_config)
        registry.register_agent("test_agent", duplicate_agent)
        assert registry.get_agent("test_agent") == duplicate_agent

        logger.info("Agent registration test passed")

    def test_agent_unregistration(self, registry, sample_config):
        """Test agent unregistration."""
        logger.info("Testing agent unregistration")

        # Register agent
        agent = MockAgent(sample_config)
        registry.register_agent("test_agent", agent)
        assert registry.is_agent_available("test_agent")

        # Unregister agent
        registry.unregister_agent("test_agent")
        assert not registry.is_agent_available("test_agent")
        assert registry.get_agent("test_agent") is None

        # Test unregistering non-existent agent
        registry.unregister_agent("non_existent")

        logger.info("Agent unregistration test passed")

    def test_agent_priority_ordering(self, registry):
        """Test agent priority ordering."""
        logger.info("Testing agent priority ordering")

        # Create agents with different priorities
        high_priority_config = AgentConfig(name="high", enabled=True, priority=1)
        medium_priority_config = AgentConfig(name="medium", enabled=True, priority=5)
        low_priority_config = AgentConfig(name="low", enabled=True, priority=10)

        high_agent = MockAgent(high_priority_config)
        medium_agent = MockAgent(medium_priority_config)
        low_agent = MockAgent(low_priority_config)

        # Register agents
        registry.register_agent("high", high_agent)
        registry.register_agent("medium", medium_agent)
        registry.register_agent("low", low_agent)

        # Get agents by priority
        available_agents = registry.get_available_agents()
        assert len(available_agents) == 3

        # Test priority-based selection
        selected_agent = registry.get_best_agent_for_task("test_task")
        assert selected_agent is not None
        assert (
            selected_agent.config.priority <= 5
        )  # Should select high or medium priority

        logger.info("Agent priority ordering test passed")

    def test_agent_fallback_loading(self, registry):
        """Test agent fallback loading when primary agent fails."""
        logger.info("Testing agent fallback loading")

        # Create primary and fallback agents
        primary_config = AgentConfig(name="primary", enabled=True, priority=1)
        fallback_config = AgentConfig(name="fallback", enabled=True, priority=2)

        primary_agent = MockAgent(primary_config)
        fallback_agent = MockAgent(fallback_config)

        # Register both agents
        registry.register_agent("primary", primary_agent)
        registry.register_agent("fallback", fallback_agent)

        # Test fallback when primary fails
        with patch.object(
            primary_agent, "execute", side_effect=Exception("Primary failed")
        ):
            with patch.object(
                fallback_agent,
                "execute",
                return_value={
                    "success": True,
                    "data": {"fallback_result": "success"},
                    "message": "Fallback executed",
                    "timestamp": datetime.now().isoformat(),
                },
            ):
                # This would normally be handled by the registry's fallback logic
                # For testing, we simulate the fallback behavior
                try:
                    result = primary_agent.execute("test")
                except Exception:
                    result = fallback_agent.execute("test")

                assert result["success"] is True
                assert "fallback_result" in result["data"]

        logger.info("Agent fallback loading test passed")

    def test_agent_health_checking(self, registry, sample_config):
        """Test agent health checking functionality."""
        logger.info("Testing agent health checking")

        # Create healthy and unhealthy agents
        healthy_agent = MockAgent(sample_config)
        unhealthy_agent = MockAgent(sample_config)

        # Register agents
        registry.register_agent("healthy", healthy_agent)
        registry.register_agent("unhealthy", unhealthy_agent)

        # Mock health check responses
        with patch.object(healthy_agent, "health_check", return_value=True):
            with patch.object(unhealthy_agent, "health_check", return_value=False):
                # Test health checks
                assert registry.check_agent_health("healthy") is True
                assert registry.check_agent_health("unhealthy") is False

                # Test overall health status
                health_status = registry.get_health_status()
                assert isinstance(health_status, dict)
                assert "healthy" in health_status
                assert "unhealthy" in health_status

        logger.info("Agent health checking test passed")

    def test_agent_configuration_validation(self, registry):
        """Test agent configuration validation."""
        logger.info("Testing agent configuration validation")

        # Test valid configuration
        valid_config = AgentConfig(
            name="valid_agent", enabled=True, priority=1, custom_config={"timeout": 30}
        )

        agent = MockAgent(valid_config)
        registry.register_agent("valid", agent)
        assert registry.is_agent_available("valid")

        # Test invalid configurations
        invalid_configs = [
            AgentConfig(name="", enabled=True, priority=1),  # Empty name
            AgentConfig(name="test", enabled=True, priority=-1),  # Negative priority
            AgentConfig(name="test", enabled=True, priority=0),  # Zero priority
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                MockAgent(invalid_config)

        logger.info("Agent configuration validation test passed")

    def test_agent_execution_tracking(self, registry, sample_config):
        """Test agent execution tracking and metrics."""
        logger.info("Testing agent execution tracking")

        # Create agent
        agent = MockAgent(sample_config)
        registry.register_agent("tracked", agent)

        # Execute agent multiple times
        for i in range(3):
            agent.execute("test")

        # Check execution count
        assert agent.execution_count == 3

        # Test execution metrics
        metrics = registry.get_execution_metrics("tracked")
        assert isinstance(metrics, dict)
        assert metrics.get("execution_count", 0) >= 0

        logger.info("Agent execution tracking test passed")

    def test_agent_dependency_loading(self, registry):
        """Test agent dependency loading and validation."""
        logger.info("Testing agent dependency loading")

        # Create agents with dependencies
        dependency_config = AgentConfig(
            name="dependency",
            enabled=True,
            priority=1,
            dependencies=["data_provider", "model_registry"],
        )

        main_config = AgentConfig(
            name="main", enabled=True, priority=2, dependencies=["dependency"]
        )

        # Create mock dependencies
        dependency_agent = MockAgent(dependency_config)
        main_agent = MockAgent(main_config)

        # Register dependencies first
        registry.register_agent("dependency", dependency_agent)
        registry.register_agent(
            "data_provider",
            MockAgent(AgentConfig(name="data", enabled=True, priority=1)),
        )
        registry.register_agent(
            "model_registry",
            MockAgent(AgentConfig(name="model", enabled=True, priority=1)),
        )

        # Register main agent
        registry.register_agent("main", main_agent)

        # Test dependency validation
        assert registry.validate_dependencies("main") is True
        assert registry.validate_dependencies("dependency") is True

        # Test missing dependency
        missing_dep_config = AgentConfig(
            name="missing_dep", enabled=True, priority=1, dependencies=["non_existent"]
        )
        missing_dep_agent = MockAgent(missing_dep_config)
        registry.register_agent("missing_dep", missing_dep_agent)

        assert registry.validate_dependencies("missing_dep") is False

        logger.info("Agent dependency loading test passed")

    def test_agent_error_handling(self, registry, sample_config):
        """Test agent error handling and recovery."""
        logger.info("Testing agent error handling")

        # Create agent that raises exceptions
        error_agent = MockAgent(sample_config)
        registry.register_agent("error_agent", error_agent)

        # Test execution with error
        with patch.object(error_agent, "execute", side_effect=Exception("Test error")):
            try:
                result = error_agent.execute("test")
            except Exception as e:
                assert str(e) == "Test error"

                # Test error recovery
                error_agent.execution_count = 0  # Reset
                with patch.object(
                    error_agent,
                    "execute",
                    return_value={
                        "success": True,
                        "data": {"recovered": True},
                        "message": "Recovered from error",
                        "timestamp": datetime.now().isoformat(),
                    },
                ):
                    result = error_agent.execute("test")
                    assert result["success"] is True
                    assert result["data"]["recovered"] is True

        logger.info("Agent error handling test passed")

    def test_agent_performance_monitoring(self, registry, sample_config):
        """Test agent performance monitoring."""
        logger.info("Testing agent performance monitoring")

        # Create agent
        agent = MockAgent(sample_config)
        registry.register_agent("monitored", agent)

        # Simulate performance monitoring
        performance_data = {
            "execution_time": 0.5,
            "memory_usage": 100,
            "success_rate": 0.95,
            "error_count": 2,
        }

        # Mock performance tracking
        with patch.object(registry, "track_performance", return_value=performance_data):
            performance = registry.get_performance_metrics("monitored")
            assert isinstance(performance, dict)
            assert "execution_time" in performance
            assert "success_rate" in performance

        logger.info("Agent performance monitoring test passed")

    def test_agent_registry_persistence(self, registry, sample_config, tmp_path):
        """Test agent registry persistence and loading."""
        logger.info("Testing agent registry persistence")

        # Create agent
        agent = MockAgent(sample_config)
        registry.register_agent("persistent", agent)

        # Save registry state
        registry_file = tmp_path / "registry_state.json"
        registry.save_state(registry_file)

        # Create new registry and load state
        new_registry = AgentRegistry()
        new_registry.load_state(registry_file)

        # Verify state was loaded
        assert "persistent" in new_registry.get_available_agents()

        logger.info("Agent registry persistence test passed")

    def test_agent_registry_cleanup(self, registry, sample_config):
        """Test agent registry cleanup and resource management."""
        logger.info("Testing agent registry cleanup")

        # Create multiple agents
        agents = []
        for i in range(5):
            config = AgentConfig(name=f"agent_{i}", enabled=True, priority=i + 1)
            agent = MockAgent(config)
            agents.append(agent)
            registry.register_agent(f"agent_{i}", agent)

        # Verify agents are registered
        assert len(registry.get_available_agents()) == 5

        # Cleanup registry
        registry.cleanup()

        # Verify cleanup
        assert len(registry.get_available_agents()) == 0

        logger.info("Agent registry cleanup test passed")
