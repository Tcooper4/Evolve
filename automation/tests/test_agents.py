import pytest
import asyncio
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch, AsyncMock

from ..agents.base_agent import BaseAgent
from ..agents.data_collection_agent import DataCollectionAgent
from ..agents.agent_manager import AgentManager
from ..core.orchestrator import Task

# Fixtures
@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch('redis.Redis') as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "storage": {
            "path": "test_data"
        }
    }

@pytest.fixture
def mock_task():
    """Create a sample task."""
    return Task(
        id="test_task_001",
        name="Test Task",
        type="data_collection",
        parameters={
            "symbol": "AAPL",
            "source": "yfinance"
        }
    )

# Base Agent Tests
@pytest.mark.asyncio
async def test_base_agent_initialization():
    """Test base agent initialization."""
    agent = BaseAgent("test_agent")
    assert agent.agent_id == "test_agent"
    assert not agent.running
    assert len(agent.tasks) == 0

@pytest.mark.asyncio
async def test_base_agent_heartbeat():
    """Test agent heartbeat update."""
    agent = BaseAgent("test_agent")
    initial_heartbeat = agent.last_heartbeat
    await asyncio.sleep(1)
    await agent.update_heartbeat()
    assert agent.last_heartbeat > initial_heartbeat

@pytest.mark.asyncio
async def test_base_agent_status():
    """Test agent status reporting."""
    agent = BaseAgent("test_agent")
    status = await agent.get_status()
    assert status["agent_id"] == "test_agent"
    assert status["status"] == "stopped"
    assert "last_heartbeat" in status
    assert status["active_tasks"] == 0

# Data Collection Agent Tests
@pytest.mark.asyncio
async def test_data_collection_agent_initialization(mock_redis, mock_config):
    """Test data collection agent initialization."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        agent = DataCollectionAgent("test_data_agent")
        assert agent.agent_id == "test_data_agent"
        assert agent.redis_client is None
        assert agent.http_session is None

@pytest.mark.asyncio
async def test_data_collection_agent_initialize(mock_redis, mock_config):
    """Test data collection agent initialization process."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        agent = DataCollectionAgent("test_data_agent")
        await agent.initialize()
        assert agent.redis_client is not None
        assert agent.http_session is not None

@pytest.mark.asyncio
async def test_data_collection_agent_process_task(mock_redis, mock_config, mock_task):
    """Test data collection agent task processing."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        agent = DataCollectionAgent("test_data_agent")
        await agent.initialize()
        result = await agent.process_task(mock_task)
        assert result["status"] == "success"
        assert "data_path" in result

# Agent Manager Tests
@pytest.mark.asyncio
async def test_agent_manager_initialization(mock_redis, mock_config):
    """Test agent manager initialization."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        manager = AgentManager()
        assert len(manager.agents) == 0
        assert len(manager.agent_types) > 0

@pytest.mark.asyncio
async def test_agent_manager_create_agent(mock_redis, mock_config):
    """Test agent creation."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        manager = AgentManager()
        agent_id = await manager.create_agent("data_collection")
        assert agent_id in manager.agents
        assert isinstance(manager.agents[agent_id], DataCollectionAgent)

@pytest.mark.asyncio
async def test_agent_manager_start_stop_agent(mock_redis, mock_config):
    """Test agent start/stop functionality."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        manager = AgentManager()
        agent_id = await manager.create_agent("data_collection")
        await manager.start_agent(agent_id)
        assert manager.agents[agent_id].running
        await manager.stop_agent(agent_id)
        assert not manager.agents[agent_id].running

@pytest.mark.asyncio
async def test_agent_manager_assign_task(mock_redis, mock_config, mock_task):
    """Test task assignment to agents."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        manager = AgentManager()
        agent_id = await manager.create_agent("data_collection")
        await manager.start_agent(agent_id)
        await manager.assign_task(mock_task)
        assert mock_task.id in manager.agents[agent_id].tasks

@pytest.mark.asyncio
async def test_agent_manager_monitor_agents(mock_redis, mock_config):
    """Test agent monitoring functionality."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        manager = AgentManager()
        agent_id = await manager.create_agent("data_collection")
        await manager.start_agent(agent_id)
        
        # Simulate agent becoming unresponsive
        manager.agents[agent_id].last_heartbeat = datetime.now() - timedelta(minutes=6)
        await manager.monitor_agents()
        
        # Agent should be restarted
        assert manager.agents[agent_id].running

# Integration Tests
@pytest.mark.asyncio
async def test_agent_system_integration(mock_redis, mock_config, mock_task):
    """Test integration of agent manager and data collection agent."""
    with patch('automation.config.config.load_config', return_value=mock_config):
        # Initialize manager
        manager = AgentManager()
        await manager.initialize()
        
        # Create and start agent
        agent_id = await manager.create_agent("data_collection")
        await manager.start_agent(agent_id)
        
        # Assign task
        await manager.assign_task(mock_task)
        
        # Verify task processing
        agent = manager.agents[agent_id]
        assert mock_task.id in agent.tasks
        
        # Process task
        result = await agent.process_task(mock_task)
        assert result["status"] == "success"
        
        # Cleanup
        await manager.stop() 