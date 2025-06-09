import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from automation.api.metrics_api import MetricsAPI
from automation.core.orchestrator import Orchestrator

@pytest.fixture
def mock_orchestrator():
    return Mock(spec=Orchestrator)

@pytest.fixture
def metrics_api(mock_orchestrator):
    return MetricsAPI(mock_orchestrator)

@pytest.mark.asyncio
async def test_get_system_metrics(metrics_api):
    """Test getting system metrics"""
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory', return_value=Mock(percent=60.0, total=1000, available=400)), \
         patch('psutil.disk_usage', return_value=Mock(percent=70.0, total=2000, free=600)):
        
        metrics = await metrics_api.get_system_metrics()
        
        assert metrics['cpu_usage'] == 50.0
        assert metrics['memory_usage'] == 60.0
        assert metrics['memory_total'] == 1000
        assert metrics['memory_available'] == 400
        assert metrics['disk_usage'] == 70.0
        assert metrics['disk_total'] == 2000
        assert metrics['disk_free'] == 600
        assert 'timestamp' in metrics
        assert 'system_status' in metrics

@pytest.mark.asyncio
async def test_get_task_metrics(metrics_api, mock_orchestrator):
    """Test getting task metrics"""
    mock_task = Mock(
        task_id='1',
        name='Test Task',
        status='running',
        progress=50,
        start_time=datetime.now(),
        end_time=None,
        error_message=None
    )
    mock_orchestrator.get_task = AsyncMock(return_value=mock_task)
    
    metrics = await metrics_api.get_task_metrics('1')
    
    assert metrics['task_id'] == '1'
    assert metrics['name'] == 'Test Task'
    assert metrics['status'] == 'running'
    assert metrics['progress'] == 50
    assert metrics['start_time'] is not None
    assert metrics['end_time'] is None
    assert metrics['execution_time'] is None
    assert metrics['error_message'] is None

@pytest.mark.asyncio
async def test_get_agent_metrics(metrics_api, mock_orchestrator):
    """Test getting agent metrics"""
    mock_agent = Mock(
        agent_id='1',
        name='Test Agent',
        status='active',
        get_tasks=Mock(return_value=['task1', 'task2']),
        last_heartbeat=datetime.now(),
        cpu_usage=30.0,
        memory_usage=40.0
    )
    mock_orchestrator.get_agents = AsyncMock(return_value=[mock_agent])
    
    metrics = await metrics_api.get_agent_metrics()
    
    assert len(metrics) == 1
    assert metrics[0]['agent_id'] == '1'
    assert metrics[0]['name'] == 'Test Agent'
    assert metrics[0]['status'] == 'active'
    assert metrics[0]['active_tasks'] == 2
    assert metrics[0]['last_heartbeat'] is not None
    assert metrics[0]['cpu_usage'] == 30.0
    assert metrics[0]['memory_usage'] == 40.0

@pytest.mark.asyncio
async def test_get_model_metrics(metrics_api, mock_orchestrator):
    """Test getting model metrics"""
    mock_model = Mock(
        model_id='1',
        name='Test Model',
        type='lstm',
        status='trained',
        accuracy=0.95,
        last_trained=datetime.now(),
        training_time=3600
    )
    mock_orchestrator.get_models = AsyncMock(return_value=[mock_model])
    
    metrics = await metrics_api.get_model_metrics()
    
    assert len(metrics) == 1
    assert metrics[0]['model_id'] == '1'
    assert metrics[0]['name'] == 'Test Model'
    assert metrics[0]['type'] == 'lstm'
    assert metrics[0]['status'] == 'trained'
    assert metrics[0]['accuracy'] == 0.95
    assert metrics[0]['last_trained'] is not None
    assert metrics[0]['training_time'] == 3600

@pytest.mark.asyncio
async def test_get_metrics_history(metrics_api):
    """Test getting metrics history"""
    # Add some test metrics
    for i in range(5):
        await metrics_api.get_system_metrics()
    
    history = await metrics_api.get_metrics_history(limit=3)
    assert len(history) == 3

def test_get_system_status(metrics_api):
    """Test system status determination"""
    assert metrics_api._get_system_status(95.0, 80.0) == 'error'
    assert metrics_api._get_system_status(75.0, 80.0) == 'warning'
    assert metrics_api._get_system_status(50.0, 60.0) == 'active'

@pytest.mark.asyncio
async def test_get_all_metrics(metrics_api):
    """Test getting all metrics in one call"""
    with patch.object(metrics_api, 'get_system_metrics', new_callable=AsyncMock) as mock_system, \
         patch.object(metrics_api, 'get_agent_metrics', new_callable=AsyncMock) as mock_agent, \
         patch.object(metrics_api, 'get_model_metrics', new_callable=AsyncMock) as mock_model:
        
        mock_system.return_value = {'cpu_usage': 50.0}
        mock_agent.return_value = [{'agent_id': '1'}]
        mock_model.return_value = [{'model_id': '1'}]
        
        metrics = await metrics_api.get_all_metrics()
        
        assert 'system' in metrics
        assert 'agents' in metrics
        assert 'models' in metrics
        assert 'timestamp' in metrics
        assert metrics['system']['cpu_usage'] == 50.0
        assert len(metrics['agents']) == 1
        assert len(metrics['models']) == 1

@pytest.mark.asyncio
async def test_error_handling(metrics_api, mock_orchestrator):
    """Test error handling in metrics API"""
    mock_orchestrator.get_task = AsyncMock(side_effect=Exception('Test error'))
    
    with pytest.raises(Exception) as exc_info:
        await metrics_api.get_task_metrics('1')
    assert str(exc_info.value) == 'Test error' 