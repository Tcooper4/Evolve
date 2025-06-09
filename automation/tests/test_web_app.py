import pytest
from automation.web.app import app
import json
import asyncio
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_task_api():
    with patch('automation.web.app.task_api') as mock:
        yield mock

@pytest.fixture
def mock_metrics_api():
    with patch('automation.web.app.metrics_api') as mock:
        yield mock

@pytest.fixture
def mock_orchestrator():
    with patch('automation.web.app.orchestrator') as mock:
        yield mock

def test_index_route(client):
    """Test the index route returns the tasks page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Task Management' in response.data

def test_dashboard_route(client):
    """Test the dashboard route returns the monitoring dashboard"""
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'Monitoring Dashboard' in response.data

@pytest.mark.asyncio
async def test_get_tasks(client, mock_task_api):
    """Test getting all tasks"""
    mock_tasks = [
        {
            'task_id': '1',
            'name': 'Test Task',
            'status': 'pending'
        }
    ]
    mock_task_api.get_tasks.return_value = mock_tasks
    
    response = await client.get('/api/tasks')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['task_id'] == '1'

@pytest.mark.asyncio
async def test_create_task(client, mock_task_api):
    """Test creating a new task"""
    task_data = {
        'name': 'New Task',
        'task_type': 'data_collection',
        'priority': 1
    }
    mock_task = {
        'task_id': '2',
        **task_data
    }
    mock_task_api.create_task.return_value = mock_task
    
    response = await client.post('/api/tasks',
                               data=json.dumps(task_data),
                               content_type='application/json')
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['task_id'] == '2'
    assert data['name'] == 'New Task'

@pytest.mark.asyncio
async def test_get_task(client, mock_task_api):
    """Test getting a specific task"""
    mock_task = {
        'task_id': '1',
        'name': 'Test Task',
        'status': 'pending'
    }
    mock_task_api.get_task.return_value = mock_task
    
    response = await client.get('/api/tasks/1')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['task_id'] == '1'
    assert data['name'] == 'Test Task'

@pytest.mark.asyncio
async def test_update_task(client, mock_task_api):
    """Test updating a task"""
    update_data = {'status': 'running'}
    mock_task = {
        'task_id': '1',
        'name': 'Test Task',
        'status': 'running'
    }
    mock_task_api.update_task.return_value = mock_task
    
    response = await client.patch('/api/tasks/1',
                                data=json.dumps(update_data),
                                content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'

@pytest.mark.asyncio
async def test_delete_task(client, mock_task_api):
    """Test deleting a task"""
    mock_task_api.delete_task.return_value = True
    
    response = await client.delete('/api/tasks/1')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'Task deleted successfully'

@pytest.mark.asyncio
async def test_execute_task(client, mock_task_api):
    """Test executing a task"""
    mock_task_api.execute_task.return_value = True
    
    response = await client.post('/api/tasks/1/execute')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'Task execution started'

@pytest.mark.asyncio
async def test_stop_task(client, mock_task_api):
    """Test stopping a task"""
    mock_task_api.stop_task.return_value = True
    
    response = await client.post('/api/tasks/1/stop')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'Task stopped successfully'

@pytest.mark.asyncio
async def test_get_task_metrics(client, mock_metrics_api):
    """Test getting task metrics"""
    mock_metrics = {
        'task_id': '1',
        'name': 'Test Task',
        'status': 'running',
        'progress': 50
    }
    mock_metrics_api.get_task_metrics.return_value = mock_metrics
    
    response = await client.get('/api/tasks/1/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['task_id'] == '1'
    assert data['progress'] == 50

@pytest.mark.asyncio
async def test_get_metrics(client, mock_metrics_api):
    """Test getting all metrics"""
    mock_metrics = {
        'system': {'cpu_usage': 50.0},
        'agents': [{'agent_id': '1'}],
        'models': [{'model_id': '1'}]
    }
    mock_metrics_api.get_all_metrics.return_value = mock_metrics
    
    response = await client.get('/api/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'system' in data
    assert 'agents' in data
    assert 'models' in data

@pytest.mark.asyncio
async def test_get_system_metrics(client, mock_metrics_api):
    """Test getting system metrics"""
    mock_metrics = {
        'cpu_usage': 50.0,
        'memory_usage': 60.0
    }
    mock_metrics_api.get_system_metrics.return_value = mock_metrics
    
    response = await client.get('/api/metrics/system')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['cpu_usage'] == 50.0
    assert data['memory_usage'] == 60.0

@pytest.mark.asyncio
async def test_get_agent_metrics(client, mock_metrics_api):
    """Test getting agent metrics"""
    mock_metrics = [
        {
            'agent_id': '1',
            'name': 'Test Agent',
            'status': 'active'
        }
    ]
    mock_metrics_api.get_agent_metrics.return_value = mock_metrics
    
    response = await client.get('/api/metrics/agents')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['agent_id'] == '1'

@pytest.mark.asyncio
async def test_get_model_metrics(client, mock_metrics_api):
    """Test getting model metrics"""
    mock_metrics = [
        {
            'model_id': '1',
            'name': 'Test Model',
            'type': 'lstm'
        }
    ]
    mock_metrics_api.get_model_metrics.return_value = mock_metrics
    
    response = await client.get('/api/metrics/models')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['model_id'] == '1'

@pytest.mark.asyncio
async def test_get_metrics_history(client, mock_metrics_api):
    """Test getting metrics history"""
    mock_history = [
        {'timestamp': '2024-01-01T00:00:00', 'cpu_usage': 50.0},
        {'timestamp': '2024-01-01T00:01:00', 'cpu_usage': 60.0}
    ]
    mock_metrics_api.get_metrics_history.return_value = mock_history
    
    response = await client.get('/api/metrics/history?limit=2')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    assert data[0]['cpu_usage'] == 50.0

@pytest.mark.asyncio
async def test_error_handling(client, mock_task_api):
    """Test error handling in API endpoints"""
    mock_task_api.get_task.side_effect = Exception('Test error')
    
    response = await client.get('/api/tasks/1')
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

@pytest.mark.asyncio
async def test_initialize_app(mock_orchestrator):
    """Test application initialization"""
    from automation.web.app import initialize
    
    await initialize()
    mock_orchestrator.initialize.assert_called_once()
    mock_orchestrator.start.assert_called_once()

def test_run_app(mock_orchestrator):
    """Test running the application"""
    with patch('automation.web.app.app.run') as mock_run:
        from automation.web.app import run_app
        run_app()
        mock_run.assert_called_once() 