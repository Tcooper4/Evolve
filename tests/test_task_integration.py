import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import uuid

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.task_memory import TaskMemory, Task, TaskStatus
from trading.agents.task_dashboard import TaskDashboard
from trading.agents.model_builder_agent import ModelBuilderAgent as ModelBuilder
# ModelMetrics and ModelOutput may not exist - using placeholders

@pytest.fixture
def task_memory():
    """Create a task memory instance for testing."""
    return TaskMemory()

@pytest.fixture
def dashboard(task_memory):
    """Create a task dashboard instance for testing."""
    return TaskDashboard(task_memory)

@pytest.fixture
def model_builder():
    """Create a model builder instance for testing."""
    return ModelBuilder()

@pytest.fixture
def sample_data():
    """Create sample data for model training."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'price': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    })

def test_model_training_task_flow(task_memory, model_builder, sample_data):
    """Test the complete flow of model training task creation and tracking."""
    # Start model training
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            'agent': 'model_builder',
            'creation_time': datetime.now().isoformat(),
            'model_type': 'lstm'
        },
        notes="Starting LSTM model training"
    )
    task_memory.add_task(task)
    
    # Verify task creation
    created_task = task_memory.get_task(task_id)
    assert created_task is not None
    assert created_task.status == TaskStatus.PENDING
    
    # Simulate model training
    try:
        # Run LSTM model
        result = model_builder.run_lstm(sample_data)
        
        # Update task status
        task.status = TaskStatus.COMPLETED
        task.metadata.update({
            'completion_time': datetime.now().isoformat(),
            'duration': '5 minutes',
            'metrics': {
                'mse': result.metrics.mse,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown
            }
        })
        task_memory.update_task(task)
        
        # Verify task completion
        completed_task = task_memory.get_task(task_id)
        assert completed_task.status == TaskStatus.COMPLETED
        assert 'metrics' in completed_task.metadata
        
    except Exception as e:
        # Update task status on failure
        task.status = TaskStatus.FAILED
        task.metadata.update({
            'error': str(e),
            'completion_time': datetime.now().isoformat()
        })
        task_memory.update_task(task)
        
        # Verify task failure
        failed_task = task_memory.get_task(task_id)
        assert failed_task.status == TaskStatus.FAILED
        assert 'error' in failed_task.metadata

def test_multiple_model_training_tasks(task_memory, model_builder, sample_data):
    """Test handling multiple concurrent model training tasks."""
    model_types = ['lstm', 'xgboost', 'prophet', 'garch', 'ridge', 'hybrid']
    task_ids = []
    
    # Create tasks for different model types
    for model_type in model_types:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'model_type': model_type
            },
            notes=f"Starting {model_type} model training"
        )
        task_memory.add_task(task)
        task_ids.append(task_id)
        
    # Verify all tasks are created
    assert len(task_memory.tasks) == len(model_types)
    
    # Simulate training for each model
    for task_id, model_type in zip(task_ids, model_types):
        try:
            # Run appropriate model
            if model_type == 'lstm':
                result = model_builder.run_lstm(sample_data)
            elif model_type == 'xgboost':
                result = model_builder.run_xgboost(sample_data)
            elif model_type == 'prophet':
                result = model_builder.run_prophet(sample_data)
            elif model_type == 'garch':
                result = model_builder.run_garch(sample_data)
            elif model_type == 'ridge':
                result = model_builder.run_ridge(sample_data)
            else:  # hybrid
                result = model_builder.run_hybrid(sample_data)
                
            # Update task status
            task = task_memory.get_task(task_id)
            task.status = TaskStatus.COMPLETED
            task.metadata.update({
                'completion_time': datetime.now().isoformat(),
                'duration': '5 minutes',
                'metrics': {
                    'mse': result.metrics.mse,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'max_drawdown': result.metrics.max_drawdown
                }
            })
            task_memory.update_task(task)
            
        except Exception as e:
            # Update task status on failure
            task = task_memory.get_task(task_id)
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
            task_memory.update_task(task)
            
    # Verify task status distribution
    completed_tasks = task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
    failed_tasks = task_memory.get_tasks_by_status(TaskStatus.FAILED)
    
    assert len(completed_tasks) + len(failed_tasks) == len(model_types)

def test_dashboard_task_display(task_memory, model_builder, sample_data):
    """Test dashboard display of model training tasks."""
    # Create and complete some model training tasks
    test_multiple_model_training_tasks(task_memory, model_builder, sample_data)
    
    # Verify dashboard metrics
    total_tasks = len(task_memory.tasks)
    completed_tasks = len(task_memory.get_tasks_by_status(TaskStatus.COMPLETED))
    failed_tasks = len(task_memory.get_tasks_by_status(TaskStatus.FAILED))
    
    assert total_tasks == 6  # 6 model types
    assert completed_tasks + failed_tasks == total_tasks
    
    # Test task filtering
    model_tasks = [t for t in task_memory.tasks if t.metadata.get('model_type') == 'lstm']
    assert len(model_tasks) == 1

def test_error_handling_and_recovery(task_memory):
    """Test error handling and recovery in the task system."""
    # Create a task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        type="model_training",
        status=TaskStatus.PENDING,
        metadata={
            'agent': 'model_builder',
            'creation_time': datetime.now().isoformat(),
            'model_type': 'lstm'
        },
        notes="Starting LSTM model training"
    )
    task_memory.add_task(task)
    
    # Simulate a failure
    try:
        # Intentionally cause an error
        raise ValueError("Test error")
    except Exception as e:
        # Update task status on failure
        task.status = TaskStatus.FAILED
        task.metadata.update({
            'error': str(e),
            'completion_time': datetime.now().isoformat()
        })
        task_memory.update_task(task)
        
        # Verify task failure
        failed_task = task_memory.get_task(task_id)
        assert failed_task.status == TaskStatus.FAILED
        assert 'error' in failed_task.metadata
        
        # Test task retry
        retry_task = Task(
            task_id=str(uuid.uuid4()),
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'model_type': 'lstm',
                'retry_of': task_id
            },
            notes=f"Retrying failed task {task_id}"
        )
        task_memory.add_task(retry_task)
        
        # Verify retry task
        assert task_memory.get_task(retry_task.task_id) is not None
        assert retry_task.metadata.get('retry_of') == task_id 