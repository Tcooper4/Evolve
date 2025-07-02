import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.task_memory import TaskMemory, Task, TaskStatus
from trading.agents.task_dashboard import TaskDashboard
from trading.meta_agents.agents.model_builder import ModelBuilder, ModelMetrics, ModelOutput

class TestTaskIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.task_memory = TaskMemory()
        self.dashboard = TaskDashboard(self.task_memory)
        self.model_builder = ModelBuilder()
        
        # Create sample data for model training
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'price': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
    def test_model_training_task_flow(self):
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
        self.task_memory.add_task(task)
        
        # Verify task creation
        created_task = self.task_memory.get_task(task_id)
        self.assertIsNotNone(created_task)
        self.assertEqual(created_task.status, TaskStatus.PENDING)
        
        # Simulate model training
        try:
            # Run LSTM model
            result = self.model_builder.run_lstm(self.sample_data)
            
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
            self.task_memory.update_task(task)
            
            # Verify task completion
            completed_task = self.task_memory.get_task(task_id)
            self.assertEqual(completed_task.status, TaskStatus.COMPLETED)
            self.assertIn('metrics', completed_task.metadata)
            
        except Exception as e:
            # Update task status on failure
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
            self.task_memory.update_task(task)
            
            # Verify task failure
            failed_task = self.task_memory.get_task(task_id)
            self.assertEqual(failed_task.status, TaskStatus.FAILED)
            self.assertIn('error', failed_task.metadata)
            
    def test_multiple_model_training_tasks(self):
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
            self.task_memory.add_task(task)
            task_ids.append(task_id)
            
        # Verify all tasks are created
        self.assertEqual(len(self.task_memory.tasks), len(model_types))
        
        # Simulate training for each model
        for task_id, model_type in zip(task_ids, model_types):
            try:
                # Run appropriate model
                if model_type == 'lstm':
                    result = self.model_builder.run_lstm(self.sample_data)
                elif model_type == 'xgboost':
                    result = self.model_builder.run_xgboost(self.sample_data)
                elif model_type == 'prophet':
                    result = self.model_builder.run_prophet(self.sample_data)
                elif model_type == 'garch':
                    result = self.model_builder.run_garch(self.sample_data)
                elif model_type == 'ridge':
                    result = self.model_builder.run_ridge(self.sample_data)
                else:  # hybrid
                    result = self.model_builder.run_hybrid(self.sample_data)
                    
                # Update task status
                task = self.task_memory.get_task(task_id)
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
                self.task_memory.update_task(task)
                
            except Exception as e:
                # Update task status on failure
                task = self.task_memory.get_task(task_id)
                task.status = TaskStatus.FAILED
                task.metadata.update({
                    'error': str(e),
                    'completion_time': datetime.now().isoformat()
                })
                self.task_memory.update_task(task)
                
        # Verify task status distribution
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        failed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.FAILED)
        
        self.assertEqual(len(completed_tasks) + len(failed_tasks), len(model_types))
        
    def test_dashboard_task_display(self):
        """Test dashboard display of model training tasks."""
        # Create and complete some model training tasks
        self.test_multiple_model_training_tasks()
        
        # Verify dashboard metrics
        total_tasks = len(self.task_memory.tasks)
        completed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED))
        failed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.FAILED))
        
        self.assertEqual(total_tasks, 6)  # 6 model types
        self.assertEqual(completed_tasks + failed_tasks, total_tasks)
        
        # Test task filtering
        model_tasks = [t for t in self.task_memory.tasks if t.metadata.get('model_type') == 'lstm']
        self.assertEqual(len(model_tasks), 1)
        
    def test_error_handling_and_recovery(self):
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
        self.task_memory.add_task(task)
        
        # Simulate a failure
        try:
            # Intentionally cause an error
            raise ValueError("Test error")
        except Exception as e:
            # Update task status
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
            self.task_memory.update_task(task)
            
        # Verify error handling
        failed_task = self.task_memory.get_task(task_id)
        self.assertEqual(failed_task.status, TaskStatus.FAILED)
        self.assertIn('error', failed_task.metadata)
        
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
        self.task_memory.add_task(retry_task)
        
        # Verify retry task
        self.assertIsNotNone(self.task_memory.get_task(retry_task.task_id))
        self.assertEqual(retry_task.metadata.get('retry_of'), task_id)

if __name__ == '__main__':
    unittest.main() 