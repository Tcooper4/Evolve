import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import uuid
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.task_memory import TaskMemory, Task, TaskStatus
from trading.agents.task_dashboard import TaskDashboard
from trading.meta_agents.agents.model_builder import ModelBuilder
from core.agents.router import RouterAgent as AgentRouter
from core.agents.self_improving_agent import SelfImprovingAgent

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.task_memory = TaskMemory()
        self.dashboard = TaskDashboard(self.task_memory)
        self.model_builder = ModelBuilder()
        self.agent_router = AgentRouter()
        self.self_improving_agent = SelfImprovingAgent()
        
    def test_empty_data_handling(self):
        """Test system behavior with empty or invalid data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        # Test model training with empty data
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
            notes="Testing empty data handling"
        )
        self.task_memory.add_task(task)
        
        try:
            self.model_builder.run_lstm(empty_df)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
            self.task_memory.update_task(task)
            
        # Verify task failure and error handling
        failed_task = self.task_memory.get_task(task_id)
        self.assertEqual(failed_task.status, TaskStatus.FAILED)
        self.assertIn('error', failed_task.metadata)
        
    def test_concurrent_task_handling(self):
        """Test system behavior with concurrent tasks."""
        # Create multiple tasks simultaneously
        task_ids = []
        for i in range(10):
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
                notes=f"Concurrent task {i}"
            )
            self.task_memory.add_task(task)
            task_ids.append(task_id)
            
        # Verify all tasks are created
        self.assertEqual(len(self.task_memory.tasks), 10)
        
        # Update tasks concurrently
        for task_id in task_ids:
            task = self.task_memory.get_task(task_id)
            task.status = TaskStatus.COMPLETED
            task.metadata.update({
                'completion_time': datetime.now().isoformat(),
                'duration': '1 minute'
            })
            self.task_memory.update_task(task)
            
        # Verify all tasks are updated
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        self.assertEqual(len(completed_tasks), 10)
        
    def test_large_data_handling(self):
        """Test system behavior with large datasets."""
        # Create large dataset
        large_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10000, freq='H'),
            'price': np.random.normal(100, 10, 10000),
            'volume': np.random.normal(1000, 100, 10000)
        })
        
        # Test task creation and tracking with large data
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'model_type': 'lstm',
                'data_size': len(large_df)
            },
            notes="Testing large data handling"
        )
        self.task_memory.add_task(task)
        
        try:
            # Simulate processing large data
            task.status = TaskStatus.COMPLETED
            task.metadata.update({
                'completion_time': datetime.now().isoformat(),
                'duration': '10 minutes',
                'memory_usage': '1GB'
            })
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
            
        self.task_memory.update_task(task)
        
        # Verify task completion
        completed_task = self.task_memory.get_task(task_id)
        self.assertEqual(completed_task.status, TaskStatus.COMPLETED)
        self.assertIn('memory_usage', completed_task.metadata)
        
    def test_task_retry_mechanism(self):
        """Test task retry mechanism after failures."""
        # Create initial task
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
            notes="Initial task"
        )
        self.task_memory.add_task(task)
        
        # Simulate task failure
        task.status = TaskStatus.FAILED
        task.metadata.update({
            'error': 'Test error',
            'completion_time': datetime.now().isoformat()
        })
        self.task_memory.update_task(task)
        
        # Create retry task
        retry_task_id = str(uuid.uuid4())
        retry_task = Task(
            task_id=retry_task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'model_type': 'lstm',
                'retry_of': task_id,
                'retry_count': 1
            },
            notes=f"Retry of task {task_id}"
        )
        self.task_memory.add_task(retry_task)
        
        # Verify retry task
        retry_task = self.task_memory.get_task(retry_task_id)
        self.assertEqual(retry_task.metadata['retry_of'], task_id)
        self.assertEqual(retry_task.metadata['retry_count'], 1)
        
    def test_invalid_task_states(self):
        """Test handling of invalid task states and transitions."""
        # Create task
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat()
            },
            notes="Testing invalid states"
        )
        self.task_memory.add_task(task)
        
        # Test invalid state transition
        task.status = TaskStatus.COMPLETED
        self.task_memory.update_task(task)
        
        # Try to transition back to pending (invalid)
        task.status = TaskStatus.PENDING
        self.task_memory.update_task(task)
        
        # Verify task remains in completed state
        updated_task = self.task_memory.get_task(task_id)
        self.assertEqual(updated_task.status, TaskStatus.COMPLETED)
        
    def test_corrupted_task_data(self):
        """Test handling of corrupted task data."""
        # Create task with corrupted metadata
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'corrupted_field': object()  # Non-serializable object
            },
            notes="Testing corrupted data"
        )
        
        # Verify task creation handles corrupted data
        try:
            self.task_memory.add_task(task)
        except Exception as e:
            self.assertIsInstance(e, (TypeError, json.JSONDecodeError))
            
    def test_task_timeout_handling(self):
        """Test handling of task timeouts."""
        # Create long-running task
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat(),
                'timeout': 300  # 5 minutes
            },
            notes="Testing timeout handling"
        )
        self.task_memory.add_task(task)
        
        # Simulate timeout
        task.status = TaskStatus.FAILED
        task.metadata.update({
            'error': 'Task timeout',
            'completion_time': datetime.now().isoformat(),
            'duration': '300 seconds'
        })
        self.task_memory.update_task(task)
        
        # Verify timeout handling
        failed_task = self.task_memory.get_task(task_id)
        self.assertEqual(failed_task.status, TaskStatus.FAILED)
        self.assertIn('timeout', failed_task.metadata.get('error', '').lower())
        
    def test_system_resource_limits(self):
        """Test system behavior under resource constraints."""
        # Create multiple resource-intensive tasks
        task_ids = []
        for i in range(5):
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat(),
                    'resource_requirements': {
                        'memory': '2GB',
                        'cpu': '100%',
                        'gpu': '1'
                    }
                },
                notes=f"Resource-intensive task {i}"
            )
            self.task_memory.add_task(task)
            task_ids.append(task_id)
            
        # Simulate resource constraints
        for task_id in task_ids:
            task = self.task_memory.get_task(task_id)
            task.status = TaskStatus.FAILED
            task.metadata.update({
                'error': 'Insufficient resources',
                'completion_time': datetime.now().isoformat(),
                'resource_status': {
                    'memory_available': '1GB',
                    'cpu_available': '50%',
                    'gpu_available': '0'
                }
            })
            self.task_memory.update_task(task)
            
        # Verify resource constraint handling
        failed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.FAILED)
        self.assertEqual(len(failed_tasks), 5)
        for task in failed_tasks:
            self.assertIn('resource_status', task.metadata)

if __name__ == '__main__':
    unittest.main() 