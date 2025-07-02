import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import uuid
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.task_memory import TaskMemory, Task, TaskStatus
from trading.agents.task_dashboard import TaskDashboard
from trading.meta_agents.agents.model_builder import ModelBuilder
from core.agents.router import AgentRouter
from core.agents.self_improving_agent import SelfImprovingAgent

class TestPerformance(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.task_memory = TaskMemory()
        self.dashboard = TaskDashboard(self.task_memory)
        self.model_builder = ModelBuilder()
        self.agent_router = AgentRouter()
        self.self_improving_agent = SelfImprovingAgent()
        
        # Initialize performance metrics
        self.metrics = {
            'task_creation_time': [],
            'task_update_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'response_times': []
        }
        
    def _measure_performance(self, func, *args, **kwargs):
        """Measure performance metrics for a function call."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        # Record metrics
        self.metrics['response_times'].append(end_time - start_time)
        self.metrics['memory_usage'].append(end_memory - start_memory)
        self.metrics['cpu_usage'].append(end_cpu - start_cpu)
        
        return result
        
    def test_task_creation_performance(self):
        """Test performance of task creation under load."""
        num_tasks = 1000
        creation_times = []
        
        for i in range(num_tasks):
            start_time = time.time()
            
            task = Task(
                task_id=str(uuid.uuid4()),
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat(),
                    'model_type': 'lstm'
                },
                notes=f"Performance test task {i}"
            )
            self.task_memory.add_task(task)
            
            end_time = time.time()
            creation_times.append(end_time - start_time)
            
        # Calculate statistics
        avg_creation_time = sum(creation_times) / len(creation_times)
        max_creation_time = max(creation_times)
        min_creation_time = min(creation_times)
        
        # Verify performance
        self.assertLess(avg_creation_time, 0.1)  # Average creation time < 100ms
        self.assertLess(max_creation_time, 0.5)  # Max creation time < 500ms
        
    def test_concurrent_task_processing(self):
        """Test system performance under concurrent task processing."""
        num_tasks = 100
        num_threads = 10
        
        def process_task(task_id):
            task = Task(
                task_id=task_id,
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat()
                }
            )
            self.task_memory.add_task(task)
            
            # Simulate task processing
            time.sleep(0.1)
            
            task.status = TaskStatus.COMPLETED
            task.metadata.update({
                'completion_time': datetime.now().isoformat()
            })
            self.task_memory.update_task(task)
            
        # Create and process tasks concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_task, str(uuid.uuid4()))
                for _ in range(num_tasks)
            ]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()
                
        # Verify all tasks are processed
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        self.assertEqual(len(completed_tasks), num_tasks)
        
    def test_memory_usage_under_load(self):
        """Test memory usage under heavy load."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create large number of tasks with substantial metadata
        for i in range(1000):
            task = Task(
                task_id=str(uuid.uuid4()),
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat(),
                    'large_data': 'x' * 10000  # 10KB of data
                }
            )
            self.task_memory.add_task(task)
            
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify memory usage is within acceptable limits
        self.assertLess(memory_increase, 100 * 1024 * 1024)  # Less than 100MB increase
        
    def test_query_performance(self):
        """Test performance of task queries and filtering."""
        # Create tasks with various attributes
        for i in range(1000):
            task = Task(
                task_id=str(uuid.uuid4()),
                type=f"task_type_{i % 5}",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': f"agent_{i % 3}",
                    'creation_time': datetime.now().isoformat(),
                    'priority': i % 10
                }
            )
            self.task_memory.add_task(task)
            
        # Test query performance
        query_times = []
        
        # Test status-based queries
        for status in TaskStatus:
            start_time = time.time()
            tasks = self.task_memory.get_tasks_by_status(status)
            end_time = time.time()
            query_times.append(end_time - start_time)
            
        # Test metadata-based queries
        for agent in ["agent_0", "agent_1", "agent_2"]:
            start_time = time.time()
            tasks = [t for t in self.task_memory.tasks if t.metadata.get('agent') == agent]
            end_time = time.time()
            query_times.append(end_time - start_time)
            
        # Verify query performance
        avg_query_time = sum(query_times) / len(query_times)
        self.assertLess(avg_query_time, 0.1)  # Average query time < 100ms
        
    def test_dashboard_rendering_performance(self):
        """Test performance of dashboard rendering with large datasets."""
        # Create large dataset
        for i in range(1000):
            task = Task(
                task_id=str(uuid.uuid4()),
                type="model_training",
                status=TaskStatus.COMPLETED,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat(),
                    'completion_time': datetime.now().isoformat(),
                    'metrics': {
                        'mse': np.random.random(),
                        'sharpe_ratio': np.random.random(),
                        'max_drawdown': np.random.random()
                    }
                }
            )
            self.task_memory.add_task(task)
            
        # Measure dashboard rendering time
        start_time = time.time()
        self.dashboard.run()
        end_time = time.time()
        
        # Verify rendering performance
        render_time = end_time - start_time
        self.assertLess(render_time, 5.0)  # Dashboard renders in less than 5 seconds
        
    def test_system_recovery_performance(self):
        """Test system recovery performance after failures."""
        # Create and process tasks
        task_ids = []
        for i in range(100):
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat()
                }
            )
            self.task_memory.add_task(task)
            task_ids.append(task_id)
            
        # Simulate system failure
        self.task_memory = TaskMemory()  # Reset task memory
        
        # Measure recovery time
        start_time = time.time()
        
        # Recover tasks
        for task_id in task_ids:
            task = Task(
                task_id=task_id,
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat()
                }
            )
            self.task_memory.add_task(task)
            
        end_time = time.time()
        recovery_time = end_time - start_time
        
        # Verify recovery performance
        self.assertLess(recovery_time, 1.0)  # Recovery in less than 1 second
        
    def test_long_running_operation_performance(self):
        """Test performance of long-running operations."""
        # Create a long-running task
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type="model_training",
            status=TaskStatus.PENDING,
            metadata={
                'agent': 'model_builder',
                'creation_time': datetime.now().isoformat()
            }
        )
        self.task_memory.add_task(task)
        
        # Simulate long-running operation
        start_time = time.time()
        time.sleep(5)  # Simulate 5-second operation
        
        # Update task status
        task.status = TaskStatus.COMPLETED
        task.metadata.update({
            'completion_time': datetime.now().isoformat(),
            'duration': '5 seconds'
        })
        self.task_memory.update_task(task)
        
        end_time = time.time()
        operation_time = end_time - start_time
        
        # Verify operation performance
        self.assertLess(operation_time, 5.1)  # Operation completes within 5.1 seconds
        
    def test_resource_cleanup_performance(self):
        """Test performance of resource cleanup after task completion."""
        # Create and complete tasks
        for i in range(100):
            task = Task(
                task_id=str(uuid.uuid4()),
                type="model_training",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': datetime.now().isoformat(),
                    'large_data': 'x' * 1000  # 1KB of data
                }
            )
            self.task_memory.add_task(task)
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.metadata.update({
                'completion_time': datetime.now().isoformat()
            })
            self.task_memory.update_task(task)
            
        # Measure cleanup time
        start_time = time.time()
        self.task_memory.cleanup_resources()  # Assuming this method exists
        end_time = time.time()
        
        # Verify cleanup performance
        cleanup_time = end_time - start_time
        self.assertLess(cleanup_time, 1.0)  # Cleanup in less than 1 second

if __name__ == '__main__':
    unittest.main() 