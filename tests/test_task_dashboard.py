import unittest
from unittest.mock import Mock, patch
import pandas as pd
import streamlit as st
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.agents.task_memory import TaskMemory, Task, TaskStatus
from trading.agents.task_dashboard import TaskDashboard

class TestTaskDashboard(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.task_memory = TaskMemory()
        self.dashboard = TaskDashboard(self.task_memory)
        
        # Create sample tasks for testing
        self.sample_tasks = [
            Task(
                task_id="test_task_1",
                type="model_training",
                status=TaskStatus.COMPLETED,
                metadata={
                    'agent': 'model_builder',
                    'creation_time': '2024-01-01T10:00:00',
                    'completion_time': '2024-01-01T10:05:00',
                    'duration': '5 minutes',
                    'metrics': {
                        'mse': 0.0012,
                        'sharpe_ratio': 1.5,
                        'max_drawdown': 0.15
                    }
                },
                notes="Test task 1 completed successfully"
            ),
            Task(
                task_id="test_task_2",
                type="forecast",
                status=TaskStatus.FAILED,
                metadata={
                    'agent': 'forecast_agent',
                    'creation_time': '2024-01-01T11:00:00',
                    'error': 'Invalid input data'
                },
                notes="Test task 2 failed due to invalid data"
            ),
            Task(
                task_id="test_task_3",
                type="strategy",
                status=TaskStatus.PENDING,
                metadata={
                    'agent': 'strategy_agent',
                    'creation_time': '2024-01-01T12:00:00'
                },
                notes="Test task 3 is pending"
            )
        ]
        
        # Add sample tasks to task memory
        for task in self.sample_tasks:
            self.task_memory.add_task(task)
            
    def test_task_metrics_calculation(self):
        """Test the calculation of task metrics."""
        # Get metrics
        total_tasks = len(self.task_memory.tasks)
        completed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED))
        failed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.FAILED))
        pending_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.PENDING))
        
        # Verify metrics
        self.assertEqual(total_tasks, 3)
        self.assertEqual(completed_tasks, 1)
        self.assertEqual(failed_tasks, 1)
        self.assertEqual(pending_tasks, 1)
        
    def test_task_filtering(self):
        """Test task filtering functionality."""
        # Test status filter
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        self.assertEqual(len(completed_tasks), 1)
        self.assertEqual(completed_tasks[0].task_id, "test_task_1")
        
        # Test search filter
        model_tasks = [t for t in self.task_memory.tasks if "model" in t.type.lower()]
        self.assertEqual(len(model_tasks), 1)
        self.assertEqual(model_tasks[0].task_id, "test_task_1")
        
    def test_task_details(self):
        """Test task details retrieval and display."""
        task = self.task_memory.get_task("test_task_1")
        
        # Verify task details
        self.assertIsNotNone(task)
        self.assertEqual(task.type, "model_training")
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.metadata['agent'], "model_builder")
        self.assertIn('metrics', task.metadata)
        self.assertEqual(task.notes, "Test task 1 completed successfully")
        
    def test_status_distribution(self):
        """Test task status distribution calculation."""
        status_counts = {
            status: len(self.task_memory.get_tasks_by_status(status))
            for status in TaskStatus
        }
        
        # Verify status distribution
        self.assertEqual(status_counts[TaskStatus.COMPLETED], 1)
        self.assertEqual(status_counts[TaskStatus.FAILED], 1)
        self.assertEqual(status_counts[TaskStatus.PENDING], 1)
        
    def test_timeline_data(self):
        """Test timeline data generation."""
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        timeline_data = []
        
        for task in completed_tasks:
            if 'completion_time' in task.metadata:
                timeline_data.append({
                    'task_id': task.task_id,
                    'completion_time': task.metadata['completion_time'],
                    'type': task.type
                })
                
        # Verify timeline data
        self.assertEqual(len(timeline_data), 1)
        self.assertEqual(timeline_data[0]['task_id'], "test_task_1")
        self.assertEqual(timeline_data[0]['type'], "model_training")
        
    @patch('streamlit.write')
    def test_dashboard_initialization(self, mock_write):
        """Test dashboard initialization and basic display."""
        # Verify dashboard setup
        self.assertIsNotNone(self.dashboard)
        self.assertIsNotNone(self.dashboard.task_memory)
        
        # Test page setup
        self.dashboard.setup_page()
        mock_write.assert_called()
        
    def test_error_handling(self):
        """Test error handling for invalid task IDs."""
        # Test non-existent task
        task = self.task_memory.get_task("non_existent_task")
        self.assertIsNone(task)
        
        # Test invalid task status
        with self.assertRaises(ValueError):
            self.task_memory.get_tasks_by_status("INVALID_STATUS")

if __name__ == '__main__':
    unittest.main() 