import pytest
from datetime import datetime
from typing import Dict, Any

from .base_test import BaseTest

class TestOrchestrator(BaseTest):
    """Test suite for the orchestrator component."""
    
    @pytest.mark.asyncio
    async def test_task_scheduling(self):
        """Test task scheduling functionality."""
        # Create test task
        task = {
            "type": "feature_implementation",
            "description": "Test feature implementation",
            "priority": 1
        }
        
        # Schedule task
        task_id = await self.orchestrator.schedule_task(task)
        
        # Assert task was scheduled
        self._assert_task_status(task_id, "scheduled")
        
        # Verify task in Redis
        task_data = self.redis_client.get(f"task:{task_id}")
        assert task_data is not None
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test task execution functionality."""
        # Create and schedule test task
        task = {
            "type": "model_training",
            "description": "Test model training",
            "priority": 2
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Execute task
        await self.orchestrator.coordinate_agents(task_id)
        
        # Assert task was completed
        self._assert_task_status(task_id, "completed")
        
        # Verify task progress
        status = self.orchestrator.get_task_status(task_id)
        assert status["progress"] == 100
        assert len(status["steps_completed"]) == 4  # All steps completed
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling functionality."""
        # Create task that will fail
        task = {
            "type": "invalid_type",
            "description": "Test error handling",
            "priority": 1
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Attempt to execute task
        with pytest.raises(ValueError):
            await self.orchestrator.coordinate_agents(task_id)
        
        # Assert task failed
        self._assert_task_status(task_id, "failed")
        
        # Verify error was recorded
        status = self.orchestrator.get_task_status(task_id)
        assert len(status["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_tasks(self):
        """Test handling of concurrent tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = {
                "type": "feature_implementation",
                "description": f"Test concurrent task {i}",
                "priority": 1
            }
            task_id = await self.orchestrator.schedule_task(task)
            tasks.append(task_id)
        
        # Execute tasks concurrently
        await asyncio.gather(*[
            self.orchestrator.coordinate_agents(task_id)
            for task_id in tasks
        ])
        
        # Verify all tasks completed
        for task_id in tasks:
            self._assert_task_status(task_id, "completed")
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        # Get initial health
        initial_health = self.orchestrator.get_system_health()
        
        # Schedule some tasks
        for i in range(5):
            task = {
                "type": "feature_implementation",
                "description": f"Test health monitoring {i}",
                "priority": 1
            }
            self.orchestrator.schedule_task(task)
        
        # Get updated health
        updated_health = self.orchestrator.get_system_health()
        
        # Verify metrics were updated
        assert updated_health["queued_tasks"] > initial_health["queued_tasks"]
    
    @pytest.mark.asyncio
    async def test_websocket_updates(self):
        """Test WebSocket real-time updates."""
        # Create test task
        task = {
            "type": "model_training",
            "description": "Test WebSocket updates",
            "priority": 1
        }
        task_id = await self.orchestrator.schedule_task(task)
        
        # Connect to WebSocket
        with self.test_client.websocket_connect("/ws") as websocket:
            # Start task execution
            execution_task = asyncio.create_task(
                self.orchestrator.coordinate_agents(task_id)
            )
            
            # Receive updates
            updates = []
            for _ in range(3):  # Expect at least 3 updates
                data = websocket.receive_json()
                updates.append(data)
            
            # Verify updates
            assert any(update["task_id"] == task_id for update in updates)
            assert any(update["status"] == "completed" for update in updates)
            
            # Wait for execution to complete
            await execution_task
    
    def test_task_dependencies(self):
        """Test task dependency handling."""
        # Create dependent tasks
        task1 = {
            "type": "data_processing",
            "description": "Process data",
            "priority": 1
        }
        task2 = {
            "type": "model_training",
            "description": "Train model",
            "priority": 2,
            "dependencies": ["task1"]  # Depends on task1
        }
        
        # Schedule tasks
        task1_id = self.orchestrator.schedule_task(task1)
        task2_id = self.orchestrator.schedule_task(task2)
        
        # Verify dependency was recorded
        task2_data = self.redis_client.get(f"task:{task2_id}")
        assert "task1" in task2_data["dependencies"]
    
    def test_task_prioritization(self):
        """Test task prioritization."""
        # Create tasks with different priorities
        tasks = []
        for priority in [3, 1, 2]:
            task = {
                "type": "feature_implementation",
                "description": f"Test priority {priority}",
                "priority": priority
            }
            task_id = self.orchestrator.schedule_task(task)
            tasks.append(task_id)
        
        # Get all tasks
        all_tasks = self.orchestrator.get_all_tasks()
        
        # Verify tasks are ordered by priority
        priorities = [task["priority"] for task in all_tasks]
        assert priorities == sorted(priorities)
    
    def test_resource_management(self):
        """Test resource management."""
        # Get initial resource usage
        initial_health = self.orchestrator.get_system_health()
        
        # Create resource-intensive task
        task = {
            "type": "model_training",
            "description": "Test resource management",
            "priority": 1,
            "resource_requirements": {
                "cpu": 80,
                "memory": 1024,
                "gpu": 1
            }
        }
        task_id = self.orchestrator.schedule_task(task)
        
        # Execute task
        self.orchestrator.coordinate_agents(task_id)
        
        # Get updated resource usage
        updated_health = self.orchestrator.get_system_health()
        
        # Verify resource usage was monitored
        assert "cpu_usage" in updated_health
        assert "memory_usage" in updated_health
        assert "gpu_usage" in updated_health 