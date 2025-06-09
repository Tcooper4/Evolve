import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path

from ..core.orchestrator import Task
from ..config.config import load_config

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: str, config_path: str = "automation/config/config.json"):
        self.agent_id = agent_id
        self.config = load_config(config_path)
        self.setup_logging()
        self.tasks: Dict[str, Task] = {}
        self.running = False
        self.last_heartbeat = datetime.now()
        
    def setup_logging(self):
        """Setup logging for the agent."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'automation/logs/agent_{self.agent_id}.log'),
                logging.StreamHandler()
            ]
        )
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent's resources and connections."""
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        pass
        
    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a single task."""
        pass
        
    async def update_heartbeat(self) -> None:
        """Update the agent's heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
        
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "status": "running" if self.running else "stopped",
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "active_tasks": len(self.tasks),
            "task_types": list(set(task.type for task in self.tasks.values()))
        }
        
    async def start(self) -> None:
        """Start the agent."""
        try:
            self.running = True
            await self.initialize()
            logger.info(f"Agent {self.agent_id} started")
            
            while self.running:
                await self.update_heartbeat()
                await self.process_tasks()
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed: {str(e)}")
            raise
        finally:
            self.running = False
            await self.cleanup()
            
    async def stop(self) -> None:
        """Stop the agent."""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
        
    async def process_tasks(self) -> None:
        """Process all active tasks."""
        for task_id, task in list(self.tasks.items()):
            try:
                result = await self.process_task(task)
                logger.info(f"Task {task_id} processed successfully: {result}")
                del self.tasks[task_id]
            except Exception as e:
                logger.error(f"Failed to process task {task_id}: {str(e)}")
                # Implement retry logic or error handling as needed
                
    async def add_task(self, task: Task) -> None:
        """Add a task to the agent's queue."""
        self.tasks[task.id] = task
        logger.info(f"Task {task.id} added to agent {self.agent_id}")
        
    async def remove_task(self, task_id: str) -> None:
        """Remove a task from the agent's queue."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Task {task_id} removed from agent {self.agent_id}")
            
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
        
    async def get_tasks(self) -> List[Task]:
        """Get all tasks in the agent's queue."""
        return list(self.tasks.values())
        
    async def clear_tasks(self) -> None:
        """Clear all tasks from the agent's queue."""
        self.tasks.clear()
        logger.info(f"All tasks cleared from agent {self.agent_id}") 