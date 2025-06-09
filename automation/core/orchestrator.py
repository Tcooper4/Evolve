import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import aiohttp
import redis
from pydantic import BaseModel, Field
import ray
from ray import serve
import kubernetes as k8s
from kubernetes import client, config
from .task_handlers import TASK_HANDLERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation/logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

class Task(BaseModel):
    """Task model for automation jobs."""
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    type: str = Field(..., description="Task type (data_collection, model_training, etc.)")
    status: str = Field(default="pending", description="Task status")
    priority: int = Field(default=1, description="Task priority (1-5)")
    dependencies: List[str] = Field(default_factory=list, description="Dependent task IDs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = Field(default=None, description="Error message if task failed")

class Orchestrator:
    def __init__(self, config_path: str = "automation/config/config.json"):
        """Initialize the orchestrator with configuration."""
        self.config = self._load_config(config_path)
        self.redis_client = self._setup_redis()
        self.ray_client = self._setup_ray()
        self.k8s_client = self._setup_kubernetes()
        self.tasks: Dict[str, Task] = {}
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection for task queue and caching."""
        try:
            redis_config = self.config["redis"]
            return redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to setup Redis: {str(e)}")
            raise
            
    def _setup_ray(self) -> None:
        """Initialize Ray for distributed computing."""
        try:
            ray_config = self.config["ray"]
            ray.init(
                address=ray_config["address"],
                namespace=ray_config["namespace"],
                runtime_env=ray_config["runtime_env"]
            )
            serve.start(detached=True)
        except Exception as e:
            logger.error(f"Failed to setup Ray: {str(e)}")
            raise
            
    def _setup_kubernetes(self) -> None:
        """Setup Kubernetes client for container orchestration."""
        try:
            k8s_config = self.config["kubernetes"]
            if k8s_config["in_cluster"]:
                config.load_incluster_config()
            else:
                config.load_kube_config()
            return client.CoreV1Api()
        except Exception as e:
            logger.error(f"Failed to setup Kubernetes: {str(e)}")
            raise

    async def create_task(self, task: Task) -> str:
        """Create a new task and add it to the queue."""
        try:
            # Validate dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Dependency task {dep_id} not found")
                    
            # Store task
            self.tasks[task.id] = task
            await self.redis_client.hset(
                "tasks",
                task.id,
                task.json()
            )
            
            # Add to priority queue
            await self.redis_client.zadd(
                "task_queue",
                {task.id: task.priority}
            )
            
            logger.info(f"Created task {task.id}: {task.name}")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        try:
            task_data = await self.redis_client.hget("tasks", task_id)
            if task_data:
                return Task.parse_raw(task_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {str(e)}")
            raise

    async def update_task_status(self, task_id: str, status: str, error: Optional[str] = None) -> None:
        """Update task status and error message."""
        try:
            task = await self.get_task(task_id)
            if task:
                task.status = status
                task.error = error
                task.updated_at = datetime.now()
                await self.redis_client.hset(
                    "tasks",
                    task_id,
                    task.json()
                )
                logger.info(f"Updated task {task_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            raise

    async def execute_task(self, task: Task) -> None:
        """Execute a task based on its type using the handler class."""
        try:
            await self.update_task_status(task.id, "running")
            handler = self._get_task_handler(task.type)
            if not handler:
                raise ValueError(f"No handler found for task type {task.type}")
            result = await handler.handle(task)
            await self.update_task_status(task.id, "completed")
            await self._process_dependencies(task)
            logger.info(f"Executed task {task.id}: {task.name} | Result: {result}")
        except Exception as e:
            logger.error(f"Failed to execute task {task.id}: {str(e)}")
            await self.update_task_status(task.id, "failed", str(e))
            raise

    def _get_task_handler(self, task_type: str):
        """Get the appropriate handler for a task type from TASK_HANDLERS."""
        handler_cls = TASK_HANDLERS.get(task_type)
        if handler_cls:
            return handler_cls()
        return None

    async def _process_dependencies(self, task: Task) -> None:
        """Process tasks that depend on the completed task."""
        for task_id, task_data in self.tasks.items():
            if task.id in task_data.dependencies:
                # Check if all dependencies are completed
                all_deps_completed = all(
                    self.tasks[dep_id].status == "completed"
                    for dep_id in task_data.dependencies
                )
                if all_deps_completed:
                    await self.execute_task(task_data)

    async def start(self) -> None:
        """Start the orchestrator."""
        try:
            self.running = True
            logger.info("Starting orchestrator")
            
            while self.running:
                # Get next task from queue
                task_id = await self.redis_client.zpopmin("task_queue")
                if task_id:
                    task = await self.get_task(task_id[0][0])
                    if task:
                        await self.execute_task(task)
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Orchestrator failed: {str(e)}")
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.running = False
        logger.info("Stopping orchestrator")

if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = Orchestrator()
        try:
            # Create a sample task
            task = Task(
                id="task_001",
                name="Train LSTM Model",
                type="model_training",
                parameters={
                    "model_type": "lstm",
                    "epochs": 100,
                    "batch_size": 32
                }
            )
            
            # Add task to queue
            await orchestrator.create_task(task)
            
            # Start orchestrator
            await orchestrator.start()
            
        except Exception as e:
            logger.error(f"Main failed: {str(e)}")
        finally:
            await orchestrator.stop()

    asyncio.run(main()) 