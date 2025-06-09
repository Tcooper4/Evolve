import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path
import aiohttp
import redis
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

import openai
from dotenv import load_dotenv

from .error_handler import ErrorHandler
from .monitor import SystemMonitor
from .documentation_manager import DocumentationManager
from .deployment_manager import DeploymentManager
from automation.web.websocket import WebSocketManager
from automation.notifications.notification_manager import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Task(BaseModel):
    """Task model for type safety and validation."""
    id: str
    type: str
    description: str
    priority: int
    dependencies: List[str] = []
    status: str = "scheduled"
    created_at: datetime
    metadata: Dict[str, Any] = {}

class DevelopmentOrchestrator:
    def __init__(self, config_path: str = "automation/config/config.json", notification_manager: NotificationManager):
        """Initialize the development orchestrator with enhanced features."""
        self.config_path = config_path
        self.config = self._load_config()
        self.tasks: List[Task] = []
        self.progress: Dict[str, Dict] = {}
        self.agents: Dict[str, Any] = {}
        
        # Initialize components
        self.error_handler = ErrorHandler(self.config)
        self.monitor = SystemMonitor(self.config)
        self.docs_manager = DocumentationManager(self.config)
        self.deployment_manager = DeploymentManager(self.config)
        
        # Initialize Redis for distributed task queue
        self.redis_client = redis.Redis(
            host=self.config.get("redis", {}).get("host", "localhost"),
            port=self.config.get("redis", {}).get("port", 6379),
            db=0
        )
        
        # Initialize FastAPI for microservices
        self.app = FastAPI(title="Development Orchestrator API")
        self.notification_manager = notification_manager
        self.websocket_manager = WebSocketManager(self.notification_manager)
        self._setup_routes()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _setup_routes(self):
        """Set up FastAPI routes for microservices."""
        @self.app.post("/tasks/")
        async def create_task(task: Task):
            return await self.schedule_task(task.dict())
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            return self.get_task_status(task_id)

    async def broadcast_task_update(self, task_id: str):
        """Broadcast task updates to all connected clients."""
        task_status = self.get_task_status(task_id)
        await self.websocket_manager.broadcast({
            "type": "task_update",
            "task_id": task_id,
            "status": task_status
        })

    async def broadcast_system_health(self):
        """Broadcast system health updates to all connected clients."""
        health_status = self.get_system_health()
        await self.websocket_manager.broadcast({
            "type": "system_health",
            "status": health_status
        })

    def _load_config(self) -> Dict:
        """Load configuration from JSON file with enhanced settings."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using default config")
            config = self._create_default_config()
        
        # Add microservices configuration
        config.update({
            "microservices": {
                "enabled": True,
                "services": {
                    "task_queue": {
                        "host": "localhost",
                        "port": 6379
                    },
                    "api": {
                        "host": "localhost",
                        "port": 8000
                    },
                    "monitoring": {
                        "host": "localhost",
                        "port": 8001
                    }
                }
            },
            "distributed": {
                "enabled": True,
                "nodes": [],
                "replication_factor": 2
            },
            "event_driven": {
                "enabled": True,
                "event_bus": "redis",
                "topics": [
                    "task_updates",
                    "system_health",
                    "model_updates",
                    "data_updates"
                ]
            }
        })
        
        return config

    def _create_default_config(self) -> Dict:
        """Create default configuration."""
        default_config = {
            "openai": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "agents": {
                "code_generation": {
                    "enabled": True,
                    "priority": 1
                },
                "testing": {
                    "enabled": True,
                    "priority": 2
                },
                "review": {
                    "enabled": True,
                    "priority": 3
                },
                "deployment": {
                    "enabled": True,
                    "priority": 4
                }
            },
            "paths": {
                "code_base": "trading",
                "tests": "tests",
                "docs": "docs"
            },
            "monitoring": {
                "enabled": True,
                "check_interval": 60,
                "alert_thresholds": {
                    "cpu": 80,
                    "memory": 85,
                    "disk": 90
                }
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 5,
                "recovery_enabled": True
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config

    async def schedule_task(self, task: Dict) -> str:
        """Schedule a new development task with enhanced features."""
        try:
            task_id = f"task_{len(self.tasks) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task["id"] = task_id
            task["status"] = "scheduled"
            task["created_at"] = datetime.now().isoformat()
            
            # Create task object
            task_obj = Task(**task)
            self.tasks.append(task_obj)
            
            # Initialize progress tracking
            self.progress[task_id] = {
                "status": "scheduled",
                "progress": 0,
                "steps_completed": [],
                "current_step": None,
                "errors": [],
                "start_time": datetime.now().isoformat(),
                "estimated_completion": None
            }
            
            # Publish task to distributed queue
            await self._publish_task(task_obj)
            
            logger.info(f"Scheduled task {task_id}: {task.get('type', 'unknown')}")
            return task_id
            
        except Exception as e:
            error_info = await self.error_handler.handle_error(e, {"task": task})
            logger.error(f"Error scheduling task: {str(e)}")
            raise

    async def _publish_task(self, task: Task):
        """Publish task to distributed queue."""
        try:
            # Serialize task
            task_data = task.json()
            
            # Publish to Redis
            await self.redis_client.publish(
                "task_queue",
                task_data
            )
            
            # Store in Redis for persistence
            await self.redis_client.set(
                f"task:{task.id}",
                task_data,
                ex=3600  # Expire after 1 hour
            )
        except Exception as e:
            logger.error(f"Error publishing task: {str(e)}")
            raise

    async def coordinate_agents(self, task_id: str) -> None:
        """Coordinate AI agents for a specific task with enhanced features."""
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        try:
            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor.start_monitoring())
            
            # Update task status
            self.progress[task_id]["status"] = "in_progress"
            await self._publish_task_update(task_id, "in_progress")
            
            # Execute task based on type
            if task.type == "feature_implementation":
                await self._handle_feature_implementation(task)
            elif task.type == "bug_fix":
                await self._handle_bug_fix(task)
            elif task.type == "code_review":
                await self._handle_code_review(task)
            elif task.type == "model_training":
                await self._handle_model_training(task)
            elif task.type == "data_processing":
                await self._handle_data_processing(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")

            # Update task status
            self.progress[task_id]["status"] = "completed"
            self.progress[task_id]["progress"] = 100
            await self._publish_task_update(task_id, "completed")
            
            # Stop monitoring
            monitor_task.cancel()
            
        except Exception as e:
            error_info = await self.error_handler.handle_error(e, {"task_id": task_id, "task": task})
            logger.error(f"Error in task {task_id}: {str(e)}")
            self.progress[task_id]["status"] = "failed"
            self.progress[task_id]["errors"].append(str(e))
            await self._publish_task_update(task_id, "failed")
            raise

    async def _publish_task_update(self, task_id: str, status: str):
        """Publish task update to event bus."""
        try:
            update = {
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            await self.redis_client.publish(
                "task_updates",
                json.dumps(update)
            )
        except Exception as e:
            logger.error(f"Error publishing task update: {str(e)}")

    async def _handle_model_training(self, task: Task) -> None:
        """Handle model training task."""
        task_id = task.id
        
        try:
            # Step 1: Data preparation
            self.progress[task_id]["current_step"] = "data_preparation"
            data = await self._prepare_training_data(task)
            self.progress[task_id]["steps_completed"].append("data_preparation")
            self.progress[task_id]["progress"] = 25
            
            # Step 2: Model training
            self.progress[task_id]["current_step"] = "model_training"
            model = await self._train_model(task, data)
            self.progress[task_id]["steps_completed"].append("model_training")
            self.progress[task_id]["progress"] = 50
            
            # Step 3: Model evaluation
            self.progress[task_id]["current_step"] = "model_evaluation"
            evaluation = await self._evaluate_model(task, model)
            self.progress[task_id]["steps_completed"].append("model_evaluation")
            self.progress[task_id]["progress"] = 75
            
            # Step 4: Model deployment
            self.progress[task_id]["current_step"] = "model_deployment"
            await self._deploy_model(task, model, evaluation)
            self.progress[task_id]["steps_completed"].append("model_deployment")
            self.progress[task_id]["progress"] = 100
            
        except Exception as e:
            error_info = await self.error_handler.handle_error(e, {"task_id": task_id, "task": task})
            raise

    async def _handle_data_processing(self, task: Task) -> None:
        """Handle data processing task."""
        task_id = task.id
        
        try:
            # Step 1: Data validation
            self.progress[task_id]["current_step"] = "data_validation"
            validation = await self._validate_data(task)
            self.progress[task_id]["steps_completed"].append("data_validation")
            self.progress[task_id]["progress"] = 25
            
            # Step 2: Data cleaning
            self.progress[task_id]["current_step"] = "data_cleaning"
            cleaned_data = await self._clean_data(task, validation)
            self.progress[task_id]["steps_completed"].append("data_cleaning")
            self.progress[task_id]["progress"] = 50
            
            # Step 3: Feature engineering
            self.progress[task_id]["current_step"] = "feature_engineering"
            features = await self._engineer_features(task, cleaned_data)
            self.progress[task_id]["steps_completed"].append("feature_engineering")
            self.progress[task_id]["progress"] = 75
            
            # Step 4: Data storage
            self.progress[task_id]["current_step"] = "data_storage"
            await self._store_data(task, features)
            self.progress[task_id]["steps_completed"].append("data_storage")
            self.progress[task_id]["progress"] = 100
            
        except Exception as e:
            error_info = await self.error_handler.handle_error(e, {"task_id": task_id, "task": task})
            raise

    def get_task_status(self, task_id: str) -> Dict:
        """Get task status with enhanced information."""
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        status = self.progress[task_id].copy()
        status.update({
            "task": task.dict(),
            "system_health": self.monitor.get_system_health(),
            "estimated_completion": self._estimate_completion(task_id)
        })
        
        return status

    def _estimate_completion(self, task_id: str) -> Optional[str]:
        """Estimate task completion time."""
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task or task.status == "completed":
            return None
            
        # Calculate estimated completion based on progress and system load
        progress = self.progress[task_id]
        if not progress.get("start_time"):
            return None
            
        start_time = datetime.fromisoformat(progress["start_time"])
        elapsed = datetime.now() - start_time
        
        if progress["progress"] == 0:
            return None
            
        estimated_total = elapsed / (progress["progress"] / 100)
        estimated_completion = start_time + estimated_total
        
        return estimated_completion.isoformat()

    def get_system_health(self) -> Dict:
        """Get enhanced system health information."""
        health = self.monitor.get_system_health()
        health.update({
            "active_tasks": len([t for t in self.tasks if t.status == "in_progress"]),
            "queued_tasks": len([t for t in self.tasks if t.status == "scheduled"]),
            "completed_tasks": len([t for t in self.tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks if t.status == "failed"]),
            "system_load": self.monitor.get_system_load(),
            "memory_usage": self.monitor.get_memory_usage(),
            "disk_usage": self.monitor.get_disk_usage()
        })
        return health 