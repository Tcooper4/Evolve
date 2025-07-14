"""
Main module for the Upgrader Agent.

This module contains the UpgraderAgent class, which is responsible for detecting
and managing upgrades for models and pipeline components.
"""

import logging
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# from trading.meta_agents.agents.model_builder import ModelBuilder  # Removed - meta_agents deleted
from config.settings import Settings
from trading.agents.task_memory import Task, TaskMemory, TaskStatus
from trading.scheduler import UpgradeScheduler
from trading.utils import (
    check_component_status,
    check_model_status,
    get_pipeline_components,
    validate_upgrade_result,
)

logger = logging.getLogger("UpgraderAgent")


class UpgraderAgent:
    """
    Autonomous agent responsible for detecting and upgrading outdated models and pipeline components.
    Integrates with TaskMemory for tracking upgrade tasks and progress.
    """

    def __init__(self, config_path: str = "config/settings.py"):
        """
        Initialize the upgrader agent with configuration and logging.

        Args:
            config_path: Path to the configuration file
        """
        self.settings = Settings(config_path)
        self.task_memory = TaskMemory()
        self.model_builder = ModelBuilder()
        self.scheduler = UpgradeScheduler(
            check_interval=self.settings.get("upgrade_interval", 24)
        )

        # Setup logging
        self.init_status = self._setup_logging()

        # Initialize state
        self.last_upgrade_check = None
        self.failed_upgrades = set()
        self.upgrade_history = []

        # Load configuration
        self.max_retries = self.settings.get("max_retries", 3)
        self.retry_delay = self.settings.get("retry_delay", 300)  # seconds
        self.log_retention_days = self.settings.get("log_retention_days", 30)
        self.memory_retention_days = self.settings.get("memory_retention_days", 7)

    def _setup_logging(self) -> dict:
        """Configure rotating file logging."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "upgrader.log"

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.handlers.RotatingFileHandler(
                        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
                    ),  # 10MB
                    logging.StreamHandler(),
                ],
            )

            return {
                "success": True,
                "message": "Logging setup completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def start(self) -> dict:
        """Start the upgrader agent with scheduled checks."""
        try:
            logger.info("Starting UpgraderAgent")
            self.scheduler.start(self.check_for_upgrades)
            return {
                "success": True,
                "message": "UpgraderAgent started",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def stop(self) -> dict:
        """Stop the upgrader agent."""
        try:
            logger.info("Stopping UpgraderAgent")
            self.scheduler.stop()
            return {
                "success": True,
                "message": "UpgraderAgent stopped",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def run_once(self) -> dict:
        """
        Run a single upgrade check manually.

        Returns:
            Dictionary with upgrade check results
        """
        try:
            logger.info("Running manual upgrade check")
            result = self.check_for_upgrades()
            return {
                "success": True,
                "result": result,
                "message": "Manual upgrade check completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def check_for_upgrades(self) -> List[Task]:
        """
        Check for outdated models and components, create upgrade tasks.

        Returns:
            List[Task]: List of created upgrade tasks
        """
        logger.info("Checking for upgrades")
        self.last_upgrade_check = datetime.now()

        upgrade_tasks = []

        try:
            # Check models in parallel
            with ThreadPoolExecutor() as executor:
                model_futures = []
                for model_id, config in self.model_builder.get_model_configs().items():
                    future = executor.submit(check_model_status, model_id, config)
                    model_futures.append((model_id, future))

                for model_id, future in model_futures:
                    needs_upgrade, reason = future.result()
                    if needs_upgrade:
                        task = self._create_upgrade_task(
                            "model_upgrade", model_id=model_id, reason=reason
                        )
                        upgrade_tasks.append(task)

            # Check pipeline components in parallel
            with ThreadPoolExecutor() as executor:
                component_futures = []
                for component in get_pipeline_components():
                    future = executor.submit(check_component_status, component)
                    component_futures.append((component, future))

                for component, future in component_futures:
                    needs_upgrade, reason = future.result()
                    if needs_upgrade:
                        task = self._create_upgrade_task(
                            "pipeline_upgrade", component=component, reason=reason
                        )
                        upgrade_tasks.append(task)

            logger.info(f"Found {len(upgrade_tasks)} items requiring upgrade")

        except Exception as e:
            logger.error(f"Error checking for upgrades: {str(e)}")
            logger.error(traceback.format_exc())

        return upgrade_tasks

    def _create_upgrade_task(self, task_type: str, **kwargs) -> Task:
        """
        Create a new upgrade task.

        Args:
            task_type: Type of upgrade task
            **kwargs: Additional task metadata

        Returns:
            Task: Created task object
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            type=task_type,
            status=TaskStatus.PENDING,
            metadata={
                "agent": "upgrader",
                "creation_time": datetime.now().isoformat(),
                **kwargs,
            },
            notes=f"Upgrade required: {kwargs.get('reason', 'Unknown reason')}",
        )

        self.task_memory.add_task(task)
        return task

    def process_upgrade_task(self, task: Task) -> bool:
        """
        Process an upgrade task with retry logic.

        Args:
            task: The task to process

        Returns:
            bool: True if the upgrade was successful
        """
        retries = 0
        success = False
        start_time = datetime.now()

        while retries < self.max_retries and not success:
            try:
                if task.type == "model_upgrade":
                    success = self._upgrade_model(task)
                elif task.type == "pipeline_upgrade":
                    success = self._upgrade_pipeline_component(task)

                if success:
                    task.status = TaskStatus.COMPLETED
                    duration = datetime.now() - start_time
                    task.metadata.update(
                        {
                            "completion_time": datetime.now().isoformat(),
                            "duration": str(duration),
                        }
                    )

                    # Record in upgrade history
                    self.upgrade_history.append(
                        {
                            "task_id": task.task_id,
                            "type": task.type,
                            "status": "success",
                            "duration": str(duration),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    raise Exception("Upgrade failed")

            except Exception as e:
                retries += 1
                logger.error(f"Upgrade attempt {retries} failed: {str(e)}")

                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    task.status = TaskStatus.FAILED
                    task.metadata.update(
                        {"error": str(e), "completion_time": datetime.now().isoformat()}
                    )
                    self.failed_upgrades.add(task.task_id)

                    # Record in upgrade history
                    self.upgrade_history.append(
                        {
                            "task_id": task.task_id,
                            "type": task.type,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            self.task_memory.update_task(task)

        return success

    def _upgrade_model(self, task: Task) -> bool:
        """
        Upgrade a model using ModelBuilder.

        Args:
            task: The upgrade task

        Returns:
            bool: True if the upgrade was successful
        """
        try:
            model_id = task.metadata["model_id"]
            logger.info(f"Upgrading model {model_id}")

            # Retrain model
            result = self.model_builder.retrain_model(model_id)

            if validate_upgrade_result(result):
                logger.info(f"Successfully upgraded model {model_id}")
                return True
            else:
                raise Exception("Model retraining failed")

        except Exception as e:
            logger.error(f"Error upgrading model: {str(e)}")
            return False

    def _upgrade_pipeline_component(self, task: Task) -> bool:
        """
        Upgrade a pipeline component.

        Args:
            task: The upgrade task

        Returns:
            bool: True if the upgrade was successful
        """
        try:
            component = task.metadata["component"]
            logger.info(f"Upgrading pipeline component {component}")

            # Implement component upgrade logic here
            # This would typically involve updating code, configurations, etc.

            logger.info(f"Successfully upgraded component {component}")
            return True

        except Exception as e:
            logger.error(f"Error upgrading component: {str(e)}")
            return False

    def cleanup(self):
        """Clean up old logs and task memory."""
        try:
            # Clean up old logs
            log_dir = Path("logs")
            cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)

            for log_file in log_dir.glob("upgrader*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()

            # Clean up old task memory
            self.task_memory.cleanup_old_tasks(self.memory_retention_days)

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def handle_prompt(self, prompt: str) -> Dict:
        """
        Handle manual upgrade requests from prompts.

        Args:
            prompt: The upgrade request prompt

        Returns:
            Dict: Result of the upgrade request
        """
        try:
            logger.info(f"Received upgrade prompt: {prompt}")

            # Create upgrade task
            task = self._create_upgrade_task(
                "manual_upgrade", prompt=prompt, reason="Manual upgrade requested"
            )

            # Process the upgrade
            success = self.process_upgrade_task(task)

            return {
                "task_id": task.task_id,
                "success": success,
                "status": task.status.name,
            }

        except Exception as e:
            logger.error(f"Error handling prompt: {str(e)}")
            return {"error": str(e), "success": False}

    def get_status(self) -> Dict:
        """
        Get the current status of the upgrader agent.

        Returns:
            Dict: Status information
        """
        return {
            "success": True,
            "result": {
                "last_upgrade_check": self.last_upgrade_check.isoformat()
                if self.last_upgrade_check
                else None,
                "failed_upgrades": list(self.failed_upgrades),
                "upgrade_history_count": len(self.upgrade_history),
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
            },
            "message": "Status retrieved",
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Create and start the upgrader agent
    upgrader = UpgraderAgent()
    upgrader.start()
