import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
import json
import argparse

# Add the automation directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from agents.orchestrator import DevelopmentOrchestrator
from agents.code_generator import CodeGenerator
from agents.test_generator import TestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomationPipeline:
    def __init__(self):
        """Initialize the automation pipeline."""
        self.orchestrator = DevelopmentOrchestrator()
        self.code_generator = CodeGenerator(self.orchestrator.config)
        self.test_generator = TestGenerator(self.orchestrator.config)

    async def run_task(self, task: Dict) -> None:
        """Run a single task through the pipeline."""
        try:
            # Schedule the task
            task_id = await self.orchestrator.schedule_task(task)
            logger.info(f"Task {task_id} scheduled")

            # Coordinate the agents
            await self.orchestrator.coordinate_agents(task_id)
            logger.info(f"Task {task_id} completed")

        except Exception as e:
            logger.error(f"Error running task: {str(e)}")
            raise

    async def run_tasks(self, tasks: List[Dict]) -> None:
        """Run multiple tasks through the pipeline."""
        for task in tasks:
            await self.run_task(task)

    def load_tasks(self, tasks_file: str) -> List[Dict]:
        """Load tasks from a JSON file."""
        try:
            with open(tasks_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading tasks: {str(e)}")
            raise

    def save_results(self, results: Dict, output_file: str) -> None:
        """Save pipeline results to a JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

async def load_tasks(tasks_file: str) -> List[Dict]:
    """Load tasks from JSON file."""
    try:
        with open(tasks_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Tasks file not found: {tasks_file}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in tasks file: {tasks_file}")
        return []

async def run_pipeline(tasks_file: str, config_file: str) -> None:
    """Run the automation pipeline."""
    try:
        # Initialize orchestrator
        orchestrator = DevelopmentOrchestrator(config_file)
        
        # Load tasks
        tasks = await load_tasks(tasks_file)
        if not tasks:
            logger.error("No tasks to process")
            return
        
        # Schedule and execute tasks
        for task in tasks:
            try:
                # Schedule task
                task_id = await orchestrator.schedule_task(task)
                logger.info(f"Scheduled task {task_id}")
                
                # Execute task
                await orchestrator.coordinate_agents(task_id)
                
                # Get task status
                status = orchestrator.get_task_status(task_id)
                logger.info(f"Task {task_id} completed with status: {status['status']}")
                
                # Check system health
                health = orchestrator.get_system_health()
                if health["alerts"]:
                    logger.warning(f"System alerts: {health['alerts']}")
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Print final summary
        all_tasks = orchestrator.get_all_tasks()
        completed = sum(1 for t in all_tasks if t["progress"]["status"] == "completed")
        failed = sum(1 for t in all_tasks if t["progress"]["status"] == "failed")
        
        logger.info(f"Pipeline completed. Tasks: {len(all_tasks)}, Completed: {completed}, Failed: {failed}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the automation pipeline")
    parser.add_argument("--tasks", default="automation/config/tasks.json", help="Path to tasks JSON file")
    parser.add_argument("--config", default="automation/config/config.json", help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Run pipeline
    asyncio.run(run_pipeline(args.tasks, args.config))

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("automation/results", exist_ok=True)
    
    main() 