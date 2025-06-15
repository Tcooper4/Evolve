"""Base class for all meta agents."""

import logging
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

@dataclass
class Task:
    """Represents a task to be processed by an agent."""
    id: str
    type: str
    data: Dict[str, Any]
    created_at: datetime = datetime.now()
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BaseMetaAgent(ABC):
    """Base class for all meta agents in the system."""
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict] = None,
        log_file_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the meta agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
            log_file_path: Optional custom path for log file
        """
        self.name = name
        self.config = config or {}
        self.running = False
        self._task_count = 0
        self._last_task: Optional[Task] = None
        self._last_run: Optional[datetime] = None
        
        # Setup logging
        self.setup_logging(log_file_path)
        
        # Create reports directory
        self.reports_dir = Path("reports/meta")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {name} agent")
    
    def setup_logging(self, log_file_path: Optional[Union[str, Path]] = None) -> None:
        """Setup logging for the agent.
        
        Args:
            log_file_path: Optional custom path for log file
        """
        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        log_dir = Path("logs/meta")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided log path or default
        if log_file_path:
            log_file = Path(log_file_path)
        else:
            log_file = log_dir / f"{self.name}.log"
            
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def start(self) -> None:
        """Start the agent and initialize resources."""
        self.logger.info(f"Starting {self.name} agent")
        self.running = True
        self.initialize()
        self.logger.info(f"{self.name} agent started successfully")
    
    def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self.logger.info(f"Stopping {self.name} agent")
        self.running = False
        self.cleanup()
        self.logger.info(f"{self.name} agent stopped")
    
    def run_once(self, task: Task) -> Dict[str, Any]:
        """Run a single task and track its status.
        
        Args:
            task: Task to process
            
        Returns:
            Dict containing task results and status
        """
        self.logger.info(f"Processing task {task.id} of type {task.type}")
        self._task_count += 1
        self._last_task = task
        self._last_run = datetime.now()
        
        try:
            result = self.process_task(task)
            task.status = "completed"
            task.result = result
            self.logger.info(f"Task {task.id} completed successfully")
            return {"result": result, "success": True, "error": None}
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.logger.error(f"Task {task.id} failed: {str(e)}")
            return {"result": None, "success": False, "error": str(e)}
    
    @abstractmethod
    def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Dict containing task results
        """
        pass
    
    def initialize(self) -> None:
        """Initialize agent resources. Override in subclasses if needed."""
        self.logger.info(f"Initializing {self.name} agent resources")
    
    def cleanup(self) -> None:
        """Cleanup agent resources. Override in subclasses if needed."""
        self.logger.info(f"Cleaning up {self.name} agent resources")
    
    def log_action(self, action: str, details: Optional[Dict] = None) -> None:
        """Log an agent action.
        
        Args:
            action: Description of the action
            details: Additional details about the action
        """
        self.logger.info(f"Action: {action}")
        if details:
            self.logger.debug(f"Details: {details}")
    
    def generate_report(self, results: Dict[str, Any]) -> Path:
        """Generate a report file.
        
        Args:
            results: Results to include in the report
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"{self.name}_{timestamp}.json"
        
        # Add metadata
        report = {
            "agent": self.name,
            "timestamp": timestamp,
            "config_hash": self._get_config_hash(),
            "task_count": self._task_count,
            "last_task": self._last_task.id if self._last_task else None,
            "results": results
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Generated report: {report_file}")
        return report_file
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.
        
        Returns:
            Dict containing agent status information
        """
        return {
            "name": self.name,
            "status": "running" if self.running else "idle",
            "config_hash": self._get_config_hash(),
            "task_count": self._task_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_task": {
                "id": self._last_task.id,
                "type": self._last_task.type,
                "status": self._last_task.status,
                "error": self._last_task.error
            } if self._last_task else None
        }
    
    def simulate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate task processing for testing.
        
        Args:
            task_data: Task data to simulate
            
        Returns:
            Dict containing simulation results
        """
        task = Task(
            id=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type="simulation",
            data=task_data
        )
        return self.run_once(task)
    
    def _get_config_hash(self) -> str:
        """Get a hash of the current configuration.
        
        Returns:
            String hash of the configuration
        """
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def __repr__(self) -> str:
        """Return a debug-friendly string representation of the agent.
        
        Returns:
            String representation of the agent
        """
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"config_hash='{self._get_config_hash()}', "
            f"running={self.running}, "
            f"task_count={self._task_count})"
        ) 