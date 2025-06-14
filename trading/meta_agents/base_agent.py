"""Base class for all meta agents."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class BaseMetaAgent(ABC):
    """Base class for all meta agents in the system."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize the meta agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(f"meta_agent.{name}")
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        self.log_dir = Path("logs/meta")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Create reports directory
        self.reports_dir = Path("reports/meta")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the agent's main task.
        
        Returns:
            Dict containing results and status
        """
        pass
    
    def log_action(self, action: str, details: Optional[Dict] = None):
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
            "results": results
        }
        
        # Save report
        import json
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
            "status": "active",
            "last_run": getattr(self, "_last_run", None),
            "config": self.config
        }
    
    def cleanup(self):
        """Cleanup resources used by the agent."""
        self.logger.info(f"Cleaning up {self.name} agent")
        # Override in subclasses if needed 