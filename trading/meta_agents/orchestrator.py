"""Orchestrator agent for managing all meta agents."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import schedule
import time
import threading

from .base_agent import BaseMetaAgent
from .code_review_agent import CodeReviewAgent
from .test_repair_agent import TestRepairAgent
from .performance_monitor_agent import PerformanceMonitorAgent
from .auto_deployment_agent import AutoDeploymentAgent
from .documentation_agent import DocumentationAgent
from .integration_agent import IntegrationAgent
from .error_handler_agent import ErrorHandlerAgent
from .security_agent import SecurityAgent

class OrchestratorAgent(BaseMetaAgent):
    """Orchestrator agent that manages all other meta agents."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the orchestrator agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("orchestrator", config)
        
        # Initialize all agents
        self.agents: Dict[str, BaseMetaAgent] = {
            "code_review": CodeReviewAgent(config),
            "test_repair": TestRepairAgent(config),
            "performance": PerformanceMonitorAgent(config),
            "deployment": AutoDeploymentAgent(config),
            "documentation": DocumentationAgent(config),
            "integration": IntegrationAgent(config),
            "error_handler": ErrorHandlerAgent(config),
            "security": SecurityAgent(config)
        }
        
        # Setup schedules
        self.schedules = {
            "daily": self._setup_daily_schedule,
            "weekly": self._setup_weekly_schedule,
            "monthly": self._setup_monthly_schedule
        }
        
        self.running = False
        self.scheduler_thread = None
    
    def run(self) -> Dict[str, Any]:
        """Run all agents and collect results.
        
        Returns:
            Dict containing results from all agents
        """
        results = {}
        for name, agent in self.agents.items():
            try:
                self.log_action(f"Running agent: {name}")
                agent_results = agent.run()
                results[name] = agent_results
            except Exception as e:
                self.logger.error(f"Error running agent {name}: {str(e)}")
                results[name] = {"error": str(e)}
        
        return results
    
    def run_specific(self, agent_name: str) -> Dict[str, Any]:
        """Run a specific agent.
        
        Args:
            agent_name: Name of the agent to run
            
        Returns:
            Dict containing results from the agent
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        self.log_action(f"Running specific agent: {agent_name}")
        return agent.run()
    
    def start_scheduler(self, schedule_type: str = "daily"):
        """Start the scheduler for automated runs.
        
        Args:
            schedule_type: Type of schedule to use (daily, weekly, monthly)
        """
        if schedule_type not in self.schedules:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            args=(schedule_type,)
        )
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def _run_scheduler(self, schedule_type: str):
        """Run the scheduler loop.
        
        Args:
            schedule_type: Type of schedule to use
        """
        # Setup schedule
        self.schedules[schedule_type]()
        
        # Run scheduler loop
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def _setup_daily_schedule(self):
        """Setup daily schedule."""
        schedule.every().day.at("00:00").do(self.run)
    
    def _setup_weekly_schedule(self):
        """Setup weekly schedule."""
        schedule.every().monday.at("00:00").do(self.run)
    
    def _setup_monthly_schedule(self):
        """Setup monthly schedule."""
        schedule.every().day.at("00:00").do(
            lambda: self.run() if datetime.now().day == 1 else None
        )
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents.
        
        Returns:
            Dict containing status of all agents
        """
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
    
    def cleanup(self):
        """Cleanup all agents."""
        self.stop_scheduler()
        for agent in self.agents.values():
            agent.cleanup() 