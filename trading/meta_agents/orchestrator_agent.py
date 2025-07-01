"""
Orchestrator Agent

This module implements a specialized agent for orchestrating and coordinating
system-wide operations, managing agent lifecycles, and handling inter-agent
communication.

Note: This module was adapted from the legacy automation/agents/orchestrator.py file.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pathlib import Path
from trading.base_agent import BaseAgent
from trading.task_agent import TaskAgent
from trading.monitor_agent import MonitorAgent
from trading.alert_agent import AlertAgent

class OrchestratorAgent(BaseAgent):
    """Agent responsible for system orchestration and coordination."""
    
    def __init__(self, config: Dict):
        """Initialize the orchestrator agent."""
        super().__init__(config)
        self.agents: Dict[str, BaseAgent] = {}
        self.setup_logging()
    def setup_logging(self):
        """Configure logging for orchestration."""
        log_path = Path("logs/orchestration")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "orchestrator_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    async def initialize(self) -> None:
        """Initialize the orchestrator agent."""
        try:
            # Initialize core agents
            self.agents = {
                "task": TaskAgent(self.config),
                "monitor": MonitorAgent(self.config),
                "alert": AlertAgent(self.config)
            }
            
            # Initialize all agents
            for name, agent in self.agents.items():
                await agent.initialize()
                self.logger.info(f"Initialized {name} agent")
            
            self.logger.info("Orchestrator agent initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator agent: {str(e)}")
            raise
    
    async def start_agent(self, agent_name: str) -> None:
        """Start a specific agent."""
        try:
            if agent_name in self.agents:
                await self.agents[agent_name].start()
                self.logger.info(f"Started {agent_name} agent")
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
        except Exception as e:
            self.logger.error(f"Error starting {agent_name} agent: {str(e)}")
            raise
    
    async def stop_agent(self, agent_name: str) -> None:
        """Stop a specific agent."""
        try:
            if agent_name in self.agents:
                await self.agents[agent_name].stop()
                self.logger.info(f"Stopped {agent_name} agent")
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
        except Exception as e:
            self.logger.error(f"Error stopping {agent_name} agent: {str(e)}")
            raise
    
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get the status of a specific agent."""
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                return {
                    "name": agent_name,
                    "status": "running" if agent.is_running else "stopped",
                    "last_heartbeat": agent.last_heartbeat,
                    "config": agent.config
                }
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
        except Exception as e:
            self.logger.error(f"Error getting status for {agent_name} agent: {str(e)}")
            raise
    
    async def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a new agent with the orchestrator."""
        try:
            if name in self.agents:
                raise ValueError(f"Agent {name} already registered")
            
            self.agents[name] = agent
            await agent.initialize()
            self.logger.info(f"Registered new agent: {name}")
        except Exception as e:
            self.logger.error(f"Error registering agent {name}: {str(e)}")
            raise
    
    async def unregister_agent(self, name: str) -> None:
        """Unregister an agent from the orchestrator."""
        try:
            if name in self.agents:
                await self.agents[name].stop()
                del self.agents[name]
                self.logger.info(f"Unregistered agent: {name}")
            else:
                raise ValueError(f"Unknown agent: {name}")
        except Exception as e:
            self.logger.error(f"Error unregistering agent {name}: {str(e)}")
            raise
    
    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all agents."""
        try:
            for name, agent in self.agents.items():
                if hasattr(agent, "handle_message"):
                    await agent.handle_message(message)
            self.logger.info("Broadcast message to all agents")
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {str(e)}")
            raise
    
    async def monitor_agents(self):
        """Monitor the health of all agents."""
        try:
            while True:
                for name, agent in self.agents.items():
                    if not agent.is_running:
                        self.logger.warning(f"Agent {name} is not running, attempting restart")
                        await self.start_agent(name)
                
                await asyncio.sleep(self.config.get("monitor_interval", 60))
                
        except Exception as e:
            self.logger.error(f"Error monitoring agents: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start the orchestrator agent."""
        try:
            await self.initialize()
            self.logger.info("Orchestrator agent started")
            
            # Start all agents
            for name in self.agents:
                await self.start_agent(name)
            
            # Start agent monitoring
            asyncio.create_task(self.monitor_agents())
            
        except Exception as e:
            self.logger.error(f"Error starting orchestrator agent: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the orchestrator agent."""
        try:
            # Stop all agents
            for name in self.agents:
                await self.stop_agent(name)
            
            self.logger.info("Orchestrator agent stopped")
        except Exception as e:
            self.logger.error(f"Error stopping orchestrator agent: {str(e)}")
            raise 