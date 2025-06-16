"""Orchestrator for managing trading agents and workflows."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from pathlib import Path

# Project imports
from ..utils.exceptions import OrchestratorError
from ..config.configuration import ConfigManager
from automation.web.websocket import WebSocketManager

class Orchestrator:
    """Orchestrator for managing trading agents and workflows."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the orchestrator.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = ConfigManager(config_path) if config_path else ConfigManager()
        self.websocket = WebSocketManager()
        self.agents = {}
        self.tasks = {}
        
    async def start(self) -> None:
        """Start the orchestrator."""
        try:
            # Initialize WebSocket connection
            await self.websocket.connect()
            
            # Start monitoring tasks
            self.tasks['monitor'] = asyncio.create_task(self._monitor_agents())
            self.tasks['heartbeat'] = asyncio.create_task(self._send_heartbeat())
            
            self.logger.info("Orchestrator started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting orchestrator: {e}")
            raise OrchestratorError(f"Failed to start orchestrator: {e}")
            
    async def stop(self) -> None:
        """Stop the orchestrator."""
        try:
            # Cancel all tasks
            for task in self.tasks.values():
                task.cancel()
                
            # Close WebSocket connection
            await self.websocket.disconnect()
            
            self.logger.info("Orchestrator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping orchestrator: {e}")
            raise OrchestratorError(f"Failed to stop orchestrator: {e}")
            
    async def _monitor_agents(self) -> None:
        """Monitor agent health and performance."""
        while True:
            try:
                for agent_id, agent in self.agents.items():
                    # Check agent health
                    if not await agent.is_healthy():
                        self.logger.warning(f"Agent {agent_id} is unhealthy")
                        await self.websocket.send_message({
                            'type': 'agent_health',
                            'agent_id': agent_id,
                            'status': 'unhealthy'
                        })
                        
                    # Get agent metrics
                    metrics = await agent.get_metrics()
                    await self.websocket.send_message({
                        'type': 'agent_metrics',
                        'agent_id': agent_id,
                        'metrics': metrics
                    })
                    
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring agents: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to connected clients."""
        while True:
            try:
                await self.websocket.send_message({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                })
                await asyncio.sleep(30)  # Send every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)  # Wait before retrying
                
    async def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance
        """
        if agent_id in self.agents:
            raise OrchestratorError(f"Agent {agent_id} already registered")
            
        self.agents[agent_id] = agent
        await self.websocket.send_message({
            'type': 'agent_registered',
            'agent_id': agent_id
        })
        
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id not in self.agents:
            raise OrchestratorError(f"Agent {agent_id} not found")
            
        del self.agents[agent_id]
        await self.websocket.send_message({
            'type': 'agent_unregistered',
            'agent_id': agent_id
        })
        
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary containing agent status
        """
        if agent_id not in self.agents:
            raise OrchestratorError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        return {
            'agent_id': agent_id,
            'is_healthy': await agent.is_healthy(),
            'metrics': await agent.get_metrics()
        }
        
    async def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all registered agents.
        
        Returns:
            List of agent status dictionaries
        """
        return [await self.get_agent_status(agent_id) for agent_id in self.agents] 