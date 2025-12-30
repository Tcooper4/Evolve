'''
Agent Hub - Central registry for all agents
This is a stub implementation to resolve import errors.
'''

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentHub:
    '''Central hub for agent management and coordination'''
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.logger = logger
        logger.info("AgentHub initialized")
    
    def register(self, name: str, agent: Any) -> None:
        '''Register an agent with the hub'''
        self.agents[name] = agent
        logger.info(f"Agent registered: {name}")
    
    def unregister(self, name: str) -> None:
        '''Unregister an agent from the hub'''
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Agent unregistered: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        '''Get an agent by name'''
        return self.agents.get(name)
    
    def list_agents(self) -> list:
        '''List all registered agents'''
        return list(self.agents.keys())
    
    def clear(self) -> None:
        '''Clear all registered agents'''
        self.agents.clear()
        logger.info("All agents cleared")


# Create a default instance
default_hub = AgentHub()
