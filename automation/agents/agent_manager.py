import asyncio
import logging
from typing import Dict, Any, Optional, List, Type
import redis
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from .data_collection_agent import DataCollectionAgent
from ..core.orchestrator import Task
from ..config.config import load_config

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages and coordinates all agents in the system."""
    
    def __init__(self, config_path: str = "automation/config/config.json"):
        self.config = load_config(config_path)
        self.setup_logging()
        self.redis_client = None
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {
            "data_collection": DataCollectionAgent
            # Add other agent types here
        }
        self.running = False
        
    def setup_logging(self):
        """Setup logging for the agent manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('automation/logs/agent_manager.log'),
                logging.StreamHandler()
            ]
        )
        
    async def initialize(self) -> None:
        """Initialize Redis connection and load existing agents."""
        try:
            # Setup Redis
            redis_config = self.config["redis"]
            self.redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True
            )
            
            # Load existing agents
            agent_data = await self.redis_client.hgetall("agents")
            for agent_id, agent_info in agent_data.items():
                agent_type = agent_info.get("type")
                if agent_type in self.agent_types:
                    agent = self.agent_types[agent_type](agent_id)
                    self.agents[agent_id] = agent
                    
            logger.info("Agent manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {str(e)}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup Redis connection and stop all agents."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            for agent in self.agents.values():
                await agent.stop()
            logger.info("Agent manager cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup agent manager: {str(e)}")
            raise
            
    async def create_agent(self, agent_type: str, agent_id: Optional[str] = None) -> str:
        """Create a new agent of the specified type."""
        try:
            if agent_type not in self.agent_types:
                raise ValueError(f"Unsupported agent type: {agent_type}")
                
            if not agent_id:
                agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            agent = self.agent_types[agent_type](agent_id)
            self.agents[agent_id] = agent
            
            # Register agent in Redis
            await self.redis_client.hset(
                "agents",
                agent_id,
                {
                    "type": agent_type,
                    "created_at": datetime.now().isoformat(),
                    "status": "created"
                }
            )
            
            logger.info(f"Created agent {agent_id} of type {agent_type}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise
            
    async def start_agent(self, agent_id: str) -> None:
        """Start an agent."""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
                
            await agent.start()
            logger.info(f"Started agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {str(e)}")
            raise
            
    async def stop_agent(self, agent_id: str) -> None:
        """Stop an agent."""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
                
            await agent.stop()
            logger.info(f"Stopped agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {str(e)}")
            raise
            
    async def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        try:
            agent = self.agents.get(agent_id)
            if agent:
                await agent.stop()
                del self.agents[agent_id]
                
            await self.redis_client.hdel("agents", agent_id)
            logger.info(f"Removed agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove agent {agent_id}: {str(e)}")
            raise
            
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the status of an agent."""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
                
            return await agent.get_status()
            
        except Exception as e:
            logger.error(f"Failed to get agent status: {str(e)}")
            raise
            
    async def assign_task(self, task: Task) -> None:
        """Assign a task to an appropriate agent."""
        try:
            # Find suitable agent
            agent = self._find_suitable_agent(task)
            if not agent:
                raise ValueError(f"No suitable agent found for task type {task.type}")
                
            await agent.add_task(task)
            logger.info(f"Assigned task {task.id} to agent {agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to assign task: {str(e)}")
            raise
            
    def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        """Find a suitable agent for a task."""
        for agent in self.agents.values():
            if task.type in agent.get_status().get("task_types", []):
                return agent
        return None
        
    async def monitor_agents(self) -> None:
        """Monitor agent health and restart failed agents."""
        try:
            for agent_id, agent in list(self.agents.items()):
                status = await agent.get_status()
                last_heartbeat = datetime.fromisoformat(status["last_heartbeat"])
                
                # Check if agent is responsive
                if datetime.now() - last_heartbeat > timedelta(minutes=5):
                    logger.warning(f"Agent {agent_id} not responsive, restarting")
                    await self.stop_agent(agent_id)
                    await self.start_agent(agent_id)
                    
        except Exception as e:
            logger.error(f"Failed to monitor agents: {str(e)}")
            raise
            
    async def start(self) -> None:
        """Start the agent manager."""
        try:
            self.running = True
            await self.initialize()
            logger.info("Agent manager started")
            
            while self.running:
                await self.monitor_agents()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Agent manager failed: {str(e)}")
            raise
        finally:
            self.running = False
            await self.cleanup()
            
    async def stop(self) -> None:
        """Stop the agent manager."""
        self.running = False
        logger.info("Agent manager stopped") 