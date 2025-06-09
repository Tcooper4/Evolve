import asyncio
import logging
from typing import Dict, Any, Optional
import redis
import aiohttp
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from ..core.orchestrator import Task
from ..core.task_handlers import DataCollectionHandler

logger = logging.getLogger(__name__)

class DataCollectionAgent(BaseAgent):
    """Agent specialized in data collection tasks."""
    
    def __init__(self, agent_id: str, config_path: str = "automation/config/config.json"):
        super().__init__(agent_id, config_path)
        self.redis_client = None
        self.http_session = None
        self.handler = DataCollectionHandler(config_path)
        
    async def initialize(self) -> None:
        """Initialize Redis connection and HTTP session."""
        try:
            # Setup Redis
            redis_config = self.config["redis"]
            self.redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True
            )
            
            # Setup HTTP session
            self.http_session = aiohttp.ClientSession()
            
            # Register agent
            await self.redis_client.hset(
                "agents",
                self.agent_id,
                self.get_status()
            )
            
            logger.info(f"Data collection agent {self.agent_id} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data collection agent: {str(e)}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup Redis connection and HTTP session."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.http_session:
                await self.http_session.close()
            logger.info(f"Data collection agent {self.agent_id} cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup data collection agent: {str(e)}")
            raise
            
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a data collection task."""
        try:
            # Validate task type
            if task.type != "data_collection":
                raise ValueError(f"Invalid task type for data collection agent: {task.type}")
                
            # Process task using handler
            result = await self.handler.handle(task)
            
            # Update task status in Redis
            await self.redis_client.hset(
                "tasks",
                task.id,
                task.json()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process data collection task: {str(e)}")
            raise
            
    async def monitor_data_sources(self) -> None:
        """Monitor data sources for updates and trigger collection tasks."""
        try:
            # Get list of symbols to monitor
            symbols = await self.redis_client.smembers("monitored_symbols")
            
            for symbol in symbols:
                # Check last update time
                last_update = await self.redis_client.hget("data_updates", symbol)
                if not last_update or (datetime.now() - datetime.fromisoformat(last_update)) > timedelta(hours=1):
                    # Create and queue new collection task
                    task = Task(
                        id=f"data_collect_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        name=f"Collect data for {symbol}",
                        type="data_collection",
                        parameters={
                            "symbol": symbol,
                            "source": "yfinance"  # Default source
                        }
                    )
                    await self.add_task(task)
                    
        except Exception as e:
            logger.error(f"Failed to monitor data sources: {str(e)}")
            raise
            
    async def start(self) -> None:
        """Start the agent with data source monitoring."""
        try:
            self.running = True
            await self.initialize()
            logger.info(f"Data collection agent {self.agent_id} started")
            
            while self.running:
                await self.update_heartbeat()
                await self.monitor_data_sources()
                await self.process_tasks()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Data collection agent failed: {str(e)}")
            raise
        finally:
            self.running = False
            await self.cleanup() 