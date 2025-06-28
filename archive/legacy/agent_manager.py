# -*- coding: utf-8 -*-
"""Master controller for all automation agents in the trading system."""

import logging
import importlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import redis
from redis.exceptions import RedisError
import schedule
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import inspect

from trading.config import config
from trading.utils.error_logger import error_logger
from trading.base_agent import BaseAgent, AgentResult
from trading.meta_agents.code_review_agent import CodeReviewAgent
from trading.meta_agents.performance_monitor_agent import PerformanceMonitorAgent
from trading.meta_agents.data_quality_agent import DataQualityAgent
from trading.meta_agents.test_repair_agent import TestRepairAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages the lifecycle and orchestration of all automation agents."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the agent manager.
        
        Args:
            redis_url: Optional Redis connection URL. If not provided, will use config.
        """
        self.redis_url = redis_url or config.get("redis_url")
        self.redis_client = None
        self.use_redis = False
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        self.scheduler = schedule.Scheduler()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
        # Initialize Redis if available
        self._init_redis()
        
        # Set up logging
        self._setup_logging()
        
        # Register agents
        self.register_agents()
    
    def _init_redis(self):
        """Initialize Redis connection with fallback."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()  # Test connection
            self.use_redis = True
            logger.info("Successfully connected to Redis")
        except (RedisError, ValueError) as e:
            logger.warning(f"Redis connection failed: {str(e)}. Running in local mode.")
            self.use_redis = False
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_config = config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_path = Path(log_config.get("path", "logs"))
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler
        file_handler = logging.FileHandler(
            log_path / "agent_manager.log",
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        
        # Configure formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(log_level)
    
    def register_agents(self):
        """Register all available agent types."""
        # Core agent types
        self.agent_types = {
            "code_review": CodeReviewAgent,
            "performance_monitor": PerformanceMonitorAgent,
            "data_quality": DataQualityAgent,
            "test_repair": TestRepairAgent
        }
        
        # Try to load additional agents from config
        try:
            agent_modules = config.get("agent_modules", [])
            for module_path in agent_modules:
                try:
                    module = importlib.import_module(module_path)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseAgent) and 
                            attr != BaseAgent):
                            self.agent_types[attr_name.lower()] = attr
                except ImportError as e:
                    logger.error(f"Failed to import agent module {module_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading agent modules: {str(e)}")
        
        logger.info(f"Registered {len(self.agent_types)} agent types")
    
    def create_agent(self, agent_type: str, **kwargs) -> Optional[BaseAgent]:
        """Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            Created agent instance or None if creation failed
        """
        try:
            if agent_type not in self.agent_types:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agent_class = self.agent_types[agent_type]
            agent = agent_class(**kwargs)
            
            # Store metadata
            self.agent_metadata[agent.name] = {
                "type": agent_type,
                "created_at": datetime.now().isoformat(),
                "last_run": None,
                "status": "created",
                "error_count": 0
            }
            
            # Store instance
            self.agent_instances[agent.name] = agent
            
            logger.info(f"Created agent {agent.name} of type {agent_type}")
            return agent
            
        except Exception as e:
            error_msg = f"Failed to create agent {agent_type}: {str(e)}"
            logger.error(error_msg)
            error_logger.log_error(error_msg)
            return None
    
    def start_all_agents(self):
        """Start all registered agents."""
        self.running = True
        for agent_id, agent in self.agent_instances.items():
            try:
                agent.start()
                self._update_agent_status(agent_id, "running")
                logger.info(f"Started agent {agent_id}")
            except Exception as e:
                error_msg = f"Failed to start agent {agent_id}: {str(e)}"
                logger.error(error_msg)
                error_logger.log_error(error_msg)
                self._update_agent_status(agent_id, "error")
    
    def stop_all_agents(self):
        """Stop all running agents."""
        self.running = False
        for agent_id, agent in self.agent_instances.items():
            try:
                agent.stop()
                self._update_agent_status(agent_id, "stopped")
                logger.info(f"Stopped agent {agent_id}")
            except Exception as e:
                error_msg = f"Failed to stop agent {agent_id}: {str(e)}"
                logger.error(error_msg)
                error_logger.log_error(error_msg)
    
    def run_agent_once(self, agent_id: str) -> bool:
        """Run a specific agent once.
        
        Args:
            agent_id: ID of the agent to run
            
        Returns:
            True if run was successful, False otherwise
        """
        if agent_id not in self.agent_instances:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        try:
            agent = self.agent_instances[agent_id]
            agent.run()
            self._update_agent_status(agent_id, "completed")
            logger.info(f"Successfully ran agent {agent_id}")
            return True
        except Exception as e:
            error_msg = f"Failed to run agent {agent_id}: {str(e)}"
            logger.error(error_msg)
            error_logger.log_error(error_msg)
            self._update_agent_status(agent_id, "error")
            return False
    
    def _update_agent_status(self, agent_id: str, status: str):
        """Update agent status in Redis or local storage.
        
        Args:
            agent_id: ID of the agent
            status: New status
        """
        metadata = self.agent_metadata[agent_id]
        metadata["status"] = status
        metadata["last_run"] = datetime.now().isoformat()
        
        if status == "error":
            metadata["error_count"] += 1
        
        if self.use_redis:
            try:
                self.redis_client.hset(
                    f"agent:{agent_id}",
                    mapping=metadata
                )
            except RedisError as e:
                logger.warning(f"Failed to update Redis for agent {agent_id}: {str(e)}")
    
    def check_redis_status(self) -> Dict[str, Any]:
        """Check Redis connection status.
        
        Returns:
            Dict containing Redis status information
        """
        status = {
            "connected": False,
            "error": None,
            "last_check": datetime.now().isoformat()
        }
        
        if not self.redis_url:
            status["error"] = "No Redis URL configured"
            return status
        
        try:
            if self.redis_client:
                self.redis_client.ping()
                status["connected"] = True
        except RedisError as e:
            status["error"] = str(e)
        
        return status
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report of all agents.
        
        Returns:
            Dict containing status information for all agents
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "redis_status": self.check_redis_status(),
            "agents": {}
        }
        
        for agent_id, metadata in self.agent_metadata.items():
            report["agents"][agent_id] = {
                "type": metadata["type"],
                "status": metadata["status"],
                "last_run": metadata["last_run"],
                "error_count": metadata["error_count"]
            }
        
        return report
    
    def schedule_agent(self, agent_id: str, interval: int):
        """Schedule an agent to run at regular intervals.
        
        Args:
            agent_id: ID of the agent to schedule
            interval: Interval in seconds between runs
        """
        if agent_id not in self.agent_instances:
            logger.error(f"Agent {agent_id} not found")
            return
        
        def run_scheduled():
            if self.running:
                self.run_agent_once(agent_id)
        
        self.scheduler.every(interval).seconds.do(run_scheduled)
        logger.info(f"Scheduled agent {agent_id} to run every {interval} seconds")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread."""
        def scheduler_loop():
            while self.running:
                self.scheduler.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=scheduler_loop)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    
    def __del__(self):
        """Cleanup when the manager is destroyed."""
        self.stop_all_agents()
        self.executor.shutdown(wait=True)

    def register_agent(self, agent: BaseAgent):
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance to register
        """
        if agent.name in self.agent_instances:
            logger.warning(f"Agent {agent.name} already registered, overwriting")
        self.agent_instances[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
        
    def register_agent_class(self, name: str, agent_class: Type[BaseAgent]):
        """
        Register an agent class.
        
        Args:
            name: Name to register the class under
            agent_class: Agent class to register
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class.__name__} must inherit from BaseAgent")
        self.agent_types[name] = agent_class
        logger.info(f"Registered agent class: {name}")
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get a registered agent by name.
        
        Args:
            name: Name of the agent to get
            
        Returns:
            Optional[BaseAgent]: Agent instance if found
        """
        return self.agent_instances.get(name)
        
    def list_agents(self) -> List[str]:
        """
        List all registered agent names.
        
        Returns:
            List[str]: List of agent names
        """
        return list(self.agent_instances.keys())
        
    def discover_agents(self, package_name: str = "trading.agents"):
        """
        Discover and register agent classes in a package.
        
        Args:
            package_name: Name of the package to search
        """
        try:
            package = importlib.import_module(package_name)
            for name, obj in inspect.getmembers(package):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAgent) and 
                    obj != BaseAgent):
                    self.register_agent_class(name, obj)
        except ImportError as e:
            logger.error(f"Failed to discover agents in {package_name}: {e}")
            
    def run_agent(self, name: str, prompt: str, **kwargs) -> AgentResult:
        """
        Run a registered agent.
        
        Args:
            name: Name of the agent to run
            prompt: Input prompt for the agent
            **kwargs: Additional arguments for the agent
            
        Returns:
            AgentResult: Result of the agent execution
        """
        agent = self.get_agent(name)
        if not agent:
            return AgentResult(
                success=False,
                message=f"Agent not found: {name}"
            )
            
        try:
            if not agent.validate_input(prompt, **kwargs):
                return AgentResult(
                    success=False,
                    message=f"Invalid input for agent {name}"
                )
                
            result = agent.run(prompt, **kwargs)
            agent.log_execution(result)
            return result
            
        except Exception as e:
            logger.error(f"Error running agent {name}: {e}")
            return agent.handle_error(e)
            
    def run_all_agents(self, prompt: str, **kwargs) -> Dict[str, AgentResult]:
        """
        Run all registered agents with the same prompt.
        
        Args:
            prompt: Input prompt for the agents
            **kwargs: Additional arguments for the agents
            
        Returns:
            Dict[str, AgentResult]: Results from all agents
        """
        results = {}
        for name in self.list_agents():
            results[name] = self.run_agent(name, prompt, **kwargs)
        return results
