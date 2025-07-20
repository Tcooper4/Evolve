"""
Agent Swarm Orchestrator

Coordinates multiple agents (Strategy, Risk, Execution, Monitoring, etc.) as async jobs/processes.
Supports Redis or SQLite for coordination. Modular, robust, and non-duplicative.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from trading.agents.base_agent_interface import AgentConfig, AgentResult
from trading.agents.alpha import AlphaGenAgent, SignalTester, RiskValidator, SentimentIngestion, AlphaRegistry
from trading.agents.walk_forward_agent import WalkForwardAgent
from trading.agents.regime_detection_agent import RegimeDetectionAgent

logger = logging.getLogger(__name__)

class AgentType(Enum):
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    ALPHA_GEN = "alpha_gen"
    SIGNAL_TESTER = "signal_tester"
    RISK_VALIDATOR = "risk_validator"
    SENTIMENT = "sentiment"
    ALPHA_REGISTRY = "alpha_registry"
    WALK_FORWARD = "walk_forward"
    REGIME_DETECTION = "regime_detection"

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SwarmTask:
    task_id: str
    agent_type: AgentType
    agent_config: AgentConfig
    input_data: Dict[str, Any]
    status: AgentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[AgentResult] = None
    error_message: Optional[str] = None
    dependencies: List[str] = None

@dataclass
class SwarmConfig:
    max_concurrent_agents: int = 5
    coordination_backend: str = "sqlite"  # or "redis"
    redis_url: Optional[str] = None
    sqlite_path: str = "logs/swarm_coordination.db"
    task_timeout: int = 300
    retry_attempts: int = 3
    enable_logging: bool = True
    enable_monitoring: bool = True

class SwarmOrchestrator:
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, SwarmTask] = {}
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, SwarmTask] = {}
        self.agent_registry: Dict[AgentType, type] = {
            AgentType.ALPHA_GEN: AlphaGenAgent,
            AgentType.SIGNAL_TESTER: SignalTester,
            AgentType.RISK_VALIDATOR: RiskValidator,
            AgentType.SENTIMENT: SentimentIngestion,
            AgentType.ALPHA_REGISTRY: AlphaRegistry,
            AgentType.WALK_FORWARD: WalkForwardAgent,
            AgentType.REGIME_DETECTION: RegimeDetectionAgent,
        }
        self.loop = None
        self.running = False
        self.lock = threading.Lock()
        self.logger.info("SwarmOrchestrator initialized")

    async def start(self):
        if self.running:
            self.logger.warning("Swarm orchestrator already running")
            return
        self.running = True
        self.loop = asyncio.get_event_loop()
        self.logger.info("Swarm orchestrator started")

    async def stop(self):
        if not self.running:
            return
        self.running = False
        with self.lock:
            for task_id in list(self.running_tasks):
                await self._cancel_task(task_id)
        self.logger.info("Swarm orchestrator stopped")

    async def submit_task(self, agent_type: AgentType, agent_config: AgentConfig, input_data: Dict[str, Any], dependencies: Optional[List[str]] = None) -> str:
        task_id = str(uuid4())
        task = SwarmTask(
            task_id=task_id,
            agent_type=agent_type,
            agent_config=agent_config,
            input_data=input_data,
            status=AgentStatus.IDLE,
            created_at=datetime.now(),
            dependencies=dependencies or []
        )
        with self.lock:
            self.tasks[task_id] = task
        asyncio.create_task(self._execute_task(task_id))
        self.logger.info(f"Task {task_id} submitted for {agent_type.value}")
        return task_id

    async def _execute_task(self, task_id: str):
        with self.lock:
            if task_id not in self.tasks:
                return
            task = self.tasks[task_id]
            if not all(dep in self.completed_tasks for dep in (task.dependencies or [])):
                return
            if len(self.running_tasks) >= self.config.max_concurrent_agents:
                await asyncio.sleep(1)
                asyncio.create_task(self._execute_task(task_id))
                return
            task.status = AgentStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks.add(task_id)
        try:
            result = await self._run_agent(task)
            with self.lock:
                task.status = AgentStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                self.running_tasks.discard(task_id)
                self.completed_tasks[task_id] = task
            self.logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            with self.lock:
                task.status = AgentStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = str(e)
                self.running_tasks.discard(task_id)
            self.logger.error(f"Task {task_id} failed: {e}")

    async def _run_agent(self, task: SwarmTask) -> AgentResult:
        agent_class = self.agent_registry.get(task.agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {task.agent_type}")
        for attempt in range(self.config.retry_attempts):
            try:
                agent = agent_class(task.agent_config)
                result = await asyncio.wait_for(
                    agent.execute(**task.input_data),
                    timeout=self.config.task_timeout
                )
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task.task_id} timed out (attempt {attempt + 1})")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(1)

    async def _cancel_task(self, task_id: str):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = AgentStatus.CANCELLED
                self.running_tasks.discard(task_id)

    def get_task_status(self, task_id: str) -> Optional[SwarmTask]:
        with self.lock:
            return self.tasks.get(task_id) or self.completed_tasks.get(task_id)

    def get_swarm_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "total_tasks": len(self.tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len([t for t in self.completed_tasks.values() if t.status == AgentStatus.FAILED]),
                "max_concurrent_agents": self.config.max_concurrent_agents
            }
