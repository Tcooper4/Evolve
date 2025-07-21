"""
Agent Management and Callback System for Trading Performance

This module provides agent management capabilities, status tracking, and callback
handlers for performance monitoring and automated responses to trading events.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# --- Enums ---


class AgentStatus(Enum):
    """Enumeration of possible agent statuses."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    EVALUATING = "evaluating"


class AgentType(Enum):
    """Enumeration of agent types."""

    TRADING = "trading"
    ANALYSIS = "analysis"
    RISK_MANAGEMENT = "risk_management"
    PERFORMANCE_MONITOR = "performance_monitor"
    ALERT = "alert"
    OPTIMIZATION = "optimization"


class EventType(Enum):
    """Enumeration of event types that can trigger agent callbacks."""

    UNDERPERFORMANCE = "underperformance"
    THRESHOLD_BREACH = "threshold_breach"
    SYSTEM_ERROR = "system_error"
    MARKET_EVENT = "market_event"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    TRAINING_COMPLETE = "training_complete"


# --- Data Models ---


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_id: str
    name: str
    agent_type: AgentType
    status: AgentStatus = AgentStatus.INACTIVE
    enabled: bool = True
    priority: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    agent_id: str
    success_rate: float = 0.0
    response_time: float = 0.0
    error_count: int = 0
    total_actions: int = 0
    last_action: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTask:
    """Task information for an agent."""

    task_id: str
    agent_id: str
    task_type: str
    status: str = "pending"
    priority: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class EventData:
    """Data structure for agent events."""

    event_id: str
    event_type: EventType
    agent_id: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    severity: str = "medium"
    handled: bool = False


@dataclass
class UIAgentInfo:
    """UI-friendly agent information."""

    agent_id: str
    name: str
    agent_type: str
    status: str
    task_count: int
    success_rate: float
    last_action: Optional[str]
    uptime: str
    priority: int
    enabled: bool


# --- Agent Manager ---


class AgentManager:
    """Manages agent registration, status tracking, and event handling."""

    def __init__(self):
        """Initialize the agent manager."""
        self.agents: Dict[str, AgentConfig] = {}
        self.metrics: Dict[str, AgentMetrics] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.callbacks: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self.event_history: List[EventData] = []
        self.agent_status_log: Dict[str, Dict[str, Any]] = (
            {}
        )  # Track agent status with timestamps
        self._load_agents()

    def register_agent(self, agent_config: AgentConfig) -> bool:
        """Register a new agent.

        Args:
            agent_config: Configuration for the agent to register

        Returns:
            True if successful, False otherwise
        """
        try:
            self.agents[agent_config.agent_id] = agent_config
            self.metrics[agent_config.agent_id] = AgentMetrics(
                agent_id=agent_config.agent_id
            )
            self._save_agents()
            logger.info(
                f"Registered agent: {agent_config.name} ({agent_config.agent_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Error registering agent {agent_config.agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: ID of the agent to unregister

        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                if agent_id in self.metrics:
                    del self.metrics[agent_id]
                # Remove associated tasks
                self.tasks = {
                    task_id: task
                    for task_id, task in self.tasks.items()
                    if task.agent_id != agent_id
                }
                self._save_agents()
                logger.info(f"Unregistered agent: {agent_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update the status of an agent.

        Args:
            agent_id: ID of the agent
            status: New status for the agent

        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_id in self.agents:
                old_status = self.agents[agent_id].status
                self.agents[agent_id].status = status
                self.agents[agent_id].updated_at = datetime.now().isoformat()

                # Track agent status with timestamps and log transitions
                self.agent_status_log[agent_id] = {
                    "status": status.value,
                    "timestamp": datetime.now().isoformat(),
                    "previous_status": old_status.value,
                }

                # Log status transition
                logger.info(
                    f"Agent {agent_id} status transition: {old_status.value} â†’ {status.value}"
                )

                self._save_agents()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating agent status {agent_id}: {e}")
            return False

    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent configuration or None if not found
        """
        return self.agents.get(agent_id)

    def get_active_agents(
        self, agent_type: Optional[AgentType] = None
    ) -> List[AgentConfig]:
        """Get all active agents, optionally filtered by type.

        Args:
            agent_type: Optional agent type filter

        Returns:
            List of active agent configurations
        """
        active_agents = [
            agent
            for agent in self.agents.values()
            if agent.status == AgentStatus.ACTIVE and agent.enabled
        ]

        if agent_type:
            active_agents = [
                agent for agent in active_agents if agent.agent_type == agent_type
            ]

        return active_agents

    def get_active_agent_names_and_task_counts(self) -> List[Tuple[str, str, int]]:
        """Get active agent names and their task counts for UI display.

        Returns:
            List of tuples containing (agent_id, agent_name, task_count)
        """
        active_agents = self.get_active_agents()
        result = []

        for agent in active_agents:
            # Count tasks for this agent
            task_count = sum(
                1
                for task in self.tasks.values()
                if task.agent_id == agent.agent_id and task.status != "completed"
            )
            result.append((agent.agent_id, agent.name, task_count))

        return result

    def get_ui_agent_info(self, agent_id: Optional[str] = None) -> List[UIAgentInfo]:
        """Get UI-friendly agent information.

        Args:
            agent_id: Optional specific agent ID to get info for

        Returns:
            List of UI agent information objects
        """
        agents_to_process = (
            [self.agents[agent_id]]
            if agent_id and agent_id in self.agents
            else self.agents.values()
        )
        result = []

        for agent in agents_to_process:
            # Count tasks for this agent
            task_count = sum(
                1
                for task in self.tasks.values()
                if task.agent_id == agent.agent_id and task.status != "completed"
            )

            # Get metrics
            metrics = self.metrics.get(
                agent.agent_id, AgentMetrics(agent_id=agent.agent_id)
            )

            # Calculate uptime
            created_time = datetime.fromisoformat(
                agent.created_at.replace("Z", "+00:00")
            )
            uptime = datetime.now(created_time.tzinfo) - created_time
            uptime_str = str(uptime).split(".")[0]  # Remove microseconds

            # Get last action time
            last_action = None
            if metrics.last_action:
                last_action = metrics.last_action

            ui_info = UIAgentInfo(
                agent_id=agent.agent_id,
                name=agent.name,
                agent_type=agent.agent_type.value,
                status=agent.status.value,
                task_count=task_count,
                success_rate=metrics.success_rate,
                last_action=last_action,
                uptime=uptime_str,
                priority=agent.priority,
                enabled=agent.enabled,
            )
            result.append(ui_info)

        return result

    def get_agent_task_summary(self) -> Dict[str, Any]:
        """Get summary of agent tasks for UI display.

        Returns:
            Dictionary containing task summary information
        """
        task_summary = {
            "total_tasks": len(self.tasks),
            "pending_tasks": 0,
            "running_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "tasks_by_agent": defaultdict(int),
            "tasks_by_type": defaultdict(int),
            "recent_tasks": [],
        }

        # Count tasks by status
        for task in self.tasks.values():
            if task.status == "pending":
                task_summary["pending_tasks"] += 1
            elif task.status == "running":
                task_summary["running_tasks"] += 1
            elif task.status == "completed":
                task_summary["completed_tasks"] += 1
            elif task.status == "failed":
                task_summary["failed_tasks"] += 1

            task_summary["tasks_by_agent"][task.agent_id] += 1
            task_summary["tasks_by_type"][task.task_type] += 1

        # Get recent tasks (last 10)
        recent_tasks = sorted(
            self.tasks.values(), key=lambda x: x.created_at, reverse=True
        )[:10]
        task_summary["recent_tasks"] = [
            {
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "task_type": task.task_type,
                "status": task.status,
                "created_at": task.created_at,
            }
            for task in recent_tasks
        ]

        return task_summary

    def add_task(
        self,
        agent_id: str,
        task_type: str,
        priority: int = 1,
        task_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new task for an agent.

        Args:
            agent_id: ID of the agent to assign the task to
            task_type: Type of task
            priority: Task priority (1-10, higher is more important)
            task_data: Optional task-specific data

        Returns:
            Task ID
        """
        task_id = f"{agent_id}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = AgentTask(
            task_id=task_id, agent_id=agent_id, task_type=task_type, priority=priority
        )

        self.tasks[task_id] = task
        logger.info(f"Added task {task_id} for agent {agent_id}")
        return task_id

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update task status.

        Args:
            task_id: ID of the task
            status: New status
            result: Optional task result
            error: Optional error message

        Returns:
            True if successful, False otherwise
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status

            if status == "running" and not task.started_at:
                task.started_at = datetime.now().isoformat()
            elif status in ["completed", "failed"]:
                task.completed_at = datetime.now().isoformat()

            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

            logger.info(f"Updated task {task_id} status to {status}")
            return True

        return False

    def get_agent_tasks(
        self, agent_id: str, status: Optional[str] = None
    ) -> List[AgentTask]:
        """Get tasks for a specific agent.

        Args:
            agent_id: ID of the agent
            status: Optional status filter

        Returns:
            List of tasks for the agent
        """
        tasks = [task for task in self.tasks.values() if task.agent_id == agent_id]

        if status:
            tasks = [task for task in tasks if task.status == status]

        return sorted(tasks, key=lambda x: x.created_at, reverse=True)

    def cleanup_completed_tasks(self, days_old: int = 7) -> int:
        """Clean up completed tasks older than specified days.

        Args:
            days_old: Number of days after which to remove completed tasks

        Returns:
            Number of tasks removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        tasks_to_remove = []

        for task_id, task in self.tasks.items():
            if (
                task.status in ["completed", "failed"]
                and task.completed_at
                and datetime.fromisoformat(task.completed_at.replace("Z", "+00:00"))
                < cutoff_date
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks[task_id]

        logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
        return len(tasks_to_remove)

    def _load_agents(self) -> None:
        """Load agent configurations from file."""
        try:
            agents_file = Path("trading/agents/agent_config.json")
            if agents_file.exists():
                with open(agents_file, "r") as f:
                    data = json.load(f)
                    for agent_data in data.get("agents", []):
                        agent_config = AgentConfig(
                            agent_id=agent_data["agent_id"],
                            name=agent_data["name"],
                            agent_type=AgentType(agent_data["agent_type"]),
                            status=AgentStatus(agent_data["status"]),
                            enabled=agent_data.get("enabled", True),
                            priority=agent_data.get("priority", 1),
                            config=agent_data.get("config", {}),
                            created_at=agent_data.get(
                                "created_at", datetime.now().isoformat()
                            ),
                            updated_at=agent_data.get(
                                "updated_at", datetime.now().isoformat()
                            ),
                        )
                        self.agents[agent_config.agent_id] = agent_config
                        self.metrics[agent_config.agent_id] = AgentMetrics(
                            agent_id=agent_config.agent_id
                        )
        except Exception as e:
            logger.error(f"Error loading agents: {e}")

    def _save_agents(self) -> None:
        """Save agent configurations to file."""
        try:
            agents_file = Path("trading/agents/agent_config.json")
            agents_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "agents": [
                    {
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "agent_type": agent.agent_type.value,
                        "status": agent.status.value,
                        "enabled": agent.enabled,
                        "priority": agent.priority,
                        "config": agent.config,
                        "created_at": agent.created_at,
                        "updated_at": agent.updated_at,
                    }
                    for agent in self.agents.values()
                ]
            }

            with open(agents_file, "w") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving agents: {e}")


# --- Event Handlers ---


class EventHandlers:
    """Collection of event handler functions."""

    @staticmethod
    def handle_underperformance(event_data: EventData) -> None:
        """Handle underperformance events with agentic logic.

        Args:
            event_data: Event data containing performance information
        """
        try:
            logger.warning(f"Underperformance detected for agent {event_data.agent_id}")

            # Extract performance data
            performance_data = event_data.data.get("performance", {})
            issues = performance_data.get("issues", [])

            # Log the issues
            for issue in issues:
                logger.warning(f"Performance issue: {issue}")

            # Implement automated responses
            logger.info(f"Implementing automated responses for {event_data.agent_id}")

            # Action 1: Trigger model retraining if performance is poor
            if performance_data["sharpe_ratio"] < 0.5:
                logger.info(f"Low Sharpe ratio detected - triggering model retraining")
                try:
                    from trading.agents.model_optimizer_agent import ModelOptimizerAgent

                    optimizer = ModelOptimizerAgent()
                    optimizer.execute(action="optimize_model", model_id="current")
                    logger.info("Model retraining triggered successfully")
                except Exception as e:
                    logger.error(f"Failed to trigger model retraining: {e}")

            # Action 2: Adjust trading parameters
            if performance_data["max_drawdown"] > 0.15:
                logger.info(f"High drawdown detected - adjusting trading parameters")
                try:
                    from trading.strategies.gatekeeper import StrategyGatekeeper

                    gatekeeper = StrategyGatekeeper()
                    gatekeeper.adjust_risk_parameters(reduce_risk=True)
                    logger.info("Trading parameters adjusted")
                except Exception as e:
                    logger.error(f"Failed to adjust trading parameters: {e}")

            # Action 3: Switch strategies if needed
            if performance_data["sharpe_ratio"] < 0.3:
                logger.info(
                    f"Very low Sharpe ratio - switching to conservative strategy"
                )
                try:
                    from trading.strategies.gatekeeper import StrategyGatekeeper

                    gatekeeper = StrategyGatekeeper()
                    gatekeeper.switch_strategy("conservative")
                    logger.info("Switched to conservative strategy")
                except Exception as e:
                    logger.error(f"Failed to switch strategy: {e}")
            if performance_data["sharpe_ratio"] < 0.5:
                logger.warning(
                    f"Low Sharpe ratio detected for {event_data.agent_id}: {performance_data['sharpe_ratio']}"
                )
                return "low_performance"
            elif performance_data["max_drawdown"] > 0.2:
                logger.warning(
                    f"High drawdown detected for {event_data.agent_id}: {performance_data['max_drawdown']}"
                )
                return "high_risk"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error handling underperformance event: {e}")

    @staticmethod
    def handle_threshold_breach(event_data: EventData) -> None:
        """Handle threshold breach events.

        Args:
            event_data: Event data containing threshold information
        """
        try:
            logger.warning(f"Threshold breach detected for agent {event_data.agent_id}")

            # Extract threshold data
            threshold_data = event_data.data.get("threshold", {})
            metric = threshold_data.get("metric")
            value = threshold_data.get("value")
            threshold = threshold_data.get("threshold")

            logger.warning(f"Metric {metric} breached threshold: {value} > {threshold}")

            # Implement threshold breach responses
            logger.info(
                f"Implementing threshold breach responses for agent {event_data.agent_id}"
            )

            # Determine breach type based on metric
            if metric in ["sharpe_ratio", "return_rate", "win_rate"]:
                breach_type = "performance"
            elif metric in ["max_drawdown", "var", "volatility"]:
                breach_type = "risk"
            else:
                breach_type = "unknown"

            if breach_type == "performance":
                logger.warning(
                    f"Performance threshold breached for {event_data.agent_id}"
                )
                # Action: Reduce position sizes and switch to conservative mode
                try:
                    from trading.strategies.gatekeeper import StrategyGatekeeper

                    gatekeeper = StrategyGatekeeper()
                    gatekeeper.adjust_risk_parameters(reduce_risk=True)
                    logger.info("Reduced position sizes due to performance breach")
                except Exception as e:
                    logger.error(f"Failed to adjust risk parameters: {e}")
                return "performance_breach"
            elif breach_type == "risk":
                logger.error(f"Risk threshold breached for {event_data.agent_id}")
                # Action: Stop trading and switch to conservative mode
                try:
                    from trading.strategies.gatekeeper import StrategyGatekeeper

                    gatekeeper = StrategyGatekeeper()
                    gatekeeper.switch_strategy("conservative")
                    gatekeeper.stop_trading()
                    logger.info("Stopped trading due to risk breach")
                except Exception as e:
                    logger.error(f"Failed to stop trading: {e}")
                return "risk_breach"
            else:
                logger.info(f"Unknown breach type: {breach_type}")
                return "unknown_breach"

        except Exception as e:
            logger.error(f"Error handling threshold breach event: {e}")

    @staticmethod
    def handle_system_error(event_data: EventData) -> None:
        """Handle system error events.

        Args:
            event_data: Event data containing error information
        """
        try:
            logger.error(f"System error detected for agent {event_data.agent_id}")

            # Extract error data
            error_data = event_data.data.get("error", {})
            error_type = error_data.get("type")
            error_message = error_data.get("message")

            logger.error(f"Error type: {error_type}, Message: {error_message}")

            # Implement error recovery
            logger.info(f"Implementing error recovery for agent {event_data.agent_id}")

            # Action 1: Restart agent if possible
            try:
                agent_manager = get_agent_manager()
                agent = agent_manager.get_agent(event_data.agent_id)
                if agent:
                    agent_manager.update_agent_status(
                        event_data.agent_id, AgentStatus.INACTIVE
                    )
                    logger.info(f"Agent {event_data.agent_id} stopped for recovery")
                    # In a real implementation, this would trigger a restart
            except Exception as e:
                logger.error(f"Failed to restart agent {event_data.agent_id}: {e}")

            # Action 2: Switch to backup system if available
            try:
                backup_agent_id = f"{event_data.agent_id}_backup"
                backup_agent = agent_manager.get_agent(backup_agent_id)
                if backup_agent:
                    agent_manager.update_agent_status(
                        backup_agent_id, AgentStatus.ACTIVE
                    )
                    logger.info(f"Switched to backup agent {backup_agent_id}")
            except Exception as e:
                logger.error(f"Failed to switch to backup agent: {e}")

            # Action 3: Send emergency alerts
            try:
                from trading.meta_agents.notification_handlers import (
                    NotificationHandler,
                )

                handler = NotificationHandler()
                handler.send_alert(
                    level="critical",
                    message=f"System error in agent {event_data.agent_id}: {error_type} - {error_message}",
                    recipients=["admin", "emergency"],
                )
                logger.info("Emergency alert sent")
            except Exception as e:
                logger.error(f"Failed to send emergency alert: {e}")

        except Exception as e:
            logger.error(f"Error handling system error event: {e}")


# --- Global Agent Manager Instance ---
_agent_manager = AgentManager()

# --- Convenience Functions ---


def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance.

    Returns:
        Global agent manager instance
    """
    return _agent_manager


def register_agent(agent_config: AgentConfig) -> bool:
    """Register a new agent using the global manager.

    Args:
        agent_config: Configuration for the agent to register

    Returns:
        True if successful, False otherwise
    """
    return _agent_manager.register_agent(agent_config)


def trigger_event(event_data: EventData) -> bool:
    """Trigger an event using the global manager.

    Args:
        event_data: Event data to trigger

    Returns:
        True if successful, False otherwise
    """
    return _agent_manager.trigger_event(event_data)


# --- Legacy Function Compatibility ---


def handle_underperformance(status_report: Dict[str, Any]) -> None:
    """Legacy function for handling underperformance.

    Args:
        status_report: Dictionary containing performance status information
    """
    try:
        # Create event data
        event_data = EventData(
            event_id=f"underperformance_{datetime.now().isoformat()}",
            event_type=EventType.UNDERPERFORMANCE,
            agent_id="performance_monitor",
            data={"performance": status_report},
            severity="high",
        )

        # Trigger the event
        trigger_event(event_data)

        # Implement agentic response actions
        performance_metrics = status_report.get("metrics", {})
        current_sharpe = performance_metrics.get("sharpe_ratio", 0)
        current_drawdown = performance_metrics.get("max_drawdown", 0)

        # Action 1: Trigger model retraining if performance is poor
        if current_sharpe < 0.5 or current_drawdown > 0.2:
            logger.info("Performance below thresholds - triggering model retraining")
            try:
                # Trigger model retraining agent
                from trading.agents.model_optimizer_agent import ModelOptimizerAgent

                optimizer = ModelOptimizerAgent()
                optimizer.execute(action="optimize_model", model_id="current")
                logger.info("Model retraining triggered successfully")
            except Exception as e:
                logger.error(f"Failed to trigger model retraining: {e}")

        # Action 2: Switch to conservative strategy if drawdown is high
        if current_drawdown > 0.15:
            logger.info("High drawdown detected - switching to conservative strategy")
            try:
                # Switch to conservative strategy
                from trading.strategies.gatekeeper import StrategyGatekeeper

                gatekeeper = StrategyGatekeeper()
                gatekeeper.switch_strategy("conservative")
                logger.info("Switched to conservative strategy")
            except Exception as e:
                logger.error(f"Failed to switch strategy: {e}")

        # Action 3: Send alert notifications
        if current_sharpe < 0.3 or current_drawdown > 0.25:
            logger.info("Critical performance issues - sending alerts")
            try:
                # Send notification
                from trading.meta_agents.notification_handlers import (
                    NotificationHandler,
                )

                handler = NotificationHandler()
                handler.send_alert(
                    level="critical",
                    message=f"Critical performance degradation: Sharpe={
                        current_sharpe:.3f}, Drawdown={
                        current_drawdown:.3f}",
                    recipients=["admin"],
                )
                logger.info("Alert notifications sent")
            except Exception as e:
                logger.error(f"Failed to send alerts: {e}")

        # Action 4: Log detailed analysis
        logger.info("[Agent Callback] Underperformance detected. Status report:")
        logger.info(status_report)
        logger.info(
            f"Actions taken: retraining={
                current_sharpe < 0.5}, strategy_switch={
                current_drawdown > 0.15}, alerts={
                current_sharpe < 0.3}")

    except Exception as e:
        logger.error(f"Error in legacy handle_underperformance: {e}")
        logger.info("[Agent Callback] Underperformance detected. Status report:")
        logger.info(status_report)
        logger.info("Agentic response failed - manual intervention may be required")


# --- Initialize Default Event Handlers ---

# Note: Legacy register_callback system removed.
# Use modern event handling patterns instead.
# If agent lifecycle events need tracking, implement proper hooks or logging systems.
