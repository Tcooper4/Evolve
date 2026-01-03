"""
Task Monitor Module

This module contains task monitoring functionality for the task orchestrator.
Extracted from the original task_orchestrator.py for modularity.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .task_models import AgentStatus, TaskExecution, TaskStatus


class TaskMonitor:
    """Monitors task execution and system health."""

    def __init__(self, config: Dict[str, Any] = None, orchestrator=None):
        self.config = config or {}
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self.agent_status: Dict[str, AgentStatus] = {}
        self.performance_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_rates = defaultdict(float)

        # Health monitoring
        self.health_check_interval = config.get("health_check_interval_minutes", 5)
        self.last_health_check = datetime.utcnow()

        # Performance thresholds
        self.performance_threshold = config.get("performance_threshold", 0.8)
        self.error_threshold = config.get("error_threshold", 5)

    def update_task_performance(self, task_name: str, execution: TaskExecution) -> None:
        """Update performance metrics for a task."""
        # Update performance metrics
        if execution.duration_seconds:
            self.performance_metrics[task_name].append(execution.duration_seconds)

            # Keep only recent metrics
            if len(self.performance_metrics[task_name]) > 100:
                self.performance_metrics[task_name] = self.performance_metrics[
                    task_name
                ][-50:]

        # Update error counts
        if execution.status == TaskStatus.FAILED:
            self.error_counts[task_name] += 1
        else:
            self.error_counts[task_name] = max(0, self.error_counts[task_name] - 1)

        # Update success rates
        recent_executions = [execution]
        success_count = sum(
            1 for e in recent_executions if e.status == TaskStatus.COMPLETED
        )
        self.success_rates[task_name] = (
            success_count / len(recent_executions) if recent_executions else 0.0
        )

        # Update agent status
        self._update_agent_status(task_name, execution)

    def _update_agent_status(self, task_name: str, execution: TaskExecution) -> None:
        """Update agent status based on task execution."""
        agent_name = self._extract_agent_name(task_name)

        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = AgentStatus(agent_name=agent_name)

        agent = self.agent_status[agent_name]

        # Update execution counts
        if execution.status == TaskStatus.COMPLETED:
            agent.success_count += 1
        elif execution.status == TaskStatus.FAILED:
            agent.failure_count += 1
            agent.error_history.append(execution.error_message or "Unknown error")

            # Keep error history manageable
            if len(agent.error_history) > 10:
                agent.error_history = agent.error_history[-5:]

        # Update timing
        agent.last_execution = execution.end_time or execution.start_time

        # Calculate average duration
        if execution.duration_seconds:
            durations = self.performance_metrics.get(task_name, [])
            if durations:
                agent.average_duration = sum(durations) / len(durations)

        # Calculate health score
        agent.health_score = self._calculate_agent_health(agent)

    def _extract_agent_name(self, task_name: str) -> str:
        """Extract agent name from task name."""
        # Simple extraction - in practice, you'd have a proper mapping
        if "model" in task_name.lower():
            return "ModelInnovationAgent"
        elif "strategy" in task_name.lower():
            return "StrategyResearchAgent"
        elif "sentiment" in task_name.lower():
            return "SentimentFetcher"
        elif "meta" in task_name.lower():
            return "MetaController"
        elif "risk" in task_name.lower():
            return "RiskManager"
        elif "execution" in task_name.lower():
            return "ExecutionAgent"
        elif "explainer" in task_name.lower():
            return "ExplainerAgent"
        else:
            return task_name

    def _calculate_agent_health(self, agent: AgentStatus) -> float:
        """Calculate agent health score."""
        total_executions = agent.success_count + agent.failure_count
        if total_executions == 0:
            return 1.0

        # Base health on success rate
        success_rate = agent.success_count / total_executions

        # Penalize for recent errors
        error_penalty = min(len(agent.error_history) * 0.1, 0.5)

        # Penalize for slow performance
        performance_penalty = 0.0
        if agent.average_duration > 300:  # 5 minutes
            performance_penalty = 0.2

        health_score = success_rate - error_penalty - performance_penalty
        return max(0.0, min(1.0, health_score))

    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # Calculate overall health metrics
            total_agents = len(self.agent_status)
            healthy_agents = sum(
                1 for agent in self.agent_status.values() if agent.health_score > 0.7
            )

            overall_health = healthy_agents / total_agents if total_agents > 0 else 1.0

            # Check for critical issues
            critical_issues = []
            for agent_name, agent in self.agent_status.items():
                if agent.health_score < 0.3:
                    critical_issues.append(
                        f"{agent_name}: Low health score ({agent.health_score:.2f})"
                    )
                if agent.failure_count > self.error_threshold:
                    critical_issues.append(
                        f"{agent_name}: High failure count ({agent.failure_count})"
                    )

            # Check performance issues
            performance_issues = []
            for task_name, metrics in self.performance_metrics.items():
                if metrics:
                    avg_duration = sum(metrics) / len(metrics)
                    if avg_duration > 600:  # 10 minutes
                        performance_issues.append(
                            f"{task_name}: Slow performance ({avg_duration:.1f}s avg)"
                        )

            health_status = {
                "overall_health": overall_health,
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "critical_issues": critical_issues,
                "performance_issues": performance_issues,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.last_health_check = datetime.utcnow()

            # Log health status
            if critical_issues:
                self.logger.warning(f"System health issues detected: {critical_issues}")
            elif overall_health < 0.8:
                self.logger.warning(f"System health degraded: {overall_health:.2f}")
            else:
                self.logger.info(f"System health good: {overall_health:.2f}")

            return health_status

        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
            return {
                "overall_health": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Calculate aggregate metrics
            for task_name, metrics in self.performance_metrics.items():
                if metrics:
                    avg_duration = sum(metrics) / len(metrics)
                    max_duration = max(metrics)
                    min_duration = min(metrics)

                    # Store aggregate metrics
                    self.performance_metrics[f"{task_name}_avg"] = [avg_duration]
                    self.performance_metrics[f"{task_name}_max"] = [max_duration]
                    self.performance_metrics[f"{task_name}_min"] = [min_duration]

            # Update success rates
            for task_name in self.success_rates:
                recent_executions = self._get_recent_executions(task_name, hours=24)
                if recent_executions:
                    success_count = sum(
                        1 for e in recent_executions if e.status == TaskStatus.COMPLETED
                    )
                    self.success_rates[task_name] = success_count / len(
                        recent_executions
                    )

        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")

    def _get_recent_executions(
        self, task_name: str, hours: int = 24
    ) -> List[TaskExecution]:
        """Get recent executions for a task."""
        # This would typically query a database or storage
        # For now, return empty list
        return []

    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get agent status information."""
        if agent_name:
            agent = self.agent_status.get(agent_name)
            if agent:
                return {
                    "agent_name": agent.agent_name,
                    "is_running": agent.is_running,
                    "last_execution": agent.last_execution,
                    "next_scheduled": agent.next_scheduled,
                    "success_count": agent.success_count,
                    "failure_count": agent.failure_count,
                    "average_duration": agent.average_duration,
                    "health_score": agent.health_score,
                    "error_history": agent.error_history,
                }
            return {}

        return {
            agent_name: {
                "is_running": agent.is_running,
                "last_execution": agent.last_execution,
                "success_count": agent.success_count,
                "failure_count": agent.failure_count,
                "health_score": agent.health_score,
            }
            for agent_name, agent in self.agent_status.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            "total_tasks": len(self.performance_metrics),
            "average_success_rate": (
                sum(self.success_rates.values()) / len(self.success_rates)
                if self.success_rates
                else 0.0
            ),
            "total_errors": sum(self.error_counts.values()),
            "performance_metrics": {},
        }

        for task_name, metrics in self.performance_metrics.items():
            if metrics and not task_name.endswith(("_avg", "_max", "_min")):
                summary["performance_metrics"][task_name] = {
                    "average_duration": sum(metrics) / len(metrics),
                    "max_duration": max(metrics),
                    "min_duration": min(metrics),
                    "execution_count": len(metrics),
                }

        return summary

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.performance_metrics.clear()
        self.error_counts.clear()
        self.success_rates.clear()
        self.logger.info("Reset all performance metrics")
