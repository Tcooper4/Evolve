"""
Financial Forecasting Agent System.

This package provides a modular system of agents for financial forecasting,
including goal planning, self-improvement, and task management.
"""

from .base_agent import BaseAgent, AgentResult
from .agent_manager import AgentManager
from .goal_planner import GoalPlanner
from .self_improving_agent import SelfImprovingAgent
from .task_memory import Task, TaskMemory, TaskStatus
from .task_dashboard import TaskDashboard, run_dashboard
from .router import Router

__all__ = [
    'BaseAgent',
    'AgentResult',
    'AgentManager',
    'GoalPlanner',
    'SelfImprovingAgent',
    'Task',
    'TaskMemory',
    'TaskStatus',
    'TaskDashboard',
    'Router',
    'run_dashboard'
]
