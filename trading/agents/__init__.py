"""
Financial Forecasting Agent System.

This package provides a modular system of agents for financial forecasting,
including goal planning, self-improvement, and task management.
"""

from trading.base_agent import BaseAgent, AgentResult
from trading.agent_manager import AgentManager
from trading.goal_planner import GoalPlanner
from trading.self_improving_agent import SelfImprovingAgent
from trading.task_memory import Task, TaskMemory, TaskStatus
from trading.task_dashboard import TaskDashboard, run_dashboard
from trading.router import Router

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
