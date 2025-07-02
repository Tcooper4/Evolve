"""
Core Trading System Components

This module provides the core components for the trading system including
performance tracking, agent management, and fundamental trading operations.
"""

from .performance import PerformanceTracker, PerformanceMetrics
from .agents import AgentManager, AgentStatus

__all__ = [
    'PerformanceTracker',
    'PerformanceMetrics', 
    'AgentManager',
    'AgentStatus'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Core Trading System Components" 