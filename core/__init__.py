"""
Core module for Evolve Trading Platform.

This module contains core orchestration and task management functionality.
"""

# Export orchestrator components
try:
    from .orchestrator.task_orchestrator import TaskOrchestrator
    from .orchestrator.task_scheduler import TaskScheduler
    from .orchestrator.task_monitor import TaskMonitor
    from .orchestrator.task_executor import TaskExecutor
    from .orchestrator.task_conditions import TaskConditions
    
    __all__ = [
        "TaskOrchestrator",
        "TaskScheduler",
        "TaskMonitor",
        "TaskExecutor",
        "TaskConditions",
    ]
except ImportError as e:
    # If imports fail, export empty list to prevent import errors
    __all__ = []
