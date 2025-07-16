"""
Core module for the Evolve trading system.
This module provides core functionality including task orchestration.
"""

try:
    from .orchestrator.task_orchestrator import TaskOrchestrator
    from .orchestrator.task_executor import TaskExecutor
    from .orchestrator.task_scheduler import TaskScheduler
    from .orchestrator.task_monitor import TaskMonitor
    from .orchestrator.task_conditions import TaskConditions
    from .orchestrator.task_providers import TaskProviders
    from .orchestrator.task_models import TaskModels
    
    __all__ = [
        "TaskOrchestrator",
        "TaskExecutor", 
        "TaskScheduler",
        "TaskMonitor",
        "TaskConditions",
        "TaskProviders",
        "TaskModels"
    ]
except ImportError as e:
    # Fallback if orchestrator modules are not available
    __all__ = [] 