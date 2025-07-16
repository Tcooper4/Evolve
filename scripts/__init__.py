"""
Scripts module for application management and utilities.
Refactored to use submodules for better organization.
"""

# Import from submodules to reduce initial load time
from .forecast_registry import ForecastRegistry
from .management_registry import ManagementRegistry
from .strategy_registry import StrategyRegistry

# Legacy imports for backward compatibility
try:
    from .manage import ApplicationManager
    from .manage_api import APIManager
    from .manage_backup import BackupManager
    from .manage_config import ConfigManager
    from .manage_dashboard import DashboardManager
    from .manage_data import DataManager
    from .manage_data_quality import DataQualityManager
    from .manage_db import DBManager
    from .manage_debug import DebugManager
    from .manage_deps import DepsManager
    from .manage_docs import DocManager
    from .manage_env import EnvManager
    from .manage_health import HealthManager
    from .manage_incident import IncidentManager
    from .manage_logs import LogManager
    from .manage_ml import MLManager
    from .manage_model import ModelManager
    from .manage_monitor import MonitorManager
    from .manage_performance import PerformanceManager
    from .manage_pipeline import PipelineManager
    from .manage_recovery import RecoveryManager
    from .manage_security import SecurityManager
except ImportError as e:
    # Log warning but don't fail - allows for partial imports
    import logging

    logging.warning(f"Some management modules could not be imported: {e}")

__all__ = [
    # Core registry modules
    "ForecastRegistry",
    "StrategyRegistry",
    "ManagementRegistry",
    # Legacy management classes (if available)
    "ApplicationManager",
    "MLManager",
    "PerformanceManager",
    "PipelineManager",
    "ModelManager",
    "LogManager",
    "EnvManager",
    "DocManager",
    "MonitorManager",
    "RecoveryManager",
    "IncidentManager",
    "HealthManager",
    "DBManager",
    "DataManager",
    "DataQualityManager",
    "DashboardManager",
    "ConfigManager",
    "BackupManager",
    "APIManager",
    "DebugManager",
    "DepsManager",
    "SecurityManager",
]
