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
    from trading.manage import ApplicationManager
    from trading.manage_api import APIManager
    from trading.manage_backup import BackupManager
    from trading.manage_config import ConfigManager
    from trading.manage_dashboard import DashboardManager
    from trading.manage_data import DataManager
    from trading.manage_data_quality import DataQualityManager
    from trading.manage_db import DBManager
    from trading.manage_debug import DebugManager
    from trading.manage_deps import DepsManager
    from trading.manage_docs import DocManager
    from trading.manage_env import EnvManager
    from trading.manage_health import HealthManager
    from trading.manage_incident import IncidentManager
    from trading.manage_logs import LogManager
    from trading.manage_ml import MLManager
    from trading.manage_model import ModelManager
    from trading.manage_monitor import MonitorManager
    from trading.manage_performance import PerformanceManager
    from trading.manage_pipeline import PipelineManager
    from trading.manage_recovery import RecoveryManager
    from trading.manage_security import SecurityManager
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
