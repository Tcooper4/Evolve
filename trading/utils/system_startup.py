"""System startup utilities for the trading platform."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import streamlit as st

# from core.agents.router import AgentRouter  # Moved to archive
from trading.agents.updater import ModelUpdater
from trading.config import config
from trading.llm.llm_interface import LLMInterface
from trading.memory.performance_memory import PerformanceMemory
from trading.utils.auto_repair import auto_repair
from trading.utils.diagnostics import diagnostics
from trading.utils.error_logger import error_logger

logger = logging.getLogger(__name__)


class StartupMonitor:
    """Monitor and log system startup process."""

    def __init__(self):
        self.start_time = time.time()
        self.init_results = {}
        self.dependencies = {}
        self.startup_log = []

    def log_init_step(
        self, component: str, success: bool, duration: float, details: Optional[str] = None, error: Optional[str] = None
    ):
        """Log initialization step result."""
        step_result = {
            "component": component,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now(),
            "details": details,
            "error": error,
        }

        self.init_results[component] = step_result
        self.startup_log.append(step_result)

        if success:
            logger.info(f"✓ {component} initialized successfully in {duration:.3f}s")
            if details:
                logger.debug(f"  {component} details: {details}")
        else:
            logger.error(f"✗ {component} initialization failed in {duration:.3f}s: {error}")

    def check_dependency(self, name: str, check_func, description: str = "") -> bool:
        """Check if a dependency is available."""
        start_time = time.time()
        try:
            result = check_func()
            duration = time.time() - start_time
            self.dependencies[name] = {"available": True, "duration": duration, "description": description}
            logger.info(f"✓ Dependency {name} available ({duration:.3f}s)")
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.dependencies[name] = {
                "available": False,
                "duration": duration,
                "description": description,
                "error": str(e),
            }
            logger.error(f"✗ Dependency {name} unavailable ({duration:.3f}s): {e}")
            return False

    def get_startup_summary(self) -> Dict[str, Any]:
        """Get comprehensive startup summary."""
        total_time = time.time() - self.start_time
        successful_inits = sum(1 for r in self.init_results.values() if r["success"])
        total_inits = len(self.init_results)

        return {
            "total_startup_time": total_time,
            "components_initialized": successful_inits,
            "total_components": total_inits,
            "success_rate": successful_inits / total_inits if total_inits > 0 else 0,
            "dependencies_available": sum(1 for d in self.dependencies.values() if d["available"]),
            "total_dependencies": len(self.dependencies),
            "init_results": self.init_results,
            "dependencies": self.dependencies,
            "startup_log": self.startup_log,
            "system_info": self._get_system_info(),
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name,
        }


def run_system_checks() -> Dict[str, Any]:
    """Run system repair and health checks.

    Returns:
        Dict containing repair and health check results
    """
    monitor = StartupMonitor()

    # Check dependencies
    monitor.check_dependency("config", lambda: config.get("model_dir"), "Configuration system")
    monitor.check_dependency("logging", lambda: logging.getLogger().handlers, "Logging system")
    monitor.check_dependency("streamlit", lambda: st.session_state, "Streamlit session")

    # Run repair and health checks
    repair_start = time.time()
    repair_result = auto_repair.run_repair()
    repair_duration = time.time() - repair_start

    health_start = time.time()
    health_result = diagnostics.run_health_check()
    health_duration = time.time() - health_start

    # Log results
    monitor.log_init_step(
        "system_repair",
        repair_result["status"] == "success",
        repair_duration,
        f"Repaired {repair_result.get('fixed_issues', 0)} issues",
    )

    monitor.log_init_step(
        "health_check",
        health_result["status"] == "healthy",
        health_duration,
        f"Health score: {health_result.get('score', 0)}",
    )

    # Log any issues
    if repair_result["status"] != "success":
        error_logger.log_error("System repair required", context=repair_result)

    if health_result["status"] != "healthy":
        error_logger.log_error("Health check failed", context=health_result)

    results = {"repair": repair_result, "health": health_result, "startup_summary": monitor.get_startup_summary()}

    # Log startup summary
    summary = monitor.get_startup_summary()
    logger.info(f"System startup completed in {summary['total_startup_time']:.3f}s")
    logger.info(f"Components: {summary['components_initialized']}/{summary['total_components']} successful")
    logger.info(f"Dependencies: {summary['dependencies_available']}/{summary['total_dependencies']} available")

    return results


def initialize_components() -> Dict[str, Any]:
    """Initialize system components with error handling and detailed logging.

    Returns:
        Dict containing initialization results
    """
    monitor = StartupMonitor()

    results = {"llm": None, "router": None, "updater": None, "memory": None, "errors": [], "startup_summary": None}

    # Initialize LLM
    llm_start = time.time()
    try:
        results["llm"] = LLMInterface()
        st.session_state.llm = results["llm"]
        llm_duration = time.time() - llm_start
        monitor.log_init_step(
            "LLMInterface", True, llm_duration, f"Provider: {getattr(results['llm'], 'provider', 'unknown')}"
        )
    except Exception as e:
        llm_duration = time.time() - llm_start
        error_msg = f"Failed to initialize LLM: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
        monitor.log_init_step("LLMInterface", False, llm_duration, error=error_msg)

    # Initialize Router (commented out)
    router_start = time.time()
    try:
        # results["router"] = AgentRouter()  # Commented out as AgentRouter is moved to archive
        # st.session_state.router = results["router"]
        router_duration = time.time() - router_start
        monitor.log_init_step("AgentRouter", True, router_duration, "Router initialization skipped (moved to archive)")
    except Exception as e:
        router_duration = time.time() - router_start
        error_msg = f"Failed to initialize AgentRouter: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
        monitor.log_init_step("AgentRouter", False, router_duration, error=error_msg)

    # Initialize ModelUpdater
    updater_start = time.time()
    try:
        results["updater"] = ModelUpdater()
        st.session_state.updater = results["updater"]
        updater_duration = time.time() - updater_start
        monitor.log_init_step(
            "ModelUpdater",
            True,
            updater_duration,
            f"Update interval: {getattr(results['updater'], 'update_interval', 'unknown')}",
        )
    except Exception as e:
        updater_duration = time.time() - updater_start
        error_msg = f"Failed to initialize ModelUpdater: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
        monitor.log_init_step("ModelUpdater", False, updater_duration, error=error_msg)

    # Initialize PerformanceMemory
    memory_start = time.time()
    try:
        results["memory"] = PerformanceMemory()
        st.session_state.memory = results["memory"]
        memory_duration = time.time() - memory_start
        monitor.log_init_step(
            "PerformanceMemory", True, memory_duration, f"Memory size: {getattr(results['memory'], 'size', 'unknown')}"
        )
    except Exception as e:
        memory_duration = time.time() - memory_start
        error_msg = f"Failed to initialize PerformanceMemory: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
        monitor.log_init_step("PerformanceMemory", False, memory_duration, error=error_msg)

    # Add startup summary to results
    results["startup_summary"] = monitor.get_startup_summary()

    # Log final summary
    summary = monitor.get_startup_summary()
    logger.info(f"Component initialization completed in {summary['total_startup_time']:.3f}s")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")

    return results


def get_system_status() -> Dict[str, Any]:
    """Get current system status with enhanced information.

    Returns:
        Dict containing system status information
    """
    # Get basic status
    status = {
        "health_status": st.session_state.get("health_status", "unknown"),
        "repair_status": st.session_state.get("repair_status", "unknown"),
        "llm_provider": st.session_state.get("llm_provider", "unknown"),
        "model_count": len(list(Path(config.get("model_dir", "trading/models")).glob("*.pkl"))),
        "last_check": st.session_state.get("last_system_check", None),
        "startup_time": st.session_state.get("startup_time", None),
        "component_status": {},
    }

    # Check component status
    components = ["llm", "updater", "memory"]
    for component in components:
        if component in st.session_state:
            status["component_status"][component] = "active"
        else:
            status["component_status"][component] = "inactive"

    # Add system resource info
    try:
        status["system_resources"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except Exception as e:
        logger.warning(f"Could not get system resources: {e}")
        status["system_resources"] = "unavailable"

    return status


def clear_session_state() -> None:
    """Clear all session state variables."""
    logger.info("Clearing session state")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.info("Session state cleared")


def get_startup_performance() -> Dict[str, Any]:
    """Get startup performance metrics.

    Returns:
        Dict containing startup performance information
    """
    startup_summary = st.session_state.get("startup_summary", {})

    if not startup_summary:
        return {"status": "no_startup_data"}

    return {
        "total_time": startup_summary.get("total_startup_time", 0),
        "success_rate": startup_summary.get("success_rate", 0),
        "components": startup_summary.get("components_initialized", 0),
        "dependencies": startup_summary.get("dependencies_available", 0),
        "system_info": startup_summary.get("system_info", {}),
        "init_details": startup_summary.get("init_results", {}),
    }
