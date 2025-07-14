"""
System Runner Module

This module provides system initialization and management functions.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from core.session_utils import safe_session_set, update_last_updated

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from core.session_utils import (
    display_system_status,
    safe_session_set,
    update_last_updated,
)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_system_initialization() -> Dict[str, Any]:
    """Initialize core system components.

    Returns:
        Dictionary containing initialization status for each module
    """
    logger.info("🔧 Starting system initialization...")

    module_status = {}

    # Initialize data management
    try:
        from trading.data.data_loader import DataLoader

        logger.info("📊 Initializing data management...")
        data_loader = DataLoader()
        safe_session_set("data_loader", data_loader)
        module_status["data_management"] = "SUCCESS"
        logger.info("✅ Data management initialized successfully")
    except Exception as e:
        logger.error(f"❌ Data management initialization failed: {e}")
        module_status["data_management"] = "FAILED"

    # Initialize market analysis
    try:
        from trading.market.market_analyzer import MarketAnalyzer

        logger.info("📈 Initializing market analysis...")
        market_analyzer = MarketAnalyzer()
        safe_session_set("market_analyzer", market_analyzer)
        module_status["market_analysis"] = "SUCCESS"
        logger.info("✅ Market analysis initialized successfully")
    except Exception as e:
        logger.error(f"❌ Market analysis initialization failed: {e}")
        module_status["market_analysis"] = "FAILED"

    # Initialize strategy engine
    try:
        from trading.strategies.strategy_engine import StrategyEngine

        logger.info("🎯 Initializing strategy engine...")
        strategy_engine = StrategyEngine()
        safe_session_set("strategy_engine", strategy_engine)
        module_status["strategy_engine"] = "SUCCESS"
        logger.info("✅ Strategy engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Strategy engine initialization failed: {e}")
        module_status["strategy_engine"] = "FAILED"

    # Initialize optimization
    try:
        from trading.optimization.optimizer_factory import OptimizerFactory

        logger.info("⚡ Initializing optimization...")
        optimizer_factory = OptimizerFactory()
        safe_session_set("optimizer_factory", optimizer_factory)
        module_status["optimization"] = "SUCCESS"
        logger.info("✅ Optimization initialized successfully")
    except Exception as e:
        logger.error(f"❌ Optimization initialization failed: {e}")
        module_status["optimization"] = "FAILED"

    # Log final status
    success_count = sum(1 for status in module_status.values() if status == "SUCCESS")
    total_count = len(module_status)

    logger.info(
        f"🎯 System initialization finished: {success_count}/{total_count} modules successful"
    )

    return module_status


def run_agentic_routing() -> Optional[str]:
    """Initialize agentic routing system.

    Returns:
        String indicating success or None if failed
    """
    try:
        from core.capability_router import CapabilityRouter

        logger.info("🤖 Initializing agentic routing...")
        capability_router = CapabilityRouter()

        # Store in session state
        safe_session_set("capability_router", capability_router)

        logger.info("✅ Agentic routing initialized successfully")
        return "SUCCESS"

    except Exception as e:
        logger.error(f"❌ Agentic routing initialization failed: {e}")
        return None


def run_portfolio_management() -> bool:
    """Initialize portfolio management system.

    Returns:
        True if successful, False otherwise
    """
    try:
        from trading.portfolio.portfolio_manager import PortfolioManager

        logger.info("📊 Initializing portfolio management...")
        portfolio_manager = PortfolioManager()

        # Store in session state
        safe_session_set("portfolio_manager", portfolio_manager)

        logger.info("✅ Portfolio management initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Portfolio management initialization failed: {e}")
        return False


def run_performance_tracking() -> bool:
    """Initialize performance tracking system.

    Returns:
        True if successful, False otherwise
    """
    try:
        from trading.optimization.performance_logger import PerformanceLogger

        logger.info("📈 Initializing performance tracking...")
        performance_logger = PerformanceLogger()

        # Store in session state
        safe_session_set("performance_logger", performance_logger)

        logger.info("✅ Performance tracking initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Performance tracking initialization failed: {e}")
        return False


def run_strategy_logging() -> bool:
    """Initialize strategy logging system.

    Returns:
        True if successful, False otherwise
    """
    try:
        from trading.memory.strategy_logger import StrategyLogger

        logger.info("📝 Initializing strategy logging...")
        strategy_logger = StrategyLogger()

        # Store in session state
        safe_session_set("strategy_logger", strategy_logger)

        logger.info("✅ Strategy logging initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy logging initialization failed: {e}")
        return False


def run_model_monitoring() -> bool:
    """Initialize model monitoring system.

    Returns:
        True if successful, False otherwise
    """
    try:
        from trading.memory.model_monitor import ModelMonitor

        logger.info("🔍 Initializing model monitoring...")
        model_monitor = ModelMonitor()

        # Store in session state
        safe_session_set("model_monitor", model_monitor)

        logger.info("✅ Model monitoring initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Model monitoring initialization failed: {e}")
        return False


def run_complete_system() -> Dict[str, Any]:
    """Run the complete system initialization and return status.

    Returns:
        Dictionary containing complete system status
    """
    logger.info("🚀 Starting complete system initialization...")

    # Initialize core system
    module_status = run_system_initialization()

    # Initialize additional components
    additional_status = {}

    # Agentic routing
    agentic_response = run_agentic_routing()
    additional_status["agentic_routing"] = "SUCCESS" if agentic_response else "FAILED"

    # Portfolio management
    portfolio_success = run_portfolio_management()
    additional_status["portfolio_management"] = (
        "SUCCESS" if portfolio_success else "FAILED"
    )

    # Performance tracking
    performance_success = run_performance_tracking()
    additional_status["performance_tracking"] = (
        "SUCCESS" if performance_success else "FAILED"
    )

    # Strategy logging
    strategy_success = run_strategy_logging()
    additional_status["strategy_logging"] = "SUCCESS" if strategy_success else "FAILED"

    # Model monitoring
    model_success = run_model_monitoring()
    additional_status["model_monitoring"] = "SUCCESS" if model_success else "FAILED"

    # Combine all status
    complete_status = {**module_status, **additional_status}

    # Log final status
    success_count = sum(1 for status in complete_status.values() if status == "SUCCESS")
    total_count = len(complete_status)

    logger.info(
        f"🎯 Complete system initialization finished: {success_count}/{total_count} components successful"
    )

    return complete_status


def display_system_status(module_status: Dict[str, Any]) -> dict:
    """Display system status information. Returns status dict."""
    from core.session_utils import display_system_status as display_status

    display_status(module_status)
    return {"status": "system_status_displayed"}


def get_system_health() -> Dict[str, Any]:
    """Get overall system health status.

    Returns:
        Dictionary containing system health information
    """
    try:
        # Get module status
        module_status = run_system_initialization()

        # Calculate health metrics
        success_count = sum(
            1 for status in module_status.values() if status == "SUCCESS"
        )
        total_count = len(module_status)
        health_percentage = (
            (success_count / total_count) * 100 if total_count > 0 else 0
        )

        # Determine overall health status
        if health_percentage >= 90:
            health_status = "excellent"
        elif health_percentage >= 75:
            health_status = "good"
        elif health_percentage >= 50:
            health_status = "fair"
        else:
            health_status = "poor"

        return {
            "overall_status": health_status,
            "health_percentage": health_percentage,
            "successful_modules": success_count,
            "total_modules": total_count,
            "module_status": module_status,
            "timestamp": update_last_updated(),
        }

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "overall_status": "error",
            "health_percentage": 0,
            "successful_modules": 0,
            "total_modules": 0,
            "module_status": {},
            "error": str(e),
        }


if __name__ == "__main__":
    # Run complete system initialization when executed directly
    status = run_complete_system()
    logger.info(f"System initialization completed: {status}")
