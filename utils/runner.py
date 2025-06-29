"""
System runner utilities for initializing and managing the trading system.

This module provides functions for initializing system modules and managing
the overall system state.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import importlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from core.session_utils import (
    initialize_session_state, 
    initialize_system_modules, 
    display_system_status,
    safe_session_get,
    safe_session_set,
    update_last_updated
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_system_initialization() -> Dict[str, Any]:
    """Initialize the entire trading system.
    
    Returns:
        Dictionary containing initialization status for all modules
    """
    logger.info("🚀 Starting system initialization...")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize system modules
    module_status = initialize_system_modules()
    
    # Update last updated timestamp
    update_last_updated()
    
    # Log initialization results
    success_count = sum(1 for status in module_status.values() if status == 'SUCCESS')
    total_count = len(module_status)
    
    if success_count == total_count:
        logger.info(f"✅ System initialization completed successfully ({success_count}/{total_count} modules)")
    elif success_count > 0:
        logger.warning(f"⚠️ Partial system initialization ({success_count}/{total_count} modules)")
    else:
        logger.error(f"❌ System initialization failed ({success_count}/{total_count} modules)")
    
    return module_status


def run_agentic_routing() -> Optional[str]:
    """Initialize and run agentic prompt routing.
    
    Returns:
        Response from agentic routing or None if failed
    """
    try:
        from trading.agents.prompt_router_agent import PromptRouterAgent
        
        logger.info("🤖 Initializing agentic prompt routing...")
        prompt_router = PromptRouterAgent()
        
        # Test the router with a simple query
        test_response = prompt_router.route_prompt("What is the system status?")
        
        if test_response:
            logger.info("✅ Agentic routing initialized successfully")
            return test_response
        else:
            logger.warning("⚠️ Agentic routing returned no response")
            return None
            
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
        safe_session_set('portfolio_manager', portfolio_manager)
        
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
        safe_session_set('performance_logger', performance_logger)
        
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
        safe_session_set('strategy_logger', strategy_logger)
        
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
        safe_session_set('model_monitor', model_monitor)
        
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
    additional_status['agentic_routing'] = 'SUCCESS' if agentic_response else 'FAILED'
    
    # Portfolio management
    portfolio_success = run_portfolio_management()
    additional_status['portfolio_management'] = 'SUCCESS' if portfolio_success else 'FAILED'
    
    # Performance tracking
    performance_success = run_performance_tracking()
    additional_status['performance_tracking'] = 'SUCCESS' if performance_success else 'FAILED'
    
    # Strategy logging
    strategy_success = run_strategy_logging()
    additional_status['strategy_logging'] = 'SUCCESS' if strategy_success else 'FAILED'
    
    # Model monitoring
    model_success = run_model_monitoring()
    additional_status['model_monitoring'] = 'SUCCESS' if model_success else 'FAILED'
    
    # Combine all status
    complete_status = {**module_status, **additional_status}
    
    # Log final status
    success_count = sum(1 for status in complete_status.values() if status == 'SUCCESS')
    total_count = len(complete_status)
    
    logger.info(f"🎯 Complete system initialization finished: {success_count}/{total_count} components successful")
    
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
        success_count = sum(1 for status in module_status.values() if status == 'SUCCESS')
        total_count = len(module_status)
        health_percentage = (success_count / total_count) * 100 if total_count > 0 else 0
        
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
            "timestamp": update_last_updated()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "overall_status": "error",
            "health_percentage": 0,
            "successful_modules": 0,
            "total_modules": 0,
            "module_status": {},
            "error": str(e)
        }


if __name__ == "__main__":
    # Run complete system initialization when executed directly
    status = run_complete_system()
    print(f"System initialization completed: {status}") 