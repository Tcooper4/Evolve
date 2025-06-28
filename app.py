"""
Main Streamlit Application

This is the entry point for the Agentic Forecasting System dashboard.
Provides a web interface for financial forecasting and trading strategy analysis.
"""

import streamlit as st
import importlib
from typing import Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Evolve - Agentic Forecasting System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation configuration
PAGES: Dict[str, str] = {
    "Home": "home",
    "Forecasting": "forecast",
    "Performance Tracker": "performance_tracker",
    "Strategy": "strategy",
    "System Scorecard": "5_📊_System_Scorecard",
    "Settings": "settings"
}


def initialize_system_modules() -> Dict[str, Any]:
    """
    Initialize all system modules and return their status.
    
    Returns:
        Dictionary with module initialization status
    """
    module_status = {
        "goal_status": {"available": False, "error": None},
        "optimizer_consolidator": {"available": False, "error": None},
        "market_analysis": {"available": False, "error": None},
        "data_pipeline": {"available": False, "error": None},
        "data_validation": {"available": False, "error": None}
    }
    
    # Initialize Goal Status Module
    try:
        from trading.memory.goals.status import get_status_summary, update_goal_progress
        module_status["goal_status"]["available"] = True
        module_status["goal_status"]["functions"] = {
            "get_status_summary": get_status_summary,
            "update_goal_progress": update_goal_progress
        }
        logger.info("[SUCCESS] Goal status module initialized")
    except Exception as e:
        module_status["goal_status"]["error"] = str(e)
        logger.warning(f"[WARNING] Goal status module not available: {e}")
    
    # Initialize Optimizer Consolidator Module
    try:
        from optimizers.consolidator import OptimizerConsolidator, run_optimizer_consolidation
        module_status["optimizer_consolidator"]["available"] = True
        module_status["optimizer_consolidator"]["functions"] = {
            "OptimizerConsolidator": OptimizerConsolidator,
            "run_optimizer_consolidation": run_optimizer_consolidation
        }
        logger.info("[SUCCESS] Optimizer consolidator module initialized")
    except Exception as e:
        module_status["optimizer_consolidator"]["error"] = str(e)
        logger.warning(f"[WARNING] Optimizer consolidator module not available: {e}")
    
    # Initialize Market Analysis Module
    try:
        from src.analysis.market_analysis import MarketAnalyzer, analyze_market_conditions
        module_status["market_analysis"]["available"] = True
        module_status["market_analysis"]["functions"] = {
            "MarketAnalyzer": MarketAnalyzer,
            "analyze_market_conditions": analyze_market_conditions
        }
        logger.info("[SUCCESS] Market analysis module initialized")
    except Exception as e:
        module_status["market_analysis"]["error"] = str(e)
        logger.warning(f"[WARNING] Market analysis module not available: {e}")
    
    # Initialize Data Pipeline Module
    try:
        from src.utils.data_pipeline import DataPipeline, run_data_pipeline
        module_status["data_pipeline"]["available"] = True
        module_status["data_pipeline"]["functions"] = {
            "DataPipeline": DataPipeline,
            "run_data_pipeline": run_data_pipeline
        }
        logger.info("[SUCCESS] Data pipeline module initialized")
    except Exception as e:
        module_status["data_pipeline"]["error"] = str(e)
        logger.warning(f"[WARNING] Data pipeline module not available: {e}")
    
    # Initialize Data Validation Module
    try:
        from src.utils.data_validation import DataValidator, validate_data_for_training, validate_data_for_forecasting
        module_status["data_validation"]["available"] = True
        module_status["data_validation"]["functions"] = {
            "DataValidator": DataValidator,
            "validate_data_for_training": validate_data_for_training,
            "validate_data_for_forecasting": validate_data_for_forecasting
        }
        logger.info("[SUCCESS] Data validation module initialized")
    except Exception as e:
        module_status["data_validation"]["error"] = str(e)
        logger.warning(f"[WARNING] Data validation module not available: {e}")
    
    return module_status


def display_system_status(module_status: Dict[str, Any]) -> None:
    """
    Display system status in the sidebar.
    
    Args:
        module_status: Dictionary with module status information
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    # Count available modules
    available_modules = sum(1 for module in module_status.values() if module["available"])
    total_modules = len(module_status)
    
    # Display overall status
    if available_modules == total_modules:
        st.sidebar.success(f"[SUCCESS] All modules available ({available_modules}/{total_modules})")
    elif available_modules > 0:
        st.sidebar.warning(f"[WARNING] {available_modules}/{total_modules} modules available")
    else:
        st.sidebar.error(f"[ERROR] No modules available ({available_modules}/{total_modules})")
    
    # Display individual module status
    with st.sidebar.expander("Module Details"):
        for module_name, status in module_status.items():
            if status["available"]:
                st.success(f"[SUCCESS] {module_name.replace('_', ' ').title()}")
            else:
                st.error(f"[ERROR] {module_name.replace('_', ' ').title()}")
                if status["error"]:
                    st.caption(f"Error: {status['error'][:50]}...")


def display_goal_status(module_status: Dict[str, Any]) -> None:
    """
    Display goal status in the sidebar.
    
    Args:
        module_status: Dictionary with module status information
    """
    if not module_status["goal_status"]["available"]:
        return
    
    try:
        get_status_summary = module_status["goal_status"]["functions"]["get_status_summary"]
        status_summary = get_status_summary()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Goal Status")
        
        # Display current status
        status_color = {
            "on_track": "success",
            "behind_schedule": "warning",
            "ahead_of_schedule": "info",
            "completed": "success",
            "not_started": "info"
        }.get(status_summary["current_status"].lower(), "info")
        
        getattr(st.sidebar, status_color)(f"Status: {status_summary['current_status']}")
        
        # Display progress
        if status_summary["progress"] is not None:
            st.sidebar.progress(status_summary["progress"])
            st.sidebar.caption(f"Progress: {status_summary['progress']:.1%}")
        
        # Display last updated
        if status_summary["last_updated"] != "Unknown":
            st.sidebar.caption(f"Updated: {status_summary['last_updated'][:10]}")
        
        # Display alerts if any
        if status_summary.get("alerts"):
            with st.sidebar.expander("[WARNING] Alerts"):
                for alert in status_summary["alerts"]:
                    st.warning(alert["message"])
        
    except Exception as e:
        st.sidebar.error(f"Error loading goal status: {str(e)}")


def load_page_module(page_name: str) -> Optional[Any]:
    """Dynamically load a page module.
    
    Args:
        page_name: Name of the page module to load
        
    Returns:
        Loaded module or None if loading fails
    """
    try:
        if page_name == "performance_tracker":
            from pages import performance_tracker
            return performance_tracker
        elif page_name == "forecast":
            from pages import forecast
            return forecast
        elif page_name == "strategy":
            from pages import strategy
            return strategy
        elif page_name == "5_📊_System_Scorecard":
            return importlib.import_module("pages.5_📊_System_Scorecard")
        elif page_name == "settings":
            from pages import settings
            return settings
        else:
            return None
    except ImportError as e:
        logger.error(f"Failed to import page module '{page_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading page '{page_name}': {e}")
        return None


def render_home_page(module_status: Dict[str, Any]) -> None:
    """Render the home page with welcome message and navigation guide."""
    st.title("🔮 Evolve: Agentic Forecasting System")
    st.markdown("""
    Welcome to **Evolve**, an autonomous financial forecasting and trading strategy platform!
    
    Use the sidebar to navigate through different features:
    
    - 📈 **Forecasting**: Generate and analyze market predictions using advanced ML models
    - 📊 **Performance Tracker**: Monitor model performance metrics and system health
    - 🎯 **Strategy**: View and manage trading strategies with backtesting results
    - 📋 **System Scorecard**: Check overall system health and performance indicators
    - ⚙️ **Settings**: Configure system parameters and preferences
    
    ### Key Features
    - **Multi-Model Forecasting**: LSTM, XGBoost, Prophet, ARIMA, and ensemble models
    - **Technical Analysis**: RSI, MACD, Bollinger Bands, and custom indicators
    - **Real-time Data**: Live market data integration and processing
    - **Interactive Dashboards**: Professional-grade visualizations and charts
    - **Risk Management**: Comprehensive backtesting and performance metrics
    - **Natural Language Interface**: Ask questions in plain English with QuantGPT
    - **Unified Access**: All features accessible through commands, UI, or prompts
    """)
    
    # Display module integration status
    st.subheader("🔧 System Integration Status")
    
    # Create columns for module status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Modules:**")
        for module_name in ["goal_status", "market_analysis", "data_validation"]:
            status = module_status[module_name]
            if status["available"]:
                st.success(f"[SUCCESS] {module_name.replace('_', ' ').title()}")
            else:
                st.error(f"[ERROR] {module_name.replace('_', ' ').title()}")
    
    with col2:
        st.markdown("**Utility Modules:**")
        for module_name in ["optimizer_consolidator", "data_pipeline"]:
            status = module_status[module_name]
            if status["available"]:
                st.success(f"[SUCCESS] {module_name.replace('_', ' ').title()}")
            else:
                st.error(f"[ERROR] {module_name.replace('_', ' ').title()}")
    
    # Display goal status if available
    if module_status["goal_status"]["available"]:
        try:
            get_status_summary = module_status["goal_status"]["functions"]["get_status_summary"]
            status_summary = get_status_summary()
            
            st.subheader("🎯 Current Goal Status")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", status_summary["current_status"])
            with col2:
                st.metric("Progress", f"{status_summary['progress']:.1%}")
            with col3:
                st.metric("Alerts", len(status_summary.get("alerts", [])))
            
            # Display recent recommendations
            if status_summary.get("recommendations"):
                st.markdown("**Recommendations:**")
                for rec in status_summary["recommendations"]:
                    st.info(f"💡 {rec}")
                    
        except Exception as e:
            st.warning(f"Could not load goal status: {str(e)}")


def main() -> None:
    """Main application entry point."""
    try:
        # Initialize system modules
        module_status = initialize_system_modules()
        
        # Sidebar navigation
        st.sidebar.title("🔮 Navigation")
        st.sidebar.markdown("---")
        
        selection = st.sidebar.radio(
            "🔍 Navigate",
            list(PAGES.keys()),
            index=0
        )
        
        selected_page = PAGES[selection]
        
        # Display system status in sidebar
        display_system_status(module_status)
        
        # Display goal status in sidebar
        display_goal_status(module_status)
        
        # Main content area
        if selection == "Home":
            render_home_page(module_status)
        else:
            # Load and execute page module
            page_module = load_page_module(selected_page)
            
            if page_module is not None:
                try:
                    if callable(page_module):
                        page_module()
                    else:
                        page_module.main()
                except AttributeError:
                    st.error(f"Page module '{selected_page}' does not have a main() function")
                    st.info("Please ensure the page module has a properly defined main() function.")
                except Exception as e:
                    st.error(f"Error executing page '{selected_page}': {str(e)}")
                    logger.error(f"Page execution error: {e}")
            else:
                st.error(f"Page '{selected_page}' not found or could not be loaded")
                st.info("Please ensure the page module exists and can be imported.")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        st.info("Please check the logs for more details.")


if __name__ == "__main__":
    main()
