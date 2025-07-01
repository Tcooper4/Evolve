"""
Shared session management utilities for Streamlit applications.

This module consolidates common session state initialization and management
functions that are used across multiple pages and components.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def initialize_session_state() -> Dict[str, Any]:
    """Initialize Streamlit session state with core components.
    
    This function sets up all necessary session state variables with safe defaults.
    It should be called at the beginning of each page to ensure consistent state.
    
    Returns:
        Dictionary containing initialization status and metadata
    """
    try:
        # Core session state variables
        if 'last_updated' not in st.session_state:
            st.session_state.last_updated = datetime.now()
        
        # Date range defaults
        if 'start_date' not in st.session_state:
            st.session_state.start_date = datetime.now().date() - timedelta(days=30)
        if 'end_date' not in st.session_state:
            st.session_state.end_date = datetime.now().date()
        
        # Ticker and model selection
        if 'selected_ticker' not in st.session_state:
            st.session_state.selected_ticker = 'AAPL'
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'LSTM'
        
        # Data storage
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
        if 'market_analysis' not in st.session_state:
            st.session_state.market_analysis = None
        
        # Portfolio management
        if 'portfolio_manager' not in st.session_state:
            st.session_state.portfolio_manager = None
        
        # Performance tracking
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = None
        if 'strategy_history' not in st.session_state:
            st.session_state.strategy_history = None
        
        # Risk management
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = None
        
        # UI state
        if 'sidebar_expanded' not in st.session_state:
            st.session_state.sidebar_expanded = True
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        # API and external services
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = False
        if 'redis_connected' not in st.session_state:
            st.session_state.redis_connected = False
        
        logger.debug("Session state initialized successfully")
        
        return {
            'success': True,
            'message': 'Session state initialized successfully',
            'timestamp': datetime.now().isoformat(),
            'variables_initialized': len(st.session_state)
        }
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def initialize_system_modules() -> Dict[str, Any]:
    """Initialize system modules and return their status.
    
    Returns:
        Dictionary containing initialization status for each module
    """
    module_status = {}
    
    # Initialize all modules upfront to verify they're working
    
    try:
        # Initialize goal status module (core)
        from trading.memory.goals.status import GoalStatusTracker
        goal_tracker = GoalStatusTracker()
        goal_tracker.generate_summary()
        module_status['goal_status'] = 'SUCCESS'
        logger.info("[SUCCESS] Goal status module initialized")
    except Exception as e:
        module_status['goal_status'] = f'FAILED: {str(e)}'
        logger.error(f"[FAILED] Goal status module: {e}")
    
    try:
        # Initialize basic data validation (core)
        from src.utils.data_validation import DataValidator
        validator = DataValidator()
        module_status['data_validation'] = 'SUCCESS'
        logger.info("[SUCCESS] Data validation module initialized")
    except Exception as e:
        module_status['data_validation'] = f'FAILED: {str(e)}'
        logger.error(f"[FAILED] Data validation: {e}")
    
    try:
        # Initialize optimizer consolidator
        from trading.optimization.utils.consolidator import OptimizerConsolidator
        consolidator = OptimizerConsolidator()
        module_status['optimizer_consolidator'] = 'SUCCESS'
        logger.info("[SUCCESS] Optimizer consolidator module initialized")
    except Exception as e:
        module_status['optimizer_consolidator'] = f'FAILED: {str(e)}'
        logger.error(f"[FAILED] Optimizer consolidator: {e}")
    
    try:
        # Initialize market analysis
        from trading.market.market_analyzer import MarketAnalyzer
        market_analyzer = MarketAnalyzer()
        module_status['market_analysis'] = 'SUCCESS'
        logger.info("[SUCCESS] Market analysis module initialized")
    except Exception as e:
        module_status['market_analysis'] = f'FAILED: {str(e)}'
        logger.error(f"[FAILED] Market analysis: {e}")
    
    try:
        # Initialize data pipeline
        from src.utils.data_pipeline import DataPipeline
        data_pipeline = DataPipeline()
        module_status['data_pipeline'] = 'SUCCESS'
        logger.info("[SUCCESS] Data pipeline module initialized")
    except Exception as e:
        module_status['data_pipeline'] = f'FAILED: {str(e)}'
        logger.error(f"[FAILED] Data pipeline: {e}")
    
    return module_status

def display_system_status(module_status: Dict[str, Any]) -> Dict[str, Any]:
    """Display system status information in the Streamlit interface.
    
    Args:
        module_status: Dictionary containing module initialization status
        
    Returns:
        Dictionary containing display status and metadata
    """
    try:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 System Status")
        
        # Count successes and failures
        success_count = sum(1 for status in module_status.values() if status == 'SUCCESS')
        failed_count = sum(1 for status in module_status.values() if 'FAILED' in str(status))
        total_count = len(module_status)
        
        # Overall status
        if failed_count == 0:
            st.sidebar.success(f"✅ All systems operational ({success_count}/{total_count})")
        else:
            st.sidebar.warning(f"⚠️ System issues detected ({failed_count} failed, {success_count} operational)")
        
        # Detailed status
        with st.sidebar.expander("📊 Module Details"):
            for module, status in module_status.items():
                if status == 'SUCCESS':
                    st.success(f"✅ {module.replace('_', ' ').title()}")
                else:
                    st.error(f"❌ {module.replace('_', ' ').title()}: {status}")
        
        # Last updated timestamp
        if 'last_updated' in st.session_state:
            st.sidebar.caption(f"Last updated: {st.session_state.last_updated.strftime('%H:%M:%S')}")
        
        return {
            'success': True,
            'message': f'System status displayed: {success_count}/{total_count} operational',
            'timestamp': datetime.now().isoformat(),
            'success_count': success_count,
            'failed_count': failed_count,
            'total_count': total_count
        }
    except Exception as e:
        logger.error(f"Error displaying system status: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def safe_session_get(key: str, default: Any = None) -> Any:
    """Safely get a value from session state with logging.
    
    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist
        
    Returns:
        The session state value or default
    """
    if key not in st.session_state:
        logger.debug(f"Session state key '{key}' not found, using default: {default}")
        return default
    return st.session_state[key]

def safe_session_set(key: str, value: Any) -> Dict[str, Any]:
    """Safely set a value in session state with logging.
    
    Args:
        key: Session state key to set
        value: Value to store in session state
        
    Returns:
        Dictionary containing operation status and metadata
    """
    try:
        st.session_state[key] = value
        logger.debug(f"Session state key '{key}' set to: {value}")
        
        return {
            'success': True,
            'message': f'Session state key "{key}" set successfully',
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'value_type': type(value).__name__
        }
    except Exception as e:
        logger.error(f"Error setting session state key '{key}': {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'key': key
        }

def update_last_updated() -> Dict[str, Any]:
    """Update the last_updated timestamp in session state.
    
    Returns:
        Dictionary containing operation status and metadata
    """
    try:
        st.session_state.last_updated = datetime.now()
        logger.debug("Last updated timestamp refreshed")
        
        return {
            'success': True,
            'message': 'Last updated timestamp refreshed',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating last_updated timestamp: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }