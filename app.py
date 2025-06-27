"""
Main Streamlit Application

This is the entry point for the Agentic Forecasting System dashboard.
Provides a web interface for financial forecasting and trading strategy analysis.
"""

import streamlit as st
import importlib
from typing import Dict, Any, Optional
import logging

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
    "Unified Interface": "unified_interface",
    "Forecasting": "forecast",
    "Performance Tracker": "performance_tracker",
    "Strategy": "strategy",
    "System Scorecard": "5_📊_System_Scorecard",
    "Settings": "settings"
}


def load_page_module(page_name: str) -> Optional[Any]:
    """Dynamically load a page module.
    
    Args:
        page_name: Name of the page module to load
        
    Returns:
        Loaded module or None if loading fails
    """
    try:
        if page_name == "unified_interface":
            from unified_interface import streamlit_ui
            return streamlit_ui
        elif page_name == "performance_tracker":
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


def render_home_page() -> None:
    """Render the home page with welcome message and navigation guide."""
    st.title("🔮 Evolve: Agentic Forecasting System")
    st.markdown("""
    Welcome to **Evolve**, an autonomous financial forecasting and trading strategy platform!
    
    Use the sidebar to navigate through different features:
    
    - 🔮 **Unified Interface**: Access all features through one comprehensive interface
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


def main() -> None:
    """Main application entry point."""
    try:
        # Sidebar navigation
        st.sidebar.title("🔮 Navigation")
        st.sidebar.markdown("---")
        
        selection = st.sidebar.radio(
            "🔍 Navigate",
            list(PAGES.keys()),
            index=0
        )
        
        selected_page = PAGES[selection]
        
        # Main content area
        if selection == "Home":
            render_home_page()
        elif selection == "Unified Interface":
            # Load and execute unified interface
            try:
                from unified_interface import streamlit_ui
                streamlit_ui()
            except ImportError as e:
                st.error(f"Unified interface not available: {e}")
                st.info("Please ensure the unified_interface.py file is present.")
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
