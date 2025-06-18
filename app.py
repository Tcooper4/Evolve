"""
Main Streamlit Application

This is the entry point for the Agentic Forecasting System dashboard.
"""

import streamlit as st
import importlib

# Page configuration
st.set_page_config(page_title="Agentic Forecasting", layout="wide")

# Navigation
PAGES = {
    "Home": "home",
    "Forecasting": "forecast",
    "Performance Tracker": "performance_tracker",
    "Strategy": "strategy",
    "System Scorecard": "5_📊_System_Scorecard",
    "Settings": "settings"
}

# Sidebar navigation
st.sidebar.title("🔮 Navigation")
selection = st.sidebar.radio("🔍 Navigate", list(PAGES.keys()))
selected_page = PAGES[selection]

# Main content area
if selection == "Home":
    st.title("🔮 Agentic Forecasting System")
    st.markdown("""
    Welcome to the Agentic Forecasting System! Use the sidebar to navigate through different features:
    
    - 📈 **Forecasting**: Generate and analyze market predictions
    - 📊 **Performance Tracker**: Monitor model performance metrics
    - 🎯 **Strategy**: View and manage trading strategies
    - 📋 **System Scorecard**: Check overall system health
    - ⚙️ **Settings**: Configure system parameters
    """)
else:
    # Dynamic view loader
    try:
        if selected_page == "performance_tracker":
            from pages import performance_tracker
            performance_tracker.main()
        elif selected_page == "forecast":
            from pages import forecast
            forecast.main()
        elif selected_page == "strategy":
            from pages import strategy
            strategy.main()
        elif selected_page == "5_📊_System_Scorecard":
            scorecard = importlib.import_module("pages.5_📊_System_Scorecard")
            scorecard.main()
        elif selected_page == "settings":
            from pages import settings
            settings.main()
        else:
            st.error(f"Page '{selected_page}' not found")
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.info("Please ensure the page module exists and has a main() function.")
