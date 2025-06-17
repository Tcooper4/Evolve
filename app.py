"""Main application entry point for the trading platform."""

# Standard library imports
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import streamlit as st
import yfinance as yf

# Local imports
from optimize.rsi_optimizer import optimize_rsi
from trading.strategies.rsi_signals import generate_rsi_signals
from utils.system_status import get_system_scorecard

def show_startup_banner():
    """Display system startup banner."""
    st.title("Trading Platform")
    st.markdown("---")
    
    # System info
    st.sidebar.markdown("### System Status")
    st.sidebar.markdown(f"Python: {platform.python_version()}")
    st.sidebar.markdown(f"Platform: {platform.platform()}")
    st.sidebar.markdown(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def initialize_system():
    """Initialize system components."""
    # Create necessary directories
    Path("memory/logs").mkdir(parents=True, exist_ok=True)
    Path("memory/strategy_settings/rsi").mkdir(parents=True, exist_ok=True)

def show_performance_log():
    """Show performance metrics in sidebar."""
    try:
        log_file = Path("memory/logs/performance_log.csv")
        if log_file.exists():
            df = pd.read_csv(log_file)
            
            st.sidebar.markdown("### Performance Metrics")
            st.sidebar.markdown(f"Total Entries: {len(df)}")
            st.sidebar.markdown(f"Avg Sharpe: {df['sharpe'].mean():.2f}")
            st.sidebar.markdown(f"Avg Accuracy: {df['accuracy'].mean():.2f}")
            
            if st.sidebar.button("View Full Log"):
                st.dataframe(df)
                
            if st.sidebar.button("Clear Log"):
                log_file.unlink()
                st.sidebar.success("Log cleared")
    except Exception as e:
        st.sidebar.error(f"Error loading performance log: {str(e)}")

def optimize_rsi_strategy(ticker: str):
    """Run RSI optimization for a ticker."""
    try:
        with st.spinner(f"Optimizing RSI strategy for {ticker}..."):
            # Get historical data
            df = yf.download(ticker, period="1y")
            
            # Run optimization
            optimal = optimize_rsi(df, ticker, generate_rsi_signals)
            
            # Display results
            st.success(f"RSI optimization complete for {ticker}")
            st.json(optimal)
            
            return optimal
    except Exception as e:
        st.error(f"Error optimizing RSI strategy: {str(e)}")
        return None

def show_rsi_optimizer():
    """Show RSI optimization panel in sidebar."""
    st.sidebar.markdown("### RSI Strategy Optimizer")
    
    # Get ticker input
    ticker = st.sidebar.text_input("Enter ticker symbol", "AAPL").upper()
    
    # Add optimization button
    if st.sidebar.button("Optimize RSI Strategy"):
        optimal = optimize_rsi_strategy(ticker)
        if optimal:
            st.sidebar.markdown("#### Current Settings")
            st.sidebar.json(optimal)

def show_sidebar_summary():
    """Show a summary of key metrics in the sidebar."""
    stats = get_system_scorecard()
    st.sidebar.title("üìà System Status")
    st.sidebar.metric("Sharpe (7d)", stats["sharpe_7d"])
    st.sidebar.metric("Win Rate (%)", stats["win_rate"])
    st.sidebar.metric("Goal Status", "‚úÖ" if stats["goal_status"].get("overall", False) else "‚ùå")

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="Trading Platform",
        page_icon="üìà",
        layout="wide"
    )
    
    # Initialize system
    initialize_system()
    
    # Show startup banner
    show_startup_banner()
    
    # Show performance log
    show_performance_log()
    
    # Show RSI optimizer
    show_rsi_optimizer()
    
    # Show sidebar summary
    show_sidebar_summary()
    
    # Main content area
    st.markdown("## Trading Dashboard")
    st.markdown("Use the sidebar to optimize RSI strategies and view performance metrics.")

if __name__ == "__main__":
    main()
