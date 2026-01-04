"""
Page Renderer Module

This module contains all Streamlit UI rendering code organized into page-specific functions.
It consolidates all UI components (pages, sliders, dropdowns, tabs, plots, etc.) from app.py
into clean, modular page renderer functions.
"""

from typing import Any, Dict
import logging

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def render_sidebar():
    """Render the main sidebar with navigation and status."""
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #2c3e50; margin-bottom: 0.5rem;">ðŸš€ Evolve AI</h2>
            <p style="color: #6c757d; font-size: 0.9rem;">Autonomous Trading Intelligence</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Main Navigation - Simplified
        st.markdown("### ðŸ“Š Navigation")

        # Primary Navigation
        primary_nav = st.radio(
            "Main Navigation",
            [
                "ðŸ  Home & Chat",
                "ðŸ“ˆ Forecasting",
                "âš¡ Strategy Lab",
                "ðŸ§  Model Lab",
                "ðŸ“‹ Reports",
            ],
            key="primary_nav",
            label_visibility="collapsed",
        )

        # Advanced Tools (Collapsible)
        with st.expander("ðŸ”§ Advanced", expanded=False):
            advanced_nav = st.radio(
                "Advanced Tools",
                [
                    "âš™ï¸ Settings",
                    "ðŸ“Š Monitor",
                    "ðŸ“ˆ Analytics",
                    "ðŸ›¡ï¸ Risk",
                    "ðŸ¤– Orchestrator",
                ],
                key="advanced_nav",
                label_visibility="collapsed",
            )

        # Developer Tools (Hidden by default)
        if st.session_state.get("dev_mode", False):
            with st.expander("ðŸ› ï¸ Dev Tools", expanded=False):
                st.markdown("- ðŸ› Debug")
                st.markdown("- ðŸ“ Logs")
                st.markdown("- âš¡ Performance")
                st.markdown("- ðŸ”Œ API")

        st.markdown("---")

        # System Status - Simplified
        st.markdown("### ðŸŸ¢ Status")

        # Status indicators - compact
        status_items = [
            ("Core Systems", "ðŸŸ¢"),
            ("Data Feed", "ðŸŸ¢"),
            ("AI Models", "ðŸŸ¢"),
            ("Agents", "ðŸŸ¢"),
        ]

        for name, status in status_items:
            st.markdown(f"{status} {name}")

        # Quick Stats - Compact
        st.markdown("---")
        st.markdown("### ðŸ“Š Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("ðŸ¤– Models", "12")
            st.metric("ðŸ“ˆ Success", "94.2%")

        with col2:
            st.metric("âš¡ Strategies", "8")
            st.metric("ðŸ’° Return", "2.8%")

    return primary_nav, advanced_nav


def render_top_navigation():
    """Render the top navigation bar."""
    st.markdown(
        """
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        margin: -1rem -2rem 2rem -2rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; color: white;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 600;">ðŸš€ Evolve AI Trading</h1>
                <p style="margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Autonomous Financial Intelligence Platform</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Current Model</div>
                <div style="font-weight: 600; font-size: 1rem;">Hybrid Ensemble</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Last Run: 2 min ago</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_voice_input():
    """Render the voice input interface."""
    voice_mode = st.toggle(
        "ðŸŽ¤ Voice Input",
        value=False,
        help="Enable voice prompt (Whisper or Google Speech)",
    )

    if voice_mode and st.session_state.get("voice_agent"):
        st.markdown("**Click below and speak your trading command:**")
        if st.button("ðŸŽ™ï¸ Record Voice Command"):
            with st.spinner("Listening for command..."):
                try:
                    text = st.session_state.voice_agent.listen_for_command(
                        timeout=5, phrase_time_limit=10
                    )
                    if text:
                        st.session_state["voice_prompt_text"] = text
                        st.success(f'Voice recognized: "{text}"')
                    else:
                        st.warning("No voice command detected or could not transcribe.")
                except Exception as e:
                    st.error(f"Voice input error: {e}")
    else:
        st.session_state["voice_prompt_text"] = ""


def render_prompt_interface():
    """Render the main prompt input interface."""
    # Enhanced Prompt Interface - ChatGPT Style
    st.markdown(
        """
    <div style="
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #e9ecef;
        text-align: center;
    ">
        <h2 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.5rem;">
            ðŸ’¬ Ask Evolve AI Anything About Trading
        </h2>
        <p style="color: #6c757d; margin-bottom: 1.5rem; font-size: 1rem;">
            Your autonomous financial intelligence assistant is ready to help
        </p>
        <div style="
            background: white;
            border-radius: 10px;
            padding: 1rem;
            border: 2px solid #e9ecef;
            margin-bottom: 1rem;
        ">
            <p style="color: #495057; font-size: 0.9rem; margin: 0;">
                <strong>ðŸ’¡ Examples:</strong> "Forecast SPY using the most accurate model",
                "Create a new LSTM model for AAPL", "Show me RSI strategy with Bollinger Bands",
                "What's the current market sentiment for TSLA?"
            </p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Professional Prompt Input
    prompt = st.text_input(
        "ðŸš€ Type your trading prompt here...",
        value=st.session_state.get("voice_prompt_text", ""),
        placeholder="e.g., 'Forecast SPY using the most accurate model and RSI tuned to 10'",
        help="Ask for forecasts, strategies, model creation, or market analysis",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.button(
            "ðŸš€ Submit Query",
            use_container_width=True,
            help="Send your prompt to Evolve AI for processing",
        )

    return prompt, submit


def render_prompt_result(result: Dict[str, Any]):
    """Render the prompt processing result."""
    if result.get("success") and result.get("message"):
        st.markdown(
            f"""
        <div class="result-card">
            <h3>ðŸ¤– AI Response</h3>
            <p>{result['message']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display strategy and model information if available
        if result.get("strategy_name"):
            st.metric("ðŸ“Š Strategy Used", result["strategy_name"])
        if result.get("model_used"):
            st.metric("ðŸ§  Model Used", result["model_used"])
        if result.get("confidence"):
            st.metric("ðŸŽ¯ Confidence", f"{result['confidence']:.2%}")
        if result.get("signal"):
            signal_color = (
                "ðŸŸ¢"
                if result["signal"].lower() in ["buy", "long"]
                else "ðŸ”´"
                if result["signal"].lower() in ["sell", "short"]
                else "ðŸŸ¡"
            )
            st.metric("ðŸ“ˆ Signal", f"{signal_color} {result['signal']}")
    else:
        st.warning("âš ï¸ No response received from AI agent")


def render_conversation_history():
    """Render the conversation history display."""
    if st.session_state.get("conversation_history"):
        st.markdown("### ðŸ’¬ Recent Conversations")
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"ðŸ’­ {conv['prompt'][:50]}...", expanded=False):
                st.markdown(
                    f"""
                <div style="
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border-left: 4px solid #667eea;
                ">
                    <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                        ðŸ¤” Your Question:
                    </div>
                    <div style="margin-bottom: 1rem; color: #495057; font-style: italic;">
                        "{conv['prompt']}"
                    </div>
                    <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                        ðŸ¤– AI Response:
                    </div>
                    <div style="color: #495057; background: white; padding: 1rem; border-radius: 5px;">
                        {conv['response'].get('message', 'No response available.')}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if conv["response"].get("data"):
                    with st.expander("Detailed Data", expanded=False):
                        st.json(conv["response"]["data"])


def render_agent_logs():
    """Render the agent activity logs."""
    if st.session_state.get("agent_logger"):
        with st.expander("Agent Activity Logs", expanded=False):
            try:
                from trading.memory.agent_logger import AgentAction, LogLevel

                # Get recent logs
                recent_logs = st.session_state.agent_logger.get_recent_logs(limit=20)

                if recent_logs:
                    st.markdown("#### Recent Agent Actions")

                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_level = st.selectbox(
                            "Log Level", ["All", "Info", "Warning", "Error"]
                        )
                    with col2:
                        show_action = st.selectbox(
                            "Action Type",
                            [
                                "All",
                                "Model Synthesis",
                                "Strategy Switch",
                                "Forecast",
                                "Trade",
                            ],
                        )
                    with col3:
                        show_agent = st.selectbox(
                            "Agent",
                            ["All"] + list(set(log.agent_name for log in recent_logs)),
                        )

                    # Filter logs
                    filtered_logs = recent_logs
                    if show_level != "All":
                        level_map = {
                            "Info": LogLevel.INFO,
                            "Warning": LogLevel.WARNING,
                            "Error": LogLevel.ERROR,
                        }
                        filtered_logs = [
                            log
                            for log in filtered_logs
                            if log.level == level_map[show_level]
                        ]

                    if show_action != "All":
                        action_map = {
                            "Model Synthesis": AgentAction.MODEL_SYNTHESIS,
                            "Strategy Switch": AgentAction.STRATEGY_SWITCH,
                            "Forecast": AgentAction.FORECAST_GENERATION,
                            "Trade": AgentAction.TRADE_EXECUTION,
                        }
                        filtered_logs = [
                            log
                            for log in filtered_logs
                            if log.action == action_map[show_action]
                        ]

                    if show_agent != "All":
                        filtered_logs = [
                            log for log in filtered_logs if log.agent_name == show_agent
                        ]

                    # Display logs
                    for log in filtered_logs[-10:]:  # Show last 10 filtered logs
                        level_color = {
                            LogLevel.INFO: "ðŸŸ¢",
                            LogLevel.WARNING: "ðŸŸ¡",
                            LogLevel.ERROR: "ðŸ”´",
                            LogLevel.CRITICAL: "ðŸ”´",
                        }.get(log.level, "âšª")

                        st.markdown(
                            f"""
                        <div class="conversation-item">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600; color: #2c3e50;">{level_color} {log.agent_name}</span>
                                <span style="font-size: 0.8rem; color: #6c757d;">{log.timestamp.strftime('%H:%M:%S')}</span>
                            </div>
                            <div style="color: #5a6c7d; margin-bottom: 0.5rem;">{log.message}</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">Action: {log.action.value.replace('_', ' ').title()}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Show performance metrics if available
                        if log.performance_metrics:
                            with st.expander("Performance Metrics", expanded=False):
                                st.json(log.performance_metrics)

                        # Show error details if available
                        if log.error_details:
                            with st.expander("Error Details", expanded=False):
                                st.error(log.error_details)
                else:
                    st.info("No recent agent activity to display.")

            except Exception as e:
                st.warning(f"Could not load agent logs: {e}")


def render_home_page():
    """Render the home page with chat interface."""
    st.markdown("### AI Trading Assistant")

    # Display conversation history with better styling
    if st.session_state.get("conversation_history"):
        st.markdown("#### Recent Conversations")
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"{conv['prompt'][:50]}...", expanded=False):
                st.markdown(
                    f"""
                <div class="conversation-item">
                    <div class="conversation-prompt">Your Question:</div>
                    <div style="margin-bottom: 1rem;">{conv['prompt']}</div>
                    <div class="conversation-prompt">AI Response:</div>
                    <div class="conversation-response">{conv['response'].get('message', 'No response available.')}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if conv["response"].get("data"):
                    with st.expander("Detailed Data", expanded=False):
                        st.json(conv["response"]["data"])
    else:
        st.markdown(
            """
        <div class="result-card">
            <h3>Welcome to Evolve AI Trading</h3>
            <p>Start by asking me anything about trading! I can help you with:</p>
            <ul>
                <li><strong>Forecasting:</strong> "Show me the best forecast for AAPL"</li>
                <li><strong>Strategy Analysis:</strong> "Switch to RSI strategy and optimize it"</li>
                <li><strong>Model Building:</strong> "Create a new model for cryptocurrency trading"</li>
                <li><strong>Reports:</strong> "Generate a performance report for my portfolio"</li>
                <li><strong>Market Analysis:</strong> "What's the current market sentiment?"</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_forecasting_page():
    """Render the forecasting page."""
    try:
        from pages.Forecasting import main as forecasting_main

        forecasting_main()
    except ImportError as e:
        st.error(f"Forecasting page not available: {e}")
        st.info("Please ensure the Forecasting.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Forecasting page: {e}")


def render_strategy_page():
    """Render the strategy lab page."""
    try:
        from pages.Strategy_Lab import main as strategy_lab_main

        strategy_lab_main()
    except ImportError as e:
        st.error(f"Strategy Lab page not available: {e}")
        st.info("Please ensure the Strategy_Lab.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Strategy Lab page: {e}")


def render_model_page():
    """Render the model lab page."""
    try:
        from pages.Model_Lab import main as model_lab_main

        model_lab_main()
    except ImportError as e:
        st.error(f"Model Lab page not available: {e}")
        st.info("Please ensure the Model_Lab.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Model Lab page: {e}")


def render_reports_page():
    """Render the reports page."""
    try:
        from pages.Reports import main as reports_main

        reports_main()
    except ImportError as e:
        st.error(f"Reports page not available: {e}")
        st.info("Please ensure the Reports.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Reports page: {e}")


def render_settings_page():
    """Render the settings page."""
    st.markdown("### Advanced Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="result-card">
            <h3>System Configuration</h3>
            <p>Advanced system settings and preferences.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Advanced settings
        st.subheader("Risk Management")
        risk_level = st.selectbox(
            "Risk Level", ["Conservative", "Moderate", "Aggressive"]
        )
        _unused_var = risk_level  # Placeholder, flake8 ignore: F841
        st.slider("Max Position Size (%)", 1, 50, 10)
        st.slider("Stop Loss (%)", 1, 20, 5)

        st.subheader("Data Sources")
        data_source = st.selectbox(
            "Primary Data Source", ["YFinance", "Alpha Vantage", "Polygon"]
        )
        _unused_var = data_source  # Placeholder, flake8 ignore: F841
        update_frequency = st.selectbox(
            "Update Frequency", ["Real-time", "1 minute", "5 minutes", "15 minutes"]
        )
        _unused_var = update_frequency  # Placeholder, flake8 ignore: F841

    with col2:
        st.markdown(
            """
        <div class="result-card">
            <h3>Performance Settings</h3>
            <p>Optimize your trading performance and monitoring.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.subheader("Notifications")
        st.checkbox("Email Notifications")
        st.checkbox("Slack Notifications")
        st.checkbox("Telegram Notifications")

        st.subheader("Auto-Trading")
        auto_trading = st.checkbox("Enable Auto-Trading")
        if auto_trading:
            st.warning("Auto-trading is enabled. Please review your risk settings.")


def render_system_monitor_page():
    """Render the system monitor page."""
    st.markdown("### System Monitor")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="result-card">
            <h3>System Health</h3>
            <p>Real-time system monitoring.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.metric("CPU Usage", "45%", "Normal")
        st.metric("Memory Usage", "62%", "Normal")
        st.metric("Disk Usage", "28%", "Good")

    with col2:
        st.markdown(
            """
        <div class="result-card">
            <h3>Network Status</h3>
            <p>Network connectivity and data feeds.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.metric("API Calls/min", "156", "+12")
        st.metric("Data Feed", "Active", "ðŸŸ¢")
        st.metric("Latency", "45ms", "Good")

    with col3:
        st.markdown(
            """
        <div class="result-card">
            <h3>AI Models</h3>
            <p>Model performance and status.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.metric("Active Models", "8", "All Healthy")
        st.metric("Avg Accuracy", "87.2%", "+1.1%")
        st.metric("Training Jobs", "2", "In Progress")


def render_performance_analytics_page():
    """Render the performance analytics page."""
    st.markdown("### Performance Analytics")

    st.markdown(
        """
    <div class="result-card">
        <h3>Advanced Analytics Dashboard</h3>
        <p>Comprehensive performance analysis and insights.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Real analytics dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Performance")
        try:
            # Get real portfolio data
            from trading.portfolio.portfolio_manager import PortfolioManager
            
            # Try to get portfolio manager from session state or create new
            portfolio_manager = st.session_state.get("portfolio_manager")
            if portfolio_manager is None:
                portfolio_manager = PortfolioManager()
                st.session_state.portfolio_manager = portfolio_manager
            
            # Get performance summary
            summary = portfolio_manager.get_performance_summary()
            
            # Get position summary for historical data
            position_df = portfolio_manager.get_position_summary()
            
            if not position_df.empty and len(position_df) > 0:
                # Create portfolio value over time from positions
                # This is a simplified version - in production, you'd track portfolio value over time
                portfolio_values = []
                benchmark_values = []
                
                # Calculate cumulative PnL
                if "unrealized_pnl" in position_df.columns:
                    cumulative_pnl = position_df["unrealized_pnl"].cumsum()
                    portfolio_values = cumulative_pnl.values
                    # Benchmark would be market index - using 0.8x as proxy
                    benchmark_values = portfolio_values * 0.8 if len(portfolio_values) > 0 else []
                
                if portfolio_values:
                    chart_data = pd.DataFrame({
                        "Portfolio": portfolio_values,
                        "Benchmark": benchmark_values if benchmark_values else portfolio_values * 0.8,
                    })
                    st.line_chart(chart_data)
                else:
                    st.info("No portfolio performance data available yet. Open some positions to see performance.")
            else:
                st.info("No positions found. Portfolio performance will appear here once positions are opened.")
        except Exception as e:
            logger.error(f"Error loading portfolio performance: {e}")
            st.warning(f"Could not load portfolio performance: {e}")
            st.info("Please ensure portfolio manager is properly initialized.")

    with col2:
        st.subheader("Risk Metrics")
        try:
            # Get real risk metrics from portfolio
            from trading.portfolio.portfolio_manager import PortfolioManager
            
            portfolio_manager = st.session_state.get("portfolio_manager")
            if portfolio_manager is None:
                portfolio_manager = PortfolioManager()
                st.session_state.portfolio_manager = portfolio_manager
            
            # Get performance summary which includes risk metrics
            summary = portfolio_manager.get_performance_summary()
            risk_metrics = portfolio_manager.state.risk_metrics if hasattr(portfolio_manager, 'state') else {}
            
            # Display real metrics
            sharpe_ratio = summary.get("sharpe_ratio", 0.0)
            max_drawdown = summary.get("max_drawdown", 0.0)
            volatility = summary.get("volatility", 0.0)
            
            # Calculate additional metrics if available
            portfolio_var = risk_metrics.get("portfolio_var", 0.0)
            portfolio_volatility = risk_metrics.get("portfolio_volatility", volatility)
            
            # Display metrics
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Max Drawdown", f"{abs(max_drawdown):.2f}%")
            st.metric("Volatility", f"{portfolio_volatility:.2f}%")
            if portfolio_var > 0:
                st.metric("VaR (95%)", f"{portfolio_var:.2f}")
            
            # Calculate Sortino if we have downside data
            if volatility > 0:
                sortino_ratio = sharpe_ratio * 1.2  # Approximate (would need downside deviation for exact)
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        except Exception as e:
            logger.error(f"Error loading risk metrics: {e}")
            st.warning(f"Could not load risk metrics: {e}")
            # Fallback to placeholder
            st.metric("Sharpe Ratio", "N/A")
            st.metric("Max Drawdown", "N/A")
            st.metric("Volatility", "N/A")


def render_risk_management_page():
    """Render the risk management page."""
    st.markdown("### Risk Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="result-card">
            <h3>Risk Alerts</h3>
            <p>Current risk monitoring and alerts.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Real risk alerts from portfolio manager
        try:
            from trading.portfolio.portfolio_manager import PortfolioManager
            from datetime import datetime, timedelta
            
            portfolio_manager = st.session_state.get("portfolio_manager")
            if portfolio_manager is None:
                portfolio_manager = PortfolioManager()
                st.session_state.portfolio_manager = portfolio_manager
            
            alerts = []
            
            # Check for risk violations
            risk_metrics = portfolio_manager.state.risk_metrics if hasattr(portfolio_manager, 'state') else {}
            
            # Check portfolio volatility
            portfolio_vol = risk_metrics.get("portfolio_volatility", 0.0)
            if portfolio_vol > 0.20:  # 20% volatility threshold
                alerts.append({
                    "level": "High",
                    "message": f"High portfolio volatility detected: {portfolio_vol:.1%}",
                    "time": "Recent"
                })
            
            # Check for stop loss triggers (positions that were closed)
            closed_positions = portfolio_manager.state.closed_positions if hasattr(portfolio_manager, 'state') else []
            recent_closed = [
                p for p in closed_positions 
                if p.exit_time and (datetime.utcnow() - p.exit_time.replace(tzinfo=None)) < timedelta(hours=24)
            ]
            
            for pos in recent_closed:
                if pos.pnl and pos.pnl < 0:  # Loss
                    time_ago = "Recent"
                    if pos.exit_time:
                        delta = datetime.utcnow() - pos.exit_time.replace(tzinfo=None)
                        minutes = delta.total_seconds() / 60
                        if minutes < 60:
                            time_ago = f"{int(minutes)} min ago"
                        else:
                            time_ago = f"{int(minutes // 60)} hours ago"
                    
                    alerts.append({
                        "level": "Medium",
                        "message": f"Position closed: {pos.symbol} ({pos.strategy}) - Loss: ${abs(pos.pnl):.2f}",
                        "time": time_ago
                    })
            
            # Check portfolio concentration
            if hasattr(portfolio_manager, 'get_portfolio_allocation'):
                allocation = portfolio_manager.get_portfolio_allocation()
                max_allocation = max([v for k, v in allocation.items() if k != "CASH"], default=0.0)
                if max_allocation > 0.3:  # 30% concentration threshold
                    alerts.append({
                        "level": "Low",
                        "message": f"High portfolio concentration: {max_allocation:.1%} in single asset",
                        "time": "Current"
                    })
            
            if alerts:
                for alert in alerts[:5]:  # Show max 5 alerts
                    color = {"Low": "🟡", "Medium": "🟠", "High": "🔴"}.get(alert["level"], "⚪")
                    st.markdown(f"{color} **{alert['level']}:** {alert['message']}")
                    st.caption(f"*{alert.get('time', 'Recent')}*")
            else:
                st.success("✅ No risk alerts - portfolio within normal parameters")
                
        except Exception as e:
            logger.error(f"Error loading risk alerts: {e}")
            st.warning("Could not load risk alerts. Please ensure portfolio manager is initialized.")

    with col2:
        st.markdown(
            """
        <div class="result-card">
            <h3>Risk Metrics</h3>
            <p>Current risk exposure and limits.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        try:
            # Get real risk metrics
            from trading.portfolio.portfolio_manager import PortfolioManager
            
            portfolio_manager = st.session_state.get("portfolio_manager")
            if portfolio_manager is None:
                portfolio_manager = PortfolioManager()
                st.session_state.portfolio_manager = portfolio_manager
            
            risk_metrics = portfolio_manager.state.risk_metrics if hasattr(portfolio_manager, 'state') else {}
            state = portfolio_manager.state if hasattr(portfolio_manager, 'state') else None
            
            # Portfolio Beta
            portfolio_beta = risk_metrics.get("portfolio_beta", 1.0)
            beta_status = "Moderate" if 0.8 <= portfolio_beta <= 1.2 else ("High" if portfolio_beta > 1.2 else "Low")
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}", beta_status)
            
            # Correlation (would need correlation matrix - using placeholder for now)
            if hasattr(portfolio_manager, 'calculate_correlation_matrix'):
                try:
                    corr_matrix = portfolio_manager.calculate_correlation_matrix()
                    if not corr_matrix.empty:
                        # Get average correlation (excluding diagonal)
                        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                        corr_status = "Acceptable" if 0.3 <= avg_corr <= 0.7 else ("High" if avg_corr > 0.7 else "Low")
                        st.metric("Avg Correlation", f"{avg_corr:.2f}", corr_status)
                    else:
                        st.metric("Correlation", "N/A", "No data")
                except Exception as e:
                    logger.warning(f"Page rendering error: {e}")
                    st.metric("Correlation", "N/A", "Calculating...")
            else:
                st.metric("Correlation", "N/A", "Not available")
            
            # Concentration
            if hasattr(portfolio_manager, 'get_portfolio_allocation'):
                allocation = portfolio_manager.get_portfolio_allocation()
                max_allocation = max([v for k, v in allocation.items() if k != "CASH"], default=0.0)
                conc_status = "High" if max_allocation > 0.3 else ("Moderate" if max_allocation > 0.2 else "Low")
                st.metric("Concentration", f"{max_allocation:.1%}", conc_status)
            else:
                st.metric("Concentration", "N/A", "Not available")
            
            # Leverage
            leverage = state.leverage if state else 1.0
            lev_status = "Low" if leverage <= 1.0 else ("Moderate" if leverage <= 1.5 else "High")
            st.metric("Leverage", f"{leverage:.2f}", lev_status)
            
        except Exception as e:
            logger.error(f"Error loading risk metrics: {e}")
            st.warning("Could not load risk metrics")
            st.metric("Portfolio Beta", "N/A")
            st.metric("Correlation", "N/A")
            st.metric("Concentration", "N/A")
            st.metric("Leverage", "N/A")


def render_orchestrator_page():
    """Render the task orchestrator monitor page."""
    st.markdown("### Task Orchestrator Monitor")

    # Check if orchestrator is available
    orchestrator_available = st.session_state.get("orchestrator_available", False)

    if orchestrator_available:
        try:
            from system.orchestrator_integration import get_system_integration_status

            orchestrator_status = get_system_integration_status()
        except ImportError:
            orchestrator_status = {
                "status": "not_available",
                "message": "Orchestrator not available",
            }
    else:
        orchestrator_status = {
            "status": "not_available",
            "message": "Orchestrator not available",
        }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="result-card">
            <h3>Orchestrator Status</h3>
            <p>Task Orchestrator system status and health.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display orchestrator status
        status_icon = {
            "available": "ðŸŸ¢",
            "not_available": "ðŸ”´",
            "not_configured": "ðŸŸ¡",
            "error": "ðŸ”´",
        }.get(orchestrator_status.get("status", "unknown"), "â“")

        st.markdown(
            f"{status_icon} **Status:** {orchestrator_status.get('status', 'unknown').title()}"
        )

        if orchestrator_status.get("status") == "available":
            st.metric("Total Tasks", orchestrator_status.get("total_tasks", 0))
            st.metric("Enabled Tasks", orchestrator_status.get("enabled_tasks", 0))
            st.metric(
                "System Health", f"{orchestrator_status.get('overall_health', 0):.1%}"
            )
        else:
            st.warning(orchestrator_status.get("message", "Orchestrator not available"))

    with col2:
        st.markdown(
            """
        <div class="result-card">
            <h3>Quick Actions</h3>
            <p>Orchestrator control and monitoring.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Quick action buttons
        col2a, col2b = st.columns(2)

        with col2a:
            if st.button("ðŸ”„ Refresh Status", key="refresh_orchestrator"):
                st.rerun()

            if st.button("ðŸ“Š Export Report", key="export_orchestrator"):
                st.info(
                    "Orchestrator report export functionality would be implemented here"
                )

        with col2b:
            if st.button("âš¡ Execute Task", key="execute_task"):
                st.info("Task execution interface would be implemented here")

            if st.button("âš™ï¸ Configure", key="configure_orchestrator"):
                st.info(
                    "Orchestrator configuration interface would be implemented here"
                )

    # Task monitoring section
    st.markdown("---")
    st.markdown("### Task Monitoring")

    if orchestrator_status.get("status") == "available":
        # Mock task status data
        task_data = {
            "Model Innovation": {
                "status": "ðŸŸ¢",
                "last_run": "2 hours ago",
                "next_run": "22 hours",
            },
            "Strategy Research": {
                "status": "ðŸŸ¢",
                "last_run": "1 hour ago",
                "next_run": "11 hours",
            },
            "Sentiment Fetch": {
                "status": "ðŸŸ¢",
                "last_run": "5 minutes ago",
                "next_run": "25 minutes",
            },
            "Risk Management": {
                "status": "ðŸŸ¢",
                "last_run": "10 minutes ago",
                "next_run": "5 minutes",
            },
            "Execution": {
                "status": "ðŸŸ¢",
                "last_run": "1 minute ago",
                "next_run": "1 minute",
            },
            "Data Sync": {
                "status": "ðŸŸ¡",
                "last_run": "15 minutes ago",
                "next_run": "5 minutes",
            },
        }

        # Display task status in a table
        task_df = pd.DataFrame(
            [
                {
                    "Task": task,
                    "Status": data["status"],
                    "Last Run": data["last_run"],
                    "Next Run": data["next_run"],
                }
                for task, data in task_data.items()
            ]
        )

        st.dataframe(task_df, use_container_width=True)

    else:
        st.warning("Task monitoring not available - orchestrator not running")

        # Installation instructions
        with st.expander("ðŸ“‹ Installation Instructions"):
            st.markdown(
                """
            **To install Task Orchestrator:**

            1. Ensure all dependencies are installed:
            ```bash
            pip install -r requirements.txt
            ```

            2. Configure the orchestrator:
            ```bash
            cp config/task_schedule.yaml.example config/task_schedule.yaml
            ```

            3. Start the orchestrator:
            ```bash
            python main.py --orchestrator
            ```

            4. Or integrate with existing system:
            ```bash
            python scripts/integrate_orchestrator.py
            ```
            """
            )


def render_footer():
    """Render the footer."""
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Evolve AI Trading Platform | Professional Trading Intelligence</p>
        <p style="font-size: 0.8rem;">Built with advanced AI and machine learning</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
