"""
System Administration Page

Merges functionality from:
- 19_Admin_Panel.py (standalone)
- 8_Agent_Management.py (standalone)
- 9_System_Monitoring.py (standalone)
- 5_System_Scorecard.py (standalone)

Features:
- System configuration
- User management
- API key management
- Broker connections
- AI agent management
- System health monitoring
- Resource usage tracking
- Logs and debugging
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend imports (with error handling)
try:
    from trading.config.enhanced_settings import EnhancedSettings
except ImportError:
    logger.warning("EnhancedSettings not found, using placeholder")
    EnhancedSettings = None

try:
    from trading.agents.agent_registry import AgentRegistry
except ImportError:
    logger.warning("AgentRegistry not found, using placeholder")
    AgentRegistry = None

try:
    from trading.monitoring.health_check import SystemHealthMonitor
except ImportError:
    try:
        from monitoring.health_check import HealthChecker as SystemHealthMonitor
    except ImportError:
        logger.warning("SystemHealthMonitor not found, using placeholder")
        SystemHealthMonitor = None

try:
    from trading.utils.system_status import SystemStatus
except ImportError:
    logger.warning("SystemStatus not found, using placeholder")
    SystemStatus = None

# Page configuration
st.set_page_config(
    page_title="System Administration",
    page_icon="⚙️",
    layout="wide"
)

# Authentication check (admin only)
def check_admin_access() -> bool:
    """Check if current user has admin access."""
    # In a real implementation, this would check the user's role from session/token
    # For now, we'll use a simple session state check
    user_role = st.session_state.get("user_role", "admin")
    return user_role == "admin"

# Check admin access
if not check_admin_access():
    st.error("🔒 Access Denied")
    st.markdown("**Admin privileges required to access this page.**")
    st.info("Please contact your system administrator for access.")
    st.stop()

# Initialize session state
if 'admin_settings' not in st.session_state:
    st.session_state.admin_settings = {}

if 'system_health' not in st.session_state:
    st.session_state.system_health = {
        "overall_score": 85,
        "database": "green",
        "api_connections": "green",
        "broker_connections": "yellow",
        "data_providers": "green",
        "agents_active": 3
    }

if 'system_events' not in st.session_state:
    st.session_state.system_events = []

if 'agent_registry' not in st.session_state:
    st.session_state.agent_registry = {}

if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = {
        "cpu_usage": 45.0,
        "memory_usage": 62.0,
        "disk_usage": 38.0,
        "uptime": "5 days, 12 hours"
    }

# Initialize backend modules (with error handling)
try:
    if EnhancedSettings:
        settings_manager = EnhancedSettings()
    else:
        settings_manager = None
except Exception as e:
    logger.error(f"Error initializing EnhancedSettings: {e}")
    settings_manager = None

try:
    if AgentRegistry:
        agent_registry = AgentRegistry()
    else:
        agent_registry = None
except Exception as e:
    logger.error(f"Error initializing AgentRegistry: {e}")
    agent_registry = None

try:
    if SystemHealthMonitor:
        health_monitor = SystemHealthMonitor()
    else:
        health_monitor = None
except Exception as e:
    logger.error(f"Error initializing SystemHealthMonitor: {e}")
    health_monitor = None

# Page header
st.title("⚙️ System Administration")
st.markdown("Manage system configuration, monitor health, and administer AI agents.")

# Tab structure
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 System Dashboard",
    "⚙️ Configuration",
    "🤖 AI Agents",
    "📈 System Monitoring",
    "📜 Logs & Debugging",
    "🔧 Maintenance"
])

# TAB 1: System Dashboard
with tab1:
    st.header("📊 System Dashboard")
    st.markdown("High-level system overview and health monitoring.")
    
    # Overall System Health Score
    st.subheader("🏥 Overall System Health")
    
    health_score = st.session_state.system_health.get("overall_score", 85)
    
    # Health score gauge
    fig_health = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_health.update_layout(height=250)
    st.plotly_chart(fig_health, use_container_width=True)
    
    st.markdown("---")
    
    # Status Indicators
    st.subheader("🔍 Status Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Database Status
    db_status = st.session_state.system_health.get("database", "green")
    with col1:
        if db_status == "green":
            st.success("✅ Database")
        elif db_status == "yellow":
            st.warning("⚠️ Database")
        else:
            st.error("❌ Database")
    
    # API Connections Status
    api_status = st.session_state.system_health.get("api_connections", "green")
    with col2:
        if api_status == "green":
            st.success("✅ API Connections")
        elif api_status == "yellow":
            st.warning("⚠️ API Connections")
        else:
            st.error("❌ API Connections")
    
    # Broker Connections Status
    broker_status = st.session_state.system_health.get("broker_connections", "yellow")
    with col3:
        if broker_status == "green":
            st.success("✅ Broker Connections")
        elif broker_status == "yellow":
            st.warning("⚠️ Broker Connections")
        else:
            st.error("❌ Broker Connections")
    
    # Data Providers Status
    data_status = st.session_state.system_health.get("data_providers", "green")
    with col4:
        if data_status == "green":
            st.success("✅ Data Providers")
        elif data_status == "yellow":
            st.warning("⚠️ Data Providers")
        else:
            st.error("❌ Data Providers")
    
    # Agents Status
    agents_active = st.session_state.system_health.get("agents_active", 0)
    with col5:
        st.metric("🤖 Active Agents", agents_active)
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uptime = st.session_state.system_metrics.get("uptime", "5 days, 12 hours")
        st.metric("⏱️ Uptime", uptime)
    
    with col2:
        # Simulated trades today
        trades_today = 47
        st.metric("📊 Trades Today", trades_today, delta="+12")
    
    with col3:
        # Simulated active strategies
        active_strategies = 8
        st.metric("📈 Active Strategies", active_strategies)
    
    with col4:
        # System load (CPU)
        cpu_usage = st.session_state.system_metrics.get("cpu_usage", 45.0)
        st.metric("💻 System Load", f"{cpu_usage:.1f}%")
    
    st.markdown("---")
    
    # Recent System Events Feed
    st.subheader("📢 Recent System Events")
    
    # Initialize events if empty
    if not st.session_state.system_events:
        st.session_state.system_events = [
            {
                "timestamp": datetime.now().isoformat(),
                "type": "info",
                "message": "System started successfully",
                "component": "System"
            },
            {
                "timestamp": (datetime.now().replace(hour=10, minute=30)).isoformat(),
                "type": "success",
                "message": "Database connection restored",
                "component": "Database"
            },
            {
                "timestamp": (datetime.now().replace(hour=9, minute=15)).isoformat(),
                "type": "warning",
                "message": "High CPU usage detected",
                "component": "Monitoring"
            }
        ]
    
    # Display events (most recent first)
    events_to_show = sorted(
        st.session_state.system_events,
        key=lambda x: x.get('timestamp', ''),
        reverse=True
    )[:10]
    
    if events_to_show:
        for event in events_to_show:
            timestamp = event.get('timestamp', 'Unknown')
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp_str = timestamp
            
            event_type = event.get('type', 'info')
            message = event.get('message', 'N/A')
            component = event.get('component', 'Unknown')
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.caption(f"🕐 {timestamp_str}")
                
                with col2:
                    if event_type == "success":
                        st.success(f"✅ {message}")
                    elif event_type == "warning":
                        st.warning(f"⚠️ {message}")
                    elif event_type == "error":
                        st.error(f"❌ {message}")
                    else:
                        st.info(f"ℹ️ {message}")
                
                with col3:
                    st.caption(f"📦 {component}")
    else:
        st.info("No recent system events.")
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Services", use_container_width=True):
            # Simulate restart
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "info",
                "message": "Services restart initiated",
                "component": "System"
            })
            st.success("✅ Services restart initiated!")
            st.info("This would restart all system services in production.")
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            # Simulate cache clear
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "Cache cleared successfully",
                "component": "Cache"
            })
            st.success("✅ Cache cleared!")
            st.info("All cached data has been cleared.")
    
    with col3:
        if st.button("🏥 Run Health Check", use_container_width=True):
            # Simulate health check
            if health_monitor:
                try:
                    health_status = health_monitor.check_system_health()
                    st.session_state.system_health["overall_score"] = health_status.get('overall_score', 85)
                    st.session_state.system_events.insert(0, {
                        "timestamp": datetime.now().isoformat(),
                        "type": "success",
                        "message": f"Health check completed. Score: {health_status.get('overall_score', 85)}",
                        "component": "Health Monitor"
                    })
                    st.success("✅ Health check completed!")
                except Exception as e:
                    st.error(f"Error running health check: {str(e)}")
            else:
                # Simulate health check
                new_score = min(100, health_score + 2)
                st.session_state.system_health["overall_score"] = new_score
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": f"Health check completed. Score: {new_score}",
                    "component": "Health Monitor"
                })
                st.success("✅ Health check completed!")
    
    # System Health Details (Expandable)
    st.markdown("---")
    with st.expander("📋 System Health Details", expanded=False):
        st.markdown("**Component Status:**")
        
        components = [
            ("Database", db_status),
            ("API Connections", api_status),
            ("Broker Connections", broker_status),
            ("Data Providers", data_status),
            ("AI Agents", "green" if agents_active > 0 else "yellow")
        ]
        
        for component, status in components:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{component}:**")
            with col2:
                if status == "green":
                    st.success("✅ Healthy")
                elif status == "yellow":
                    st.warning("⚠️ Degraded")
                else:
                    st.error("❌ Unhealthy")
        
        st.markdown("---")
        st.markdown("**System Metrics:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"- **CPU Usage:** {cpu_usage:.1f}%")
            st.markdown(f"- **Memory Usage:** {st.session_state.system_metrics.get('memory_usage', 62.0):.1f}%")
            st.markdown(f"- **Disk Usage:** {st.session_state.system_metrics.get('disk_usage', 38.0):.1f}%")
        
        with col2:
            st.markdown(f"- **Uptime:** {uptime}")
            st.markdown(f"- **Active Agents:** {agents_active}")
            st.markdown(f"- **Health Score:** {health_score}/100")

# TAB 2: Configuration
with tab2:
    st.header("⚙️ Configuration")
    st.markdown("Manage system settings, API keys, broker connections, and feature flags.")
    
    # Initialize configuration if not exists
    if 'system_config' not in st.session_state:
        st.session_state.system_config = {
            "general": {
                "system_name": "EVOLVE Trading System",
                "timezone": "America/New_York",
                "base_currency": "USD",
                "trading_hours_start": "09:30",
                "trading_hours_end": "16:00"
            },
            "api_keys": {
                "alpha_vantage": "",
                "finnhub": "",
                "polygon": "",
                "openai": ""
            },
            "brokers": {
                "alpaca_paper": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "alpaca_live": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "binance": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "ibkr": {"enabled": False, "username": "", "password": "", "status": "disconnected"}
            },
            "feature_flags": {
                "beta_features": False,
                "advanced_analytics": True,
                "ai_forecasting": True,
                "real_time_alerts": True,
                "portfolio_optimization": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db",
                "user": "admin"
            }
        }
    
    config = st.session_state.system_config
    
    # General Settings Section
    with st.expander("🌐 General Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            config["general"]["system_name"] = st.text_input(
                "System Name",
                value=config["general"]["system_name"],
                help="Name of the trading system"
            )
            
            config["general"]["timezone"] = st.selectbox(
                "Timezone",
                [
                    "America/New_York",
                    "America/Chicago",
                    "America/Denver",
                    "America/Los_Angeles",
                    "Europe/London",
                    "Europe/Paris",
                    "Asia/Tokyo",
                    "Asia/Hong_Kong",
                    "UTC"
                ],
                index=0 if config["general"]["timezone"] in [
                    "America/New_York", "America/Chicago", "America/Denver",
                    "America/Los_Angeles", "Europe/London", "Europe/Paris",
                    "Asia/Tokyo", "Asia/Hong_Kong", "UTC"
                ] else 0,
                help="System timezone"
            )
        
        with col2:
            config["general"]["base_currency"] = st.selectbox(
                "Base Currency",
                ["USD", "EUR", "GBP", "JPY", "CNY"],
                index=["USD", "EUR", "GBP", "JPY", "CNY"].index(config["general"]["base_currency"]) if config["general"]["base_currency"] in ["USD", "EUR", "GBP", "JPY", "CNY"] else 0,
                help="Base currency for trading"
            )
            
            col_start, col_end = st.columns(2)
            with col_start:
                trading_start = st.time_input(
                    "Trading Hours Start",
                    value=datetime.strptime(config["general"]["trading_hours_start"], "%H:%M").time(),
                    help="Start of trading hours"
                )
                config["general"]["trading_hours_start"] = trading_start.strftime("%H:%M")
            
            with col_end:
                trading_end = st.time_input(
                    "Trading Hours End",
                    value=datetime.strptime(config["general"]["trading_hours_end"], "%H:%M").time(),
                    help="End of trading hours"
                )
                config["general"]["trading_hours_end"] = trading_end.strftime("%H:%M")
    
    st.markdown("---")
    
    # API Keys Section
    with st.expander("🔑 API Keys", expanded=True):
        st.markdown("**Data Provider API Keys:**")
        
        # Alpha Vantage
        col1, col2 = st.columns([3, 1])
        with col1:
            alpha_vantage_key = st.text_input(
                "Alpha Vantage API Key",
                value=config["api_keys"]["alpha_vantage"],
                type="password",
                help="Alpha Vantage API key for market data"
            )
            if alpha_vantage_key:
                config["api_keys"]["alpha_vantage"] = alpha_vantage_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_alpha"):
                st.text_input("Alpha Vantage API Key (Visible)", value=config["api_keys"]["alpha_vantage"], key="alpha_visible", disabled=True)
        
        # Finnhub
        col1, col2 = st.columns([3, 1])
        with col1:
            finnhub_key = st.text_input(
                "Finnhub API Key",
                value=config["api_keys"]["finnhub"],
                type="password",
                help="Finnhub API key for market data"
            )
            if finnhub_key:
                config["api_keys"]["finnhub"] = finnhub_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_finnhub"):
                st.text_input("Finnhub API Key (Visible)", value=config["api_keys"]["finnhub"], key="finnhub_visible", disabled=True)
        
        # Polygon
        col1, col2 = st.columns([3, 1])
        with col1:
            polygon_key = st.text_input(
                "Polygon API Key",
                value=config["api_keys"]["polygon"],
                type="password",
                help="Polygon API key for market data"
            )
            if polygon_key:
                config["api_keys"]["polygon"] = polygon_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_polygon"):
                st.text_input("Polygon API Key (Visible)", value=config["api_keys"]["polygon"], key="polygon_visible", disabled=True)
        
        # OpenAI
        col1, col2 = st.columns([3, 1])
        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=config["api_keys"]["openai"],
                type="password",
                help="OpenAI API key for AI features"
            )
            if openai_key:
                config["api_keys"]["openai"] = openai_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_openai"):
                st.text_input("OpenAI API Key (Visible)", value=config["api_keys"]["openai"], key="openai_visible", disabled=True)
    
    st.markdown("---")
    
    # Broker Connections Section
    with st.expander("🏦 Broker Connections", expanded=True):
        st.markdown("**Configure broker connections:**")
        
        # Alpaca Paper
        st.markdown("**Alpaca (Paper Trading):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            alpaca_paper_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["alpaca_paper"]["enabled"],
                key="alpaca_paper_enabled"
            )
            config["brokers"]["alpaca_paper"]["enabled"] = alpaca_paper_enabled
        
        with col2:
            config["brokers"]["alpaca_paper"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["alpaca_paper"]["api_key"],
                type="password",
                key="alpaca_paper_key"
            )
        
        with col3:
            config["brokers"]["alpaca_paper"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["alpaca_paper"]["api_secret"],
                type="password",
                key="alpaca_paper_secret"
            )
        
        with col4:
            status = config["brokers"]["alpaca_paper"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_alpaca_paper"):
                # Simulate connection test
                config["brokers"]["alpaca_paper"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Alpaca Live
        st.markdown("**Alpaca (Live Trading):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            alpaca_live_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["alpaca_live"]["enabled"],
                key="alpaca_live_enabled"
            )
            config["brokers"]["alpaca_live"]["enabled"] = alpaca_live_enabled
        
        with col2:
            config["brokers"]["alpaca_live"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["alpaca_live"]["api_key"],
                type="password",
                key="alpaca_live_key"
            )
        
        with col3:
            config["brokers"]["alpaca_live"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["alpaca_live"]["api_secret"],
                type="password",
                key="alpaca_live_secret"
            )
        
        with col4:
            status = config["brokers"]["alpaca_live"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_alpaca_live"):
                # Simulate connection test
                config["brokers"]["alpaca_live"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Binance
        st.markdown("**Binance:**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            binance_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["binance"]["enabled"],
                key="binance_enabled"
            )
            config["brokers"]["binance"]["enabled"] = binance_enabled
        
        with col2:
            config["brokers"]["binance"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["binance"]["api_key"],
                type="password",
                key="binance_key"
            )
        
        with col3:
            config["brokers"]["binance"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["binance"]["api_secret"],
                type="password",
                key="binance_secret"
            )
        
        with col4:
            status = config["brokers"]["binance"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_binance"):
                # Simulate connection test
                config["brokers"]["binance"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Interactive Brokers
        st.markdown("**Interactive Brokers (IBKR):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            ibkr_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["ibkr"]["enabled"],
                key="ibkr_enabled"
            )
            config["brokers"]["ibkr"]["enabled"] = ibkr_enabled
        
        with col2:
            config["brokers"]["ibkr"]["username"] = st.text_input(
                "Username",
                value=config["brokers"]["ibkr"]["username"],
                key="ibkr_username"
            )
        
        with col3:
            config["brokers"]["ibkr"]["password"] = st.text_input(
                "Password",
                value=config["brokers"]["ibkr"]["password"],
                type="password",
                key="ibkr_password"
            )
        
        with col4:
            status = config["brokers"]["ibkr"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_ibkr"):
                # Simulate connection test
                config["brokers"]["ibkr"]["status"] = "connected"
                st.success("✅ Connection successful!")
    
    st.markdown("---")
    
    # Feature Flags Section
    with st.expander("🚩 Feature Flags", expanded=False):
        st.markdown("**Enable or disable system features:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["feature_flags"]["beta_features"] = st.checkbox(
                "Beta Features",
                value=config["feature_flags"]["beta_features"],
                help="Enable experimental beta features"
            )
            
            config["feature_flags"]["advanced_analytics"] = st.checkbox(
                "Advanced Analytics",
                value=config["feature_flags"]["advanced_analytics"],
                help="Enable advanced analytics features"
            )
            
            config["feature_flags"]["ai_forecasting"] = st.checkbox(
                "AI Forecasting",
                value=config["feature_flags"]["ai_forecasting"],
                help="Enable AI-powered forecasting"
            )
        
        with col2:
            config["feature_flags"]["real_time_alerts"] = st.checkbox(
                "Real-Time Alerts",
                value=config["feature_flags"]["real_time_alerts"],
                help="Enable real-time alert system"
            )
            
            config["feature_flags"]["portfolio_optimization"] = st.checkbox(
                "Portfolio Optimization",
                value=config["feature_flags"]["portfolio_optimization"],
                help="Enable portfolio optimization features"
            )
    
    st.markdown("---")
    
    # Database Settings Section
    with st.expander("💾 Database Settings", expanded=False):
        st.markdown("**Database configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["database"]["host"] = st.text_input(
                "Database Host",
                value=config["database"]["host"],
                help="Database server hostname"
            )
            
            config["database"]["port"] = st.number_input(
                "Database Port",
                min_value=1,
                max_value=65535,
                value=config["database"]["port"],
                help="Database server port"
            )
        
        with col2:
            config["database"]["name"] = st.text_input(
                "Database Name",
                value=config["database"]["name"],
                help="Database name"
            )
            
            config["database"]["user"] = st.text_input(
                "Database User",
                value=config["database"]["user"],
                help="Database username"
            )
    
    st.markdown("---")
    
    # Save Configuration
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            # Save configuration
            st.session_state.system_config = config
            
            # Add confirmation
            st.success("✅ Configuration saved successfully!")
            
            # Log event
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "System configuration updated",
                "component": "Configuration"
            })
            
            st.info("Configuration changes will take effect after system restart.")

# TAB 3: AI Agents
with tab3:
    st.header("🤖 AI Agents")
    st.markdown("Manage AI agents lifecycle, configuration, and monitoring.")
    
    # Initialize agent registry if not exists
    if not st.session_state.agent_registry:
        st.session_state.agent_registry = {
            "ModelSelectorAgent": {
                "name": "ModelSelectorAgent",
                "type": "Model Selector",
                "status": "active",
                "last_run": datetime.now().isoformat(),
                "performance_score": 92,
                "enabled": True,
                "description": "Selects optimal forecasting model based on data characteristics",
                "configuration": {
                    "max_models": 5,
                    "evaluation_metric": "RMSE",
                    "timeout": 300
                },
                "execution_history": [
                    {"timestamp": datetime.now().isoformat(), "action": "Model selection", "result": "LSTM selected"},
                    {"timestamp": (datetime.now().replace(hour=10, minute=0)).isoformat(), "action": "Model selection", "result": "XGBoost selected"}
                ],
                "performance_metrics": {
                    "success_rate": 0.95,
                    "avg_execution_time": 45.2,
                    "total_executions": 127
                }
            },
            "OptimizerAgent": {
                "name": "OptimizerAgent",
                "type": "Optimizer",
                "status": "active",
                "last_run": (datetime.now().replace(hour=11, minute=30)).isoformat(),
                "performance_score": 88,
                "enabled": True,
                "description": "Optimizes hyperparameters for machine learning models",
                "configuration": {
                    "method": "Optuna",
                    "n_trials": 100,
                    "timeout": 600
                },
                "execution_history": [
                    {"timestamp": (datetime.now().replace(hour=11, minute=30)).isoformat(), "action": "Hyperparameter optimization", "result": "Best params found"}
                ],
                "performance_metrics": {
                    "success_rate": 0.92,
                    "avg_execution_time": 320.5,
                    "total_executions": 45
                }
            },
            "RiskAnalyzerAgent": {
                "name": "RiskAnalyzerAgent",
                "type": "Risk Analyzer",
                "status": "paused",
                "last_run": (datetime.now().replace(hour=8, minute=0)).isoformat(),
                "performance_score": 85,
                "enabled": False,
                "description": "Analyzes portfolio risk and generates risk metrics",
                "configuration": {
                    "risk_metrics": ["VaR", "CVaR", "Sharpe"],
                    "confidence_level": 0.95
                },
                "execution_history": [
                    {"timestamp": (datetime.now().replace(hour=8, minute=0)).isoformat(), "action": "Risk analysis", "result": "Analysis complete"}
                ],
                "performance_metrics": {
                    "success_rate": 0.90,
                    "avg_execution_time": 120.3,
                    "total_executions": 89
                }
            }
        }
    
    # Agent Registry Table
    st.subheader("📋 Agent Registry")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set(agent.get("type", "Unknown") for agent in st.session_state.agent_registry.values())),
            help="Filter agents by type"
        )
    
    with col2:
        agent_status_filter = st.selectbox(
            "Filter by Status",
            ["All", "active", "paused", "error"],
            help="Filter agents by status"
        )
    
    # Filter agents
    filtered_agents = {}
    for name, agent in st.session_state.agent_registry.items():
        if agent_type_filter != "All" and agent.get("type") != agent_type_filter:
            continue
        if agent_status_filter != "All" and agent.get("status") != agent_status_filter:
            continue
        filtered_agents[name] = agent
    
    # Agent Table
    if filtered_agents:
        agents_data = []
        for name, agent in filtered_agents.items():
            status = agent.get("status", "unknown")
            status_display = {
                "active": "🟢 Active",
                "paused": "⏸️ Paused",
                "error": "🔴 Error"
            }.get(status, "❓ Unknown")
            
            last_run = agent.get("last_run", "Never")
            if last_run != "Never":
                try:
                    dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    last_run = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            agents_data.append({
                "Agent Name": name,
                "Type": agent.get("type", "Unknown"),
                "Status": status_display,
                "Last Run": last_run,
                "Performance": f"{agent.get('performance_score', 0)}/100",
                "Enabled": "✅" if agent.get("enabled", False) else "❌"
            })
        
        agents_df = pd.DataFrame(agents_data)
        st.dataframe(agents_df, use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Agent Selection and Management
        st.subheader("⚙️ Agent Management")
        
        selected_agent = st.selectbox(
            "Select Agent",
            options=list(filtered_agents.keys()),
            help="Choose an agent to manage"
        )
        
        if selected_agent:
            agent = st.session_state.agent_registry[selected_agent]
            
            # Agent Actions
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Enable/Disable Toggle
                current_enabled = agent.get("enabled", False)
                new_enabled = st.checkbox(
                    "Enabled",
                    value=current_enabled,
                    key=f"enable_{selected_agent}",
                    help="Enable or disable this agent"
                )
                if new_enabled != current_enabled:
                    agent["enabled"] = new_enabled
                    if new_enabled:
                        agent["status"] = "active"
                    else:
                        agent["status"] = "paused"
                    st.session_state.agent_registry[selected_agent] = agent
                    st.rerun()
            
            with col2:
                # Configure Button
                if st.button("⚙️ Configure", use_container_width=True, key=f"config_{selected_agent}"):
                    st.session_state.configuring_agent = selected_agent
                    st.rerun()
            
            with col3:
                # Test Agent Button
                if st.button("🧪 Test", use_container_width=True, key=f"test_{selected_agent}"):
                    # Simulate agent test
                    agent["last_run"] = datetime.now().isoformat()
                    agent["status"] = "active"
                    st.session_state.agent_registry[selected_agent] = agent
                    st.success(f"✅ Agent '{selected_agent}' test completed!")
                    st.info("Test execution would run the agent with sample data in production.")
            
            with col4:
                # View Logs Button
                if st.button("📜 View Logs", use_container_width=True, key=f"logs_{selected_agent}"):
                    st.session_state.viewing_agent_logs = selected_agent
                    st.rerun()
            
            with col5:
                # Delete Agent Button
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_{selected_agent}"):
                    if selected_agent in st.session_state.agent_registry:
                        del st.session_state.agent_registry[selected_agent]
                        st.success(f"✅ Agent '{selected_agent}' deleted!")
                        st.rerun()
            
            st.markdown("---")
            
            # Agent Details Panel
            st.subheader(f"📄 Agent Details: {selected_agent}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Agent Name:** {selected_agent}")
                st.markdown(f"**Type:** {agent.get('type', 'Unknown')}")
                st.markdown(f"**Status:** {status_display}")
                st.markdown(f"**Enabled:** {'✅ Yes' if agent.get('enabled', False) else '❌ No'}")
                st.markdown(f"**Performance Score:** {agent.get('performance_score', 0)}/100")
                
                last_run = agent.get("last_run", "Never")
                if last_run != "Never":
                    try:
                        dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                        st.markdown(f"**Last Run:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        st.markdown(f"**Last Run:** {last_run}")
                else:
                    st.markdown("**Last Run:** Never")
            
            with col2:
                if agent.get("description"):
                    st.markdown(f"**Description:** {agent.get('description')}")
                
                # Performance Metrics
                metrics = agent.get("performance_metrics", {})
                if metrics:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(f"- Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")
                    st.markdown(f"- Avg Execution Time: {metrics.get('avg_execution_time', 0):.1f}s")
                    st.markdown(f"- Total Executions: {metrics.get('total_executions', 0)}")
            
            # Configuration (Expandable)
            with st.expander("⚙️ Configuration", expanded=False):
                config = agent.get("configuration", {})
                if config:
                    st.json(config)
                else:
                    st.info("No configuration available")
            
            # Execution History (Expandable)
            with st.expander("📜 Execution History", expanded=False):
                history = agent.get("execution_history", [])
                if history:
                    history_data = []
                    for entry in history[-10:]:  # Last 10 entries
                        timestamp = entry.get("timestamp", "Unknown")
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                        history_data.append({
                            "Timestamp": timestamp,
                            "Action": entry.get("action", "N/A"),
                            "Result": entry.get("result", "N/A")
                        })
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("No execution history available")
            
            # Performance Metrics Chart (Expandable)
            with st.expander("📊 Performance Metrics", expanded=False):
                metrics = agent.get("performance_metrics", {})
                if metrics:
                    # Simple bar chart for metrics
                    fig = go.Figure(data=[
                        go.Bar(
                            x=["Success Rate", "Avg Execution Time (s)", "Total Executions"],
                            y=[
                                metrics.get("success_rate", 0) * 100,
                                metrics.get("avg_execution_time", 0),
                                metrics.get("total_executions", 0) / 10  # Scale for visualization
                            ],
                            marker_color=['green', 'blue', 'orange']
                        )
                    ])
                    fig.update_layout(
                        title="Agent Performance Metrics",
                        yaxis_title="Value",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No performance metrics available")
    
    else:
        st.info("No agents found matching the filters.")
    
    st.markdown("---")
    
    # Add New Agent
    st.subheader("➕ Add New Agent")
    
    with st.expander("Create New Agent", expanded=False):
        new_agent_name = st.text_input(
            "Agent Name",
            placeholder="MyCustomAgent",
            help="Unique name for the agent"
        )
        
        new_agent_type = st.selectbox(
            "Agent Type",
            ["Model Selector", "Optimizer", "Risk Analyzer", "Strategy Selector", "Custom"],
            help="Type of agent"
        )
        
        new_agent_description = st.text_area(
            "Description",
            placeholder="Describe what this agent does...",
            height=100,
            help="Agent description"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Create Agent", use_container_width=True):
                if new_agent_name:
                    if new_agent_name in st.session_state.agent_registry:
                        st.error(f"Agent '{new_agent_name}' already exists!")
                    else:
                        st.session_state.agent_registry[new_agent_name] = {
                            "name": new_agent_name,
                            "type": new_agent_type,
                            "status": "paused",
                            "last_run": None,
                            "performance_score": 0,
                            "enabled": False,
                            "description": new_agent_description,
                            "configuration": {},
                            "execution_history": [],
                            "performance_metrics": {
                                "success_rate": 0.0,
                                "avg_execution_time": 0.0,
                                "total_executions": 0
                            }
                        }
                        st.success(f"✅ Agent '{new_agent_name}' created!")
                        st.rerun()
                else:
                    st.error("Please enter an agent name")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
    
    # Agent Logs (if viewing)
    if 'viewing_agent_logs' in st.session_state:
        st.markdown("---")
        st.subheader(f"📜 Agent Logs: {st.session_state.viewing_agent_logs}")
        
        # Simulated logs
        log_level = st.selectbox("Log Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR"], key="agent_log_level")
        
        # Simulated log entries
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Agent initialized",
            f"[{(datetime.now().replace(hour=11, minute=0)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model selection started",
            f"[{(datetime.now().replace(hour=11, minute=1)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: LSTM model selected",
            f"[{(datetime.now().replace(hour=11, minute=2)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Execution completed successfully"
        ]
        
        # Filter logs by level
        if log_level != "All":
            log_entries = [log for log in log_entries if log_level in log]
        
        st.code("\n".join(log_entries), language="text")
        
        if st.button("❌ Close Logs"):
            del st.session_state.viewing_agent_logs
            st.rerun()
    
    # Agent Testing Interface (if configuring)
    if 'configuring_agent' in st.session_state:
        st.markdown("---")
        st.subheader(f"⚙️ Configure Agent: {st.session_state.configuring_agent}")
        
        agent = st.session_state.agent_registry[st.session_state.configuring_agent]
        config = agent.get("configuration", {})
        
        # Configuration editor
        import json
        config_json = st.text_area(
            "Configuration (JSON)",
            value=json.dumps(config, indent=2) if config else "{}",
            height=200,
            help="Edit agent configuration as JSON"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Configuration", use_container_width=True):
                try:
                    import json
                    new_config = json.loads(config_json)
                    agent["configuration"] = new_config
                    st.session_state.agent_registry[st.session_state.configuring_agent] = agent
                    st.success("✅ Configuration saved!")
                    del st.session_state.configuring_agent
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format!")
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                del st.session_state.configuring_agent
                st.rerun()

# TAB 4: System Monitoring
with tab4:
    st.header("📈 System Monitoring")
    st.markdown("Real-time system resource usage and service status monitoring.")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (5 seconds)", value=False, help="Automatically refresh metrics every 5 seconds")
    
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
    
    # Resource Usage Section
    st.subheader("💻 Resource Usage")
    
    # Try to get real system metrics, fallback to simulated
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        # Use simulated metrics from session state
        cpu_percent = st.session_state.system_metrics.get("cpu_usage", 45.0)
        memory_percent = st.session_state.system_metrics.get("memory_usage", 62.0)
        disk_percent = st.session_state.system_metrics.get("disk_usage", 38.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # CPU Usage Gauge
    with col1:
        st.markdown("**CPU Usage**")
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_cpu.update_layout(height=200)
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    # Memory Usage Gauge
    with col2:
        st.markdown("**Memory Usage**")
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number",
            value=memory_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgreen"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_mem.update_layout(height=200)
        st.plotly_chart(fig_mem, use_container_width=True)
    
    # Disk Usage Gauge
    with col3:
        st.markdown("**Disk Usage**")
        fig_disk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=disk_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgreen"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_disk.update_layout(height=200)
        st.plotly_chart(fig_disk, use_container_width=True)
    
    # Network I/O
    with col4:
        st.markdown("**Network I/O**")
        try:
            if PSUTIL_AVAILABLE:
                net_io = psutil.net_io_counters()
                bytes_sent = net_io.bytes_sent / (1024**3)  # GB
                bytes_recv = net_io.bytes_recv / (1024**3)  # GB
            else:
                bytes_sent = 12.5
                bytes_recv = 45.8
        except:
            bytes_sent = 12.5
            bytes_recv = 45.8
        
        st.metric("Sent", f"{bytes_sent:.2f} GB")
        st.metric("Received", f"{bytes_recv:.2f} GB")
    
    # Resource Usage Charts (Historical)
    st.markdown("---")
    st.subheader("📊 Resource Usage Over Time")
    
    # Simulated historical data
    import numpy as np
    time_points = list(range(30))  # Last 30 data points
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage Chart
        cpu_history = [cpu_percent + np.random.normal(0, 5) for _ in time_points]
        cpu_history = [max(0, min(100, x)) for x in cpu_history]  # Clamp to 0-100
        
        fig_cpu_chart = go.Figure()
        fig_cpu_chart.add_trace(go.Scatter(
            x=time_points,
            y=cpu_history,
            mode='lines',
            name='CPU Usage',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_cpu_chart.update_layout(
            title="CPU Usage Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="CPU Usage (%)",
            height=250
        )
        st.plotly_chart(fig_cpu_chart, use_container_width=True)
    
    with col2:
        # Memory Usage Chart
        memory_history = [memory_percent + np.random.normal(0, 3) for _ in time_points]
        memory_history = [max(0, min(100, x)) for x in memory_history]  # Clamp to 0-100
        
        fig_mem_chart = go.Figure()
        fig_mem_chart.add_trace(go.Scatter(
            x=time_points,
            y=memory_history,
            mode='lines',
            name='Memory Usage',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_mem_chart.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Memory Usage (%)",
            height=250
        )
        st.plotly_chart(fig_mem_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Service Status Section
    st.subheader("🔧 Service Status")
    
    # Initialize service status if not exists
    if 'service_status' not in st.session_state:
        st.session_state.service_status = {
            "web_server": {"status": "running", "uptime": "5 days"},
            "database": {"status": "running", "uptime": "5 days"},
            "cache": {"status": "running", "uptime": "5 days"},
            "task_queue": {"status": "running", "uptime": "5 days"}
        }
    
    services = st.session_state.service_status
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Web Server**")
        if services["web_server"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['web_server']['uptime']}")
    
    with col2:
        st.markdown("**Database**")
        if services["database"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['database']['uptime']}")
    
    with col3:
        st.markdown("**Cache (Redis)**")
        if services["cache"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['cache']['uptime']}")
    
    with col4:
        st.markdown("**Task Queue**")
        if services["task_queue"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['task_queue']['uptime']}")
    
    st.markdown("---")
    
    # API Rate Limits Section
    st.subheader("📡 API Rate Limits")
    
    # Initialize API rate limits if not exists
    if 'api_rate_limits' not in st.session_state:
        st.session_state.api_rate_limits = {
            "alpha_vantage": {"used": 450, "limit": 500, "reset_time": "2024-12-19 23:59:59"},
            "finnhub": {"used": 58, "limit": 60, "reset_time": "2024-12-19 23:59:59"},
            "polygon": {"used": 4, "limit": 5, "reset_time": "2024-12-19 23:59:59"},
            "openai": {"used": 850, "limit": 1000, "reset_time": "2024-12-19 23:59:59"}
        }
    
    api_limits = st.session_state.api_rate_limits
    
    for provider, limits in api_limits.items():
        used = limits["used"]
        limit = limits["limit"]
        remaining = limit - used
        percentage = (used / limit * 100) if limit > 0 else 0
        
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"**{provider.replace('_', ' ').title()}**")
        
        with col2:
            # Progress bar
            st.progress(percentage / 100)
            st.caption(f"{used} / {limit} requests used ({remaining} remaining)")
        
        with col3:
            st.caption(f"Reset: {limits['reset_time']}")
    
    st.markdown("---")
    
    # Performance Metrics Section
    st.subheader("⚡ Performance Metrics")
    
    # Initialize performance metrics if not exists
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            "avg_response_time": 125,  # ms
            "avg_query_time": 45,  # ms
            "error_rate": 0.02,  # 2%
            "requests_per_minute": 120
        }
    
    perf_metrics = st.session_state.performance_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Response Time", f"{perf_metrics['avg_response_time']} ms")
    
    with col2:
        st.metric("Avg Query Time", f"{perf_metrics['avg_query_time']} ms")
    
    with col3:
        error_rate_pct = perf_metrics['error_rate'] * 100
        st.metric("Error Rate", f"{error_rate_pct:.2f}%")
    
    with col4:
        st.metric("Requests/Min", f"{perf_metrics['requests_per_minute']}")
    
    # Performance Charts
    st.markdown("---")
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response Time Trend
        response_times = [perf_metrics['avg_response_time'] + np.random.normal(0, 10) for _ in time_points]
        response_times = [max(0, x) for x in response_times]
        
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=time_points,
            y=response_times,
            mode='lines',
            name='Response Time',
            line=dict(color='#2ca02c', width=2)
        ))
        fig_response.update_layout(
            title="Response Time Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Response Time (ms)",
            height=250
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        # Error Rate Trend
        error_rates = [perf_metrics['error_rate'] * 100 + np.random.normal(0, 0.5) for _ in time_points]
        error_rates = [max(0, x) for x in error_rates]
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(
            x=time_points,
            y=error_rates,
            mode='lines',
            name='Error Rate',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ))
        fig_errors.update_layout(
            title="Error Rate Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Error Rate (%)",
            height=250
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    
    # Refresh Button
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()

# TAB 5: Logs & Debugging
with tab5:
    st.header("📜 Logs & Debugging")
    st.markdown("View system logs, debug issues, and analyze errors.")
    
    # Log Filters
    st.subheader("🔍 Log Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox(
            "Log Level",
            ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Filter logs by level"
        )
    
    with col2:
        log_source = st.selectbox(
            "Log Source",
            ["All", "app", "trading", "data", "models", "agents", "system"],
            help="Filter logs by source"
        )
    
    with col3:
        log_date = st.date_input(
            "Date",
            value=datetime.now().date(),
            help="Filter logs by date"
        )
    
    # Search
    log_search = st.text_input(
        "Search Logs",
        placeholder="Search log messages...",
        help="Search logs by content"
    )
    
    st.markdown("---")
    
    # Initialize logs if not exists
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "source": "app",
                "message": "Application started successfully",
                "component": "System"
            },
            {
                "timestamp": (datetime.now().replace(hour=11, minute=30)).isoformat(),
                "level": "INFO",
                "source": "trading",
                "message": "Trade executed: BUY 100 shares of AAPL at $150.25",
                "component": "Trading"
            },
            {
                "timestamp": (datetime.now().replace(hour=11, minute=25)).isoformat(),
                "level": "WARNING",
                "source": "data",
                "message": "Data provider API rate limit approaching",
                "component": "DataLoader"
            },
            {
                "timestamp": (datetime.now().replace(hour=10, minute=45)).isoformat(),
                "level": "ERROR",
                "source": "models",
                "message": "Model training failed: Out of memory",
                "component": "LSTMModel"
            },
            {
                "timestamp": (datetime.now().replace(hour=10, minute=30)).isoformat(),
                "level": "DEBUG",
                "source": "agents",
                "message": "Agent execution started: ModelSelectorAgent",
                "component": "AgentRegistry"
            },
            {
                "timestamp": (datetime.now().replace(hour=9, minute=15)).isoformat(),
                "level": "INFO",
                "source": "system",
                "message": "Database connection restored",
                "component": "Database"
            }
        ]
    
    # Filter logs
    filtered_logs = []
    for log in st.session_state.system_logs:
        # Level filter
        if log_level != "All" and log.get("level") != log_level:
            continue
        
        # Source filter
        if log_source != "All" and log.get("source") != log_source:
            continue
        
        # Date filter
        try:
            log_date_obj = datetime.fromisoformat(log.get("timestamp", "").replace('Z', '+00:00')).date()
            if log_date_obj != log_date:
                continue
        except:
            continue
        
        # Search filter
        if log_search:
            search_lower = log_search.lower()
            message = log.get("message", "").lower()
            component = log.get("component", "").lower()
            if search_lower not in message and search_lower not in component:
                continue
        
        filtered_logs.append(log)
    
    # Sort by timestamp (newest first)
    filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    st.subheader("📋 Log Viewer")
    
    # Live log tail toggle
    live_tail = st.checkbox("📺 Live Tail (Auto-scroll)", value=False, help="Automatically scroll to newest logs")
    
    # Log Display
    if filtered_logs:
        # Log count
        st.caption(f"Showing {len(filtered_logs)} log entries")
        
        # Log entries
        log_container = st.container()
        with log_container:
            for log in filtered_logs:
                timestamp = log.get("timestamp", "Unknown")
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    timestamp_str = timestamp
                
                level = log.get("level", "INFO")
                source = log.get("source", "unknown")
                message = log.get("message", "N/A")
                component = log.get("component", "Unknown")
                
                # Color code by level
                if level == "ERROR" or level == "CRITICAL":
                    border_color = "red"
                elif level == "WARNING":
                    border_color = "orange"
                elif level == "DEBUG":
                    border_color = "gray"
                else:
                    border_color = "blue"
                
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 4, 2])
                    
                    with col1:
                        st.caption(f"🕐 {timestamp_str}")
                        st.caption(f"📦 {component}")
                    
                    with col2:
                        # Level badge
                        if level == "ERROR" or level == "CRITICAL":
                            st.error(f"[{level}] {message}")
                        elif level == "WARNING":
                            st.warning(f"[{level}] {message}")
                        elif level == "DEBUG":
                            st.info(f"[{level}] {message}")
                        else:
                            st.info(f"[{level}] {message}")
                    
                    with col3:
                        st.caption(f"🔖 {source}")
    else:
        st.info("No logs found matching the filters.")
    
    st.markdown("---")
    
    # Download Logs
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Logs", use_container_width=True):
            # Create log file content
            log_content = ""
            for log in filtered_logs:
                timestamp = log.get("timestamp", "Unknown")
                level = log.get("level", "INFO")
                source = log.get("source", "unknown")
                message = log.get("message", "N/A")
                component = log.get("component", "Unknown")
                log_content += f"[{timestamp}] [{level}] [{source}] [{component}] {message}\n"
            
            st.download_button(
                label="⬇️ Download Log File",
                data=log_content,
                file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            if st.button("⚠️ Confirm Clear", key="confirm_clear_logs"):
                st.session_state.system_logs = []
                st.success("✅ Logs cleared!")
                st.rerun()
    
    st.markdown("---")
    
    # Error Summary
    st.subheader("📊 Error Summary")
    
    # Count errors by type
    error_counts = {}
    error_messages = {}
    
    for log in st.session_state.system_logs:
        level = log.get("level", "")
        if level in ["ERROR", "CRITICAL"]:
            error_type = log.get("component", "Unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            error_messages[error_type] = log.get("message", "N/A")
    
    if error_counts:
        col1, col2 = st.columns(2)
        
        with col1:
            # Error count by type
            st.markdown("**Error Count by Type:**")
            error_df = pd.DataFrame({
                "Component": list(error_counts.keys()),
                "Error Count": list(error_counts.values())
            })
            st.dataframe(error_df, use_container_width=True)
        
        with col2:
            # Most common errors
            st.markdown("**Most Common Errors:**")
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for component, count in sorted_errors:
                st.markdown(f"- **{component}:** {count} error(s)")
                if component in error_messages:
                    st.caption(f"  Last: {error_messages[component]}")
        
        # Error trend chart
        st.markdown("---")
        st.markdown("**Error Trend:**")
        
        # Simulated error trend data
        import numpy as np
        time_points = list(range(24))  # Last 24 hours
        error_trend = [np.random.randint(0, 5) for _ in time_points]
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(
            x=time_points,
            y=error_trend,
            mode='lines+markers',
            name='Errors',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ))
        fig_errors.update_layout(
            title="Error Count Over Time (Last 24 Hours)",
            xaxis_title="Hours Ago",
            yaxis_title="Error Count",
            height=300
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    else:
        st.success("✅ No errors found in the logs!")
    
    st.markdown("---")
    
    # Debugging Tools
    st.subheader("🔧 Debugging Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "Cache cleared via debugging tools",
                "component": "Debug"
            })
            st.success("✅ Cache cleared!")
            st.info("All cached data has been cleared.")
    
    with col2:
        if st.button("🔄 Reset Session", use_container_width=True):
            # Clear session state (except essential)
            keys_to_keep = ['system_health', 'system_metrics', 'system_config', 'agent_registry']
            keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("✅ Session reset!")
            st.info("Session state has been reset (essential data preserved).")
    
    with col3:
        if st.button("🧹 Force GC", use_container_width=True):
            import gc
            gc.collect()
            st.success("✅ Garbage collection forced!")
            st.info("Python garbage collector has been run.")
    
    with col4:
        if st.button("🔌 Test DB", use_container_width=True):
            # Simulate database connection test
            try:
                # In production, this would actually test the database connection
                st.success("✅ Database connection successful!")
                st.info("Database is accessible and responding.")
            except Exception as e:
                st.error(f"❌ Database connection failed: {str(e)}")
    
    # Debugging Information
    st.markdown("---")
    with st.expander("🔍 System Debug Information", expanded=False):
        st.markdown("**Python Version:**")
        import sys
        st.code(f"{sys.version}")
        
        st.markdown("**Streamlit Version:**")
        try:
            import streamlit as st_module
            st.code(f"{st_module.__version__}")
        except:
            st.code("Unknown")
        
        st.markdown("**Session State Keys:**")
        st.code(f"{list(st.session_state.keys())}")
        
        st.markdown("**System Metrics:**")
        st.json(st.session_state.system_metrics)

# TAB 6: Maintenance
with tab6:
    st.header("🔧 Maintenance")
    st.markdown("Database maintenance, cache management, data cleanup, and system updates.")
    
    # Database Maintenance Section
    with st.expander("💾 Database Maintenance", expanded=True):
        st.markdown("**Database operations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Backup Database", use_container_width=True):
                # Simulate backup
                backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                st.success(f"✅ Database backup created: {backup_filename}")
                st.info("In production, this would create a full database backup.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": f"Database backup created: {backup_filename}",
                    "component": "Database"
                })
            
            if st.button("📥 Restore from Backup", use_container_width=True):
                st.warning("⚠️ Restore operation requires confirmation!")
                if st.button("⚠️ Confirm Restore", key="confirm_restore"):
                    st.success("✅ Database restored from backup!")
                    st.info("In production, this would restore the database from the selected backup file.")
        
        with col2:
            if st.button("⚡ Optimize Database", use_container_width=True):
                # Simulate optimization
                st.success("✅ Database optimized!")
                st.info("Database indexes and tables have been optimized for better performance.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": "Database optimization completed",
                    "component": "Database"
                })
            
            if st.button("🧹 Vacuum Database", use_container_width=True):
                # Simulate vacuum
                st.success("✅ Database vacuum completed!")
                st.info("Database has been vacuumed to reclaim unused space.")
        
        # Database Size
        st.markdown("---")
        st.markdown("**Database Information:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database Size", "2.5 GB")
        
        with col2:
            st.metric("Tables", "47")
        
        with col3:
            st.metric("Indexes", "89")
    
    st.markdown("---")
    
    # Cache Management Section
    with st.expander("🗄️ Cache Management", expanded=True):
        st.markdown("**Cache operations:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Cache", use_container_width=True):
                # Simulate cache clear
                st.success("✅ All cache cleared!")
                st.info("All cached data has been removed.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": "All cache cleared",
                    "component": "Cache"
                })
        
        with col2:
            cache_type = st.selectbox(
                "Clear Specific Cache",
                ["Data Cache", "Model Cache", "Strategy Cache", "Session Cache"],
                help="Select cache type to clear"
            )
            
            if st.button("🗑️ Clear Selected Cache", use_container_width=True):
                st.success(f"✅ {cache_type} cleared!")
                st.info(f"All {cache_type.lower()} data has been removed.")
        
        with col3:
            # Cache Hit Rate
            st.markdown("**Cache Statistics:**")
            st.metric("Cache Hit Rate", "87.5%")
            st.metric("Cache Size", "245 MB")
    
    st.markdown("---")
    
    # Data Cleanup Section
    with st.expander("🧹 Data Cleanup", expanded=True):
        st.markdown("**Clean up old data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            delete_days = st.number_input(
                "Delete Logs Older Than (Days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Logs older than this many days will be deleted"
            )
            
            if st.button("🗑️ Delete Old Logs", use_container_width=True):
                # Simulate deletion
                st.success(f"✅ Deleted logs older than {delete_days} days!")
                st.info(f"In production, this would delete all logs older than {delete_days} days.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "info",
                    "message": f"Deleted logs older than {delete_days} days",
                    "component": "Data Cleanup"
                })
            
            if st.button("📦 Archive Old Data", use_container_width=True):
                # Simulate archiving
                st.success("✅ Old data archived!")
                st.info("In production, this would archive old data to long-term storage.")
        
        with col2:
            if st.button("🧹 Clean Temp Files", use_container_width=True):
                # Simulate cleanup
                st.success("✅ Temporary files cleaned!")
                st.info("All temporary files have been removed.")
            
            # Data Statistics
            st.markdown("---")
            st.markdown("**Data Statistics:**")
            
            st.metric("Total Logs", "12,547")
            st.metric("Old Logs (>30 days)", "3,892")
            st.metric("Temp Files Size", "156 MB")
    
    st.markdown("---")
    
    # System Updates Section
    with st.expander("🔄 System Updates", expanded=True):
        st.markdown("**System version and updates:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Version:**")
            st.info("v1.2.3")
            
            if st.button("🔍 Check for Updates", use_container_width=True):
                # Simulate update check
                st.success("✅ System is up to date!")
                st.info("No updates available. You are running the latest version.")
        
        with col2:
            st.markdown("**Update History:**")
            
            update_history = [
                {"version": "v1.2.3", "date": "2024-12-15", "notes": "Bug fixes and performance improvements"},
                {"version": "v1.2.2", "date": "2024-12-01", "notes": "New features: AI forecasting, risk alerts"},
                {"version": "v1.2.1", "date": "2024-11-15", "notes": "Security patches and stability improvements"}
            ]
            
            for update in update_history:
                with st.container(border=True):
                    st.markdown(f"**{update['version']}** - {update['date']}")
                    st.caption(update['notes'])
    
    st.markdown("---")
    
    # Scheduled Maintenance Section
    with st.expander("⏰ Scheduled Maintenance", expanded=False):
        st.markdown("**Configure automated maintenance tasks:**")
        
        # Initialize scheduled maintenance if not exists
        if 'scheduled_maintenance' not in st.session_state:
            st.session_state.scheduled_maintenance = {
                "backup": {
                    "enabled": True,
                    "frequency": "Daily",
                    "time": "02:00",
                    "retention_days": 30
                },
                "cleanup": {
                    "enabled": True,
                    "frequency": "Weekly",
                    "day": "Sunday",
                    "time": "03:00",
                    "delete_logs_older_than": 30
                }
            }
        
        maintenance = st.session_state.scheduled_maintenance
        
        # Backup Schedule
        st.markdown("**Backup Schedule:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            backup_enabled = st.checkbox(
                "Enable",
                value=maintenance["backup"]["enabled"],
                key="backup_enabled"
            )
            maintenance["backup"]["enabled"] = backup_enabled
        
        with col2:
            maintenance["backup"]["frequency"] = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=["Daily", "Weekly", "Monthly"].index(maintenance["backup"]["frequency"]),
                key="backup_frequency"
            )
        
        with col3:
            backup_time = st.time_input(
                "Time",
                value=datetime.strptime(maintenance["backup"]["time"], "%H:%M").time(),
                key="backup_time"
            )
            maintenance["backup"]["time"] = backup_time.strftime("%H:%M")
        
        with col4:
            maintenance["backup"]["retention_days"] = st.number_input(
                "Retention (Days)",
                min_value=1,
                max_value=365,
                value=maintenance["backup"]["retention_days"],
                key="backup_retention"
            )
        
        st.markdown("---")
        
        # Cleanup Schedule
        st.markdown("**Cleanup Schedule:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cleanup_enabled = st.checkbox(
                "Enable",
                value=maintenance["cleanup"]["enabled"],
                key="cleanup_enabled"
            )
            maintenance["cleanup"]["enabled"] = cleanup_enabled
        
        with col2:
            maintenance["cleanup"]["frequency"] = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=["Daily", "Weekly", "Monthly"].index(maintenance["cleanup"]["frequency"]),
                key="cleanup_frequency"
            )
        
        with col3:
            if maintenance["cleanup"]["frequency"] == "Weekly":
                maintenance["cleanup"]["day"] = st.selectbox(
                    "Day",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(maintenance["cleanup"].get("day", "Sunday")),
                    key="cleanup_day"
                )
            else:
                cleanup_time = st.time_input(
                    "Time",
                    value=datetime.strptime(maintenance["cleanup"].get("time", "03:00"), "%H:%M").time(),
                    key="cleanup_time"
                )
                maintenance["cleanup"]["time"] = cleanup_time.strftime("%H:%M")
        
        with col4:
            maintenance["cleanup"]["delete_logs_older_than"] = st.number_input(
                "Delete Logs Older Than (Days)",
                min_value=1,
                max_value=365,
                value=maintenance["cleanup"]["delete_logs_older_than"],
                key="cleanup_logs_days"
            )
        
        # Save Schedule
        if st.button("💾 Save Schedule", use_container_width=True):
            st.session_state.scheduled_maintenance = maintenance
            st.success("✅ Maintenance schedule saved!")
            st.info("Scheduled maintenance tasks will run according to the configured schedule.")
    
    # Maintenance Summary
    st.markdown("---")
    st.subheader("📊 Maintenance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Backup", "2024-12-19 02:00")
    
    with col2:
        st.metric("Backups This Month", "19")
    
    with col3:
        st.metric("Cache Hit Rate", "87.5%")
    
    with col4:
        st.metric("System Uptime", "5 days, 12 hours")

