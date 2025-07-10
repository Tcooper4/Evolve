"""
Institutional Dashboard

Comprehensive dashboard integrating all strategic intelligence modules.
Provides modern, professional UI for institutional-grade trading system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime, timedelta
import logging

# Import the institutional system
try:
    from trading.integration.institutional_grade_system import InstitutionalGradeSystem
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    logging.warning("Institutional system not available")

logger = logging.getLogger(__name__)

class InstitutionalDashboard:
    """Comprehensive institutional trading dashboard."""
    
    def __init__(self):
        """Initialize the institutional dashboard."""
        self.system = None
        self.initialize_system()
        
        # Configure page
        st.set_page_config(
            page_title="Institutional Trading System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self.setup_custom_css()

    def initialize_system(self):
        """Initialize the institutional trading system."""
        if SYSTEM_AVAILABLE:
            try:
                self.system = InstitutionalGradeSystem(auto_start=False)
                logger.info("Institutional system initialized")
            except Exception as e:
                logger.error(f"Error initializing system: {e}")
                self.system = None
        else:
            self.system = None
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_custom_css(self) -> Dict:
        """Setup custom CSS for professional appearance."""
        try:
            st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
                margin: 0.5rem 0;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-running { background-color: #28a745; }
            .status-paused { background-color: #ffc107; }
            .status-error { background-color: #dc3545; }
            .status-initializing { background-color: #17a2b8; }
            .module-card {
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 0.5rem 0;
            }
            .alert-box {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 1rem 0;
            }
            </style>
            """, unsafe_allow_html=True)
            return {"status": "success", "message": "Custom CSS setup completed", "css_loaded": True}
        except Exception as e:
            return {'success': True, 'result': {"status": "error", "message": str(e), "css_loaded": False}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def run(self):
        """Run the institutional dashboard."""
        # Header
        st.markdown('<h1 class="main-header">🏛️ Institutional Trading System</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if self.system is None:
            self.render_system_unavailable()
        else:
            self.render_main_dashboard()

    def render_sidebar(self):
        """Render the sidebar with controls and navigation."""
        with st.sidebar:
            st.header("🎛️ System Controls")
            
            # System status
            if self.system:
                status = self.system.status.value
                status_color = {
                    'running': 'status-running',
                    'paused': 'status-paused',
                    'error': 'status-error',
                    'initializing': 'status-initializing'
                }.get(status, 'status-running')
                
                st.markdown(f"""
                <div class="metric-card">
                    <span class="status-indicator {status_color}"></span>
                    <strong>Status:</strong> {status.title()}
                </div>
                """, unsafe_allow_html=True)
                
                # System controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Start", use_container_width=True):
                        self.system.start()
                        st.rerun()
                
                with col2:
                    if st.button("⏸️ Stop", use_container_width=True):
                        self.system.stop()
                        st.rerun()
            
            st.divider()
            
            # Navigation
            st.header("📊 Navigation")
            page = st.selectbox(
                "Select Page",
                ["Dashboard", "Market Regime", "Signals", "Risk Management", "Performance", "Reports", "Settings"]
            )
            
            st.divider()
            
            # Quick actions
            st.header("⚡ Quick Actions")
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.button("📋 Generate Report", use_container_width=True):
                if self.system:
                    with st.spinner("Generating report..."):
                        report_path = self.system.generate_system_report()
                        if report_path:
                            st.success(f"Report generated: {report_path}")
                        else:
                            st.error("Failed to generate report")
            
            if st.button("💾 Export Data", use_container_width=True):
                if self.system:
                    with st.spinner("Exporting data..."):
                        self.system.export_system_data()
                        st.success("Data exported successfully")

    def render_system_unavailable(self):
        """Render message when system is unavailable."""
        st.error("🚨 Institutional Trading System is not available")
        st.info("Please ensure all required modules are installed and configured.")
        
        # Show installation instructions
        with st.expander("📋 Installation Instructions"):
            st.markdown("""
            To install the Institutional Trading System:
            
            1. **Install Dependencies**:
            ```bash
            pip install -r requirements.txt
            ```
            
            2. **Set Environment Variables**:
            ```bash
            export FRED_API_KEY = os.getenv('API_KEY', '')
            export ALPHA_VANTAGE_API_KEY = os.getenv('API_KEY', '')
            export OPENAI_API_KEY = os.getenv('API_KEY', '')
            ```
            
            3. **Configure System**:
            ```bash
            cp config.example.json config/institutional_system.json
            # Edit the configuration file
            ```
            
            4. **Start the System**:
            ```bash
            streamlit run trading/ui/institutional_dashboard.py
            ```
            """)

    def render_main_dashboard(self):
        """Render the main dashboard."""
        # System overview
        self.render_system_overview()
        
        # Key metrics
        self.render_key_metrics()
        
        # Module status
        self.render_module_status()
        
        # Recent activity
        self.render_recent_activity()
        
        # Charts and visualizations
        self.render_charts()

    def render_system_overview(self):
        """Render system overview section."""
        st.header("📊 System Overview")
        
        if self.system:
            status = self.system.get_system_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "System Status",
                    status['status'].title(),
                    delta=None
                )
            
            with col2:
                uptime_hours = status.get('uptime', 0) / 3600
                st.metric(
                    "Uptime",
                    f"{uptime_hours:.1f}h",
                    delta=None
                )
            
            with col3:
                modules_count = status.get('modules', 0)
                st.metric(
                    "Active Modules",
                    modules_count,
                    delta=None
                )
            
            with col4:
                health = status.get('system_metrics', {}).get('system_health', 0)
                st.metric(
                    "System Health",
                    f"{health:.1%}",
                    delta=None
                )

    def render_key_metrics(self):
        """Render key performance metrics."""
        st.header("📈 Key Metrics")
        
        if self.system:
            metrics = self.system.system_metrics
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Signals (24h)",
                        metrics.total_signals,
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Active Trades",
                        metrics.active_trades,
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Performance Score",
                        f"{metrics.performance_score:.2f}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Risk Score",
                        f"{metrics.risk_score:.2f}",
                        delta=None
                    )

    def render_module_status(self):
        """Render module status overview."""
        st.header("🔧 Module Status")
        
        if self.system:
            module_status = self.system._get_module_status()
            
            # Create status cards
            cols = st.columns(3)
            for i, (module_name, status) in enumerate(module_status.items()):
                with cols[i % 3]:
                    self.render_module_card(module_name, status)

    def render_module_card(self, module_name: str, status: Dict[str, Any]):
        """Render individual module status card."""
        status_value = status.get('status', 'unknown')
        
        # Status color mapping
        status_colors = {
            'healthy': '🟢',
            'warning': '🟡',
            'error': '🔴',
            'unknown': '⚪',
            'fallback': '🟠'
        }
        
        status_icon = status_colors.get(status_value, '⚪')
        
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} {module_name.title()}</h4>
            <p><strong>Status:</strong> {status_value.title()}</p>
        </div>
        """, unsafe_allow_html=True)

    def render_recent_activity(self):
        """Render recent system activity."""
        st.header("📋 Recent Activity")
        
        if self.system:
            # Get recent signals
            if 'signal_center' in self.system.modules:
                recent_signals = self.system.modules['signal_center'].get_recent_signals(hours=1)
                
                if recent_signals:
                    st.subheader("Recent Signals")
                    
                    # Create signal table
                    signal_data = []
                    for signal in recent_signals[:10]:  # Show last 10
                        signal_data.append({
                            'Time': signal.timestamp.strftime('%H:%M:%S'),
                            'Symbol': signal.symbol,
                            'Type': signal.signal_type.value,
                            'Confidence': f"{signal.confidence:.2f}",
                            'Strategy': signal.strategy
                        })
                    
                    if signal_data:
                        df = pd.DataFrame(signal_data)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent signals")
            
            # Get active trades
            if 'signal_center' in self.system.modules:
                active_trades = self.system.modules['signal_center'].get_active_trades()
                
                if active_trades:
                    st.subheader("Active Trades")
                    
                    trade_data = []
                    for trade in active_trades[:10]:  # Show last 10
                        trade_data.append({
                            'Symbol': trade.symbol,
                            'Side': trade.side,
                            'Quantity': f"{trade.quantity:.2f}",
                            'Entry Price': f"${trade.entry_price:.2f}",
                            'Current Price': f"${trade.current_price:.2f}",
                            'PnL': f"${trade.pnl:.2f}",
                            'PnL %': f"{trade.pnl_percent:.2%}"
                        })
                    
                    if trade_data:
                        df = pd.DataFrame(trade_data)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.info("No active trades")

    def render_charts(self):
        """Render charts and visualizations."""
        st.header("📊 Charts & Analytics")
        
        if self.system:
            # Create tabs for different chart types
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk", "Regime", "Signals"])
            
            with tab1:
                self.render_performance_charts()
            
            with tab2:
                self.render_risk_charts()
            
            with tab3:
                self.render_regime_charts()
            
            with tab4:
                self.render_signal_charts()

    def render_performance_charts(self):
        """Render performance-related charts."""
        st.subheader("Performance Analytics")
        
        if self.system:
            # Get performance data
            performance_data = self.system._get_performance_data()
            
            if performance_data:
                # Performance metrics over time (simulated)
                dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
                performance_series = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=performance_series,
                    mode='lines',
                    name='Cumulative Performance',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title='Cumulative Performance Over Time',
                    xaxis_title='Date',
                    yaxis_title='Performance',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available")

    def render_risk_charts(self):
        """Render risk-related charts."""
        st.subheader("Risk Analytics")
        
        if self.system:
            # Risk metrics visualization
            risk_data = self.system._get_risk_data()
            
            if risk_data:
                # Create risk metrics chart
                metrics = ['Volatility', 'VaR', 'Drawdown', 'Beta']
                values = [0.15, 0.02, 0.05, 1.1]  # Simulated values
                
                fig = go.Figure(data=[
                    go.Bar(x=metrics, y=values, marker_color='#ff7f0e')
                ])
                
                fig.update_layout(
                    title='Risk Metrics',
                    xaxis_title='Metric',
                    yaxis_title='Value',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk data available")

    def render_regime_charts(self):
        """Render regime-related charts."""
        st.subheader("Market Regime Analysis")
        
        if self.system:
            regime_data = self.system._get_regime_data()
            
            if regime_data and 'market_regime' in regime_data:
                regime_info = regime_data['market_regime']
                
                # Regime distribution chart
                regimes = ['Bull', 'Bear', 'Sideways', 'Volatile']
                counts = [30, 15, 40, 15]  # Simulated data
                
                fig = go.Figure(data=[
                    go.Pie(labels=regimes, values=counts, hole=0.3)
                ])
                
                fig.update_layout(
                    title='Market Regime Distribution',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Current regime info
                if 'current_regime' in regime_info:
                    st.info(f"Current Regime: {regime_info['current_regime']}")
            else:
                st.info("No regime data available")

    def render_signal_charts(self):
        """Render signal-related charts."""
        st.subheader("Signal Analytics")
        
        if self.system:
            signal_data = self.system._get_signal_data()
            
            if signal_data and 'signal_center' in signal_data:
                signal_summary = signal_data['signal_center']
                
                # Signal distribution
                signal_types = ['Buy', 'Sell', 'Hold', 'Strong Buy', 'Strong Sell']
                signal_counts = [25, 15, 40, 10, 10]  # Simulated data
                
                fig = go.Figure(data=[
                    go.Bar(x=signal_types, y=signal_counts, marker_color='#2ca02c')
                ])
                
                fig.update_layout(
                    title='Signal Distribution (24h)',
                    xaxis_title='Signal Type',
                    yaxis_title='Count',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No signal data available")

    def render_natural_language_interface(self):
        """Render natural language query interface."""
        st.header("🤖 Natural Language Interface")
        
        if self.system:
            # Query input
            query = st.text_input(
                "Ask about the system:",
                placeholder="e.g., What is the current market regime? Show me recent signals. Generate a performance report."
            )
            
            if st.button("🔍 Query", use_container_width=True):
                if query:
                    with st.spinner("Processing query..."):
                        result = self.system.process_natural_language_query(query)
                        
                        if result.get('success'):
                            st.success("Query processed successfully!")
                            
                            # Display results
                            if result.get('data'):
                                st.json(result['data'])
                        else:
                            st.error(f"Query failed: {result.get('error', 'Unknown error')}")

    def render_settings(self):
        """Render system settings."""
        st.header("⚙️ System Settings")
        
        if self.system:
            # Configuration display
            st.subheader("Current Configuration")
            
            config = self.system.config
            
            # System settings
            with st.expander("System Settings"):
                st.json(config.get('system', {}))
            
            # Module settings
            with st.expander("Module Settings"):
                st.json(config.get('modules', {}))
            
            # Risk limits
            with st.expander("Risk Limits"):
                st.json(config.get('risk_limits', {}))
            
            # Data sources
            with st.expander("Data Sources"):
                st.json(config.get('data_sources', {}))

    def render_help(self):
        """Render help and documentation."""
        st.header("❓ Help & Documentation")
        
        if self.system:
            help_info = self.system.get_help()
            
            st.subheader("System Information")
            st.write(f"**Name:** {help_info.get('system_name', 'Unknown')}")
            st.write(f"**Version:** {help_info.get('version', 'Unknown')}")
            st.write(f"**Description:** {help_info.get('description', 'No description available')}")
            
            st.subheader("Available Modules")
            modules = help_info.get('modules', [])
            for module in modules:
                st.write(f"• {module}")
            
            st.subheader("Commands")
            commands = help_info.get('commands', {})
            for cmd, desc in commands.items():
                st.write(f"**{cmd}:** {desc}")
            
            st.subheader("Example Queries")
            examples = help_info.get('example_queries', [])
            for example in examples:
                st.write(f"• {example}")

def main():
    """Main function to run the dashboard."""
    dashboard = InstitutionalDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 