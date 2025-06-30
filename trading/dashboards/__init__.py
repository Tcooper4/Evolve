"""Dashboard components for trading visualization."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TradingDashboard:
    """Main trading dashboard component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading dashboard."""
        self.config = config or {}
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def render_portfolio_overview(self, portfolio_data: pd.DataFrame):
        """Render portfolio overview chart."""
        try:
            if portfolio_data.empty:
                st.warning("No portfolio data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            fig = go.Figure()
            
            # Portfolio value over time
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering portfolio overview: {e}")
            st.error("Error rendering portfolio chart")
    
    def render_returns_distribution(self, returns: pd.Series):
        """Render returns distribution chart."""
        try:
            if returns.empty:
                st.warning("No returns data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            fig = px.histogram(
                returns,
                title='Returns Distribution',
                nbins=50,
                color_discrete_sequence=['blue']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering returns distribution: {e}")
            st.error("Error rendering returns chart")
    
    def render_drawdown_chart(self, portfolio_data: pd.DataFrame):
        """Render drawdown chart."""
        try:
            if portfolio_data.empty:
                st.warning("No portfolio data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Calculate drawdown
            peak = portfolio_data['total_value'].expanding().max()
            drawdown = (portfolio_data['total_value'] - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=drawdown,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red'),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering drawdown chart: {e}")
            st.error("Error rendering drawdown chart")
    
    def render_asset_allocation(self, allocation_data: Dict[str, float]):
        """Render asset allocation pie chart."""
        try:
            if not allocation_data:
                st.warning("No allocation data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            fig = go.Figure(data=[go.Pie(
                labels=list(allocation_data.keys()),
                values=list(allocation_data.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title='Asset Allocation',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering asset allocation: {e}")
            st.error("Error rendering allocation chart")
    
    def render_performance_metrics(self, metrics: Dict[str, float]):
        """Render performance metrics table."""
        try:
            if not metrics:
                st.warning("No metrics available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in metrics.items()
            ])
            
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering performance metrics: {e}")
            st.error("Error rendering metrics table")

class StrategyDashboard:
    """Strategy-specific dashboard component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy dashboard."""
        self.config = config or {}
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def render_strategy_performance(self, strategy_data: pd.DataFrame):
        """Render strategy performance chart."""
        try:
            if strategy_data.empty:
                st.warning("No strategy data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            fig = go.Figure()
            
            # Strategy returns
            fig.add_trace(go.Scatter(
                x=strategy_data.index,
                y=strategy_data['cumulative_returns'],
                mode='lines',
                name='Strategy Returns',
                line=dict(color='green')
            ))
            
            # Benchmark if available
            if 'benchmark_returns' in strategy_data.columns:
                fig.add_trace(go.Scatter(
                    x=strategy_data.index,
                    y=strategy_data['benchmark_returns'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', dash='dash')
                ))
            
            fig.update_layout(
                title='Strategy Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Returns',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering strategy performance: {e}")
            st.error("Error rendering strategy chart")
    
    def render_trade_analysis(self, trades_data: pd.DataFrame):
        """Render trade analysis."""
        try:
            if trades_data.empty:
                st.warning("No trades data available")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            # Trade P&L distribution
            fig = px.histogram(
                trades_data,
                x='pnl',
                title='Trade P&L Distribution',
                nbins=30,
                color_discrete_sequence=['orange']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade statistics
            st.subheader("Trade Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", len(trades_data))
            with col2:
                win_rate = (trades_data['pnl'] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                avg_pnl = trades_data['pnl'].mean()
                st.metric("Avg P&L", f"${avg_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error rendering trade analysis: {e}")
            st.error("Error rendering trade analysis")

# Global dashboard instances
trading_dashboard = TradingDashboard()
strategy_dashboard = StrategyDashboard()

def get_trading_dashboard() -> TradingDashboard:
    """Get the global trading dashboard instance."""
    return {'success': True, 'result': trading_dashboard, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def get_strategy_dashboard() -> StrategyDashboard:
    """Get the global strategy dashboard instance."""
    return {'success': True, 'result': strategy_dashboard, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}