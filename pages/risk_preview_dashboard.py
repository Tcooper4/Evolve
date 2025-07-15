"""
Risk Preview Dashboard

Enhanced with Batch 11 features: Risk preview dashboard with expected loss, margin usage, VaR,
and toggle-able overlays for each model/strategy via Streamlit sidebar.

This module provides a comprehensive risk management dashboard for monitoring portfolio risk
metrics and position sizing recommendations.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading modules
try:
    from trading.risk.risk_manager import RiskManager, VolatilityModel, DynamicVolatilityModel
    from trading.portfolio.portfolio_manager import PortfolioManager
    from trading.strategies.strategy_composer import StrategyComposer
    from trading.visualization.visualizer import EnhancedVisualizer
    RISK_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Risk modules not available: {e}")
    RISK_MODULES_AVAILABLE = False

# Import utility modules
try:
    from utils.math_helpers import (
        calculate_tail_risk_metrics,
        calculate_volatility_regime,
        calculate_correlation_regime
    )
    from utils.logging import EnhancedLogManager
    UTILS_AVAILABLE = True
except ImportError as e:
    st.error(f"Utility modules not available: {e}")
    UTILS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class RiskPreviewDashboard:
    """Risk preview dashboard with comprehensive risk monitoring."""
    
    def __init__(self):
        """Initialize the risk preview dashboard."""
        self.risk_manager = None
        self.portfolio_manager = None
        self.strategy_composer = None
        self.visualizer = None
        
        if RISK_MODULES_AVAILABLE:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize dashboard components."""
        try:
            # Initialize risk manager with dynamic volatility model
            risk_config = {
                "volatility_model": {
                    "type": "hybrid",
                    "window": 252,
                    "alpha": 0.94,
                    "garch_order": (1, 1)
                },
                "position_sizing": {
                    "max_position_size": 0.1,
                    "volatility_scaling": True,
                    "risk_adjustment": True,
                    "confidence_threshold": 0.7
                },
                "target_volatility": 0.15
            }
            
            self.risk_manager = RiskManager(risk_config)
            self.portfolio_manager = PortfolioManager()
            self.strategy_composer = StrategyComposer()
            self.visualizer = EnhancedVisualizer()
            
            logger.info("Risk preview dashboard components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard components: {e}")
            st.error(f"Failed to initialize dashboard components: {e}")
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.set_page_config(
            page_title="Risk Preview Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("🎯 Risk Preview Dashboard")
        st.markdown("Comprehensive risk monitoring and position sizing recommendations")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main dashboard content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_risk_metrics()
            self._render_volatility_analysis()
        
        with col2:
            self._render_position_recommendations()
            self._render_risk_alerts()
        
        # Full-width charts
        self._render_risk_charts()
        self._render_strategy_overlays()
    
    def _render_sidebar(self):
        """Render sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Risk model selection
        st.sidebar.subheader("Risk Model")
        model_type = st.sidebar.selectbox(
            "Volatility Model",
            ["rolling_std", "ewma", "garch", "hybrid"],
            index=3
        )
        
        # Model parameters
        window = st.sidebar.slider("Rolling Window", 30, 500, 252)
        alpha = st.sidebar.slider("EWMA Alpha", 0.8, 0.99, 0.94, 0.01)
        
        # Position sizing parameters
        st.sidebar.subheader("Position Sizing")
        max_position = st.sidebar.slider("Max Position Size (%)", 1, 20, 10) / 100
        target_vol = st.sidebar.slider("Target Volatility (%)", 5, 30, 15) / 100
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
        
        # Strategy overlays
        st.sidebar.subheader("Strategy Overlays")
        show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
        show_macd = st.sidebar.checkbox("MACD", value=True)
        show_rsi = st.sidebar.checkbox("RSI", value=True)
        show_volume = st.sidebar.checkbox("Volume", value=True)
        
        # Update configuration
        if self.risk_manager:
            try:
                self.risk_manager.update_volatility_model(
                    VolatilityModel(model_type),
                    window=window,
                    alpha=alpha
                )
                
                self.risk_manager.position_config.update({
                    "max_position_size": max_position,
                    "confidence_threshold": confidence_threshold
                })
                
                self.risk_manager.config["target_volatility"] = target_vol
                
            except Exception as e:
                st.sidebar.error(f"Error updating configuration: {e}")
        
        # Store overlay settings
        st.session_state.overlay_settings = {
            "bollinger": show_bollinger,
            "macd": show_macd,
            "rsi": show_rsi,
            "volume": show_volume
        }
    
    def _render_risk_metrics(self):
        """Render key risk metrics."""
        st.subheader("📈 Key Risk Metrics")
        
        if not self.risk_manager or not self.risk_manager.current_metrics:
            st.info("No risk metrics available. Please load portfolio data.")
            return
        
        metrics = self.risk_manager.current_metrics
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.sharpe_ratio:.3f}",
                delta=f"{metrics.sharpe_ratio - 1.0:.3f}"
            )
            st.metric(
                "VaR (95%)",
                f"{metrics.var_95:.2%}",
                delta=f"{metrics.var_95 - 0.02:.2%}"
            )
        
        with col2:
            st.metric(
                "Sortino Ratio",
                f"{metrics.sortino_ratio:.3f}",
                delta=f"{metrics.sortino_ratio - 1.0:.3f}"
            )
            st.metric(
                "CVaR (95%)",
                f"{metrics.cvar_95:.2%}",
                delta=f"{metrics.cvar_95 - 0.03:.2%}"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown:.2%}",
                delta=f"{metrics.max_drawdown + 0.15:.2%}"
            )
            st.metric(
                "Volatility",
                f"{metrics.volatility:.2%}",
                delta=f"{metrics.volatility - 0.15:.2%}"
            )
        
        with col4:
            st.metric(
                "Beta",
                f"{metrics.beta:.3f}",
                delta=f"{metrics.beta - 1.0:.3f}"
            )
            st.metric(
                "Kelly Fraction",
                f"{metrics.kelly_fraction:.3f}",
                delta=f"{metrics.kelly_fraction - 0.5:.3f}"
            )
    
    def _render_volatility_analysis(self):
        """Render volatility analysis."""
        st.subheader("📊 Volatility Analysis")
        
        if not self.risk_manager or not self.risk_manager.returns is not None:
            st.info("No returns data available for volatility analysis.")
            return
        
        # Get volatility forecast
        volatility_forecast = self.risk_manager.get_volatility_forecast()
        
        if volatility_forecast:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Current Volatility",
                    f"{volatility_forecast.current_volatility:.2%}"
                )
                st.metric(
                    "Forecasted Volatility",
                    f"{volatility_forecast.forecasted_volatility:.2%}",
                    delta=f"{volatility_forecast.forecasted_volatility - volatility_forecast.current_volatility:.2%}"
                )
            
            with col2:
                st.metric(
                    "Model Type",
                    volatility_forecast.model_type.upper()
                )
                st.metric(
                    "Confidence Interval",
                    f"[{volatility_forecast.confidence_interval[0]:.2%}, {volatility_forecast.confidence_interval[1]:.2%}]"
                )
        
        # Volatility regime
        if self.risk_manager.returns is not None:
            try:
                volatility_regime = calculate_volatility_regime(
                    self.risk_manager.returns,
                    window=60
                )
                
                if not volatility_regime.empty:
                    current_regime = volatility_regime.iloc[-1]
                    st.info(f"Current Volatility Regime: **{current_regime}**")
                    
                    # Regime distribution
                    regime_counts = volatility_regime.value_counts()
                    fig = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title="Volatility Regime Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not calculate volatility regime: {e}")
    
    def _render_position_recommendations(self):
        """Render position sizing recommendations."""
        st.subheader("🎯 Position Recommendations")
        
        # Sample symbols for demonstration
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        recommendations = []
        
        for symbol in symbols:
            try:
                # Generate sample recommendation
                base_size = 0.05  # 5% base position
                confidence = np.random.uniform(0.6, 0.9)
                
                if self.risk_manager:
                    recommendation = self.risk_manager.calculate_dynamic_position_size(
                        symbol=symbol,
                        base_position_size=base_size,
                        confidence_score=confidence
                    )
                    recommendations.append(recommendation)
                else:
                    # Fallback recommendation
                    recommendations.append({
                        "symbol": symbol,
                        "final_recommendation": base_size * confidence,
                        "confidence_score": confidence,
                        "volatility_model": "fallback"
                    })
                    
            except Exception as e:
                st.warning(f"Error calculating position for {symbol}: {e}")
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                if hasattr(rec, 'symbol'):
                    symbol = rec.symbol
                    size = rec.final_recommendation
                    confidence = rec.confidence_score
                    model = rec.volatility_model
                else:
                    symbol = rec["symbol"]
                    size = rec["final_recommendation"]
                    confidence = rec["confidence_score"]
                    model = rec["volatility_model"]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{symbol}**")
                
                with col2:
                    st.write(f"{size:.2%}")
                
                with col3:
                    st.write(f"{confidence:.1%}")
                
                st.caption(f"Model: {model}")
                st.divider()
    
    def _render_risk_alerts(self):
        """Render risk alerts and warnings."""
        st.subheader("⚠️ Risk Alerts")
        
        alerts = []
        
        if self.risk_manager and self.risk_manager.current_metrics:
            metrics = self.risk_manager.current_metrics
            
            # Check for high volatility
            if metrics.volatility > 0.25:
                alerts.append({
                    "level": "High",
                    "message": f"High volatility detected: {metrics.volatility:.2%}",
                    "color": "red"
                })
            
            # Check for high drawdown
            if metrics.max_drawdown < -0.15:
                alerts.append({
                    "level": "High",
                    "message": f"High drawdown: {metrics.max_drawdown:.2%}",
                    "color": "red"
                })
            
            # Check for low Sharpe ratio
            if metrics.sharpe_ratio < 0.5:
                alerts.append({
                    "level": "Medium",
                    "message": f"Low Sharpe ratio: {metrics.sharpe_ratio:.3f}",
                    "color": "orange"
                })
            
            # Check for high VaR
            if metrics.var_95 < -0.05:
                alerts.append({
                    "level": "Medium",
                    "message": f"High VaR: {metrics.var_95:.2%}",
                    "color": "orange"
                })
        
        if alerts:
            for alert in alerts:
                if alert["color"] == "red":
                    st.error(f"🚨 {alert['message']}")
                elif alert["color"] == "orange":
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No risk alerts - portfolio within normal parameters")
    
    def _render_risk_charts(self):
        """Render comprehensive risk charts."""
        st.subheader("📈 Risk Charts")
        
        if not self.risk_manager or self.risk_manager.returns is None:
            st.info("No data available for risk charts.")
            return
        
        returns = self.risk_manager.returns
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Cumulative Returns", "Rolling Volatility",
                "Drawdown", "VaR Over Time",
                "Correlation Matrix", "Tail Risk Distribution"
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values, name="Cumulative Returns"),
            row=1, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="30-day Volatility"),
            row=1, col=2
        )
        
        # Drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill='tonexty'),
            row=2, col=1
        )
        
        # VaR over time
        rolling_var = returns.rolling(window=60).quantile(0.05)
        fig.add_trace(
            go.Scatter(x=rolling_var.index, y=rolling_var.values, name="60-day VaR"),
            row=2, col=2
        )
        
        # Correlation matrix (if multiple assets)
        if len(returns) > 100:
            # Use rolling correlation with market (simplified)
            market_returns = returns.rolling(window=60).mean()  # Simplified market proxy
            correlation = returns.rolling(window=60).corr(market_returns)
            fig.add_trace(
                go.Scatter(x=correlation.index, y=correlation.values, name="Market Correlation"),
                row=3, col=1
            )
        
        # Tail risk distribution
        fig.add_trace(
            go.Histogram(x=returns.values, name="Returns Distribution", nbinsx=50),
            row=3, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_overlays(self):
        """Render strategy overlays based on sidebar settings."""
        st.subheader("🎛️ Strategy Overlays")
        
        overlay_settings = getattr(st.session_state, 'overlay_settings', {})
        
        if not overlay_settings:
            st.info("No overlay settings configured.")
            return
        
        # Generate sample data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        prices = 100 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Create overlay chart
        fig = go.Figure()
        
        # Base price line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add overlays based on settings
        if overlay_settings.get('bollinger', False):
            # Simple Bollinger Bands
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=sma + 2*std,
                    mode='lines',
                    name='Bollinger Upper',
                    line=dict(color='blue', dash='dash')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=sma - 2*std,
                    mode='lines',
                    name='Bollinger Lower',
                    line=dict(color='blue', dash='dash'),
                    fill='tonexty'
                )
            )
        
        if overlay_settings.get('macd', False):
            # Simple MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=macd,
                    mode='lines',
                    name='MACD',
                    line=dict(color='orange'),
                    yaxis='y2'
                )
            )
        
        if overlay_settings.get('rsi', False):
            # Simple RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple'),
                    yaxis='y3'
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Strategy Overlays",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(title="MACD", overlaying="y", side="right"),
            yaxis3=dict(title="RSI", overlaying="y", side="right", position=0.95),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume subplot if enabled
        if overlay_settings.get('volume', False):
            fig_volume = go.Figure()
            
            fig_volume.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume'
                )
            )
            
            fig_volume.update_layout(
                title="Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)


def main():
    """Main function to run the dashboard."""
    try:
        dashboard = RiskPreviewDashboard()
        dashboard.render_dashboard()
        
    except Exception as e:
        st.error(f"Error running dashboard: {e}")
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    main() 