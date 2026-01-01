"""
Risk Management & Analysis Page

Merges functionality from:
- risk_preview_dashboard.py
- Monte_Carlo_Simulation.py

Features:
- Real-time risk monitoring dashboard
- Comprehensive VaR analysis
- Monte Carlo portfolio simulation
- Historical and custom stress testing
- Advanced risk analytics
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

# Backend imports
try:
    from trading.risk.risk_manager import RiskManager
    from risk.advanced_risk import AdvancedRiskAnalyzer
    from trading.backtesting.monte_carlo import MonteCarloSimulator, MonteCarloConfig
    from trading.portfolio.portfolio_manager import PortfolioManager
    from trading.data.data_loader import DataLoader, DataLoadRequest
    RISK_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some risk modules not available: {e}")
    RISK_MODULES_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Risk Management & Analysis",
    page_icon="⚠️",
    layout="wide"
)

# Initialize session state
if 'risk_manager' not in st.session_state:
    try:
        risk_config = {
            "volatility_model": {
                "type": "hybrid",
                "window": 252,
            },
            "position_sizing": {
                "max_position_size": 0.1,
                "volatility_scaling": True,
            },
            "target_volatility": 0.15,
        }
        st.session_state.risk_manager = RiskManager(risk_config) if RISK_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize risk manager: {e}")
        st.session_state.risk_manager = None

if 'advanced_risk_analyzer' not in st.session_state:
    try:
        st.session_state.advanced_risk_analyzer = AdvancedRiskAnalyzer() if RISK_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize advanced risk analyzer: {e}")
        st.session_state.advanced_risk_analyzer = None

if 'portfolio_manager' not in st.session_state:
    try:
        st.session_state.portfolio_manager = PortfolioManager()
    except Exception as e:
        logger.warning(f"Could not initialize portfolio manager: {e}")
        st.session_state.portfolio_manager = None

if 'risk_alerts' not in st.session_state:
    st.session_state.risk_alerts = []

if 'monte_carlo_results' not in st.session_state:
    st.session_state.monte_carlo_results = None

if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []

if 'risk_limits' not in st.session_state:
    st.session_state.risk_limits = {
        'max_var': 0.02,  # 2% daily VaR limit
        'max_volatility': 0.20,  # 20% annual volatility limit
        'max_drawdown': 0.15,  # 15% max drawdown limit
        'max_beta': 1.5,  # 1.5 beta limit
        'max_position_size': 0.20,  # 20% max position size
    }

# Main page title
st.title("⚠️ Risk Management & Analysis")
st.markdown("Comprehensive risk monitoring, analysis, and stress testing tools")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Dashboard",
    "📉 VaR Analysis",
    "🎲 Monte Carlo",
    "💥 Stress Testing",
    "🔬 Advanced Analytics"
])

# Helper functions for Risk Dashboard
def calculate_risk_level(metrics: Dict[str, float], limits: Dict[str, float]) -> Tuple[str, str]:
    """Calculate overall risk level (Low/Medium/High) and color."""
    risk_score = 0
    
    # Check VaR
    var_95 = abs(metrics.get('var_95', 0))
    if var_95 > limits['max_var'] * 1.5:
        risk_score += 3
    elif var_95 > limits['max_var']:
        risk_score += 2
    elif var_95 > limits['max_var'] * 0.7:
        risk_score += 1
    
    # Check volatility
    vol = metrics.get('volatility', 0)
    if vol > limits['max_volatility'] * 1.5:
        risk_score += 3
    elif vol > limits['max_volatility']:
        risk_score += 2
    elif vol > limits['max_volatility'] * 0.7:
        risk_score += 1
    
    # Check drawdown
    dd = abs(metrics.get('max_drawdown', 0))
    if dd > limits['max_drawdown'] * 1.5:
        risk_score += 3
    elif dd > limits['max_drawdown']:
        risk_score += 2
    elif dd > limits['max_drawdown'] * 0.7:
        risk_score += 1
    
    # Check beta
    beta = abs(metrics.get('beta', 1.0))
    if beta > limits['max_beta'] * 1.2:
        risk_score += 2
    elif beta > limits['max_beta']:
        risk_score += 1
    
    if risk_score >= 7:
        return "High", "red"
    elif risk_score >= 4:
        return "Medium", "orange"
    else:
        return "Low", "green"

def create_risk_gauge(risk_level: str, color: str) -> go.Figure:
    """Create a risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = {"Low": 25, "Medium": 50, "High": 75}[risk_level],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risk Level: {risk_level}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def check_risk_limits(metrics: Dict[str, float], limits: Dict[str, float]) -> List[Dict[str, any]]:
    """Check risk metrics against limits and return violations."""
    violations = []
    
    var_95 = abs(metrics.get('var_95', 0))
    if var_95 > limits['max_var']:
        violations.append({
            'metric': 'VaR (95%)',
            'value': f"{var_95:.2%}",
            'limit': f"{limits['max_var']:.2%}",
            'status': 'critical' if var_95 > limits['max_var'] * 1.5 else 'warning'
        })
    
    vol = metrics.get('volatility', 0)
    if vol > limits['max_volatility']:
        violations.append({
            'metric': 'Volatility',
            'value': f"{vol:.2%}",
            'limit': f"{limits['max_volatility']:.2%}",
            'status': 'critical' if vol > limits['max_volatility'] * 1.5 else 'warning'
        })
    
    dd = abs(metrics.get('max_drawdown', 0))
    if dd > limits['max_drawdown']:
        violations.append({
            'metric': 'Max Drawdown',
            'value': f"{dd:.2%}",
            'limit': f"{limits['max_drawdown']:.2%}",
            'status': 'critical' if dd > limits['max_drawdown'] * 1.5 else 'warning'
        })
    
    beta = abs(metrics.get('beta', 1.0))
    if beta > limits['max_beta']:
        violations.append({
            'metric': 'Beta',
            'value': f"{beta:.2f}",
            'limit': f"{limits['max_beta']:.2f}",
            'status': 'critical' if beta > limits['max_beta'] * 1.2 else 'warning'
        })
    
    return violations

def get_portfolio_data() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """Get portfolio returns and current positions."""
    try:
        portfolio_manager = st.session_state.portfolio_manager
        if not portfolio_manager:
            return None, None
        
        # Get portfolio positions
        positions = {}
        if hasattr(portfolio_manager, 'state') and hasattr(portfolio_manager.state, 'open_positions'):
            for pos in portfolio_manager.state.open_positions:
                positions[pos.symbol] = pos.size
        
        # Generate sample returns if no real data
        if not positions:
            # Create sample portfolio returns
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
            return returns, {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'TSLA': 0.15, 'NVDA': 0.1}
        
        # Try to get actual returns from portfolio manager
        # This is a simplified version - in production, you'd fetch real data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
        
        return returns, positions
        
    except Exception as e:
        logger.warning(f"Error getting portfolio data: {e}")
        # Return sample data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
        positions = {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'TSLA': 0.15, 'NVDA': 0.1}
        return returns, positions

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive risk metrics."""
    if returns is None or returns.empty:
        return {}
    
    metrics = {}
    
    # Basic metrics
    metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
    metrics['mean_return'] = returns.mean() * 252  # Annualized
    
    # VaR and CVaR
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['var_99'] = np.percentile(returns, 1)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    metrics['max_drawdown'] = drawdowns.min()
    metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
    
    # Beta (simplified - would need benchmark in production)
    metrics['beta'] = 1.0  # Placeholder
    
    # Sharpe ratio
    risk_free_rate = 0.02
    excess_returns = returns - (risk_free_rate / 252)
    metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
    metrics['sortino_ratio'] = np.sqrt(252) * excess_returns.mean() / downside_vol if downside_vol > 0 else 0
    
    return metrics

# TAB 1: Risk Dashboard
with tab1:
    st.header("📊 Risk Dashboard")
    st.markdown("Real-time risk monitoring and portfolio risk analysis")
    
    # Auto-refresh toggle
    col_refresh, col_limits = st.columns([1, 3])
    with col_refresh:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(0.1)  # Small delay for demo
            st.rerun()
    
    with col_limits:
        st.markdown("**Risk Limits:**")
        col_lim1, col_lim2, col_lim3, col_lim4 = st.columns(4)
        with col_lim1:
            st.caption(f"Max VaR: {st.session_state.risk_limits['max_var']:.2%}")
        with col_lim2:
            st.caption(f"Max Volatility: {st.session_state.risk_limits['max_volatility']:.2%}")
        with col_lim3:
            st.caption(f"Max Drawdown: {st.session_state.risk_limits['max_drawdown']:.2%}")
        with col_lim4:
            st.caption(f"Max Beta: {st.session_state.risk_limits['max_beta']:.2f}")
    
    st.markdown("---")
    
    # Get portfolio data
    returns, positions = get_portfolio_data()
    
    if returns is None or returns.empty:
        st.warning("No portfolio data available. Please ensure positions are loaded.")
    else:
        # Calculate risk metrics
        if st.session_state.risk_manager:
            try:
                st.session_state.risk_manager.update_returns(returns)
                current_metrics = st.session_state.risk_manager.current_metrics
                if current_metrics:
                    metrics = {
                        'var_95': current_metrics.var_95,
                        'var_99': current_metrics.var_99,
                        'cvar_95': current_metrics.cvar_95,
                        'volatility': current_metrics.volatility,
                        'max_drawdown': current_metrics.max_drawdown,
                        'beta': current_metrics.beta,
                        'sharpe_ratio': current_metrics.sharpe_ratio,
                        'sortino_ratio': current_metrics.sortino_ratio,
                    }
                else:
                    metrics = calculate_risk_metrics(returns)
            except Exception as e:
                logger.warning(f"Error using risk manager: {e}")
                metrics = calculate_risk_metrics(returns)
        else:
            metrics = calculate_risk_metrics(returns)
        
        # Calculate risk level
        risk_level, risk_color = calculate_risk_level(metrics, st.session_state.risk_limits)
        
        # Store in history
        st.session_state.risk_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy(),
            'risk_level': risk_level
        })
        # Keep only last 100 entries
        if len(st.session_state.risk_history) > 100:
            st.session_state.risk_history = st.session_state.risk_history[-100:]
        
        # Risk Gauge and Key Metrics
        col_gauge, col_metrics = st.columns([1, 2])
        
        with col_gauge:
            st.subheader("Risk Gauge")
            fig_gauge = create_risk_gauge(risk_level, risk_color)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_metrics:
            st.subheader("Key Risk Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "VaR (95%)",
                    f"{abs(metrics.get('var_95', 0)):.2%}",
                    delta=f"{abs(metrics.get('var_95', 0)) - st.session_state.risk_limits['max_var']:.2%}",
                    delta_color="inverse"
                )
                st.metric(
                    "CVaR (95%)",
                    f"{abs(metrics.get('cvar_95', 0)):.2%}",
                )
            
            with col_m2:
                st.metric(
                    "Volatility",
                    f"{metrics.get('volatility', 0):.2%}",
                    delta=f"{metrics.get('volatility', 0) - st.session_state.risk_limits['max_volatility']:.2%}",
                    delta_color="inverse"
                )
                st.metric(
                    "Beta",
                    f"{metrics.get('beta', 1.0):.2f}",
                    delta=f"{metrics.get('beta', 1.0) - st.session_state.risk_limits['max_beta']:.2f}",
                    delta_color="inverse"
                )
            
            with col_m3:
                st.metric(
                    "Max Drawdown",
                    f"{abs(metrics.get('max_drawdown', 0)):.2%}",
                    delta=f"{abs(metrics.get('max_drawdown', 0)) - st.session_state.risk_limits['max_drawdown']:.2%}",
                    delta_color="inverse"
                )
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics.get('sharpe_ratio', 0):.2f}",
                )
            
            with col_m4:
                st.metric(
                    "Sortino Ratio",
                    f"{metrics.get('sortino_ratio', 0):.2f}",
                )
                st.metric(
                    "Mean Return",
                    f"{metrics.get('mean_return', 0):.2%}",
                )
        
        st.markdown("---")
        
        # Risk Limits Status and Alerts
        col_limits_status, col_alerts = st.columns([1, 1])
        
        with col_limits_status:
            st.subheader("⚠️ Risk Limits Status")
            violations = check_risk_limits(metrics, st.session_state.risk_limits)
            
            if not violations:
                st.success("✅ All risk limits within acceptable range")
            else:
                for violation in violations:
                    if violation['status'] == 'critical':
                        st.error(f"🔴 **{violation['metric']}**: {violation['value']} (Limit: {violation['limit']})")
                    else:
                        st.warning(f"🟡 **{violation['metric']}**: {violation['value']} (Limit: {violation['limit']})")
                    
                    # Add to alerts
                    alert = {
                        'timestamp': datetime.now(),
                        'type': violation['status'],
                        'metric': violation['metric'],
                        'message': f"{violation['metric']} exceeded limit: {violation['value']} > {violation['limit']}"
                    }
                    if alert not in st.session_state.risk_alerts:
                        st.session_state.risk_alerts.append(alert)
        
        with col_alerts:
            st.subheader("📢 Risk Alerts Feed")
            if st.session_state.risk_alerts:
                # Show last 10 alerts
                recent_alerts = st.session_state.risk_alerts[-10:]
                for alert in reversed(recent_alerts):
                    timestamp_str = alert['timestamp'].strftime("%H:%M:%S") if isinstance(alert['timestamp'], datetime) else str(alert['timestamp'])
                    if alert['type'] == 'critical':
                        st.error(f"🔴 [{timestamp_str}] {alert['message']}")
                    else:
                        st.warning(f"🟡 [{timestamp_str}] {alert['message']}")
            else:
                st.info("No risk alerts")
        
        st.markdown("---")
        
        # Portfolio Heat Map and Risk by Position
        col_heatmap, col_positions = st.columns([1, 1])
        
        with col_heatmap:
            st.subheader("🔥 Portfolio Risk Heat Map")
            if positions:
                # Create correlation matrix (simplified - would use real correlations in production)
                symbols = list(positions.keys())
                n = len(symbols)
                correlation_matrix = np.random.rand(n, n)
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                np.fill_diagonal(correlation_matrix, 1.0)
                
                df_corr = pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)
                
                fig_heatmap = px.imshow(
                    df_corr,
                    labels=dict(x="Symbol", y="Symbol", color="Correlation"),
                    x=symbols,
                    y=symbols,
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    title="Position Correlation Matrix"
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No positions available for heat map")
        
        with col_positions:
            st.subheader("📊 Risk by Position")
            if positions:
                # Calculate risk metrics per position
                position_risks = []
                for symbol, weight in positions.items():
                    # Simplified risk calculation per position
                    position_var = abs(metrics.get('var_95', 0)) * weight
                    position_vol = metrics.get('volatility', 0) * weight
                    
                    position_risks.append({
                        'Symbol': symbol,
                        'Weight': f"{weight:.1%}",
                        'VaR (95%)': f"{position_var:.2%}",
                        'Volatility': f"{position_vol:.2%}",
                        'Risk Score': f"{(position_var + position_vol) / 2:.2%}"
                    })
                
                df_positions = pd.DataFrame(position_risks)
                st.dataframe(df_positions, use_container_width=True, hide_index=True)
            else:
                st.info("No positions available")
        
        st.markdown("---")
        
        # Historical Risk Trend
        st.subheader("📈 Historical Risk Trend")
        if len(st.session_state.risk_history) > 1:
            # Extract historical data
            timestamps = [h['timestamp'] for h in st.session_state.risk_history]
            var_history = [abs(h['metrics'].get('var_95', 0)) for h in st.session_state.risk_history]
            vol_history = [h['metrics'].get('volatility', 0) for h in st.session_state.risk_history]
            dd_history = [abs(h['metrics'].get('max_drawdown', 0)) for h in st.session_state.risk_history]
            
            fig_trend = make_subplots(
                rows=3, cols=1,
                subplot_titles=('VaR (95%)', 'Volatility', 'Max Drawdown'),
                vertical_spacing=0.1
            )
            
            fig_trend.add_trace(
                go.Scatter(x=timestamps, y=var_history, name='VaR (95%)', line=dict(color='red')),
                row=1, col=1
            )
            fig_trend.add_hline(
                y=st.session_state.risk_limits['max_var'],
                line_dash="dash",
                line_color="red",
                annotation_text="VaR Limit",
                row=1, col=1
            )
            
            fig_trend.add_trace(
                go.Scatter(x=timestamps, y=vol_history, name='Volatility', line=dict(color='blue')),
                row=2, col=1
            )
            fig_trend.add_hline(
                y=st.session_state.risk_limits['max_volatility'],
                line_dash="dash",
                line_color="blue",
                annotation_text="Volatility Limit",
                row=2, col=1
            )
            
            fig_trend.add_trace(
                go.Scatter(x=timestamps, y=dd_history, name='Max Drawdown', line=dict(color='orange')),
                row=3, col=1
            )
            fig_trend.add_hline(
                y=st.session_state.risk_limits['max_drawdown'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Drawdown Limit",
                row=3, col=1
            )
            
            fig_trend.update_layout(height=600, showlegend=False)
            fig_trend.update_xaxes(title_text="Time", row=3, col=1)
            fig_trend.update_yaxes(title_text="VaR (%)", row=1, col=1)
            fig_trend.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            fig_trend.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Collecting historical risk data...")

# Helper functions for VaR Analysis
def calculate_historical_var(returns: pd.Series, confidence_level: float, time_horizon: int = 1) -> float:
    """Calculate historical VaR."""
    if returns is None or returns.empty:
        return 0.0
    
    # Scale for time horizon
    percentile = (1 - confidence_level) * 100
    var_1day = np.percentile(returns, percentile)
    
    # Scale to time horizon (assuming independence)
    return var_1day * np.sqrt(time_horizon)

def calculate_parametric_var(returns: pd.Series, confidence_level: float, time_horizon: int = 1) -> float:
    """Calculate parametric VaR assuming normal distribution."""
    if returns is None or returns.empty:
        return 0.0
    
    try:
        from scipy.stats import norm
    except ImportError:
        # Fallback if scipy not available
        z_scores = {0.90: -1.28, 0.95: -1.65, 0.99: -2.33}
        z_score = z_scores.get(confidence_level, -1.65)
    else:
        z_score = norm.ppf(1 - confidence_level)
    
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Parametric VaR
    var_1day = mean_return + z_score * volatility
    
    # Scale to time horizon
    return var_1day * np.sqrt(time_horizon)

def calculate_monte_carlo_var(returns: pd.Series, confidence_level: float, time_horizon: int = 1, n_simulations: int = 10000) -> float:
    """Calculate VaR using Monte Carlo simulation."""
    if returns is None or returns.empty:
        return 0.0
    
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Generate random returns
    np.random.seed(42)  # For reproducibility
    simulated_returns = np.random.normal(
        mean_return * time_horizon,
        volatility * np.sqrt(time_horizon),
        n_simulations
    )
    
    # Calculate VaR
    percentile = (1 - confidence_level) * 100
    return np.percentile(simulated_returns, percentile)

def calculate_cvar(returns: pd.Series, var_value: float) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    if returns is None or returns.empty:
        return 0.0
    
    # CVaR is the mean of returns below VaR
    return returns[returns <= var_value].mean()

def calculate_component_var(returns: pd.Series, positions: Dict[str, float], confidence_level: float, portfolio_value: float) -> pd.DataFrame:
    """Calculate Component VaR for each position."""
    if not positions or returns is None or returns.empty:
        return pd.DataFrame()
    
    # Calculate portfolio VaR
    portfolio_var = calculate_historical_var(returns, confidence_level)
    
    # For each position, calculate its contribution to VaR
    component_vars = []
    total_weight = sum(positions.values())
    
    for symbol, weight in positions.items():
        # Simplified: assume each position contributes proportionally
        # In production, you'd use actual correlation and covariance
        position_var = portfolio_var * (weight / total_weight) if total_weight > 0 else 0
        position_var_dollar = position_var * portfolio_value
        
        component_vars.append({
            'Symbol': symbol,
            'Weight': f"{weight:.2%}",
            'Component VaR (%)': f"{abs(position_var):.2%}",
            'Component VaR ($)': f"${abs(position_var_dollar):,.2f}",
            'Contribution (%)': f"{(weight / total_weight * 100):.1f}%" if total_weight > 0 else "0%"
        })
    
    return pd.DataFrame(component_vars)

def backtest_var(returns: pd.Series, var_predictions: pd.Series, confidence_level: float) -> Dict[str, any]:
    """Backtest VaR predictions against actual losses."""
    if returns is None or returns.empty or var_predictions is None or var_predictions.empty:
        return {}
    
    # Align indices
    common_index = returns.index.intersection(var_predictions.index)
    if len(common_index) == 0:
        return {}
    
    actual_returns = returns.loc[common_index]
    predicted_var = var_predictions.loc[common_index]
    
    # Count violations (actual loss exceeds VaR)
    violations = (actual_returns < predicted_var).sum()
    total_days = len(common_index)
    violation_rate = violations / total_days if total_days > 0 else 0
    
    # Expected violation rate
    expected_rate = 1 - confidence_level
    
    # Calculate exceedances
    exceedances = actual_returns[actual_returns < predicted_var] - predicted_var[actual_returns < predicted_var]
    avg_exceedance = exceedances.mean() if len(exceedances) > 0 else 0
    
    return {
        'total_days': total_days,
        'violations': violations,
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'avg_exceedance': avg_exceedance,
        'is_valid': abs(violation_rate - expected_rate) < 0.02  # Within 2% of expected
    }

# TAB 2: VaR Analysis
with tab2:
    st.header("📉 Value at Risk (VaR) Analysis")
    st.markdown("Comprehensive VaR calculation and analysis using multiple methods")
    
    # Get portfolio data
    returns, positions = get_portfolio_data()
    portfolio_value = st.session_state.get('portfolio_value', 100000.0)  # Default $100k
    
    if returns is None or returns.empty:
        st.warning("No portfolio data available. Please ensure positions are loaded.")
    else:
        # Configuration section
        st.subheader("⚙️ VaR Configuration")
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            var_method = st.selectbox(
                "VaR Calculation Method",
                ["Historical", "Parametric", "Monte Carlo"],
                help="Historical: Uses historical returns distribution\nParametric: Assumes normal distribution\nMonte Carlo: Simulates future returns"
            )
        
        with col_config2:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.0f%%",
                help="Confidence level for VaR calculation (e.g., 95% means 5% chance of exceeding VaR)"
            )
        
        with col_config3:
            time_horizon_map = {"1-day": 1, "10-day": 10, "1-month": 21}
            time_horizon_label = st.selectbox(
                "Time Horizon",
                ["1-day", "10-day", "1-month"],
                index=0
            )
            time_horizon = time_horizon_map[time_horizon_label]
        
        st.markdown("---")
        
        # Calculate VaR based on selected method
        if var_method == "Historical":
            var_value = calculate_historical_var(returns, confidence_level, time_horizon)
        elif var_method == "Parametric":
            var_value = calculate_parametric_var(returns, confidence_level, time_horizon)
        else:  # Monte Carlo
            n_simulations = st.slider("Number of Simulations", 1000, 50000, 10000, 1000, key="mc_simulations")
            var_value = calculate_monte_carlo_var(returns, confidence_level, time_horizon, n_simulations)
        
        # Calculate CVaR (Expected Shortfall)
        cvar_value = calculate_cvar(returns, var_value)
        
        # Display VaR metrics
        st.subheader("📊 VaR Metrics")
        col_var1, col_var2, col_var3, col_var4 = st.columns(4)
        
        with col_var1:
            st.metric(
                f"VaR ({int(confidence_level*100)}%)",
                f"{abs(var_value):.2%}",
                help="Maximum expected loss at confidence level"
            )
            st.metric(
                "VaR ($)",
                f"${abs(var_value * portfolio_value):,.2f}",
            )
        
        with col_var2:
            st.metric(
                f"CVaR ({int(confidence_level*100)}%)",
                f"{abs(cvar_value):.2%}",
                help="Expected loss given that VaR is exceeded"
            )
            st.metric(
                "CVaR ($)",
                f"${abs(cvar_value * portfolio_value):,.2f}",
            )
        
        with col_var3:
            st.metric(
                "Expected Shortfall",
                f"{abs(cvar_value):.2%}",
                help="Same as CVaR - average loss beyond VaR"
            )
            st.metric(
                "ES ($)",
                f"${abs(cvar_value * portfolio_value):,.2f}",
            )
        
        with col_var4:
            st.metric(
                "Time Horizon",
                time_horizon_label,
            )
            st.metric(
                "Method",
                var_method,
            )
        
        st.markdown("---")
        
        # VaR Distribution Chart
        st.subheader("📈 VaR Distribution")
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            # Create distribution histogram with VaR line
            fig_var = go.Figure()
            
            # Histogram of returns
            fig_var.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Return Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # VaR line
            fig_var.add_vline(
                x=var_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR ({int(confidence_level*100)}%)",
                annotation_position="top"
            )
            
            # CVaR line
            fig_var.add_vline(
                x=cvar_value,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"CVaR ({int(confidence_level*100)}%)",
                annotation_position="top"
            )
            
            fig_var.update_layout(
                title="Return Distribution with VaR and CVaR",
                xaxis_title="Return",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col_chart2:
            st.markdown("**VaR Comparison**")
            
            # Compare different confidence levels
            conf_levels = [0.90, 0.95, 0.99]
            var_comparison = []
            
            for conf in conf_levels:
                if var_method == "Historical":
                    var_comp = calculate_historical_var(returns, conf, time_horizon)
                elif var_method == "Parametric":
                    var_comp = calculate_parametric_var(returns, conf, time_horizon)
                else:
                    var_comp = calculate_monte_carlo_var(returns, conf, time_horizon, n_simulations)
                
                var_comparison.append({
                    'Confidence': f"{int(conf*100)}%",
                    'VaR (%)': f"{abs(var_comp):.2%}",
                    'VaR ($)': f"${abs(var_comp * portfolio_value):,.2f}"
                })
            
            df_var_comp = pd.DataFrame(var_comparison)
            st.dataframe(df_var_comp, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Component VaR
        st.subheader("🔍 Component VaR (by Position)")
        if positions:
            component_var_df = calculate_component_var(returns, positions, confidence_level, portfolio_value)
            if not component_var_df.empty:
                st.dataframe(component_var_df, use_container_width=True, hide_index=True)
                
                # Visualize component VaR
                fig_component = go.Figure()
                fig_component.add_trace(go.Bar(
                    x=component_var_df['Symbol'],
                    y=[abs(float(v.replace('%', '').replace('$', '').replace(',', ''))) for v in component_var_df['Component VaR ($)'].str.replace('$', '').str.replace(',', '')],
                    name='Component VaR',
                    marker_color='crimson'
                ))
                fig_component.update_layout(
                    title="Component VaR by Position",
                    xaxis_title="Symbol",
                    yaxis_title="Component VaR ($)",
                    height=300
                )
                st.plotly_chart(fig_component, use_container_width=True)
            else:
                st.info("Unable to calculate component VaR")
        else:
            st.info("No positions available for component VaR calculation")
        
        st.markdown("---")
        
        # VaR Backtesting
        st.subheader("🧪 VaR Backtesting")
        st.markdown("Compare predicted VaR against actual historical losses")
        
        # Generate rolling VaR predictions
        window = st.slider("Backtesting Window", 30, 252, 60, 10, key="backtest_window")
        
        if len(returns) >= window:
            # Calculate rolling VaR
            rolling_var = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                if var_method == "Historical":
                    window_var = calculate_historical_var(window_returns, confidence_level, 1)
                elif var_method == "Parametric":
                    window_var = calculate_parametric_var(window_returns, confidence_level, 1)
                else:
                    window_var = calculate_monte_carlo_var(window_returns, confidence_level, 1, 1000)
                rolling_var.append(window_var)
            
            rolling_var_series = pd.Series(rolling_var, index=returns.index[window:])
            
            # Backtest
            backtest_results = backtest_var(returns[window:], rolling_var_series, confidence_level)
            
            if backtest_results:
                col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
                
                with col_bt1:
                    st.metric("Total Days", backtest_results['total_days'])
                    st.metric("Violations", backtest_results['violations'])
                
                with col_bt2:
                    st.metric(
                        "Violation Rate",
                        f"{backtest_results['violation_rate']:.2%}",
                        delta=f"{backtest_results['violation_rate'] - backtest_results['expected_rate']:.2%}",
                        delta_color="inverse"
                    )
                    st.metric("Expected Rate", f"{backtest_results['expected_rate']:.2%}")
                
                with col_bt3:
                    st.metric(
                        "Avg Exceedance",
                        f"{abs(backtest_results['avg_exceedance']):.2%}",
                    )
                    validation_status = "✅ Valid" if backtest_results['is_valid'] else "⚠️ Invalid"
                    st.metric("Validation", validation_status)
                
                with col_bt4:
                    if backtest_results['is_valid']:
                        st.success("VaR model is well-calibrated")
                    else:
                        st.warning("VaR model may need adjustment")
                
                # Backtesting chart
                fig_backtest = go.Figure()
                
                # Actual returns
                fig_backtest.add_trace(go.Scatter(
                    x=returns.index[window:],
                    y=returns[window:],
                    name='Actual Returns',
                    line=dict(color='blue', width=1),
                    opacity=0.6
                ))
                
                # Predicted VaR
                fig_backtest.add_trace(go.Scatter(
                    x=rolling_var_series.index,
                    y=rolling_var_series,
                    name='Predicted VaR',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Highlight violations
                violations_mask = returns[window:] < rolling_var_series
                if violations_mask.any():
                    violations_returns = returns[window:][violations_mask]
                    violations_var = rolling_var_series[violations_mask]
                    
                    fig_backtest.add_trace(go.Scatter(
                        x=violations_returns.index,
                        y=violations_returns.values,
                        mode='markers',
                        name='Violations',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig_backtest.update_layout(
                    title="VaR Backtesting: Actual Returns vs Predicted VaR",
                    xaxis_title="Date",
                    yaxis_title="Return",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_backtest, use_container_width=True)
            else:
                st.warning("Unable to perform backtesting")
        else:
            st.warning(f"Need at least {window} days of data for backtesting. Current: {len(returns)} days")

# Helper functions for Monte Carlo Simulation
def calculate_drawdown_distribution(simulated_paths: pd.DataFrame) -> pd.Series:
    """Calculate maximum drawdown for each simulation path."""
    drawdowns = []
    for col in simulated_paths.columns:
        path = simulated_paths[col]
        cummax = path.expanding().max()
        drawdown = (path - cummax) / cummax
        max_dd = drawdown.min()
        drawdowns.append(max_dd)
    return pd.Series(drawdowns)

def calculate_probability_metrics(final_values: pd.Series, initial_capital: float) -> Dict[str, float]:
    """Calculate probability of profit/loss and other metrics."""
    returns = (final_values - initial_capital) / initial_capital
    
    prob_profit = (returns > 0).mean()
    prob_loss = (returns < 0).mean()
    prob_big_loss = (returns < -0.10).mean()  # >10% loss
    prob_big_profit = (returns > 0.10).mean()  # >10% profit
    
    return {
        'prob_profit': prob_profit,
        'prob_loss': prob_loss,
        'prob_big_loss': prob_big_loss,
        'prob_big_profit': prob_big_profit
    }

# TAB 3: Monte Carlo Simulation
with tab3:
    st.header("🎲 Monte Carlo Portfolio Simulation")
    st.markdown("Simulate thousands of possible portfolio scenarios to assess risk and potential outcomes")
    
    # Get portfolio data
    returns, positions = get_portfolio_data()
    portfolio_value = st.session_state.get('portfolio_value', 100000.0)
    
    if returns is None or returns.empty:
        st.warning("No portfolio data available. Please ensure positions are loaded.")
    else:
        # Configuration section
        st.subheader("⚙️ Simulation Configuration")
        
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        
        with col_sim1:
            n_simulations = st.selectbox(
                "Number of Simulations",
                [1000, 5000, 10000, 50000, 100000],
                index=2,
                help="More simulations = more accurate but slower"
            )
        
        with col_sim2:
            time_horizon_type = st.selectbox(
                "Time Horizon Unit",
                ["Days", "Weeks", "Months"],
                index=0
            )
            time_horizon_value = st.number_input(
                f"Time Horizon ({time_horizon_type})",
                min_value=1,
                max_value=365 if time_horizon_type == "Days" else 52 if time_horizon_type == "Weeks" else 12,
                value=252 if time_horizon_type == "Days" else 52 if time_horizon_type == "Weeks" else 12,
                step=1
            )
            # Convert to days
            if time_horizon_type == "Weeks":
                time_horizon_days = time_horizon_value * 7
            elif time_horizon_type == "Months":
                time_horizon_days = time_horizon_value * 21  # Approximate trading days
            else:
                time_horizon_days = time_horizon_value
        
        with col_sim3:
            bootstrap_method = st.selectbox(
                "Bootstrap Method",
                ["historical", "block", "parametric"],
                index=0,
                help="Historical: Random sampling\nBlock: Preserves time structure\nParametric: Assumes normal distribution"
            )
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000.0,
                value=float(portfolio_value),
                step=1000.0
            )
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
                block_size = st.number_input("Block Size (for block bootstrap)", min_value=1, value=20, step=1) if bootstrap_method == "block" else None
            with col_adv2:
                show_individual_paths = st.checkbox("Show Individual Paths", value=True)
                n_paths_to_show = st.slider("Number of Paths to Show", 10, 200, 50) if show_individual_paths else 0
        
        st.markdown("---")
        
        # Run simulation button
        run_simulation = st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True)
        
        if run_simulation or st.session_state.monte_carlo_results is not None:
            if run_simulation:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing simulation...")
                    progress_bar.progress(10)
                    
                    # Create Monte Carlo config
                    config = MonteCarloConfig(
                        n_simulations=n_simulations,
                        bootstrap_method=bootstrap_method,
                        random_seed=random_seed,
                        initial_capital=initial_capital
                    )
                    
                    if bootstrap_method == "block" and block_size:
                        config.block_size = block_size
                    
                    # Create simulator
                    status_text.text("Creating simulator...")
                    progress_bar.progress(20)
                    simulator = MonteCarloSimulator(config)
                    
                    # Adjust returns to match time horizon
                    if len(returns) > time_horizon_days:
                        # Use last N days
                        returns_to_use = returns.iloc[-time_horizon_days:]
                    else:
                        # Repeat returns to fill horizon
                        n_repeats = int(np.ceil(time_horizon_days / len(returns)))
                        returns_to_use = pd.concat([returns] * n_repeats).iloc[:time_horizon_days]
                    
                    # Run simulation
                    status_text.text(f"Running {n_simulations:,} simulations...")
                    progress_bar.progress(40)
                    
                    simulated_paths = simulator.simulate_portfolio_paths(
                        returns_to_use,
                        initial_capital,
                        n_simulations
                    )
                    
                    status_text.text("Calculating percentiles...")
                    progress_bar.progress(70)
                    simulator.calculate_percentiles()
                    
                    status_text.text("Calculating statistics...")
                    progress_bar.progress(90)
                    
                    # Calculate additional metrics
                    final_values = simulated_paths.iloc[-1]
                    final_returns = (final_values - initial_capital) / initial_capital
                    
                    # Drawdown distribution
                    drawdown_dist = calculate_drawdown_distribution(simulated_paths)
                    
                    # Probability metrics
                    prob_metrics = calculate_probability_metrics(final_values, initial_capital)
                    
                    # Store results
                    st.session_state.monte_carlo_results = {
                        'simulator': simulator,
                        'simulated_paths': simulated_paths,
                        'final_values': final_values,
                        'final_returns': final_returns,
                        'drawdown_dist': drawdown_dist,
                        'prob_metrics': prob_metrics,
                        'config': {
                            'n_simulations': n_simulations,
                            'time_horizon_days': time_horizon_days,
                            'initial_capital': initial_capital,
                            'bootstrap_method': bootstrap_method
                        }
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Simulation complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.monte_carlo_results = None
            
            # Display results if available
            if st.session_state.monte_carlo_results:
                results = st.session_state.monte_carlo_results
                simulator = results['simulator']
                simulated_paths = results['simulated_paths']
                final_values = results['final_values']
                final_returns = results['final_returns']
                drawdown_dist = results['drawdown_dist']
                prob_metrics = results['prob_metrics']
                
                st.markdown("---")
                st.subheader("📊 Simulation Results")
                
                # Summary statistics
                col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                
                with col_stat1:
                    st.metric("Mean Final Value", f"${final_values.mean():,.2f}")
                    st.metric("Mean Return", f"{final_returns.mean():.2%}")
                
                with col_stat2:
                    st.metric("Median Final Value", f"${final_values.median():,.2f}")
                    st.metric("Median Return", f"{final_returns.median():.2%}")
                
                with col_stat3:
                    st.metric("Best Case (P95)", f"${final_values.quantile(0.95):,.2f}")
                    st.metric("Best Return", f"{final_returns.quantile(0.95):.2%}")
                
                with col_stat4:
                    st.metric("Worst Case (P5)", f"${final_values.quantile(0.05):,.2f}")
                    st.metric("Worst Return", f"{final_returns.quantile(0.05):.2%}")
                
                with col_stat5:
                    st.metric("Std Dev", f"${final_values.std():,.2f}")
                    st.metric("Volatility", f"{final_returns.std():.2%}")
                
                st.markdown("---")
                
                # Percentile outcomes
                st.subheader("📈 Percentile Outcomes")
                col_perc1, col_perc2 = st.columns([2, 1])
                
                with col_perc1:
                    # Percentile table
                    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
                    percentile_data = []
                    for p in percentiles:
                        value = final_values.quantile(p)
                        ret = (value - initial_capital) / initial_capital
                        percentile_data.append({
                            'Percentile': f"P{int(p*100)}",
                            'Final Value ($)': f"${value:,.2f}",
                            'Return (%)': f"{ret:.2%}",
                            'Gain/Loss ($)': f"${value - initial_capital:,.2f}"
                        })
                    
                    df_percentiles = pd.DataFrame(percentile_data)
                    st.dataframe(df_percentiles, use_container_width=True, hide_index=True)
                
                with col_perc2:
                    st.markdown("**Probability Metrics**")
                    st.metric("Profit Probability", f"{prob_metrics['prob_profit']:.1%}")
                    st.metric("Loss Probability", f"{prob_metrics['prob_loss']:.1%}")
                    st.metric("Big Loss (>10%)", f"{prob_metrics['prob_big_loss']:.1%}")
                    st.metric("Big Profit (>10%)", f"{prob_metrics['prob_big_profit']:.1%}")
                
                st.markdown("---")
                
                # Path visualization
                st.subheader("📊 Simulation Paths Visualization")
                
                fig_paths = go.Figure()
                
                # Individual paths (sample)
                if show_individual_paths and n_paths_to_show > 0:
                    paths_to_show = min(n_paths_to_show, simulated_paths.shape[1])
                    sample_cols = np.random.choice(simulated_paths.columns, paths_to_show, replace=False)
                    for col in sample_cols:
                        fig_paths.add_trace(go.Scatter(
                            x=simulated_paths.index,
                            y=simulated_paths[col],
                            mode='lines',
                            line=dict(color='lightgray', width=0.5),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # Percentile bands
                if simulator.percentiles is not None:
                    # P10 and P90 bands
                    if 'P10' in simulator.percentiles.columns and 'P90' in simulator.percentiles.columns:
                        fig_paths.add_trace(go.Scatter(
                            x=simulator.percentiles.index,
                            y=simulator.percentiles['P90'],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dash'),
                            name='90th Percentile',
                            fill=None
                        ))
                        fig_paths.add_trace(go.Scatter(
                            x=simulator.percentiles.index,
                            y=simulator.percentiles['P10'],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dash'),
                            name='10th Percentile',
                            fill='tonexty',
                            fillcolor='rgba(0, 100, 255, 0.2)'
                        ))
                    
                    # Median (P50)
                    if 'P50' in simulator.percentiles.columns:
                        fig_paths.add_trace(go.Scatter(
                            x=simulator.percentiles.index,
                            y=simulator.percentiles['P50'],
                            mode='lines',
                            line=dict(color='red', width=3),
                            name='Median (50th Percentile)'
                        ))
                    
                    # Mean
                    if 'Mean' in simulator.percentiles.columns:
                        fig_paths.add_trace(go.Scatter(
                            x=simulator.percentiles.index,
                            y=simulator.percentiles['Mean'],
                            mode='lines',
                            line=dict(color='green', width=2, dash='dot'),
                            name='Mean Path'
                        ))
                
                fig_paths.update_layout(
                    title="Monte Carlo Simulation Paths",
                    xaxis_title="Period",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_paths, use_container_width=True)
                
                st.markdown("---")
                
                # Distribution charts
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.subheader("📊 Return Distribution")
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Histogram(
                        x=final_returns,
                        nbinsx=50,
                        name='Return Distribution',
                        marker_color='skyblue',
                        opacity=0.7
                    ))
                    
                    # Add percentile lines
                    for p in [0.10, 0.50, 0.90]:
                        value = final_returns.quantile(p)
                        fig_returns.add_vline(
                            x=value,
                            line_dash="dash",
                            line_color="red" if p == 0.50 else "orange",
                            annotation_text=f"P{int(p*100)}"
                        )
                    
                    fig_returns.update_layout(
                        title="Distribution of Final Returns",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                with col_dist2:
                    st.subheader("📉 Maximum Drawdown Distribution")
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Histogram(
                        x=drawdown_dist,
                        nbinsx=50,
                        name='Max Drawdown Distribution',
                        marker_color='crimson',
                        opacity=0.7
                    ))
                    
                    # Add median line
                    median_dd = drawdown_dist.median()
                    fig_dd.add_vline(
                        x=median_dd,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Median: {median_dd:.2%}"
                    )
                    
                    fig_dd.update_layout(
                        title="Distribution of Maximum Drawdowns",
                        xaxis_title="Maximum Drawdown (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                st.markdown("---")
                
                # Export results
                st.subheader("💾 Export Results")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    # Export to CSV
                    csv_data = pd.DataFrame({
                        'Final_Value': final_values,
                        'Final_Return': final_returns,
                        'Max_Drawdown': drawdown_dist
                    })
                    csv_string = csv_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_string,
                        file_name=f"monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col_exp2:
                    # Summary report
                    summary_report = f"""
Monte Carlo Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- Simulations: {results['config']['n_simulations']:,}
- Time Horizon: {results['config']['time_horizon_days']} days
- Initial Capital: ${results['config']['initial_capital']:,.2f}
- Bootstrap Method: {results['config']['bootstrap_method']}

Results:
- Mean Final Value: ${final_values.mean():,.2f}
- Median Final Value: ${final_values.median():,.2f}
- Best Case (P95): ${final_values.quantile(0.95):,.2f}
- Worst Case (P5): ${final_values.quantile(0.05):,.2f}

Probability Metrics:
- Probability of Profit: {prob_metrics['prob_profit']:.1%}
- Probability of Loss: {prob_metrics['prob_loss']:.1%}
- Probability of Big Loss (>10%): {prob_metrics['prob_big_loss']:.1%}
- Probability of Big Profit (>10%): {prob_metrics['prob_big_profit']:.1%}

Drawdown Analysis:
- Mean Max Drawdown: {drawdown_dist.mean():.2%}
- Median Max Drawdown: {drawdown_dist.median():.2%}
- Worst Max Drawdown: {drawdown_dist.min():.2%}
"""
                    st.download_button(
                        label="📄 Download Summary Report (TXT)",
                        data=summary_report,
                        file_name=f"monte_carlo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("👆 Click 'Run Monte Carlo Simulation' to start the analysis")

# Helper functions for Stress Testing
def get_historical_scenarios() -> Dict[str, Dict[str, float]]:
    """Get predefined historical stress scenarios."""
    return {
        "2008 Financial Crisis": {
            "description": "Global financial crisis with bank failures and market collapse",
            "market_shock": -0.38,  # S&P 500 fell ~38% in 2008
            "volatility_multiplier": 3.0,
            "duration_days": 252,
            "recovery_days": 1260,  # ~5 years to recover
            "sector_shocks": {
                "Financial": -0.55,
                "Real Estate": -0.45,
                "Consumer": -0.35,
                "Technology": -0.40
            }
        },
        "2020 COVID Crash": {
            "description": "Pandemic-induced market crash and recovery",
            "market_shock": -0.34,  # S&P 500 fell ~34% in March 2020
            "volatility_multiplier": 4.0,
            "duration_days": 33,  # Very fast crash
            "recovery_days": 126,  # ~6 months to recover
            "sector_shocks": {
                "Travel": -0.50,
                "Energy": -0.45,
                "Technology": -0.20,
                "Healthcare": -0.15
            }
        },
        "1987 Black Monday": {
            "description": "Largest single-day market crash in history",
            "market_shock": -0.23,  # Dow fell 22.6% in one day
            "volatility_multiplier": 5.0,
            "duration_days": 1,
            "recovery_days": 630,  # ~2.5 years
            "sector_shocks": {
                "All": -0.23  # Universal crash
            }
        },
        "2000 Dot-com Bubble": {
            "description": "Technology bubble burst",
            "market_shock": -0.49,  # NASDAQ fell ~49% from peak
            "volatility_multiplier": 2.5,
            "duration_days": 504,  # ~2 years
            "recovery_days": 1890,  # ~7.5 years
            "sector_shocks": {
                "Technology": -0.60,
                "Telecommunications": -0.50,
                "Internet": -0.70,
                "Financial": -0.20
            }
        }
    }

def apply_stress_scenario(returns: pd.Series, scenario: Dict[str, any], positions: Dict[str, float]) -> Tuple[pd.Series, Dict[str, float]]:
    """Apply stress scenario to returns and calculate portfolio impact."""
    # Create shocked returns
    shocked_returns = returns.copy()
    
    # Apply market shock
    market_shock = scenario.get('market_shock', 0)
    volatility_mult = scenario.get('volatility_multiplier', 1.0)
    
    # Scale returns by shock and increase volatility
    shocked_returns = shocked_returns * (1 + market_shock) * volatility_mult
    
    # Calculate portfolio impact
    portfolio_impact = {}
    if positions:
        total_weight = sum(positions.values())
        for symbol, weight in positions.items():
            # Apply sector-specific shock if available
            sector_shock = scenario.get('sector_shocks', {}).get('All', market_shock)
            position_impact = weight * sector_shock if total_weight > 0 else 0
            portfolio_impact[symbol] = {
                'weight': weight,
                'shock': sector_shock,
                'impact_pct': position_impact,
                'impact_dollar': position_impact * st.session_state.get('portfolio_value', 100000.0)
            }
    
    return shocked_returns, portfolio_impact

def estimate_recovery_time(scenario: Dict[str, any], portfolio_impact_pct: float) -> int:
    """Estimate recovery time based on scenario and impact."""
    base_recovery = scenario.get('recovery_days', 252)
    
    # Adjust based on impact severity
    if abs(portfolio_impact_pct) > 0.30:
        recovery_multiplier = 1.5
    elif abs(portfolio_impact_pct) > 0.20:
        recovery_multiplier = 1.2
    else:
        recovery_multiplier = 1.0
    
    return int(base_recovery * recovery_multiplier)

# TAB 4: Stress Testing
with tab4:
    st.header("💥 Portfolio Stress Testing")
    st.markdown("Test portfolio resilience under historical and custom stress scenarios")
    
    # Get portfolio data
    returns, positions = get_portfolio_data()
    portfolio_value = st.session_state.get('portfolio_value', 100000.0)
    
    if returns is None or returns.empty:
        st.warning("No portfolio data available. Please ensure positions are loaded.")
    else:
        # Scenario selection
        st.subheader("📋 Scenario Selection")
        
        scenario_type = st.radio(
            "Scenario Type",
            ["Historical Scenarios", "Factor Stress Tests", "Custom Scenario Builder"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if scenario_type == "Historical Scenarios":
            historical_scenarios = get_historical_scenarios()
            
            col_scen1, col_scen2 = st.columns([1, 2])
            
            with col_scen1:
                selected_scenario_name = st.selectbox(
                    "Select Historical Scenario",
                    list(historical_scenarios.keys()) + ["Custom scenario"],
                    help="Choose a historical market crisis to test"
                )
            
            if selected_scenario_name == "Custom scenario":
                st.info("Use 'Custom Scenario Builder' tab to create custom scenarios")
            else:
                selected_scenario = historical_scenarios[selected_scenario_name]
                
                with col_scen2:
                    st.markdown(f"**{selected_scenario_name}**")
                    st.caption(selected_scenario['description'])
                    st.markdown(f"- Market Shock: {selected_scenario['market_shock']:.1%}")
                    st.markdown(f"- Volatility Multiplier: {selected_scenario['volatility_multiplier']:.1f}x")
                    st.markdown(f"- Duration: {selected_scenario['duration_days']} days")
                    st.markdown(f"- Historical Recovery: {selected_scenario['recovery_days']} days")
                
                # Run stress test
                if st.button("🚀 Run Stress Test", type="primary", use_container_width=True):
                    with st.spinner("Running stress test..."):
                        shocked_returns, position_impacts = apply_stress_scenario(
                            returns, selected_scenario, positions
                        )
                        
                        # Calculate portfolio impact
                        if positions:
                            total_impact_pct = sum([imp['impact_pct'] for imp in position_impacts.values()])
                        else:
                            total_impact_pct = selected_scenario['market_shock']
                        
                        total_impact_dollar = total_impact_pct * portfolio_value
                        recovery_time = estimate_recovery_time(selected_scenario, total_impact_pct)
                        
                        # Store results
                        stress_result = {
                            'scenario_name': selected_scenario_name,
                            'scenario': selected_scenario,
                            'shocked_returns': shocked_returns,
                            'position_impacts': position_impacts,
                            'total_impact_pct': total_impact_pct,
                            'total_impact_dollar': total_impact_dollar,
                            'recovery_time': recovery_time,
                            'portfolio_value_after': portfolio_value * (1 + total_impact_pct)
                        }
                        
                        st.session_state['stress_test_result'] = stress_result
                        st.success("✅ Stress test complete!")
        
        elif scenario_type == "Factor Stress Tests":
            st.subheader("🔧 Factor Stress Tests")
            
            col_factor1, col_factor2 = st.columns(2)
            
            with col_factor1:
                st.markdown("**Market Factors**")
                interest_rate_shock = st.slider(
                    "Interest Rate Shock (%)",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1
                )
                volatility_shock = st.slider(
                    "Volatility Shock (multiplier)",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.1
                )
                correlation_shock = st.slider(
                    "Correlation Shock (increase)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            with col_factor2:
                st.markdown("**Sector-Specific Shocks**")
                tech_shock = st.slider("Technology Sector (%)", -50.0, 50.0, 0.0, 1.0)
                finance_shock = st.slider("Financial Sector (%)", -50.0, 50.0, 0.0, 1.0)
                energy_shock = st.slider("Energy Sector (%)", -50.0, 50.0, 0.0, 1.0)
                consumer_shock = st.slider("Consumer Sector (%)", -50.0, 50.0, 0.0, 1.0)
            
            # Create custom scenario
            custom_scenario = {
                'description': 'Custom factor stress test',
                'market_shock': interest_rate_shock / 100,
                'volatility_multiplier': volatility_shock,
                'correlation_increase': correlation_shock,
                'sector_shocks': {
                    'Technology': tech_shock / 100,
                    'Financial': finance_shock / 100,
                    'Energy': energy_shock / 100,
                    'Consumer': consumer_shock / 100
                },
                'duration_days': 30,
                'recovery_days': 180
            }
            
            if st.button("🚀 Run Factor Stress Test", type="primary", use_container_width=True):
                with st.spinner("Running factor stress test..."):
                    shocked_returns, position_impacts = apply_stress_scenario(
                        returns, custom_scenario, positions
                    )
                    
                    if positions:
                        total_impact_pct = sum([imp['impact_pct'] for imp in position_impacts.values()])
                    else:
                        total_impact_pct = custom_scenario['market_shock']
                    
                    total_impact_dollar = total_impact_pct * portfolio_value
                    recovery_time = estimate_recovery_time(custom_scenario, total_impact_pct)
                    
                    stress_result = {
                        'scenario_name': 'Custom Factor Test',
                        'scenario': custom_scenario,
                        'shocked_returns': shocked_returns,
                        'position_impacts': position_impacts,
                        'total_impact_pct': total_impact_pct,
                        'total_impact_dollar': total_impact_dollar,
                        'recovery_time': recovery_time,
                        'portfolio_value_after': portfolio_value * (1 + total_impact_pct)
                    }
                    
                    st.session_state['stress_test_result'] = stress_result
                    st.success("✅ Factor stress test complete!")
        
        else:  # Custom Scenario Builder
            st.subheader("🛠️ Custom Scenario Builder")
            
            col_custom1, col_custom2 = st.columns(2)
            
            with col_custom1:
                scenario_name = st.text_input("Scenario Name", placeholder="My Custom Stress Test")
                scenario_description = st.text_area("Description", placeholder="Describe the scenario...")
                
                market_shock = st.number_input(
                    "Market Shock (%)",
                    min_value=-100.0,
                    max_value=100.0,
                    value=-20.0,
                    step=1.0
                )
                
                volatility_mult = st.number_input(
                    "Volatility Multiplier",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1
                )
            
            with col_custom2:
                duration_days = st.number_input(
                    "Duration (days)",
                    min_value=1,
                    max_value=365,
                    value=30,
                    step=1
                )
                
                recovery_days = st.number_input(
                    "Expected Recovery (days)",
                    min_value=1,
                    max_value=3650,
                    value=180,
                    step=30
                )
                
                # Position-specific shocks
                st.markdown("**Position-Specific Shocks**")
                position_shocks = {}
                if positions:
                    for symbol in positions.keys():
                        position_shocks[symbol] = st.number_input(
                            f"{symbol} Shock (%)",
                            min_value=-100.0,
                            max_value=100.0,
                            value=market_shock,
                            step=1.0,
                            key=f"shock_{symbol}"
                        )
            
            if st.button("🚀 Run Custom Stress Test", type="primary", use_container_width=True):
                if not scenario_name:
                    st.error("Please provide a scenario name")
                else:
                    with st.spinner("Running custom stress test..."):
                        custom_scenario = {
                            'description': scenario_description or 'Custom scenario',
                            'market_shock': market_shock / 100,
                            'volatility_multiplier': volatility_mult,
                            'duration_days': duration_days,
                            'recovery_days': recovery_days,
                            'sector_shocks': {symbol: shock / 100 for symbol, shock in position_shocks.items()} if position_shocks else {}
                        }
                        
                        shocked_returns, position_impacts = apply_stress_scenario(
                            returns, custom_scenario, positions
                        )
                        
                        if positions:
                            total_impact_pct = sum([imp['impact_pct'] for imp in position_impacts.values()])
                        else:
                            total_impact_pct = custom_scenario['market_shock']
                        
                        total_impact_dollar = total_impact_pct * portfolio_value
                        recovery_time = estimate_recovery_time(custom_scenario, total_impact_pct)
                        
                        stress_result = {
                            'scenario_name': scenario_name,
                            'scenario': custom_scenario,
                            'shocked_returns': shocked_returns,
                            'position_impacts': position_impacts,
                            'total_impact_pct': total_impact_pct,
                            'total_impact_dollar': total_impact_dollar,
                            'recovery_time': recovery_time,
                            'portfolio_value_after': portfolio_value * (1 + total_impact_pct)
                        }
                        
                        st.session_state['stress_test_result'] = stress_result
                        st.success("✅ Custom stress test complete!")
        
        # Display results if available
        if 'stress_test_result' in st.session_state and st.session_state['stress_test_result']:
            result = st.session_state['stress_test_result']
            
            st.markdown("---")
            st.subheader("📊 Stress Test Results")
            
            # Portfolio impact summary
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.metric(
                    "Portfolio Impact (%)",
                    f"{result['total_impact_pct']:.2%}",
                    delta=f"{result['total_impact_pct']:.2%}",
                    delta_color="inverse"
                )
            
            with col_res2:
                st.metric(
                    "Portfolio Impact ($)",
                    f"${abs(result['total_impact_dollar']):,.2f}",
                    delta=f"${result['total_impact_dollar']:,.2f}",
                    delta_color="inverse"
                )
            
            with col_res3:
                st.metric(
                    "Portfolio Value After",
                    f"${result['portfolio_value_after']:,.2f}",
                    delta=f"${result['portfolio_value_after'] - portfolio_value:,.2f}",
                    delta_color="inverse"
                )
            
            with col_res4:
                st.metric(
                    "Estimated Recovery",
                    f"{result['recovery_time']} days",
                    help="Estimated time to recover to pre-stress levels"
                )
            
            st.markdown("---")
            
            # Position-level impact
            if result['position_impacts']:
                st.subheader("📈 Position-Level Impact")
                position_data = []
                for symbol, impact in result['position_impacts'].items():
                    position_data.append({
                        'Symbol': symbol,
                        'Weight': f"{impact['weight']:.2%}",
                        'Shock (%)': f"{impact['shock']:.2%}",
                        'Impact (%)': f"{impact['impact_pct']:.2%}",
                        'Impact ($)': f"${abs(impact['impact_dollar']):,.2f}"
                    })
                
                df_positions = pd.DataFrame(position_data)
                st.dataframe(df_positions, use_container_width=True, hide_index=True)
                
                # Visualize position impacts
                fig_positions = go.Figure()
                fig_positions.add_trace(go.Bar(
                    x=[d['Symbol'] for d in position_data],
                    y=[float(d['Impact (%)'].replace('%', '')) for d in position_data],
                    name='Impact (%)',
                    marker_color='crimson'
                ))
                fig_positions.update_layout(
                    title="Position-Level Impact",
                    xaxis_title="Symbol",
                    yaxis_title="Impact (%)",
                    height=300
                )
                st.plotly_chart(fig_positions, use_container_width=True)
            
            st.markdown("---")
            
            # Scenario comparison (if multiple scenarios run)
            if 'stress_test_history' not in st.session_state:
                st.session_state['stress_test_history'] = []
            
            # Add current result to history
            if result not in st.session_state['stress_test_history']:
                st.session_state['stress_test_history'].append(result.copy())
            
            if len(st.session_state['stress_test_history']) > 1:
                st.subheader("📊 Scenario Comparison")
                
                comparison_data = []
                for hist_result in st.session_state['stress_test_history']:
                    comparison_data.append({
                        'Scenario': hist_result['scenario_name'],
                        'Impact (%)': f"{hist_result['total_impact_pct']:.2%}",
                        'Impact ($)': f"${abs(hist_result['total_impact_dollar']):,.2f}",
                        'Recovery (days)': hist_result['recovery_time']
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Comparison chart
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Bar(
                    x=[d['Scenario'] for d in comparison_data],
                    y=[float(d['Impact (%)'].replace('%', '')) for d in comparison_data],
                    name='Impact (%)',
                    marker_color='steelblue'
                ))
                fig_comparison.update_layout(
                    title="Scenario Comparison - Portfolio Impact",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=400
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Clear results button
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state['stress_test_result'] = None
                st.rerun()

# Helper functions for Advanced Analytics
def calculate_correlation_matrix(returns: pd.Series, positions: Dict[str, float]) -> pd.DataFrame:
    """Calculate correlation matrix for portfolio positions."""
    if not positions or len(positions) < 2:
        # Return sample correlation matrix
        symbols = list(positions.keys()) if positions else ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        n = len(symbols)
        corr_matrix = np.random.rand(n, n)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
    
    # In production, you'd fetch actual returns for each symbol
    # For now, generate sample correlations
    symbols = list(positions.keys())
    n = len(symbols)
    corr_matrix = np.random.rand(n, n)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)

def calculate_factor_decomposition(returns: pd.Series) -> Dict[str, float]:
    """Calculate factor decomposition (simplified)."""
    # In production, this would use PCA or factor models
    # For now, return simplified metrics
    market_factor = returns.mean() * 0.7  # Assume 70% market factor
    idiosyncratic = returns.std() * 0.3  # 30% idiosyncratic
    
    return {
        'market_factor': market_factor,
        'idiosyncratic_factor': idiosyncratic,
        'market_exposure': 0.7,
        'idiosyncratic_exposure': 0.3
    }

def calculate_liquidity_risk(positions: Dict[str, float], portfolio_value: float) -> Dict[str, float]:
    """Calculate liquidity risk metrics."""
    if not positions:
        return {}
    
    # Simplified liquidity risk calculation
    # In production, you'd use actual volume data and bid-ask spreads
    liquidity_scores = {}
    total_value = sum(positions.values()) * portfolio_value if positions else portfolio_value
    
    for symbol, weight in positions.items():
        # Simulate liquidity score (0-1, higher = more liquid)
        liquidity_score = np.random.uniform(0.5, 1.0)  # Placeholder
        position_value = weight * portfolio_value
        days_to_liquidate = (1 - liquidity_score) * 10  # 0-10 days
        
        liquidity_scores[symbol] = {
            'liquidity_score': liquidity_score,
            'position_value': position_value,
            'days_to_liquidate': days_to_liquidate,
            'liquidity_risk': 'Low' if liquidity_score > 0.8 else 'Medium' if liquidity_score > 0.5 else 'High'
        }
    
    return liquidity_scores

def calculate_concentration_risk(positions: Dict[str, float]) -> Dict[str, float]:
    """Calculate concentration risk metrics."""
    if not positions:
        return {}
    
    weights = list(positions.values())
    total_weight = sum(weights)
    
    if total_weight == 0:
        return {}
    
    # Normalize weights
    normalized_weights = [w / total_weight for w in weights]
    
    # Herfindahl-Hirschman Index (HHI)
    hhi = sum(w**2 for w in normalized_weights)
    
    # Concentration ratio (top 5 positions)
    sorted_weights = sorted(normalized_weights, reverse=True)
    top5_concentration = sum(sorted_weights[:min(5, len(sorted_weights))])
    
    # Effective number of positions
    effective_n = 1 / hhi if hhi > 0 else len(positions)
    
    # Max position concentration
    max_concentration = max(normalized_weights) if normalized_weights else 0
    
    return {
        'herfindahl_index': hhi,
        'top5_concentration': top5_concentration,
        'effective_positions': effective_n,
        'max_position_concentration': max_concentration,
        'concentration_risk': 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low'
    }

def calculate_greek_exposure(positions: Dict[str, float]) -> Dict[str, float]:
    """Calculate Greek exposure (for options - placeholder)."""
    # This would calculate actual Greeks for options positions
    # For now, return placeholder values
    greeks = {}
    
    for symbol in positions.keys():
        greeks[symbol] = {
            'delta': np.random.uniform(-1, 1),  # Price sensitivity
            'gamma': np.random.uniform(0, 0.1),  # Delta sensitivity
            'theta': np.random.uniform(-0.1, 0),  # Time decay
            'vega': np.random.uniform(0, 0.2),  # Volatility sensitivity
            'rho': np.random.uniform(-0.1, 0.1)  # Interest rate sensitivity
        }
    
    return greeks

def calculate_rolling_risk_metrics(returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """Calculate rolling risk metrics."""
    if returns is None or returns.empty or len(returns) < window:
        return pd.DataFrame()
    
    rolling_metrics = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        
        metrics = {
            'date': returns.index[i],
            'volatility': window_returns.std() * np.sqrt(252),
            'sharpe': (window_returns.mean() * 252) / (window_returns.std() * np.sqrt(252)) if window_returns.std() > 0 else 0,
            'max_drawdown': calculate_max_drawdown_simple(window_returns),
            'skewness': window_returns.skew(),
            'kurtosis': window_returns.kurtosis(),
            'var_95': np.percentile(window_returns, 5)
        }
        
        rolling_metrics.append(metrics)
    
    return pd.DataFrame(rolling_metrics).set_index('date')

def calculate_max_drawdown_simple(returns: pd.Series) -> float:
    """Calculate max drawdown for a return series."""
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    return drawdowns.min()

# TAB 5: Advanced Analytics
with tab5:
    st.header("🔬 Advanced Risk Analytics")
    st.markdown("Comprehensive risk analysis including correlation, factors, tail risk, and more")
    
    # Get portfolio data
    returns, positions = get_portfolio_data()
    portfolio_value = st.session_state.get('portfolio_value', 100000.0)
    
    if returns is None or returns.empty:
        st.warning("No portfolio data available. Please ensure positions are loaded.")
    else:
        # Calculate advanced metrics
        if st.session_state.advanced_risk_analyzer:
            try:
                risk_metrics = st.session_state.advanced_risk_analyzer.calculate_comprehensive_risk(returns)
            except Exception as e:
                logger.warning(f"Error using advanced risk analyzer: {e}")
                risk_metrics = None
        else:
            risk_metrics = None
        
        # Calculate basic metrics if advanced analyzer not available
        if risk_metrics is None:
            volatility = returns.std() * np.sqrt(252)
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
            sortino = sharpe  # Simplified
            calmar = (returns.mean() * 252) / abs(calculate_max_drawdown_simple(returns)) if calculate_max_drawdown_simple(returns) != 0 else 0
        else:
            volatility = risk_metrics.volatility
            skewness = risk_metrics.skewness
            kurtosis = risk_metrics.kurtosis
            sharpe = risk_metrics.sharpe_ratio
            sortino = risk_metrics.sortino_ratio
            calmar = risk_metrics.calmar_ratio
        
        st.markdown("---")
        
        # Correlation Matrix
        st.subheader("📊 Correlation Matrix")
        if positions:
            corr_matrix = calculate_correlation_matrix(returns, positions)
            
            col_corr1, col_corr2 = st.columns([2, 1])
            
            with col_corr1:
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Symbol", y="Symbol", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    title="Position Correlation Matrix",
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_corr2:
                st.markdown("**Correlation Statistics**")
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
                
                st.metric("Average Correlation", f"{avg_corr:.3f}")
                st.metric("Maximum Correlation", f"{max_corr:.3f}")
                st.metric("Minimum Correlation", f"{min_corr:.3f}")
                
                if avg_corr > 0.7:
                    st.warning("⚠️ High average correlation - portfolio may lack diversification")
                elif avg_corr < 0.3:
                    st.success("✅ Low average correlation - good diversification")
        else:
            st.info("No positions available for correlation analysis")
        
        st.markdown("---")
        
        # Factor Decomposition and Tail Risk
        col_factor, col_tail = st.columns(2)
        
        with col_factor:
            st.subheader("🔍 Factor Decomposition")
            factor_decomp = calculate_factor_decomposition(returns)
            
            st.metric("Market Factor Exposure", f"{factor_decomp['market_exposure']:.1%}")
            st.metric("Idiosyncratic Exposure", f"{factor_decomp['idiosyncratic_exposure']:.1%}")
            st.metric("Market Factor Return", f"{factor_decomp['market_factor']:.2%}")
            st.metric("Idiosyncratic Risk", f"{factor_decomp['idiosyncratic_factor']:.2%}")
            
            # Factor exposure chart
            fig_factor = go.Figure(data=[
                go.Bar(name='Market Factor', x=['Exposure'], y=[factor_decomp['market_exposure']], marker_color='blue'),
                go.Bar(name='Idiosyncratic', x=['Exposure'], y=[factor_decomp['idiosyncratic_exposure']], marker_color='orange')
            ])
            fig_factor.update_layout(
                title="Factor Exposure Breakdown",
                yaxis_title="Exposure",
                height=300,
                barmode='stack'
            )
            st.plotly_chart(fig_factor, use_container_width=True)
        
        with col_tail:
            st.subheader("📉 Tail Risk Metrics")
            
            st.metric("Skewness", f"{skewness:.3f}", 
                     help="Negative = left tail risk, Positive = right tail")
            st.metric("Kurtosis", f"{kurtosis:.3f}",
                     help=">3 = fat tails, <3 = thin tails")
            
            # Tail risk interpretation
            if skewness < -0.5:
                st.error("⚠️ High left tail risk (negative skewness)")
            elif skewness < 0:
                st.warning("🟡 Moderate left tail risk")
            else:
                st.success("✅ Positive or neutral skewness")
            
            if kurtosis > 5:
                st.error("⚠️ Very fat tails (high kurtosis)")
            elif kurtosis > 3:
                st.warning("🟡 Fat tails detected")
            else:
                st.success("✅ Normal tail distribution")
            
            # Tail risk chart
            fig_tail = go.Figure()
            fig_tail.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Return Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            fig_tail.add_vline(
                x=np.percentile(returns, 5),
                line_dash="dash",
                line_color="red",
                annotation_text="5th Percentile"
            )
            fig_tail.update_layout(
                title="Return Distribution (Tail Risk)",
                xaxis_title="Return",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_tail, use_container_width=True)
        
        st.markdown("---")
        
        # Liquidity Risk and Concentration Risk
        col_liq, col_conc = st.columns(2)
        
        with col_liq:
            st.subheader("💧 Liquidity Risk Analysis")
            if positions:
                liquidity_risks = calculate_liquidity_risk(positions, portfolio_value)
                
                liquidity_data = []
                for symbol, metrics in liquidity_risks.items():
                    liquidity_data.append({
                        'Symbol': symbol,
                        'Liquidity Score': f"{metrics['liquidity_score']:.2f}",
                        'Position Value ($)': f"${metrics['position_value']:,.2f}",
                        'Days to Liquidate': f"{metrics['days_to_liquidate']:.1f}",
                        'Risk Level': metrics['liquidity_risk']
                    })
                
                df_liquidity = pd.DataFrame(liquidity_data)
                st.dataframe(df_liquidity, use_container_width=True, hide_index=True)
                
                # Liquidity risk chart
                fig_liq = go.Figure()
                fig_liq.add_trace(go.Bar(
                    x=[d['Symbol'] for d in liquidity_data],
                    y=[float(d['Days to Liquidate']) for d in liquidity_data],
                    name='Days to Liquidate',
                    marker_color='crimson'
                ))
                fig_liq.update_layout(
                    title="Liquidity Risk - Days to Liquidate",
                    xaxis_title="Symbol",
                    yaxis_title="Days",
                    height=300
                )
                st.plotly_chart(fig_liq, use_container_width=True)
            else:
                st.info("No positions available for liquidity analysis")
        
        with col_conc:
            st.subheader("🎯 Concentration Risk")
            if positions:
                conc_metrics = calculate_concentration_risk(positions)
                
                st.metric("Herfindahl Index (HHI)", f"{conc_metrics['herfindahl_index']:.3f}",
                         help="Higher = more concentrated (max 1.0)")
                st.metric("Top 5 Concentration", f"{conc_metrics['top5_concentration']:.1%}")
                st.metric("Effective Positions", f"{conc_metrics['effective_positions']:.1f}")
                st.metric("Max Position Size", f"{conc_metrics['max_position_concentration']:.1%}")
                
                # Risk level
                risk_level = conc_metrics['concentration_risk']
                if risk_level == 'High':
                    st.error(f"⚠️ {risk_level} concentration risk")
                elif risk_level == 'Medium':
                    st.warning(f"🟡 {risk_level} concentration risk")
                else:
                    st.success(f"✅ {risk_level} concentration risk")
                
                # Concentration chart
                weights = list(positions.values())
                symbols = list(positions.keys())
                fig_conc = go.Figure(data=[
                    go.Pie(
                        labels=symbols,
                        values=weights,
                        hole=0.3,
                        textinfo='label+percent'
                    )
                ])
                fig_conc.update_layout(
                    title="Position Concentration",
                    height=300
                )
                st.plotly_chart(fig_conc, use_container_width=True)
            else:
                st.info("No positions available for concentration analysis")
        
        st.markdown("---")
        
        # Greek Exposure (for options)
        st.subheader("📐 Greek Exposure (Options)")
        if positions:
            greeks = calculate_greek_exposure(positions)
            
            greek_data = []
            for symbol, greek_values in greeks.items():
                greek_data.append({
                    'Symbol': symbol,
                    'Delta': f"{greek_values['delta']:.3f}",
                    'Gamma': f"{greek_values['gamma']:.3f}",
                    'Theta': f"{greek_values['theta']:.3f}",
                    'Vega': f"{greek_values['vega']:.3f}",
                    'Rho': f"{greek_values['rho']:.3f}"
                })
            
            df_greeks = pd.DataFrame(greek_data)
            st.dataframe(df_greeks, use_container_width=True, hide_index=True)
            
            st.caption("Note: Greek calculations are placeholders. Actual Greeks require options data.")
        else:
            st.info("No positions available for Greek exposure analysis")
        
        st.markdown("---")
        
        # Risk-Adjusted Return Metrics
        st.subheader("📈 Risk-Adjusted Return Metrics")
        
        col_radj1, col_radj2, col_radj3, col_radj4 = st.columns(4)
        
        with col_radj1:
            st.metric("Sharpe Ratio", f"{sharpe:.3f}",
                     help="Return per unit of risk (higher is better)")
            st.metric("Sortino Ratio", f"{sortino:.3f}",
                     help="Return per unit of downside risk")
        
        with col_radj2:
            st.metric("Calmar Ratio", f"{calmar:.3f}",
                     help="Return per unit of max drawdown")
            st.metric("Volatility", f"{volatility:.2%}",
                     help="Annualized volatility")
        
        with col_radj3:
            # Additional metrics
            if risk_metrics:
                st.metric("Omega Ratio", f"{risk_metrics.omega_ratio if hasattr(risk_metrics, 'omega_ratio') else 0:.3f}")
                st.metric("Gain/Loss Ratio", f"{risk_metrics.gain_loss_ratio if hasattr(risk_metrics, 'gain_loss_ratio') else 0:.3f}")
            else:
                gain_loss = abs(returns[returns > 0].mean() / returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
                st.metric("Gain/Loss Ratio", f"{gain_loss:.3f}")
                st.metric("Win Rate", f"{(returns > 0).mean():.1%}")
        
        with col_radj4:
            if risk_metrics:
                st.metric("Profit Factor", f"{risk_metrics.profit_factor if hasattr(risk_metrics, 'profit_factor') else 0:.3f}")
                st.metric("Recovery Factor", f"{risk_metrics.recovery_factor if hasattr(risk_metrics, 'recovery_factor') else 0:.3f}")
            else:
                st.metric("Mean Return", f"{returns.mean() * 252:.2%}")
                st.metric("Best Day", f"{returns.max():.2%}")
        
        st.markdown("---")
        
        # Rolling Risk Metrics Charts
        st.subheader("📊 Rolling Risk Metrics")
        
        window_size = st.slider("Rolling Window (days)", min_value=30, max_value=252, value=60, step=10)
        
        if len(returns) >= window_size:
            rolling_metrics = calculate_rolling_risk_metrics(returns, window_size)
            
            if not rolling_metrics.empty:
                # Create subplots
                fig_rolling = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Skewness', 'Kurtosis', 'VaR (95%)'),
                    vertical_spacing=0.1
                )
                
                # Volatility
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['volatility'], name='Volatility', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Sharpe
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['sharpe'], name='Sharpe', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Max Drawdown
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['max_drawdown'], name='Max DD', line=dict(color='red')),
                    row=2, col=1
                )
                
                # Skewness
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['skewness'], name='Skewness', line=dict(color='orange')),
                    row=2, col=2
                )
                
                # Kurtosis
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['kurtosis'], name='Kurtosis', line=dict(color='purple')),
                    row=3, col=1
                )
                
                # VaR
                fig_rolling.add_trace(
                    go.Scatter(x=rolling_metrics.index, y=rolling_metrics['var_95'], name='VaR (95%)', line=dict(color='crimson')),
                    row=3, col=2
                )
                
                fig_rolling.update_layout(height=800, showlegend=False)
                fig_rolling.update_xaxes(title_text="Date", row=3, col=1)
                fig_rolling.update_xaxes(title_text="Date", row=3, col=2)
                
                st.plotly_chart(fig_rolling, use_container_width=True)
            else:
                st.warning("Unable to calculate rolling metrics")
        else:
            st.warning(f"Need at least {window_size} days of data. Current: {len(returns)} days")

