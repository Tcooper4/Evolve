"""Base UI components for the trading system.

This module provides reusable UI components that can be used across different pages
and can be integrated with agentic systems for monitoring and control.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

from .config.registry import registry, ModelConfig, StrategyConfig

logger = logging.getLogger(__name__)

def create_date_range_selector(
    default_days: int = 30,
    min_days: int = 1,
    max_days: int = 365,
    key: str = "date_range"
) -> Tuple[datetime, datetime]:
    """Create a date range selector with validation.
    
    Args:
        default_days: Default number of days to look back
        min_days: Minimum allowed days
        max_days: Maximum allowed days
        key: Unique key for the Streamlit component
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=default_days)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=start_date,
            min_value=end_date - timedelta(days=max_days),
            max_value=end_date,
            key=f"{key}_start"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=end_date,
            min_value=start_date,
            max_value=datetime.now(),
            key=f"{key}_end"
        )
    
    # Validate date range
    if (end_date - start_date).days < min_days:
        st.warning(f"Date range must be at least {min_days} days")

    return start_date, end_date

def create_model_selector(
    category: Optional[str] = None,
    default_model: Optional[str] = None,
    key: str = "model_selector"
) -> Optional[str]:
    """Create a model selector with dynamic options from registry.
    
    Args:
        category: Optional category to filter models
        default_model: Optional default model to select
        key: Unique key for the Streamlit component
        
    Returns:
        Selected model name or None if no selection
    """
    models = registry.get_available_models(category)
    if not models:
        st.warning("No models available for the selected category")

    model_names = [model.name for model in models]
    selected_model = st.selectbox(
        "Select Model",
        options=model_names,
        index=model_names.index(default_model) if default_model in model_names else 0,
        key=key
    )
    
    # Log selection for agentic monitoring
    logger.info(f"Model selected: {selected_model}")
    
    return selected_model

def create_strategy_selector(
    category: Optional[str] = None,
    default_strategy: Optional[str] = None,
    key: str = "strategy_selector"
) -> Optional[str]:
    """Create a strategy selector with dynamic options from registry.
    
    Args:
        category: Optional category to filter strategies
        default_strategy: Optional default strategy to select
        key: Unique key for the Streamlit component
        
    Returns:
        Selected strategy name or None if no selection
    """
    strategies = registry.get_available_strategies(category)
    if not strategies:
        st.warning("No strategies available for the selected category")

    strategy_names = [strategy.name for strategy in strategies]
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=strategy_names,
        index=strategy_names.index(default_strategy) if default_strategy in strategy_names else 0,
        key=key
    )
    
    # Log selection for agentic monitoring
    logger.info(f"Strategy selected: {selected_strategy}")
    
    return selected_strategy

def create_parameter_inputs(
    parameters: Dict[str, Union[str, float, int, bool]],
    key_prefix: str = "param"
) -> Dict[str, Any]:
    """Create input fields for model or strategy parameters.
    
    Args:
        parameters: Dictionary of parameter names and default values
        key_prefix: Prefix for Streamlit component keys
        
    Returns:
        Dictionary of parameter values
    """
    values = {}
    for param_name, default_value in parameters.items():
        key = f"{key_prefix}_{param_name}"
        
        if isinstance(default_value, bool):
            values[param_name] = st.checkbox(param_name, value=default_value, key=key)
        elif isinstance(default_value, int):
            values[param_name] = st.number_input(
                param_name,
                value=default_value,
                step=1,
                key=key
            )
        elif isinstance(default_value, float):
            values[param_name] = st.number_input(
                param_name,
                value=default_value,
                step=0.01,
                key=key
            )
        else:
            values[param_name] = st.text_input(param_name, value=str(default_value), key=key)
    
    # Log parameter changes for agentic monitoring
    logger.info(f"Parameters updated: {values}")
    
    return values

def create_asset_selector(
    assets: List[str],
    default_asset: Optional[str] = None,
    key: str = "asset_selector"
) -> Optional[str]:
    """Create an asset selector with validation.
    
    Args:
        assets: List of available assets
        default_asset: Optional default asset to select
        key: Unique key for the Streamlit component
        
    Returns:
        Selected asset symbol or None if no selection
    """
    if not assets:
        st.warning("No assets available")

    selected_asset = st.selectbox(
        "Select Asset",
        options=assets,
        index=assets.index(default_asset) if default_asset in assets else 0,
        key=key
    )
    
    # Log selection for agentic monitoring
    logger.info(f"Asset selected: {selected_asset}")
    
    return selected_asset

def create_timeframe_selector(
    timeframes: List[str],
    default_timeframe: Optional[str] = None,
    key: str = "timeframe_selector"
) -> Optional[str]:
    """Create a timeframe selector with validation.
    
    Args:
        timeframes: List of available timeframes
        default_timeframe: Optional default timeframe to select
        key: Unique key for the Streamlit component
        
    Returns:
        Selected timeframe or None if no selection
    """
    if not timeframes:
        st.warning("No timeframes available")

    selected_timeframe = st.selectbox(
        "Select Timeframe",
        options=timeframes,
        index=timeframes.index(default_timeframe) if default_timeframe in timeframes else 0,
        key=key
    )
    
    # Log selection for agentic monitoring
    logger.info(f"Timeframe selected: {selected_timeframe}")
    
    return selected_timeframe

def create_confidence_interval(
    data: pd.DataFrame,
    confidence_level: float = 0.95
) -> Tuple[pd.Series, pd.Series]:
    """Calculate confidence intervals for predictions.
    
    Args:
        data: DataFrame containing predictions and standard errors
        confidence_level: Confidence level for the interval
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = 1.96  # 95% confidence interval
    lower_bound = data['prediction'] - z_score * data['std_error']
    upper_bound = data['prediction'] + z_score * data['std_error']
    return {'success': True, 'result': lower_bound, upper_bound, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def create_benchmark_overlay(
    data: pd.DataFrame,
    benchmark_column: str,
    prediction_column: str
) -> go.Figure:
    """Create a plot with benchmark overlay.
    
    Args:
        data: DataFrame containing predictions and benchmark data
        benchmark_column: Name of the benchmark column
        prediction_column: Name of the prediction column
        
    Returns:
        Plotly figure with benchmark overlay
    """
    fig = go.Figure()
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[prediction_column],
        name='Prediction',
        line=dict(color='blue')
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[benchmark_column],
        name='Benchmark',
        line=dict(color='gray', dash='dash')
    ))
    
    # Add confidence interval if available
    if 'std_error' in data.columns:
        lower, upper = create_confidence_interval(data)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
    
    fig.update_layout(
        title='Prediction with Benchmark Overlay',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified'
    )
    
    return fig

def create_prompt_input() -> Dict[str, Any]:
    """Create a prompt input component.
    
    Returns:
        Dictionary with prompt text and metadata
    """
    try:
        prompt = st.text_area(
            "Enter your prompt",
            placeholder="Describe what you want to analyze or predict...",
            height=100,
            key="prompt_input"
        )
        
        if prompt:
            return {
                'success': True,
                'prompt': prompt,
                'length': len(prompt),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'prompt': '',
                'message': 'No prompt entered',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error creating prompt input: {e}")
        return {
            'success': False,
            'prompt': '',
            'message': f'Error creating prompt input: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def create_sidebar() -> Dict[str, Any]:
    """Create the main sidebar with navigation and settings.
    
    Returns:
        Dictionary with sidebar configuration and selections
    """
    try:
        with st.sidebar:
            st.title("Evolve Trading")
            
            # Navigation
            page = st.selectbox(
                "Navigation",
                ["Dashboard", "Forecast", "Strategy", "Analysis", "Settings"],
                key="nav_select"
            )
            
            # Model settings
            st.subheader("Model Settings")
            model_type = st.selectbox(
                "Model Type",
                ["LSTM", "Transformer", "Ensemble", "Custom"],
                key="model_type"
            )
            
            # Strategy settings
            st.subheader("Strategy Settings")
            strategy_type = st.selectbox(
                "Strategy Type",
                ["Momentum", "Mean Reversion", "Breakout", "Custom"],
                key="strategy_type"
            )
            
            # Risk settings
            st.subheader("Risk Settings")
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=100,
                value=10,
                key="max_position"
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=50,
                value=5,
                key="stop_loss"
            )
            
            # System settings
            st.subheader("System Settings")
            auto_refresh = st.checkbox("Auto Refresh", value=True, key="auto_refresh")
            debug_mode = st.checkbox("Debug Mode", value=False, key="debug_mode")
            
            return {
                'success': True,
                'page': page,
                'model_type': model_type,
                'strategy_type': strategy_type,
                'max_position_size': max_position_size,
                'stop_loss': stop_loss,
                'auto_refresh': auto_refresh,
                'debug_mode': debug_mode,
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error creating sidebar: {e}")
        return {
            'success': False,
            'message': f'Error creating sidebar: {str(e)}',
            'page': 'Dashboard',
            'timestamp': datetime.now().isoformat()
        }

def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    title: str = "Price Forecast"
) -> go.Figure:
    """Create an interactive forecast chart.
    
    Args:
        historical_data: Historical price data
        forecast_data: Forecast data with confidence intervals
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            name='Historical',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['upper'],
            name='Upper Bound',
            line=dict(color='gray', dash='dot'),
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['lower'],
            name='Lower Bound',
            line=dict(color='gray', dash='dot'),
            fill='tonexty',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        height=600
    )
    
    return fig

def create_strategy_chart(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    title: str = "Strategy Signals"
) -> go.Figure:
    """Create an interactive strategy chart.
    
    Args:
        data: Price data
        signals: Trading signals
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Add price data
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'],
            name='Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add signals
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['signal'],
            name='Signal',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        showlegend=True,
        height=600
    )
    
    return fig

def create_performance_report(
    results: Dict[str, Any],
    title: str = "Performance Report"
) -> Dict[str, Any]:
    """Create an expandable performance report section.
    
    Args:
        results: Dictionary containing performance metrics
        title: Section title
        
    Returns:
        Dictionary with report creation status and metrics summary
    """
    with st.expander(title, expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
        
        with col2:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
            st.metric("Avg Trade", f"{results.get('avg_trade', 0):.2%}")
            st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        
        with col3:
            st.metric("CAGR", f"{results.get('cagr', 0):.2%}")
            st.metric("Volatility", f"{results.get('volatility', 0):.2%}")
            st.metric("Calmar Ratio", f"{results.get('calmar_ratio', 0):.2f}")
        
        # Create equity curve
        if 'equity_curve' in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve'].values,
                    name='Equity',
                    line=dict(color='blue')
                )
            )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create drawdown chart
        if 'drawdowns' in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results['drawdowns'].index,
                    y=results['drawdowns'].values,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red')
                )
            )
            fig.update_layout(
                title="Drawdowns",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Log report creation for agentic monitoring
    logger.info(f"Performance report created: {title}")
    
    return {
        'success': True,
        'title': title,
        'metrics_count': len(results),
        'has_equity_curve': 'equity_curve' in results,
        'has_drawdowns': 'drawdowns' in results,
        'key_metrics': {
            'total_return': results.get('total_return', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0)
        }
    }

def create_error_block(message: str) -> Dict[str, Any]:
    """Create an error message block.
    
    Args:
        message: Error message to display
        
    Returns:
        Dictionary with error display status
    """
    st.error(message)
    st.info("Try adjusting your parameters or selecting a different strategy.")
    
    # Log error for agentic monitoring
    logger.error(f"Error block displayed: {message}")
    
    return {
        'success': True,
        'error_displayed': True,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }

def create_loading_spinner(message: str = "Processing...") -> Dict[str, Any]:
    """Create a loading spinner with message.
    
    Args:
        message: Message to display during loading
        
    Returns:
        Dictionary with spinner creation status
    """
    spinner = st.spinner(message)
    
    # Log spinner creation for agentic monitoring
    logger.info(f"Loading spinner created: {message}")
    
    return {
        'success': True,
        'spinner_created': True,
        'message': message,
        'spinner_object': spinner
    }

def create_forecast_metrics(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create forecast metrics display from forecast results.
    
    Args:
        forecast_results: Dictionary containing forecast results and metrics
        
    Returns:
        Dictionary with metrics display status
    """
    st.subheader("Forecast Metrics")
    
    metrics = forecast_results.get('metrics', {})
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    
    with col2:
        st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
    
    with col3:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    
    with col4:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    
    # Log metrics for agentic monitoring
    logger.info(f"Forecast metrics displayed for {forecast_results.get('ticker', 'unknown')}")
    
    return {
        'success': True,
        'metrics_displayed': True,
        'ticker': forecast_results.get('ticker', 'unknown'),
        'metrics_count': len(metrics),
        'key_metrics': {
            'accuracy': metrics.get('accuracy', 0),
            'mse': metrics.get('mse', 0),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0)
        }
    }

def create_forecast_table(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create forecast table display from forecast results.
    
    Args:
        forecast_results: Dictionary containing forecast results
        
    Returns:
        Dictionary with table creation status
    """
    st.subheader("Forecast Results")
    
    # Create a simple table with forecast information
    forecast_data = {
        'Metric': ['Ticker', 'Model', 'Strategy', 'Prediction'],
        'Value': [
            forecast_results.get('ticker', 'N/A'),
            forecast_results.get('model', 'N/A'),
            forecast_results.get('strategy', 'N/A'),
            forecast_results.get('prediction', 'N/A')
        ]
    }
    
    df = pd.DataFrame(forecast_data)
    st.table(df)
    
    # Log table creation for agentic monitoring
    logger.info(f"Forecast table displayed for {forecast_results.get('ticker', 'unknown')}")
    
    return {
        'success': True,
        'table_created': True,
        'ticker': forecast_results.get('ticker', 'unknown'),
        'table_rows': len(forecast_data['Metric']),
        'forecast_data': forecast_data
    }

def create_system_metrics_panel(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create a system-wide metrics panel showing key performance indicators.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        Dictionary with panel creation status and metrics summary
    """
    st.subheader("📊 System Performance Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sharpe = metrics.get('sharpe_ratio', 0.0)
        if sharpe >= 1.5:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Excellent", delta_color="normal")
        elif sharpe >= 1.0:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Good", delta_color="normal")
        elif sharpe >= 0.5:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Fair", delta_color="off")
        else:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Poor", delta_color="inverse")
    
    with col2:
        total_return = metrics.get('total_return', 0.0)
        if total_return >= 0.20:
            st.metric("Total Return", f"{total_return:.1%}", delta="Strong", delta_color="normal")
        elif total_return >= 0.10:
            st.metric("Total Return", f"{total_return:.1%}", delta="Good", delta_color="normal")
        elif total_return >= 0.05:
            st.metric("Total Return", f"{total_return:.1%}", delta="Moderate", delta_color="off")
        else:
            st.metric("Total Return", f"{total_return:.1%}", delta="Weak", delta_color="inverse")
    
    with col3:
        max_dd = metrics.get('max_drawdown', 0.0)
        if max_dd <= 0.10:
            st.metric("Max Drawdown", f"{max_dd:.1%}", delta="Low Risk", delta_color="normal")
        elif max_dd <= 0.20:
            st.metric("Max Drawdown", f"{max_dd:.1%}", delta="Moderate", delta_color="off")
        elif max_dd <= 0.30:
            st.metric("Max Drawdown", f"{max_dd:.1%}", delta="High Risk", delta_color="off")
        else:
            st.metric("Max Drawdown", f"{max_dd:.1%}", delta="Very High Risk", delta_color="inverse")
    
    with col4:
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate >= 0.60:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Excellent", delta_color="normal")
        elif win_rate >= 0.50:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Good", delta_color="normal")
        elif win_rate >= 0.40:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Fair", delta_color="off")
        else:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Poor", delta_color="inverse")
    
    with col5:
        pnl = metrics.get('total_pnl', 0.0)
        if pnl >= 10000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Strong", delta_color="normal")
        elif pnl >= 5000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Good", delta_color="normal")
        elif pnl >= 1000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Moderate", delta_color="off")
        else:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Weak", delta_color="inverse")
    
    # Log panel creation for agentic monitoring
    logger.info(f"System metrics panel created with {len(metrics)} metrics")
    
    return {
        'success': True,
        'panel_created': True,
        'metrics_count': len(metrics),
        'key_metrics': {
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'total_return': metrics.get('total_return', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'total_pnl': metrics.get('total_pnl', 0.0)
        }
    } 