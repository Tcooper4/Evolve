"""Report generation utilities for trading analysis."""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

def generate_performance_report(
    performance_data: Dict[str, Any],
    output_dir: str,
    filename: Optional[str] = None
) -> str:
    """Generate performance report with charts and metrics.
    
    Args:
        performance_data: Dictionary containing performance data
        output_dir: Directory to save report
        filename: Optional filename for report
        
    Returns:
        Path to generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.html"
    
    # Create report content
    content = []
    
    # Add header
    content.append("<h1>Trading Performance Report</h1>")
    content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add performance metrics
    content.append("<h2>Performance Metrics</h2>")
    metrics_table = _create_metrics_table(performance_data["metrics"])
    content.append(metrics_table)
    
    # Add equity curve
    content.append("<h2>Equity Curve</h2>")
    equity_chart = _create_equity_chart(performance_data["equity_curve"])
    content.append(equity_chart)
    
    # Add drawdown chart
    content.append("<h2>Drawdown Analysis</h2>")
    drawdown_chart = _create_drawdown_chart(performance_data["drawdowns"])
    content.append(drawdown_chart)
    
    # Add trade analysis
    content.append("<h2>Trade Analysis</h2>")
    trade_charts = _create_trade_charts(performance_data["trades"])
    content.append(trade_charts)
    
    # Add risk metrics
    content.append("<h2>Risk Metrics</h2>")
    risk_table = _create_risk_table(performance_data["risk_metrics"])
    content.append(risk_table)
    
    # Combine content
    report_html = "\n".join(content)
    
    # Save report
    report_path = os.path.join(output_dir, filename)
    with open(report_path, "w") as f:
        f.write(report_html)
    
    return report_path

def generate_forecast_report(
    forecast_data: Dict[str, Any],
    output_dir: str,
    filename: Optional[str] = None
) -> str:
    """Generate forecast report with predictions and analysis.
    
    Args:
        forecast_data: Dictionary containing forecast data
        output_dir: Directory to save report
        filename: Optional filename for report
        
    Returns:
        Path to generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_report_{timestamp}.html"
    
    # Create report content
    content = []
    
    # Add header
    content.append("<h1>Price Forecast Report</h1>")
    content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add forecast chart
    content.append("<h2>Price Forecast</h2>")
    forecast_chart = _create_forecast_chart(forecast_data)
    content.append(forecast_chart)
    
    # Add forecast metrics
    content.append("<h2>Forecast Metrics</h2>")
    metrics_table = _create_forecast_metrics_table(forecast_data["metrics"])
    content.append(metrics_table)
    
    # Add confidence intervals
    content.append("<h2>Confidence Intervals</h2>")
    confidence_chart = _create_confidence_chart(forecast_data)
    content.append(confidence_chart)
    
    # Add forecast explanation
    content.append("<h2>Forecast Analysis</h2>")
    content.append(f"<p>{forecast_data['explanation']}</p>")
    
    # Add key insights
    content.append("<h2>Key Insights</h2>")
    insights_list = _create_insights_list(forecast_data["insights"])
    content.append(insights_list)
    
    # Combine content
    report_html = "\n".join(content)
    
    # Save report
    report_path = os.path.join(output_dir, filename)
    with open(report_path, "w") as f:
        f.write(report_html)
    
    return report_path

def _create_metrics_table(metrics: Dict[str, float]) -> str:
    """Create HTML table for performance metrics.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        HTML table string
    """
    table = ["<table>", "<tr><th>Metric</th><th>Value</th></tr>"]
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.2%}" if "return" in metric or "rate" in metric else f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        table.append(f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>")
    
    table.append("</table>")
    return "\n".join(table)

def _create_equity_chart(equity_curve: pd.Series) -> str:
    """Create equity curve chart.
    
    Args:
        equity_curve: Series containing equity values
        
    Returns:
        HTML chart string
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name="Equity",
            line=dict(color="blue")
        )
    )
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        showlegend=True
    )
    
    return fig.to_html(full_html=False)

def _create_drawdown_chart(drawdowns: pd.Series) -> str:
    """Create drawdown chart.
    
    Args:
        drawdowns: Series containing drawdown values
        
    Returns:
        HTML chart string
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red")
        )
    )
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        showlegend=True
    )
    
    return fig.to_html(full_html=False)

def _create_trade_charts(trades: pd.DataFrame) -> str:
    """Create trade analysis charts.
    
    Args:
        trades: DataFrame containing trade data
        
    Returns:
        HTML charts string
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Trade Returns Distribution",
            "Cumulative Trade Returns",
            "Trade Duration",
            "Win/Loss Ratio"
        )
    )
    
    # Add trade returns distribution
    fig.add_trace(
        go.Histogram(
            x=trades["returns"],
            name="Returns",
            nbinsx=50
        ),
        row=1,
        col=1
    )
    
    # Add cumulative returns
    fig.add_trace(
        go.Scatter(
            x=trades.index,
            y=trades["returns"].cumsum(),
            name="Cumulative Returns"
        ),
        row=1,
        col=2
    )
    
    # Add trade duration
    fig.add_trace(
        go.Bar(
            x=trades.index,
            y=trades["duration"],
            name="Duration"
        ),
        row=2,
        col=1
    )
    
    # Add win/loss ratio
    wins = (trades["returns"] > 0).sum()
    losses = (trades["returns"] < 0).sum()
    fig.add_trace(
        go.Pie(
            values=[wins, losses],
            labels=["Wins", "Losses"],
            name="Win/Loss"
        ),
        row=2,
        col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    
    return fig.to_html(full_html=False)

def _create_risk_table(risk_metrics: Dict[str, float]) -> str:
    """Create HTML table for risk metrics.
    
    Args:
        risk_metrics: Dictionary containing risk metrics
        
    Returns:
        HTML table string
    """
    table = ["<table>", "<tr><th>Risk Metric</th><th>Value</th></tr>"]
    
    for metric, value in risk_metrics.items():
        formatted_value = f"{value:.2%}" if "var" in metric or "volatility" in metric else f"{value:.2f}"
        table.append(f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>")
    
    table.append("</table>")
    return "\n".join(table)

def _create_forecast_chart(forecast_data: Dict[str, Any]) -> str:
    """Create forecast chart.
    
    Args:
        forecast_data: Dictionary containing forecast data
        
    Returns:
        HTML chart string
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=forecast_data["dates"],
            y=forecast_data["historical"],
            name="Historical",
            line=dict(color="blue")
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["forecast"],
            name="Forecast",
            line=dict(color="red", dash="dash")
        )
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["upper_bound"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="Upper Bound"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["lower_bound"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="Lower Bound"
        )
    )
    
    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    
    return fig.to_html(full_html=False)

def _create_forecast_metrics_table(metrics: Dict[str, float]) -> str:
    """Create HTML table for forecast metrics.
    
    Args:
        metrics: Dictionary containing forecast metrics
        
    Returns:
        HTML table string
    """
    table = ["<table>", "<tr><th>Metric</th><th>Value</th></tr>"]
    
    for metric, value in metrics.items():
        formatted_value = f"{value:.2%}" if "accuracy" in metric or "confidence" in metric else f"{value:.4f}"
        table.append(f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>")
    
    table.append("</table>")
    return "\n".join(table)

def _create_confidence_chart(forecast_data: Dict[str, Any]) -> str:
    """Create confidence intervals chart.
    
    Args:
        forecast_data: Dictionary containing forecast data
        
    Returns:
        HTML chart string
    """
    fig = go.Figure()
    
    # Add forecast with confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["forecast"],
            name="Forecast",
            line=dict(color="red")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["upper_bound"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="95% Confidence"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["lower_bound"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="Lower Bound"
        )
    )
    
    fig.update_layout(
        title="Forecast Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    
    return fig.to_html(full_html=False)

def _create_insights_list(insights: List[str]) -> str:
    """Create HTML list for key insights.
    
    Args:
        insights: List of key insights
        
    Returns:
        HTML list string
    """
    list_items = ["<ul>"]
    
    for insight in insights:
        list_items.append(f"<li>{insight}</li>")
    
    list_items.append("</ul>")
    return "\n".join(list_items) 