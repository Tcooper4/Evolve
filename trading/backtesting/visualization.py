"""
Visualization for Backtesting

This module contains the BacktestVisualizer class for plotting and visualizing
backtest results using Plotly and Matplotlib, with graceful fallbacks.
"""

from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import Plotly and Matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

class BacktestVisualizer:
    """Visualizes backtest results using Plotly or Matplotlib."""
    def __init__(self):
        pass

    def plot_equity_curve(self, df: pd.DataFrame, use_plotly: bool = True) -> Optional[Any]:
        """Plot the equity curve of the backtest."""
        if use_plotly and PLOTLY_AVAILABLE:
            return self._plot_equity_curve_plotly(df)
        elif MPL_AVAILABLE:
            return self._plot_equity_curve_matplotlib(df)
        else:
            logger.warning("No visualization library available.")
            return None

    def _plot_equity_curve_plotly(self, df: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['equity_curve'], mode='lines', name='Equity Curve'))
        fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value')
        fig.show()
        return fig

    def _plot_equity_curve_matplotlib(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_curve'], label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return plt

    def plot_trades(self, df: pd.DataFrame, trades: pd.DataFrame, use_plotly: bool = True) -> Optional[Any]:
        """Plot trades on the price chart."""
        if use_plotly and PLOTLY_AVAILABLE:
            return self._plot_trades_plotly(df, trades)
        elif MPL_AVAILABLE:
            return self._plot_trades_matplotlib(df, trades)
        else:
            logger.warning("No visualization library available.")
            return None

    def _plot_trades_plotly(self, df: pd.DataFrame, trades: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'))
        if not trades.empty:
            buys = trades[trades['type'] == 'buy']
            sells = trades[trades['type'] == 'sell']
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', marker=dict(color='green', symbol='triangle-up'), name='Buy'))
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', marker=dict(color='red', symbol='triangle-down'), name='Sell'))
        fig.update_layout(title='Trades on Price Chart', xaxis_title='Date', yaxis_title='Price')
        fig.show()
        return fig

    def _plot_trades_matplotlib(self, df: pd.DataFrame, trades: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Price')
        if not trades.empty:
            buys = trades[trades['type'] == 'buy']
            sells = trades[trades['type'] == 'sell']
            plt.scatter(buys['timestamp'], buys['price'], marker='^', color='green', label='Buy')
            plt.scatter(sells['timestamp'], sells['price'], marker='v', color='red', label='Sell')
        plt.title('Trades on Price Chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return plt

    def plot_risk_metrics(self, risk_metrics: Dict[str, Any], use_plotly: bool = True) -> Optional[Any]:
        """Plot risk metrics as a bar chart."""
        if use_plotly and PLOTLY_AVAILABLE:
            return self._plot_risk_metrics_plotly(risk_metrics)
        elif MPL_AVAILABLE:
            return self._plot_risk_metrics_matplotlib(risk_metrics)
        else:
            logger.warning("No visualization library available.")
            return None

    def _plot_risk_metrics_plotly(self, risk_metrics: Dict[str, Any]):
        fig = go.Figure([go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values()))])
        fig.update_layout(title='Risk Metrics', xaxis_title='Metric', yaxis_title='Value')
        fig.show()
        return fig

    def _plot_risk_metrics_matplotlib(self, risk_metrics: Dict[str, Any]):
        plt.figure(figsize=(10, 5))
        plt.bar(list(risk_metrics.keys()), list(risk_metrics.values()))
        plt.title('Risk Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.show()
        return plt 