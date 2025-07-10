"""
Performance Analysis for Backtesting

This module contains the PerformanceAnalyzer class for aggregating and reporting
backtest performance metrics, attribution, and summary statistics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Aggregates and reports backtest performance metrics and attribution."""
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []

    def compute_metrics(self, df: pd.DataFrame, trade_log: pd.DataFrame) -> Dict[str, Any]:
        """Compute performance metrics from price and trade data."""
        metrics = {}
        try:
            metrics['total_return'] = (df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0]) - 1 if 'equity_curve' in df else np.nan
            
            # Use log returns for accurate compounding
            if 'returns' in df:
                log_returns = np.log1p(df['returns'])
                metrics['annualized_return'] = log_returns.mean() * 252
                metrics['volatility'] = log_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else np.nan
            else:
                metrics['annualized_return'] = np.nan
                metrics['volatility'] = np.nan
                metrics['sharpe_ratio'] = np.nan
            
            metrics['max_drawdown'] = self._max_drawdown(df['equity_curve']) if 'equity_curve' in df else np.nan
            metrics['num_trades'] = len(trade_log) if trade_log is not None else 0
            metrics['win_rate'] = (trade_log['pnl'] > 0).mean() if 'pnl' in trade_log else np.nan
            metrics['avg_win'] = trade_log[trade_log['pnl'] > 0]['pnl'].mean() if 'pnl' in trade_log else np.nan
            metrics['avg_loss'] = trade_log[trade_log['pnl'] < 0]['pnl'].mean() if 'pnl' in trade_log else np.nan
            metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] else np.nan
        except Exception as e:
            logger.warning(f"Performance metric calculation failed: {e}")
        return metrics

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def combine_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple metrics dictionaries into a summary."""
        if not metrics_list:
            return {}
        summary = {}
        try:
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m and m[key] is not None]
                if values:
                    summary[key] = np.mean(values)
                else:
                    summary[key] = np.nan
        except Exception as e:
            logger.warning(f"Failed to combine metrics: {e}")
        return summary

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add a metrics dictionary to the history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of all computed metrics."""
        return self.metrics_history.copy()

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metrics_history.clear()

    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Return a fallback metrics dictionary in case of failure."""
        return {
            'total_return': np.nan,
            'annualized_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'num_trades': 0,
            'win_rate': np.nan,
            'avg_win': np.nan,
            'avg_loss': np.nan,
            'profit_factor': np.nan
        } 