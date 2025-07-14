"""
Backtesting engine for trading strategies.

This module provides a comprehensive backtesting framework with support for:
- Multiple strategy backtesting
- Advanced position sizing (equal-weighted, risk-based, Kelly, fixed, volatility-adjusted, optimal f)
- Detailed trade logging and analysis
- Comprehensive performance metrics
- Advanced visualization capabilities
- Sophisticated risk management
"""

import logging

# Standard library imports
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Third-party imports
import pandas as pd

from trading.backtesting.performance_analysis import PerformanceAnalyzer
from trading.backtesting.position_sizing import PositionSizing, PositionSizingEngine
from trading.backtesting.risk_metrics import RiskMetricsEngine

# Local imports
from trading.backtesting.trade_models import Trade, TradeType
from trading.backtesting.visualization import BacktestVisualizer

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SLIPPAGE = 0.001  # 0.1%
DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
DEFAULT_SPREAD = 0.0005  # 0.05%


class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100000.0,
        slippage: float = DEFAULT_SLIPPAGE,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        spread: float = DEFAULT_SPREAD,
        max_leverage: float = 1.0,
        max_trades: int = 10000,
        trade_log_path: Optional[str] = None,
        position_sizing: PositionSizing = PositionSizing.EQUAL_WEIGHTED,
        risk_per_trade: float = 0.02,
        risk_free_rate: float = 0.02,
        benchmark: Optional[pd.Series] = None,
    ):
        """Initialize backtester.

        Args:
            data: Historical price data
            initial_cash: Starting cash amount
            slippage: Slippage as fraction of price
            transaction_cost: Transaction cost as fraction of trade value
            spread: Bid-ask spread as fraction of price
            max_leverage: Maximum leverage allowed
            max_trades: Maximum number of trades
            trade_log_path: Path to save trade log
            position_sizing: Position sizing method
            risk_per_trade: Risk per trade as fraction of portfolio
            risk_free_rate: Risk-free rate for calculations
            benchmark: Benchmark return series
        """
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.spread = spread
        self.max_leverage = max_leverage
        self.max_trades = max_trades
        self.trade_log_path = trade_log_path
        self.risk_per_trade = risk_per_trade
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark

        # Initialize state
        self.portfolio_values = np.zeros(len(data))
        self.trades: List[Trade] = []
        self.trade_log = []
        self.positions: Dict[str, float] = {}
        self.asset_values: Dict[str, float] = {}
        self.strategy_results: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, float] = {}

        # Initialize modular components
        self.position_sizing_engine = PositionSizingEngine(
            cash=initial_cash,
            risk_per_trade=risk_per_trade,
            risk_free_rate=risk_free_rate,
            max_leverage=max_leverage,
        )
        self.risk_metrics_engine = RiskMetricsEngine(
            risk_free_rate=risk_free_rate, period=TRADING_DAYS_PER_YEAR
        )
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _calculate_position_size(
        self, asset: str, price: float, strategy: str, signal: float
    ) -> float:
        """Calculate position size using the position sizing engine."""
        return self.position_sizing_engine.calculate_position_size(
            method=PositionSizing.EQUAL_WEIGHTED,  # Default method
            asset=asset,
            price=price,
            strategy=strategy,
            signal=signal,
            data=self.data,
            positions=self.positions,
        )

    def _calculate_risk_metrics(
        self, asset: str, price: float, quantity: float
    ) -> Dict[str, float]:
        """Calculate risk metrics using the risk metrics engine."""
        if asset not in self.data.columns:
            return {}

        returns = self.data[asset].pct_change().dropna()
        if len(returns) < 30:
            return {}

        return self.risk_metrics_engine.calculate(returns)

    def execute_trade(
        self,
        timestamp: datetime,
        asset: str,
        quantity: float,
        price: float,
        trade_type: TradeType,
        strategy: str,
        signal: float,
    ) -> Trade:
        """Execute a trade and record it."""
        # Calculate position size
        position_size = self._calculate_position_size(asset, price, strategy, signal)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(asset, price, quantity)

        # Create trade object
        trade = Trade(
            timestamp=timestamp,
            asset=asset,
            quantity=quantity,
            price=price,
            type=trade_type,
            slippage=self.slippage,
            transaction_cost=self.transaction_cost,
            spread=self.spread,
            cash_balance=self.cash,
            portfolio_value=self._calculate_portfolio_value(),
            strategy=strategy,
            position_size=position_size,
            risk_metrics=risk_metrics,
        )

        # Update positions and cash
        if trade_type == TradeType.BUY:
            self.positions[asset] = self.positions.get(asset, 0) + quantity
            self.cash -= trade.calculate_total_cost()
        elif trade_type == TradeType.SELL:
            self.positions[asset] = self.positions.get(asset, 0) - quantity
            self.cash += trade.calculate_total_cost()

        # Add trade to history
        self.trades.append(trade)
        self.trade_log.append(trade.to_dict())

        # Add trade to position sizing engine history
        self.position_sizing_engine.add_trade(trade.to_dict())

        return trade

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash
        for asset, quantity in self.positions.items():
            if asset in self.data.columns and quantity != 0:
                current_price = self.data[asset].iloc[-1]
                portfolio_value += quantity * current_price
        return portfolio_value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.trades:
            return self.performance_analyzer.get_fallback_metrics()

        # Create equity curve
        equity_curve = self._calculate_equity_curve()

        # Create trade log DataFrame
        trade_log_df = pd.DataFrame(self.trade_log)

        # Calculate metrics
        metrics = self.performance_analyzer.compute_metrics(equity_curve, trade_log_df)

        # Add risk metrics
        if "returns" in equity_curve.columns:
            risk_metrics = self.risk_metrics_engine.calculate(equity_curve["returns"])
            metrics.update(risk_metrics)

        return metrics

    def _calculate_equity_curve(self) -> pd.DataFrame:
        """Calculate equity curve from trades."""
        if not self.trades:
            return pd.DataFrame()

        # Create date range
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Initialize equity curve
        equity_curve = pd.DataFrame(index=date_range)
        equity_curve["equity_curve"] = self.initial_cash
        equity_curve["returns"] = 0.0

        # Calculate portfolio value over time
        current_cash = self.initial_cash
        current_positions = {}

        for trade in self.trades:
            trade_date = trade.timestamp.date()

            # Update positions
            if trade.type == TradeType.BUY:
                current_positions[trade.asset] = (
                    current_positions.get(trade.asset, 0) + trade.quantity
                )
                current_cash -= trade.calculate_total_cost()
            elif trade.type == TradeType.SELL:
                current_positions[trade.asset] = (
                    current_positions.get(trade.asset, 0) - trade.quantity
                )
                current_cash += trade.calculate_total_cost()

            # Calculate portfolio value
            portfolio_value = current_cash
            for asset, quantity in current_positions.items():
                if asset in self.data.columns and quantity != 0:
                    # Find closest price to trade date
                    asset_data = self.data[asset]
                    closest_date = asset_data.index[
                        asset_data.index.get_indexer([trade_date], method="ffill")[0]
                    ]
                    price = asset_data.loc[closest_date]
                    portfolio_value += quantity * price

            # Update equity curve from trade date onwards
            mask = equity_curve.index >= trade_date
            equity_curve.loc[mask, "equity_curve"] = portfolio_value

        # Calculate returns
        equity_curve["returns"] = equity_curve["equity_curve"].pct_change()

        return equity_curve

    def plot_results(self, use_plotly: bool = True) -> None:
        """Plot backtest results."""
        if not self.trades:
            self.logger.warning("No trades to plot")
            return

        # Calculate equity curve
        equity_curve = self._calculate_equity_curve()

        # Plot equity curve
        self.visualizer.plot_equity_curve(equity_curve, use_plotly=use_plotly)

        # Plot trades if we have price data
        if not equity_curve.empty and len(self.data.columns) > 0:
            asset = list(self.data.columns)[0]  # Use first asset
            price_data = pd.DataFrame(
                {
                    "close": self.data[asset],
                    "equity_curve": equity_curve["equity_curve"],
                }
            )
            trade_log_df = pd.DataFrame(self.trade_log)
            self.visualizer.plot_trades(price_data, trade_log_df, use_plotly=use_plotly)

        # Plot risk metrics
        metrics = self.get_performance_metrics()
        self.visualizer.plot_risk_metrics(metrics, use_plotly=use_plotly)

    def save_trade_log(self, filepath: Optional[str] = None) -> None:
        """Save trade log to file."""
        if not self.trade_log:
            self.logger.warning("No trades to save")
            return

        filepath = filepath or self.trade_log_path
        if not filepath:
            self.logger.warning("No filepath specified for trade log")
            return

        try:
            trade_log_df = pd.DataFrame(self.trade_log)
            trade_log_df.to_csv(filepath, index=False)
            self.logger.info(f"Trade log saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save trade log: {e}")

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades."""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl and t.pnl < 0])

        total_pnl = sum([t.pnl or 0 for t in self.trades])
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "final_portfolio_value": self._calculate_portfolio_value(),
            "total_return": (self._calculate_portfolio_value() / self.initial_cash) - 1,
        }

    def reset(self) -> None:
        """Reset backtester to initial state."""
        self.cash = self.initial_cash
        self.trades.clear()
        self.trade_log.clear()
        self.positions.clear()
        self.asset_values.clear()
        self.strategy_results.clear()
        self.risk_metrics.clear()
        self.portfolio_values = np.zeros(len(self.data))

        # Reset modular components
        self.position_sizing_engine.clear_trade_history()
        self.performance_analyzer.clear_history()

    def __del__(self):
        """Cleanup when backtester is destroyed."""
        if self.trade_log_path and self.trade_log:
            self.save_trade_log()


# Utility functions for backward compatibility


def run_backtest(
    strategy: Union[str, List[str]], plot: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run a backtest with the given strategy.

    Args:
        strategy: Strategy name or list of strategy names
        plot: Whether to plot results

    Returns:
        Tuple of (equity_curve, trade_log, metrics)
    """
    # This is a placeholder - actual implementation would depend on strategy definitions
    logger.warning(
        "run_backtest function is a placeholder - use Backtester class directly"
    )

    # Return empty results
    empty_df = pd.DataFrame()
    empty_metrics = {}

    return empty_df, empty_df, empty_metrics
