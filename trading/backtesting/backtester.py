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

from trading.backtesting.cost_model import CostModel, get_retail_cost_config
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
        enable_leverage: bool = True,
        enable_fractional_sizing: bool = True,
        slippage_model: str = "fixed",  # "fixed", "proportional", "dynamic"
        transaction_cost_model: str = "bps",  # "fixed", "bps", "tiered"
        cost_model: Optional[CostModel] = None,  # New argument
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

        # Enhanced features
        self.enable_leverage = enable_leverage
        self.enable_fractional_sizing = enable_fractional_sizing
        self.slippage_model = slippage_model
        self.transaction_cost_model = transaction_cost_model

        # Simulated broker ledger
        self.cash_account = initial_cash
        self.equity_account = initial_cash
        self.leverage_used = 0.0
        self.margin_used = 0.0
        self.account_history = []

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
        self.performance_analyzer = PerformanceAnalyzer(
            cost_model.config if cost_model else None
        )
        self.visualizer = BacktestVisualizer()

        self.cost_model = (
            cost_model
            if cost_model is not None
            else CostModel(get_retail_cost_config(), data)
        )

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
        """Calculate position size using the position sizing engine with leverage and fractional support."""
        base_size = self.position_sizing_engine.calculate_position_size(
            method=PositionSizing.EQUAL_WEIGHTED,  # Default method
            asset=asset,
            price=price,
            strategy=strategy,
            signal=signal,
            data=self.data,
            positions=self.positions,
        )

        # Apply leverage if enabled
        if self.enable_leverage and self.leverage_used < self.max_leverage:
            leverage_multiplier = min(2.0, self.max_leverage - self.leverage_used)
            base_size *= leverage_multiplier

        # Apply fractional sizing if enabled
        if self.enable_fractional_sizing:
            # Allow fractional positions (e.g., 0.5 shares)
            return base_size
        else:
            # Round to whole shares
            return int(base_size)

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
        """Execute a trade and record it with cost model adjustments."""
        # Log decision for parity checking
        try:
            from testing.parity_checker import get_parity_checker
            
            parity_checker = get_parity_checker()
            
            # Extract features from current state (if available)
            features = {
                "price": price,
                "quantity": quantity,
                "signal": signal,
                "strategy": strategy,
            }
            
            # Create signal dict
            signal_dict = {
                "action": "buy" if trade_type == TradeType.BUY else "sell",
                "quantity": quantity,
                "price": price,
                "type": trade_type.name,
            }
            
            # Log backtest decision
            parity_checker.log_backtest_decision(
                timestamp=timestamp,
                symbol=asset,
                signal=signal_dict,
                features=features,
                context={"strategy": strategy, "backtest": True},
            )
        except Exception as e:
            # Don't fail if parity checker not available
            logger.debug(f"Could not log backtest decision for parity: {e}")
        
        position_size = self._calculate_position_size(asset, price, strategy, signal)

        # Use the robust cost model
        cost_breakdown = self.cost_model.calculate_total_cost(
            price=price,
            quantity=quantity,
            trade_type=trade_type.name.lower(),
            asset=asset,
            timestamp=timestamp,
            volume=None,  # Optionally pass volume if available
        )
        # Unpack costs
        fees = cost_breakdown["fees"]
        spread = cost_breakdown["spread"]
        slippage = cost_breakdown["slippage"]
        total_cost = cost_breakdown["total_cost"]
        effective_price = cost_breakdown["effective_price"]

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(asset, price, quantity)

        # Create trade object with cost details
        trade = Trade(
            timestamp=timestamp,
            asset=asset,
            quantity=quantity,
            price=price,
            type=trade_type,
            slippage=slippage,
            transaction_cost=fees,
            spread=spread,
            cash_balance=self.cash_account,
            portfolio_value=self._calculate_portfolio_value(),
            strategy=strategy,
            position_size=position_size,
            risk_metrics=risk_metrics,
            total_cost=total_cost,
            effective_price=effective_price,
        )

        # Update positions and accounts
        if trade_type == TradeType.BUY:
            self.positions[asset] = self.positions.get(asset, 0) + quantity
        elif trade_type == TradeType.SELL:
            self.positions[asset] = self.positions.get(asset, 0) - quantity

        # Update account ledger
        self._update_account_ledger(trade)

        # Add trade to history
        self.trades.append(trade)
        self.trade_log.append(trade.to_dict())

        # Add trade to position sizing engine history
        self.position_sizing_engine.add_trade(trade.to_dict())

        return trade

    def _update_account_ledger(self, trade: Trade) -> None:
        """Update the simulated broker ledger with trade information.

        Args:
            trade: Trade object containing trade details
        """
        try:
            # Calculate trade impact
            trade_value = trade.price * trade.quantity
            total_cost = trade.calculate_total_cost()

            if trade.type == TradeType.BUY:
                # Debit cash account
                self.cash_account -= total_cost
                # Credit equity account (position value)
                self.equity_account += trade_value
                # Update leverage
                if self.enable_leverage:
                    self.leverage_used = min(
                        self.max_leverage,
                        (self.equity_account - self.cash_account) / self.cash_account,
                    )
            else:  # SELL
                # Credit cash account
                self.cash_account += total_cost
                # Debit equity account
                self.equity_account -= trade_value
                # Update leverage
                if self.enable_leverage:
                    self.leverage_used = max(
                        0, (self.equity_account - self.cash_account) / self.cash_account
                    )

            # Record account state
            account_state = {
                "timestamp": trade.timestamp,
                "cash_account": self.cash_account,
                "equity_account": self.equity_account,
                "leverage_used": self.leverage_used,
                "margin_used": self.margin_used,
                "total_value": self.cash_account + self.equity_account,
            }
            self.account_history.append(account_state)

        except Exception as e:
            self.logger.error(f"Error updating account ledger: {e}")

    def get_account_summary(self) -> Dict[str, Any]:
        """Get current account summary.

        Returns:
            Dictionary with account information
        """
        return {
            "cash_account": self.cash_account,
            "equity_account": self.equity_account,
            "leverage_used": self.leverage_used,
            "margin_used": self.margin_used,
            "total_value": self.cash_account + self.equity_account,
            "account_history": self.account_history,
        }

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

    def _calculate_equity_curve(self) -> pd.DataFrame:
        """Calculate equity curve from trades, using true cost of each trade."""
        if not self.trades:
            return pd.DataFrame()

        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        equity_curve = pd.DataFrame(index=date_range)
        equity_curve["equity_curve"] = self.initial_cash
        equity_curve["returns"] = 0.0

        current_cash = self.initial_cash
        current_positions = {}

        for trade in self.trades:
            trade_date = trade.timestamp.date()

            # Update positions
            if trade.type == TradeType.BUY:
                current_positions[trade.asset] = (
                    current_positions.get(trade.asset, 0) + trade.quantity
                )
                current_cash -= (
                    trade.total_cost
                    if hasattr(trade, "total_cost")
                    else trade.calculate_total_cost()
                )
            elif trade.type == TradeType.SELL:
                current_positions[trade.asset] = (
                    current_positions.get(trade.asset, 0) - trade.quantity
                )
                current_cash += (
                    trade.total_cost
                    if hasattr(trade, "total_cost")
                    else trade.calculate_total_cost()
                )

            # Calculate portfolio value
            portfolio_value = current_cash
            for asset, quantity in current_positions.items():
                if asset in self.data.columns and quantity != 0:
                    asset_data = self.data[asset]
                    closest_date = asset_data.index[
                        asset_data.index.get_indexer([trade_date], method="ffill")[0]
                    ]
                    price = asset_data.loc[closest_date]
                    portfolio_value += quantity * price

            mask = equity_curve.index >= trade_date
            equity_curve.loc[mask, "equity_curve"] = portfolio_value

        equity_curve["returns"] = equity_curve["equity_curve"].pct_change()
        return equity_curve

    def process_signals_dataframe(
        self, signals_df: pd.DataFrame, fill_method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Process signals DataFrame to handle NaN values and ensure data quality.

        Args:
            signals_df: DataFrame containing trading signals
            fill_method: Method to fill NaN values ("ffill", "bfill", "drop", "zero")

        Returns:
            Processed DataFrame with NaN values handled
        """
        if signals_df is None or signals_df.empty:
            self.logger.warning("Signals DataFrame is empty or None")
            return pd.DataFrame()

        try:
            # Check for NaN values
            nan_count = signals_df.isna().sum().sum()
            if nan_count > 0:
                self.logger.info(f"Found {nan_count} NaN values in signals DataFrame")

                # Handle NaN values based on method
                if fill_method == "ffill":
                    signals_df = signals_df.fillna(method="ffill")
                    self.logger.info("Filled NaN values using forward fill")
                elif fill_method == "bfill":
                    signals_df = signals_df.fillna(method="bfill")
                    self.logger.info("Filled NaN values using backward fill")
                elif fill_method == "drop":
                    signals_df = signals_df.dropna()
                    self.logger.info("Dropped rows with NaN values")
                elif fill_method == "zero":
                    signals_df = signals_df.fillna(0)
                    self.logger.info("Filled NaN values with zero")
                else:
                    self.logger.warning(
                        f"Unknown fill method: {fill_method}, using forward fill"
                    )
                    signals_df = signals_df.fillna(method="ffill")

            # Check for infinite values
            inf_count = (
                np.isinf(signals_df.select_dtypes(include=[np.number])).sum().sum()
            )
            if inf_count > 0:
                self.logger.warning(
                    f"Found {inf_count} infinite values in signals DataFrame"
                )
                signals_df = signals_df.replace([np.inf, -np.inf], np.nan)
                signals_df = signals_df.fillna(method="ffill")

            # Validate final DataFrame
            if signals_df.isna().any().any():
                self.logger.error("NaN values still present after processing")
                return pd.DataFrame()

            self.logger.info(
                f"Successfully processed signals DataFrame: {signals_df.shape}"
            )
            return signals_df

        except Exception as e:
            self.logger.error(f"Error processing signals DataFrame: {e}")
            return pd.DataFrame()

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

    def run_rolling_window_validation(
        self,
        model,
        strategy,
        symbol: str,
        window_size: int = 252,
        step_size: int = 21,
        test_size: int = 21,
        interval: str = "1d",
        metrics: Optional[List[str]] = None,
        output_csv: Optional[str] = None,
        output_heatmap: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run rolling-window validation:
        - For each window, train model on window_1, test on window_2, roll forward.
        - Average performance metrics across all windows.
        - Output as CSV and/or heatmap.

        Args:
            model: Model instance (must support fit(X, y) and predict(X))
            strategy: Strategy instance
            symbol: Symbol to backtest
            window_size: Training window size (in bars)
            step_size: Step size to roll window (in bars)
            test_size: Test window size (in bars)
            interval: Data interval
            metrics: List of metrics to collect (default: all)
            output_csv: Path to save CSV report
            output_heatmap: Path to save heatmap image (optional)
        Returns:
            DataFrame of window-by-window metrics
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        # Get full data for symbol
        data = self.data if symbol not in self.data.columns else self.data[[symbol]]
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")
        n = len(data)
        results = []
        window_starts = range(0, n - window_size - test_size + 1, step_size)
        for start in window_starts:
            train_idx = slice(start, start + window_size)
            test_idx = slice(start + window_size, start + window_size + test_size)
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            if len(test_data) < test_size:
                continue
            # Fit model on train_data
            X_train, y_train = (
                train_data.drop("Close", axis=1, errors="ignore"),
                train_data["Close"],
            )
            X_test, y_test = (
                test_data.drop("Close", axis=1, errors="ignore"),
                test_data["Close"],
            )
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception as e:
                self.logger.error(f"Model training/prediction failed: {e}")
                continue
            # Run backtest on test window
            # (Assume strategy can use preds as signals)
            # You may need to adapt this for your strategy/model interface
            test_df = test_data.copy()
            test_df["pred"] = preds
            # Simulate simple returns as example
            test_df["returns"] = test_df["Close"].pct_change().shift(-1) * np.sign(
                test_df["pred"].diff()
            )
            perf = {
                "window_start": test_data.index[0],
                "window_end": test_data.index[-1],
                "mean_return": test_df["returns"].mean(),
                "sharpe": test_df["returns"].mean() / (test_df["returns"].std() + 1e-9),
                "drawdown": (
                    test_df["returns"].cumsum().cummax() - test_df["returns"].cumsum()
                ).max(),
                "win_rate": (test_df["returns"] > 0).mean(),
            }
            results.append(perf)
        results_df = pd.DataFrame(results)
        if output_csv:
            results_df.to_csv(output_csv, index=False)
        if output_heatmap and not results_df.empty:
            # Create a heatmap of mean_return by window
            plt.figure(figsize=(12, 4))
            data_for_heatmap = results_df[["mean_return"]].T
            sns.heatmap(data_for_heatmap, annot=True, fmt=".4f", cmap="coolwarm")
            plt.title("Rolling Window Mean Return")
            plt.xlabel("Window")
            plt.ylabel("Metric")
            plt.tight_layout()
            plt.savefig(output_heatmap)
            plt.close()
        return results_df


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
    print(
        "Warning: run_backtest function is a placeholder - use Backtester class directly"
    )

    # Return empty results
    empty_df = pd.DataFrame()
    empty_metrics = {}

    return empty_df, empty_df, empty_metrics
