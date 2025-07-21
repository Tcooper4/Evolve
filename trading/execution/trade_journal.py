"""
Trade Journal

This module provides comprehensive trade tracking and analysis with
performance metrics and PnL calculations.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record."""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    status: str = "open"  # 'open', 'closed', 'cancelled'
    strategy: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSummary:
    """Trade summary statistics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    average_return_per_trade: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    average_holding_period: timedelta
    total_volume: float
    total_commission: float
    total_slippage: float
    start_date: datetime
    end_date: datetime
    symbols_traded: List[str]
    strategies_used: List[str]


class TradeJournal:
    """
    Comprehensive trade journal with performance tracking and analysis.

    Features:
    - Individual trade tracking
    - Performance metrics calculation
    - PnL analysis
    - Risk metrics
    - Trade summarization
    - Export capabilities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trade journal.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []

        # Performance tracking
        self.daily_pnl: Dict[str, float] = {}
        self.monthly_pnl: Dict[str, float] = {}

        # Configuration
        self.auto_calculate_pnl = self.config.get("auto_calculate_pnl", True)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2% annual
        self.min_trades_for_metrics = self.config.get("min_trades_for_metrics", 5)

        # File storage
        self.storage_path = Path(self.config.get("storage_path", "data/trade_journal"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing trades
        self._load_trades()

        self.logger.info("Trade journal initialized")

    def add_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        strategy: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Add a new trade to the journal.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            entry_price: Entry price
            strategy: Trading strategy name
            notes: Trade notes
            **kwargs: Additional trade parameters

        Returns:
            Trade ID
        """
        try:
            # Generate trade ID
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.trades):04d}"

            # Create trade
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=datetime.now(),
                strategy=strategy,
                notes=notes,
                metadata=kwargs,
            )

            # Store trade
            self.trades[trade_id] = trade
            self.trade_history.append(trade)

            # Update daily PnL
            self._update_daily_pnl(trade)

            self.logger.info(
                f"Added trade {trade_id}: {side} {quantity} {symbol} @ {entry_price}"
            )

            return trade_id

        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")
            raise

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Close a trade with exit information.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_time: Exit time (default: now)
            commission: Commission paid
            slippage: Slippage incurred
            notes: Additional notes

        Returns:
            True if successful
        """
        try:
            if trade_id not in self.trades:
                raise ValueError(f"Trade {trade_id} not found")

            trade = self.trades[trade_id]

            if trade.status != "open":
                raise ValueError(f"Trade {trade_id} is not open")

            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = exit_time or datetime.now()
            trade.commission = commission
            trade.slippage = slippage
            trade.status = "closed"

            if notes:
                trade.notes = f"{trade.notes or ''}\n{notes}".strip()

            # Calculate PnL
            if self.auto_calculate_pnl:
                self._calculate_trade_pnl(trade)

            # Update daily PnL
            self._update_daily_pnl(trade)

            # Save trades
            self._save_trades()

            self.logger.info(
                f"Closed trade {trade_id}: PnL = {trade.pnl:.2f} "
                f"({trade.pnl_percentage:.2%})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {e}")
            return False

    def _calculate_trade_pnl(self, trade: Trade) -> None:
        """Calculate PnL for a trade."""
        if trade.exit_price is None:
            return

        # Calculate raw PnL
        if trade.side == "buy":
            # Long position
            raw_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:
            # Short position
            raw_pnl = (trade.entry_price - trade.exit_price) * trade.quantity

        # Subtract costs
        total_costs = trade.commission + (
            trade.slippage * trade.quantity * trade.entry_price
        )
        trade.pnl = raw_pnl - total_costs

        # Calculate percentage return
        trade_value = trade.quantity * trade.entry_price
        trade.pnl_percentage = trade.pnl / trade_value if trade_value > 0 else 0.0

    def _update_daily_pnl(self, trade: Trade) -> None:
        """Update daily PnL tracking."""
        if trade.entry_time:
            date_key = trade.entry_time.date().isoformat()
            self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + (
                trade.pnl or 0
            )

        if trade.exit_time:
            date_key = trade.exit_time.date().isoformat()
            self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + (
                trade.pnl or 0
            )

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a specific trade."""
        return self.trades.get(trade_id)

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [trade for trade in self.trades.values() if trade.status == "open"]

    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [trade for trade in self.trades.values() if trade.status == "closed"]

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get all trades for a specific symbol."""
        return [trade for trade in self.trades.values() if trade.symbol == symbol]

    def get_trades_by_strategy(self, strategy: str) -> List[Trade]:
        """Get all trades for a specific strategy."""
        return [trade for trade in self.trades.values() if trade.strategy == strategy]

    def get_trades_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """Get trades within a date range."""
        return [
            trade
            for trade in self.trades.values()
            if trade.entry_time and start_date <= trade.entry_time <= end_date
        ]

    def summarize(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> TradeSummary:
        """
        Generate comprehensive trade summary.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            TradeSummary object
        """
        try:
            # Filter trades by date range
            if start_date or end_date:
                trades = self.get_trades_by_date_range(
                    start_date or datetime.min, end_date or datetime.max
                )
            else:
                trades = list(self.trades.values())

            # Only consider closed trades for summary
            closed_trades = [
                trade
                for trade in trades
                if trade.status == "closed" and trade.pnl is not None
            ]

            if not closed_trades:
                return self._create_empty_summary()

            # Basic statistics
            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

            # PnL statistics
            total_pnl = sum(trade.pnl for trade in closed_trades)
            total_pnl_percentage = sum(trade.pnl_percentage for trade in closed_trades)
            average_return_per_trade = (
                total_pnl / total_trades if total_trades > 0 else 0.0
            )

            # Win/Loss statistics
            average_win = (
                np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            )
            average_loss = (
                np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            )
            largest_win = max([t.pnl for t in closed_trades]) if closed_trades else 0.0
            largest_loss = min([t.pnl for t in closed_trades]) if closed_trades else 0.0

            # Risk metrics
            profit_factor = (
                abs(
                    sum(t.pnl for t in winning_trades)
                    / sum(t.pnl for t in losing_trades)
                )
                if losing_trades
                else float("inf")
            )

            # Calculate Sharpe ratio
            returns = [t.pnl_percentage for t in closed_trades]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(returns)

            # Holding period statistics
            holding_periods = []
            for trade in closed_trades:
                if trade.entry_time and trade.exit_time:
                    holding_periods.append(trade.exit_time - trade.entry_time)

            average_holding_period = (
                np.mean(holding_periods) if holding_periods else timedelta(0)
            )

            # Volume and cost statistics
            total_volume = sum(
                trade.quantity * trade.entry_price for trade in closed_trades
            )
            total_commission = sum(trade.commission for trade in closed_trades)
            total_slippage = sum(
                trade.slippage * trade.quantity * trade.entry_price
                for trade in closed_trades
            )

            # Date range
            entry_times = [t.entry_time for t in closed_trades if t.entry_time]
            exit_times = [t.exit_time for t in closed_trades if t.exit_time]

            start_date = min(entry_times) if entry_times else datetime.now()
            end_date = max(exit_times) if exit_times else datetime.now()

            # Symbols and strategies
            symbols_traded = list(set(trade.symbol for trade in closed_trades))
            strategies_used = list(
                set(trade.strategy for trade in closed_trades if trade.strategy)
            )

            summary = TradeSummary(
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_percentage=total_pnl_percentage,
                average_return_per_trade=average_return_per_trade,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                average_holding_period=average_holding_period,
                total_volume=total_volume,
                total_commission=total_commission,
                total_slippage=total_slippage,
                start_date=start_date,
                end_date=end_date,
                symbols_traded=symbols_traded,
                strategies_used=strategies_used,
            )

            self.logger.info(
                f"Generated trade summary: {total_trades} trades, "
                f"PnL: {total_pnl:.2f}, Win rate: {win_rate:.2%}"
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating trade summary: {e}")
            return self._create_empty_summary()

    def _create_empty_summary(self) -> TradeSummary:
        """Create empty trade summary."""
        return TradeSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_percentage=0.0,
            average_return_per_trade=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            average_holding_period=timedelta(0),
            total_volume=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            symbols_traded=[],
            strategies_used=[],
        )

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < self.min_trades_for_metrics:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (
            self.risk_free_rate / 252
        )  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        )  # Annualized
        return sharpe

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return abs(np.min(drawdown))

    def export_trades(self, filepath: str, format: str = "csv") -> bool:
        """
        Export trades to file.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel')

        Returns:
            True if successful
        """
        try:
            trades_data = []
            for trade in self.trades.values():
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_time": (
                        trade.entry_time.isoformat() if trade.entry_time else None
                    ),
                    "exit_time": (
                        trade.exit_time.isoformat() if trade.exit_time else None
                    ),
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "pnl": trade.pnl,
                    "pnl_percentage": trade.pnl_percentage,
                    "status": trade.status,
                    "strategy": trade.strategy,
                    "notes": trade.notes,
                    "metadata": trade.metadata,
                }
                trades_data.append(trade_dict)

            df = pd.DataFrame(trades_data)

            if format.lower() == "csv":
                df.to_csv(filepath, index=False)
            elif format.lower() == "json":
                df.to_json(filepath, orient="records", indent=2)
            elif format.lower() == "excel":
                df.to_excel(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Exported {len(trades_data)} trades to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting trades: {e}")
            return False

    def _save_trades(self) -> None:
        """Save trades to file."""
        try:
            filepath = self.storage_path / "trades.json"
            trades_data = []

            for trade in self.trades.values():
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_time": (
                        trade.entry_time.isoformat() if trade.entry_time else None
                    ),
                    "exit_time": (
                        trade.exit_time.isoformat() if trade.exit_time else None
                    ),
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "pnl": trade.pnl,
                    "pnl_percentage": trade.pnl_percentage,
                    "status": trade.status,
                    "strategy": trade.strategy,
                    "notes": trade.notes,
                    "metadata": trade.metadata,
                }
                trades_data.append(trade_dict)

            with open(filepath, "w") as f:
                json.dump(trades_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving trades: {e}")

    def _load_trades(self) -> None:
        """Load trades from file."""
        try:
            filepath = self.storage_path / "trades.json"
            if not filepath.exists():
                return

            with open(filepath, "r") as f:
                trades_data = json.load(f)

            for trade_dict in trades_data:
                trade = Trade(
                    trade_id=trade_dict["trade_id"],
                    symbol=trade_dict["symbol"],
                    side=trade_dict["side"],
                    quantity=trade_dict["quantity"],
                    entry_price=trade_dict["entry_price"],
                    exit_price=trade_dict.get("exit_price"),
                    entry_time=(
                        datetime.fromisoformat(trade_dict["entry_time"])
                        if trade_dict.get("entry_time")
                        else None
                    ),
                    exit_time=(
                        datetime.fromisoformat(trade_dict["exit_time"])
                        if trade_dict.get("exit_time")
                        else None
                    ),
                    commission=trade_dict.get("commission", 0.0),
                    slippage=trade_dict.get("slippage", 0.0),
                    pnl=trade_dict.get("pnl"),
                    pnl_percentage=trade_dict.get("pnl_percentage"),
                    status=trade_dict.get("status", "open"),
                    strategy=trade_dict.get("strategy"),
                    notes=trade_dict.get("notes"),
                    metadata=trade_dict.get("metadata", {}),
                )

                self.trades[trade.trade_id] = trade
                self.trade_history.append(trade)

            self.logger.info(f"Loaded {len(trades_data)} trades from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading trades: {e}")

    def clear_trades(self) -> None:
        """Clear all trades."""
        self.trades.clear()
        self.trade_history.clear()
        self.daily_pnl.clear()
        self.monthly_pnl.clear()
        self.logger.info("All trades cleared")


def create_trade_journal(config: Optional[Dict[str, Any]] = None) -> TradeJournal:
    """Create a trade journal instance."""
    return TradeJournal(config)
