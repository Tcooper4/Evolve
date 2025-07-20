"""Real-Time Signal Center.

This module provides live signal streaming dashboard with active trades,
time since signal, strategy that triggered it, and Discord/email webhook alerts.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

# Try to import webhook libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    ALERT = "alert"


class SignalStatus(Enum):
    """Signal status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    EXECUTED = "executed"


@dataclass
class Signal:
    """Trading signal."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    strategy: str
    confidence: float
    price: float
    target_price: float
    stop_loss: float
    timestamp: datetime
    expiry: datetime
    status: SignalStatus
    metadata: Dict[str, Any]


@dataclass
class ActiveTrade:
    """Active trade information."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    time_open: datetime
    strategy: str
    signal_id: str
    metadata: Dict[str, Any]


class SignalCenter:
    """Real-time signal center with streaming and alerts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize signal center.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Signal settings
        self.signal_expiry_hours = self.config.get("signal_expiry_hours", 24)
        self.max_active_signals = self.config.get("max_active_signals", 100)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.6)

        # Alert settings
        self.enable_discord_alerts = self.config.get("enable_discord_alerts", False)
        self.enable_email_alerts = self.config.get("enable_email_alerts", False)
        self.enable_slack_alerts = self.config.get("enable_slack_alerts", False)
        self.discord_webhook_url = self.config.get("discord_webhook_url", "")
        self.email_webhook_url = self.config.get("email_webhook_url", "")
        self.slack_webhook_url = self.config.get("slack_webhook_url", "")

        # Data storage
        self.signals: Dict[str, Signal] = {}
        self.trades: Dict[str, ActiveTrade] = {}
        self.performance_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        self.total_signals = 0
        self.total_trades = 0
        self.successful_trades = 0

        logger.info("SignalCenter initialized successfully")

    def add_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strategy: str,
        confidence: float,
        price: float,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new trading signal.

        Args:
            symbol: Trading symbol
            signal_type: Type of signal
            strategy: Strategy name
            confidence: Signal confidence (0.0 to 1.0)
            price: Current price
            target_price: Target price (optional)
            stop_loss: Stop loss price (optional)
            metadata: Additional metadata

        Returns:
            Signal ID
        """
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Invalid confidence {confidence}, clamping to [0, 1]")
            confidence = max(0.0, min(1.0, confidence))

        # Check confidence threshold
        if confidence < self.min_confidence_threshold:
            logger.info(f"Signal confidence {confidence} below threshold {self.min_confidence_threshold}, skipping")
            return ""

        # Generate signal ID
        signal_id = f"signal_{len(self.signals)}_{int(datetime.now().timestamp())}"

        # Calculate expiry time
        expiry = datetime.now() + timedelta(hours=self.signal_expiry_hours)

        # Set default target and stop loss if not provided
        if target_price is None:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                target_price = price * 1.05  # 5% target
            else:
                target_price = price * 0.95  # 5% target

        if stop_loss is None:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = price * 0.98  # 2% stop loss
            else:
                stop_loss = price * 1.02  # 2% stop loss

        # Create signal
        signal = Signal(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            strategy=strategy,
            confidence=confidence,
            price=price,
            target_price=target_price,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            expiry=expiry,
            status=SignalStatus.ACTIVE,
            metadata=metadata or {}
        )

        # Store signal
        self.signals[signal_id] = signal
        self.total_signals += 1

        # Cleanup old signals if needed
        if len(self.signals) > self.max_active_signals:
            self._cleanup_expired_signals()

        # Send alert
        if self.enable_discord_alerts or self.enable_email_alerts or self.enable_slack_alerts:
            self._send_alert(signal)

        logger.info(f"Added signal {signal_id}: {symbol} {signal_type.value} (confidence: {confidence:.2f})")
        return signal_id

    def update_signal_status(
        self,
        signal_id: str,
        status: SignalStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update signal status.

        Args:
            signal_id: Signal ID
            status: New status
            metadata: Additional metadata

        Returns:
            Updated signal information
        """
        if signal_id not in self.signals:
            logger.error(f"Signal {signal_id} not found")
            return {"error": "Signal not found"}

        signal = self.signals[signal_id]
        signal.status = status

        if metadata:
            signal.metadata.update(metadata)

        logger.info(f"Updated signal {signal_id} status to {status.value}")
        return self._signal_to_dict(signal)

    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        signal_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a new active trade.

        Args:
            trade_id: Trade ID
            symbol: Trading symbol
            side: Trade side (buy/sell)
            entry_price: Entry price
            quantity: Trade quantity
            strategy: Strategy name
            signal_id: Associated signal ID
            metadata: Additional metadata

        Returns:
            Trade information
        """
        # Create trade
        trade = ActiveTrade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            pnl=0.0,
            pnl_pct=0.0,
            time_open=datetime.now(),
            strategy=strategy,
            signal_id=signal_id,
            metadata=metadata or {}
        )

        # Store trade
        self.trades[trade_id] = trade
        self.total_trades += 1

        # Update signal status if associated
        if signal_id in self.signals:
            self.update_signal_status(signal_id, SignalStatus.EXECUTED)

        logger.info(f"Added trade {trade_id}: {symbol} {side} {quantity} @ {entry_price}")
        return self._trade_to_dict(trade)

    def update_trade_price(self, trade_id: str, current_price: float) -> Dict[str, Any]:
        """Update trade current price and P&L.

        Args:
            trade_id: Trade ID
            current_price: Current market price

        Returns:
            Updated trade information
        """
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return {"error": "Trade not found"}

        trade = self.trades[trade_id]
        trade.current_price = current_price

        # Calculate P&L
        if trade.side.lower() == "buy":
            trade.pnl = (current_price - trade.entry_price) * trade.quantity
            trade.pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl = (trade.entry_price - current_price) * trade.quantity
            trade.pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100

        return self._trade_to_dict(trade)

    def close_trade(
        self, trade_id: str, exit_price: float, exit_reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close an active trade.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for closing

        Returns:
            Closed trade information
        """
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return {"error": "Trade not found"}

        trade = self.trades[trade_id]

        # Calculate final P&L
        if trade.side.lower() == "buy":
            final_pnl = (exit_price - trade.entry_price) * trade.quantity
            final_pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            final_pnl = (trade.entry_price - exit_price) * trade.quantity
            final_pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        # Update performance tracking
        self._update_performance_tracking(trade, final_pnl_pct)

        # Remove from active trades
        closed_trade = self.trades.pop(trade_id)

        logger.info(f"Closed trade {trade_id}: P&L {final_pnl:.2f} ({final_pnl_pct:.2f}%)")
        return {
            "trade_id": trade_id,
            "final_pnl": final_pnl,
            "final_pnl_pct": final_pnl_pct,
            "exit_reason": exit_reason,
            "duration": datetime.now() - trade.time_open
        }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get active signals.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active signals
        """
        active_signals = []
        for signal in self.signals.values():
            if signal.status == SignalStatus.ACTIVE:
                if symbol is None or signal.symbol == symbol:
                    active_signals.append(signal)
        return active_signals

    def get_active_trades(self, symbol: Optional[str] = None) -> List[ActiveTrade]:
        """Get active trades.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active trades
        """
        active_trades = []
        for trade in self.trades.values():
            if symbol is None or trade.symbol == symbol:
                active_trades.append(trade)
        return active_trades

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get signal summary statistics."""
        active_signals = self.get_active_signals()
        active_trades = self.get_active_trades()

        summary = {
            "total_signals": self.total_signals,
            "active_signals": len(active_signals),
            "total_trades": self.total_trades,
            "active_trades": len(active_trades),
            "successful_trades": self.successful_trades,
            "success_rate": (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0,
            "signals_by_type": {},
            "signals_by_strategy": {}
        }

        # Count signals by type
        for signal in active_signals:
            signal_type = signal.signal_type.value
            summary["signals_by_type"][signal_type] = summary["signals_by_type"].get(signal_type, 0) + 1

        # Count signals by strategy
        for signal in active_signals:
            strategy = signal.strategy
            summary["signals_by_strategy"][strategy] = summary["signals_by_strategy"].get(strategy, 0) + 1

        return summary

    def get_signal_performance(
        self, strategy: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get signal performance statistics.

        Args:
            strategy: Filter by strategy (optional)
            days: Number of days to look back

        Returns:
            Performance statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        performance_data = []

        for record in self.performance_history:
            if record["timestamp"] >= cutoff_date:
                if strategy is None or record["strategy"] == strategy:
                    performance_data.append(record)

        if not performance_data:
            return {"message": "No performance data available"}

        # Calculate statistics
        pnl_values = [record["pnl_pct"] for record in performance_data]
        total_trades = len(performance_data)
        successful_trades = len([p for p in pnl_values if p > 0])

        return {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "success_rate": (successful_trades / total_trades * 100) if total_trades > 0 else 0.0,
            "average_pnl": sum(pnl_values) / len(pnl_values) if pnl_values else 0.0,
            "max_profit": max(pnl_values) if pnl_values else 0.0,
            "max_loss": min(pnl_values) if pnl_values else 0.0,
            "strategy": strategy or "all"
        }

    def _send_alert(self, signal: Signal):
        """Send alert for a new signal."""
        if not (self.enable_discord_alerts or self.enable_email_alerts or self.enable_slack_alerts):
            return

        message = self._create_alert_message(signal)

        if self.enable_discord_alerts and self.discord_webhook_url:
            self._send_discord_alert(message, signal)

        if self.enable_email_alerts and self.email_webhook_url:
            self._send_email_alert(message, signal)

        if self.enable_slack_alerts and self.slack_webhook_url:
            self._send_slack_alert(message, signal)

    def _create_alert_message(self, signal: Signal) -> str:
        """Create alert message for a signal."""
        emoji_map = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.STRONG_BUY: "🟢🟢",
            SignalType.STRONG_SELL: "🔴🔴",
            SignalType.HOLD: "🟡",
            SignalType.ALERT: "⚠️"
        }

        emoji = emoji_map.get(signal.signal_type, "📊")
        confidence_pct = signal.confidence * 100

        message = f"""
{emoji} **NEW SIGNAL** {emoji}

**Symbol**: {signal.symbol}
**Type**: {signal.signal_type.value.upper()}
**Strategy**: {signal.strategy}
**Confidence**: {confidence_pct:.1f}%
**Price**: ${signal.price:.2f}
**Target**: ${signal.target_price:.2f}
**Stop Loss**: ${signal.stop_loss:.2f}

*Signal ID: {signal.signal_id}*
        """.strip()

        return message

    def _send_discord_alert(self, message: str, signal: Signal):
        """Send Discord webhook alert."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, skipping Discord alert")
            return

        try:
            payload = {
                "content": message,
                "username": "Trading Bot",
                "avatar_url": "https://example.com/bot-avatar.png"
            }

            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Discord alert sent for signal {signal.signal_id}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def _send_email_alert(self, message: str, signal: Signal):
        """Send email webhook alert."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, skipping email alert")
            return

        try:
            payload = {
                "to": "trading@example.com",
                "subject": f"Trading Signal: {signal.symbol} {signal.signal_type.value.upper()}",
                "body": message
            }

            response = requests.post(self.email_webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Email alert sent for signal {signal.signal_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack_alert(self, message: str, signal: Signal):
        """Send Slack webhook alert."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, skipping Slack alert")
            return

        try:
            payload = {
                "text": message,
                "username": "Trading Bot",
                "icon_emoji": ":chart_with_upwards_trend:"
            }

            response = requests.post(self.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Slack alert sent for signal {signal.signal_id}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _cleanup_expired_signals(self):
        """Remove expired signals."""
        current_time = datetime.now()
        expired_signals = []

        for signal_id, signal in self.signals.items():
            if signal.expiry < current_time:
                expired_signals.append(signal_id)

        for signal_id in expired_signals:
            self.signals[signal_id].status = SignalStatus.EXPIRED
            logger.info(f"Signal {signal_id} expired")

    def _update_performance_tracking(self, trade: ActiveTrade, final_pnl_pct: float):
        """Update performance tracking."""
        record = {
            "timestamp": datetime.now(),
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "pnl_pct": final_pnl_pct,
            "duration": datetime.now() - trade.time_open
        }

        self.performance_history.append(record)

        if final_pnl_pct > 0:
            self.successful_trades += 1

        # Update strategy performance
        if trade.strategy not in self.strategy_performance:
            self.strategy_performance[trade.strategy] = {
                "total_trades": 0,
                "successful_trades": 0,
                "total_pnl": 0.0
            }

        perf = self.strategy_performance[trade.strategy]
        perf["total_trades"] += 1
        perf["total_pnl"] += final_pnl_pct

        if final_pnl_pct > 0:
            perf["successful_trades"] += 1

    def export_signal_report(self, filepath: str) -> Dict[str, Any]:
        """Export signal report to JSON file.

        Args:
            filepath: Output file path

        Returns:
            Export result
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_signal_summary(),
                "active_signals": [self._signal_to_dict(s) for s in self.get_active_signals()],
                "active_trades": [self._trade_to_dict(t) for t in self.get_active_trades()],
                "performance_history": self.performance_history[-100:],  # Last 100 records
                "strategy_performance": self.strategy_performance
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Signal report exported to {filepath}")
            return {"success": True, "filepath": filepath}

        except Exception as e:
            logger.error(f"Failed to export signal report: {e}")
            return {"success": False, "error": str(e)}

    def _signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "strategy": signal.strategy,
            "confidence": signal.confidence,
            "price": signal.price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "timestamp": signal.timestamp.isoformat(),
            "expiry": signal.expiry.isoformat(),
            "status": signal.status.value,
            "metadata": signal.metadata
        }

    def _trade_to_dict(self, trade: ActiveTrade) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "current_price": trade.current_price,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "time_open": trade.time_open.isoformat(),
            "strategy": trade.strategy,
            "signal_id": trade.signal_id,
            "metadata": trade.metadata
        }


def get_signal_center() -> Dict[str, Any]:
    """Get signal center instance."""
    return {
        "message": "Signal center functionality available",
        "status": "ready"
    } 