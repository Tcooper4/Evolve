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

        # Webhook settings
        self.webhook_config = self.config.get(
            "webhook_config",
            {
                "discord_webhook_url": "",
                "email_webhook_url": "",
                "slack_webhook_url": "",
                "enable_alerts": True,
                "alert_confidence_threshold": 0.7,
            },
        )

        # Signal storage
        self.active_signals = {}
        self.signal_history = []
        self.active_trades = {}
        self.trade_history = []

        # Performance tracking
        self.signal_performance = {}
        self.strategy_performance = {}

        # Alert history
        self.alert_history = []

        logger.info("Signal Center initialized")

        return {
            "success": True,
            "message": "Signal Center initialized successfully",
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

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
            signal_type: Signal type
            strategy: Strategy name
            confidence: Signal confidence (0-1)
            price: Current price
            target_price: Target price (optional)
            stop_loss: Stop loss price (optional)
            metadata: Additional metadata

        Returns:
            Signal ID
        """
        try:
            # Generate signal ID
            signal_id = f"{symbol}_{signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Set default values
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
                expiry=datetime.now() + timedelta(hours=self.signal_expiry_hours),
                status=SignalStatus.ACTIVE,
                metadata=metadata or {},
            )

            # Store signal
            self.active_signals[signal_id] = signal
            self.signal_history.append(signal)

            # Check if signal should trigger alert
            if (
                self.webhook_config["enable_alerts"]
                and confidence >= self.webhook_config["alert_confidence_threshold"]
            ):
                self._send_alert(signal)

            # Clean up old signals
            self._cleanup_expired_signals()

            logger.info(
                f"Added signal {signal_id}: {signal_type.value} {symbol} "
                f"at {price:.2f} (confidence: {confidence:.2%})"
            )

            return signal_id

        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

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
            Update result
        """
        try:
            if signal_id not in self.active_signals:
                return {
                    "success": False,
                    "error": "Signal not found",
                    "timestamp": datetime.now().isoformat(),
                }

            signal = self.active_signals[signal_id]
            signal.status = status

            if metadata:
                signal.metadata.update(metadata)

            logger.info(f"Updated signal {signal_id} status to {status.value}")

            return {
                "success": True,
                "message": "Signal status updated successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

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
            Trade result
        """
        try:
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
                metadata=metadata or {},
            )

            # Store trade
            self.active_trades[trade_id] = trade
            self.trade_history.append(trade)

            logger.info(
                f"Added trade {trade_id}: {side} {quantity} {symbol} at {entry_price:.2f}"
            )

            return {
                "success": True,
                "message": "Trade added successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def update_trade_price(self, trade_id: str, current_price: float) -> Dict[str, Any]:
        """Update trade current price and P&L.

        Args:
            trade_id: Trade ID
            current_price: Current market price

        Returns:
            Update result
        """
        try:
            if trade_id not in self.active_trades:
                return {
                    "success": False,
                    "error": "Trade not found",
                    "timestamp": datetime.now().isoformat(),
                }

            trade = self.active_trades[trade_id]
            trade.current_price = current_price

            # Calculate P&L
            if trade.side.lower() == "buy":
                trade.pnl = (current_price - trade.entry_price) * trade.quantity
                trade.pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl = (trade.entry_price - current_price) * trade.quantity
                trade.pnl_pct = (trade.entry_price - current_price) / trade.entry_price

            return {
                "success": True,
                "message": "Trade price updated successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating trade price: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def close_trade(
        self, trade_id: str, exit_price: float, exit_reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close an active trade.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for closing

        Returns:
            Close result
        """
        try:
            if trade_id not in self.active_trades:
                return {
                    "success": False,
                    "error": "Trade not found",
                    "timestamp": datetime.now().isoformat(),
                }

            trade = self.active_trades[trade_id]

            # Calculate final P&L
            if trade.side.lower() == "buy":
                final_pnl = (exit_price - trade.entry_price) * trade.quantity
                final_pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            else:
                final_pnl = (trade.entry_price - exit_price) * trade.quantity
                final_pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

            # Update performance tracking
            self._update_performance_tracking(trade, final_pnl_pct)

            # Remove from active trades
            del self.active_trades[trade_id]

            logger.info(
                f"Closed trade {trade_id}: P&L {final_pnl:.2f} ({final_pnl_pct:.2%})"
            )

            return {
                "success": True,
                "message": "Trade closed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get active signals.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active signals
        """
        try:
            if symbol:
                signals = [
                    s for s in self.active_signals.values() if s.symbol == symbol
                ]
            else:
                signals = list(self.active_signals.values())

            return {
                "success": True,
                "result": signals,
                "message": "Active signals retrieved successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_active_trades(self, symbol: Optional[str] = None) -> List[ActiveTrade]:
        """Get active trades.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active trades
        """
        try:
            if symbol:
                trades = [t for t in self.active_trades.values() if t.symbol == symbol]
            else:
                trades = list(self.active_trades.values())

            return {
                "success": True,
                "result": trades,
                "message": "Active trades retrieved successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get signal summary statistics.

        Returns:
            Summary statistics
        """
        try:
            total_signals = len(self.signal_history)
            active_signals = len(self.active_signals)
            total_trades = len(self.trade_history)
            active_trades = len(self.active_trades)

            # Calculate performance metrics
            if self.trade_history:
                total_pnl = sum(
                    trade.pnl for trade in self.trade_history if hasattr(trade, "pnl")
                )
                avg_pnl = total_pnl / len(self.trade_history)
            else:
                total_pnl = 0
                avg_pnl = 0

            summary = {
                "total_signals": total_signals,
                "active_signals": active_signals,
                "total_trades": total_trades,
                "active_trades": active_trades,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "timestamp": datetime.now().isoformat(),
            }

            return {
                "success": True,
                "result": summary,
                "message": "Signal summary retrieved successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_signal_performance(
        self, strategy: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get signal performance metrics.

        Args:
            strategy: Filter by strategy (optional)
            days: Number of days to analyze

        Returns:
            Performance metrics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Filter signals by date and strategy
            recent_signals = [
                s for s in self.signal_history if s.timestamp >= cutoff_date
            ]

            if strategy:
                recent_signals = [s for s in recent_signals if s.strategy == strategy]

            if not recent_signals:
                return {
                    "success": True,
                    "result": {},
                    "message": "No signals found for period",
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate metrics
            total_signals = len(recent_signals)
            avg_confidence = sum(s.confidence for s in recent_signals) / total_signals

            # Signal type distribution
            signal_types = {}
            for signal in recent_signals:
                signal_type = signal.signal_type.value
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

            performance = {
                "total_signals": total_signals,
                "avg_confidence": avg_confidence,
                "signal_types": signal_types,
                "period_days": days,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat(),
            }

            return {
                "success": True,
                "result": performance,
                "message": "Performance metrics retrieved successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _send_alert(self, signal: Signal):
        """Send alert for signal."""
        try:
            message = self._create_alert_message(signal)

            # Send to different platforms
            if self.webhook_config.get("discord_webhook_url"):
                self._send_discord_alert(message, signal)

            if self.webhook_config.get("email_webhook_url"):
                self._send_email_alert(message, signal)

            if self.webhook_config.get("slack_webhook_url"):
                self._send_slack_alert(message, signal)

            # Log alert
            self.alert_history.append(
                {
                    "signal_id": signal.signal_id,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return {
                "success": True,
                "message": "Alert sent successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _create_alert_message(self, signal: Signal) -> str:
        """Create alert message for signal."""
        try:
            message = f"🚨 **{signal.signal_type.value.upper()}** Signal\n"
            message += f"**Symbol:** {signal.symbol}\n"
            message += f"**Strategy:** {signal.strategy}\n"
            message += f"**Price:** ${signal.price:.2f}\n"
            message += f"**Target:** ${signal.target_price:.2f}\n"
            message += f"**Stop Loss:** ${signal.stop_loss:.2f}\n"
            message += f"**Confidence:** {signal.confidence:.1%}\n"
            message += f"**Time:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

            return {
                "success": True,
                "result": message,
                "message": "Alert message created successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating alert message: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _send_discord_alert(self, message: str, signal: Signal):
        """Send Discord alert."""
        try:
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests library not available for Discord alerts")
                return {
                    "success": False,
                    "error": "Requests library not available",
                    "timestamp": datetime.now().isoformat(),
                }

            webhook_url = self.webhook_config.get("discord_webhook_url")
            if not webhook_url:
                return {
                    "success": False,
                    "error": "Discord webhook URL not configured",
                    "timestamp": datetime.now().isoformat(),
                }

            payload = {"content": message, "username": "Trading Signal Bot"}

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            return {
                "success": True,
                "message": "Discord alert sent successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _send_email_alert(self, message: str, signal: Signal):
        """Send email alert."""
        try:
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests library not available for email alerts")
                return {
                    "success": False,
                    "error": "Requests library not available",
                    "timestamp": datetime.now().isoformat(),
                }

            webhook_url = self.webhook_config.get("email_webhook_url")
            if not webhook_url:
                return {
                    "success": False,
                    "error": "Email webhook URL not configured",
                    "timestamp": datetime.now().isoformat(),
                }

            payload = {
                "subject": f"Trading Signal: {signal.signal_type.value.upper()} {signal.symbol}",
                "body": message,
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            return {
                "success": True,
                "message": "Email alert sent successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _send_slack_alert(self, message: str, signal: Signal):
        """Send Slack alert."""
        try:
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests library not available for Slack alerts")
                return {
                    "success": False,
                    "error": "Requests library not available",
                    "timestamp": datetime.now().isoformat(),
                }

            webhook_url = self.webhook_config.get("slack_webhook_url")
            if not webhook_url:
                return {
                    "success": False,
                    "error": "Slack webhook URL not configured",
                    "timestamp": datetime.now().isoformat(),
                }

            payload = {"text": message}

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            return {
                "success": True,
                "message": "Slack alert sent successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _cleanup_expired_signals(self):
        """Clean up expired signals."""
        try:
            current_time = datetime.now()
            expired_signals = []

            for signal_id, signal in self.active_signals.items():
                if signal.expiry < current_time:
                    expired_signals.append(signal_id)
                    signal.status = SignalStatus.EXPIRED

            for signal_id in expired_signals:
                del self.active_signals[signal_id]

            if expired_signals:
                logger.info(f"Cleaned up {len(expired_signals)} expired signals")

            return {
                "success": True,
                "message": f"Cleaned up {len(expired_signals)} expired signals",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _update_performance_tracking(self, trade: ActiveTrade, final_pnl_pct: float):
        """Update performance tracking for trade."""
        try:
            # Update strategy performance
            strategy = trade.strategy
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                }

            perf = self.strategy_performance[strategy]
            perf["total_trades"] += 1
            perf["total_pnl"] += final_pnl_pct

            if final_pnl_pct > 0:
                perf["winning_trades"] += 1

            perf["avg_pnl"] = perf["total_pnl"] / perf["total_trades"]

            return {
                "success": True,
                "message": "Performance tracking updated successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def export_signal_report(self, filepath: str) -> Dict[str, Any]:
        """Export signal report to file.

        Args:
            filepath: Output file path

        Returns:
            Export result
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_signal_summary(),
                "active_signals": [
                    self._signal_to_dict(s) for s in self.active_signals.values()
                ],
                "active_trades": [
                    self._trade_to_dict(t) for t in self.active_trades.values()
                ],
                "strategy_performance": self.strategy_performance,
                "alert_history": self.alert_history[-100:],  # Last 100 alerts
            }

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Signal report exported to {filepath}")

            return {
                "success": True,
                "message": f"Signal report exported to {filepath}",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error exporting signal report: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

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
            "metadata": signal.metadata,
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
            "metadata": trade.metadata,
        }


# Global signal center instance
signal_center = SignalCenter()


def get_signal_center() -> Dict[str, Any]:
    """Get signal center instance."""
    try:
        # This would typically return a singleton instance
        # For now, return a new instance
        signal_center = SignalCenter()
        return {
            "success": True,
            "result": signal_center,
            "message": "Signal center instance created successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error creating signal center: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
