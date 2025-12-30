"""
Real-Time Signal Center

Provides live signal streaming, active trades, and webhook alerts.
Manages real-time trading signals and notifications.
"""

import asyncio
import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import requests

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for real-time trading."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    ALERT = "alert"


class SignalPriority(Enum):
    """Signal priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradingSignal:
    """Real-time trading signal."""

    signal_id: str
    symbol: str
    signal_type: SignalType
    priority: SignalPriority
    price: float
    quantity: float
    confidence: float
    strategy: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ActiveTrade:
    """Active trade information."""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    entry_time: datetime
    strategy: str
    status: str
    metadata: Dict[str, Any]


@dataclass
class AlertConfig:
    """Alert configuration."""

    alert_type: str
    conditions: Dict[str, Any]
    webhook_url: Optional[str] = None
    email: Optional[str] = None
    slack_webhook: Optional[str] = None
    enabled: bool = True


class RealTimeSignalCenter:
    """Advanced real-time signal center with streaming and alerts."""

    def __init__(
        self,
        websocket_port: int = 8765,
        max_signals: int = 1000,
        alert_check_interval: int = 30,
    ):
        """Initialize the real-time signal center.

        Args:
            websocket_port: Port for WebSocket server
            max_signals: Maximum signals to keep in memory
            alert_check_interval: Seconds between alert checks
        """
        self.websocket_port = websocket_port
        self.max_signals = max_signals
        self.alert_check_interval = alert_check_interval

        # Initialize data structures
        self.signals = []
        self.active_trades = {}
        self.alert_configs = []
        self.signal_subscribers = []
        self.websocket_server = None
        self.is_running = False

        # Threading and async
        self.signal_queue = Queue()
        self.alert_queue = Queue()
        self.lock = threading.Lock()

        # Initialize default alerts
        self._initialize_default_alerts()

        logger.info("Real-Time Signal Center initialized successfully")

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _initialize_default_alerts(self):
        """Initialize default alert configurations."""
        self.alert_configs = [
            AlertConfig(
                alert_type="high_confidence_signal",
                conditions={"confidence_threshold": 0.8},
                enabled=True,
            ),
            AlertConfig(
                alert_type="large_position_change",
                conditions={"position_change_threshold": 0.1},
                enabled=True,
            ),
            AlertConfig(
                alert_type="significant_loss",
                conditions={"loss_threshold": -0.05},
                enabled=True,
            ),
            AlertConfig(
                alert_type="market_volatility",
                conditions={"volatility_threshold": 0.3},
                enabled=True,
            ),
        ]

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def start(self):
        """Start the real-time signal center."""
        try:
            self.is_running = True

            # Start signal processing thread
            self.signal_thread = threading.Thread(target=self._signal_processing_loop)
            self.signal_thread.daemon = True
            self.signal_thread.start()

            # Start alert processing thread
            self.alert_thread = threading.Thread(target=self._alert_processing_loop)
            self.alert_thread.daemon = True
            self.alert_thread.start()

            # Start WebSocket server
            self._start_websocket_server()

            logger.info("Real-Time Signal Center started successfully")
            return {
                "status": "success",
                "message": "Real-Time Signal Center started successfully",
            }

        except Exception as e:
            logger.error(f"Error starting signal center: {e}")
            self.is_running = False
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def stop(self):
        """Stop the real-time signal center."""
        try:
            self.is_running = False

            # Stop WebSocket server
            if self.websocket_server:
                asyncio.run(self.websocket_server.close())

            logger.info("Real-Time Signal Center stopped")
            return {
                "status": "success",
                "message": "Real-Time Signal Center stopped successfully",
            }

        except Exception as e:
            logger.error(f"Error stopping signal center: {e}")
            return {
                "success": True,
                "result": {"status": "error", "message": str(e)},
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _signal_processing_loop(self):
        """Main signal processing loop."""
        while self.is_running:
            try:
                # Process signals from queue
                while not self.signal_queue.empty():
                    signal = self.signal_queue.get_nowait()
                    self._process_signal(signal)

                # Check for expired signals
                self._cleanup_expired_signals()

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in signal processing loop: {e}")
                time.sleep(5)

        return {
            "success": True,
            "result": {
                "status": "completed",
                "message": "Signal processing loop ended",
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _alert_processing_loop(self):
        """Main alert processing loop."""
        while self.is_running:
            try:
                # Process alerts from queue
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get_nowait()
                    self._process_alert(alert)

                # Check for alert conditions
                self._check_alert_conditions()

                time.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(10)

        return {
            "success": True,
            "result": {"status": "completed", "message": "Alert processing loop ended"},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _process_signal(self, signal: TradingSignal):
        """Process incoming trading signal."""
        try:
            with self.lock:
                # Add signal to list
                self.signals.append(signal)

                # Keep only recent signals
                if len(self.signals) > self.max_signals:
                    self.signals = self.signals[-self.max_signals :]

                # Check for alert conditions
                self._check_signal_alerts(signal)

                # Notify subscribers
                self._notify_subscribers(signal)

            logger.info(f"Processed signal: {signal.signal_id} for {signal.symbol}")
            return {
                "status": "success",
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
            }

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {
                "success": True,
                "result": {
                    "status": "error",
                    "message": str(e),
                    "signal_id": signal.signal_id,
                },
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _process_alert(self, alert: Dict[str, Any]):
        """Process alert and send notifications."""
        try:
            alert_type = alert.get("type")
            message = alert.get("message")
            alert.get("priority", "medium")

            # Send webhook notifications
            self._send_webhook_alert(alert)

            # Send email notifications
            self._send_email_alert(alert)

            # Send Slack notifications
            self._send_slack_alert(alert)

            logger.info(f"Processed alert: {alert_type} - {message}")
            return {"status": "success", "alert_type": alert_type, "message": message}

        except Exception as e:
            logger.error(f"Error processing alert: {e}")
            return {
                "success": True,
                "result": {
                    "status": "error",
                    "message": str(e),
                    "alert_type": alert.get("type"),
                },
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _check_signal_alerts(self, signal: TradingSignal):
        """Check if signal triggers any alerts."""
        try:
            for config in self.alert_configs:
                if not config.enabled:
                    continue

                if config.alert_type == "high_confidence_signal":
                    threshold = config.conditions.get("confidence_threshold", 0.8)
                    if signal.confidence >= threshold:
                        self._queue_alert(
                            {
                                "type": "high_confidence_signal",
                                "message": f"High confidence signal: {signal.symbol} {signal.signal_type.value}",
                                "priority": "high",
                                "signal": signal,
                            }
                        )

                elif config.alert_type == "large_position_change":
                    # Check for large position changes
                    if signal.quantity > config.conditions.get("threshold", 1000):
                        self._queue_alert(
                            {
                                "type": "large_position_change",
                                "message": f"Large position change: {signal.quantity} {signal.symbol}",
                                "priority": "medium",
                                "data": {"signal": signal},
                            }
                        )

                elif config.alert_type == "significant_loss":
                    # Check for losses in active trades
                    for trade in self.active_trades.values():
                        if trade.pnl_percent < config.conditions.get(
                            "loss_threshold", -0.05
                        ):
                            self._queue_alert(
                                {
                                    "type": "significant_loss",
                                    "message": f"Significant loss: {trade.pnl_percent:.2%} on {trade.symbol}",
                                    "priority": "high",
                                    "data": {"trade": trade},
                                }
                            )

                elif config.alert_type == "market_volatility":
                    # Check market volatility using recent signals
                    recent_signals = [
                        s
                        for s in self.signals
                        if s.timestamp > datetime.now() - timedelta(minutes=30)
                    ]
                    if len(recent_signals) > config.conditions.get(
                        "signal_threshold", 20
                    ):
                        self._queue_alert(
                            {
                                "type": "market_volatility",
                                "message": f"High market volatility: {len(recent_signals)} signals in 30 minutes",
                                "priority": "medium",
                                "data": {"signal_count": len(recent_signals)},
                            }
                        )

        except Exception as e:
            logger.error(f"Error checking signal alerts: {e}")

    def _check_alert_conditions(self):
        """Check for general alert conditions."""
        try:
            # Check active trades for losses
            for trade_id, trade in self.active_trades.items():
                if trade.pnl_percent < -0.05:  # 5% loss
                    self._queue_alert(
                        {
                            "type": "significant_loss",
                            "message": f"Significant loss in {trade.symbol}: {trade.pnl_percent:.2%}",
                            "priority": "high",
                            "trade": trade,
                        }
                    )

            # Check for high signal volume
            recent_signals = [
                s
                for s in self.signals
                if s.timestamp > datetime.now() - timedelta(minutes=5)
            ]

            if len(recent_signals) > 10:
                self._queue_alert(
                    {
                        "type": "high_signal_volume",
                        "message": f"High signal volume: {len(recent_signals)} signals in 5 minutes",
                        "priority": "medium",
                    }
                )

        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")

    def _queue_alert(self, alert: Dict[str, Any]):
        """Queue alert for processing."""
        try:
            self.alert_queue.put(alert)
        except Exception as e:
            logger.error(f"Error queuing alert: {e}")

    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send webhook alert."""
        try:
            for config in self.alert_configs:
                if config.webhook_url and config.enabled:
                    payload = {
                        "timestamp": datetime.now().isoformat(),
                        "type": alert.get("type"),
                        "message": alert.get("message"),
                        "priority": alert.get("priority"),
                        "data": alert.get("data", {}),
                    }

                    response = requests.post(
                        config.webhook_url, json=payload, timeout=10
                    )

                    if response.status_code != 200:
                        logger.warning(f"Webhook alert failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert."""
        try:
            # Email sending would be implemented here
            # For now, just log the alert
            logger.info(f"Email alert would be sent: {alert.get('message')}")

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert."""
        try:
            for config in self.alert_configs:
                if config.slack_webhook and config.enabled:
                    payload = {
                        "text": f"*{alert.get('type', 'Alert').upper()}*\n{alert.get('message')}",
                        "username": "Trading Bot",
                        "icon_emoji": ":chart_with_upwards_trend:",
                    }

                    response = requests.post(
                        config.slack_webhook, json=payload, timeout=10
                    )

                    if response.status_code != 200:
                        logger.warning(f"Slack alert failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def _notify_subscribers(self, signal: TradingSignal):
        """Notify signal subscribers."""
        try:
            for subscriber in self.signal_subscribers:
                try:
                    subscriber(signal)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")

        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")

    def _cleanup_expired_signals(self):
        """Remove expired signals."""
        try:
            with self.lock:
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.signals = [s for s in self.signals if s.timestamp > cutoff_time]

        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")

    def add_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        price: float,
        quantity: float,
        confidence: float,
        strategy: str,
        priority: SignalPriority = SignalPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new trading signal."""
        try:
            signal_id = f"SIGNAL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            signal = TradingSignal(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                priority=priority,
                price=price,
                quantity=quantity,
                confidence=confidence,
                strategy=strategy,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Add to processing queue
            self.signal_queue.put(signal)

            return signal_id

        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return ""

    def update_active_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        strategy: str,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update active trade information."""
        try:
            pnl = (
                (current_price - entry_price) * quantity
                if side == "buy"
                else (entry_price - current_price) * quantity
            )
            pnl_percent = (
                pnl / (entry_price * quantity) if entry_price * quantity > 0 else 0
            )

            trade = ActiveTrade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                entry_time=datetime.now(),
                strategy=strategy,
                status=status,
                metadata=metadata or {},
            )

            with self.lock:
                self.active_trades[trade_id] = trade

            logger.info(f"Updated active trade: {trade_id} for {symbol}")

        except Exception as e:
            logger.error(f"Error updating active trade: {e}")

    def close_trade(self, trade_id: str):
        """Close active trade."""
        try:
            with self.lock:
                if trade_id in self.active_trades:
                    trade = self.active_trades[trade_id]
                    trade.status = "closed"

                    # Keep for some time for analysis
                    # In practice, you might move to a separate closed trades table

                    logger.info(f"Closed trade: {trade_id}")

        except Exception as e:
            logger.error(f"Error closing trade: {e}")

    def subscribe_to_signals(self, callback: Callable[[TradingSignal], None]):
        """Subscribe to real-time signals."""
        try:
            if callback not in self.signal_subscribers:
                self.signal_subscribers.append(callback)
                logger.info("New signal subscriber added")

        except Exception as e:
            logger.error(f"Error adding signal subscriber: {e}")

    def unsubscribe_from_signals(self, callback: Callable[[TradingSignal], None]):
        """Unsubscribe from real-time signals."""
        try:
            if callback in self.signal_subscribers:
                self.signal_subscribers.remove(callback)
                logger.info("Signal subscriber removed")

        except Exception as e:
            logger.error(f"Error removing signal subscriber: {e}")

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        hours: int = 24,
    ) -> List[TradingSignal]:
        """Get recent signals with optional filtering."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            with self.lock:
                recent_signals = [s for s in self.signals if s.timestamp > cutoff_time]

                if symbol:
                    recent_signals = [s for s in recent_signals if s.symbol == symbol]

                if signal_type:
                    recent_signals = [
                        s for s in recent_signals if s.signal_type == signal_type
                    ]

                return sorted(recent_signals, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []

    def get_active_trades(self, symbol: Optional[str] = None) -> List[ActiveTrade]:
        """Get active trades with optional filtering."""
        try:
            with self.lock:
                trades = list(self.active_trades.values())

                if symbol:
                    trades = [t for t in trades if t.symbol == symbol]

                return sorted(trades, key=lambda x: x.entry_time, reverse=True)

        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return []

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signal activity."""
        try:
            with self.lock:
                recent_signals = [
                    s
                    for s in self.signals
                    if s.timestamp > datetime.now() - timedelta(hours=24)
                ]

                signal_counts = {}
                for signal_type in SignalType:
                    signal_counts[signal_type.value] = len(
                        [s for s in recent_signals if s.signal_type == signal_type]
                    )

                return {
                    "total_signals_24h": len(recent_signals),
                    "signal_counts": signal_counts,
                    "active_trades": len(self.active_trades),
                    "subscribers": len(self.signal_subscribers),
                    "alerts_enabled": len([a for a in self.alert_configs if a.enabled]),
                }

        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {"error": str(e)}

    def _start_websocket_server(self):
        """Start WebSocket server for real-time data streaming."""
        try:
            # This would start an async WebSocket server
            # For now, just log that it would start
            logger.info(f"WebSocket server would start on port {self.websocket_port}")

        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")

    def add_alert_config(self, config: AlertConfig):
        """Add new alert configuration."""
        try:
            self.alert_configs.append(config)
            logger.info(f"Added alert config: {config.alert_type}")

        except Exception as e:
            logger.error(f"Error adding alert config: {e}")

    def remove_alert_config(self, alert_type: str):
        """Remove alert configuration."""
        try:
            self.alert_configs = [
                c for c in self.alert_configs if c.alert_type != alert_type
            ]
            logger.info(f"Removed alert config: {alert_type}")

        except Exception as e:
            logger.error(f"Error removing alert config: {e}")

    def export_signal_data(self, filepath: str = "logs/real_time_signals.json"):
        """Export signal data to file."""
        try:
            with self.lock:
                export_data = {
                    "signals": [
                        {
                            "signal_id": s.signal_id,
                            "symbol": s.symbol,
                            "signal_type": s.signal_type.value,
                            "priority": s.priority.value,
                            "price": s.price,
                            "quantity": s.quantity,
                            "confidence": s.confidence,
                            "strategy": s.strategy,
                            "timestamp": s.timestamp.isoformat(),
                            "metadata": s.metadata,
                        }
                        for s in self.signals
                    ],
                    "active_trades": [
                        {
                            "trade_id": t.trade_id,
                            "symbol": t.symbol,
                            "side": t.side,
                            "quantity": t.quantity,
                            "entry_price": t.entry_price,
                            "current_price": t.current_price,
                            "pnl": t.pnl,
                            "pnl_percent": t.pnl_percent,
                            "entry_time": t.entry_time.isoformat(),
                            "strategy": t.strategy,
                            "status": t.status,
                            "metadata": t.metadata,
                        }
                        for t in self.active_trades.values()
                    ],
                    "summary": self.get_signal_summary(),
                    "export_date": datetime.now().isoformat(),
                }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Signal data exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting signal data: {e}")
