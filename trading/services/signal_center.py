"""Real-Time Signal Center.

This module provides live signal streaming dashboard with active trades,
time since signal, strategy that triggered it, and Discord/email webhook alerts.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import warnings
warnings.filterwarnings('ignore')

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
        self.signal_expiry_hours = self.config.get('signal_expiry_hours', 24)
        self.max_active_signals = self.config.get('max_active_signals', 100)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        
        # Webhook settings
        self.webhook_config = self.config.get('webhook_config', {
            'discord_webhook_url': '',
            'email_webhook_url': '',
            'slack_webhook_url': '',
            'enable_alerts': True,
            'alert_confidence_threshold': 0.7
        })
        
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
    
    def add_signal(self,
                   symbol: str,
                   signal_type: SignalType,
                   strategy: str,
                   confidence: float,
                   price: float,
                   target_price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
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
                metadata=metadata or {}
            )
            
            # Store signal
            self.active_signals[signal_id] = signal
            self.signal_history.append(signal)
            
            # Check if signal should trigger alert
            if (self.webhook_config['enable_alerts'] and 
                confidence >= self.webhook_config['alert_confidence_threshold']):
                self._send_alert(signal)
            
            # Clean up old signals
            self._cleanup_expired_signals()
            
            logger.info(f"Added signal {signal_id}: {signal_type.value} {symbol} "
                       f"at {price:.2f} (confidence: {confidence:.2%})")
            
            return signal_id
            
        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return ""
    
    def update_signal_status(self, signal_id: str, status: SignalStatus, metadata: Optional[Dict[str, Any]] = None):
        """Update signal status.
        
        Args:
            signal_id: Signal ID
            status: New status
            metadata: Additional metadata
        """
        try:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]
                signal.status = status
                
                if metadata:
                    signal.metadata.update(metadata)
                
                # Remove from active if not active
                if status != SignalStatus.ACTIVE:
                    del self.active_signals[signal_id]
                
                logger.info(f"Updated signal {signal_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
    
    def add_trade(self,
                  trade_id: str,
                  symbol: str,
                  side: str,
                  entry_price: float,
                  quantity: float,
                  strategy: str,
                  signal_id: str,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
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
            True if trade added successfully
        """
        try:
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
            
            self.active_trades[trade_id] = trade
            self.trade_history.append(trade)
            
            logger.info(f"Added trade {trade_id}: {side} {quantity} {symbol} at {entry_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False
    
    def update_trade_price(self, trade_id: str, current_price: float):
        """Update trade with current price.
        
        Args:
            trade_id: Trade ID
            current_price: Current market price
        """
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade.current_price = current_price
                
                # Calculate P&L
                if trade.side.lower() == 'buy':
                    trade.pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - current_price) * trade.quantity
                
                trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
                
        except Exception as e:
            logger.error(f"Error updating trade price: {e}")
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str = "manual"):
        """Close an active trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                
                # Calculate final P&L
                if trade.side.lower() == 'buy':
                    final_pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    final_pnl = (trade.entry_price - exit_price) * trade.quantity
                
                final_pnl_pct = (final_pnl / (trade.entry_price * trade.quantity)) * 100
                
                # Update trade metadata
                trade.metadata.update({
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'final_pnl': final_pnl,
                    'final_pnl_pct': final_pnl_pct,
                    'time_closed': datetime.now(),
                    'duration_hours': (datetime.now() - trade.time_open).total_seconds() / 3600
                })
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                # Update performance tracking
                self._update_performance_tracking(trade, final_pnl_pct)
                
                logger.info(f"Closed trade {trade_id}: {final_pnl:.2f} ({final_pnl_pct:.2%})")
                
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get active signals.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of active signals
        """
        try:
            signals = list(self.active_signals.values())
            
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Sort by confidence (highest first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def get_active_trades(self, symbol: Optional[str] = None) -> List[ActiveTrade]:
        """Get active trades.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of active trades
        """
        try:
            trades = list(self.active_trades.values())
            
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            
            # Sort by P&L (highest first)
            trades.sort(key=lambda x: x.pnl, reverse=True)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return []
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get signal center summary.
        
        Returns:
            Summary dictionary
        """
        try:
            # Active signals summary
            active_signals = list(self.active_signals.values())
            signal_types = {}
            strategy_signals = {}
            
            for signal in active_signals:
                signal_types[signal.signal_type.value] = signal_types.get(signal.signal_type.value, 0) + 1
                strategy_signals[signal.strategy] = strategy_signals.get(signal.strategy, 0) + 1
            
            # Active trades summary
            active_trades = list(self.active_trades.values())
            total_pnl = sum(trade.pnl for trade in active_trades)
            total_pnl_pct = np.mean([trade.pnl_pct for trade in active_trades]) if active_trades else 0
            
            # Performance summary
            winning_trades = [t for t in self.trade_history if t.metadata.get('final_pnl', 0) > 0]
            win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
            
            return {
                'total_active_signals': len(active_signals),
                'total_active_trades': len(active_trades),
                'signal_types': signal_types,
                'strategy_signals': strategy_signals,
                'total_pnl': total_pnl,
                'avg_pnl_pct': total_pnl_pct,
                'win_rate': win_rate,
                'total_trades': len(self.trade_history),
                'total_alerts': len(self.alert_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}
    
    def get_signal_performance(self, strategy: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get signal performance metrics.
        
        Args:
            strategy: Filter by strategy (optional)
            days: Number of days to look back
            
        Returns:
            Performance metrics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter signals by date and strategy
            recent_signals = [s for s in self.signal_history if s.timestamp > cutoff_date]
            if strategy:
                recent_signals = [s for s in recent_signals if s.strategy == strategy]
            
            if not recent_signals:
                return {}
            
            # Calculate metrics
            total_signals = len(recent_signals)
            high_confidence_signals = len([s for s in recent_signals if s.confidence >= 0.8])
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            
            # Signal type distribution
            signal_types = {}
            for signal in recent_signals:
                signal_types[signal.signal_type.value] = signal_types.get(signal.signal_type.value, 0) + 1
            
            # Strategy performance
            strategy_perf = {}
            for signal in recent_signals:
                if signal.strategy not in strategy_perf:
                    strategy_perf[signal.strategy] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'signals': []
                    }
                
                strategy_perf[signal.strategy]['count'] += 1
                strategy_perf[signal.strategy]['signals'].append(signal.confidence)
            
            # Calculate average confidence per strategy
            for strat, data in strategy_perf.items():
                data['avg_confidence'] = np.mean(data['signals'])
            
            return {
                'total_signals': total_signals,
                'high_confidence_signals': high_confidence_signals,
                'avg_confidence': avg_confidence,
                'signal_types': signal_types,
                'strategy_performance': strategy_perf,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {}
    
    def _send_alert(self, signal: Signal):
        """Send alert for high-confidence signal.
        
        Args:
            signal: Trading signal
        """
        try:
            # Create alert message
            message = self._create_alert_message(signal)
            
            # Send to Discord
            if self.webhook_config.get('discord_webhook_url'):
                self._send_discord_alert(message, signal)
            
            # Send to email
            if self.webhook_config.get('email_webhook_url'):
                self._send_email_alert(message, signal)
            
            # Send to Slack
            if self.webhook_config.get('slack_webhook_url'):
                self._send_slack_alert(message, signal)
            
            # Store alert
            self.alert_history.append({
                'signal_id': signal.signal_id,
                'timestamp': datetime.now(),
                'message': message,
                'webhooks_sent': ['discord', 'email', 'slack']
            })
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _create_alert_message(self, signal: Signal) -> str:
        """Create alert message for signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Alert message
        """
        try:
            emoji = {
                SignalType.BUY: "🟢",
                SignalType.SELL: "🔴",
                SignalType.STRONG_BUY: "🟢💪",
                SignalType.STRONG_SELL: "🔴💪",
                SignalType.HOLD: "🟡",
                SignalType.ALERT: "⚠️"
            }.get(signal.signal_type, "📊")
            
            message = f"{emoji} **{signal.signal_type.value.upper()} Signal**\n"
            message += f"**Symbol:** {signal.symbol}\n"
            message += f"**Strategy:** {signal.strategy}\n"
            message += f"**Price:** ${signal.price:.2f}\n"
            message += f"**Target:** ${signal.target_price:.2f}\n"
            message += f"**Stop Loss:** ${signal.stop_loss:.2f}\n"
            message += f"**Confidence:** {signal.confidence:.1%}\n"
            message += f"**Time:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating alert message: {e}")
            return f"Signal: {signal.signal_type.value} {signal.symbol}"
    
    def _send_discord_alert(self, message: str, signal: Signal):
        """Send Discord webhook alert.
        
        Args:
            message: Alert message
            signal: Trading signal
        """
        try:
            if not REQUESTS_AVAILABLE:
                return
            
            webhook_url = self.webhook_config.get('discord_webhook_url')
            if not webhook_url:
                return
            
            payload = {
                "content": message,
                "username": "Evolve Trading Bot",
                "avatar_url": "https://example.com/bot-avatar.png"
            }
            
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 204:
                logger.info(f"Discord alert sent for signal {signal.signal_id}")
            else:
                logger.warning(f"Discord alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    def _send_email_alert(self, message: str, signal: Signal):
        """Send email webhook alert.
        
        Args:
            message: Alert message
            signal: Trading signal
        """
        try:
            if not REQUESTS_AVAILABLE:
                return
            
            webhook_url = self.webhook_config.get('email_webhook_url')
            if not webhook_url:
                return
            
            payload = {
                "subject": f"Trading Signal: {signal.signal_type.value.upper()} {signal.symbol}",
                "body": message,
                "priority": "high" if signal.confidence >= 0.8 else "normal"
            }
            
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Email alert sent for signal {signal.signal_id}")
            else:
                logger.warning(f"Email alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, message: str, signal: Signal):
        """Send Slack webhook alert.
        
        Args:
            message: Alert message
            signal: Trading signal
        """
        try:
            if not REQUESTS_AVAILABLE:
                return
            
            webhook_url = self.webhook_config.get('slack_webhook_url')
            if not webhook_url:
                return
            
            payload = {
                "text": message,
                "channel": "#trading-signals",
                "username": "Evolve Trading Bot"
            }
            
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for signal {signal.signal_id}")
            else:
                logger.warning(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _cleanup_expired_signals(self):
        """Remove expired signals from active list."""
        try:
            current_time = datetime.now()
            expired_signals = []
            
            for signal_id, signal in self.active_signals.items():
                if current_time > signal.expiry:
                    expired_signals.append(signal_id)
                    signal.status = SignalStatus.EXPIRED
            
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
            
            if expired_signals:
                logger.info(f"Cleaned up {len(expired_signals)} expired signals")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")
    
    def _update_performance_tracking(self, trade: ActiveTrade, final_pnl_pct: float):
        """Update performance tracking for closed trade.
        
        Args:
            trade: Closed trade
            final_pnl_pct: Final P&L percentage
        """
        try:
            # Update signal performance
            signal_id = trade.signal_id
            if signal_id not in self.signal_performance:
                self.signal_performance[signal_id] = []
            
            self.signal_performance[signal_id].append({
                'trade_id': trade.trade_id,
                'pnl_pct': final_pnl_pct,
                'duration_hours': trade.metadata.get('duration_hours', 0),
                'exit_reason': trade.metadata.get('exit_reason', 'unknown')
            })
            
            # Update strategy performance
            strategy = trade.strategy
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = []
            
            self.strategy_performance[strategy].append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'pnl_pct': final_pnl_pct,
                'duration_hours': trade.metadata.get('duration_hours', 0)
            })
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def export_signal_report(self, filepath: str) -> bool:
        """Export signal report.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            report_data = []
            
            # Add active signals
            for signal in self.active_signals.values():
                row = {
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type.value,
                    'strategy': signal.strategy,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'timestamp': signal.timestamp,
                    'status': signal.status.value
                }
                report_data.append(row)
            
            # Add active trades
            for trade in self.active_trades.values():
                row = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'entry_price': trade.entry_price,
                    'current_price': trade.current_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'strategy': trade.strategy,
                    'signal_id': trade.signal_id,
                    'time_open': trade.time_open
                }
                report_data.append(row)
            
            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Signal report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting signal report: {e}")
            return False

# Global signal center instance
signal_center = SignalCenter()

def get_signal_center() -> SignalCenter:
    """Get the global signal center instance."""
    return signal_center 