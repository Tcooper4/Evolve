"""
Execution Risk Control Agent

Enforces trade constraints, cooling periods, and risk limits.
Provides comprehensive risk management and trade execution controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for trade execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradeStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"

@dataclass
class TradeRequest:
    """Trade request with risk parameters."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit', 'stop'
    timestamp: datetime
    strategy: str
    confidence: float
    risk_score: float
    metadata: Dict[str, Any]

@dataclass
class RiskCheck:
    """Result of risk check."""
    passed: bool
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    max_allowed_quantity: float
    cooling_period_remaining: int  # minutes

@dataclass
class ExecutionResult:
    """Result of trade execution."""
    trade_id: str
    status: TradeStatus
    executed_quantity: float
    executed_price: float
    execution_time: datetime
    slippage: float
    commission: float
    risk_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class ExecutionRiskControlAgent:
    """Advanced execution risk control agent with comprehensive risk management."""
    
    def __init__(self, 
                 max_position_size: float = 0.25,
                 max_daily_trades: int = 50,
                 max_daily_loss: float = 0.05,
                 cooling_period_minutes: int = 30,
                 correlation_threshold: float = 0.7,
                 volatility_threshold: float = 0.5):
        """Initialize the execution risk control agent.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_daily_trades: Maximum trades per day
            max_daily_loss: Maximum daily loss as fraction of portfolio
            cooling_period_minutes: Minutes to wait between trades
            correlation_threshold: Threshold for high correlation warning
            volatility_threshold: Threshold for high volatility warning
        """
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.cooling_period_minutes = cooling_period_minutes
        self.correlation_threshold = correlation_threshold
        self.volatility_threshold = volatility_threshold
        
        # Initialize tracking
        self.trade_history = []
        self.risk_history = []
        self.position_tracker = defaultdict(float)
        self.daily_trade_count = defaultdict(int)
        self.daily_pnl = defaultdict(float)
        self.last_trade_time = {}
        self.risk_limits = self._initialize_risk_limits()
        
        # Load existing history
        self._load_history()
        
        logger.info("Execution Risk Control Agent initialized successfully")
    
    def _initialize_risk_limits(self) -> Dict[str, Any]:
        """Initialize risk limits and thresholds."""
        return {
            'position_limits': {
                'max_single_position': 0.25,
                'max_sector_exposure': 0.4,
                'max_correlation_exposure': 0.6
            },
            'trade_limits': {
                'max_trade_size': 0.1,
                'min_trade_size': 0.001,
                'max_slippage': 0.02
            },
            'risk_limits': {
                'max_drawdown': 0.15,
                'max_var_95': 0.03,
                'max_beta': 1.5
            },
            'timing_limits': {
                'market_hours_only': True,
                'avoid_earnings': True,
                'avoid_high_impact_news': True
            }
        }
    
    def _load_history(self):
        """Load existing trade and risk history."""
        try:
            # Load trade history
            trade_history_path = "logs/trade_history.json"
            if os.path.exists(trade_history_path):
                with open(trade_history_path, 'r') as f:
                    self.trade_history = json.load(f)
            
            # Load risk history
            risk_history_path = "logs/risk_history.json"
            if os.path.exists(risk_history_path):
                with open(risk_history_path, 'r') as f:
                    self.risk_history = json.load(f)
            
            # Reconstruct position tracker
            for trade in self.trade_history:
                symbol = trade['symbol']
                side = trade['side']
                quantity = trade['executed_quantity']
                
                if side == 'buy':
                    self.position_tracker[symbol] += quantity
                else:
                    self.position_tracker[symbol] -= quantity
            
            logger.info(f"Loaded {len(self.trade_history)} trades from history")
            
        except Exception as e:
            logger.warning(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save trade and risk history."""
        try:
            # Save trade history
            trade_history_path = "logs/trade_history.json"
            os.makedirs(os.path.dirname(trade_history_path), exist_ok=True)
            with open(trade_history_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            
            # Save risk history
            risk_history_path = "logs/risk_history.json"
            with open(risk_history_path, 'w') as f:
                json.dump(self.risk_history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def check_trade_risk(self, trade_request: TradeRequest) -> RiskCheck:
        """Comprehensive risk check for trade request."""
        try:
            violations = []
            warnings = []
            recommendations = []
            
            # 1. Position size check
            position_check = self._check_position_size(trade_request)
            if not position_check['passed']:
                violations.append(position_check['reason'])
            if position_check.get('warning'):
                warnings.append(position_check['warning'])
            
            # 2. Daily limits check
            daily_check = self._check_daily_limits(trade_request)
            if not daily_check['passed']:
                violations.append(daily_check['reason'])
            if daily_check.get('warning'):
                warnings.append(daily_check['warning'])
            
            # 3. Cooling period check
            cooling_check = self._check_cooling_period(trade_request)
            if not cooling_check['passed']:
                violations.append(cooling_check['reason'])
            
            # 4. Risk score check
            risk_check = self._check_risk_score(trade_request)
            if not risk_check['passed']:
                violations.append(risk_check['reason'])
            if risk_check.get('warning'):
                warnings.append(risk_check['warning'])
            
            # 5. Market conditions check
            market_check = self._check_market_conditions(trade_request)
            if not market_check['passed']:
                violations.append(market_check['reason'])
            if market_check.get('warning'):
                warnings.append(market_check['warning'])
            
            # 6. Correlation check
            correlation_check = self._check_correlation_risk(trade_request)
            if correlation_check.get('warning'):
                warnings.append(correlation_check['warning'])
            
            # Determine risk level
            risk_level = self._determine_risk_level(violations, warnings, trade_request.risk_score)
            
            # Calculate max allowed quantity
            max_quantity = self._calculate_max_allowed_quantity(trade_request, violations)
            
            # Calculate cooling period remaining
            cooling_remaining = self._calculate_cooling_remaining(trade_request.symbol)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(violations, warnings, trade_request)
            
            # Check if trade passes
            passed = len(violations) == 0
            
            return RiskCheck(
                passed=passed,
                risk_level=risk_level,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                max_allowed_quantity=max_quantity,
                cooling_period_remaining=cooling_remaining
            )
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return RiskCheck(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                violations=[f"Risk check error: {str(e)}"],
                warnings=[],
                recommendations=["Contact system administrator"],
                max_allowed_quantity=0.0,
                cooling_period_remaining=0
            )
    
    def _check_position_size(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check position size limits."""
        try:
            current_position = self.position_tracker.get(trade_request.symbol, 0.0)
            new_position = current_position + (trade_request.quantity if trade_request.side == 'buy' else -trade_request.quantity)
            
            # Check single position limit
            position_value = abs(new_position) * trade_request.price
            portfolio_value = self._get_portfolio_value()
            position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_ratio > self.max_position_size:
                return {
                    'passed': False,
                    'reason': f"Position size {position_ratio:.2%} exceeds limit {self.max_position_size:.2%}"
                }
            
            # Warning for large positions
            if position_ratio > self.max_position_size * 0.8:
                return {
                    'passed': True,
                    'warning': f"Large position size: {position_ratio:.2%}"
                }
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error checking position size: {e}")
            return {'passed': False, 'reason': 'Position size check failed'}
    
    def _check_daily_limits(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check daily trading limits."""
        try:
            today = datetime.now().date().isoformat()
            
            # Check daily trade count
            daily_trades = self.daily_trade_count.get(today, 0)
            if daily_trades >= self.max_daily_trades:
                return {
                    'passed': False,
                    'reason': f"Daily trade limit {self.max_daily_trades} exceeded"
                }
            
            # Check daily loss limit
            daily_loss = abs(self.daily_pnl.get(today, 0.0))
            portfolio_value = self._get_portfolio_value()
            loss_ratio = daily_loss / portfolio_value if portfolio_value > 0 else 0
            
            if loss_ratio > self.max_daily_loss:
                return {
                    'passed': False,
                    'reason': f"Daily loss limit {self.max_daily_loss:.2%} exceeded"
                }
            
            # Warning for approaching limits
            if daily_trades >= self.max_daily_trades * 0.8:
                return {
                    'passed': True,
                    'warning': f"Approaching daily trade limit: {daily_trades}/{self.max_daily_trades}"
                }
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error checking daily limits: {e}")
            return {'passed': False, 'reason': 'Daily limits check failed'}
    
    def _check_cooling_period(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check cooling period between trades."""
        try:
            last_trade = self.last_trade_time.get(trade_request.symbol)
            
            if last_trade:
                time_since_last = datetime.now() - datetime.fromisoformat(last_trade)
                minutes_since_last = time_since_last.total_seconds() / 60
                
                if minutes_since_last < self.cooling_period_minutes:
                    return {
                        'passed': False,
                        'reason': f"Cooling period not met: {self.cooling_period_minutes - minutes_since_last:.0f} minutes remaining"
                    }
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error checking cooling period: {e}")
            return {'passed': False, 'reason': 'Cooling period check failed'}
    
    def _check_risk_score(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check trade risk score."""
        try:
            if trade_request.risk_score > 0.8:
                return {
                    'passed': False,
                    'reason': f"Risk score {trade_request.risk_score:.2f} too high"
                }
            
            if trade_request.risk_score > 0.6:
                return {
                    'passed': True,
                    'warning': f"High risk score: {trade_request.risk_score:.2f}"
                }
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error checking risk score: {e}")
            return {'passed': False, 'reason': 'Risk score check failed'}
    
    def _check_market_conditions(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check market conditions for trade."""
        try:
            # Check if within market hours (simplified)
            current_hour = datetime.now().hour
            if not (9 <= current_hour <= 16):  # 9 AM to 4 PM
                return {
                    'passed': False,
                    'reason': "Outside market hours"
                }
            
            # Check for high volatility (simplified)
            # In practice, this would use real market data
            if trade_request.metadata.get('volatility', 0) > self.volatility_threshold:
                return {
                    'passed': True,
                    'warning': f"High volatility: {trade_request.metadata.get('volatility', 0):.2f}"
                }
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return {'passed': False, 'reason': 'Market conditions check failed'}
    
    def _check_correlation_risk(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check correlation risk with existing positions."""
        try:
            # Simplified correlation check
            # In practice, this would calculate actual correlations
            existing_positions = [sym for sym, pos in self.position_tracker.items() if abs(pos) > 0]
            
            if len(existing_positions) > 0:
                # Assume some correlation risk for demonstration
                correlation_risk = 0.3  # Simplified
                
                if correlation_risk > self.correlation_threshold:
                    return {
                        'warning': f"High correlation with existing positions: {correlation_risk:.2f}"
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return {}
    
    def _determine_risk_level(self, 
                            violations: List[str], 
                            warnings: List[str], 
                            risk_score: float) -> RiskLevel:
        """Determine overall risk level."""
        try:
            if violations:
                return RiskLevel.CRITICAL
            elif risk_score > 0.7 or len(warnings) > 2:
                return RiskLevel.HIGH
            elif risk_score > 0.5 or len(warnings) > 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.CRITICAL
    
    def _calculate_max_allowed_quantity(self, 
                                      trade_request: TradeRequest, 
                                      violations: List[str]) -> float:
        """Calculate maximum allowed quantity based on violations."""
        try:
            if violations:
                return 0.0
            
            # Start with requested quantity
            max_quantity = trade_request.quantity
            
            # Reduce based on risk score
            risk_reduction = 1.0 - trade_request.risk_score
            max_quantity *= risk_reduction
            
            # Apply position size limit
            portfolio_value = self._get_portfolio_value()
            max_position_value = portfolio_value * self.max_position_size
            max_quantity_by_value = max_position_value / trade_request.price
            
            max_quantity = min(max_quantity, max_quantity_by_value)
            
            return max(0.0, max_quantity)
            
        except Exception as e:
            logger.error(f"Error calculating max quantity: {e}")
            return 0.0
    
    def _calculate_cooling_remaining(self, symbol: str) -> int:
        """Calculate remaining cooling period in minutes."""
        try:
            last_trade = self.last_trade_time.get(symbol)
            
            if not last_trade:
                return 0
            
            time_since_last = datetime.now() - datetime.fromisoformat(last_trade)
            minutes_since_last = time_since_last.total_seconds() / 60
            
            remaining = max(0, self.cooling_period_minutes - minutes_since_last)
            return int(remaining)
            
        except Exception as e:
            logger.error(f"Error calculating cooling remaining: {e}")
            return 0
    
    def _generate_recommendations(self, 
                                violations: List[str], 
                                warnings: List[str], 
                                trade_request: TradeRequest) -> List[str]:
        """Generate recommendations based on violations and warnings."""
        recommendations = []
        
        try:
            # Position size recommendations
            if any('position size' in v.lower() for v in violations):
                recommendations.append("Consider reducing position size")
            
            # Risk score recommendations
            if trade_request.risk_score > 0.6:
                recommendations.append("Consider waiting for better entry point")
            
            # Cooling period recommendations
            if any('cooling period' in v.lower() for v in violations):
                recommendations.append("Wait for cooling period to expire")
            
            # Market conditions recommendations
            if any('market hours' in v.lower() for v in violations):
                recommendations.append("Execute during market hours")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Trade approved with current parameters")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Contact system administrator"]
    
    def execute_trade(self, trade_request: TradeRequest) -> ExecutionResult:
        """Execute trade with risk controls."""
        try:
            # Perform risk check
            risk_check = self.check_trade_risk(trade_request)
            
            if not risk_check.passed:
                return ExecutionResult(
                    trade_id=self._generate_trade_id(),
                    status=TradeStatus.REJECTED,
                    executed_quantity=0.0,
                    executed_price=0.0,
                    execution_time=datetime.now(),
                    slippage=0.0,
                    commission=0.0,
                    risk_metrics={},
                    metadata={'rejection_reason': risk_check.violations}
                )
            
            # Adjust quantity if needed
            actual_quantity = min(trade_request.quantity, risk_check.max_allowed_quantity)
            
            # Simulate execution
            execution_price = self._simulate_execution(trade_request, actual_quantity)
            slippage = abs(execution_price - trade_request.price) / trade_request.price
            commission = self._calculate_commission(actual_quantity, execution_price)
            
            # Update tracking
            self._update_trade_tracking(trade_request, actual_quantity, execution_price)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_execution_risk_metrics(trade_request, actual_quantity, execution_price)
            
            # Create execution result
            result = ExecutionResult(
                trade_id=self._generate_trade_id(),
                status=TradeStatus.EXECUTED,
                executed_quantity=actual_quantity,
                executed_price=execution_price,
                execution_time=datetime.now(),
                slippage=slippage,
                commission=commission,
                risk_metrics=risk_metrics,
                metadata={
                    'risk_check': {
                        'risk_level': risk_check.risk_level.value,
                        'warnings': risk_check.warnings,
                        'recommendations': risk_check.recommendations
                    }
                }
            )
            
            # Store trade history
            self._store_trade_history(result)
            
            logger.info(f"Trade executed: {result.trade_id}, Quantity: {actual_quantity}, "
                       f"Price: {execution_price:.2f}, Risk Level: {risk_check.risk_level.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return ExecutionResult(
                trade_id=self._generate_trade_id(),
                status=TradeStatus.REJECTED,
                executed_quantity=0.0,
                executed_price=0.0,
                execution_time=datetime.now(),
                slippage=0.0,
                commission=0.0,
                risk_metrics={},
                metadata={'error': str(e)}
            )
    
    def _simulate_execution(self, trade_request: TradeRequest, quantity: float) -> float:
        """Simulate trade execution with slippage."""
        try:
            base_price = trade_request.price
            
            # Add slippage based on order type and size
            if trade_request.order_type == 'market':
                slippage_factor = 0.001  # 0.1% for market orders
            else:
                slippage_factor = 0.0005  # 0.05% for limit orders
            
            # Size-based slippage
            size_slippage = min(0.005, quantity * 0.001)  # Max 0.5%
            
            total_slippage = slippage_factor + size_slippage
            
            # Apply slippage
            if trade_request.side == 'buy':
                execution_price = base_price * (1 + total_slippage)
            else:
                execution_price = base_price * (1 - total_slippage)
            
            return execution_price
            
        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            return trade_request.price
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for trade."""
        try:
            trade_value = quantity * price
            
            # Simple commission structure
            if trade_value < 10000:
                commission_rate = 0.01  # 1%
            elif trade_value < 100000:
                commission_rate = 0.005  # 0.5%
            else:
                commission_rate = 0.002  # 0.2%
            
            return trade_value * commission_rate
            
        except Exception as e:
            logger.error(f"Error calculating commission: {e}")
            return 0.0
    
    def _update_trade_tracking(self, trade_request: TradeRequest, quantity: float, price: float):
        """Update trade tracking data."""
        try:
            # Update position tracker
            if trade_request.side == 'buy':
                self.position_tracker[trade_request.symbol] += quantity
            else:
                self.position_tracker[trade_request.symbol] -= quantity
            
            # Update daily trade count
            today = datetime.now().date().isoformat()
            self.daily_trade_count[today] += 1
            
            # Update last trade time
            self.last_trade_time[trade_request.symbol] = datetime.now().isoformat()
            
            # Update daily PnL (simplified)
            trade_pnl = quantity * price * (1 if trade_request.side == 'sell' else -1)
            self.daily_pnl[today] += trade_pnl
            
        except Exception as e:
            logger.error(f"Error updating trade tracking: {e}")
    
    def _calculate_execution_risk_metrics(self, 
                                        trade_request: TradeRequest, 
                                        quantity: float, 
                                        price: float) -> Dict[str, float]:
        """Calculate risk metrics for executed trade."""
        try:
            trade_value = quantity * price
            portfolio_value = self._get_portfolio_value()
            
            return {
                'trade_value': trade_value,
                'portfolio_exposure': trade_value / portfolio_value if portfolio_value > 0 else 0,
                'position_risk': trade_request.risk_score * trade_value,
                'slippage_cost': abs(price - trade_request.price) * quantity,
                'commission_cost': self._calculate_commission(quantity, price)
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution risk metrics: {e}")
            return {}
    
    def _store_trade_history(self, execution_result: ExecutionResult):
        """Store trade in history."""
        try:
            trade_record = {
                'trade_id': execution_result.trade_id,
                'symbol': execution_result.metadata.get('symbol', ''),
                'side': execution_result.metadata.get('side', ''),
                'executed_quantity': execution_result.executed_quantity,
                'executed_price': execution_result.executed_price,
                'execution_time': execution_result.execution_time.isoformat(),
                'status': execution_result.status.value,
                'slippage': execution_result.slippage,
                'commission': execution_result.commission,
                'risk_metrics': execution_result.risk_metrics
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only last 10000 trades
            if len(self.trade_history) > 10000:
                self.trade_history = self.trade_history[-10000:]
            
            # Save history
            self._save_history()
            
        except Exception as e:
            logger.error(f"Error storing trade history: {e}")
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        return f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value (simplified)."""
        # In practice, this would get the actual portfolio value
        return 100000.0  # Default value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            # Calculate risk metrics
            total_trades = len(self.trade_history)
            executed_trades = len([t for t in self.trade_history if t['status'] == 'executed'])
            rejected_trades = len([t for t in self.trade_history if t['status'] == 'rejected'])
            
            # Position summary
            total_positions = len([p for p in self.position_tracker.values() if abs(p) > 0])
            largest_position = max([abs(p) for p in self.position_tracker.values()]) if self.position_tracker else 0
            
            # Daily summary
            today = datetime.now().date().isoformat()
            daily_trades = self.daily_trade_count.get(today, 0)
            daily_pnl = self.daily_pnl.get(today, 0.0)
            
            return {
                'total_trades': total_trades,
                'executed_trades': executed_trades,
                'rejected_trades': rejected_trades,
                'execution_rate': executed_trades / total_trades if total_trades > 0 else 0,
                'total_positions': total_positions,
                'largest_position': largest_position,
                'daily_trades': daily_trades,
                'daily_pnl': daily_pnl,
                'current_drawdown': self._calculate_current_drawdown(),
                'risk_limits': self.risk_limits
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {'error': str(e)}
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown."""
        try:
            # Simplified drawdown calculation
            # In practice, this would use actual portfolio performance
            return 0.05  # 5% drawdown for demonstration
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def export_risk_data(self, filepath: str = "logs/execution_risk_data.json"):
        """Export risk control data to file."""
        try:
            export_data = {
                'trade_history': self.trade_history,
                'risk_history': self.risk_history,
                'position_tracker': dict(self.position_tracker),
                'daily_trade_count': dict(self.daily_trade_count),
                'daily_pnl': dict(self.daily_pnl),
                'last_trade_time': self.last_trade_time,
                'risk_limits': self.risk_limits,
                'summary': self.get_risk_summary(),
                'export_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Risk control data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting risk data: {e}") 