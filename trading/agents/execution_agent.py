"""
Execution Agent

This agent handles trade execution, position tracking, and portfolio management.
It currently operates in simulation mode with hooks for real execution via
Alpaca, Interactive Brokers, or Robinhood APIs.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# Local imports
from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentStatus, AgentResult
from trading.portfolio.portfolio_manager import PortfolioManager, Position, TradeDirection, PositionStatus
from trading.portfolio.position_sizer import (
    PositionSizer, SizingParameters, SizingStrategy,
    MarketContext, SignalContext, PortfolioContext
)
from trading.memory.agent_memory import AgentMemory

class ExecutionMode(Enum):
    """Execution mode enum."""
    SIMULATION = "simulation"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ROBINHOOD = "robinhood"

class RiskThresholdType(Enum):
    """Risk threshold type enum."""
    PERCENTAGE = "percentage"
    ATR_BASED = "atr_based"
    FIXED = "fixed"

class ExitReason(Enum):
    """Exit reason enum."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MAX_HOLDING_PERIOD = "max_holding_period"
    MANUAL = "manual"
    RISK_LIMIT = "risk_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    CORRELATION_LIMIT = "correlation_limit"

@dataclass
class RiskThreshold:
    """Risk threshold configuration."""
    threshold_type: RiskThresholdType
    value: float
    atr_multiplier: Optional[float] = None
    atr_period: int = 14
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'threshold_type': self.threshold_type.value,
            'value': self.value,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskThreshold':
        """Create from dictionary."""
        data['threshold_type'] = RiskThresholdType(data['threshold_type'])
        return cls(**data)

@dataclass
class RiskControls:
    """Risk controls configuration."""
    stop_loss: RiskThreshold
    take_profit: RiskThreshold
    max_position_size: float = 0.2  # 20% of capital
    max_portfolio_risk: float = 0.05  # 5% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_correlation: float = 0.7  # Maximum correlation between positions
    volatility_limit: float = 0.5  # Maximum volatility for new positions
    trailing_stop: bool = False
    trailing_stop_distance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stop_loss': self.stop_loss.to_dict(),
            'take_profit': self.take_profit.to_dict(),
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_daily_loss': self.max_daily_loss,
            'max_correlation': self.max_correlation,
            'volatility_limit': self.volatility_limit,
            'trailing_stop': self.trailing_stop,
            'trailing_stop_distance': self.trailing_stop_distance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskControls':
        """Create from dictionary."""
        data['stop_loss'] = RiskThreshold.from_dict(data['stop_loss'])
        data['take_profit'] = RiskThreshold.from_dict(data['take_profit'])
        return cls(**data)

@dataclass
class ExitEvent:
    """Exit event data class."""
    timestamp: datetime
    symbol: str
    position_id: str
    exit_price: float
    exit_reason: ExitReason
    pnl: float
    holding_period: timedelta
    risk_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        exit_dict = asdict(self)
        exit_dict['timestamp'] = self.timestamp.isoformat()
        exit_dict['exit_reason'] = self.exit_reason.value
        exit_dict['holding_period'] = self.holding_period.total_seconds()
        return exit_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExitEvent':
        """Create from dictionary."""
        data['exit_reason'] = ExitReason(data['exit_reason'])
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data['holding_period'], (int, float)):
            data['holding_period'] = timedelta(seconds=data['holding_period'])
        return cls(**data)

@dataclass
class TradeSignal:
    """Trade signal data class."""
    symbol: str
    direction: TradeDirection
    strategy: str
    confidence: float
    entry_price: float
    size: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    max_holding_period: Optional[timedelta] = None
    market_data: Optional[Dict[str, Any]] = None
    risk_controls: Optional[RiskControls] = None
    timestamp: datetime = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        signal_dict = asdict(self)
        signal_dict['timestamp'] = self.timestamp.isoformat()
        if self.max_holding_period:
            signal_dict['max_holding_period'] = self.max_holding_period.total_seconds()
        if self.risk_controls:
            signal_dict['risk_controls'] = self.risk_controls.to_dict()
        return signal_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeSignal':
        """Create from dictionary."""
        # Convert string enums back to enum values
        data['direction'] = TradeDirection(data['direction'])
        
        # Convert string timestamp to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert max_holding_period
        if 'max_holding_period' in data and isinstance(data['max_holding_period'], (int, float)):
            data['max_holding_period'] = timedelta(seconds=data['max_holding_period'])
        
        # Convert risk controls
        if 'risk_controls' in data and data['risk_controls']:
            data['risk_controls'] = RiskControls.from_dict(data['risk_controls'])
        
        return cls(**data)

@dataclass
class ExecutionRequest:
    """Request for execution agent operations."""
    operation_type: str  # 'execute', 'exit', 'status', etc.
    signal: Optional[TradeSignal] = None
    market_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.utcnow()

@dataclass
class ExecutionResult:
    """Execution result data class."""
    success: bool
    signal: TradeSignal
    position: Optional[Position] = None
    execution_price: Optional[float] = None
    slippage: float = 0.0
    fees: float = 0.0
    message: str = ""
    error: Optional[str] = None
    risk_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        result_dict['signal'] = self.signal.to_dict()
        if self.position:
            result_dict['position'] = self.position.to_dict()
        return result_dict

class ExecutionAgent(BaseAgent):
    """Agent responsible for trade execution and position tracking."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the execution agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        # Execution settings
        self.execution_mode = ExecutionMode(config.custom_config.get('execution_mode', 'simulation'))
        self.max_positions = config.custom_config.get('max_positions', 10)
        self.min_confidence = config.custom_config.get('min_confidence', 0.7)
        self.max_slippage = config.custom_config.get('max_slippage', 0.001)  # 10 bps
        self.execution_delay = config.custom_config.get('execution_delay', 1.0)  # seconds
        
        # Risk controls
        self.default_risk_controls = self._load_default_risk_controls()
        self.risk_monitoring_enabled = config.custom_config.get('risk_monitoring_enabled', True)
        self.auto_exit_enabled = config.custom_config.get('auto_exit_enabled', True)
        
        # Portfolio management
        self.portfolio_manager = PortfolioManager(config.custom_config.get('portfolio_config', {}))
        
        # Memory for tracking
        self.memory = AgentMemory()
        
        # Trade history
        self.trade_log_path = Path("trading/agents/logs/trade_log.json")
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Risk monitoring
        self.exit_log_path = Path("trading/agents/logs/exit_log.json")
        self.risk_log_path = Path("trading/agents/logs/risk_log.json")
        
        # Execution history
        self.execution_history: List[ExecutionResult] = []
        self.exit_events: List[ExitEvent] = []
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Risk monitoring state
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Price history for ATR calculations
        self.price_history: Dict[str, List[float]] = {}
        
        # Global market metrics
        self.global_metrics: Dict[str, Any] = {
            'market_regime': 'unknown',
            'volatility_regime': 'normal',
            'correlation_regime': 'normal'
        }
        
        # Position sizer
        self.position_sizer = PositionSizer(config.custom_config.get('position_sizing_config', {}))
        
        # Initialize execution providers (for future real execution)
        self._initialize_execution_providers()
        
        self.logger.info(f"ExecutionAgent initialized in {self.execution_mode.value} mode with risk controls")
    
    def _load_default_risk_controls(self) -> RiskControls:
        """Load default risk controls from config."""
        risk_config = self.config.custom_config.get('risk_controls', {})
        
        # Default stop loss (2% or 2x ATR)
        stop_loss_config = risk_config.get('stop_loss', {
            'threshold_type': 'percentage',
            'value': 0.02,
            'atr_multiplier': 2.0,
            'atr_period': 14
        })
        stop_loss = RiskThreshold.from_dict(stop_loss_config)
        
        # Default take profit (6% or 3x ATR)
        take_profit_config = risk_config.get('take_profit', {
            'threshold_type': 'percentage',
            'value': 0.06,
            'atr_multiplier': 3.0,
            'atr_period': 14
        })
        take_profit = RiskThreshold.from_dict(take_profit_config)
        
        return RiskControls(
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position_size=risk_config.get('max_position_size', 0.2),
            max_portfolio_risk=risk_config.get('max_portfolio_risk', 0.05),
            max_daily_loss=risk_config.get('max_daily_loss', 0.02),
            max_correlation=risk_config.get('max_correlation', 0.7),
            volatility_limit=risk_config.get('volatility_limit', 0.5),
            trailing_stop=risk_config.get('trailing_stop', False),
            trailing_stop_distance=risk_config.get('trailing_stop_distance', None)
        )
    
    def _initialize_execution_providers(self) -> None:
        """Initialize execution providers for different brokers."""
        self.execution_providers = {
            ExecutionMode.ALPACA: None,
            ExecutionMode.INTERACTIVE_BROKERS: None,
            ExecutionMode.ROBINHOOD: None
        }
        
        # Future: Initialize real execution providers
        if self.execution_mode == ExecutionMode.ALPACA:
            self._initialize_alpaca_provider()
        elif self.execution_mode == ExecutionMode.INTERACTIVE_BROKERS:
            self._initialize_ib_provider()
        elif self.execution_mode == ExecutionMode.ROBINHOOD:
            self._initialize_robinhood_provider()
    
    def _initialize_alpaca_provider(self) -> None:
        """Initialize Alpaca execution provider."""
        try:
            # Future: Add Alpaca SDK integration
            # from alpaca.trading.client import TradingClient
            # self.execution_providers[ExecutionMode.ALPACA] = TradingClient(...)
            self.logger.info("Alpaca provider placeholder initialized")
        except ImportError:
            self.logger.warning("Alpaca SDK not available")
    
    def _initialize_ib_provider(self) -> None:
        """Initialize Interactive Brokers execution provider."""
        try:
            # Future: Add IB API integration
            # from ibapi.client import EClient
            # self.execution_providers[ExecutionMode.INTERACTIVE_BROKERS] = EClient(...)
            self.logger.info("Interactive Brokers provider placeholder initialized")
        except ImportError:
            self.logger.warning("Interactive Brokers API not available")
    
    def _initialize_robinhood_provider(self) -> None:
        """Initialize Robinhood execution provider."""
        try:
            # Future: Add Robinhood API integration
            # from robin_stocks import robinhood
            # self.execution_providers[ExecutionMode.ROBINHOOD] = robinhood
            self.logger.info("Robinhood provider placeholder initialized")
        except ImportError:
            self.logger.warning("Robinhood API not available")
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main logic.
        
        Args:
            **kwargs: May include:
                - signals: List of TradeSignal objects
                - market_data: Current market data
                - portfolio_update: Whether to update portfolio
                - risk_check: Whether to perform risk checks
                
        Returns:
            AgentResult: Result of the execution
        """
        try:
            signals = kwargs.get('signals', [])
            market_data = kwargs.get('market_data', {})
            portfolio_update = kwargs.get('portfolio_update', True)
            risk_check = kwargs.get('risk_check', True)
            
            # Update market data cache
            self._update_market_data_cache(market_data)
            
            # Perform risk monitoring if enabled
            if risk_check and self.risk_monitoring_enabled:
                await self._monitor_risk_limits(market_data)
            
            # Process trade signals
            execution_results = []
            for signal in signals:
                result = await self._process_trade_signal(signal, market_data)
                execution_results.append(result)
                
                # Log execution
                self._log_execution_result(result)
                
                # Add to memory
                self.memory.log_decision(
                    agent_name=self.config.name,
                    decision_type='trade_execution',
                    details=result.to_dict()
                )
            
            # Update portfolio if requested
            if portfolio_update:
                self._update_portfolio(market_data)
            
            # Generate summary
            summary = self._generate_execution_summary(execution_results)
            
            return AgentResult(
                success=True,
                message=f"Processed {len(signals)} trade signals",
                data={
                    'execution_results': [r.to_dict() for r in execution_results],
                    'summary': summary,
                    'portfolio_state': self.portfolio_manager.get_status(),
                    'risk_metrics': self._calculate_portfolio_risk_metrics(market_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in execution: {e}")
            return AgentResult(
                success=False,
                message=f"Execution error: {str(e)}",
                error=e
            )
    
    async def _monitor_risk_limits(self, market_data: Dict[str, Any]) -> None:
        """Monitor and enforce risk limits.
        
        Args:
            market_data: Current market data
        """
        try:
            # Reset daily PnL if it's a new day
            current_time = datetime.utcnow()
            if current_time.date() > self.daily_reset_time.date():
                self.daily_pnl = 0.0
                self.daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check daily loss limit
            if abs(self.daily_pnl) > self.default_risk_controls.max_daily_loss:
                await self._emergency_exit_all_positions(
                    ExitReason.RISK_LIMIT,
                    f"Daily loss limit exceeded: {self.daily_pnl:.2%}",
                    market_data
                )
                return
            
            # Check each open position for risk limits
            for position in self.portfolio_manager.state.open_positions[:]:  # Copy to allow modification
                await self._check_position_risk_limits(position, market_data)
            
            # Check portfolio-level risk limits
            await self._check_portfolio_risk_limits(market_data)
            
        except Exception as e:
            self.logger.error(f"Error in risk monitoring: {e}")
    
    async def _check_position_risk_limits(self, position: Position, 
                                        market_data: Dict[str, Any]) -> None:
        """Check risk limits for a specific position.
        
        Args:
            position: Position to check
            market_data: Current market data
        """
        try:
            symbol = position.symbol
            if symbol not in market_data:
                return
            
            current_price = market_data[symbol].get('price', position.entry_price)
            
            # Get risk controls for this position
            risk_controls = self._get_position_risk_controls(position)
            
            # Check stop loss
            stop_loss_price = self._calculate_stop_loss_price(position, risk_controls, market_data)
            if self._should_exit_position(position, current_price, stop_loss_price, "stop_loss"):
                await self._exit_position(
                    position, 
                    current_price, 
                    ExitReason.STOP_LOSS,
                    f"Stop loss triggered at ${current_price:.2f}",
                    market_data
                )
                return
            
            # Check take profit
            take_profit_price = self._calculate_take_profit_price(position, risk_controls, market_data)
            if self._should_exit_position(position, current_price, take_profit_price, "take_profit"):
                await self._exit_position(
                    position, 
                    current_price, 
                    ExitReason.TAKE_PROFIT,
                    f"Take profit triggered at ${current_price:.2f}",
                    market_data
                )
                return
            
            # Check max holding period
            if position.max_holding_period:
                holding_time = datetime.utcnow() - position.entry_time
                if holding_time > position.max_holding_period:
                    await self._exit_position(
                        position, 
                        current_price, 
                        ExitReason.MAX_HOLDING_PERIOD,
                        f"Max holding period exceeded: {holding_time}",
                        market_data
                    )
                    return
            
            # Check volatility limit
            volatility = market_data[symbol].get('volatility', 0.0)
            if volatility > risk_controls.volatility_limit:
                await self._exit_position(
                    position, 
                    current_price, 
                    ExitReason.VOLATILITY_LIMIT,
                    f"Volatility limit exceeded: {volatility:.2%}",
                    market_data
                )
                return
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits for {position.symbol}: {e}")
    
    async def _check_portfolio_risk_limits(self, market_data: Dict[str, Any]) -> None:
        """Check portfolio-level risk limits.
        
        Args:
            market_data: Current market data
        """
        try:
            # Calculate portfolio correlation
            correlation = self._calculate_portfolio_correlation(market_data)
            if correlation > self.default_risk_controls.max_correlation:
                await self._emergency_exit_all_positions(
                    ExitReason.CORRELATION_LIMIT,
                    f"Portfolio correlation too high: {correlation:.2%}",
                    market_data
                )
                return
            
            # Check portfolio risk exposure
            portfolio_risk = self._calculate_portfolio_risk_exposure(market_data)
            if portfolio_risk > self.default_risk_controls.max_portfolio_risk:
                await self._emergency_exit_all_positions(
                    ExitReason.RISK_LIMIT,
                    f"Portfolio risk too high: {portfolio_risk:.2%}",
                    market_data
                )
                return
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk limits: {e}")
    
    def _get_position_risk_controls(self, position: Position) -> RiskControls:
        """Get risk controls for a position.
        
        Args:
            position: Position to get controls for
            
        Returns:
            Risk controls for the position
        """
        # Use position-specific controls if available, otherwise use defaults
        if hasattr(position, 'risk_controls') and position.risk_controls:
            return position.risk_controls
        return self.default_risk_controls
    
    def _calculate_stop_loss_price(self, position: Position, risk_controls: RiskControls,
                                 market_data: Dict[str, Any]) -> float:
        """Calculate stop loss price for a position.
        
        Args:
            position: Position to calculate for
            risk_controls: Risk controls to use
            market_data: Current market data
            
        Returns:
            Stop loss price
        """
        threshold = risk_controls.stop_loss
        
        if threshold.threshold_type == RiskThresholdType.PERCENTAGE:
            if position.direction == TradeDirection.LONG:
                return position.entry_price * (1 - threshold.value)
            else:
                return position.entry_price * (1 + threshold.value)
        
        elif threshold.threshold_type == RiskThresholdType.ATR_BASED:
            atr = self._calculate_atr(position.symbol, threshold.atr_period, market_data)
            atr_distance = atr * threshold.atr_multiplier
            
            if position.direction == TradeDirection.LONG:
                return position.entry_price - atr_distance
            else:
                return position.entry_price + atr_distance
        
        elif threshold.threshold_type == RiskThresholdType.FIXED:
            if position.direction == TradeDirection.LONG:
                return position.entry_price - threshold.value
            else:
                return position.entry_price + threshold.value
        
        return position.entry_price
    
    def _calculate_take_profit_price(self, position: Position, risk_controls: RiskControls,
                                   market_data: Dict[str, Any]) -> float:
        """Calculate take profit price for a position.
        
        Args:
            position: Position to calculate for
            risk_controls: Risk controls to use
            market_data: Current market data
            
        Returns:
            Take profit price
        """
        threshold = risk_controls.take_profit
        
        if threshold.threshold_type == RiskThresholdType.PERCENTAGE:
            if position.direction == TradeDirection.LONG:
                return position.entry_price * (1 + threshold.value)
            else:
                return position.entry_price * (1 - threshold.value)
        
        elif threshold.threshold_type == RiskThresholdType.ATR_BASED:
            atr = self._calculate_atr(position.symbol, threshold.atr_period, market_data)
            atr_distance = atr * threshold.atr_multiplier
            
            if position.direction == TradeDirection.LONG:
                return position.entry_price + atr_distance
            else:
                return position.entry_price - atr_distance
        
        elif threshold.threshold_type == RiskThresholdType.FIXED:
            if position.direction == TradeDirection.LONG:
                return position.entry_price + threshold.value
            else:
                return position.entry_price - threshold.value
        
        return position.entry_price
    
    def _should_exit_position(self, position: Position, current_price: float, 
                            threshold_price: float, exit_type: str) -> bool:
        """Check if position should be exited based on price.
        
        Args:
            position: Position to check
            current_price: Current market price
            threshold_price: Threshold price for exit
            exit_type: Type of exit ('stop_loss' or 'take_profit')
            
        Returns:
            True if position should be exited
        """
        if position.direction == TradeDirection.LONG:
            if exit_type == "stop_loss":
                return current_price <= threshold_price
            else:  # take_profit
                return current_price >= threshold_price
        else:  # SHORT
            if exit_type == "stop_loss":
                return current_price >= threshold_price
            else:  # take_profit
                return current_price <= threshold_price
    
    async def _exit_position(self, position: Position, exit_price: float, 
                           exit_reason: ExitReason, message: str, 
                           market_data: Dict[str, Any]) -> None:
        """Exit a position with detailed logging.
        
        Args:
            position: Position to exit
            exit_price: Exit price
            exit_reason: Reason for exit
            message: Exit message
            market_data: Current market data
        """
        try:
            # Calculate PnL
            if position.direction == TradeDirection.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            
            # Apply fees and slippage
            fees = self._calculate_fees(position, exit_price)
            pnl -= fees
            
            # Calculate holding period
            holding_period = datetime.utcnow() - position.entry_time
            
            # Calculate risk metrics
            risk_metrics = self._calculate_position_risk_metrics(position, exit_price, market_data)
            
            # Close position through portfolio manager
            self.portfolio_manager.close_position(position, exit_price)
            
            # Update daily PnL
            self.daily_pnl += pnl
            
            # Create exit event
            exit_event = ExitEvent(
                timestamp=datetime.utcnow(),
                symbol=position.symbol,
                position_id=str(id(position)),
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                holding_period=holding_period,
                risk_metrics=risk_metrics,
                market_conditions=self._get_market_conditions(position.symbol, market_data)
            )
            
            # Log exit event
            self._log_exit_event(exit_event)
            
            # Log to memory
            self.memory.log_decision(
                agent_name=self.config.name,
                decision_type='position_exit',
                details={
                    'symbol': position.symbol,
                    'exit_reason': exit_reason.value,
                    'pnl': pnl,
                    'message': message,
                    'exit_event': exit_event.to_dict()
                }
            )
            
            self.logger.info(f"Exited position {position.symbol}: {message} "
                           f"(PnL: ${pnl:.2f}, Reason: {exit_reason.value})")
            
        except Exception as e:
            self.logger.error(f"Error exiting position {position.symbol}: {e}")
    
    async def _emergency_exit_all_positions(self, exit_reason: ExitReason, 
                                          message: str, market_data: Dict[str, Any]) -> None:
        """Emergency exit all positions.
        
        Args:
            exit_reason: Reason for emergency exit
            message: Exit message
            market_data: Current market data
        """
        self.logger.warning(f"Emergency exit all positions: {message}")
        
        # Exit all open positions
        for position in self.portfolio_manager.state.open_positions[:]:
            symbol = position.symbol
            current_price = market_data.get(symbol, {}).get('price', position.entry_price)
            
            await self._exit_position(
                position,
                current_price,
                exit_reason,
                f"Emergency exit: {message}",
                market_data
            )
    
    def _calculate_atr(self, symbol: str, period: int, market_data: Dict[str, Any]) -> float:
        """Calculate Average True Range for a symbol.
        
        Args:
            symbol: Symbol to calculate ATR for
            period: ATR period
            market_data: Market data
            
        Returns:
            ATR value
        """
        try:
            if symbol not in self.market_data_cache:
                return 0.0
            
            # Get price history
            prices = self.market_data_cache[symbol]
            if len(prices) < period + 1:
                return 0.0
            
            # Calculate True Range
            true_ranges = []
            for i in range(1, len(prices)):
                high = prices[i]  # Simplified - assume current price is high
                low = prices[i-1]  # Simplified - assume previous price is low
                prev_close = prices[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR
            if len(true_ranges) >= period:
                return np.mean(true_ranges[-period:])
            else:
                return np.mean(true_ranges) if true_ranges else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0
    
    def _calculate_portfolio_correlation(self, market_data: Dict[str, Any]) -> float:
        """Calculate portfolio correlation.
        
        Args:
            market_data: Current market data
            
        Returns:
            Portfolio correlation
        """
        try:
            positions = self.portfolio_manager.state.open_positions
            if len(positions) < 2:
                return 0.0
            
            # Get price data for all positions
            price_data = {}
            for position in positions:
                symbol = position.symbol
                if symbol in self.market_data_cache and self.market_data_cache[symbol]:
                    price_data[symbol] = list(self.market_data_cache[symbol])
            
            if len(price_data) < 2:
                return 0.0
            
            # Calculate correlations
            correlations = []
            symbols = list(price_data.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    min_len = min(len(price_data[sym1]), len(price_data[sym2]))
                    
                    if min_len > 10:
                        corr = np.corrcoef(
                            price_data[sym1][-min_len:],
                            price_data[sym2][-min_len:]
                        )[0, 1]
                        
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation: {e}")
            return 0.0
    
    def _calculate_portfolio_risk_exposure(self, market_data: Dict[str, Any]) -> float:
        """Calculate portfolio risk exposure.
        
        Args:
            market_data: Current market data
            
        Returns:
            Portfolio risk exposure as percentage
        """
        try:
            total_exposure = 0.0
            portfolio_value = self.portfolio_manager.state.equity
            
            for position in self.portfolio_manager.state.open_positions:
                symbol = position.symbol
                current_price = market_data.get(symbol, {}).get('price', position.entry_price)
                position_value = position.size * current_price
                total_exposure += position_value
            
            return total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk exposure: {e}")
            return 0.0
    
    def _calculate_position_risk_metrics(self, position: Position, current_price: float,
                                       market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for a position.
        
        Args:
            position: Position to calculate metrics for
            current_price: Current price
            market_data: Market data
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            # Calculate basic metrics
            if position.direction == TradeDirection.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.size
                price_change = (current_price - position.entry_price) / position.entry_price
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size
                price_change = (position.entry_price - current_price) / position.entry_price
            
            # Calculate volatility
            volatility = market_data.get(position.symbol, {}).get('volatility', 0.0)
            
            # Calculate VaR (simplified)
            var_95 = -price_change * 1.645  # 95% VaR
            
            return {
                'unrealized_pnl': unrealized_pnl,
                'price_change': price_change,
                'volatility': volatility,
                'var_95': var_95,
                'position_size': position.size,
                'position_value': position.size * current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics for {position.symbol}: {e}")
            return {}
    
    def _calculate_portfolio_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary of portfolio risk metrics
        """
        try:
            total_pnl = 0.0
            total_exposure = 0.0
            position_metrics = []
            
            for position in self.portfolio_manager.state.open_positions:
                symbol = position.symbol
                current_price = market_data.get(symbol, {}).get('price', position.entry_price)
                
                metrics = self._calculate_position_risk_metrics(position, current_price, market_data)
                position_metrics.append(metrics)
                
                total_pnl += metrics.get('unrealized_pnl', 0.0)
                total_exposure += metrics.get('position_value', 0.0)
            
            portfolio_value = self.portfolio_manager.state.equity
            
            return {
                'total_unrealized_pnl': total_pnl,
                'total_exposure': total_exposure,
                'exposure_ratio': total_exposure / portfolio_value if portfolio_value > 0 else 0.0,
                'daily_pnl': self.daily_pnl,
                'position_count': len(self.portfolio_manager.state.open_positions),
                'portfolio_correlation': self._calculate_portfolio_correlation(market_data),
                'portfolio_risk': self._calculate_portfolio_risk_exposure(market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    def _get_market_conditions(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get market conditions for a symbol.
        
        Args:
            symbol: Symbol to get conditions for
            market_data: Market data
            
        Returns:
            Dictionary of market conditions
        """
        try:
            symbol_data = market_data.get(symbol, {})
            
            return {
                'price': symbol_data.get('price', 0.0),
                'volume': symbol_data.get('volume', 0.0),
                'volatility': symbol_data.get('volatility', 0.0),
                'price_change': symbol_data.get('price_change', 0.0),
                'market_regime': self.global_metrics.get('market_regime', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market conditions for {symbol}: {e}")
            return {}
    
    def _calculate_fees_for_exit(self, position: Position, exit_price: float) -> float:
        """Calculate fees for position exit.
        
        Args:
            position: Position being exited
            exit_price: Exit price
            
        Returns:
            Fee amount
        """
        # Get fee settings
        base_fee = self.config.custom_config.get('base_fee', 0.001)  # 10 bps
        min_fee = self.config.custom_config.get('min_fee', 1.0)
        
        # Calculate fees
        fee = max(min_fee, exit_price * position.size * base_fee)
        
        return fee
    
    def _log_exit_event(self, exit_event: ExitEvent) -> None:
        """Log exit event to file.
        
        Args:
            exit_event: Exit event to log
        """
        try:
            # Add to exit events list
            self.exit_events.append(exit_event)
            
            # Keep only recent events (last 1000)
            if len(self.exit_events) > 1000:
                self.exit_events = self.exit_events[-1000:]
            
            # Save to JSON file
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'exit_event': exit_event.to_dict()
            }
            
            with open(self.exit_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error logging exit event: {e}")
    
    def get_exit_events(self, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get exit events.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of exit events
        """
        if not self.exit_log_path.exists():
            return []
        
        events = []
        try:
            with open(self.exit_log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        event_timestamp = datetime.fromisoformat(entry['timestamp'])
                        
                        # Apply date filters
                        if start_date and event_timestamp < start_date:
                            continue
                        if end_date and event_timestamp > end_date:
                            continue
                        
                        events.append(entry)
        except Exception as e:
            self.logger.error(f"Error reading exit events: {e}")
        
        return events
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary.
        
        Returns:
            Risk management summary
        """
        try:
            # Calculate exit statistics
            exit_reasons = {}
            total_exits = len(self.exit_events)
            
            for event in self.exit_events:
                reason = event.exit_reason.value
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Calculate PnL by exit reason
            pnl_by_reason = {}
            total_pnl = 0.0
            
            for event in self.exit_events:
                reason = event.exit_reason.value
                pnl = event.pnl
                
                if reason not in pnl_by_reason:
                    pnl_by_reason[reason] = {'total': 0.0, 'count': 0, 'avg': 0.0}
                
                pnl_by_reason[reason]['total'] += pnl
                pnl_by_reason[reason]['count'] += 1
                total_pnl += pnl
            
            # Calculate averages
            for reason in pnl_by_reason:
                count = pnl_by_reason[reason]['count']
                if count > 0:
                    pnl_by_reason[reason]['avg'] = pnl_by_reason[reason]['total'] / count
            
            return {
                'total_exits': total_exits,
                'exit_reasons': exit_reasons,
                'pnl_by_reason': pnl_by_reason,
                'total_pnl': total_pnl,
                'daily_pnl': self.daily_pnl,
                'risk_controls': self.default_risk_controls.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}")
            return {}
    
    def _update_market_data_cache(self, market_data: Dict[str, Any]) -> None:
        """Update market data cache and price history.
        
        Args:
            market_data: Current market data
        """
        try:
            # Update market data cache
            for symbol, data in market_data.items():
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {}
                
                self.market_data_cache[symbol].update(data)
                
                # Update price history for ATR calculations
                if 'price' in data:
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    self.price_history[symbol].append(data['price'])
                    
                    # Keep only recent prices (last 100)
                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Update global metrics
            self._update_global_metrics(market_data)
            
        except Exception as e:
            self.logger.error(f"Error updating market data cache: {e}")
    
    def _update_global_metrics(self, market_data: Dict[str, Any]) -> None:
        """Update global market metrics.
        
        Args:
            market_data: Current market data
        """
        try:
            # Calculate average volatility
            volatilities = []
            for symbol_data in market_data.values():
                if 'volatility' in symbol_data:
                    volatilities.append(symbol_data['volatility'])
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                
                # Determine volatility regime
                if avg_volatility > 0.3:
                    self.global_metrics['volatility_regime'] = 'high'
                elif avg_volatility > 0.15:
                    self.global_metrics['volatility_regime'] = 'normal'
                else:
                    self.global_metrics['volatility_regime'] = 'low'
            
            # Determine market regime based on price movements
            price_changes = []
            for symbol_data in market_data.values():
                if 'price_change' in symbol_data:
                    price_changes.append(symbol_data['price_change'])
            
            if price_changes:
                avg_change = np.mean(price_changes)
                
                if abs(avg_change) > 0.02:
                    self.global_metrics['market_regime'] = 'trending'
                elif abs(avg_change) > 0.005:
                    self.global_metrics['market_regime'] = 'volatile'
                else:
                    self.global_metrics['market_regime'] = 'sideways'
            
        except Exception as e:
            self.logger.error(f"Error updating global metrics: {e}")
    
    async def _process_trade_signal(self, signal: TradeSignal, 
                                    market_data: Dict[str, Any]) -> ExecutionResult:
        """Process a single trade signal.
        
        Args:
            signal: Trade signal to process
            market_data: Current market data
            
        Returns:
            ExecutionResult: Result of the signal processing
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return ExecutionResult(
                    success=False,
                    signal=signal,
                    message="Signal validation failed",
                    error="Invalid signal parameters"
                )
            
            # Check position limits
            if not self._check_position_limits(signal):
                return ExecutionResult(
                    success=False,
                    signal=signal,
                    message="Position limit exceeded",
                    error="Too many positions or insufficient capital"
                )
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(signal, market_data)
            
            # Check slippage
            slippage = abs(execution_price - signal.entry_price) / signal.entry_price
            if slippage > self.max_slippage:
                return ExecutionResult(
                    success=False,
                    signal=signal,
                    message="Slippage too high",
                    error=f"Slippage {slippage:.4f} exceeds limit {self.max_slippage:.4f}"
                )
            
            # Execute trade
            if self.execution_mode == ExecutionMode.SIMULATION:
                position = self._execute_simulation_trade_sync(signal, execution_price)
            else:
                position = self._execute_real_trade_sync(signal, execution_price)
            
            # Calculate fees
            fees = self._calculate_fees(signal, execution_price)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_position_risk_metrics(
                position, execution_price, market_data
            ) if position else None
            
            return ExecutionResult(
                success=True,
                signal=signal,
                position=position,
                execution_price=execution_price,
                slippage=slippage,
                fees=fees,
                message="Trade executed successfully",
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error processing trade signal: {e}")
            return ExecutionResult(
                success=False,
                signal=signal,
                message="Signal processing failed",
                error=str(e)
            )
    
    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trade signal parameters.
        
        Args:
            signal: Trade signal to validate
            
        Returns:
            True if signal is valid
        """
        try:
            # Check required fields
            if not signal.symbol or not signal.strategy:
                return False
            
            # Check confidence level
            if signal.confidence < 0.0 or signal.confidence > 1.0:
                return False
            
            # Check entry price
            if signal.entry_price <= 0:
                return False
            
            # Check minimum confidence
            if signal.confidence < self.min_confidence:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def _check_position_limits(self, signal: TradeSignal) -> bool:
        """Check position limits before execution.
        
        Args:
            signal: Trade signal to check
            
        Returns:
            True if within limits
        """
        try:
            # Check maximum positions
            if len(self.portfolio_manager.state.open_positions) >= self.max_positions:
                return False
            
            # Check if already have position in this symbol
            for position in self.portfolio_manager.state.open_positions:
                if position.symbol == signal.symbol:
                    return False
            
            # Check capital availability
            portfolio_value = self.portfolio_manager.state.equity
            if portfolio_value <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False
    
    def _calculate_execution_price(self, signal: TradeSignal, 
                                 market_data: Dict[str, Any]) -> float:
        """Calculate execution price with slippage simulation.
        
        Args:
            signal: Trade signal
            market_data: Current market data
            
        Returns:
            Execution price
        """
        try:
            # Get current market price
            current_price = market_data.get(signal.symbol, {}).get('price', signal.entry_price)
            
            # Simulate slippage based on order size and market conditions
            volatility = market_data.get(signal.symbol, {}).get('volatility', 0.15)
            volume = market_data.get(signal.symbol, {}).get('volume', 1000000)
            
            # Calculate slippage factor
            slippage_factor = min(0.001, volatility * 0.01)  # Max 10 bps
            
            # Apply slippage
            if signal.direction == TradeDirection.LONG:
                execution_price = current_price * (1 + slippage_factor)
            else:
                execution_price = current_price * (1 - slippage_factor)
            
            return execution_price
            
        except Exception as e:
            self.logger.error(f"Error calculating execution price: {e}")
            return signal.entry_price
    
    async def _execute_simulation_trade(self, signal: TradeSignal, 
                                      execution_price: float) -> Position:
        """Execute trade in simulation mode.
        
        Args:
            signal: Trade signal
            execution_price: Execution price
            
        Returns:
            Created position
        """
        try:
            # Calculate position size
            position_size, sizing_details = self._calculate_position_size(signal, execution_price)
            
            # Create position through portfolio manager
            position = self.portfolio_manager.open_position(
                symbol=signal.symbol,
                direction=signal.direction,
                size=position_size,
                entry_price=execution_price,
                strategy=signal.strategy,
                max_holding_period=signal.max_holding_period
            )
            
            # Add risk controls to position if provided
            if signal.risk_controls:
                position.risk_controls = signal.risk_controls
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing simulation trade: {e}")
            raise
    
    async def _execute_real_trade(self, signal: TradeSignal, 
                                execution_price: float) -> Position:
        """Execute trade through real broker (placeholder).
        
        Args:
            signal: Trade signal
            execution_price: Execution price
            
        Returns:
            Created position
        """
        # Placeholder for real execution
        self.logger.info(f"Real execution not implemented yet for {signal.symbol}")
        return await self._execute_simulation_trade(signal, execution_price)
    
    def _calculate_position_size(
        self, 
        signal: TradeSignal, 
        execution_price: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate position size using the PositionSizer.
        
        Args:
            signal: Trade signal
            execution_price: Execution price
            market_data: Current market data
            
        Returns:
            Tuple of (position_size, sizing_details)
        """
        try:
            # Use signal size if provided
            if signal.size:
                return signal.size, {'strategy': 'manual_size', 'size': signal.size}
            
            # Get stop loss price for risk calculation
            stop_loss_price = self._get_stop_loss_price(signal, market_data or {})
            
            # Create market context
            market_context = self._create_market_context(signal, market_data or {})
            
            # Create signal context
            signal_context = self._create_signal_context(signal)
            
            # Create portfolio context
            portfolio_context = self._create_portfolio_context()
            
            # Get sizing parameters from signal or use defaults
            sizing_params = self._get_sizing_parameters(signal)
            
            # Calculate position size using PositionSizer
            position_size, sizing_details = self.position_sizer.calculate_position_size(
                entry_price=execution_price,
                stop_loss_price=stop_loss_price,
                market_context=market_context,
                signal_context=signal_context,
                portfolio_context=portfolio_context,
                sizing_params=sizing_params
            )
            
            # Convert percentage to actual shares
            portfolio_value = self.portfolio_manager.state.equity
            actual_shares = (position_size * portfolio_value) / execution_price
            
            # Add additional details
            sizing_details.update({
                'execution_price': execution_price,
                'stop_loss_price': stop_loss_price,
                'portfolio_value': portfolio_value,
                'actual_shares': actual_shares,
                'position_value': actual_shares * execution_price
            })
            
            return actual_shares, sizing_details
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Return conservative fallback
            portfolio_value = self.portfolio_manager.state.equity
            conservative_size = min(0.01, portfolio_value * 0.01 / execution_price)
            return conservative_size, {'strategy': 'conservative_fallback', 'error': str(e), 'size': conservative_size}
    
    def _get_stop_loss_price(self, signal: TradeSignal, market_data: Dict[str, Any]) -> float:
        """Get stop loss price for position sizing.
        
        Args:
            signal: Trade signal
            market_data: Market data
            
        Returns:
            Stop loss price
        """
        # Use signal stop loss if provided
        if signal.stop_loss:
            return signal.stop_loss
        
        # Use risk controls if available
        if signal.risk_controls:
            # Create a mock position for stop loss calculation
            mock_position = type('Position', (), {
                'symbol': signal.symbol,
                'direction': signal.direction,
                'entry_price': signal.entry_price
            })()
            
            return self._calculate_stop_loss_price(
                mock_position, signal.risk_controls, market_data
            )
        
        # Default to 2% below entry price
        if signal.direction == TradeDirection.LONG:
            return signal.entry_price * 0.98
        else:
            return signal.entry_price * 1.02
    
    def _create_market_context(self, signal: TradeSignal, market_data: Dict[str, Any]) -> MarketContext:
        """Create market context for position sizing.
        
        Args:
            signal: Trade signal
            market_data: Market data
            
        Returns:
            Market context
        """
        symbol_data = market_data.get(signal.symbol, {})
        
        # Get price history for correlation calculation
        price_history = None
        if signal.symbol in self.price_history:
            price_history = pd.Series(self.price_history[signal.symbol])
        
        return MarketContext(
            symbol=signal.symbol,
            current_price=symbol_data.get('price', signal.entry_price),
            volatility=symbol_data.get('volatility', 0.15),
            volume=symbol_data.get('volume', 1000000),
            market_regime=self.global_metrics.get('market_regime', 'normal'),
            correlation=self._calculate_symbol_correlation(signal.symbol, market_data),
            liquidity_score=symbol_data.get('liquidity_score', 1.0),
            bid_ask_spread=symbol_data.get('bid_ask_spread', 0.001),
            price_history=price_history
        )
    
    def _create_signal_context(self, signal: TradeSignal) -> SignalContext:
        """Create signal context for position sizing.
        
        Args:
            signal: Trade signal
            
        Returns:
            Signal context
        """
        # Get strategy performance from memory or use defaults
        strategy_performance = self._get_strategy_performance(signal.strategy)
        
        return SignalContext(
            confidence=signal.confidence,
            forecast_certainty=signal.market_data.get('forecast_certainty', 0.5) if signal.market_data else 0.5,
            strategy_performance=strategy_performance.get('performance', 0.0),
            win_rate=strategy_performance.get('win_rate', 0.5),
            avg_win=strategy_performance.get('avg_win', 0.02),
            avg_loss=strategy_performance.get('avg_loss', -0.01),
            sharpe_ratio=strategy_performance.get('sharpe_ratio', 0.0),
            max_drawdown=strategy_performance.get('max_drawdown', 0.1),
            signal_strength=signal.market_data.get('signal_strength', 0.5) if signal.market_data else 0.5
        )
    
    def _create_portfolio_context(self) -> PortfolioContext:
        """Create portfolio context for position sizing.
        
        Returns:
            Portfolio context
        """
        portfolio_state = self.portfolio_manager.get_status()
        
        return PortfolioContext(
            total_capital=portfolio_state['equity'],
            available_capital=portfolio_state['cash'],
            current_exposure=portfolio_state['total_exposure'],
            open_positions=len(portfolio_state['open_positions']),
            daily_pnl=self.daily_pnl,
            portfolio_volatility=self._calculate_portfolio_volatility(),
            correlation_matrix=self._get_correlation_matrix()
        )
    
    def _get_sizing_parameters(self, signal: TradeSignal) -> Optional[SizingParameters]:
        """Get sizing parameters for the signal.
        
        Args:
            signal: Trade signal
            
        Returns:
            Sizing parameters or None to use defaults
        """
        # Check if signal has custom sizing parameters
        if hasattr(signal, 'sizing_params') and signal.sizing_params:
            return signal.sizing_params
        
        # Check if signal has sizing strategy in market data
        if signal.market_data and 'sizing_strategy' in signal.market_data:
            strategy_name = signal.market_data['sizing_strategy']
            try:
                strategy = SizingStrategy(strategy_name)
                return SizingParameters(
                    strategy=strategy,
                    risk_per_trade=signal.market_data.get('risk_per_trade', 0.02),
                    max_position_size=signal.market_data.get('max_position_size', 0.2),
                    confidence_multiplier=signal.market_data.get('confidence_multiplier', 1.0)
                )
            except ValueError:
                self.logger.warning(f"Invalid sizing strategy: {strategy_name}")
        
        # Use default parameters
        return None
    
    def _get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """Get strategy performance metrics from memory.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Strategy performance metrics
        """
        try:
            # Get strategy performance from memory
            strategy_data = self.memory.get_strategy_performance(strategy_name)
            if strategy_data:
                return strategy_data
            
            # Return default metrics if not found
            return {
                'performance': 0.0,
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': -0.01,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {
                'performance': 0.0,
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': -0.01,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.1
            }
    
    def _calculate_symbol_correlation(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate correlation between symbol and existing positions.
        
        Args:
            symbol: Symbol to calculate correlation for
            market_data: Market data
            
        Returns:
            Correlation value
        """
        try:
            # Get existing positions
            positions = self.portfolio_manager.state.open_positions
            if not positions:
                return 0.0
            
            # Calculate correlation with existing positions
            correlations = []
            for position in positions:
                if position.symbol in market_data and symbol in market_data:
                    # Simple correlation calculation
                    # In practice, this would use historical price data
                    correlations.append(0.3)  # Default correlation
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating symbol correlation: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility.
        
        Returns:
            Portfolio volatility
        """
        try:
            # Get portfolio returns from memory or calculate
            portfolio_returns = self.memory.get_portfolio_returns()
            if portfolio_returns and len(portfolio_returns) > 10:
                return np.std(portfolio_returns)
            
            # Default volatility
            return 0.15
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15
    
    def _get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix for portfolio positions.
        
        Returns:
            Correlation matrix or None
        """
        try:
            # Get position symbols
            positions = self.portfolio_manager.state.open_positions
            if len(positions) < 2:
                return None
            
            symbols = [pos.symbol for pos in positions]
            
            # Create correlation matrix from price history
            price_data = {}
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) > 10:
                    price_data[symbol] = self.price_history[symbol]
            
            if len(price_data) < 2:
                return None
            
            # Calculate correlation matrix
            df = pd.DataFrame(price_data)
            return df.corr()
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {e}")
            return None
    
    def _calculate_fees(self, signal: TradeSignal, execution_price: float) -> float:
        """Calculate trading fees.
        
        Args:
            signal: Trade signal
            execution_price: Execution price
            
        Returns:
            Fee amount
        """
        try:
            # Get fee settings
            base_fee = self.config.custom_config.get('base_fee', 0.001)  # 10 bps
            min_fee = self.config.custom_config.get('min_fee', 1.0)
            
            # Calculate position size (returns tuple of size and details)
            position_size, sizing_details = self._calculate_position_size(signal, execution_price)
            
            # Calculate fees
            fee = max(min_fee, execution_price * position_size * base_fee)
            
            return fee
            
        except Exception as e:
            self.logger.error(f"Error calculating fees: {e}")
            return 0.0
    
    def _update_portfolio(self, market_data: Dict[str, Any]) -> None:
        """Update portfolio with current market data.
        
        Args:
            market_data: Current market data
        """
        try:
            # Update portfolio manager with current prices
            for symbol, data in market_data.items():
                if 'price' in data:
                    self.portfolio_manager.update_position_price(symbol, data['price'])
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def _log_execution_result(self, result: ExecutionResult) -> None:
        """Log execution result to file.
        
        Args:
            result: Execution result to log
        """
        try:
            # Add to execution history
            self.execution_history.append(result)
            
            # Keep only recent history (last 1000)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            # Save to JSON file
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'execution_result': result.to_dict()
            }
            
            with open(self.trade_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error logging execution result: {e}")
    
    def _generate_execution_summary(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Generate execution summary.
        
        Args:
            results: List of execution results
            
        Returns:
            Summary dictionary
        """
        try:
            if not results:
                return {
                    'success_rate': 0.0,
                    'total_signals': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_slippage': 0.0,
                    'total_fees': 0.0
                }
            
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            total_slippage = sum(r.slippage for r in successful)
            total_fees = sum(r.fees for r in successful)
            
            return {
                'success_rate': len(successful) / len(results),
                'total_signals': len(results),
                'successful_executions': len(successful),
                'failed_executions': len(failed),
                'total_slippage': total_slippage,
                'total_fees': total_fees,
                'average_confidence': np.mean([r.signal.confidence for r in results]) if results else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating execution summary: {e}")
            return {}
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status.
        
        Returns:
            Portfolio status dictionary
        """
        return self.portfolio_manager.get_status()
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of execution history entries
        """
        try:
            history = []
            for result in self.execution_history[-limit:]:
                history.append(result.to_dict())
            return history
        except Exception as e:
            self.logger.error(f"Error getting execution history: {e}")
            return []
    
    def get_trade_log(self, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get trade log entries.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of trade log entries
        """
        if not self.trade_log_path.exists():
            return []
        
        entries = []
        try:
            with open(self.trade_log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry_timestamp = datetime.fromisoformat(entry['timestamp'])
                        
                        # Apply date filters
                        if start_date and entry_timestamp < start_date:
                            continue
                        if end_date and entry_timestamp > end_date:
                            continue
                        
                        entries.append(entry)
        except Exception as e:
            self.logger.error(f"Error reading trade log: {e}")
        
        return entries
    
    def clear_trade_log(self) -> None:
        """Clear the trade log file."""
        try:
            if self.trade_log_path.exists():
                self.trade_log_path.unlink()
            self.execution_history.clear()
        except Exception as e:
            self.logger.error(f"Error clearing trade log: {e}")

# Factory function for easy creation
def create_execution_agent(config: Optional[Dict[str, Any]] = None) -> ExecutionAgent:
    """Create an execution agent with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        ExecutionAgent instance
    """
    default_config = {
        'name': 'execution_agent',
        'enabled': True,
        'priority': 1,
        'max_concurrent_runs': 1,
        'timeout_seconds': 300,
        'retry_attempts': 3,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 10,
            'min_confidence': 0.7,
            'max_slippage': 0.001,
            'execution_delay': 1.0,
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'base_fee': 0.001,
            'min_fee': 1.0
        }
    }
    
    if config:
        default_config.update(config)
    
    agent_config = AgentConfig(**default_config)
    return ExecutionAgent(agent_config)