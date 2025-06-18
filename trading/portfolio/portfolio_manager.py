import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from redis.exceptions import RedisError

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.optimization.performance_logger import PerformanceLogger, PerformanceMetrics
from trading.portfolio.llm_utils import LLMInterface, TradeRationale, DailyCommentary

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/portfolio/logs/portfolio_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

class PositionStatus(Enum):
    """Position status enum."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

class TradeDirection(Enum):
    """Trade direction enum."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Position:
    """Position data class."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    size: float
    strategy: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    max_holding_period: Optional[timedelta] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    rationale: Optional[TradeRationale] = None
    unrealized_pnl: Optional[float] = None
    risk_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        pos_dict = asdict(self)
        if self.rationale:
            pos_dict['rationale'] = self.rationale.to_dict()
        return pos_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        # Convert string enums back to enum values
        data['direction'] = TradeDirection(data['direction'])
        data['status'] = PositionStatus(data['status'])
        
        # Convert string timestamps to datetime
        if isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        if isinstance(data['max_holding_period'], str):
            data['max_holding_period'] = timedelta(seconds=float(data['max_holding_period']))
        
        # Convert rationale
        if 'rationale' in data:
            data['rationale'] = TradeRationale.from_dict(data['rationale'])
            
        return cls(**data)

@dataclass
class PortfolioState:
    """Portfolio state data class."""
    timestamp: datetime
    cash: float
    equity: float
    leverage: float
    available_capital: float
    total_pnl: float
    unrealized_pnl: float
    open_positions: List[Position]
    closed_positions: List[Position]
    metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_regime: str
    strategy_weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        state_dict = asdict(self)
        state_dict['timestamp'] = self.timestamp.isoformat()
        state_dict['open_positions'] = [p.to_dict() for p in self.open_positions]
        state_dict['closed_positions'] = [p.to_dict() for p in self.closed_positions]
        return state_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioState':
        """Create state from dictionary."""
        # Convert string timestamp to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
        # Convert position dictionaries to Position objects
        data['open_positions'] = [Position.from_dict(p) for p in data['open_positions']]
        data['closed_positions'] = [Position.from_dict(p) for p in data['closed_positions']]
        
        return cls(**data)

class PortfolioManager:
    """Portfolio manager with full agentic support and interactive tracking."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize portfolio manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize Redis connection if available
        self.redis = None
        self.pubsub = None
        try:
            self.redis = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0)
            )
            self.redis.ping()  # Test connection
            self.pubsub = self.redis.pubsub()
            self.pubsub.subscribe('portfolio_updates')
            logger.info("Connected to Redis")
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Redis not available: {e}. Using in-memory storage.")
            self.redis = None
        
        # Initialize components
        self.strategy_agent = StrategySelectionAgent()
        self.performance_logger = PerformanceLogger()
        self.llm_interface = LLMInterface(self.config.get('llm_config'))
        
        # Initialize state
        self.state = PortfolioState(
            timestamp=datetime.utcnow(),
            cash=self.config.get('initial_cash', 100000.0),
            equity=self.config.get('initial_cash', 100000.0),
            leverage=self.config.get('max_leverage', 1.0),
            available_capital=self.config.get('initial_cash', 100000.0),
            total_pnl=0.0,
            unrealized_pnl=0.0,
            open_positions=[],
            closed_positions=[],
            metrics={},
            risk_metrics={},
            market_regime='neutral',
            strategy_weights={}
        )
        
        # Create necessary directories
        os.makedirs('trading/portfolio/logs', exist_ok=True)
        os.makedirs('trading/portfolio/data', exist_ok=True)
        
        logger.info(f"Initialized PortfolioManager with config: {self.config}")
    
    def open_position(self, symbol: str, direction: TradeDirection, price: float,
                     size: float, strategy: str, market_data: Dict[str, Any],
                     take_profit: Optional[float] = None, stop_loss: Optional[float] = None,
                     max_holding_period: Optional[timedelta] = None) -> Position:
        """Open a new position.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            price: Entry price
            size: Position size
            strategy: Strategy name
            market_data: Market data and context
            take_profit: Optional take profit level
            stop_loss: Optional stop loss level
            max_holding_period: Optional maximum holding period
            
        Returns:
            New Position object
        """
        # Calculate position size based on risk
        size = self._calculate_position_size(symbol, price, strategy, market_data)
        
        # Validate position size
        if size * price > self.state.available_capital:
            raise ValueError("Position size exceeds available capital")
        
        # Generate trade rationale
        rationale = self.llm_interface.generate_trade_rationale(
            symbol=symbol,
            direction=direction.value,
            strategy=strategy,
            market_data=market_data
        )
        
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            entry_time=datetime.utcnow(),
            size=size,
            strategy=strategy,
            take_profit=take_profit,
            stop_loss=stop_loss,
            max_holding_period=max_holding_period,
            rationale=rationale
        )
        
        # Update state
        self.state.open_positions.append(position)
        self.state.cash -= size * price
        self.state.available_capital -= size * price
        
        # Update strategy weights
        self._update_strategy_weights()
        
        # Log position
        self._log_position(position, "open")
        
        # Publish update
        self._publish_update("position_opened", position.to_dict())
        
        return position
    
    def close_position(self, position: Position, price: float) -> None:
        """Close an existing position.
        
        Args:
            position: Position to close
            price: Exit price
        """
        # Calculate PnL with slippage and fees
        if position.direction == TradeDirection.LONG:
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * position.size
        
        # Apply slippage and fees
        slippage = self._calculate_slippage(position, price)
        fees = self._calculate_fees(position, price)
        pnl -= slippage + fees
        
        # Update position
        position.exit_price = price
        position.exit_time = datetime.utcnow()
        position.pnl = pnl
        position.status = PositionStatus.CLOSED
        
        # Update state
        self.state.open_positions.remove(position)
        self.state.closed_positions.append(position)
        self.state.cash += position.size * price
        self.state.available_capital += position.size * price
        self.state.total_pnl += pnl
        
        # Update strategy weights
        self._update_strategy_weights()
        
        # Log position
        self._log_position(position, "close")
        
        # Publish update
        self._publish_update("position_closed", position.to_dict())
        
        # Update metrics
        self._update_metrics()
    
    def update_positions(self, prices: Dict[str, float], market_data: Dict[str, Any]) -> None:
        """Update all open positions with current prices.
        
        Args:
            prices: Dictionary of current prices
            market_data: Market data and context
        """
        # Update market regime
        self.state.market_regime = self.strategy_agent.get_market_regime(market_data)
        
        # Update positions
        total_unrealized_pnl = 0.0
        for position in self.state.open_positions[:]:  # Copy list to allow modification
            if position.symbol not in prices:
                continue
                
            price = prices[position.symbol]
            
            # Calculate unrealized PnL
            if position.direction == TradeDirection.LONG:
                unrealized_pnl = (price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - price) * position.size
            
            # Update position
            position.unrealized_pnl = unrealized_pnl
            total_unrealized_pnl += unrealized_pnl
            
            # Update risk metrics
            position.risk_metrics = self._calculate_position_risk(position, price, market_data)
            
            # Check take profit
            if position.take_profit is not None:
                if (position.direction == TradeDirection.LONG and price >= position.take_profit) or \
                   (position.direction == TradeDirection.SHORT and price <= position.take_profit):
                    self.close_position(position, price)
                    continue
            
            # Check stop loss
            if position.stop_loss is not None:
                if (position.direction == TradeDirection.LONG and price <= position.stop_loss) or \
                   (position.direction == TradeDirection.SHORT and price >= position.stop_loss):
                    self.close_position(position, price)
                    continue
            
            # Check max holding period
            if position.max_holding_period is not None:
                if datetime.utcnow() - position.entry_time > position.max_holding_period:
                    self.close_position(position, price)
                    continue
        
        # Update state
        self.state.unrealized_pnl = total_unrealized_pnl
        self.state.equity = self.state.cash + total_unrealized_pnl
        
        # Update risk metrics
        self._update_risk_metrics(prices, market_data)
        
        # Update metrics
        self._update_metrics()
        
        # Generate daily commentary if needed
        self._generate_daily_commentary(market_data)
    
    def _calculate_position_size(self, symbol: str, price: float, strategy: str,
                               market_data: Dict[str, Any]) -> float:
        """Calculate position size based on risk and strategy confidence.
        
        Args:
            symbol: Trading symbol
            price: Current price
            strategy: Strategy name
            market_data: Market data and context
            
        Returns:
            Position size
        """
        # Get base size from risk settings
        risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2% per trade
        base_size = self.state.available_capital * risk_per_trade / price
        
        # Adjust for volatility
        volatility = market_data.get('volatility', {}).get(symbol, 0.2)
        volatility_factor = 1.0 / (1.0 + volatility)
        
        # Adjust for strategy confidence
        strategy_confidence = self.strategy_agent.get_strategy_confidence(strategy, market_data)
        confidence_factor = 0.5 + strategy_confidence  # 0.5-1.5 range
        
        # Adjust for market regime
        regime_factor = 1.0
        if self.state.market_regime == 'trending':
            regime_factor = 1.2
        elif self.state.market_regime == 'ranging':
            regime_factor = 0.8
        
        # Calculate final size
        size = base_size * volatility_factor * confidence_factor * regime_factor
        
        # Apply position limits
        max_position_size = self.config.get('max_position_size', 0.2)  # 20% of capital
        size = min(size, self.state.available_capital * max_position_size / price)
        
        return size
    
    def _calculate_slippage(self, position: Position, price: float) -> float:
        """Calculate slippage for a trade.
        
        Args:
            position: Position to calculate slippage for
            price: Current price
            
        Returns:
            Slippage amount
        """
        # Get slippage settings
        base_slippage = self.config.get('base_slippage', 0.0001)  # 1 bps
        size_factor = position.size / 1000  # Increase with size
        
        # Calculate slippage
        slippage = price * position.size * base_slippage * (1 + size_factor)
        
        return slippage
    
    def _calculate_fees(self, position: Position, price: float) -> float:
        """Calculate fees for a trade.
        
        Args:
            position: Position to calculate fees for
            price: Current price
            
        Returns:
            Fee amount
        """
        # Get fee settings
        base_fee = self.config.get('base_fee', 0.001)  # 10 bps
        min_fee = self.config.get('min_fee', 1.0)
        
        # Calculate fees
        fee = max(min_fee, price * position.size * base_fee)
        
        return fee
    
    def _calculate_position_risk(self, position: Position, price: float,
                               market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for a position.
        
        Args:
            position: Position to calculate risk for
            price: Current price
            market_data: Market data and context
            
        Returns:
            Dictionary of risk metrics
        """
        # Calculate VaR
        returns = market_data.get('returns', {}).get(position.symbol, [])
        if returns:
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
        else:
            var_95 = var_99 = 0.0
        
        # Calculate volatility
        volatility = market_data.get('volatility', {}).get(position.symbol, 0.0)
        
        # Calculate beta
        beta = market_data.get('beta', {}).get(position.symbol, 1.0)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'volatility': volatility,
            'beta': beta
        }
    
    def _update_risk_metrics(self, prices: Dict[str, float],
                           market_data: Dict[str, Any]) -> None:
        """Update portfolio risk metrics.
        
        Args:
            prices: Dictionary of current prices
            market_data: Market data and context
        """
        # Calculate portfolio VaR
        position_var = []
        for position in self.state.open_positions:
            if position.symbol in prices:
                var = position.risk_metrics.get('var_95', 0) * position.size * prices[position.symbol]
                position_var.append(var)
        
        portfolio_var = np.sqrt(np.sum(np.square(position_var)))
        
        # Calculate portfolio volatility
        position_vol = []
        for position in self.state.open_positions:
            if position.symbol in prices:
                vol = position.risk_metrics.get('volatility', 0) * position.size * prices[position.symbol]
                position_vol.append(vol)
        
        portfolio_vol = np.sqrt(np.sum(np.square(position_vol)))
        
        # Calculate portfolio beta
        position_beta = []
        for position in self.state.open_positions:
            if position.symbol in prices:
                beta = position.risk_metrics.get('beta', 1.0) * position.size * prices[position.symbol]
                position_beta.append(beta)
        
        portfolio_beta = np.average(position_beta) if position_beta else 1.0
        
        # Update state
        self.state.risk_metrics = {
            'portfolio_var': portfolio_var,
            'portfolio_volatility': portfolio_vol,
            'portfolio_beta': portfolio_beta
        }
    
    def _update_strategy_weights(self) -> None:
        """Update strategy weights based on performance."""
        # Calculate strategy PnL
        strategy_pnl = {}
        for position in self.state.closed_positions:
            if position.pnl is not None:
                strategy_pnl[position.strategy] = strategy_pnl.get(position.strategy, 0) + position.pnl
        
        # Calculate weights
        total_pnl = sum(strategy_pnl.values())
        if total_pnl > 0:
            self.state.strategy_weights = {
                strategy: pnl / total_pnl
                for strategy, pnl in strategy_pnl.items()
            }
        else:
            # Equal weights if no profit
            strategies = set(p.strategy for p in self.state.closed_positions)
            self.state.strategy_weights = {
                strategy: 1.0 / len(strategies)
                for strategy in strategies
            }
    
    def _generate_daily_commentary(self, market_data: Dict[str, Any]) -> None:
        """Generate daily trading commentary.
        
        Args:
            market_data: Market data and context
        """
        # Check if we need to generate commentary
        last_commentary = self.state.timestamp.date()
        current_date = datetime.utcnow().date()
        
        if current_date > last_commentary:
            # Get trades for the day
            daily_trades = [
                p.to_dict() for p in self.state.closed_positions
                if p.exit_time and p.exit_time.date() == current_date
            ]
            
            # Generate commentary
            commentary = self.llm_interface.generate_daily_commentary(
                portfolio_state=self.state.to_dict(),
                trades=daily_trades,
                market_data=market_data
            )
            
            if commentary:
                # Log commentary
                self._log_commentary(commentary)
                
                # Publish update
                self._publish_update("daily_commentary", commentary.to_dict())
    
    def _publish_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish portfolio update to Redis.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if self.redis is not None:
            try:
                message = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': event_type,
                    'data': data
                }
                self.redis.publish('portfolio_updates', json.dumps(message))
            except RedisError as e:
                logger.error(f"Failed to publish update: {e}")
    
    def _log_position(self, position: Position, action: str) -> None:
        """Log position action.
        
        Args:
            position: Position to log
            action: Action performed ("open" or "close")
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "position": position.to_dict()
        }
        
        # Save to JSON
        log_path = f"trading/portfolio/logs/positions_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Save to CSV
        csv_path = f"trading/portfolio/data/positions_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        df = pd.DataFrame([position.to_dict()])
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        logger.info(f"Logged {action} position for {position.symbol}")
    
    def _log_commentary(self, commentary: DailyCommentary) -> None:
        """Log daily commentary.
        
        Args:
            commentary: Daily commentary to log
        """
        # Save to JSON
        log_path = f"trading/portfolio/logs/commentary_{commentary.date.strftime('%Y%m%d')}.json"
        with open(log_path, "w") as f:
            json.dump(commentary.to_dict(), f, indent=2)
        
        logger.info(f"Logged daily commentary for {commentary.date.date()}")

__all__ = ["PortfolioManager", "PortfolioState", "Position", "PositionStatus", "TradeDirection"]
