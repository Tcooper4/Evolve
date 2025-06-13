import os
import json
import redis
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Position:
    symbol: str
    type: PositionType
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    current_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.type == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.type == PositionType.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def holding_period(self) -> timedelta:
        """Calculate position holding period."""
        return self.current_time - self.entry_time

class PortfolioError(Exception):
    """Custom exception for portfolio errors."""
    pass

class PortfolioManager:
    """Manages trading positions and portfolio value."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize portfolio manager.
        
        Args:
            config: Configuration dictionary containing:
                - redis_host: Redis host (default: localhost)
                - redis_port: Redis port (default: 6379)
                - redis_db: Redis database (default: 0)
                - redis_password: Redis password
                - redis_ssl: Whether to use SSL (default: false)
                - max_positions: Maximum number of positions (default: 100)
                - position_size_limit: Maximum position size (default: 100000)
                - risk_limit: Maximum risk per position (default: 0.02)
        """
        if config is None:
            config = {}
            
        # Determine if Redis should be used. Fallback to in-memory storage if
        # disabled or connection fails. This allows the portfolio manager to be
        # used in environments where Redis is not available.
        self.use_redis = os.getenv('USE_REDIS', 'true').lower() == 'true'
        self.redis_client = None
        self.positions: Dict[str, Dict[str, Any]] = {}

        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 0)),
                    password=os.getenv('REDIS_PASSWORD'),
                    ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true'
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                # Fall back to in-memory storage on connection failure
                self.logger = logging.getLogger(self.__class__.__name__)
                self.logger.warning(
                    "Redis unavailable, falling back to in-memory storage: %s",
                    str(e),
                )
                self.redis_client = None
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = {
            'max_positions': int(os.getenv('MAX_POSITIONS', 100)),
            'position_size_limit': float(os.getenv('POSITION_SIZE_LIMIT', 100000)),
            'risk_limit': float(os.getenv('RISK_LIMIT', 0.02))
        }
        self.config.update(config)
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_handler = RotatingFileHandler(
            os.getenv('LOG_FILE', 'trading.log'),
            maxBytes=int(os.getenv('LOG_MAX_SIZE', 10485760)),
            backupCount=int(os.getenv('LOG_BACKUP_COUNT', 5))
        )
        log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
        
    def _validate_position(self, position: Dict[str, Any]) -> None:
        """Validate position data.
        
        Args:
            position: Position data to validate
            
        Raises:
            PortfolioError: If position data is invalid
        """
        required_fields = ['symbol', 'quantity', 'entry_price', 'entry_time']
        if not all(field in position for field in required_fields):
            raise PortfolioError(f"Position missing required fields: {required_fields}")
            
        if position['quantity'] <= 0:
            raise PortfolioError("Position quantity must be positive")
            
        if position['entry_price'] <= 0:
            raise PortfolioError("Entry price must be positive")
            
        try:
            datetime.fromisoformat(position['entry_time'])
        except ValueError:
            raise PortfolioError("Invalid entry time format")
            
    def add_position(self, position: Dict[str, Any]) -> str:
        """Add a new position.
        
        Args:
            position: Position data
            
        Returns:
            Position ID
            
        Raises:
            PortfolioError: If position cannot be added
        """
        try:
            self._validate_position(position)
            
            # Check position limits
            if len(self.get_positions()) >= self.config['max_positions']:
                raise PortfolioError("Maximum number of positions reached")
                
            position_value = position['quantity'] * position['entry_price']
            if position_value > self.config['position_size_limit']:
                raise PortfolioError("Position size exceeds limit")
                
            # Generate position ID
            position_id = f"{position['symbol']}_{datetime.utcnow().isoformat()}"
            
            # Store position either in Redis or in-memory
            if self.redis_client:
                self.redis_client.hset(
                    'positions',
                    position_id,
                    json.dumps(position)
                )
            else:
                self.positions[position_id] = position
            
            self.logger.info(f"Added position {position_id}")
            return position_id
            
        except Exception as e:
            raise PortfolioError(f"Failed to add position: {str(e)}")
            
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing position.
        
        Args:
            position_id: Position ID
            updates: Position updates
            
        Raises:
            PortfolioError: If position cannot be updated
        """
        try:
            position = self.get_position(position_id)
            if not position:
                raise PortfolioError(f"Position {position_id} not found")
                
            # Update position
            position.update(updates)
            self._validate_position(position)
            
            # Store updated position
            if self.redis_client:
                self.redis_client.hset(
                    'positions',
                    position_id,
                    json.dumps(position)
                )
            else:
                self.positions[position_id] = position
            
            self.logger.info(f"Updated position {position_id}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to update position: {str(e)}")
            
    def remove_position(self, position_id: str) -> None:
        """Remove a position.
        
        Args:
            position_id: Position ID
            
        Raises:
            PortfolioError: If position cannot be removed
        """
        try:
            if self.redis_client:
                if not self.redis_client.hexists('positions', position_id):
                    raise PortfolioError(f"Position {position_id} not found")
                self.redis_client.hdel('positions', position_id)
            else:
                if position_id not in self.positions:
                    raise PortfolioError(f"Position {position_id} not found")
                del self.positions[position_id]
            self.logger.info(f"Removed position {position_id}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to remove position: {str(e)}")
            
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position data if found, None otherwise
            
        Raises:
            PortfolioError: If position cannot be retrieved
        """
        try:
            if self.redis_client:
                position_data = self.redis_client.hget('positions', position_id)
                if position_data:
                    return json.loads(position_data)
                return None
            else:
                return self.positions.get(position_id)
            
        except Exception as e:
            raise PortfolioError(f"Failed to get position: {str(e)}")
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions.
        
        Returns:
            List of position data
            
        Raises:
            PortfolioError: If positions cannot be retrieved
        """
        try:
            if self.redis_client:
                positions = []
                for position_data in self.redis_client.hgetall('positions').values():
                    positions.append(json.loads(position_data))
                return positions
            else:
                return list(self.positions.values())
            
        except Exception as e:
            raise PortfolioError(f"Failed to get positions: {str(e)}")
            
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update position prices.
        
        Args:
            prices: Dictionary of symbol to price mappings
            
        Raises:
            PortfolioError: If prices cannot be updated
        """
        try:
            if self.redis_client:
                positions = self.get_positions()
                for position in positions:
                    symbol = position['symbol']
                    if symbol in prices:
                        position['current_price'] = prices[symbol]
                        self.redis_client.hset(
                            'positions',
                            f"{symbol}_{position['entry_time']}",
                            json.dumps(position)
                        )
            else:
                for pid, position in list(self.positions.items()):
                    symbol = position['symbol']
                    if symbol in prices:
                        position['current_price'] = prices[symbol]
                        self.positions[pid] = position

            self.logger.info("Updated position prices")
            
        except Exception as e:
            raise PortfolioError(f"Failed to update prices: {str(e)}")
            
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value.
        
        Returns:
            Total portfolio value
            
        Raises:
            PortfolioError: If portfolio value cannot be calculated
        """
        try:
            positions = self.get_positions()
            total_value = 0
            
            for position in positions:
                if 'current_price' in position:
                    total_value += position['quantity'] * position['current_price']
                else:
                    total_value += position['quantity'] * position['entry_price']
                    
            return total_value
            
        except Exception as e:
            raise PortfolioError(f"Failed to calculate portfolio value: {str(e)}")
            
    def calculate_position_pnl(self, position_id: str) -> float:
        """Calculate position P&L.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position P&L
            
        Raises:
            PortfolioError: If P&L cannot be calculated
        """
        try:
            position = self.get_position(position_id)
            if not position:
                raise PortfolioError(f"Position {position_id} not found")
                
            if 'current_price' not in position:
                return 0
                
            return (position['current_price'] - position['entry_price']) * position['quantity']
            
        except Exception as e:
            raise PortfolioError(f"Failed to calculate P&L: {str(e)}")
            
    def calculate_portfolio_pnl(self) -> float:
        """Calculate total portfolio P&L.
        
        Returns:
            Total portfolio P&L
            
        Raises:
            PortfolioError: If P&L cannot be calculated
        """
        try:
            positions = self.get_positions()
            total_pnl = 0
            
            for position in positions:
                if 'current_price' in position:
                    total_pnl += (position['current_price'] - position['entry_price']) * position['quantity']
                    
            return total_pnl
            
        except Exception as e:
            raise PortfolioError(f"Failed to calculate portfolio P&L: {str(e)}")
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.
        
        Returns:
            Portfolio summary dictionary
            
        Raises:
            PortfolioError: If summary cannot be generated
        """
        try:
            positions = self.get_positions()
            total_value = self.calculate_portfolio_value()
            total_pnl = self.calculate_portfolio_pnl()
            
            return {
                'total_positions': len(positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'positions': positions
            }
            
        except Exception as e:
            raise PortfolioError(f"Failed to generate portfolio summary: {str(e)}")
            
    def save_portfolio(self, path: str) -> None:
        """Save portfolio to file.
        
        Args:
            path: Path to save portfolio
            
        Raises:
            PortfolioError: If portfolio cannot be saved
        """
        try:
            portfolio_data = {
                'positions': self.get_positions(),
                'config': self.config,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(portfolio_data, f, indent=4)
                
            self.logger.info(f"Saved portfolio to {path}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to save portfolio: {str(e)}")
            
    def load_portfolio(self, path: str) -> None:
        """Load portfolio from file.
        
        Args:
            path: Path to load portfolio from
            
        Raises:
            PortfolioError: If portfolio cannot be loaded
        """
        try:
            with open(path, 'r') as f:
                portfolio_data = json.load(f)
                
            # Clear existing positions
            if self.redis_client:
                self.redis_client.delete('positions')
            else:
                self.positions.clear()
            
            # Load positions
            for position in portfolio_data['positions']:
                self._validate_position(position)
                if self.redis_client:
                    self.redis_client.hset(
                        'positions',
                        f"{position['symbol']}_{position['entry_time']}",
                        json.dumps(position)
                    )
                else:
                    pid = f"{position['symbol']}_{position['entry_time']}"
                    self.positions[pid] = position
                
            # Update configuration
            self.config.update(portfolio_data['config'])
            
            self.logger.info(f"Loaded portfolio from {path}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to load portfolio: {str(e)}")
            
    def export_to_csv(self, path: str) -> None:
        """Export portfolio to CSV.
        
        Args:
            path: Path to save CSV
            
        Raises:
            PortfolioError: If portfolio cannot be exported
        """
        try:
            positions = self.get_positions()
            df = pd.DataFrame(positions)
            
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(save_path, index=False)
            self.logger.info(f"Exported portfolio to {path}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to export portfolio: {str(e)}")
            
    def import_from_csv(self, path: str) -> None:
        """Import portfolio from CSV.
        
        Args:
            path: Path to load CSV from
            
        Raises:
            PortfolioError: If portfolio cannot be imported
        """
        try:
            df = pd.read_csv(path)
            required_columns = ['symbol', 'quantity', 'entry_price', 'entry_time']
            
            if not all(col in df.columns for col in required_columns):
                raise PortfolioError(f"CSV missing required columns: {required_columns}")
                
            # Clear existing positions
            if self.redis_client:
                self.redis_client.delete('positions')
            else:
                self.positions.clear()
            
            # Import positions
            for _, row in df.iterrows():
                position = row.to_dict()
                self._validate_position(position)
                if self.redis_client:
                    self.redis_client.hset(
                        'positions',
                        f"{position['symbol']}_{position['entry_time']}",
                        json.dumps(position)
                    )
                else:
                    pid = f"{position['symbol']}_{position['entry_time']}"
                    self.positions[pid] = position
                
            self.logger.info(f"Imported portfolio from {path}")
            
        except Exception as e:
            raise PortfolioError(f"Failed to import portfolio: {str(e)}")

    def open_position(self,
                     symbol: str,
                     position_type: PositionType,
                     size: float,
                     price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """Open a new position.
        
        Args:
            symbol: Trading symbol
            position_type: Position type (LONG/SHORT)
            size: Position size
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position opened successfully, False otherwise
        """
        # Check if we have enough cash
        required_margin = self._calculate_required_margin(symbol, size, price)
        if required_margin > self.calculate_portfolio_value():
            return False
        
        # Create and store position
        position = Position(
            symbol=symbol,
            type=position_type,
            size=size,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price,
            current_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.add_position(position.dict())
        
        return True
    
    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            
        Returns:
            Realized P&L if position closed successfully, None otherwise
        """
        if symbol not in self.get_positions():
            return None
        
        position = self.get_position(symbol)
        if not position:
            return None
        
        realized_pnl = position['quantity'] * (price - position['entry_price'])
        
        # Update cash
        self.update_position(symbol, {'current_price': price})
        
        return realized_pnl
    
    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update position prices and check for stop loss/take profit.
        
        Args:
            prices: Dictionary of current prices by symbol
        """
        current_time = datetime.now()
        positions_to_close = []
        
        for position in self.get_positions():
            if position['symbol'] not in prices:
                continue
                
            current_price = prices[position['symbol']]
            
            # Check stop loss
            if position['stop_loss'] is not None:
                if (position['type'] == PositionType.LONG and current_price <= position['stop_loss']) or \
                   (position['type'] == PositionType.SHORT and current_price >= position['stop_loss']):
                    positions_to_close.append(position['symbol'])
            
            # Check take profit
            if position['take_profit'] is not None:
                if (position['type'] == PositionType.LONG and current_price >= position['take_profit']) or \
                   (position['type'] == PositionType.SHORT and current_price <= position['take_profit']):
                    positions_to_close.append(position['symbol'])
        
        # Close positions that hit stop loss or take profit
        for symbol in positions_to_close:
            self.close_position(symbol, prices[symbol])
    
    def calculate_performance_metrics(self, prices: Dict[str, float]) -> Dict:
        """Calculate portfolio performance metrics.
        
        Args:
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of performance metrics
        """
        total_value = self.calculate_portfolio_value()
        
        # Calculate returns
        total_return = (total_value - 1000000.0) / 1000000.0
        
        # Calculate position metrics
        position_metrics = {
            symbol: {
                'unrealized_pnl': position['pnl'],
                'unrealized_pnl_pct': position['pnl_pct'],
                'holding_period': (datetime.now() - position['timestamp']).total_seconds() / 3600  # hours
            }
            for symbol, position in self.get_positions().items()
        }
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics()
        
        return {
            'total_value': total_value,
            'cash': self.calculate_portfolio_value() - total_value,
            'total_return': total_return,
            'position_metrics': position_metrics,
            'trade_metrics': trade_metrics
        }
    
    def rebalance_portfolio(self,
                          target_weights: Dict[str, float],
                          prices: Dict[str, float]) -> List[Tuple[str, float, float]]:
        """Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary of target weights by symbol
            prices: Dictionary of current prices by symbol
            
        Returns:
            List of rebalancing trades (symbol, size, price)
        """
        current_value = self.calculate_portfolio_value()
        rebalance_trades = []
        
        # Calculate target positions
        target_positions = {
            symbol: (weight * current_value) / price
            for symbol, (weight, price) in zip(target_weights.items(), prices.items())
        }
        
        # Close positions not in target weights
        for symbol in list(self.get_positions()):
            if symbol not in target_weights:
                rebalance_trades.append((symbol, -self.get_position(symbol)['quantity'], prices[symbol]))
                self.remove_position(symbol)
        
        # Adjust existing positions
        for symbol, target_size in target_positions.items():
            current_size = self.get_position(symbol)['quantity'] if symbol in self.get_positions() else 0
            size_diff = target_size - current_size
            
            if abs(size_diff) > 1e-6:  # Avoid tiny trades
                rebalance_trades.append((symbol, size_diff, prices[symbol]))
                
                if size_diff > 0:
                    self.open_position(symbol, PositionType.LONG, size_diff, prices[symbol])
                else:
                    self.open_position(symbol, PositionType.SHORT, abs(size_diff), prices[symbol])
        
        return rebalance_trades
    
    def _calculate_required_margin(self, symbol: str, size: float, price: float) -> float:
        """Calculate required margin for a position."""
        # Simple margin calculation (can be enhanced with more sophisticated rules)
        margin_requirement = 0.1
        return size * price * margin_requirement
    
    def _calculate_trade_metrics(self) -> Dict:
        """Calculate trade performance metrics."""
        if len(self.get_positions()) == 0:
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.get_positions())
        winning_trades = sum(1 for position in self.get_positions().values() if position['type'] == PositionType.LONG)
        losing_trades = sum(1 for position in self.get_positions().values() if position['type'] == PositionType.SHORT)
        
        # Calculate P&L metrics
        total_pnl = sum(position['pnl'] for position in self.get_positions().values())
        avg_win = np.mean([position['pnl'] for position in self.get_positions().values() if position['type'] == PositionType.LONG]) if winning_trades > 0 else 0
        avg_loss = np.mean([position['pnl'] for position in self.get_positions().values() if position['type'] == PositionType.SHORT]) if losing_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        } 