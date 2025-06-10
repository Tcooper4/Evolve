import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

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

class PortfolioManager:
    def __init__(self):
        """Initialize the portfolio manager."""
        self.positions = pd.DataFrame(columns=['symbol', 'quantity', 'avg_price', 'current_price', 'pnl', 'pnl_pct'])
        self.cash = 1000000.0  # Starting cash
        self.trades = []
    
    def add_position(self, symbol: str, quantity: int, price: float):
        """Add a new position or update existing one."""
        if symbol in self.positions['symbol'].values:
            # Update existing position
            idx = self.positions[self.positions['symbol'] == symbol].index[0]
            old_quantity = self.positions.loc[idx, 'quantity']
            old_avg_price = self.positions.loc[idx, 'avg_price']
            
            new_quantity = old_quantity + quantity
            new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
            
            self.positions.loc[idx, 'quantity'] = new_quantity
            self.positions.loc[idx, 'avg_price'] = new_avg_price
            self.positions.loc[idx, 'current_price'] = price
        else:
            # Add new position
            new_position = pd.DataFrame({
                'symbol': [symbol],
                'quantity': [quantity],
                'avg_price': [price],
                'current_price': [price],
                'pnl': [0.0],
                'pnl_pct': [0.0]
            })
            self.positions = pd.concat([self.positions, new_position], ignore_index=True)
        
        # Update cash
        self.cash -= quantity * price
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'type': 'buy' if quantity > 0 else 'sell'
        })
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices and calculate P&L."""
        for symbol, price in prices.items():
            if symbol in self.positions['symbol'].values:
                idx = self.positions[self.positions['symbol'] == symbol].index[0]
                self.positions.loc[idx, 'current_price'] = price
                
                # Calculate P&L
                quantity = self.positions.loc[idx, 'quantity']
                avg_price = self.positions.loc[idx, 'avg_price']
                pnl = quantity * (price - avg_price)
                pnl_pct = (price - avg_price) / avg_price * 100
                
                self.positions.loc[idx, 'pnl'] = pnl
                self.positions.loc[idx, 'pnl_pct'] = pnl_pct
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = (self.positions['quantity'] * self.positions['current_price']).sum()
        return positions_value + self.cash
    
    def get_portfolio_summary(self) -> Dict[str, float]:
        """Get portfolio summary statistics."""
        total_value = self.get_portfolio_value()
        total_pnl = self.positions['pnl'].sum()
        total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'num_positions': len(self.positions)
        }
    
    def get_recent_trades(self, n: int = 5) -> List[Dict]:
        """Get n most recent trades."""
        return self.trades[-n:]

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
        if required_margin > self.cash:
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
        
        self.positions.loc[len(self.positions)] = [
            symbol, size, price, price, 0.0, 0.0
        ]
        
        self.cash -= required_margin
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': size,
            'price': price,
            'type': 'buy' if size > 0 else 'sell'
        })
        
        return True
    
    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            
        Returns:
            Realized P&L if position closed successfully, None otherwise
        """
        if symbol not in self.positions['symbol'].values:
            return None
        
        idx = self.positions[self.positions['symbol'] == symbol].index[0]
        quantity = self.positions.loc[idx, 'quantity']
        avg_price = self.positions.loc[idx, 'avg_price']
        realized_pnl = quantity * (price - avg_price)
        
        # Update cash
        self.cash += self._calculate_required_margin(symbol, quantity, price)
        self.cash += realized_pnl
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': -quantity,
            'price': price,
            'type': 'sell' if quantity > 0 else 'buy'
        })
        
        # Remove position
        self.positions.drop(idx, inplace=True)
        
        return realized_pnl
    
    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update position prices and check for stop loss/take profit.
        
        Args:
            prices: Dictionary of current prices by symbol
        """
        current_time = datetime.now()
        positions_to_close = []
        
        for symbol, position in self.positions.iterrows():
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]
            
            # Check stop loss
            if position['stop_loss'] is not None:
                if (position['type'] == PositionType.LONG and current_price <= position['stop_loss']) or \
                   (position['type'] == PositionType.SHORT and current_price >= position['stop_loss']):
                    positions_to_close.append((symbol, current_price))
            
            # Check take profit
            if position['take_profit'] is not None:
                if (position['type'] == PositionType.LONG and current_price >= position['take_profit']) or \
                   (position['type'] == PositionType.SHORT and current_price <= position['take_profit']):
                    positions_to_close.append((symbol, current_price))
        
        # Close positions that hit stop loss or take profit
        for symbol, price in positions_to_close:
            self.close_position(symbol, price)
    
    def calculate_performance_metrics(self, prices: Dict[str, float]) -> Dict:
        """Calculate portfolio performance metrics.
        
        Args:
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of performance metrics
        """
        total_value = self.get_portfolio_value()
        
        # Calculate returns
        total_return = (total_value - 1000000.0) / 1000000.0
        
        # Calculate position metrics
        position_metrics = {
            symbol: {
                'unrealized_pnl': position['pnl'],
                'unrealized_pnl_pct': position['pnl_pct'],
                'holding_period': (datetime.now() - position['timestamp']).total_seconds() / 3600  # hours
            }
            for symbol, position in self.positions.iterrows()
        }
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics()
        
        return {
            'total_value': total_value,
            'cash': self.cash,
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
        current_value = self.get_portfolio_value()
        rebalance_trades = []
        
        # Calculate target positions
        target_positions = {
            symbol: (weight * current_value) / price
            for symbol, (weight, price) in zip(target_weights.items(), prices.items())
        }
        
        # Close positions not in target weights
        for symbol in list(self.positions['symbol'].values):
            if symbol not in target_weights:
                rebalance_trades.append((symbol, -self.positions[self.positions['symbol'] == symbol].iloc[0]['quantity'], prices[symbol]))
                self.close_position(symbol, prices[symbol])
        
        # Adjust existing positions
        for symbol, target_size in target_positions.items():
            current_size = self.positions[self.positions['symbol'] == symbol].iloc[0]['quantity'] if symbol in self.positions['symbol'].values else 0
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
        if len(self.trades) == 0:
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['type'] == 'buy')
        losing_trades = sum(1 for trade in self.trades if trade['type'] == 'sell')
        
        # Calculate P&L metrics
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        avg_win = np.mean([trade['pnl'] for trade in self.trades if trade['type'] == 'buy']) if winning_trades > 0 else 0
        avg_loss = np.mean([trade['pnl'] for trade in self.trades if trade['type'] == 'sell']) if losing_trades > 0 else 0
        
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