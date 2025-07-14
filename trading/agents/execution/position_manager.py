"""
Position Manager Module

This module contains position management functionality for the execution agent.
Extracted from the original execution_agent.py for modularity.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .risk_controls import ExitEvent, ExitReason
from trading.portfolio.portfolio_manager import Position


class PositionManager:
    """Manages position tracking, exits, and risk monitoring."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.exit_events: List[ExitEvent] = []
        
        # Risk monitoring
        self.daily_pnl = 0.0
        self.max_daily_loss = config.get("max_daily_loss", 0.02)
        self.risk_metrics_cache = {}
        
        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage for positions and exit events."""
        storage_dir = Path("data/execution")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.positions_file = storage_dir / "positions.json"
        self.exit_events_file = storage_dir / "exit_events.json"
        
        # Load existing data
        self._load_positions()
        self._load_exit_events()

    def _load_positions(self) -> None:
        """Load positions from storage."""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, 'r') as f:
                    positions_data = json.load(f)
                    for pos_data in positions_data:
                        position = Position.from_dict(pos_data)
                        self.positions[position.position_id] = position
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")

    def _load_exit_events(self) -> None:
        """Load exit events from storage."""
        try:
            if self.exit_events_file.exists():
                with open(self.exit_events_file, 'r') as f:
                    events_data = json.load(f)
                    for event_data in events_data:
                        exit_event = ExitEvent.from_dict(event_data)
                        self.exit_events.append(exit_event)
        except Exception as e:
            self.logger.error(f"Failed to load exit events: {e}")

    def _save_positions(self) -> None:
        """Save positions to storage."""
        try:
            positions_data = [pos.to_dict() for pos in self.positions.values()]
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save positions: {e}")

    def _save_exit_events(self) -> None:
        """Save exit events to storage."""
        try:
            events_data = [event.to_dict() for event in self.exit_events]
            with open(self.exit_events_file, 'w') as f:
                json.dump(events_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save exit events: {e}")

    def add_position(self, position: Position) -> None:
        """Add a new position."""
        self.positions[position.position_id] = position
        self._save_positions()

    def remove_position(self, position_id: str) -> None:
        """Remove a position."""
        if position_id in self.positions:
            del self.positions[position_id]
            self._save_positions()

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol."""
        return [pos for pos in self.positions.values() if pos.symbol == symbol]

    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())

    def update_position(self, position_id: str, updates: Dict[str, Any]) -> None:
        """Update a position with new data."""
        if position_id in self.positions:
            position = self.positions[position_id]
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            self._save_positions()

    def calculate_position_risk_metrics(
        self, position: Position, current_price: float, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for a position."""
        entry_price = position.entry_price
        size = position.size
        
        # Basic metrics
        pnl = (current_price - entry_price) * size if position.direction.value == 'long' else (entry_price - current_price) * size
        pnl_percentage = (pnl / (entry_price * size)) * 100
        
        # Volatility-based metrics
        volatility = market_data.get(f"{position.symbol}_volatility", 0.0)
        var_95 = size * entry_price * volatility * 1.645  # 95% VaR
        
        # Drawdown calculation
        max_price = position.max_price if hasattr(position, 'max_price') else entry_price
        drawdown = ((max_price - current_price) / max_price) * 100 if position.direction.value == 'long' else ((current_price - max_price) / max_price) * 100
        
        return {
            "pnl": pnl,
            "pnl_percentage": pnl_percentage,
            "volatility": volatility,
            "var_95": var_95,
            "drawdown": drawdown,
            "current_price": current_price,
            "entry_price": entry_price
        }

    def calculate_portfolio_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        if not self.positions:
            return {
                "total_pnl": 0.0,
                "total_value": 0.0,
                "portfolio_volatility": 0.0,
                "max_drawdown": 0.0,
                "correlation": 0.0
            }
        
        # Calculate total portfolio metrics
        total_pnl = 0.0
        total_value = 0.0
        position_values = []
        
        for position in self.positions.values():
            current_price = market_data.get(f"{position.symbol}_price", position.entry_price)
            position_value = current_price * position.size
            position_pnl = (current_price - position.entry_price) * position.size if position.direction.value == 'long' else (position.entry_price - current_price) * position.size
            
            total_pnl += position_pnl
            total_value += position_value
            position_values.append(position_value)
        
        # Calculate portfolio volatility
        if len(position_values) > 1:
            weights = np.array(position_values) / total_value
            correlation_matrix = self._get_correlation_matrix(market_data)
            if correlation_matrix is not None:
                portfolio_volatility = np.sqrt(weights.T @ correlation_matrix @ weights)
            else:
                portfolio_volatility = 0.0
        else:
            portfolio_volatility = 0.0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_portfolio_drawdown()
        
        # Calculate correlation
        correlation = self._calculate_portfolio_correlation(market_data)
        
        return {
            "total_pnl": total_pnl,
            "total_value": total_value,
            "portfolio_volatility": portfolio_volatility,
            "max_drawdown": max_drawdown,
            "correlation": correlation,
            "position_count": len(self.positions)
        }

    def _get_correlation_matrix(self, market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get correlation matrix for portfolio positions."""
        try:
            symbols = [pos.symbol for pos in self.positions.values()]
            if len(symbols) < 2:
                return None
            
            # Get price data for correlation calculation
            price_data = {}
            for symbol in symbols:
                price_key = f"{symbol}_price_history"
                if price_key in market_data:
                    price_data[symbol] = market_data[price_key]
            
            if len(price_data) < 2:
                return None
            
            # Create DataFrame and calculate correlation
            df = pd.DataFrame(price_data)
            return df.corr()
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation matrix: {e}")
            return None

    def _calculate_portfolio_correlation(self, market_data: Dict[str, Any]) -> float:
        """Calculate average correlation between portfolio positions."""
        try:
            symbols = [pos.symbol for pos in self.positions.values()]
            if len(symbols) < 2:
                return 0.0
            
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    corr_key = f"{symbol1}_{symbol2}_correlation"
                    if corr_key in market_data:
                        correlations.append(market_data[corr_key])
            
            return np.mean(correlations) if correlations else 0.0
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio correlation: {e}")
            return 0.0

    def _calculate_portfolio_drawdown(self) -> float:
        """Calculate portfolio drawdown."""
        try:
            if not self.exit_events:
                return 0.0
            
            # Calculate cumulative PnL over time
            pnl_history = []
            cumulative_pnl = 0.0
            
            for event in sorted(self.exit_events, key=lambda x: x.timestamp):
                cumulative_pnl += event.pnl
                pnl_history.append(cumulative_pnl)
            
            if not pnl_history:
                return 0.0
            
            # Calculate drawdown
            peak = pnl_history[0]
            max_drawdown = 0.0
            
            for pnl in pnl_history:
                if pnl > peak:
                    peak = pnl
                drawdown = (peak - pnl) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown * 100  # Convert to percentage
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio drawdown: {e}")
            return 0.0

    def add_exit_event(self, exit_event: ExitEvent) -> None:
        """Add an exit event."""
        self.exit_events.append(exit_event)
        self._save_exit_events()
        
        # Update daily PnL
        if exit_event.timestamp.date() == datetime.utcnow().date():
            self.daily_pnl += exit_event.pnl

    def get_exit_events(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get exit events within a date range."""
        events = self.exit_events
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        return [event.to_dict() for event in events]

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for all positions."""
        if not self.positions:
            return {
                "total_positions": 0,
                "total_value": 0.0,
                "total_pnl": 0.0,
                "risk_level": "low"
            }
        
        total_value = sum(pos.entry_price * pos.size for pos in self.positions.values())
        total_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.size if pos.direction.value == 'long' 
            else (pos.entry_price - pos.current_price) * pos.size 
            for pos in self.positions.values()
        )
        
        # Determine risk level
        if total_pnl < -total_value * 0.05:
            risk_level = "high"
        elif total_pnl < -total_value * 0.02:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "total_positions": len(self.positions),
            "total_value": total_value,
            "total_pnl": total_pnl,
            "risk_level": risk_level,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss
        }

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL tracking."""
        self.daily_pnl = 0.0

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded."""
        return abs(self.daily_pnl) > self.max_daily_loss 