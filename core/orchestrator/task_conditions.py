"""
Task Conditions Module

This module contains task condition checking functionality for the task orchestrator.
Extracted from the original task_orchestrator.py for modularity.
"""

import logging
from datetime import datetime, time
from typing import Any, Dict, Optional


class TaskConditions:
    """Handles task condition checking."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    async def check_condition(self, condition: str, value: Any) -> bool:
        """Check if a condition is met."""
        try:
            if condition == "market_hours":
                return await self._is_market_hours()
            elif condition == "system_health":
                return await self._get_system_health() > value
            elif condition == "market_volatility":
                return await self._check_market_volatility(value)
            elif condition == "news_events":
                return await self._check_news_events()
            elif condition == "position_count":
                return await self._check_position_count(value)
            elif condition == "pending_orders":
                return await self._check_pending_orders()
            elif condition == "new_trades":
                return await self._check_new_trades()
            elif condition == "trading_activity":
                return await self._check_trading_activity()
            else:
                self.logger.warning(f"Unknown condition: {condition}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to check condition {condition}: {e}")
            return False

    async def _is_market_hours(self) -> bool:
        """Check if it's currently market hours."""
        try:
            now = datetime.utcnow()
            current_time = now.time()
            
            # US Market Hours (9:30 AM - 4:00 PM ET, Monday-Friday)
            market_start = time(9, 30)  # 9:30 AM
            market_end = time(16, 0)    # 4:00 PM
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's within market hours
            return market_start <= current_time <= market_end
            
        except Exception as e:
            self.logger.error(f"Failed to check market hours: {e}")
            return True  # Default to True to avoid blocking

    async def _get_system_health(self) -> float:
        """Get system health score."""
        try:
            # This would typically query system health metrics
            # For now, return a default health score
            return 0.9
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return 0.5

    async def _check_market_volatility(self, expected_level: str) -> bool:
        """Check if market volatility matches expected level."""
        try:
            # This would typically query market data
            # For now, return True to avoid blocking
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check market volatility: {e}")
            return True

    async def _check_news_events(self) -> bool:
        """Check if there are significant news events."""
        try:
            # This would typically query news APIs
            # For now, return False (no significant news)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check news events: {e}")
            return False

    async def _check_position_count(self, condition: str) -> bool:
        """Check position count conditions."""
        try:
            # This would typically query portfolio data
            # For now, return True to avoid blocking
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check position count: {e}")
            return True

    async def _check_pending_orders(self) -> bool:
        """Check if there are pending orders."""
        try:
            # This would typically query order management system
            # For now, return False (no pending orders)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check pending orders: {e}")
            return False

    async def _check_new_trades(self) -> bool:
        """Check if there are new trades."""
        try:
            # This would typically query trade history
            # For now, return False (no new trades)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check new trades: {e}")
            return False

    async def _check_trading_activity(self) -> bool:
        """Check if there's recent trading activity."""
        try:
            # This would typically query trading activity metrics
            # For now, return True to allow trading
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check trading activity: {e}")
            return True

    def get_available_conditions(self) -> Dict[str, str]:
        """Get list of available conditions."""
        return {
            "market_hours": "Check if it's currently market hours",
            "system_health": "Check if system health is above threshold",
            "market_volatility": "Check if market volatility matches expected level",
            "news_events": "Check if there are significant news events",
            "position_count": "Check position count conditions",
            "pending_orders": "Check if there are pending orders",
            "new_trades": "Check if there are new trades",
            "trading_activity": "Check if there's recent trading activity"
        }

    def validate_condition(self, condition: str, value: Any) -> bool:
        """Validate a condition and its value."""
        available_conditions = self.get_available_conditions()
        
        if condition not in available_conditions:
            return False
        
        # Validate value based on condition type
        if condition == "system_health":
            return isinstance(value, (int, float)) and 0 <= value <= 1
        elif condition == "market_volatility":
            return isinstance(value, str) and value in ["low", "medium", "high"]
        elif condition == "position_count":
            return isinstance(value, str) and value in ["<5", "5-10", ">10"]
        else:
            return True  # Other conditions don't require specific value validation 