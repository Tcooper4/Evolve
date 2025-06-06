import logging
from typing import Dict, List, Optional
from datetime import datetime

class ExecutionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.trades: List[Dict] = []

    def execute_market_order(self, asset: str, quantity: float, price: float) -> Dict:
        """Execute a market order and log the details."""
        trade = {
            'asset': asset,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'type': 'market'
        }
        self.trades.append(trade)
        self.logger.info(f"Executed market order: {trade}")
        return trade

    def execute_limit_order(self, asset: str, quantity: float, limit_price: float) -> Optional[Dict]:
        """Execute a limit order if the current price is favorable."""
        # Simulate current market price (replace with actual market data)
        current_price = limit_price * 0.99  # Example: current price is slightly lower
        if current_price <= limit_price:
            trade = {
                'asset': asset,
                'quantity': quantity,
                'price': current_price,
                'timestamp': datetime.now(),
                'type': 'limit'
            }
            self.trades.append(trade)
            self.logger.info(f"Executed limit order: {trade}")
            return trade
        else:
            self.logger.info(f"Limit order not executed: current price {current_price} > limit price {limit_price}")
            return None

    def get_trade_history(self) -> List[Dict]:
        """Return the history of executed trades."""
        return self.trades 