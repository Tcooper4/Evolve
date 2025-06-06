import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class PortfolioManager:
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # asset -> quantity
        self.prices: Dict[str, float] = {}  # asset -> price

    def add_asset(self, asset: str, quantity: float, price: float) -> None:
        """Add an asset to the portfolio."""
        if asset in self.positions:
            self.positions[asset] += quantity
        else:
            self.positions[asset] = quantity
        self.prices[asset] = price
        self.cash -= quantity * price

    def remove_asset(self, asset: str, quantity: float) -> None:
        """Remove an asset from the portfolio."""
        if asset not in self.positions or self.positions[asset] < quantity:
            raise ValueError(f"Not enough {asset} to remove.")
        self.positions[asset] -= quantity
        self.cash += quantity * self.prices[asset]
        if self.positions[asset] == 0:
            del self.positions[asset]
            del self.prices[asset]

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update the prices of assets in the portfolio."""
        for asset, price in prices.items():
            if asset in self.positions:
                self.prices[asset] = price

    def get_portfolio_value(self) -> float:
        """Calculate the total value of the portfolio."""
        total_value = self.cash
        for asset, quantity in self.positions.items():
            total_value += quantity * self.prices[asset]
        return total_value

    def rebalance(self, target_weights: Dict[str, float]) -> None:
        """Rebalance the portfolio to match target weights."""
        total_value = self.get_portfolio_value()
        for asset, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_value = self.positions.get(asset, 0) * self.prices.get(asset, 0)
            if current_value < target_value:
                # Buy more
                quantity_to_buy = (target_value - current_value) / self.prices[asset]
                self.add_asset(asset, quantity_to_buy, self.prices[asset])
            elif current_value > target_value:
                # Sell some
                quantity_to_sell = (current_value - target_value) / self.prices[asset]
                self.remove_asset(asset, quantity_to_sell) 