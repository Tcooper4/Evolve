import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_cash: float = 100000.0):
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # asset -> quantity
        self.trades: List[Dict] = []
        self.portfolio_values: List[float] = [initial_cash]

    def run_backtest(self, strategy: Callable) -> None:
        """Run a backtest using the provided strategy function."""
        for i in range(len(self.data)):
            current_data = self.data.iloc[:i+1]
            signals = strategy(current_data)
            self._execute_trades(signals, current_data.iloc[-1])
            self.portfolio_values.append(self._calculate_portfolio_value(current_data.iloc[-1]))

    def _execute_trades(self, signals: Dict[str, float], current_prices: pd.Series) -> None:
        """Execute trades based on the signals."""
        for asset, signal in signals.items():
            if signal > 0:  # Buy
                quantity = signal * self.cash / current_prices[asset]
                self.positions[asset] = self.positions.get(asset, 0) + quantity
                self.cash -= quantity * current_prices[asset]
                self.trades.append({'asset': asset, 'quantity': quantity, 'price': current_prices[asset], 'type': 'buy'})
            elif signal < 0:  # Sell
                quantity = abs(signal) * self.positions.get(asset, 0)
                if quantity > 0:
                    self.positions[asset] -= quantity
                    self.cash += quantity * current_prices[asset]
                    self.trades.append({'asset': asset, 'quantity': quantity, 'price': current_prices[asset], 'type': 'sell'})

    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calculate the current portfolio value."""
        total_value = self.cash
        for asset, quantity in self.positions.items():
            total_value += quantity * current_prices[asset]
        return total_value

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the backtest."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        return {
            'total_return': (self.portfolio_values[-1] / self.initial_cash) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (pd.Series(self.portfolio_values).cummax() - self.portfolio_values).max() / pd.Series(self.portfolio_values).cummax()
        }

    def plot_results(self) -> None:
        """Plot the portfolio value over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show() 