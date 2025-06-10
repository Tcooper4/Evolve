import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
from datetime import datetime

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

class BacktestEngine:
    def __init__(self, data: pd.DataFrame):
        """Initialize the backtest engine with historical data."""
        self.data = data
        self.results = {}
        self.positions = pd.DataFrame()
        self.trades = []
    
    def run_backtest(self, strategy: str, params: Dict) -> Dict:
        """Run backtest with specified strategy and parameters."""
        if strategy == "Momentum":
            return self._run_momentum_strategy(params)
        elif strategy == "Mean Reversion":
            return self._run_mean_reversion_strategy(params)
        elif strategy == "ML-Based":
            return self._run_ml_strategy(params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _run_momentum_strategy(self, params: Dict) -> Dict:
        """Run momentum strategy backtest."""
        # Calculate returns
        returns = self.data['Close'].pct_change()
        
        # Calculate momentum signal
        lookback = params.get('lookback', 20)
        momentum = returns.rolling(window=lookback).mean()
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[momentum > params.get('threshold', 0)] = 1
        signals[momentum < -params.get('threshold', 0)] = -1
        
        # Calculate performance
        strategy_returns = signals.shift(1) * returns
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'returns': strategy_returns
        }
    
    def _run_mean_reversion_strategy(self, params: Dict) -> Dict:
        """Run mean reversion strategy backtest."""
        # Calculate moving average
        ma_period = params.get('ma_period', 20)
        ma = self.data['Close'].rolling(window=ma_period).mean()
        
        # Calculate z-score
        std = self.data['Close'].rolling(window=ma_period).std()
        z_score = (self.data['Close'] - ma) / std
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[z_score < -params.get('entry_threshold', 2)] = 1
        signals[z_score > params.get('exit_threshold', 2)] = -1
        
        # Calculate performance
        returns = self.data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'returns': strategy_returns
        }
    
    def _run_ml_strategy(self, params: Dict) -> Dict:
        """Run ML-based strategy backtest."""
        # This is a placeholder for ML strategy implementation
        # In a real implementation, you would use your trained ML model here
        
        # For now, return dummy results
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'returns': pd.Series(0, index=self.data.index)
        }
    
    def get_backtest_metrics(self) -> Dict[str, float]:
        """Get backtest metrics."""
        if not self.results:
            return {}
        
        return {
            'total_return': self.results.get('total_return', 0.0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0.0),
            'max_drawdown': self.results.get('max_drawdown', 0.0),
            'win_rate': self.results.get('win_rate', 0.0)
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades) 