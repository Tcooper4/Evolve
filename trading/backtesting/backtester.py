# Standard library imports
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union, Any
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SLIPPAGE = 0.001  # 0.1%
DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
DEFAULT_SPREAD = 0.0005  # 0.05%

class TradeType(Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    asset: str
    quantity: float
    price: float
    type: TradeType
    slippage: float
    transaction_cost: float
    spread: float
    cash_balance: float
    portfolio_value: float

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 initial_cash: float = 100000.0,
                 slippage: float = DEFAULT_SLIPPAGE,
                 transaction_cost: float = DEFAULT_TRANSACTION_COST,
                 spread: float = DEFAULT_SPREAD,
                 max_leverage: float = 1.0):
        """Initialize backtester.
        
        Args:
            data: Historical price data with OHLCV columns
            initial_cash: Initial cash balance
            slippage: Slippage percentage per trade
            transaction_cost: Transaction cost percentage per trade
            spread: Bid-ask spread percentage
            max_leverage: Maximum allowed leverage
        """
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # asset -> quantity
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = [initial_cash]
        self.asset_values: Dict[str, List[float]] = {}
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.spread = spread
        self.max_leverage = max_leverage
        self.trade_log: List[Dict[str, Any]] = []
        
        # Initialize asset value tracking
        for column in data.columns:
            if column not in ['open', 'high', 'low', 'close', 'volume']:
                self.asset_values[column] = [0.0]
    
    def run_backtest(self, 
                    strategy: Union[Callable, Any], 
                    config: Optional[Dict] = None,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    custom_exit: Optional[Callable] = None) -> Dict[str, Any]:
        """Run a backtest using the provided strategy.
        
        Args:
            strategy: Strategy function or class instance
            config: Optional strategy configuration
            stop_loss: Optional stop loss percentage
            take_profit: Optional take profit percentage
            custom_exit: Optional custom exit function
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        for i in range(len(self.data)):
            current_data = self.data.iloc[:i+1]
            current_prices = current_data.iloc[-1]
            
            # Generate signals
            if hasattr(strategy, 'generate_signals'):
                signals = strategy.generate_signals(current_data)
            else:
                signals = strategy(current_data)
            
            # Check for early exit conditions
            if self._check_exit_conditions(current_prices, stop_loss, take_profit, custom_exit):
                self._execute_trades({'exit': -1}, current_prices)
                break
            
            # Execute trades
            self._execute_trades(signals, current_prices)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.portfolio_values.append(portfolio_value)
            
            # Update asset values
            self._update_asset_values(current_prices)
        
        return self.get_performance_metrics()
    
    def _check_exit_conditions(self, 
                             current_prices: pd.Series,
                             stop_loss: Optional[float],
                             take_profit: Optional[float],
                             custom_exit: Optional[Callable]) -> bool:
        """Check if any exit conditions are met."""
        if not self.positions:
            return False
        
        # Check stop loss
        if stop_loss:
            for asset, quantity in self.positions.items():
                entry_price = self._get_entry_price(asset)
                if entry_price and current_prices[asset] < entry_price * (1 - stop_loss):
                    return True
        
        # Check take profit
        if take_profit:
            for asset, quantity in self.positions.items():
                entry_price = self._get_entry_price(asset)
                if entry_price and current_prices[asset] > entry_price * (1 + take_profit):
                    return True
        
        # Check custom exit
        if custom_exit:
            return custom_exit(self.positions, current_prices)
        
        return False
    
    def _get_entry_price(self, asset: str) -> Optional[float]:
        """Get the entry price for an asset."""
        for trade in reversed(self.trades):
            if trade.asset == asset and trade.type == TradeType.BUY:
                return trade.price
        return None
    
    def _execute_trades(self, signals: Dict[str, float], current_prices: pd.Series) -> None:
        """Execute trades based on the signals."""
        for asset, signal in signals.items():
            if asset == 'exit':
                # Close all positions
                for pos_asset, quantity in list(self.positions.items()):
                    if quantity > 0:
                        self._execute_single_trade(pos_asset, -quantity, current_prices[pos_asset])
                continue
            
            if signal > 0:  # Buy
                # Calculate maximum position size based on cash and leverage
                max_position = (self.cash * self.max_leverage) / current_prices[asset]
                quantity = min(signal * max_position, max_position)
                self._execute_single_trade(asset, quantity, current_prices[asset])
            
            elif signal < 0:  # Sell
                quantity = abs(signal) * self.positions.get(asset, 0)
                if quantity > 0:
                    self._execute_single_trade(asset, -quantity, current_prices[asset])
    
    def _execute_single_trade(self, asset: str, quantity: float, price: float) -> None:
        """Execute a single trade with slippage and transaction costs."""
        # Apply slippage
        if quantity > 0:  # Buy
            price *= (1 + self.slippage)
        else:  # Sell
            price *= (1 - self.slippage)
        
        # Apply spread
        price *= (1 + self.spread)
        
        # Calculate transaction cost
        cost = abs(quantity * price * self.transaction_cost)
        
        # Check if we have enough cash
        if quantity > 0 and (quantity * price + cost) > self.cash:
            return
        
        # Update positions and cash
        self.positions[asset] = self.positions.get(asset, 0) + quantity
        self.cash -= (quantity * price + cost)
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            asset=asset,
            quantity=quantity,
            price=price,
            type=TradeType.BUY if quantity > 0 else TradeType.SELL,
            slippage=self.slippage,
            transaction_cost=self.transaction_cost,
            spread=self.spread,
            cash_balance=self.cash,
            portfolio_value=self._calculate_portfolio_value(pd.Series({asset: price}))
        )
        self.trades.append(trade)
        
        # Update trade log
        self.trade_log.append({
            'timestamp': trade.timestamp,
            'asset': asset,
            'quantity': quantity,
            'price': price,
            'type': trade.type.value,
            'cost': cost,
            'cash_balance': self.cash,
            'portfolio_value': trade.portfolio_value
        })
    
    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calculate the current portfolio value."""
        total_value = self.cash
        for asset, quantity in self.positions.items():
            if asset in current_prices and not pd.isna(current_prices[asset]):
                total_value += quantity * current_prices[asset]
        return total_value
    
    def _update_asset_values(self, current_prices: pd.Series) -> None:
        """Update the time series of individual asset values."""
        for asset in self.asset_values:
            if asset in current_prices and not pd.isna(current_prices[asset]):
                value = self.positions.get(asset, 0) * current_prices[asset]
                self.asset_values[asset].append(value)
            else:
                self.asset_values[asset].append(self.asset_values[asset][-1])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        # Calculate basic metrics
        total_return = (self.portfolio_values[-1] / self.initial_cash) - 1
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Calculate ratios
        sharpe_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / returns.std() if returns.std() != 0 else 0
        sortino_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / downside_volatility if downside_volatility != 0 else 0
        
        # Calculate drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trades if t.quantity > 0 and t.price > self._get_entry_price(t.asset)]
        losing_trades = [t for t in self.trades if t.quantity > 0 and t.price <= self._get_entry_price(t.asset)]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        avg_trade = np.mean([t.price - self._get_entry_price(t.asset) for t in self.trades]) if self.trades else 0
        avg_win = np.mean([t.price - self._get_entry_price(t.asset) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.price - self._get_entry_price(t.asset) for t in losing_trades]) if losing_trades else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'returns': returns,
            'drawdowns': drawdowns,
            'trade_log': self.trade_log
        }
    
    def plot_results(self, use_plotly: bool = True) -> None:
        """Plot backtest results with interactive features."""
        if use_plotly:
            self._plot_plotly()
        else:
            self._plot_matplotlib()
    
    def _plot_plotly(self) -> None:
        """Create interactive Plotly visualization."""
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Portfolio Value', 'Drawdown', 'Asset Values'))
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(y=self.portfolio_values, name='Portfolio Value'),
            row=1, col=1
        )
        
        # Drawdown
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(y=drawdowns, name='Drawdown', fill='tozeroy'),
            row=2, col=1
        )
        
        # Asset values
        for asset, values in self.asset_values.items():
            fig.add_trace(
                go.Scatter(y=values, name=asset),
                row=3, col=1
            )
        
        # Add trade markers
        for trade in self.trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade.timestamp],
                    y=[trade.portfolio_value],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.type == TradeType.BUY else 'triangle-down',
                        size=10,
                        color='green' if trade.type == TradeType.BUY else 'red'
                    ),
                    name=f"{trade.type.value} {trade.asset}"
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True,
            height=900
        )
        
        fig.show()
    
    def _plot_matplotlib(self) -> None:
        """Create static Matplotlib visualization."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Portfolio value
        ax1.plot(self.portfolio_values)
        ax1.set_title('Portfolio Value')
        ax1.grid(True)
        
        # Drawdown
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.grid(True)
        
        # Asset values
        for asset, values in self.asset_values.items():
            ax3.plot(values, label=asset)
        ax3.set_title('Asset Values')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
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