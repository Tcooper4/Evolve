"""RSI optimizer with regime awareness and enhanced features."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading.risk.risk_metrics import calculate_regime_metrics
from trading.strategies.rsi_signals import generate_rsi_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading/optimization/logs/rsi_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RSIParameters:
    """Container for RSI parameters."""
    period: int
    overbought: float
    oversold: float
    stop_loss: float
    take_profit: float
    regime: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    parameters: RSIParameters
    returns: pd.Series
    metrics: Dict[str, float]
    signals: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series

class RSIOptimizer:
    """RSI strategy optimizer with regime awareness."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        slippage: float = 0.0001,
        transaction_cost: float = 0.0002,
        verbose: bool = False
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize RSI optimizer.
        
        Args:
            data: DataFrame with OHLCV data
            slippage: Slippage per trade (default: 0.01%)
            transaction_cost: Transaction cost per trade (default: 0.02%)
            verbose: Enable verbose logging
        """
        self.data = data
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.verbose = verbose
        self.regime_data = None
        self._detect_regimes()
    
    def _detect_regimes(self):
        """Detect market regimes in the data."""
        returns = self.data['close'].pct_change().dropna()
        self.regime_data = calculate_regime_metrics(returns)
        
        if self.verbose:
            logger.info(f"Detected regime: {self.regime_data['regime']}")
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def calculate_rsi(
        self,
        data: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate RSI indicator.
        
        Args:
            data: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {'success': True, 'result': rsi, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def calculate_returns(
        self,
        parameters: RSIParameters,
        regime_filter: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate strategy returns with slippage and costs.
        
        Args:
            parameters: RSI parameters
            regime_filter: Filter by market regime
            
        Returns:
            Tuple of (returns, signals, equity_curve)
        """
        # Generate signals using the strategy module
        signals = generate_rsi_signals(
            self.data,
            period=parameters.period,
            buy_threshold=parameters.oversold,
            sell_threshold=parameters.overbought
        )['signal']
        
        # Apply regime filter if specified
        if regime_filter and self.regime_data:
            regime_mask = self.regime_data['regime'] == regime_filter
            signals[~regime_mask] = 0
        
        # Calculate returns
        price_returns = self.data['close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        
        # Apply slippage and costs
        trade_mask = signals.diff().abs() > 0
        strategy_returns[trade_mask] -= (
            self.slippage + self.transaction_cost
        )
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        return {'success': True, 'result': strategy_returns, signals, equity_curve, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        signals: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            returns: Strategy returns
            signals: Trading signals
            equity_curve: Equity curve
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        total_return = equity_curve.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Win rate
        winning_trades = returns[returns > 0]
        total_trades = returns[returns != 0]
        win_rate = len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Signal confidence
        signal_changes = signals.diff().abs()
        recent_signals = signal_changes.rolling(20).sum()
        signal_confidence = 1 - (recent_signals / 20)
        
        return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'signal_confidence': signal_confidence.mean()
        }
    
    def optimize_rsi_parameters(
        self,
        objective: str = 'sharpe',
        n_top: int = 3,
        regime_filter: Optional[str] = None,
        param_ranges: Optional[Dict] = None
    ) -> List[OptimizationResult]:
        """Optimize RSI parameters.
        
        Args:
            objective: Optimization objective ('sharpe', 'win_rate', 'drawdown')
            n_top: Number of top parameter sets to return
            regime_filter: Filter by market regime
            param_ranges: Parameter ranges for optimization
            
        Returns:
            List of top N optimization results
        """
        if param_ranges is None:
            param_ranges = {
                'period': range(5, 31, 5),
                'overbought': np.arange(60, 81, 5),
                'oversold': np.arange(20, 41, 5),
                'stop_loss': np.arange(0.01, 0.06, 0.01),
                'take_profit': np.arange(0.02, 0.11, 0.02)
            }
        
        results = []
        
        for period in param_ranges['period']:
            for overbought in param_ranges['overbought']:
                for oversold in param_ranges['oversold']:
                    for stop_loss in param_ranges['stop_loss']:
                        for take_profit in param_ranges['take_profit']:
                            if oversold >= overbought:
                                continue
                            
                            parameters = RSIParameters(
                                period=period,
                                overbought=overbought,
                                oversold=oversold,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                regime=regime_filter
                            )
                            
                            returns, signals, equity_curve = self.calculate_returns(
                                parameters,
                                regime_filter
                            )
                            
                            metrics = self.calculate_metrics(
                                returns,
                                signals,
                                equity_curve
                            )
                            
                            results.append(OptimizationResult(
                                parameters=parameters,
                                returns=returns,
                                metrics=metrics,
                                signals=signals,
                                equity_curve=equity_curve,
                                drawdown=(equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()
                            ))
                            
                            if self.verbose:
                                logger.info(
                                    f"Tested parameters: {parameters}, "
                                    f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                                    f"Win Rate: {metrics['win_rate']:.2f}"
                                )
        
        # Sort results by objective
        if objective == 'sharpe':
            results.sort(key=lambda x: x.metrics['sharpe_ratio'], reverse=True)
        elif objective == 'win_rate':
            results.sort(key=lambda x: x.metrics['win_rate'], reverse=True)
        elif objective == 'drawdown':
            results.sort(key=lambda x: x.metrics['max_drawdown'])
        
        return {'success': True, 'result': results[:n_top], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_equity_curve(
        self,
        result: OptimizationResult,
        title: str = "Equity Curve"
    ) -> go.Figure:
        """Plot equity curve.
        
        Args:
            result: Optimization result
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            name="Equity Curve"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_drawdown(
        self,
        result: OptimizationResult,
        title: str = "Drawdown"
    ) -> go.Figure:
        """Plot drawdown.
        
        Args:
            result: Optimization result
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result.drawdown.index,
            y=result.drawdown,
            name="Drawdown",
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_signals(
        self,
        result: OptimizationResult,
        title: str = "RSI Signals"
    ) -> go.Figure:
        """Plot RSI signals.
        
        Args:
            result: Optimization result
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['close'],
            name="Price"
        ), row=1, col=1)
        
        # RSI
        rsi = self.calculate_rsi(
            self.data['close'],
            result.parameters.period
        )
        
        fig.add_trace(go.Scatter(
            x=rsi.index,
            y=rsi,
            name="RSI"
        ), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_hline(
            y=result.parameters.overbought,
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )
        
        fig.add_hline(
            y=result.parameters.oversold,
            line_dash="dash",
            line_color="green",
            row=2, col=1
        )
        
        # Add signals
        buy_signals = result.signals[result.signals == 1]
        sell_signals = result.signals[result.signals == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=self.data.loc[buy_signals.index, 'close'],
            mode='markers',
            name="Buy Signal",
            marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=self.data.loc[sell_signals.index, 'close'],
            mode='markers',
            name="Sell Signal",
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}