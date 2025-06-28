"""
Backtesting engine for trading strategies.

This module provides a comprehensive backtesting framework with support for:
- Multiple strategy backtesting
- Advanced position sizing (equal-weighted, risk-based, Kelly, fixed, volatility-adjusted, optimal f)
- Detailed trade logging and analysis
- Comprehensive performance metrics
- Advanced visualization capabilities
- Sophisticated risk management
"""

# Standard library imports
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import json
import csv
import math

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
import empyrical as ep
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import plotly.express as px
import plotly.io as pio
from plotly.offline import plot
import mplfinance as mpf
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import networkx as nx
from scipy.stats import norm, t, skew, kurtosis
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Try to import empyrical, with fallback
try:
    import empyrical as ep
    EMPYRICIAL_AVAILABLE = True
except ImportError:
    EMPYRICIAL_AVAILABLE = False
    # Create a simple fallback for empyrical functions
    class EmpyricalFallback:
        @staticmethod
        def sharpe_ratio(returns, risk_free=0.0, period=252):
            """Simple Sharpe ratio calculation."""
            excess_returns = returns - risk_free
            if len(excess_returns) == 0 or excess_returns.std() == 0:
                return 0.0
            return (excess_returns.mean() * period) / (excess_returns.std() * np.sqrt(period))
        
        @staticmethod
        def sortino_ratio(returns, risk_free=0.0, period=252):
            """Simple Sortino ratio calculation."""
            excess_returns = returns - risk_free
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return (excess_returns.mean() * period) / (downside_returns.std() * np.sqrt(period))
        
        @staticmethod
        def max_drawdown(returns):
            """Simple max drawdown calculation."""
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        @staticmethod
        def calmar_ratio(returns, risk_free=0.0, period=252):
            """Simple Calmar ratio calculation."""
            max_dd = EmpyricalFallback.max_drawdown(returns)
            if max_dd == 0:
                return 0.0
            annual_return = (1 + returns.mean()) ** period - 1
            return annual_return / abs(max_dd)
    
    ep = EmpyricalFallback()

from scipy.optimize import minimize

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SLIPPAGE = 0.001  # 0.1%
DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
DEFAULT_SPREAD = 0.0005  # 0.05%
MAX_POSITION_SIZE = 0.25  # Maximum 25% of portfolio in single position
MIN_POSITION_SIZE = 0.01  # Minimum 1% of portfolio in single position

class PositionSizing(Enum):
    """Position sizing methods."""
    EQUAL_WEIGHTED = "equal_weighted"
    RISK_BASED = "risk_based"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_SIZE = "fixed_size"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"
    HALF_KELLY = "half_kelly"
    DYNAMIC_KELLY = "dynamic_kelly"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    MOMENTUM_WEIGHTED = "momentum_weighted"
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    RISK_EFFICIENT = "risk_efficient"
    ADAPTIVE_WEIGHT = "adaptive_weight"
    REGIME_BASED = "regime_based"
    FACTOR_BASED = "factor_based"
    MACHINE_LEARNING = "machine_learning"

class RiskMetric(Enum):
    """Risk metrics for analysis."""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    CALMAR = "calmar_ratio"
    OMEGA = "omega_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    DRAWDOWN_DURATION = "drawdown_duration"
    TREYNOR = "treynor_ratio"
    INFORMATION = "information_ratio"
    JENSEN_ALPHA = "jensen_alpha"
    ULGER = "ulger_ratio"
    MODIGLIANI = "modigliani_ratio"
    BURKE = "burke_ratio"
    STERLING = "sterling_ratio"
    KAPPA = "kappa_ratio"
    GINI = "gini_coefficient"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    VAR_RATIO = "var_ratio"
    CONDITIONAL_SHARPE = "conditional_sharpe"
    TAIL_RATIO = "tail_ratio"
    PAIN_RATIO = "pain_ratio"
    GAIN_LOSS_RATIO = "gain_loss_ratio"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    RECOVERY_FACTOR = "recovery_factor"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    OMEGA_SHARPE = "omega_sharpe"
    CONDITIONAL_VAR = "conditional_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    SEMI_VARIANCE = "semi_variance"
    DOWNSIDE_DEVIATION = "downside_deviation"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_DRAWDOWN = "conditional_drawdown"
    REGIME_RISK = "regime_risk"
    FACTOR_RISK = "factor_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    LEVERAGE_RISK = "leverage_risk"
    CURRENCY_RISK = "currency_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    INFLATION_RISK = "inflation_risk"
    POLITICAL_RISK = "political_risk"
    SYSTEMIC_RISK = "systemic_risk"
    IDIOSYNCRATIC_RISK = "idiosyncratic_risk"

class TradeType(Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"

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
    strategy: str
    position_size: float
    risk_metrics: Dict[str, float]
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    holding_period: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 initial_cash: float = 100000.0,
                 slippage: float = DEFAULT_SLIPPAGE,
                 transaction_cost: float = DEFAULT_TRANSACTION_COST,
                 spread: float = DEFAULT_SPREAD,
                 max_leverage: float = 1.0,
                 max_trades: int = 10000,
                 trade_log_path: Optional[str] = None,
                 position_sizing: PositionSizing = PositionSizing.EQUAL_WEIGHTED,
                 risk_per_trade: float = 0.02,
                 risk_free_rate: float = 0.02,
                 benchmark: Optional[pd.Series] = None):
        """Initialize backtester."""
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.spread = spread
        self.max_leverage = max_leverage
        self.max_trades = max_trades
        self.trade_log_path = trade_log_path
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark
        
        # Initialize state
        self.portfolio_values = np.zeros(len(data))
        self.trades = []
        self.trade_log = []
        self.positions = {}
        self.asset_values = {}
        self.strategy_results = {}
        self.risk_metrics = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def _calculate_position_size(self, 
                               asset: str,
                               price: float,
                               strategy: str,
                               signal: float) -> float:
        """Calculate position size based on selected method."""
        sizing_methods = {
            PositionSizing.EQUAL_WEIGHTED: self._calculate_equal_weighted_size,
            PositionSizing.RISK_BASED: self._calculate_risk_based_size,
            PositionSizing.KELLY_CRITERION: self._calculate_kelly_size,
            PositionSizing.FIXED_SIZE: self._calculate_fixed_size,
            PositionSizing.VOLATILITY_ADJUSTED: self._calculate_volatility_adjusted_size,
            PositionSizing.OPTIMAL_F: self._calculate_optimal_f_size,
            PositionSizing.RISK_PARITY: self._calculate_risk_parity_size,
            PositionSizing.BLACK_LITTERMAN: self._calculate_black_litterman_size,
            PositionSizing.MARTINGALE: self._calculate_martingale_size,
            PositionSizing.ANTI_MARTINGALE: self._calculate_anti_martingale_size,
            PositionSizing.HALF_KELLY: self._calculate_half_kelly_size,
            PositionSizing.DYNAMIC_KELLY: self._calculate_dynamic_kelly_size,
            PositionSizing.CORRELATION_ADJUSTED: self._calculate_correlation_adjusted_size,
            PositionSizing.MOMENTUM_WEIGHTED: self._calculate_momentum_weighted_size,
            PositionSizing.MEAN_VARIANCE: self._calculate_mean_variance_size,
            PositionSizing.MINIMUM_VARIANCE: self._calculate_minimum_variance_size,
            PositionSizing.MAXIMUM_DIVERSIFICATION: self._calculate_maximum_diversification_size,
            PositionSizing.RISK_EFFICIENT: self._calculate_risk_efficient_size,
            PositionSizing.ADAPTIVE_WEIGHT: self._calculate_adaptive_weight_size,
            PositionSizing.REGIME_BASED: self._calculate_regime_based_size,
            PositionSizing.FACTOR_BASED: self._calculate_factor_based_size,
            PositionSizing.MACHINE_LEARNING: self._calculate_machine_learning_size
        }
        
        sizing_method = sizing_methods.get(self.position_sizing)
        if not sizing_method:
            raise ValueError(f"Unknown position sizing method: {self.position_sizing}")
            
        position_size = sizing_method(asset, price, strategy, signal)
        
        # Apply position size limits
        return max(min(position_size, MAX_POSITION_SIZE), MIN_POSITION_SIZE)
        
    def _calculate_volatility_adjusted_size(self, 
                                          asset: str,
                                          price: float,
                                          strategy: str,
                                          signal: float) -> float:
        """Calculate volatility-adjusted position size."""
        # Calculate rolling volatility
        returns = self.data[asset].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Adjust position size inversely to volatility
        base_size = self._calculate_equal_weighted_size(price)
        return base_size * (1 / (1 + volatility))
        
    def _calculate_optimal_f_size(self, 
                                asset: str,
                                price: float,
                                strategy: str,
                                signal: float) -> float:
        """Calculate position size using Optimal f."""
        # Calculate win/loss ratio and win rate
        trades = [t for t in self.trades if t.asset == asset]
        if not trades:
            return self._calculate_equal_weighted_size(price)
            
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        if not winning_trades or not losing_trades:
            return self._calculate_equal_weighted_size(price)
            
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t.pnl for t in winning_trades])
        avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
        
        # Calculate Optimal f
        f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return f * self.cash / price
        
    def _calculate_risk_parity_size(self, 
                                  asset: str,
                                  price: float,
                                  strategy: str,
                                  signal: float) -> float:
        """Calculate position size using Risk Parity approach."""
        # Calculate risk contribution
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate risk contribution
        risk_contrib = np.sqrt(np.diag(cov_matrix))
        total_risk = np.sum(risk_contrib)
        
        # Calculate target weights
        target_weights = risk_contrib / total_risk
        
        # Calculate position size
        return target_weights[asset] * self.cash / price
        
    def _calculate_black_litterman_size(self, 
                                      asset: str,
                                      price: float,
                                      strategy: str,
                                      signal: float) -> float:
        """Calculate position size using Black-Litterman model."""
        # Calculate market equilibrium returns
        returns = self.data[asset].pct_change().dropna()
        market_return = returns.mean()
        market_risk = returns.std()
        
        # Calculate investor views
        view_return = signal * market_return
        view_confidence = abs(signal)
        
        # Calculate Black-Litterman weights
        tau = 0.05  # Prior uncertainty
        omega = np.diag([1/view_confidence])  # View uncertainty
        
        # Calculate posterior returns and weights
        prior_return = market_return
        prior_cov = market_risk ** 2
        
        post_return = (prior_return + tau * view_return) / (1 + tau)
        post_cov = prior_cov * (1 + tau)
        
        # Calculate position size
        position_size = (post_return - self.risk_free_rate) / (post_cov * self.risk_per_trade)
        return position_size * self.cash / price
        
    def _calculate_martingale_size(self, 
                                 asset: str,
                                 price: float,
                                 strategy: str,
                                 signal: float) -> float:
        """Calculate position size using Martingale strategy."""
        # Get recent trades for this asset
        trades = [t for t in self.trades if t.asset == asset]
        if not trades:
            return self._calculate_equal_weighted_size(price)
        
        # Double position size after losses
        last_trade = trades[-1]
        if last_trade.pnl and last_trade.pnl < 0:
            return last_trade.position_size * 2
        return self._calculate_equal_weighted_size(price)
        
    def _calculate_anti_martingale_size(self, 
                                      asset: str,
                                      price: float,
                                      strategy: str,
                                      signal: float) -> float:
        """Calculate position size using Anti-Martingale strategy."""
        # Get recent trades for this asset
        trades = [t for t in self.trades if t.asset == asset]
        if not trades:
            return self._calculate_equal_weighted_size(price)
        
        # Double position size after wins
        last_trade = trades[-1]
        if last_trade.pnl and last_trade.pnl > 0:
            return last_trade.position_size * 2
        return self._calculate_equal_weighted_size(price)
        
    def _calculate_half_kelly_size(self, 
                                 asset: str,
                                 price: float,
                                 strategy: str,
                                 signal: float) -> float:
        """Calculate position size using Half-Kelly strategy."""
        kelly_size = self._calculate_kelly_size(asset, price, strategy, signal)
        return kelly_size * 0.5
        
    def _calculate_dynamic_kelly_size(self, 
                                    asset: str,
                                    price: float,
                                    strategy: str,
                                    signal: float) -> float:
        """Calculate position size using Dynamic Kelly strategy."""
        # Calculate Kelly fraction
        kelly_size = self._calculate_kelly_size(asset, price, strategy, signal)
        
        # Adjust based on market conditions
        returns = self.data[asset].pct_change().dropna()
        volatility = returns.std()
        trend_strength = abs(returns.rolling(window=20).mean().iloc[-1])
        
        # Reduce position size in high volatility or weak trend
        adjustment = 1 / (1 + volatility) * (1 + trend_strength)
        return kelly_size * adjustment
        
    def _calculate_correlation_adjusted_size(self, 
                                           asset: str,
                                           price: float,
                                           strategy: str,
                                           signal: float) -> float:
        """Calculate position size adjusted for correlation with existing positions."""
        # Calculate correlations with existing positions
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate average correlation
        correlations = returns.corr()[asset].drop(asset)
        avg_correlation = correlations.mean()
        
        # Adjust position size inversely to correlation
        base_size = self._calculate_equal_weighted_size(price)
        return base_size * (1 - abs(avg_correlation))
        
    def _calculate_momentum_weighted_size(self, 
                                        asset: str,
                                        price: float,
                                        strategy: str,
                                        signal: float) -> float:
        """Calculate position size weighted by momentum strength."""
        # Calculate momentum indicators
        returns = self.data[asset].pct_change().dropna()
        momentum = returns.rolling(window=20).mean().iloc[-1]
        rsi = self._calculate_rsi(returns)
        
        # Combine momentum signals
        momentum_score = (momentum + (rsi - 50) / 50) / 2
        
        # Adjust position size based on momentum
        base_size = self._calculate_equal_weighted_size(price)
        return base_size * (1 + momentum_score)
        
    def _calculate_rsi(self, returns: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
        
    def _calculate_mean_variance_size(self, 
                                    asset: str,
                                    price: float,
                                    strategy: str,
                                    signal: float) -> float:
        """Calculate position size using Mean-Variance optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Calculate optimal weights using mean-variance optimization
        n_assets = len(returns.columns)
        weights = np.array([1/n_assets] * n_assets)  # Initial guess
        
        def portfolio_stats(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = portfolio_return / portfolio_vol
            return -sharpe_ratio  # Minimize negative Sharpe ratio
        
        # Optimize weights
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(portfolio_stats, weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Get weight for the current asset
        asset_idx = returns.columns.get_loc(asset)
        return result.x[asset_idx]

    def _calculate_minimum_variance_size(self, 
                                       asset: str,
                                       price: float,
                                       strategy: str,
                                       signal: float) -> float:
        """Calculate position size using Minimum Variance optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate optimal weights using minimum variance optimization
        n_assets = len(returns.columns)
        weights = np.array([1/n_assets] * n_assets)  # Initial guess
        
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        # Optimize weights
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(portfolio_variance, weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Get weight for the current asset
        asset_idx = returns.columns.get_loc(asset)
        return result.x[asset_idx]

    def _calculate_maximum_diversification_size(self, 
                                              asset: str,
                                              price: float,
                                              strategy: str,
                                              signal: float) -> float:
        """Calculate position size using Maximum Diversification optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate volatility and correlation matrix
        vol = returns.std()
        corr_matrix = returns.corr()
        
        # Calculate diversification ratio
        def diversification_ratio(weights):
            portfolio_vol = np.sqrt(weights.T @ (corr_matrix * np.outer(vol, vol)) @ weights)
            weighted_vol = np.sum(weights * vol)
            return -weighted_vol / portfolio_vol  # Minimize negative ratio
        
        # Optimize weights
        n_assets = len(returns.columns)
        weights = np.array([1/n_assets] * n_assets)  # Initial guess
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(diversification_ratio, weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Get weight for the current asset
        asset_idx = returns.columns.get_loc(asset)
        return result.x[asset_idx]

    def _calculate_risk_efficient_size(self, 
                                     asset: str,
                                     price: float,
                                     strategy: str,
                                     signal: float) -> float:
        """Calculate position size using Risk-Efficient optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate risk metrics
        vol = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Calculate risk efficiency score
        risk_score = vol * (1 + abs(skew)) * (1 + kurt/3)
        
        # Calculate weights inversely proportional to risk score
        weights = 1 / risk_score
        weights = weights / weights.sum()
        
        # Get weight for the current asset
        return weights[asset]

    def _calculate_adaptive_weight_size(self, 
                                      asset: str,
                                      price: float,
                                      strategy: str,
                                      signal: float) -> float:
        """Calculate position size using Adaptive Weight optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Calculate adaptive metrics
        vol = returns.std()
        momentum = returns.rolling(window=20).mean().iloc[-1]
        trend = returns.rolling(window=50).mean().iloc[-1]
        
        # Calculate adaptive score
        adaptive_score = (momentum + trend) / (2 * vol)
        
        # Calculate weights based on adaptive score
        weights = adaptive_score / adaptive_score.sum()
        
        # Get weight for the current asset
        return weights[asset]

    def _calculate_regime_based_size(self, 
                                   asset: str,
                                   price: float,
                                   strategy: str,
                                   signal: float) -> float:
        """Calculate position size using Regime-Based optimization."""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        for pos_asset in self.positions:
            returns[pos_asset] = self.data[pos_asset].pct_change()
        returns[asset] = self.data[asset].pct_change()
        
        # Detect market regime
        vol = returns.std()
        trend = returns.rolling(window=50).mean().iloc[-1]
        momentum = returns.rolling(window=20).mean().iloc[-1]
        
        # Define regime based on market conditions
        if vol[asset] > vol.mean() * 1.5:
            regime = 'high_volatility'
        elif trend[asset] > 0 and momentum[asset] > 0:
            regime = 'bullish'
        elif trend[asset] < 0 and momentum[asset] < 0:
            regime = 'bearish'
        else:
            regime = 'neutral'
        
        # Adjust position size based on regime
        base_size = self._calculate_equal_weighted_size(price)
        regime_multipliers = {
            'high_volatility': 0.5,
            'bullish': 1.2,
            'bearish': 0.8,
            'neutral': 1.0
        }
        
        return base_size * regime_multipliers[regime]

    def _calculate_factor_based_size(self, 
                                   asset: str,
                                   price: float,
                                   strategy: str,
                                   signal: float) -> float:
        """Calculate position size using Factor-Based optimization."""
        # Calculate factor exposures
        returns = self.data[asset].pct_change().dropna()
        
        # Calculate common factors
        market_factor = returns.rolling(window=20).mean()
        volatility_factor = returns.rolling(window=20).std()
        momentum_factor = returns.rolling(window=20).sum()
        value_factor = self._calculate_value_factor(asset)
        quality_factor = self._calculate_quality_factor(asset)
        
        # Combine factors
        factor_score = (
            0.3 * market_factor.iloc[-1] +
            0.2 * volatility_factor.iloc[-1] +
            0.2 * momentum_factor.iloc[-1] +
            0.15 * value_factor +
            0.15 * quality_factor
        )
        
        # Calculate position size based on factor score
        base_size = self._calculate_equal_weighted_size(price)
        return base_size * (1 + factor_score)

    def _calculate_value_factor(self, asset: str) -> float:
        """Calculate value factor for an asset."""
        # Implement value factor calculation
        # This is a placeholder - implement actual value metrics
        return 0.0

    def _calculate_quality_factor(self, asset: str) -> float:
        """Calculate quality factor for an asset."""
        # Implement quality factor calculation
        # This is a placeholder - implement actual quality metrics
        return 0.0

    def _calculate_machine_learning_size(self, 
                                      asset: str,
                                      price: float,
                                      strategy: str,
                                      signal: float) -> float:
        """Calculate position size using Machine Learning optimization."""
        # Prepare features
        returns = self.data[asset].pct_change().dropna()
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(window=20).std(),
            'momentum': returns.rolling(window=20).mean(),
            'rsi': self._calculate_rsi(returns),
            'macd': self._calculate_macd(returns),
            'bollinger': self._calculate_bollinger_bands(returns)
        })
        
        # Prepare target (future returns)
        target = returns.shift(-1).dropna()
        
        # Align features and target
        features = features.iloc[:-1]
        target = target.iloc[:-1]
        
        # Train model (placeholder - implement actual ML model)
        # This is a simplified example using linear regression
        model = LinearRegression()
        model.fit(features, target)
        
        # Predict optimal position size
        current_features = features.iloc[-1:].values
        predicted_return = model.predict(current_features)[0]
        
        # Calculate position size based on prediction
        base_size = self._calculate_equal_weighted_size(price)
        return base_size * (1 + predicted_return)

    def _calculate_macd(self, returns: pd.Series) -> float:
        """Calculate MACD indicator."""
        exp1 = returns.ewm(span=12, adjust=False).mean()
        exp2 = returns.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd.iloc[-1]

    def _calculate_bollinger_bands(self, returns: pd.Series) -> float:
        """Calculate Bollinger Bands indicator."""
        sma = returns.rolling(window=20).mean()
        std = returns.rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return (returns.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])

    def _calculate_stochastic(self, returns: pd.Series) -> float:
        """Calculate Stochastic Oscillator."""
        high = returns.rolling(window=14).max()
        low = returns.rolling(window=14).min()
        close = returns
        k = 100 * (close - low) / (high - low)
        d = k.rolling(window=3).mean()
        return d.iloc[-1]

    def _calculate_adx(self, returns: pd.Series) -> float:
        """Calculate Average Directional Index."""
        high = returns.rolling(window=14).max()
        low = returns.rolling(window=14).min()
        close = returns
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        atr = tr.rolling(window=14).mean()
        return atr.iloc[-1]

    def _calculate_cci(self, returns: pd.Series) -> float:
        """Calculate Commodity Channel Index."""
        tp = returns.rolling(window=20).mean()
        md = returns.rolling(window=20).std()
        cci = (returns - tp) / (0.015 * md)
        return cci.iloc[-1]

    def _calculate_mfi(self, returns: pd.Series) -> float:
        """Calculate Money Flow Index."""
        # This is a simplified version - implement full MFI calculation
        return 0.0

    def _calculate_obv(self, returns: pd.Series) -> float:
        """Calculate On-Balance Volume."""
        # This is a simplified version - implement full OBV calculation
        return 0.0

    def _calculate_technical_indicators(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive technical indicators."""
        # Convert returns to price series
        prices = (1 + returns).cumprod()
        
        # Calculate Ichimoku Cloud
        ichimoku = ta.ichimoku(prices)
        tenkan = ichimoku['ITS_9']
        kijun = ichimoku['IKS_26']
        senkou_span_a = ichimoku['ISA_9']
        senkou_span_b = ichimoku['ISB_26']
        
        # Calculate Fibonacci levels
        high = prices.max()
        low = prices.min()
        diff = high - low
        fib_levels = {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
        
        # Calculate Elliott Wave
        wave_points = self._calculate_elliott_wave(prices)
        
        # Calculate Keltner Channels
        keltner = ta.kc(prices)
        keltner_upper = keltner['KCUe_20_2']
        keltner_middle = keltner['KCBe_20_2']
        keltner_lower = keltner['KCLe_20_2']
        
        # Calculate Parabolic SAR
        psar = ta.psar(prices)
        psar_up = psar['PSARl_0.02_0.2']
        psar_down = psar['PSARs_0.02_0.2']
        
        # Calculate additional indicators
        macd = self._calculate_macd(returns)
        rsi = self._calculate_rsi(returns)
        bollinger = self._calculate_bollinger_bands(returns)
        stoch = self._calculate_stochastic(returns)
        adx = self._calculate_adx(returns)
        cci = self._calculate_cci(returns)
        mfi = self._calculate_mfi(returns)
        obv = self._calculate_obv(returns)
        
        # Calculate additional advanced indicators
        williams_r = self._calculate_williams_r(returns)
        roc = self._calculate_rate_of_change(returns)
        cmf = self._calculate_chaikin_money_flow(returns)
        vwap = self._calculate_vwap(returns)
        atr = self._calculate_average_true_range(returns)
        
        return {
            'ichimoku': {
                'tenkan': tenkan.iloc[-1],
                'kijun': kijun.iloc[-1],
                'senkou_span_a': senkou_span_a.iloc[-1],
                'senkou_span_b': senkou_span_b.iloc[-1]
            },
            'fibonacci': fib_levels,
            'elliott_wave': wave_points,
            'keltner': {
                'upper': keltner_upper.iloc[-1],
                'middle': keltner_middle.iloc[-1],
                'lower': keltner_lower.iloc[-1]
            },
            'parabolic_sar': {
                'up': psar_up.iloc[-1],
                'down': psar_down.iloc[-1]
            },
            'macd': macd,
            'rsi': rsi,
            'bollinger': bollinger,
            'stochastic': stoch,
            'adx': adx,
            'cci': cci,
            'mfi': mfi,
            'obv': obv,
            'williams_r': williams_r,
            'roc': roc,
            'cmf': cmf,
            'vwap': vwap,
            'atr': atr
        }

    def _calculate_williams_r(self, returns: pd.Series) -> float:
        """Calculate Williams %R."""
        high = returns.rolling(window=14).max()
        low = returns.rolling(window=14).min()
        close = returns
        wr = -100 * (high - close) / (high - low)
        return wr.iloc[-1]

    def _calculate_rate_of_change(self, returns: pd.Series) -> float:
        """Calculate Rate of Change."""
        roc = returns.pct_change(periods=10) * 100
        return roc.iloc[-1]

    def _calculate_chaikin_money_flow(self, returns: pd.Series) -> float:
        """Calculate Chaikin Money Flow."""
        # This is a simplified version - implement full CMF calculation
        return 0.0

    def _calculate_vwap(self, returns: pd.Series) -> float:
        """Calculate Volume Weighted Average Price."""
        # This is a simplified version - implement full VWAP calculation
        return 0.0

    def _calculate_average_true_range(self, returns: pd.Series) -> float:
        """Calculate Average True Range."""
        high = returns.rolling(window=14).max()
        low = returns.rolling(window=14).min()
        close = returns
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        return atr.iloc[-1]

    def _calculate_elliott_wave(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate Elliott Wave points."""
        # This is a simplified version - implement full Elliott Wave analysis
        # using proper wave identification and rules
        high = prices.max()
        low = prices.min()
        current = prices.iloc[-1]
        
        # Identify potential wave points
        wave_points = {
            'wave1': low,
            'wave2': low + (high - low) * 0.382,
            'wave3': low + (high - low) * 0.618,
            'wave4': low + (high - low) * 0.786,
            'wave5': high
        }
        
        return wave_points

    def _calculate_risk_metrics(self, 
                              asset: str, 
                              price: float, 
                              quantity: float) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for a trade."""
        # Calculate basic metrics
        returns = self.data[asset].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calculate position metrics
        position_value = abs(quantity * price)
        portfolio_value = self._calculate_portfolio_value(pd.Series({asset: price}))
        position_weight = position_value / portfolio_value
        
        # Calculate correlation with benchmark
        if self.benchmark is not None:
            benchmark_returns = self.benchmark.pct_change().dropna()
            correlation = returns.corr(benchmark_returns)
            beta = returns.cov(benchmark_returns) / benchmark_returns.var()
        else:
            correlation = 0.0
            beta = 0.0
            
        # Calculate drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns.mean()
        
        # Calculate additional risk metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        var_ratio = returns.var() / returns.rolling(window=20).var().mean()
        
        # Calculate conditional metrics
        up_market = returns[returns > 0]
        down_market = returns[returns < 0]
        conditional_sharpe = up_market.mean() / up_market.std() if len(up_market) > 0 else 0
        
        # Calculate tail metrics
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05))
        
        # Calculate regime risk
        regime_risk = self._calculate_regime_risk(returns)
        
        # Calculate factor risk
        factor_risk = self._calculate_factor_risk(returns)
        
        # Calculate liquidity risk
        liquidity_risk = self._calculate_liquidity_risk(asset)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(asset)
        
        # Calculate leverage risk
        leverage_risk = self._calculate_leverage_risk(asset)
        
        # Calculate currency risk
        currency_risk = self._calculate_currency_risk(asset)
        
        # Calculate interest rate risk
        interest_rate_risk = self._calculate_interest_rate_risk(asset)
        
        # Calculate inflation risk
        inflation_risk = self._calculate_inflation_risk(asset)
        
        # Calculate political risk
        political_risk = self._calculate_political_risk(asset)
        
        # Calculate systemic risk
        systemic_risk = self._calculate_systemic_risk(returns)
        
        # Calculate idiosyncratic risk
        idiosyncratic_risk = self._calculate_idiosyncratic_risk(returns)
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'position_value': position_value,
            'position_weight': position_weight,
            'correlation': correlation,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_ratio': var_ratio,
            'conditional_sharpe': conditional_sharpe,
            'tail_ratio': tail_ratio,
            'regime_risk': regime_risk,
            'factor_risk': factor_risk,
            'liquidity_risk': liquidity_risk,
            'concentration_risk': concentration_risk,
            'leverage_risk': leverage_risk,
            'currency_risk': currency_risk,
            'interest_rate_risk': interest_rate_risk,
            'inflation_risk': inflation_risk,
            'political_risk': political_risk,
            'systemic_risk': systemic_risk,
            'idiosyncratic_risk': idiosyncratic_risk
        }

    def _calculate_regime_risk(self, returns: pd.Series) -> float:
        """Calculate regime risk based on market conditions."""
        # Calculate volatility regime
        vol = returns.std()
        vol_regime = 1 if vol > returns.std() * 1.5 else 0
        
        # Calculate trend regime
        trend = returns.rolling(window=50).mean().iloc[-1]
        trend_regime = 1 if trend > 0 else 0
        
        # Calculate momentum regime
        momentum = returns.rolling(window=20).mean().iloc[-1]
        momentum_regime = 1 if momentum > 0 else 0
        
        # Calculate regime risk score
        regime_risk = (vol_regime * 0.4 + trend_regime * 0.3 + momentum_regime * 0.3)
        
        return regime_risk

    def _calculate_factor_risk(self, returns: pd.Series) -> float:
        """Calculate factor risk based on factor exposures."""
        # Calculate factor exposures
        market_factor = returns.rolling(window=20).mean().iloc[-1]
        size_factor = returns.rolling(window=20).std().iloc[-1]
        value_factor = self._calculate_value_factor(returns.name)
        momentum_factor = returns.rolling(window=20).sum().iloc[-1]
        volatility_factor = returns.std()
        
        # Calculate factor risk score
        factor_risk = (
            abs(market_factor) * 0.3 +
            abs(size_factor) * 0.2 +
            abs(value_factor) * 0.2 +
            abs(momentum_factor) * 0.15 +
            abs(volatility_factor) * 0.15
        )
        
        return factor_risk

    def _calculate_liquidity_risk(self, asset: str) -> float:
        """Calculate liquidity risk for an asset."""
        # Calculate trading volume
        volume = self.data[asset].rolling(window=20).std()
        
        # Calculate bid-ask spread (simplified)
        spread = self.data[asset].rolling(window=20).std() * 0.001
        
        # Calculate liquidity risk score
        liquidity_risk = 1 / (volume.iloc[-1] * (1 - spread.iloc[-1]))
        
        return liquidity_risk

    def _calculate_concentration_risk(self, asset: str) -> float:
        """Calculate concentration risk for an asset."""
        # Calculate position weight
        position_value = abs(self.positions.get(asset, 0) * self.data[asset].iloc[-1])
        portfolio_value = self._calculate_portfolio_value(self.data.iloc[-1])
        weight = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate concentration risk score
        concentration_risk = weight ** 2
        
        return concentration_risk

    def _calculate_leverage_risk(self, asset: str) -> float:
        """Calculate leverage risk for an asset."""
        # Calculate leverage ratio
        position_value = abs(self.positions.get(asset, 0) * self.data[asset].iloc[-1])
        equity = self.cash
        leverage = position_value / equity if equity > 0 else 0
        
        # Calculate leverage risk score
        leverage_risk = min(leverage / 2, 1)
        
        return leverage_risk

    def _calculate_currency_risk(self, asset: str) -> float:
        """Calculate currency risk for an asset."""
        # This is a placeholder - implement actual currency risk calculation
        # using exchange rate data and currency exposure
        return 0.0

    def _calculate_interest_rate_risk(self, asset: str) -> float:
        """Calculate interest rate risk for an asset."""
        # This is a placeholder - implement actual interest rate risk calculation
        # using yield curve data and duration
        return 0.0

    def _calculate_inflation_risk(self, asset: str) -> float:
        """Calculate inflation risk for an asset."""
        # This is a placeholder - implement actual inflation risk calculation
        # using inflation data and sensitivity
        return 0.0

    def _calculate_political_risk(self, asset: str) -> float:
        """Calculate political risk for an asset."""
        # This is a placeholder - implement actual political risk calculation
        # using political stability indices and country exposure
        return 0.0

    def _calculate_systemic_risk(self, returns: pd.Series) -> float:
        """Calculate systemic risk based on market conditions."""
        # Calculate correlation with market
        if self.benchmark is not None:
            market_returns = self.benchmark.pct_change().dropna()
            correlation = returns.corr(market_returns)
        else:
            correlation = 0
        
        # Calculate beta
        if self.benchmark is not None:
            beta = returns.cov(market_returns) / market_returns.var()
        else:
            beta = 0
        
        # Calculate systemic risk score
        systemic_risk = (abs(correlation) * 0.5 + abs(beta) * 0.5)
        
        return systemic_risk

    def _calculate_idiosyncratic_risk(self, returns: pd.Series) -> float:
        """Calculate idiosyncratic risk for an asset."""
        # Calculate total risk
        total_risk = returns.std()
        
        # Calculate systematic risk
        if self.benchmark is not None:
            market_returns = self.benchmark.pct_change().dropna()
            beta = returns.cov(market_returns) / market_returns.var()
            systematic_risk = beta * market_returns.std()
        else:
            systematic_risk = 0
        
        # Calculate idiosyncratic risk
        idiosyncratic_risk = np.sqrt(total_risk ** 2 - systematic_risk ** 2)
        
        return idiosyncratic_risk

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
        calmar_ratio = annual_return / abs(self.get_performance_metrics()['max_drawdown']) if self.get_performance_metrics()['max_drawdown'] != 0 else 0
        
        # Calculate drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns.mean()
        drawdown_duration = (drawdowns < 0).astype(int).groupby((drawdowns < 0).astype(int).cumsum()).cumsum().max()
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        avg_trade = np.mean([t.pnl for t in self.trades if t.pnl]) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate position metrics
        position_metrics = self._calculate_position_metrics()
        
        # Calculate benchmark metrics if available
        benchmark_metrics = self._calculate_benchmark_metrics() if self.benchmark is not None else {}
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration': drawdown_duration,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'position_metrics': position_metrics,
            'benchmark_metrics': benchmark_metrics,
            'trades': self._get_trade_summary(),
            'equity_curve': self.portfolio_values.tolist()
        }
        
    def _calculate_benchmark_metrics(self) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        if self.benchmark is None:
            return {}
            
        strategy_returns = pd.Series(self.portfolio_values).pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()
        
        # Calculate relative performance
        relative_return = strategy_returns.mean() - benchmark_returns.mean()
        tracking_error = (strategy_returns - benchmark_returns).std()
        information_ratio = relative_return / tracking_error if tracking_error != 0 else 0
        
        # Calculate beta and alpha
        beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        
        # Calculate correlation
        correlation = strategy_returns.corr(benchmark_returns)
        
        return {
            'relative_return': relative_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation
        }
        
    def plot_results(self, use_plotly: bool = True) -> None:
        """Plot comprehensive backtest results."""
        if use_plotly:
            self._plot_plotly()
        else:
            self._plot_matplotlib()
            
    def _plot_plotly(self) -> None:
        """Create interactive Plotly visualization."""
        # Create subplot figure
        fig = make_subplots(
            rows=7, cols=2,
            subplot_titles=(
                'Portfolio Value',
                'Returns Distribution',
                'Drawdown',
                'Trade P&L',
                'Position Weights',
                'Risk Metrics',
                'Benchmark Comparison',
                'Trade Analysis',
                'Risk Decomposition',
                'Performance Attribution',
                'Technical Indicators',
                'ML Predictions',
                'Correlation Heatmap',
                'Network Graph'
            )
        )
        
        # Add portfolio value
        fig.add_trace(
            go.Scatter(
                y=self.portfolio_values,
                name='Portfolio Value'
            ),
            row=1, col=1
        )
        
        # Add returns distribution
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns Distribution'
            ),
            row=1, col=2
        )
        
        # Add drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(
                y=drawdowns,
                name='Drawdown',
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add trade P&L
        trade_pnl = [t.pnl for t in self.trades if t.pnl]
        fig.add_trace(
            go.Scatter(
                y=trade_pnl,
                name='Trade P&L'
            ),
            row=2, col=2
        )
        
        # Add position weights
        position_metrics = self._calculate_position_metrics()
        if position_metrics:
            weights = position_metrics['position_weights']
            fig.add_trace(
                go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    name='Position Weights'
                ),
                row=3, col=1
            )
        
        # Add risk metrics
        risk_metrics = {
            'Sharpe Ratio': self.get_performance_metrics()['sharpe_ratio'],
            'Sortino Ratio': self.get_performance_metrics()['sortino_ratio'],
            'Calmar Ratio': self.get_performance_metrics()['calmar_ratio'],
            'Win Rate': self.get_performance_metrics()['win_rate']
        }
        fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                name='Risk Metrics'
            ),
            row=3, col=2
        )
        
        # Add benchmark comparison
        if self.benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_metrics()
            fig.add_trace(
                go.Bar(
                    x=list(benchmark_metrics.keys()),
                    y=list(benchmark_metrics.values()),
                    name='Benchmark Metrics'
                ),
                row=4, col=1
            )
        
        # Add trade analysis
        trade_analysis = {
            'Avg Trade': self.get_performance_metrics()['avg_trade'],
            'Avg Win': self.get_performance_metrics()['avg_win'],
            'Avg Loss': self.get_performance_metrics()['avg_loss'],
            'Win Rate': self.get_performance_metrics()['win_rate']
        }
        fig.add_trace(
            go.Bar(
                x=list(trade_analysis.keys()),
                y=list(trade_analysis.values()),
                name='Trade Analysis'
            ),
            row=4, col=2
        )
        
        # Add risk decomposition
        risk_decomp = self._calculate_risk_decomposition()
        fig.add_trace(
            go.Bar(
                x=list(risk_decomp.keys()),
                y=list(risk_decomp.values()),
                name='Risk Decomposition'
            ),
            row=5, col=1
        )
        
        # Add performance attribution
        perf_attr = self._calculate_performance_attribution()
        fig.add_trace(
            go.Bar(
                x=list(perf_attr.keys()),
                y=list(perf_attr.values()),
                name='Performance Attribution'
            ),
            row=5, col=2
        )
        
        # Add technical indicators
        for asset in self.positions:
            indicators = self._calculate_technical_indicators(self.data[asset].pct_change())
            fig.add_trace(
                go.Scatter(
                    y=list(indicators['ichimoku'].values()),
                    name=f'{asset} Ichimoku'
                ),
                row=6, col=1
            )
        
        # Add ML predictions
        if hasattr(self, 'ml_predictions'):
            fig.add_trace(
                go.Scatter(
                    y=self.ml_predictions,
                    name='ML Predictions'
                ),
                row=6, col=2
            )
        
        # Add correlation heatmap
        corr_matrix = self._calculate_correlation_matrix()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu'
            ),
            row=7, col=1
        )
        
        # Add network graph
        network_graph = self._create_network_graph()
        fig.add_trace(
            go.Scatter(
                x=network_graph['x'],
                y=network_graph['y'],
                mode='markers+lines+text',
                text=network_graph['labels'],
                name='Network Graph'
            ),
            row=7, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=2800,
            width=2000,
            title_text='Backtest Results',
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html('backtest_results.html')
        
        # Create candlestick chart
        for asset in self.positions:
            df = self.data[asset].copy()
            df.index = pd.to_datetime(df.index)
            
            fig = mpf.figure(figsize=(20, 10))
            mpf.plot(df, type='candle', style='charles',
                    title=f'{asset} Price Chart',
                    volume=True,
                    savefig=f'{asset}_candlestick.png')
        
    def _plot_matplotlib(self) -> None:
        """Create static Matplotlib visualization."""
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, axes = plt.subplots(7, 2, figsize=(20, 20))
        
        # Plot portfolio value
        axes[0, 0].plot(self.portfolio_values)
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].grid(True)
        
        # Plot returns distribution
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        sns.histplot(returns, ax=axes[0, 1], bins=50)
        axes[0, 1].set_title('Returns Distribution')
        
        # Plot drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        axes[1, 0].fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].grid(True)
        
        # Plot trade P&L
        trade_pnl = [t.pnl for t in self.trades if t.pnl]
        axes[1, 1].plot(trade_pnl)
        axes[1, 1].set_title('Trade P&L')
        axes[1, 1].grid(True)
        
        # Plot position weights
        position_metrics = self._calculate_position_metrics()
        if position_metrics:
            weights = position_metrics['position_weights']
            axes[2, 0].pie(
                list(weights.values()),
                labels=list(weights.keys()),
                autopct='%1.1f%%'
            )
            axes[2, 0].set_title('Position Weights')
        
        # Plot risk metrics
        risk_metrics = {
            'Sharpe Ratio': self.get_performance_metrics()['sharpe_ratio'],
            'Sortino Ratio': self.get_performance_metrics()['sortino_ratio'],
            'Calmar Ratio': self.get_performance_metrics()['calmar_ratio'],
            'Win Rate': self.get_performance_metrics()['win_rate']
        }
        axes[2, 1].bar(
            list(risk_metrics.keys()),
            list(risk_metrics.values())
        )
        axes[2, 1].set_title('Risk Metrics')
        plt.setp(axes[2, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot benchmark comparison
        if self.benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_metrics()
            axes[3, 0].bar(
                list(benchmark_metrics.keys()),
                list(benchmark_metrics.values())
            )
            axes[3, 0].set_title('Benchmark Comparison')
            plt.setp(axes[3, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot trade analysis
        trade_analysis = {
            'Avg Trade': self.get_performance_metrics()['avg_trade'],
            'Avg Win': self.get_performance_metrics()['avg_win'],
            'Avg Loss': self.get_performance_metrics()['avg_loss'],
            'Win Rate': self.get_performance_metrics()['win_rate']
        }
        axes[3, 1].bar(
            list(trade_analysis.keys()),
            list(trade_analysis.values())
        )
        axes[3, 1].set_title('Trade Analysis')
        plt.setp(axes[3, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot risk decomposition
        risk_decomp = self._calculate_risk_decomposition()
        axes[4, 0].bar(
            list(risk_decomp.keys()),
            list(risk_decomp.values())
        )
        axes[4, 0].set_title('Risk Decomposition')
        plt.setp(axes[4, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot performance attribution
        perf_attr = self._calculate_performance_attribution()
        axes[4, 1].bar(
            list(perf_attr.keys()),
            list(perf_attr.values())
        )
        axes[4, 1].set_title('Performance Attribution')
        plt.setp(axes[4, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot technical indicators
        for asset in self.positions:
            indicators = self._calculate_technical_indicators(self.data[asset].pct_change())
            axes[5, 0].plot(indicators['ichimoku'].values(), label=f'{asset} Ichimoku')
            axes[5, 0].plot(indicators['fibonacci'].values(), label=f'{asset} Fibonacci')
            axes[5, 0].plot(indicators['elliott_wave'].values(), label=f'{asset} Elliott Wave')
            axes[5, 0].plot(indicators['macd'], label=f'{asset} MACD')
            axes[5, 0].plot(indicators['rsi'], label=f'{asset} RSI')
            axes[5, 0].plot(indicators['bollinger'], label=f'{asset} Bollinger')
            axes[5, 0].plot(indicators['stochastic'], label=f'{asset} Stochastic')
            axes[5, 0].plot(indicators['adx'], label=f'{asset} ADX')
            axes[5, 0].plot(indicators['cci'], label=f'{asset} CCI')
            axes[5, 0].plot(indicators['mfi'], label=f'{asset} MFI')
            axes[5, 0].plot(indicators['obv'], label=f'{asset} OBV')
            axes[5, 0].set_title('Technical Indicators')
            axes[5, 0].legend()
        
        # Plot ML predictions
        if hasattr(self, 'ml_predictions'):
            axes[5, 1].plot(self.ml_predictions)
            axes[5, 1].set_title('ML Predictions')
        
        # Add correlation heatmap
        corr_matrix = self._calculate_correlation_matrix()
        axes[6, 0].imshow(corr_matrix, cmap='RdBu', interpolation='nearest')
        axes[6, 0].set_title('Correlation Heatmap')
        
        # Add network graph
        network_graph = self._create_network_graph()
        axes[6, 1].scatter(network_graph['x'], network_graph['y'], c=network_graph['colors'], s=100)
        axes[6, 1].set_title('Network Graph')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'trade_log_file'):
            self.trade_log_file.close()

    def _calculate_risk_decomposition(self) -> Dict[str, float]:
        """Calculate risk decomposition across assets."""
        if not self.positions:
            return {}
        
        # Calculate returns for all positions
        returns = pd.DataFrame()
        for asset in self.positions:
            returns[asset] = self.data[asset].pct_change()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate portfolio variance
        weights = np.array([self.positions[asset] for asset in self.positions])
        portfolio_var = weights.T @ cov_matrix @ weights
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix @ weights / np.sqrt(portfolio_var)
        
        # Calculate risk contribution
        risk_contrib = weights * mcr
        
        return {
            asset: contrib / np.sqrt(portfolio_var)
            for asset, contrib in zip(self.positions.keys(), risk_contrib)
        }

    def _calculate_performance_attribution(self) -> Dict[str, float]:
        """Calculate performance attribution across assets."""
        if not self.positions:
            return {}
        
        # Calculate returns for all positions
        returns = pd.DataFrame()
        for asset in self.positions:
            returns[asset] = self.data[asset].pct_change()
        
        # Calculate portfolio return
        portfolio_return = returns.mean().dot(
            [self.positions[asset] for asset in self.positions]
        )
        
        # Calculate contribution to return
        return {
            asset: ret * self.positions[asset] / portfolio_return
            for asset, ret in zip(self.positions.keys(), returns.mean())
        }

    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all assets."""
        returns = pd.DataFrame()
        for asset in self.positions:
            returns[asset] = self.data[asset].pct_change()
        return returns.corr()

    def _create_network_graph(self) -> Dict[str, List]:
        """Create network graph of asset relationships."""
        corr_matrix = self._calculate_correlation_matrix()
        
        # Create graph
        G = nx.Graph()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) > 0.5:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                              weight=abs(corr_matrix.iloc[i,j]))
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Prepare data for plotting
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = []
        node_y = []
        node_labels = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
        
        return {
            'x': node_x,
            'y': node_y,
            'labels': node_labels,
            'edge_x': edge_x,
            'edge_y': edge_y
        }

class GRUModel(nn.Module):
    """GRU model for time series prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def run_backtest(self, strategy: Union[str, List[str]], plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run backtest for one or more strategies.
    
    Args:
        strategy: Single strategy or list of strategies to backtest
        plot: Whether to generate plots (default: True)
        
    Returns:
        Tuple containing:
        - DataFrame with backtest results
        - DataFrame with trade log
        - Dictionary with performance metrics
    """
    if isinstance(strategy, str):
        strategy = [strategy]
        
    # Initialize results storage
    self.portfolio_values = []
    self.trades = []
    self.positions = {}
    self.strategy_results = {}
    
    # Validate data
    for asset in self.data.columns:
        if 'Signal' not in self.data[asset].columns:
            raise ValueError(f"'Signal' column not found in data for asset {asset}")
    
    # Run backtest for each strategy
    for strat in strategy:
        self.strategy_results[strat] = {
            'trades': [],
            'portfolio_values': [],
            'positions': {}
        }
        
        # Process each time step
        for timestamp, row in self.data.iterrows():
            for asset in row.index:
                if asset == 'Signal':
                    continue
                    
                signal = row[asset]['Signal']
                price = row[asset]['Close']
                
                # Check for exit conditions
                if self._check_exit_conditions(asset, price, strat):
                    self._execute_trade(asset, price, 0, TradeType.EXIT, strat)
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(asset, price, strat, signal)
                
                # Execute trade if signal exists
                if signal != 0:
                    self._execute_trade(asset, price, position_size, 
                                     TradeType.BUY if signal > 0 else TradeType.SELL,
                                     strat)
        
        # Calculate results for this strategy
        df = self._calculate_equity_curve(strat)
        trade_log = self._generate_trade_log(strat)
        metrics = self._compute_metrics(df, trade_log)
        
        self.strategy_results[strat].update({
            'equity_curve': df,
            'trade_log': trade_log,
            'metrics': metrics
        })
    
    # Generate plots if requested
    if plot:
        self.plot_results()
    
    # Return results for the last strategy (or combine if multiple)
    if len(strategy) == 1:
        return (self.strategy_results[strategy[0]]['equity_curve'],
                self.strategy_results[strategy[0]]['trade_log'],
                self.strategy_results[strategy[0]]['metrics'])
    else:
        # Combine results from all strategies
        combined_df = pd.concat([res['equity_curve'] for res in self.strategy_results.values()])
        combined_trades = pd.concat([res['trade_log'] for res in self.strategy_results.values()])
        combined_metrics = self._combine_metrics([res['metrics'] for res in self.strategy_results.values()])
        return combined_df, combined_trades, combined_metrics

def _calculate_equity_curve(self, strategy: str) -> pd.DataFrame:
    """Calculate equity curve for a strategy.
    
    Args:
        strategy: Strategy identifier
        
    Returns:
        DataFrame with equity curve data
    """
    portfolio_values = self.strategy_results[strategy]['portfolio_values']
    positions = self.strategy_results[strategy]['positions']
    
    df = pd.DataFrame({
        'Portfolio Value': portfolio_values,
        'Cash': [self.cash] * len(portfolio_values)
    })
    
    # Add position values
    for asset, position in positions.items():
        df[f'{asset} Position'] = position
    
    # Calculate returns
    df['Returns'] = df['Portfolio Value'].pct_change()
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod()
    
    return df

def _generate_trade_log(self, strategy: str) -> pd.DataFrame:
    """Generate trade log for a strategy.
    
    Args:
        strategy: Strategy identifier
        
    Returns:
        DataFrame with trade log
    """
    trades = self.strategy_results[strategy]['trades']
    
    if not trades:
        return pd.DataFrame()
    
    # Convert trades to DataFrame
    trade_log = pd.DataFrame([{
        'Timestamp': t.timestamp,
        'Asset': t.asset,
        'Type': t.type.value,
        'Quantity': t.quantity,
        'Price': t.price,
        'Value': t.quantity * t.price,
        'Slippage': t.slippage,
        'Transaction Cost': t.transaction_cost,
        'Spread': t.spread,
        'Cash Balance': t.cash_balance,
        'Portfolio Value': t.portfolio_value,
        'Strategy': t.strategy,
        'Position Size': t.position_size,
        'Risk Metrics': t.risk_metrics,
        'Entry Price': t.entry_price,
        'Exit Price': t.exit_price,
        'Holding Period': t.holding_period,
        'PnL': t.pnl,
        'PnL %': t.pnl_pct
    } for t in trades])
    
    return trade_log

def _compute_metrics(self, df: pd.DataFrame, trade_log: pd.DataFrame) -> Dict[str, Any]:
    """Compute performance metrics for a strategy.
    
    Args:
        df: Equity curve DataFrame
        trade_log: Trade log DataFrame
        
    Returns:
        Dictionary with performance metrics
    """
    returns = df['Returns']
    
    # Basic metrics
    total_return = (df['Portfolio Value'].iloc[-1] / self.initial_cash) - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / returns.std()
    
    # Drawdown metrics
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Trade metrics
    if not trade_log.empty:
        win_rate = len(trade_log[trade_log['PnL'] > 0]) / len(trade_log)
        avg_trade = trade_log['PnL'].mean()
        avg_win = trade_log[trade_log['PnL'] > 0]['PnL'].mean()
        avg_loss = trade_log[trade_log['PnL'] < 0]['PnL'].mean()
        profit_factor = abs(trade_log[trade_log['PnL'] > 0]['PnL'].sum() / 
                          trade_log[trade_log['PnL'] < 0]['PnL'].sum())
    else:
        win_rate = avg_trade = avg_win = avg_loss = profit_factor = 0.0
    
    # Risk metrics
    risk_metrics = self._calculate_risk_metrics()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'risk_metrics': risk_metrics
    }

def _combine_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine metrics from multiple strategies."""
    combined = {
        'total_return': np.mean([m['total_return'] for m in metrics_list]),
        'annual_return': np.mean([m['annual_return'] for m in metrics_list]),
        'volatility': np.mean([m['volatility'] for m in metrics_list]),
        'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in metrics_list]),
        'max_drawdown': np.mean([m['max_drawdown'] for m in metrics_list]),
        'win_rate': np.mean([m['win_rate'] for m in metrics_list]),
        'avg_trade': np.mean([m['avg_trade'] for m in metrics_list]),
        'avg_win': np.mean([m['avg_win'] for m in metrics_list]),
        'avg_loss': np.mean([m['avg_loss'] for m in metrics_list]),
        'profit_factor': np.mean([m['profit_factor'] for m in metrics_list]),
        'risk_metrics': {}
    }
    
    # Combine risk metrics
    risk_metric_keys = metrics_list[0]['risk_metrics'].keys()
    for key in risk_metric_keys:
        combined['risk_metrics'][key] = np.mean([m['risk_metrics'][key] for m in metrics_list])
    
    return combined

def plot_results(self, use_plotly: bool = True) -> None:
    """Plot backtest results."""
    # Import matplotlib only when needed
    import matplotlib.pyplot as plt
    
    if use_plotly:
        self._plot_plotly()
    else:
        self._plot_matplotlib()