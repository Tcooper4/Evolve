import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
import os

class RiskManager:
    """Risk management and portfolio optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize risk manager.
        
        Args:
            config: Configuration dictionary containing:
                - risk_limits: Dictionary of risk limits
                    - max_drawdown: Maximum drawdown limit (default: 0.2)
                    - max_volatility: Maximum volatility limit (default: 0.3)
                    - min_sharpe_ratio: Minimum Sharpe ratio limit (default: 1.0)
                - results_dir: Directory for saving results (default: risk_results)
                - log_level: Logging level (default: INFO)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'risk_limits': {
                'max_drawdown': float(os.getenv('RISK_MAX_DRAWDOWN', 0.2)),
                'max_volatility': float(os.getenv('RISK_MAX_VOLATILITY', 0.3)),
                'min_sharpe_ratio': float(os.getenv('RISK_MIN_SHARPE_RATIO', 1.0))
            },
            'results_dir': os.getenv('RISK_RESULTS_DIR', 'risk_results'),
            'log_level': os.getenv('RISK_LOG_LEVEL', 'INFO')
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'var': [],
            'cvar': [],
            'volatility': [],
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': []
        }
        
        # Initialize risk metrics
        self.var_95 = None
        self.var_99 = None
        self.beta = None
        self.correlation = None
        self.returns = None
        self.sortino = None
    
    def calculate_position_size(self, asset_price: float, volatility: float,
                              account_value: float, risk_per_trade: float) -> float:
        """Calculate optimal position size.
        
        Args:
            asset_price: Current asset price
            volatility: Asset volatility
            account_value: Total account value
            risk_per_trade: Maximum risk per trade as fraction of account
            
        Returns:
            Optimal position size
        """
        # Calculate position size using Kelly Criterion
        win_rate = 0.5  # Conservative estimate
        avg_win = volatility * 2  # Conservative estimate
        avg_loss = volatility
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limit to 50% of Kelly
        
        # Calculate position size
        risk_amount = account_value * risk_per_trade
        position_size = risk_amount / (volatility * asset_price)
        
        # Apply Kelly fraction
        position_size *= kelly_fraction
        
        return position_size
    
    def calculate_portfolio_risk(self, positions: Dict[str, float],
                               returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio risk metrics.
        
        Args:
            positions: Dictionary of position sizes
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * pd.Series(positions)).sum(axis=1)
        
        # Calculate metrics
        var_95 = self._calculate_var(portfolio_returns, 0.95)
        cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        volatility = self._calculate_portfolio_volatility(positions, returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Update metrics history
        self.metrics_history['var'].append(var_95)
        self.metrics_history['cvar'].append(cvar_95)
        self.metrics_history['volatility'].append(volatility)
        self.metrics_history['sharpe_ratio'].append(sharpe_ratio)
        self.metrics_history['sortino_ratio'].append(sortino_ratio)
        self.metrics_history['max_drawdown'].append(max_drawdown)
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame,
                         risk_free_rate: float = 0.0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Optimize portfolio weights.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Tuple of (optimal weights, metrics)
        """
        n_assets = len(returns.columns)
        
        # Define objective function (negative Sharpe ratio)
        def neg_sharpe_ratio(weights):
            portfolio_returns = (returns * weights).sum(axis=1)
            return -self._calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((0, 1) for _ in range(n_assets))  # weights between 0 and 1
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
        
        # Calculate optimal weights
        optimal_weights = dict(zip(returns.columns, result.x))
        
        # Calculate metrics
        portfolio_returns = (returns * result.x).sum(axis=1)
        metrics = {
            'sharpe_ratio': -result.fun,
            'volatility': self._calculate_portfolio_volatility(optimal_weights, returns),
            'returns': portfolio_returns.mean() * 252,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
        }
        
        return optimal_weights, metrics
    
    def check_risk_limits(self, positions: Dict[str, float],
                         returns: pd.DataFrame) -> Dict[str, bool]:
        """Check if current positions violate risk limits.
        
        Args:
            positions: Dictionary of position sizes
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of limit violations
        """
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_risk(positions, returns)
        
        # Check limits
        limits = self.config.get('risk_limits', {})
        violations = {
            'max_drawdown': metrics['max_drawdown'] > limits.get('max_drawdown', 0.2),
            'max_volatility': metrics['volatility'] > limits.get('max_volatility', 0.3),
            'min_sharpe_ratio': metrics['sharpe_ratio'] < limits.get('min_sharpe_ratio', 1.0)
        }
        
        return violations
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float],
                                     returns: pd.DataFrame) -> float:
        """Calculate portfolio volatility."""
        cov_matrix = returns.cov()
        weights_array = np.array(list(weights.values()))
        return np.sqrt(weights_array.T @ cov_matrix @ weights_array) * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series,
                              risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series,
                                risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        downside = returns[returns < 0]
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        excess = returns.mean() - risk_free_rate/252
        return np.sqrt(252) * excess / downside_std
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
    
    def save_results(self, results: Dict, filename: str):
        """Save optimization results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved results to {filepath}")
    
    def load_results(self, filename: str) -> Dict:
        """Load optimization results from disk."""
        filepath = self.results_dir / filename
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        self.logger.info(f"Loaded results from {filepath}")
        
        return results
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics."""
        if len(self.returns) > 0:
            self.var_95 = np.percentile(self.returns, 5)
            self.var_99 = np.percentile(self.returns, 1)
            self.beta = self.returns.cov(self.returns) / self.returns.var()
            self.correlation = self.returns.corr(self.returns)
            self.sortino = self._calculate_sortino_ratio(self.returns)
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        return {
            'var_95': self.var_95 if self.var_95 is not None else 0.0,
            'var_99': self.var_99 if self.var_99 is not None else 0.0,
            'beta': self.beta if self.beta is not None else 0.0,
            'correlation': self.correlation if self.correlation is not None else 0.0,
            'sortino_ratio': self.sortino if hasattr(self, 'sortino') else 0.0
        }
    
    def update_returns(self, new_returns: pd.Series):
        """Update historical returns and recalculate risk metrics.
        
        Args:
            new_returns: New returns series
        """
        self.returns = new_returns
        self._calculate_risk_metrics()
        self.logger.info("Updated returns and risk metrics")
    
    def get_position_limits(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate position size limits based on risk parameters.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of position limits
        """
        if not self.returns is not None:
            self.logger.warning("No returns data available for position limits")
            return {}
            
        volatility = self.returns.std() * np.sqrt(252)
        var = self._calculate_var(self.returns, 0.95)
        
        return {
            'max_position_size': portfolio_value * self.config['risk_limits']['max_drawdown'],
            'max_leverage': 1 / (volatility * 2),  # Conservative leverage limit
            'max_var_exposure': abs(var) * portfolio_value
        } 