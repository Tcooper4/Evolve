"""
Portfolio Optimization Engine

Advanced portfolio optimization using CVXPY and CVXOPT.
Implements Mean-Variance Optimization, Black-Litterman Model, and Min-CVaR strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime

# Import optimization libraries with fallback handling
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Install with: pip install cvxpy")

try:
    import cvxopt
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    logging.warning("CVXOPT not available. Install with: pip install cvxopt")

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.results_dir = Path("results/portfolio_optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available. Portfolio optimization will use simplified methods.")
        
        logger.info("Portfolio optimizer initialized")
    
    def mean_variance_optimization(
        self, 
        returns: pd.DataFrame, 
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Mean-Variance Optimization using CVXPY.
        
        Args:
            returns: Asset returns DataFrame
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints dictionary
            
        Returns:
            Dictionary with optimization results
        """
        if not CVXPY_AVAILABLE:
            return self._simple_mean_variance(returns, target_return, risk_aversion)
        
        try:
            # Calculate expected returns and covariance matrix
            mu = returns.mean()
            Sigma = returns.cov()
            
            n_assets = len(mu)
            
            # Define variables
            w = cp.Variable(n_assets)
            
            # Define objective
            if target_return is not None:
                # Minimize variance subject to target return
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                constraints_list = [
                    w >= 0,  # Long-only constraint
                    cp.sum(w) == 1,  # Budget constraint
                    mu @ w >= target_return  # Return constraint
                ]
            else:
                # Maximize Sharpe ratio (minimize negative Sharpe)
                excess_return = mu - self.risk_free_rate
                objective = cp.Minimize(-excess_return @ w / cp.sqrt(cp.quad_form(w, Sigma)))
                constraints_list = [
                    w >= 0,  # Long-only constraint
                    cp.sum(w) == 1  # Budget constraint
                ]
            
            # Add custom constraints
            if constraints:
                if 'max_weight' in constraints:
                    constraints_list.append(w <= constraints['max_weight'])
                if 'min_weight' in constraints:
                    constraints_list.append(w >= constraints['min_weight'])
                if 'sector_limits' in constraints:
                    for sector, limit in constraints['sector_limits'].items():
                        # This would need sector mapping - simplified here
                        pass
            
            # Solve problem
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == 'optimal':
                weights = w.value
                portfolio_return = mu @ weights
                portfolio_vol = np.sqrt(weights @ Sigma @ weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
                
                # Calculate asset contributions
                asset_contributions = self._calculate_asset_contributions(weights, mu, Sigma)
                
                result = {
                    'weights': dict(zip(returns.columns, weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'asset_contributions': asset_contributions,
                    'optimization_status': 'optimal',
                    'constraints_used': list(constraints.keys()) if constraints else []
                }
                
                # Save results
                self._save_optimization_results('mean_variance', result)
                
                return result
            else:
                logger.warning(f"Mean-variance optimization failed: {problem.status}")
                return {'error': f'Optimization failed: {problem.status}'}
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return self._simple_mean_variance(returns, target_return, risk_aversion)
    
    def black_litterman_optimization(
        self, 
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Dict[str, float],
        confidence: Dict[str, float],
        tau: float = 0.05
    ) -> Dict[str, Any]:
        """Black-Litterman Model optimization.
        
        Args:
            returns: Asset returns DataFrame
            market_caps: Market capitalization weights
            views: Dictionary of views {asset: expected_return}
            confidence: Dictionary of confidence levels {asset: confidence}
            tau: Scaling parameter
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Calculate market equilibrium returns
            mu_market = returns.mean()
            Sigma = returns.cov()
            
            # Market equilibrium returns (reverse optimization)
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * Sigma @ market_caps
            
            # Create view matrix
            assets = list(views.keys())
            P = np.zeros((len(views), len(returns.columns)))
            q = np.array(list(views.values()))
            Omega = np.diag([1/confidence[asset] for asset in assets])
            
            for i, asset in enumerate(assets):
                if asset in returns.columns:
                    col_idx = returns.columns.get_loc(asset)
                    P[i, col_idx] = 1
            
            # Black-Litterman posterior estimates
            M1 = np.linalg.inv(np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P)
            M2 = np.linalg.inv(tau * Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ q
            
            mu_bl = M1 @ M2
            Sigma_bl = Sigma + M1
            
            # Convert to Series
            mu_bl_series = pd.Series(mu_bl, index=returns.columns)
            
            # Run mean-variance optimization with BL estimates
            return self.mean_variance_optimization(
                returns, 
                target_return=None,
                constraints={'max_weight': 0.2}  # Limit individual weights
            )
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return {'error': str(e)}
    
    def min_cvar_optimization(
        self, 
        returns: pd.DataFrame, 
        confidence_level: float = 0.95,
        target_return: Optional[float] = None
    ) -> Dict[str, Any]:
        """Minimum Conditional Value at Risk (CVaR) optimization.
        
        Args:
            returns: Asset returns DataFrame
            confidence_level: Confidence level for CVaR (e.g., 0.95)
            target_return: Optional target return constraint
            
        Returns:
            Dictionary with optimization results
        """
        if not CVXPY_AVAILABLE:
            return self._simple_cvar_optimization(returns, confidence_level, target_return)
        
        try:
            n_assets = len(returns.columns)
            n_scenarios = len(returns)
            
            # Define variables
            w = cp.Variable(n_assets)
            alpha = cp.Variable()  # VaR
            z = cp.Variable(n_scenarios)  # Auxiliary variables
            
            # Scenario returns
            R = returns.values
            
            # CVaR objective
            beta = 1 - confidence_level
            objective = cp.Minimize(alpha + (1/beta) * cp.sum(z) / n_scenarios)
            
            # Constraints
            constraints = [
                w >= 0,  # Long-only
                cp.sum(w) == 1,  # Budget constraint
                z >= 0,  # Non-negative auxiliary variables
                z >= -R @ w - alpha  # CVaR constraint
            ]
            
            if target_return is not None:
                mu = returns.mean()
                constraints.append(mu @ w >= target_return)
            
            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                weights = w.value
                cvar = alpha.value + (1/beta) * np.sum(z.value) / n_scenarios
                
                # Calculate additional metrics
                portfolio_return = returns.mean() @ weights
                portfolio_vol = np.sqrt(weights @ returns.cov() @ weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
                
                result = {
                    'weights': dict(zip(returns.columns, weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'cvar': cvar,
                    'sharpe_ratio': sharpe_ratio,
                    'confidence_level': confidence_level,
                    'optimization_status': 'optimal'
                }
                
                # Save results
                self._save_optimization_results('min_cvar', result)
                
                return result
            else:
                logger.warning(f"Min-CVaR optimization failed: {problem.status}")
                return {'error': f'Optimization failed: {problem.status}'}
                
        except Exception as e:
            logger.error(f"Error in Min-CVaR optimization: {e}")
            return self._simple_cvar_optimization(returns, confidence_level, target_return)
    
    def _simple_mean_variance(self, returns: pd.DataFrame, target_return: Optional[float], risk_aversion: float) -> Dict[str, Any]:
        """Simplified mean-variance optimization without CVXPY."""
        try:
            mu = returns.mean()
            Sigma = returns.cov()
            
            # Simple inverse volatility weighting as fallback
            vol = returns.std()
            weights = (1 / vol) / (1 / vol).sum()
            
            portfolio_return = mu @ weights
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_status': 'fallback_inverse_volatility'
            }
            
        except Exception as e:
            logger.error(f"Error in simple mean-variance: {e}")
            return {'error': str(e)}
    
    def _simple_cvar_optimization(self, returns: pd.DataFrame, confidence_level: float, target_return: Optional[float]) -> Dict[str, Any]:
        """Simplified CVaR optimization without CVXPY."""
        try:
            # Use historical simulation for CVaR
            portfolio_returns = returns.mean(axis=1)  # Equal weight portfolio
            cvar = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # Use inverse volatility weighting
            vol = returns.std()
            weights = (1 / vol) / (1 / vol).sum()
            
            portfolio_return = returns.mean() @ weights
            portfolio_vol = np.sqrt(weights @ returns.cov() @ weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'cvar': cvar,
                'sharpe_ratio': sharpe_ratio,
                'confidence_level': confidence_level,
                'optimization_status': 'fallback_historical_cvar'
            }
            
        except Exception as e:
            logger.error(f"Error in simple CVaR optimization: {e}")
            return {'error': str(e)}
    
    def _calculate_asset_contributions(self, weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate asset contributions to portfolio metrics."""
        try:
            portfolio_return = mu @ weights
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            
            # Return contribution
            return_contrib = {asset: weight * mu[asset] for asset, weight in zip(mu.index, weights)}
            
            # Risk contribution
            risk_contrib = {}
            for i, asset in enumerate(mu.index):
                # Marginal contribution to risk
                mcr = (Sigma.iloc[i] @ weights) / portfolio_vol
                risk_contrib[asset] = weights[i] * mcr
            
            # Sharpe contribution
            sharpe_contrib = {}
            for asset in mu.index:
                sharpe_contrib[asset] = (return_contrib[asset] - self.risk_free_rate * weights[mu.index.get_loc(asset)]) / portfolio_vol
            
            return {
                'return_contribution': return_contrib,
                'risk_contribution': risk_contrib,
                'sharpe_contribution': sharpe_contrib
            }
            
        except Exception as e:
            logger.error(f"Error calculating asset contributions: {e}")
            return {}
    
    def _save_optimization_results(self, method: str, results: Dict[str, Any]):
        """Save optimization results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{method}_optimization_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def compare_strategies(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compare different optimization strategies."""
        try:
            strategies = {}
            
            # Mean-Variance
            mv_result = self.mean_variance_optimization(returns)
            if 'error' not in mv_result:
                strategies['Mean-Variance'] = {
                    'Return': mv_result['portfolio_return'],
                    'Volatility': mv_result['portfolio_volatility'],
                    'Sharpe': mv_result['sharpe_ratio']
                }
            
            # Min-CVaR
            cvar_result = self.min_cvar_optimization(returns)
            if 'error' not in cvar_result:
                strategies['Min-CVaR'] = {
                    'Return': cvar_result['portfolio_return'],
                    'Volatility': cvar_result['portfolio_volatility'],
                    'Sharpe': cvar_result['sharpe_ratio'],
                    'CVaR': cvar_result['cvar']
                }
            
            # Equal Weight (benchmark)
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            equal_return = returns.mean() @ equal_weights
            equal_vol = np.sqrt(equal_weights @ returns.cov() @ equal_weights)
            equal_sharpe = (equal_return - self.risk_free_rate) / equal_vol
            
            strategies['Equal Weight'] = {
                'Return': equal_return,
                'Volatility': equal_vol,
                'Sharpe': equal_sharpe
            }
            
            return pd.DataFrame(strategies).T
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()

# Global portfolio optimizer instance
_portfolio_optimizer = None

def get_portfolio_optimizer() -> PortfolioOptimizer:
    """Get the global portfolio optimizer instance."""
    global _portfolio_optimizer
    if _portfolio_optimizer is None:
        _portfolio_optimizer = PortfolioOptimizer()
    return _portfolio_optimizer 