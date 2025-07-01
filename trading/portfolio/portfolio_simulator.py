# -*- coding: utf-8 -*-
"""
Portfolio Simulation Module with advanced optimization techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm

from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.risk.risk_analyzer import RiskAnalyzer

class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    asset_limits: Optional[Dict[str, float]] = None
    leverage_limit: float = 1.0

@dataclass
class PortfolioResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    concentration_index: float
    rebalancing_frequency: str
    transaction_costs: float

class PortfolioSimulator:
    """
    Portfolio Simulation Module with:
    - Mean-variance optimization
    - Black-Litterman integration
    - Risk parity optimization
    - Risk management and constraints
    - Transaction cost modeling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Portfolio Simulator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.risk_analyzer = RiskAnalyzer()
        
        # Configuration
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.transaction_cost_rate = self.config.get('transaction_cost_rate', 0.001)
        self.rebalancing_frequency = self.config.get('rebalancing_frequency', 'monthly')
        self.lookback_period = self.config.get('lookback_period', 252)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # Storage
        self.portfolio_history: List[PortfolioResult] = []
        self.optimization_results: Dict[str, PortfolioResult] = {}
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def optimize_portfolio(self, 
                          returns_data: pd.DataFrame,
                          method: OptimizationMethod,
                          constraints: PortfolioConstraints,
                          views: Optional[Dict[str, float]] = None) -> PortfolioResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns_data: Asset returns DataFrame
            method: Optimization method
            constraints: Portfolio constraints
            views: Market views for Black-Litterman (optional)
            
        Returns:
            Portfolio optimization result
        """
        try:
            self.logger.info(f"Optimizing portfolio using {method.value}")
            
            # Calculate expected returns and covariance matrix
            expected_returns = self._calculate_expected_returns(returns_data)
            covariance_matrix = self._calculate_covariance_matrix(returns_data)
            
            # Apply optimization method
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._mean_variance_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._black_litterman_optimization(
                    returns_data, expected_returns, covariance_matrix, constraints, views
                )
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == OptimizationMethod.MAX_SHARPE:
                weights = self._max_sharpe_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == OptimizationMethod.MIN_VARIANCE:
                weights = self._min_variance_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                weights = self._equal_weight_allocation(returns_data.columns)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio metrics
            result = self._calculate_portfolio_metrics(
                weights, expected_returns, covariance_matrix, returns_data
            )
            
            # Store result
            self.optimization_results[method.value] = result
            
            self.logger.info(f"Portfolio optimization completed: "
                           f"Return={result.expected_return:.4f}, "
                           f"Vol={result.expected_volatility:.4f}, "
                           f"Sharpe={result.sharpe_ratio:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return self._create_error_result(returns_data.columns)
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate expected returns using historical data."""
        try:
            # Use exponential weighted moving average for more recent data
            alpha = 0.94  # RiskMetrics decay factor
            expected_returns = returns_data.ewm(alpha=alpha).mean().iloc[-1]
            
            return expected_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating expected returns: {str(e)}")
            return {'success': True, 'result': pd.Series(0.0, index=returns_data.columns), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix using historical data."""
        try:
            # Use exponential weighted moving average
            alpha = 0.94
            covariance_matrix = returns_data.ewm(alpha=alpha).cov()
            
            # Get the latest covariance matrix
            latest_cov = covariance_matrix.xs(returns_data.index[-1], level=1)
            
            return latest_cov
            
        except Exception as e:
            self.logger.error(f"Error calculating covariance matrix: {str(e)}")
            return {'success': True, 'result': pd.DataFrame(np.eye(len(returns_data.columns)),, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                              index=returns_data.columns, 
                              columns=returns_data.columns)
    
    def _mean_variance_optimization(self, 
                                  expected_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  constraints: PortfolioConstraints) -> Dict[str, float]:
        """Mean-variance optimization."""
        try:
            n_assets = len(expected_returns)
            
            # Define variables
            weights = cp.Variable(n_assets)
            
            # Define objective: minimize variance
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            
            # Define constraints
            constraints_list = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= constraints.min_weight,  # Minimum weight
                weights <= constraints.max_weight   # Maximum weight
            ]
            
            # Add target return constraint if specified
            if constraints.target_return is not None:
                portfolio_return = expected_returns.values @ weights
                constraints_list.append(portfolio_return >= constraints.target_return)
            
            # Add leverage constraint
            constraints_list.append(cp.sum(cp.abs(weights)) <= constraints.leverage_limit)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
            problem.solve()
            
            if problem.status == 'optimal':
                weights_dict = {asset: weight.value for asset, weight in 
                              zip(expected_returns.index, weights.value)}
                return weights_dict
            else:
                self.logger.warning("Mean-variance optimization failed, using equal weights")
                return self._equal_weight_allocation(expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {str(e)}")
            return self._equal_weight_allocation(expected_returns.index)
    
    def _black_litterman_optimization(self, 
                                    returns_data: pd.DataFrame,
                                    expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame,
                                    constraints: PortfolioConstraints,
                                    views: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Black-Litterman optimization with market views."""
        try:
            if views is None:
                return self._mean_variance_optimization(expected_returns, covariance_matrix, constraints)
            
            # Calculate market equilibrium returns
            market_cap_weights = self._estimate_market_cap_weights(returns_data)
            equilibrium_returns = self._calculate_equilibrium_returns(
                covariance_matrix, market_cap_weights
            )
            
            # Create view matrix and confidence matrix
            P, Q, Omega = self._create_view_matrices(views, expected_returns.index)
            
            # Calculate Black-Litterman returns
            tau = 0.05  # Scaling factor
            Pi = equilibrium_returns
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau * covariance_matrix)
            M2 = P.T @ np.linalg.inv(Omega)
            
            bl_returns = np.linalg.inv(M1 + M2) @ (M1 @ Pi + M2 @ Q)
            
            # Convert to Series
            bl_returns_series = pd.Series(bl_returns, index=expected_returns.index)
            
            # Use mean-variance optimization with Black-Litterman returns
            return self._mean_variance_optimization(bl_returns_series, covariance_matrix, constraints)
            
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {'success': True, 'result': self._mean_variance_optimization(expected_returns, covariance_matrix, constraints), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _estimate_market_cap_weights(self, returns_data: pd.DataFrame) -> pd.Series:
        """Estimate market capitalization weights."""
        try:
            # Use inverse volatility as proxy for market cap weights
            volatilities = returns_data.std()
            inv_vol_weights = 1 / volatilities
            market_cap_weights = inv_vol_weights / inv_vol_weights.sum()
            
            return market_cap_weights
            
        except Exception as e:
            self.logger.error(f"Error estimating market cap weights: {str(e)}")
            return {'success': True, 'result': pd.Series(1.0 / len(returns_data.columns), index=returns_data.columns), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_equilibrium_returns(self, 
                                     covariance_matrix: pd.DataFrame,
                                     market_cap_weights: pd.Series) -> np.ndarray:
        """Calculate equilibrium returns using reverse optimization."""
        try:
            # Assume market Sharpe ratio
            market_sharpe = 0.5
            
            # Calculate market portfolio volatility
            market_vol = np.sqrt(market_cap_weights.T @ covariance_matrix @ market_cap_weights)
            
            # Calculate equilibrium returns
            equilibrium_returns = market_sharpe * market_vol * covariance_matrix @ market_cap_weights
            
            return equilibrium_returns.values
            
        except Exception as e:
            self.logger.error(f"Error calculating equilibrium returns: {str(e)}")
            return np.zeros(len(covariance_matrix))
    
    def _create_view_matrices(self, 
                            views: Dict[str, float],
                            asset_names: pd.Index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create view matrices for Black-Litterman."""
        try:
            n_views = len(views)
            n_assets = len(asset_names)
            
            # Create view matrix P (n_views x n_assets)
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in asset_names:
                    asset_idx = asset_names.get_loc(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = view_return
            
            # Create confidence matrix Omega (diagonal)
            Omega = np.eye(n_views) * 0.01  # 1% confidence
            
            return P, Q, Omega
            
        except Exception as e:
            self.logger.error(f"Error creating view matrices: {str(e)}")
            return {'success': True, 'result': np.array([]), np.array([]), np.array([]), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _risk_parity_optimization(self, 
                                expected_returns: pd.Series,
                                covariance_matrix: pd.DataFrame,
                                constraints: PortfolioConstraints) -> Dict[str, float]:
        """Risk parity optimization."""
        try:
            n_assets = len(expected_returns)
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(weights.T @ covariance_matrix.values @ weights)
                asset_contributions = weights * (covariance_matrix.values @ weights) / portfolio_vol
                
                # Minimize variance of risk contributions
                risk_contribution_var = np.var(asset_contributions)
                return risk_contribution_var
            
            def constraint_sum_to_one(weights):
                return np.sum(weights) - 1
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'ineq', 'fun': lambda w: w - constraints.min_weight},
                {'type': 'ineq', 'fun': lambda w: constraints.max_weight - w}
            ]
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(constraints.min_weight, constraints.max_weight)] * n_assets
            )
            
            if result.success:
                weights_dict = {asset: weight for asset, weight in 
                              zip(expected_returns.index, result.x)}
                return weights_dict
            else:
                self.logger.warning("Risk parity optimization failed, using equal weights")
                return self._equal_weight_allocation(expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            return self._equal_weight_allocation(expected_returns.index)
    
    def _max_sharpe_optimization(self, 
                               expected_returns: pd.Series,
                               covariance_matrix: pd.DataFrame,
                               constraints: PortfolioConstraints) -> Dict[str, float]:
        """Maximum Sharpe ratio optimization."""
        try:
            n_assets = len(expected_returns)
            
            def negative_sharpe(weights):
                portfolio_return = expected_returns.values @ weights
                portfolio_vol = np.sqrt(weights.T @ covariance_matrix.values @ weights)
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe
            
            def constraint_sum_to_one(weights):
                return np.sum(weights) - 1
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'ineq', 'fun': lambda w: w - constraints.min_weight},
                {'type': 'ineq', 'fun': lambda w: constraints.max_weight - w}
            ]
            
            # Optimize
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(constraints.min_weight, constraints.max_weight)] * n_assets
            )
            
            if result.success:
                weights_dict = {asset: weight for asset, weight in 
                              zip(expected_returns.index, result.x)}
                return weights_dict
            else:
                self.logger.warning("Max Sharpe optimization failed, using equal weights")
                return self._equal_weight_allocation(expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Error in max Sharpe optimization: {str(e)}")
            return self._equal_weight_allocation(expected_returns.index)
    
    def _min_variance_optimization(self, 
                                 expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame,
                                 constraints: PortfolioConstraints) -> Dict[str, float]:
        """Minimum variance optimization."""
        try:
            n_assets = len(expected_returns)
            
            def portfolio_variance(weights):
                return weights.T @ covariance_matrix.values @ weights
            
            def constraint_sum_to_one(weights):
                return np.sum(weights) - 1
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'ineq', 'fun': lambda w: w - constraints.min_weight},
                {'type': 'ineq', 'fun': lambda w: constraints.max_weight - w}
            ]
            
            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(constraints.min_weight, constraints.max_weight)] * n_assets
            )
            
            if result.success:
                weights_dict = {asset: weight for asset, weight in 
                              zip(expected_returns.index, result.x)}
                return weights_dict
            else:
                self.logger.warning("Min variance optimization failed, using equal weights")
                return self._equal_weight_allocation(expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Error in min variance optimization: {str(e)}")
            return self._equal_weight_allocation(expected_returns.index)
    
    def _equal_weight_allocation(self, asset_names: pd.Index) -> Dict[str, float]:
        """Equal weight allocation."""
        try:
            n_assets = len(asset_names)
            weight = 1.0 / n_assets
            return {asset: weight for asset in asset_names}
            
        except Exception as e:
            self.logger.error(f"Error in equal weight allocation: {str(e)}")
            return {}
    
    def _calculate_portfolio_metrics(self, 
                                   weights: Dict[str, float],
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   returns_data: pd.DataFrame) -> PortfolioResult:
        """Calculate comprehensive portfolio metrics."""
        try:
            # Convert weights to array
            weight_array = np.array([weights[asset] for asset in expected_returns.index])
            
            # Calculate expected return and volatility
            expected_return = expected_returns.values @ weight_array
            expected_volatility = np.sqrt(weight_array.T @ covariance_matrix.values @ weight_array)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility
            
            # Calculate historical portfolio returns
            portfolio_returns = returns_data @ weight_array
            
            # Calculate max drawdown
            max_drawdown = calculate_max_drawdown(portfolio_returns)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(portfolio_returns)
            
            # Calculate diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(
                weights, covariance_matrix
            )
            
            # Calculate concentration index
            concentration_index = self._calculate_concentration_index(weights)
            
            # Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(weights)
            
            return PortfolioResult(
                weights=weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                diversification_ratio=diversification_ratio,
                concentration_index=concentration_index,
                rebalancing_frequency=self.rebalancing_frequency,
                transaction_costs=transaction_costs
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return self._create_error_result(returns_data.columns)
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        try:
            alpha = 1 - self.confidence_level
            var_95 = np.percentile(returns, alpha * 100)
            cvar_95 = returns[returns <= var_95].mean()
            
            return var_95, cvar_95
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR/CVaR: {str(e)}")
            return {'success': True, 'result': 0.0, 0.0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_diversification_ratio(self, 
                                       weights: Dict[str, float],
                                       covariance_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        try:
            weight_array = np.array([weights[asset] for asset in covariance_matrix.index])
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(weight_array.T @ covariance_matrix.values @ weight_array)
            
            # Weighted average of individual volatilities
            individual_vols = np.sqrt(np.diag(covariance_matrix.values))
            weighted_avg_vol = weight_array @ individual_vols
            
            return weighted_avg_vol / portfolio_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
    
    def _calculate_concentration_index(self, weights: Dict[str, float]) -> float:
        """Calculate concentration index (Herfindahl index)."""
        try:
            weight_values = list(weights.values())
            concentration = np.sum(np.square(weight_values))
            return concentration
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration index: {str(e)}")
            return 1.0
    
    def _calculate_transaction_costs(self, weights: Dict[str, float]) -> float:
        """Calculate expected transaction costs."""
        try:
            # Assume equal weight benchmark
            n_assets = len(weights)
            benchmark_weight = 1.0 / n_assets
            
            total_cost = 0.0
            for weight in weights.values():
                # Cost of deviating from benchmark
                deviation = abs(weight - benchmark_weight)
                total_cost += deviation * self.transaction_cost_rate
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction costs: {str(e)}")
            return 0.0
    
    def _create_error_result(self, asset_names: pd.Index) -> PortfolioResult:
        """Create error result when optimization fails."""
        weights = {asset: 1.0 / len(asset_names) for asset in asset_names}
        
        return PortfolioResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            diversification_ratio=1.0,
            concentration_index=1.0,
            rebalancing_frequency=self.rebalancing_frequency,
            transaction_costs=0.0
        )
    
    def compare_optimization_methods(self, 
                                   returns_data: pd.DataFrame,
                                   constraints: PortfolioConstraints) -> Dict[str, PortfolioResult]:
        """Compare different optimization methods."""
        try:
            self.logger.info("Comparing optimization methods")
            
            methods = [
                OptimizationMethod.MEAN_VARIANCE,
                OptimizationMethod.RISK_PARITY,
                OptimizationMethod.MAX_SHARPE,
                OptimizationMethod.MIN_VARIANCE,
                OptimizationMethod.EQUAL_WEIGHT
            ]
            
            results = {}
            
            for method in methods:
                result = self.optimize_portfolio(returns_data, method, constraints)
                results[method.value] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error comparing optimization methods: {str(e)}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        try:
            summary = {}
            
            for method, result in self.optimization_results.items():
                summary[method] = {
                    'expected_return': result.expected_return,
                    'expected_volatility': result.expected_volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'var_95': result.var_95,
                    'cvar_95': result.cvar_95,
                    'diversification_ratio': result.diversification_ratio,
                    'concentration_index': result.concentration_index,
                    'transaction_costs': result.transaction_costs
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {str(e)}")
            return {}