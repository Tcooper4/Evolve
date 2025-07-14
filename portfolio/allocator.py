"""
Portfolio Allocator

This module implements various portfolio allocation strategies:
- Modern Portfolio Theory (MPT)
- Risk Parity
- Kelly Criterion
- Black-Litterman Model
- Equal Weight
- Minimum Variance
- Maximum Sharpe Ratio

Supports dynamic rebalancing and risk-adjusted optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Local imports
from utils.cache_utils import cache_result
from utils.common_helpers import safe_json_save, load_config


class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_VARIANCE = "mean_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"


@dataclass
class AssetMetrics:
    """Asset-specific metrics for allocation"""
    ticker: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation: Dict[str, float]
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    sentiment_score: Optional[float] = None


@dataclass
class AllocationResult:
    """Result of portfolio allocation"""
    strategy: AllocationStrategy
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contribution: Dict[str, float]
    diversification_ratio: float
    constraints_satisfied: bool
    optimization_status: str
    metadata: Dict[str, Any]


class PortfolioAllocator:
    """
    Portfolio allocation engine with multiple strategies
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.portfolio_config = self.config.get('portfolio', {})
        
        # Allocation constraints
        self.max_weight = self.portfolio_config.get('max_weight', 0.3)
        self.min_weight = self.portfolio_config.get('min_weight', 0.01)
        self.target_volatility = self.portfolio_config.get('target_volatility', 0.15)
        self.risk_free_rate = self.portfolio_config.get('risk_free_rate', 0.02)
        
        # Optimization parameters
        self.max_iterations = self.portfolio_config.get('max_iterations', 1000)
        self.tolerance = self.portfolio_config.get('tolerance', 1e-6)
        
        # Kelly criterion parameters
        self.kelly_fraction = self.portfolio_config.get('kelly_fraction', 0.25)
        self.max_kelly_weight = self.portfolio_config.get('max_kelly_weight', 0.5)
    
    def allocate_portfolio(self, 
                         assets: List[AssetMetrics], 
                         strategy: AllocationStrategy,
                         constraints: Optional[Dict[str, Any]] = None) -> AllocationResult:
        """
        Allocate portfolio using specified strategy
        """
        if not assets:
            raise ValueError("No assets provided for allocation")
        
        # Extract asset data
        tickers = [asset.ticker for asset in assets]
        expected_returns = np.array([asset.expected_return for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(assets)
        covariance_matrix = self._build_covariance_matrix(volatilities, correlation_matrix)
        
        # Apply strategy-specific allocation
        if strategy == AllocationStrategy.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(assets)
        elif strategy == AllocationStrategy.MINIMUM_VARIANCE:
            weights = self._minimum_variance_allocation(covariance_matrix, constraints)
        elif strategy == AllocationStrategy.MAXIMUM_SHARPE:
            weights = self._maximum_sharpe_allocation(expected_returns, covariance_matrix, constraints)
        elif strategy == AllocationStrategy.RISK_PARITY:
            weights = self._risk_parity_allocation(covariance_matrix, constraints)
        elif strategy == AllocationStrategy.KELLY_CRITERION:
            weights = self._kelly_criterion_allocation(assets, constraints)
        elif strategy == AllocationStrategy.BLACK_LITTERMAN:
            weights = self._black_litterman_allocation(assets, constraints)
        elif strategy == AllocationStrategy.MEAN_VARIANCE:
            weights = self._mean_variance_allocation(expected_returns, covariance_matrix, constraints)
        elif strategy == AllocationStrategy.MAXIMUM_DIVERSIFICATION:
            weights = self._maximum_diversification_allocation(covariance_matrix, constraints)
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
        
        # Apply constraints and normalize
        weights = self._apply_constraints(weights, constraints)
        weights = self._normalize_weights(weights)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(weights, expected_returns, covariance_matrix)
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(weights, volatilities, covariance_matrix)
        
        return AllocationResult(
            strategy=strategy,
            weights=dict(zip(tickers, weights)),
            expected_return=portfolio_metrics['expected_return'],
            expected_volatility=portfolio_metrics['expected_volatility'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            risk_contribution=dict(zip(tickers, risk_contributions)),
            diversification_ratio=diversification_ratio,
            constraints_satisfied=self._check_constraints(weights, constraints),
            optimization_status="success",
            metadata=portfolio_metrics
        )
    
    def _equal_weight_allocation(self, assets: List[AssetMetrics]) -> np.ndarray:
        """Equal weight allocation"""
        n_assets = len(assets)
        return np.ones(n_assets) / n_assets
    
    def _minimum_variance_allocation(self, 
                                   covariance_matrix: np.ndarray,
                                   constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Minimum variance portfolio allocation"""
        n_assets = covariance_matrix.shape[0]
        
        # Objective: minimize w'Σw
        # Constraints: sum(w) = 1, w >= 0
        
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                return weights.T @ covariance_matrix @ weights
            
            def constraint_sum(weights):
                return np.sum(weights) - 1
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimization
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets
    
    def _maximum_sharpe_allocation(self,
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Maximum Sharpe ratio portfolio allocation"""
        n_assets = expected_returns.shape[0]
        
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            def constraint_sum(weights):
                return np.sum(weights) - 1
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets
    
    def _risk_parity_allocation(self,
                              covariance_matrix: np.ndarray,
                              constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Risk parity portfolio allocation"""
        n_assets = covariance_matrix.shape[0]
        
        try:
            from scipy.optimize import minimize
            
            def risk_contribution(weights):
                portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
                marginal_risk = covariance_matrix @ weights / portfolio_volatility
                return weights * marginal_risk
            
            def objective(weights):
                risk_contrib = risk_contribution(weights)
                # Minimize variance of risk contributions
                return np.var(risk_contrib)
            
            def constraint_sum(weights):
                return np.sum(weights) - 1
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets
    
    def _kelly_criterion_allocation(self,
                                  assets: List[AssetMetrics],
                                  constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Kelly criterion portfolio allocation"""
        n_assets = len(assets)
        kelly_weights = np.zeros(n_assets)
        
        for i, asset in enumerate(assets):
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            
            # Simplified Kelly for expected return and volatility
            expected_return = asset.expected_return
            volatility = asset.volatility
            
            if volatility > 0:
                # Kelly fraction based on Sharpe ratio
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
                kelly_fraction = max(0, sharpe_ratio) * self.kelly_fraction
                
                # Cap at maximum Kelly weight
                kelly_weights[i] = min(kelly_fraction, self.max_kelly_weight)
        
        # Normalize weights
        if np.sum(kelly_weights) > 0:
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        else:
            # Fallback to equal weights if no positive Kelly weights
            kelly_weights = np.ones(n_assets) / n_assets
        
        return kelly_weights
    
    def _black_litterman_allocation(self,
                                  assets: List[AssetMetrics],
                                  constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Black-Litterman model allocation"""
        n_assets = len(assets)
        
        # Market equilibrium returns (simplified)
        market_caps = np.array([asset.market_cap or 1.0 for asset in assets])
        market_weights = market_caps / np.sum(market_caps)
        
        # Equilibrium returns (simplified)
        equilibrium_returns = np.array([asset.expected_return for asset in assets])
        
        # Investor views (simplified - using sentiment scores)
        views = np.array([asset.sentiment_score or 0.0 for asset in assets])
        view_confidence = np.ones(n_assets) * 0.1  # Low confidence in views
        
        # Black-Litterman formula (simplified)
        tau = 0.05  # Scaling factor
        omega = np.diag(view_confidence ** 2)  # View uncertainty
        
        # Prior returns
        prior_returns = equilibrium_returns
        
        # Posterior returns
        try:
            # Simplified Black-Litterman calculation
            posterior_returns = prior_returns + tau * views
        except:
            posterior_returns = prior_returns
        
        # Use posterior returns for mean-variance optimization
        covariance_matrix = self._build_covariance_matrix(
            [asset.volatility for asset in assets],
            self._build_correlation_matrix(assets)
        )
        
        return self._mean_variance_allocation(posterior_returns, covariance_matrix, constraints)
    
    def _mean_variance_allocation(self,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Mean-variance optimization allocation"""
        n_assets = expected_returns.shape[0]
        
        try:
            from scipy.optimize import minimize
            
            # Risk aversion parameter
            risk_aversion = constraints.get('risk_aversion', 1.0) if constraints else 1.0
            
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = weights.T @ covariance_matrix @ weights
                # Maximize: return - 0.5 * risk_aversion * variance
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
            
            def constraint_sum(weights):
                return np.sum(weights) - 1
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets
    
    def _maximum_diversification_allocation(self,
                                          covariance_matrix: np.ndarray,
                                          constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Maximum diversification portfolio allocation"""
        n_assets = covariance_matrix.shape[0]
        
        try:
            from scipy.optimize import minimize
            
            def diversification_ratio(weights):
                portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
                weighted_volatilities = np.sum(weights * np.sqrt(np.diag(covariance_matrix)))
                return weighted_volatilities / portfolio_volatility
            
            def objective(weights):
                return -diversification_ratio(weights)  # Maximize diversification ratio
            
            def constraint_sum(weights):
                return np.sum(weights) - 1
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            # Fallback without scipy
            return np.ones(n_assets) / n_assets
    
    def _build_correlation_matrix(self, assets: List[AssetMetrics]) -> np.ndarray:
        """Build correlation matrix from asset correlations"""
        n_assets = len(assets)
        correlation_matrix = np.eye(n_assets)
        
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i != j:
                    # Use correlation if available, otherwise estimate
                    if asset_j.ticker in asset_i.correlation:
                        correlation_matrix[i, j] = asset_i.correlation[asset_j.ticker]
                    else:
                        # Default correlation based on sector similarity
                        if asset_i.sector and asset_j.sector:
                            if asset_i.sector == asset_j.sector:
                                correlation_matrix[i, j] = 0.7
                            else:
                                correlation_matrix[i, j] = 0.3
                        else:
                            correlation_matrix[i, j] = 0.5
        
        return correlation_matrix
    
    def _build_covariance_matrix(self, volatilities: np.ndarray, correlation_matrix: np.ndarray) -> np.ndarray:
        """Build covariance matrix from volatilities and correlations"""
        volatility_matrix = np.diag(volatilities)
        return volatility_matrix @ correlation_matrix @ volatility_matrix
    
    def _apply_constraints(self, weights: np.ndarray, constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Apply allocation constraints"""
        if not constraints:
            return weights
        
        # Apply sector constraints
        if 'sector_limits' in constraints:
            weights = self._apply_sector_constraints(weights, constraints['sector_limits'])
        
        # Apply volatility targeting
        if 'target_volatility' in constraints:
            weights = self._apply_volatility_targeting(weights, constraints['target_volatility'])
        
        return weights
    
    def _apply_sector_constraints(self, weights: np.ndarray, sector_limits: Dict[str, float]) -> np.ndarray:
        """Apply sector allocation limits"""
        # This is a simplified implementation
        # In practice, you'd need asset sector information
        return weights
    
    def _apply_volatility_targeting(self, weights: np.ndarray, target_vol: float) -> np.ndarray:
        """Apply volatility targeting"""
        # This would require covariance matrix
        # Simplified implementation
        return weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1"""
        total_weight = np.sum(weights)
        if total_weight > 0:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)
    
    def _calculate_portfolio_metrics(self, 
                                   weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'variance': portfolio_variance
        }
    
    def _calculate_risk_contributions(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset"""
        portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
        if portfolio_volatility > 0:
            marginal_risk = covariance_matrix @ weights / portfolio_volatility
            return weights * marginal_risk
        else:
            return np.zeros(len(weights))
    
    def _calculate_diversification_ratio(self, 
                                       weights: np.ndarray,
                                       volatilities: np.ndarray,
                                       covariance_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
        weighted_volatilities = np.sum(weights * volatilities)
        
        if portfolio_volatility > 0:
            return weighted_volatilities / portfolio_volatility
        else:
            return 1.0
    
    def _check_constraints(self, weights: np.ndarray, constraints: Optional[Dict[str, Any]] = None) -> bool:
        """Check if weights satisfy constraints"""
        if not constraints:
            return True
        
        # Check weight bounds
        if np.any(weights < self.min_weight) or np.any(weights > self.max_weight):
            return False
        
        # Check sum constraint
        if abs(np.sum(weights) - 1.0) > self.tolerance:
            return False
        
        return True
    
    def compare_strategies(self, assets: List[AssetMetrics]) -> Dict[str, AllocationResult]:
        """Compare all allocation strategies"""
        results = {}
        
        for strategy in AllocationStrategy:
            try:
                result = self.allocate_portfolio(assets, strategy)
                results[strategy.value] = result
            except Exception as e:
                print(f"Failed to allocate using {strategy.value}: {e}")
        
        return results
    
    def get_optimal_strategy(self, assets: List[AssetMetrics], 
                           objective: str = 'sharpe') -> Tuple[AllocationStrategy, AllocationResult]:
        """Find optimal strategy based on objective"""
        results = self.compare_strategies(assets)
        
        if not results:
            raise ValueError("No allocation strategies succeeded")
        
        if objective == 'sharpe':
            best_strategy = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
        elif objective == 'return':
            best_strategy = max(results.keys(), key=lambda k: results[k].expected_return)
        elif objective == 'volatility':
            best_strategy = min(results.keys(), key=lambda k: results[k].expected_volatility)
        elif objective == 'diversification':
            best_strategy = max(results.keys(), key=lambda k: results[k].diversification_ratio)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        return AllocationStrategy(best_strategy), results[best_strategy]


# Convenience functions
def create_allocator(config_path: str = "config/app_config.yaml") -> PortfolioAllocator:
    """Create a portfolio allocator instance"""
    return PortfolioAllocator(config_path)


def allocate_portfolio(assets: List[AssetMetrics], 
                     strategy: AllocationStrategy,
                     config_path: str = "config/app_config.yaml") -> AllocationResult:
    """Quick function to allocate portfolio"""
    allocator = PortfolioAllocator(config_path)
    return allocator.allocate_portfolio(assets, strategy)


if __name__ == "__main__":
    # Example usage
    allocator = PortfolioAllocator()
    
    # Create sample assets
    assets = [
        AssetMetrics(
            ticker="AAPL",
            expected_return=0.12,
            volatility=0.20,
            sharpe_ratio=0.5,
            beta=1.1,
            correlation={"TSLA": 0.3, "NVDA": 0.4},
            market_cap=2000000000000,
            sector="Technology",
            sentiment_score=0.6
        ),
        AssetMetrics(
            ticker="TSLA",
            expected_return=0.18,
            volatility=0.35,
            sharpe_ratio=0.46,
            beta=1.8,
            correlation={"AAPL": 0.3, "NVDA": 0.5},
            market_cap=800000000000,
            sector="Automotive",
            sentiment_score=0.7
        ),
        AssetMetrics(
            ticker="NVDA",
            expected_return=0.15,
            volatility=0.30,
            sharpe_ratio=0.43,
            beta=1.5,
            correlation={"AAPL": 0.4, "TSLA": 0.5},
            market_cap=1200000000000,
            sector="Technology",
            sentiment_score=0.8
        )
    ]
    
    # Test different strategies
    strategies = [AllocationStrategy.EQUAL_WEIGHT, 
                 AllocationStrategy.MAXIMUM_SHARPE,
                 AllocationStrategy.RISK_PARITY,
                 AllocationStrategy.KELLY_CRITERION]
    
    for strategy in strategies:
        try:
            result = allocator.allocate_portfolio(assets, strategy)
            print(f"\n{strategy.value.upper()} Allocation:")
            print(f"  Expected Return: {result.expected_return:.3f}")
            print(f"  Expected Volatility: {result.expected_volatility:.3f}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
            print(f"  Weights: {result.weights}")
        except Exception as e:
            print(f"Failed {strategy.value}: {e}")
    
    # Compare all strategies
    print("\n" + "="*50)
    print("STRATEGY COMPARISON")
    print("="*50)
    
    comparison = allocator.compare_strategies(assets)
    
    for strategy_name, result in comparison.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Return: {result.expected_return:.3f}")
        print(f"  Volatility: {result.expected_volatility:.3f}")
        print(f"  Sharpe: {result.sharpe_ratio:.3f}")
        print(f"  Diversification: {result.diversification_ratio:.3f}")
    
    # Find optimal strategy
    optimal_strategy, optimal_result = allocator.get_optimal_strategy(assets, 'sharpe')
    print(f"\nOptimal Strategy (Sharpe): {optimal_strategy.value}")
    print(f"Optimal Sharpe Ratio: {optimal_result.sharpe_ratio:.3f}") 