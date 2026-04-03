"""Portfolio Allocator

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

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.common_helpers import load_config

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Try to import scipy
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError as e:
    logger.warning(
        "⚠️ scipy not available. Disabling optimization-based portfolio allocation."
    )
    logger.warning(f"   Missing: {e}")
    minimize = None
    SCIPY_AVAILABLE = False


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
        self.portfolio_config = self.config.get("portfolio", {})

        # Allocation constraints
        self.max_weight = self.portfolio_config.get("max_weight", 0.3)
        self.min_weight = self.portfolio_config.get("min_weight", 0.01)
        self.target_volatility = self.portfolio_config.get("target_volatility", 0.15)
        self.risk_free_rate = self.portfolio_config.get("risk_free_rate", 0.02)

        # Optimization parameters
        self.max_iterations = self.portfolio_config.get("max_iterations", 1000)
        self.tolerance = self.portfolio_config.get("tolerance", 1e-6)

        # Kelly criterion parameters
        self.kelly_fraction = self.portfolio_config.get("kelly_fraction", 0.25)
        self.max_kelly_weight = self.portfolio_config.get("max_kelly_weight", 0.5)

    def allocate_portfolio(
        self,
        assets: List[AssetMetrics],
        strategy: AllocationStrategy,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> AllocationResult:
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
            weights = self._maximum_sharpe_allocation(
                expected_returns, covariance_matrix, constraints
            )
        elif strategy == AllocationStrategy.RISK_PARITY:
            weights = self._risk_parity_allocation(covariance_matrix, constraints)
        elif strategy == AllocationStrategy.KELLY_CRITERION:
            weights = self._kelly_criterion_allocation(assets, constraints)
        elif strategy == AllocationStrategy.BLACK_LITTERMAN:
            weights = self._black_litterman_allocation(assets, constraints)
        elif strategy == AllocationStrategy.MEAN_VARIANCE:
            weights = self._mean_variance_allocation(
                expected_returns, covariance_matrix, constraints
            )
        elif strategy == AllocationStrategy.MAXIMUM_DIVERSIFICATION:
            weights = self._maximum_diversification_allocation(
                covariance_matrix, constraints
            )
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

        # Apply constraints and normalize
        weights = self._apply_constraints(weights, constraints)
        weights = self._normalize_weights(weights)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            weights, expected_returns, covariance_matrix
        )

        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)

        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(
            weights, volatilities, covariance_matrix
        )

        return AllocationResult(
            strategy=strategy,
            weights=dict(zip(tickers, weights)),
            expected_return=portfolio_metrics["expected_return"],
            expected_volatility=portfolio_metrics["expected_volatility"],
            sharpe_ratio=portfolio_metrics["sharpe_ratio"],
            risk_contribution=dict(zip(tickers, risk_contributions)),
            diversification_ratio=diversification_ratio,
            constraints_satisfied=self._check_constraints(weights, constraints),
            optimization_status="success",
            metadata=portfolio_metrics,
        )

    def _equal_weight_allocation(self, assets: List[AssetMetrics]) -> np.ndarray:
        """Equal weight allocation"""
        n_assets = len(assets)
        return np.ones(n_assets) / n_assets

    # Remaining strategy and helper implementations follow the archived allocator.

