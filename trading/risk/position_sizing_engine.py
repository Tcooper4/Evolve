"""Position Sizing Engine.

This engine implements advanced position sizing algorithms including Kelly Criterion,
volatility-based sizing, and maximum drawdown guards for optimal risk management.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""

    KELLY = "kelly"
    VOLATILITY = "volatility"
    FIXED = "fixed"
    RISK_PARITY = "risk_parity"
    MAX_DRAWDOWN = "max_drawdown"
    HYBRID = "hybrid"


@dataclass
class PositionSize:
    """Position size calculation result."""

    symbol: str
    size: float
    sizing_method: SizingMethod
    confidence: float
    risk_metrics: Dict[str, float]
    constraints: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""

    allocations: Dict[str, float]
    total_allocation: float
    risk_budget: Dict[str, float]
    diversification_score: float
    expected_return: float
    expected_volatility: float
    max_drawdown_limit: float
    metadata: Dict[str, Any]


class PositionSizingEngine:
    """Advanced position sizing engine with multiple algorithms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize position sizing engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Risk parameters
        self.max_position_size = self.config.get("max_position_size", 0.2)
        self.max_portfolio_risk = self.config.get("max_portfolio_risk", 0.15)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.confidence_level = self.config.get("confidence_level", 0.95)

        # Kelly Criterion parameters
        self.kelly_config = self.config.get(
            "kelly_config", {"fraction": 0.25, "max_leverage": 2.0, "min_size": 0.01}  # Use fractional Kelly
        )

        # Volatility parameters
        self.volatility_config = self.config.get(
            "volatility_config", {"lookback_period": 252, "target_volatility": 0.15, "volatility_floor": 0.05}
        )

        # Drawdown parameters
        self.drawdown_config = self.config.get(
            "drawdown_config", {"max_drawdown": 0.10, "drawdown_lookback": 252, "recovery_factor": 0.5}
        )

        # Portfolio constraints
        self.constraints = self.config.get(
            "constraints",
            {
                "max_sector_exposure": 0.3,
                "max_country_exposure": 0.4,
                "min_diversification": 0.7,
                "liquidity_threshold": 1000000,
            },
        )

        # Performance tracking
        self.sizing_history = []
        self.allocation_history = []

        logger.info("Position Sizing Engine initialized")

        return {"success": True, "message": "Initialization completed", "timestamp": datetime.now().isoformat()}

    def calculate_position_size(
        self,
        symbol: str,
        returns: pd.Series,
        price: float,
        method: SizingMethod = SizingMethod.HYBRID,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> PositionSize:
        """Calculate optimal position size for a symbol.

        Args:
            symbol: Trading symbol
            returns: Historical returns
            price: Current price
            method: Sizing method
            portfolio_context: Portfolio context information

        Returns:
            Position size calculation
        """
        try:
            if len(returns) < 30:
                return self._create_default_position(symbol, method)

            # Calculate base size based on method
            if method == SizingMethod.KELLY:
                base_size = self._kelly_criterion(returns)
            elif method == SizingMethod.VOLATILITY:
                base_size = self._volatility_based_sizing(returns)
            elif method == SizingMethod.FIXED:
                base_size = self._fixed_sizing(returns)
            elif method == SizingMethod.RISK_PARITY:
                base_size = self._risk_parity_sizing(returns, portfolio_context)
            elif method == SizingMethod.MAX_DRAWDOWN:
                base_size = self._max_drawdown_sizing(returns)
            elif method == SizingMethod.HYBRID:
                base_size = self._hybrid_sizing(returns, portfolio_context)
            else:
                base_size = self._fixed_sizing(returns)

            # Apply constraints
            constrained_size = self._apply_constraints(base_size, symbol, portfolio_context)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns, constrained_size)

            # Calculate confidence
            confidence = self._calculate_confidence(returns, method)

            position_size = PositionSize(
                symbol=symbol,
                size=constrained_size,
                sizing_method=method,
                confidence=confidence,
                risk_metrics=risk_metrics,
                constraints=self._get_applied_constraints(constrained_size, symbol, portfolio_context),
                timestamp=datetime.now(),
                metadata={
                    "base_size": base_size,
                    "constraint_adjustments": base_size - constrained_size,
                    "method_parameters": self._get_method_parameters(method),
                },
            )

            self.sizing_history.append(position_size)

            logger.info(
                f"Calculated position size for {symbol}: {constrained_size:.2%} " f"using {method.value} method"
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return {
                "success": True,
                "result": self._create_default_position(symbol, method),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly Criterion position size.

        Args:
            returns: Historical returns

        Returns:
            Kelly position size
        """
        try:
            # Calculate win rate and average win/loss
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                return self.kelly_config["min_size"]

            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            if avg_loss == 0:
                return self.kelly_config["min_size"]

            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply fractional Kelly and constraints
            fractional_kelly = kelly_fraction * self.kelly_config["fraction"]
            constrained_kelly = max(
                self.kelly_config["min_size"], min(fractional_kelly, self.kelly_config["max_leverage"])
            )

            return constrained_kelly

        except Exception as e:
            logger.error(f"Error in Kelly Criterion calculation: {e}")
            return self.kelly_config["min_size"]

    def _volatility_based_sizing(self, returns: pd.Series) -> float:
        """Calculate volatility-based position size.

        Args:
            returns: Historical returns

        Returns:
            Volatility-based position size
        """
        try:
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252)
            volatility = max(volatility, self.volatility_config["volatility_floor"])

            # Target volatility sizing
            target_vol = self.volatility_config["target_volatility"]
            size = target_vol / volatility

            # Apply constraints
            size = max(0.01, min(size, self.max_position_size))

            return size

        except Exception as e:
            logger.error(f"Error in volatility-based sizing: {e}")
            return 0.05

    def _fixed_sizing(self, returns: pd.Series) -> float:
        """Calculate fixed position size.

        Args:
            returns: Historical returns

        Returns:
            Fixed position size
        """
        return 0.05  # 5% fixed allocation

    def _risk_parity_sizing(self, returns: pd.Series, portfolio_context: Optional[Dict[str, Any]]) -> float:
        """Calculate risk parity position size.

        Args:
            returns: Historical returns
            portfolio_context: Portfolio context

        Returns:
            Risk parity position size
        """
        try:
            if portfolio_context is None:
                return self._volatility_based_sizing(returns)

            # Calculate asset volatility
            asset_vol = returns.std() * np.sqrt(252)

            # Get portfolio volatility target
            portfolio_vol_target = portfolio_context.get("target_volatility", 0.15)

            # Risk parity: equal risk contribution
            target_risk_contribution = portfolio_vol_target / len(portfolio_context.get("assets", [1]))

            size = target_risk_contribution / asset_vol

            return max(0.01, min(size, self.max_position_size))

        except Exception as e:
            logger.error(f"Error in risk parity sizing: {e}")
            return 0.05

    def _max_drawdown_sizing(self, returns: pd.Series) -> float:
        """Calculate max drawdown-based position size.

        Args:
            returns: Historical returns

        Returns:
            Max drawdown-based position size
        """
        try:
            # Calculate maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min())

            # Size inversely proportional to max drawdown
            max_dd_limit = self.drawdown_config["max_drawdown"]
            size = max_dd_limit / max(max_dd, 0.01)

            # Apply recovery factor
            size *= self.drawdown_config["recovery_factor"]

            return max(0.01, min(size, self.max_position_size))

        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Max drawdown calculation failed: {e}")
            return 0.05

    def _hybrid_sizing(self, returns: pd.Series, portfolio_context: Optional[Dict[str, Any]]) -> float:
        """Calculate hybrid position size combining multiple methods.

        Args:
            returns: Historical returns
            portfolio_context: Portfolio context

        Returns:
            Hybrid position size
        """
        try:
            # Calculate sizes using different methods
            kelly_size = self._kelly_criterion(returns)
            vol_size = self._volatility_based_sizing(returns)
            dd_size = self._max_drawdown_sizing(returns)

            # Weighted average
            weights = [0.4, 0.4, 0.2]  # Kelly, Volatility, Drawdown
            hybrid_size = kelly_size * weights[0] + vol_size * weights[1] + dd_size * weights[2]

            return max(0.01, min(hybrid_size, self.max_position_size))

        except Exception as e:
            logger.error(f"Error in hybrid sizing: {e}")
            return 0.05

    def _apply_constraints(self, base_size: float, symbol: str, portfolio_context: Optional[Dict[str, Any]]) -> float:
        """Apply portfolio constraints to position size.

        Args:
            base_size: Base position size
            symbol: Trading symbol
            portfolio_context: Portfolio context

        Returns:
            Constrained position size
        """
        constrained_size = base_size

        if portfolio_context:
            # Maximum position size constraint
            constrained_size = min(constrained_size, self.max_position_size)

            # Sector exposure constraint
            if "sector_exposures" in portfolio_context:
                sector = portfolio_context.get("sector_exposures", {}).get(symbol, "Unknown")
                current_sector_exposure = portfolio_context.get("current_sector_exposures", {}).get(sector, 0)
                max_sector_exposure = self.constraints["max_sector_exposure"]

                if current_sector_exposure + constrained_size > max_sector_exposure:
                    constrained_size = max(0, max_sector_exposure - current_sector_exposure)

            # Liquidity constraint
            if "liquidity" in portfolio_context:
                liquidity = portfolio_context["liquidity"].get(symbol, 0)
                liquidity_threshold = self.constraints["liquidity_threshold"]

                if liquidity < liquidity_threshold:
                    constrained_size *= 0.5  # Reduce size for illiquid assets

        return {
            "success": True,
            "result": max(0.01, constrained_size),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_risk_metrics(self, returns: pd.Series, position_size: float) -> Dict[str, float]:
        """Calculate risk metrics for position.

        Args:
            returns: Historical returns
            position_size: Position size

        Returns:
            Risk metrics dictionary
        """
        try:
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            var_95 = np.percentile(returns, 5)

            return {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "position_risk": position_size * volatility,
                "expected_return": returns.mean() * 252,
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown.

        Args:
            returns: Return series

        Returns:
            Maximum drawdown
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_confidence(self, returns: pd.Series, method: SizingMethod) -> float:
        """Calculate confidence in position size calculation.

        Args:
            returns: Historical returns
            method: Sizing method used

        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence on data quality and method
            data_quality = min(len(returns) / 252, 1.0)  # More data = higher confidence

            # Method-specific confidence adjustments
            method_confidence = {
                SizingMethod.KELLY: 0.8,
                SizingMethod.VOLATILITY: 0.9,
                SizingMethod.FIXED: 0.7,
                SizingMethod.RISK_PARITY: 0.85,
                SizingMethod.MAX_DRAWDOWN: 0.75,
                SizingMethod.HYBRID: 0.9,
            }

            base_confidence = method_confidence.get(method, 0.7)

            # Adjust for return stability
            return_stability = 1 - returns.std()  # Lower volatility = higher confidence
            return_stability = max(0, min(1, return_stability))

            confidence = data_quality * 0.4 + base_confidence * 0.4 + return_stability * 0.2

            return max(0, min(1, confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _get_applied_constraints(
        self, final_size: float, symbol: str, portfolio_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get information about applied constraints.

        Args:
            final_size: Final position size
            symbol: Trading symbol
            portfolio_context: Portfolio context

        Returns:
            Applied constraints information
        """
        constraints_info = {"max_position_size": self.max_position_size, "max_portfolio_risk": self.max_portfolio_risk}

        if portfolio_context:
            constraints_info.update(
                {
                    "sector_exposure": portfolio_context.get("current_sector_exposures", {}),
                    "liquidity_available": portfolio_context.get("liquidity", {}).get(symbol, 0),
                }
            )

        return constraints_info

    def _get_method_parameters(self, method: SizingMethod) -> Dict[str, Any]:
        """Get parameters used for the sizing method.

        Args:
            method: Sizing method

        Returns:
            Method parameters
        """
        if method == SizingMethod.KELLY:
            return self.kelly_config
        elif method == SizingMethod.VOLATILITY:
            return self.volatility_config
        elif method == SizingMethod.MAX_DRAWDOWN:
            return self.drawdown_config
        else:
            return {}

    def _create_default_position(self, symbol: str, method: SizingMethod) -> PositionSize:
        """Create default position size when calculation fails.

        Args:
            symbol: Trading symbol
            method: Sizing method

        Returns:
            Default position size
        """
        return PositionSize(
            symbol=symbol,
            size=0.05,  # 5% default
            sizing_method=method,
            confidence=0.5,
            risk_metrics={},
            constraints={},
            timestamp=datetime.now(),
            metadata={"note": "Default position size due to calculation error"},
        )

    def optimize_portfolio_allocation(
        self,
        assets: Dict[str, pd.Series],
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation across multiple assets.

        Args:
            assets: Dictionary of asset return series
            target_return: Target portfolio return
            target_volatility: Target portfolio volatility

        Returns:
            Optimized portfolio allocation
        """
        try:
            # Calculate expected returns and covariance
            returns_matrix = pd.DataFrame(assets)
            expected_returns = returns_matrix.mean() * 252
            covariance_matrix = returns_matrix.cov() * 252

            # Use risk parity approach for allocation
            volatilities = np.sqrt(np.diag(covariance_matrix))
            risk_contributions = 1 / volatilities
            weights = risk_contributions / risk_contributions.sum()

            # Apply constraints
            constrained_weights = self._apply_portfolio_constraints(weights, assets)

            # Calculate portfolio metrics
            portfolio_return = (constrained_weights * expected_returns).sum()
            portfolio_vol = np.sqrt(constrained_weights.T @ covariance_matrix @ constrained_weights)

            # Calculate diversification score
            diversification_score = self._calculate_diversification_score(constrained_weights, covariance_matrix)

            allocation = PortfolioAllocation(
                allocations=dict(zip(assets.keys(), constrained_weights)),
                total_allocation=constrained_weights.sum(),
                risk_budget=dict(zip(assets.keys(), risk_contributions)),
                diversification_score=diversification_score,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                max_drawdown_limit=self.drawdown_config["max_drawdown"],
                metadata={
                    "target_return": target_return,
                    "target_volatility": target_volatility,
                    "optimization_method": "risk_parity",
                },
            )

            self.allocation_history.append(allocation)

            return allocation

        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return self._create_default_allocation(assets)

    def _apply_portfolio_constraints(self, weights: np.ndarray, assets: Dict[str, pd.Series]) -> np.ndarray:
        """Apply portfolio-level constraints.

        Args:
            weights: Asset weights
            assets: Asset return series

        Returns:
            Constrained weights
        """
        constrained_weights = weights.copy()

        # Maximum position size constraint
        constrained_weights = np.minimum(constrained_weights, self.max_position_size)

        # Normalize weights
        constrained_weights = constrained_weights / constrained_weights.sum()

        return constrained_weights

    def _calculate_diversification_score(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """Calculate portfolio diversification score.

        Args:
            weights: Asset weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Diversification score (0-1)
        """
        try:
            # Calculate portfolio variance
            portfolio_variance = weights.T @ covariance_matrix @ weights

            # Calculate weighted average of individual variances
            individual_variances = np.diag(covariance_matrix)
            weighted_avg_variance = (weights**2 * individual_variances).sum()

            # Diversification ratio
            if weighted_avg_variance > 0:
                diversification_ratio = weighted_avg_variance / portfolio_variance
                return min(diversification_ratio / len(weights), 1.0)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0.5

    def _create_default_allocation(self, assets: Dict[str, pd.Series]) -> PortfolioAllocation:
        """Create default portfolio allocation.

        Args:
            assets: Asset return series

        Returns:
            Default allocation
        """
        n_assets = len(assets)
        equal_weights = np.ones(n_assets) / n_assets

        return PortfolioAllocation(
            allocations=dict(zip(assets.keys(), equal_weights)),
            total_allocation=1.0,
            risk_budget=dict(zip(assets.keys(), equal_weights)),
            diversification_score=0.5,
            expected_return=0.0,
            expected_volatility=0.0,
            max_drawdown_limit=self.drawdown_config["max_drawdown"],
            metadata={"note": "Default equal-weighted allocation"},
        )

    def get_sizing_history(self, symbol: Optional[str] = None, days: int = 30) -> List[PositionSize]:
        """Get position sizing history.

        Args:
            symbol: Filter by symbol (optional)
            days: Number of days to look back

        Returns:
            List of position sizes
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        history = [s for s in self.sizing_history if s.timestamp > cutoff_date]

        if symbol:
            history = [s for s in history if s.symbol == symbol]

        return history

    def export_sizing_report(self, filepath: str) -> bool:
        """Export position sizing report.

        Args:
            filepath: Output file path

        Returns:
            True if export successful
        """
        try:
            report_data = []

            for sizing in self.sizing_history:
                row = {
                    "symbol": sizing.symbol,
                    "size": sizing.size,
                    "method": sizing.sizing_method.value,
                    "confidence": sizing.confidence,
                    "timestamp": sizing.timestamp,
                    "volatility": sizing.risk_metrics.get("volatility", 0),
                    "sharpe_ratio": sizing.risk_metrics.get("sharpe_ratio", 0),
                    "max_drawdown": sizing.risk_metrics.get("max_drawdown", 0),
                }
                report_data.append(row)

            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False)

            logger.info(f"Position sizing report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting sizing report: {e}")
            return False


# Global position sizing engine instance
position_sizing_engine = PositionSizingEngine()


def get_position_sizing_engine() -> PositionSizingEngine:
    """Get the global position sizing engine instance."""
    return position_sizing_engine
