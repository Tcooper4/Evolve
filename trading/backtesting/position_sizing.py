"""
Position Sizing for Backtesting

This module contains position sizing methods and calculations for the backtesting system.
"""

import logging
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from trading.utils.safe_math import safe_divide, safe_rsi, safe_price_momentum

# Try to import scipy
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError as e:
    print("⚠️ scipy not available. Disabling optimization-based position sizing.")
    print(f"   Missing: {e}")
    minimize = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
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


class PositionSizingEngine:
    """Engine for calculating position sizes using various methods."""

    def __init__(
        self,
        cash: float,
        risk_per_trade: float = 0.02,
        risk_free_rate: float = 0.02,
        max_leverage: float = 1.0,
    ):
        """Initialize position sizing engine.

        Args:
            cash: Available cash for trading
            risk_per_trade: Risk per trade as fraction of portfolio
            risk_free_rate: Risk-free rate for calculations
            max_leverage: Maximum leverage allowed
        """
        self.cash = cash
        self.risk_per_trade = risk_per_trade
        self.risk_free_rate = risk_free_rate
        self.max_leverage = max_leverage
        self.trade_history: List[Dict[str, Any]] = []

    def calculate_position_size(
        self,
        method: PositionSizing,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using specified method.

        Args:
            method: Position sizing method to use
            asset: Asset symbol
            price: Current asset price
            strategy: Strategy name
            signal: Trading signal (-1 to 1)
            data: Historical price data
            positions: Current positions

        Returns:
            Position size as fraction of portfolio
        """
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
            PositionSizing.MACHINE_LEARNING: self._calculate_machine_learning_size,
        }

        sizing_method = sizing_methods.get(method)
        if not sizing_method:
            raise ValueError(f"Unknown position sizing method: {method}")

        try:
            position_size = sizing_method(
                asset, price, strategy, signal, data, positions
            )

            # Apply position size limits
            position_size = max(
                min(position_size, MAX_POSITION_SIZE), MIN_POSITION_SIZE
            )

            # Apply leverage limits
            position_size = min(position_size, self.max_leverage)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size with {method}: {e}")
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

    def _calculate_equal_weighted_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate equal-weighted position size."""
        return 0.1  # 10% of portfolio

    def _calculate_risk_based_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate risk-based position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        returns = data[asset].pct_change().dropna()
        volatility = returns.std()

        if volatility == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Risk-based sizing: risk_per_trade / volatility
        if volatility > 1e-10:
            position_size = self.risk_per_trade / volatility
        else:
            # Use equal-weighted sizing if volatility is zero
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )
        return min(position_size, MAX_POSITION_SIZE)

    def _calculate_kelly_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate Kelly criterion position size."""
        # Get historical trades for this asset/strategy
        asset_trades = [
            t
            for t in self.trade_history
            if t["asset"] == asset and t["strategy"] == strategy
        ]

        if len(asset_trades) < 10:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate win rate and average win/loss
        winning_trades = [t for t in asset_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in asset_trades if t.get("pnl", 0) < 0]

        if not winning_trades or not losing_trades:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        win_rate = len(winning_trades) / len(asset_trades)
        avg_win = np.mean([t["pnl"] for t in winning_trades])
        avg_loss = abs(np.mean([t["pnl"] for t in losing_trades]))

        if avg_loss == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Kelly formula using safe division utility
        from trading.utils.safe_math import safe_kelly_fraction
        kelly_fraction = safe_kelly_fraction(avg_win, avg_loss, win_rate)

        # Apply signal strength
        position_size = kelly_fraction * abs(signal)
        return max(0, min(position_size, MAX_POSITION_SIZE))

    def _calculate_fixed_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate fixed position size."""
        return 0.05  # 5% of portfolio

    def _calculate_volatility_adjusted_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate volatility-adjusted position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        returns = data[asset].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]

        if np.isnan(volatility) or volatility == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Adjust position size inversely to volatility
        base_size = self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )
        return base_size * (1 / (1 + volatility))

    def _calculate_optimal_f_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using Optimal f."""
        # Get historical trades for this asset/strategy
        asset_trades = [
            t
            for t in self.trade_history
            if t["asset"] == asset and t["strategy"] == strategy
        ]

        if len(asset_trades) < 10:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate win/loss ratio and win rate
        winning_trades = [t for t in asset_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in asset_trades if t.get("pnl", 0) < 0]

        if not winning_trades or not losing_trades:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        win_rate = len(winning_trades) / len(asset_trades)
        avg_win = np.mean([t["pnl"] for t in winning_trades])
        avg_loss = abs(np.mean([t["pnl"] for t in losing_trades]))

        if avg_win == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate Optimal f
        f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        position_size = f * abs(signal)
        return max(0, min(position_size, MAX_POSITION_SIZE))

    def _calculate_risk_parity_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using Risk Parity approach."""
        if not positions:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate risk contribution for all assets
        returns_data = {}
        for pos_asset in positions:
            if pos_asset in data.columns:
                returns_data[pos_asset] = data[pos_asset].pct_change().dropna()

        if asset in data.columns:
            returns_data[asset] = data[asset].pct_change().dropna()

        if len(returns_data) < 2:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate covariance matrix
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov()

        if cov_matrix.empty or cov_matrix.isnull().any().any():
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate risk contribution
        risk_contrib = np.sqrt(np.diag(cov_matrix))
        total_risk = np.sum(risk_contrib)

        if total_risk == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate target weights
        target_weights = safe_divide(risk_contrib, total_risk, default=1.0 / len(risk_contrib))

        # Calculate position size for new asset
        if asset in target_weights:
            return target_weights[asset] * abs(signal)
        else:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

    def _calculate_black_litterman_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using Black-Litterman model."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        returns = data[asset].pct_change().dropna()
        market_return = returns.mean()
        market_risk = returns.std()

        if market_risk == 0:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate investor views
        view_return = signal * market_return
        view_confidence = abs(signal)

        # Calculate Black-Litterman weights
        tau = 0.05  # Prior uncertainty
        1 / view_confidence if view_confidence > 0 else 1

        # Calculate posterior returns and weights
        prior_return = market_return
        prior_cov = market_risk**2

        post_return = (prior_return + tau * view_return) / (1 + tau)
        post_cov = prior_cov * (1 + tau)

        # Calculate position size
        position_size = (post_return - self.risk_free_rate) / (
            post_cov * self.risk_per_trade
        )
        return max(0, min(position_size, MAX_POSITION_SIZE))

    def _calculate_martingale_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using Martingale strategy."""
        # Get recent trades for this asset
        asset_trades = [
            t
            for t in self.trade_history
            if t["asset"] == asset and t["strategy"] == strategy
        ]

        if not asset_trades:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Double position size after losses
        last_trade = asset_trades[-1]
        if last_trade.get("pnl", 0) < 0:
            return min(last_trade.get("position_size", 0.1) * 2, MAX_POSITION_SIZE)

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_anti_martingale_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate position size using Anti-Martingale strategy."""
        # Get recent trades for this asset
        asset_trades = [
            t
            for t in self.trade_history
            if t["asset"] == asset and t["strategy"] == strategy
        ]

        if not asset_trades:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Increase position size after wins
        last_trade = asset_trades[-1]
        if last_trade.get("pnl", 0) > 0:
            return min(last_trade.get("position_size", 0.1) * 1.5, MAX_POSITION_SIZE)

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_half_kelly_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate half Kelly position size."""
        kelly_size = self._calculate_kelly_size(
            asset, price, strategy, signal, data, positions
        )
        return kelly_size * 0.5

    def _calculate_dynamic_kelly_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate dynamic Kelly position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Use recent data for dynamic Kelly
        recent_data = data[asset].tail(252)  # Last year
        returns = recent_data.pct_change().dropna()

        if len(returns) < 30:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate rolling Kelly
        rolling_kelly = []
        window = 60

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window : i]
            win_rate = (window_returns > 0).mean()
            avg_win = window_returns[window_returns > 0].mean()
            avg_loss = abs(window_returns[window_returns < 0].mean())

            if avg_loss > 0 and avg_win > 0:
                b = safe_divide(avg_win, avg_loss, default=0.0)
                p = win_rate
                q = 1 - p
                kelly = (b * p - q) / b
                rolling_kelly.append(max(0, kelly))
            else:
                rolling_kelly.append(0)

        if rolling_kelly:
            dynamic_kelly = np.mean(rolling_kelly[-10:])  # Average of last 10
            return dynamic_kelly * abs(signal)

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_correlation_adjusted_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate correlation-adjusted position size."""
        if not positions or asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate correlation with existing positions
        correlations = []
        asset_returns = data[asset].pct_change().dropna()

        for pos_asset in positions:
            if pos_asset in data.columns:
                pos_returns = data[pos_asset].pct_change().dropna()
                # Align returns
                common_index = asset_returns.index.intersection(pos_returns.index)
                if len(common_index) > 10:
                    corr = asset_returns.loc[common_index].corr(
                        pos_returns.loc[common_index]
                    )
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

        if correlations:
            avg_correlation = np.mean(correlations)
            # Reduce position size for high correlation
            base_size = self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )
            return base_size * (1 - avg_correlation * 0.5)

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_momentum_weighted_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate momentum-weighted position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate momentum
        returns = data[asset].pct_change().dropna()
        if len(returns) < 20:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        momentum = returns.rolling(window=20).mean().iloc[-1]

        if np.isnan(momentum):
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Adjust position size based on momentum
        base_size = self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )
        momentum_factor = 1 + momentum * 2  # Scale momentum effect
        return base_size * momentum_factor

    def _calculate_mean_variance_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available. Using equal-weighted position sizing.")
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )
        """Calculate mean-variance optimal position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Get all available assets
        available_assets = [asset] + list(positions.keys())
        available_assets = [a for a in available_assets if a in data.columns]

        if len(available_assets) < 2:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate returns for all assets
        returns_data = {}
        for a in available_assets:
            returns_data[a] = data[a].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 30:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        if cov_matrix.empty or cov_matrix.isnull().any().any():
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Mean-variance optimization
        n_assets = len(available_assets)

        def portfolio_stats(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe_ratio  # Minimize negative Sharpe ratio

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1

        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                portfolio_stats,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                asset_index = available_assets.index(asset)
                return optimal_weights[asset_index] * abs(signal)
        except Exception as e:
            logger.warning(f"Mean-variance optimization failed: {e}")

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_minimum_variance_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate minimum variance position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Get all available assets
        available_assets = [asset] + list(positions.keys())
        available_assets = [a for a in available_assets if a in data.columns]

        if len(available_assets) < 2:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate returns for all assets
        returns_data = {}
        for a in available_assets:
            returns_data[a] = data[a].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 30:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate covariance matrix
        cov_matrix = returns_df.cov()

        if cov_matrix.empty or cov_matrix.isnull().any().any():
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Minimum variance optimization
        n_assets = len(available_assets)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1

        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                portfolio_variance,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                asset_index = available_assets.index(asset)
                return optimal_weights[asset_index] * abs(signal)
        except Exception as e:
            logger.warning(f"Minimum variance optimization failed: {e}")

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_maximum_diversification_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate maximum diversification position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Get all available assets
        available_assets = [asset] + list(positions.keys())
        available_assets = [a for a in available_assets if a in data.columns]

        if len(available_assets) < 2:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate returns for all assets
        returns_data = {}
        for a in available_assets:
            returns_data[a] = data[a].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 30:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate covariance matrix and volatilities
        cov_matrix = returns_df.cov()
        volatilities = returns_df.std()

        if cov_matrix.empty or cov_matrix.isnull().any().any():
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Maximum diversification optimization
        n_assets = len(available_assets)

        def diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            weighted_vol = np.sum(weights * volatilities)
            return (
                -weighted_vol / portfolio_vol
            )  # Minimize negative diversification ratio

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1

        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                diversification_ratio,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                asset_index = available_assets.index(asset)
                return optimal_weights[asset_index] * abs(signal)
        except Exception as e:
            logger.warning(f"Maximum diversification optimization failed: {e}")

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_risk_efficient_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate risk-efficient position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Get all available assets
        available_assets = [asset] + list(positions.keys())
        available_assets = [a for a in available_assets if a in data.columns]

        if len(available_assets) < 2:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate returns for all assets
        returns_data = {}
        for a in available_assets:
            returns_data[a] = data[a].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 30:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        if cov_matrix.empty or cov_matrix.isnull().any().any():
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Risk-efficient optimization (maximize return per unit of risk)
        n_assets = len(available_assets)

        def risk_efficiency(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1

        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                risk_efficiency,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                asset_index = available_assets.index(asset)
                return optimal_weights[asset_index] * abs(signal)
        except Exception as e:
            logger.warning(f"Risk-efficient optimization failed: {e}")

        return self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

    def _calculate_adaptive_weight_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate adaptive weight position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate market regime
        returns = data[asset].pct_change().dropna()
        if len(returns) < 60:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate volatility regime
        recent_vol = returns.tail(20).std()
        long_term_vol = returns.tail(252).std()

        # Safely calculate vol_ratio with division-by-zero protection
        vol_ratio = safe_divide(recent_vol, long_term_vol, default=1.0)
        if vol_ratio <= 1e-10:
            # Default to neutral if no long-term volatility, or use equal-weighted sizing
            if recent_vol == 0:
                return self._calculate_equal_weighted_size(
                    asset, price, strategy, signal, data, positions
                )
            vol_ratio = 1.0  # Default to neutral if no long-term volatility

        # Calculate trend regime
        recent_return = returns.tail(20).mean()
        long_term_return = returns.tail(252).mean()

        # Adaptive sizing based on regime
        base_size = self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

        # Adjust for volatility regime
        if vol_ratio > 1.5:  # High volatility
            vol_adjustment = 0.5
        elif vol_ratio < 0.5:  # Low volatility
            vol_adjustment = 1.5
        else:
            vol_adjustment = 1.0

        # Adjust for trend regime
        if recent_return > long_term_return * 1.2:  # Strong uptrend
            trend_adjustment = 1.2
        elif recent_return < long_term_return * 0.8:  # Strong downtrend
            trend_adjustment = 0.8
        else:
            trend_adjustment = 1.0

        return base_size * vol_adjustment * trend_adjustment

    def _calculate_regime_based_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate regime-based position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        returns = data[asset].pct_change().dropna()
        if len(returns) < 60:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate regime indicators
        volatility = returns.rolling(window=20).std().iloc[-1]
        momentum = returns.rolling(window=20).mean().iloc[-1]
        skewness = returns.rolling(window=60).skew().iloc[-1]

        if np.isnan(volatility) or np.isnan(momentum) or np.isnan(skewness):
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Determine regime
        base_size = self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

        # High volatility regime
        if volatility > returns.std() * 1.5:
            regime_adjustment = 0.5
        # Low volatility regime
        elif volatility < returns.std() * 0.5:
            regime_adjustment = 1.5
        # Normal volatility regime
        else:
            regime_adjustment = 1.0

        # Momentum regime
        if momentum > 0.01:  # Strong positive momentum
            momentum_adjustment = 1.3
        elif momentum < -0.01:  # Strong negative momentum
            momentum_adjustment = 0.7
        else:
            momentum_adjustment = 1.0

        # Skewness regime
        if skewness > 1:  # Positive skew (fat right tail)
            skew_adjustment = 1.2
        elif skewness < -1:  # Negative skew (fat left tail)
            skew_adjustment = 0.8
        else:
            skew_adjustment = 1.0

        return base_size * regime_adjustment * momentum_adjustment * skew_adjustment

    def _calculate_factor_based_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate factor-based position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Calculate factor scores
        returns = data[asset].pct_change().dropna()
        if len(returns) < 60:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Value factor (inverse of price relative to moving average)
        sma_20 = data[asset].rolling(window=20).mean().iloc[-1]
        sma_60 = data[asset].rolling(window=60).mean().iloc[-1]

        # Use safe_price_momentum for ratio calculation
        # Note: safe_price_momentum calculates (current - reference) / reference
        # For sma_60/sma_20 ratio, we use: 1 + safe_price_momentum(sma_60, sma_20)
        value_factor = 1.0 + safe_price_momentum(sma_60, sma_20)

        # Momentum factor
        momentum_factor = returns.rolling(window=20).mean().iloc[-1]
        if np.isnan(momentum_factor):
            momentum_factor = 0

        # Volatility factor (inverse)
        volatility_factor = 1 / (1 + returns.rolling(window=20).std().iloc[-1])
        if np.isnan(volatility_factor):
            volatility_factor = 1.0

        # Quality factor (based on return consistency)
        positive_returns = (returns > 0).rolling(window=20).mean().iloc[-1]
        if np.isnan(positive_returns):
            positive_returns = 0.5
        quality_factor = positive_returns

        # Combine factors
        base_size = self._calculate_equal_weighted_size(
            asset, price, strategy, signal, data, positions
        )

        factor_score = (
            value_factor + (1 + momentum_factor) + volatility_factor + quality_factor
        ) / 4
        return base_size * factor_score

    def _calculate_machine_learning_size(
        self,
        asset: str,
        price: float,
        strategy: str,
        signal: float,
        data: pd.DataFrame,
        positions: Dict[str, float],
    ) -> float:
        """Calculate machine learning-based position size."""
        if asset not in data.columns:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        # Get historical trades for training
        asset_trades = [
            t
            for t in self.trade_history
            if t["asset"] == asset and t["strategy"] == strategy
        ]

        if len(asset_trades) < 20:
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

        try:
            # Prepare features
            returns = data[asset].pct_change().dropna()
            if len(returns) < 60:
                return self._calculate_equal_weighted_size(
                    asset, price, strategy, signal, data, positions
                )

            # Calculate technical features
            features = {
                "volatility": returns.rolling(window=20).std().iloc[-1],
                "momentum": returns.rolling(window=20).mean().iloc[-1],
                "rsi": self._calculate_rsi(returns),
                "signal": signal,
                "price": price,
            }

            # Prepare training data
            X_train = []
            y_train = []

            for trade in asset_trades[-20:]:  # Use last 20 trades
                if "features" in trade and "pnl" in trade:
                    X_train.append(list(trade["features"].values()))
                    y_train.append(1 if trade["pnl"] > 0 else 0)

            if len(X_train) < 10:
                return self._calculate_equal_weighted_size(
                    asset, price, strategy, signal, data, positions
                )

            # Train simple model (Random Forest)
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            # Predict optimal position size
            current_features = list(features.values())
            predicted_prob = model.predict([current_features])[0]

            # Convert probability to position size
            base_size = self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )
            return base_size * predicted_prob

        except Exception as e:
            logger.warning(f"Machine learning sizing failed: {e}")
            return self._calculate_equal_weighted_size(
                asset, price, strategy, signal, data, positions
            )

    def _calculate_rsi(self, returns: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator using safe division."""
        if len(returns) < period:
            return 50.0

        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)

        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]

        # Use safe_divide to handle zero avg_loss
        rs = safe_divide(avg_gain, avg_loss, default=0.0)
        # Use safe_divide for final RSI calculation to handle edge case where rs = -1
        denominator = 1 + rs
        rsi = 100 - safe_divide(100, denominator, default=50.0)

        return rsi if not np.isnan(rsi) and np.isfinite(rsi) else 50.0

    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add trade to history for position sizing calculations."""
        self.trade_history.append(trade_data)

        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history.copy()

    def clear_trade_history(self) -> None:
        """Clear trade history."""
        self.trade_history.clear()
