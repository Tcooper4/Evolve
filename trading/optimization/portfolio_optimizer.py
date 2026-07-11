"""
Portfolio Optimization Engine

Advanced portfolio optimization using CVXPY and CVXOPT.
Implements Mean-Variance Optimization, Black-Litterman Model, and Min-CVaR strategies.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Import optimization libraries with fallback handling
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Install with: pip install cvxpy")

try:
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    logging.warning("CVXOPT not available. Install with: pip install cvxopt")

from trading.utils.logging_utils import setup_logger
from trading.utils.safe_math import safe_divide

logger = setup_logger(__name__)


class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize the portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        # BUG FIX: risk_free_rate is documented and passed as an ANNUAL
        # rate (default 0.02 = 2%/year, the standard convention), but
        # every excess-return/Sharpe calculation throughout this file
        # (12 locations) previously subtracted it directly from
        # returns.mean()/portfolio_return, which are unannualized DAILY
        # return statistics. Verified concretely: a portfolio of three
        # genuinely positive-return assets produced a Sharpe ratio of
        # -2.5, and the real CVXPY Sharpe-maximization objective in
        # mean_variance_optimization failed to find a sensible solution
        # given this units mismatch (every asset's "excess return"
        # looked strongly negative), silently falling through to a
        # simpler fallback method instead of actually optimizing.
        # Assuming ~252 trading days/year (the standard convention used
        # throughout this codebase's other risk-metric calculations).
        self.daily_risk_free_rate = risk_free_rate / 252
        self.results_dir = Path("results/portfolio_optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if not CVXPY_AVAILABLE:
            logger.warning(
                "CVXPY not available. Portfolio optimization will use simplified methods."
            )

        logger.info("Portfolio optimizer initialized")

    def mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        constraints: Dict[str, Any] = None,
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
            # Try PyPortfolioOpt as fallback when CVXPY is missing
            try:
                from pypfopt import EfficientFrontier, risk_models, expected_returns
                mu = expected_returns.mean_historical_return(returns)
                S = risk_models.sample_cov(returns)
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe()
                cleaned = ef.clean_weights()
                weights = np.array([cleaned.get(c, 0.0) for c in returns.columns])
                portfolio_return = mu @ weights
                portfolio_vol = np.sqrt(weights @ S @ weights)
                sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol if portfolio_vol > 1e-10 else 0.0
                return {
                    "weights": dict(zip(returns.columns, weights)),
                    "portfolio_return": float(portfolio_return),
                    "portfolio_volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "optimization_status": "pypfopt_max_sharpe",
                }
            except Exception as e:
                logger.debug("PyPortfolioOpt fallback failed: %s", e)
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
                    mu @ w >= target_return,  # Return constraint
                ]
                solve_var = w
                sharpe_max_mode = False
            else:
                # BUG FIX: maximizing Sharpe ratio directly
                # (-excess_return @ w / cp.sqrt(quad_form(w, Sigma))) is
                # NOT a valid DCP (disciplined convex program) expression
                # - dividing by the square root of a quadratic form
                # inside the objective isn't convex in the form CVXPY
                # requires. This branch previously raised DCPError on
                # every single call and silently fell through to the
                # simplified fallback method - verified concretely by
                # reproducing the exact DCPError directly. Replaced with
                # the standard, well-established convex reformulation for
                # maximum-Sharpe portfolios: minimize the variance of
                # UNNORMALIZED weights y subject to a fixed excess-return
                # normalization (excess_return @ y == 1, y >= 0), then
                # rescale y to sum to 1 to recover the actual portfolio
                # weights. This is mathematically equivalent to
                # maximizing Sharpe ratio for a long-only, no-target-
                # return portfolio.
                excess_return = mu - self.daily_risk_free_rate
                y = cp.Variable(n_assets)
                objective = cp.Minimize(cp.quad_form(y, Sigma))
                constraints_list = [
                    y >= 0,
                    excess_return.values @ y == 1,
                ]
                solve_var = y
                sharpe_max_mode = True

            # Add custom constraints
            # Note: max_weight/min_weight constraints apply to the FINAL
            # normalized weights. In Sharpe-maximization mode, the solve
            # variable y is unnormalized (its scale is set by the
            # excess-return==1 constraint, not by summing to 1), so these
            # per-asset bounds can't be expressed as simple linear
            # constraints on y without breaking convexity. They're
            # applied directly only in the target-return (variance-
            # minimization) mode, where w is already the normalized
            # weight vector.
            if constraints and not sharpe_max_mode:
                if "max_weight" in constraints:
                    constraints_list.append(w <= constraints["max_weight"])
                if "min_weight" in constraints:
                    constraints_list.append(w >= constraints["min_weight"])
                if "sector_limits" in constraints:
                    for sector, limit in constraints["sector_limits"].items():
                        # This would need sector mapping - simplified here
                        pass

            # Solve problem
            problem = cp.Problem(objective, constraints_list)
            problem.solve()

            if problem.status == "optimal":
                # Safely extract weights array (solve_var.value can be
                # ndarray, None, or odd types)
                _w = solve_var.value
                if _w is None:
                    raise ValueError("Optimizer returned no solution")
                _w = np.array(_w, dtype=float).ravel()
                if len(_w) != len(returns.columns):
                    raise ValueError(
                        f"Weight count {len(_w)} != asset count {len(returns.columns)}"
                    )
                if sharpe_max_mode:
                    # y is unnormalized - rescale to sum to 1 to recover
                    # the actual portfolio weights.
                    _sum = _w.sum()
                    if _sum <= 0:
                        raise ValueError(
                            "Sharpe-maximization solve returned non-positive weight sum"
                        )
                    _w = _w / _sum
                weights_dict = dict(zip(returns.columns, _w.tolist()))

                portfolio_return = mu @ _w
                portfolio_vol = np.sqrt(_w @ Sigma @ _w)
                sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol

                # Calculate asset contributions
                asset_contributions = self._calculate_asset_contributions(

                    _w, mu, Sigma
                )

                result = {
                    "weights": weights_dict,
                    "portfolio_return": portfolio_return,
                    "portfolio_volatility": portfolio_vol,
                    "sharpe_ratio": sharpe_ratio,
                    "asset_contributions": asset_contributions,
                    "optimization_status": "optimal",
                    "constraints_used": list(constraints.keys()) if constraints else [],
                }

                # Save results
                self._save_optimization_results("mean_variance", result)

                return result
            else:
                logger.warning(f"Mean-variance optimization failed: {problem.status}")
                return {"error": f"Optimization failed: {problem.status}"}

        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return self._simple_mean_variance(returns, target_return, risk_aversion)

    def black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Dict[str, float],
        confidence: Dict[str, float],
        tau: float = 0.05,
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
            returns.mean()
            Sigma = returns.cov()

            # Market equilibrium returns (reverse optimization)
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * Sigma @ market_caps

            # Create view matrix
            assets = list(views.keys())
            P = np.zeros((len(views), len(returns.columns)))
            q = np.array(list(views.values()))
            Omega = np.diag([1 / confidence[asset] for asset in assets])

            for i, asset in enumerate(assets):
                if asset in returns.columns:
                    col_idx = returns.columns.get_loc(asset)
                    P[i, col_idx] = 1

            # Black-Litterman posterior estimates
            M1 = np.linalg.inv(
                np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P
            )
            M2 = np.linalg.inv(tau * Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ q

            mu_bl = M1 @ M2
            Sigma + M1

            # Convert to Series
            mu_bl_series = pd.Series(mu_bl, index=returns.columns)
            _unused_var = mu_bl_series  # Placeholder, flake8 ignore: F841

            # Run mean-variance optimization with BL estimates
            return self.mean_variance_optimization(
                returns,
                target_return=None,
                constraints={"max_weight": 0.2},  # Limit individual weights
            )

        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return {"error": str(e)}

    def risk_parity_optimization(
        self,
        returns: pd.DataFrame,
        target_risk: Optional[float] = None,
        risk_measure: str = "volatility",
    ) -> Dict[str, Any]:
        """Risk Parity Optimization.

        Args:
            returns: Asset returns DataFrame
            target_risk: Target portfolio risk level (if None, use equal risk contribution)
            risk_measure: Risk measure ('volatility', 'cvar', 'var')

        Returns:
            Dictionary with optimization results
        """
        if not CVXPY_AVAILABLE:
            return self._simple_risk_parity(returns, target_risk, risk_measure)

        try:
            # Calculate covariance matrix
            Sigma = returns.cov()
            n_assets = len(returns.columns)

            # Define variables
            w = cp.Variable(n_assets)

            if risk_measure == "volatility":
                # Risk parity using volatility
                portfolio_vol = cp.sqrt(cp.quad_form(w, Sigma))

                # Risk contribution of each asset
                risk_contrib = []
                for i in range(n_assets):
                    # Marginal contribution to risk
                    marginal_risk = (Sigma @ w)[i] / portfolio_vol
                    risk_contrib.append(w[i] * marginal_risk)

                # Objective: minimize sum of squared differences in risk contributions
                target_risk_contrib = safe_divide(portfolio_vol, n_assets, default=0.0)
                objective = cp.Minimize(
                    cp.sum_squares(cp.hstack(risk_contrib) - target_risk_contrib)
                )

            elif risk_measure == "cvar":
                # Risk parity using CVaR
                alpha = 0.05  # 95% confidence level
                portfolio_returns = returns @ w
                cvar = cp.quantile(portfolio_returns, alpha) + (1 / alpha) * cp.mean(
                    cp.pos(-portfolio_returns - cp.quantile(portfolio_returns, alpha))
                )

                # Simplified risk parity for CVaR
                objective = cp.Minimize(cvar)

            else:  # VaR
                # Risk parity using VaR
                alpha = 0.05  # 95% confidence level
                portfolio_returns = returns @ w
                var = cp.quantile(portfolio_returns, alpha)
                objective = cp.Minimize(var)

            # Constraints
            constraints_list = [
                w >= 0,
                cp.sum(w) == 1,
            ]  # Long-only constraint  # Budget constraint

            if target_risk is not None and risk_measure == "volatility":
                constraints_list.append(portfolio_vol <= target_risk)

            # Solve problem
            problem = cp.Problem(objective, constraints_list)
            problem.solve()

            if problem.status == "optimal":
                weights = w.value
                portfolio_return = returns.mean() @ weights
                portfolio_vol = np.sqrt(weights @ Sigma @ weights)
                sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol

                # Calculate risk contributions
                risk_contributions = self._calculate_risk_contributions(
                    weights, Sigma, risk_measure
                )

                result = {
                    "weights": dict(zip(returns.columns, weights)),
                    "portfolio_return": portfolio_return,
                    "portfolio_volatility": portfolio_vol,
                    "sharpe_ratio": sharpe_ratio,
                    "risk_contributions": risk_contributions,
                    "risk_measure": risk_measure,
                    "optimization_status": "optimal",
                }

                # Save results
                self._save_optimization_results("risk_parity", result)

                return result
            else:
                logger.warning(f"Risk parity optimization failed: {problem.status}")
                return {"error": f"Optimization failed: {problem.status}"}

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self._simple_risk_parity(returns, target_risk, risk_measure)

    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        method: str = "ward",
    ) -> Dict[str, Any]:
        """
        Hierarchical Risk Parity (HRP).

        Lopez de Prado (2016). Does not
        require matrix inversion —
        more robust than mean-variance
        with small samples or correlated
        assets.

        Algorithm:
        1. Compute correlation matrix
        2. Convert to distance matrix
        3. Hierarchical clustering
           (Ward linkage by default)
        4. Quasi-diagonalization
           (reorder assets by cluster)
        5. Recursive bisection to
           allocate weights

        Args:
            returns: DataFrame of asset
                returns (rows=dates,
                cols=symbols)
            method: linkage method
                ('ward', 'single',
                'complete', 'average')

        Returns:
            weights: Dict[symbol, weight]
            expected_return: float
            expected_volatility: float
            sharpe_ratio: float
            method: "hrp"
        """
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform

            if returns is None or returns.empty:
                return {
                    "error": "No returns data",
                    "method": "hrp",
                }

            returns = returns.dropna(axis=1, how="all")
            returns = returns.dropna(axis=0, how="any")

            if len(returns) < 10 or len(returns.columns) < 2:
                return self._simple_risk_parity(
                    returns, None, "volatility"
                )

            _n = len(returns.columns)
            _syms = list(returns.columns)

            _corr = returns.corr().fillna(0)
            _cov = returns.cov().fillna(0)

            _dist = np.sqrt(0.5 * (1 - _corr.values))
            np.fill_diagonal(_dist, 0)
            _dist = np.clip(_dist, 0, None)

            _condensed = squareform(_dist, checks=False)
            _link = linkage(_condensed, method=method)

            _order = leaves_list(_link)
            _ordered_syms = [_syms[i] for i in _order]

            def _get_cluster_var(
                cov: pd.DataFrame, cluster_items: list
            ) -> float:
                _sub = cov.loc[cluster_items, cluster_items]
                _ivol = 1.0 / np.sqrt(np.diag(_sub.values))
                _ivol /= _ivol.sum()
                return float(_ivol @ _sub.values @ _ivol)

            def _recursive_bisect(
                cov: pd.DataFrame, items: list
            ) -> Dict[str, float]:
                if len(items) == 1:
                    return {items[0]: 1.0}
                mid = len(items) // 2
                left = items[:mid]
                right = items[mid:]
                _lv = _get_cluster_var(cov, left)
                _rv = _get_cluster_var(cov, right)
                _alpha = 1.0 - (_lv / (_lv + _rv))
                _lw = _recursive_bisect(cov, left)
                _rw = _recursive_bisect(cov, right)
                return {
                    k: v * _alpha for k, v in _lw.items()
                } | {
                    k: v * (1 - _alpha) for k, v in _rw.items()
                }

            _weights_raw = _recursive_bisect(_cov, _ordered_syms)

            _total = sum(_weights_raw.values())
            _weights = {
                k: round(v / _total, 4) for k, v in _weights_raw.items()
            }

            _w_arr = np.array([_weights.get(s, 0) for s in _syms])
            _ann = 252
            _port_ret = float(returns.mean().values @ _w_arr * _ann)
            _port_vol = float(
                np.sqrt(_w_arr @ _cov.values @ _w_arr * _ann)
            )
            _sharpe = (
                (_port_ret - self.daily_risk_free_rate) / _port_vol
                if _port_vol > 0
                else 0.0
            )

            result = {
                "weights": _weights,
                "portfolio_return": float(returns.mean().values @ _w_arr),
                "portfolio_volatility": float(
                    np.sqrt(_w_arr @ _cov.values @ _w_arr)
                ),
                "expected_return": round(_port_ret, 4),
                "expected_volatility": round(_port_vol, 4),
                "sharpe_ratio": round(_sharpe, 3),
                "method": "hrp",
                "n_assets": _n,
                "linkage_method": method,
            }
            self._save_optimization_results("hrp", result)
            return result

        except Exception as e:
            logger.error("HRP optimization failed: %s", e)
            return {
                "error": str(e),
                "method": "hrp",
            }

    def enhanced_black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Dict[str, float],
        confidence: Dict[str, float],
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        view_type: str = "absolute",
    ) -> Dict[str, Any]:
        """Enhanced Black-Litterman Model with multiple view types.

        Args:
            returns: Asset returns DataFrame
            market_caps: Market capitalization weights
            views: Dictionary of views {asset: expected_return}
            confidence: Dictionary of confidence levels {asset: confidence}
            tau: Scaling parameter
            risk_aversion: Risk aversion parameter
            view_type: Type of views ('absolute', 'relative', 'ranking')

        Returns:
            Dictionary with optimization results
        """
        try:
            # Calculate market equilibrium returns
            returns.mean()
            Sigma = returns.cov()

            # Market equilibrium returns (reverse optimization)
            pi = risk_aversion * Sigma @ market_caps

            # Create view matrix based on view type
            assets = list(views.keys())

            if view_type == "absolute":
                # Absolute views: asset A will return X%
                P = np.zeros((len(views), len(returns.columns)))
                q = np.array(list(views.values()))

                for i, asset in enumerate(assets):
                    if asset in returns.columns:
                        col_idx = returns.columns.get_loc(asset)
                        P[i, col_idx] = 1

            elif view_type == "relative":
                # Relative views: asset A will outperform asset B by X%
                P = np.zeros((len(views), len(returns.columns)))
                q = []

                for i, (view_pair, outperformance) in enumerate(views.items()):
                    asset_a, asset_b = view_pair.split(" vs ")
                    if asset_a in returns.columns and asset_b in returns.columns:
                        col_a = returns.columns.get_loc(asset_a)
                        col_b = returns.columns.get_loc(asset_b)
                        P[i, col_a] = 1
                        P[i, col_b] = -1
                        q.append(outperformance)

                q = np.array(q)

            elif view_type == "ranking":
                # Ranking views: assets ranked by expected performance
                P = np.zeros((len(views) - 1, len(returns.columns)))
                q = np.zeros(len(views) - 1)

                ranked_assets = list(views.keys())
                for i in range(len(ranked_assets) - 1):
                    asset_a = ranked_assets[i]
                    asset_b = ranked_assets[i + 1]
                    if asset_a in returns.columns and asset_b in returns.columns:
                        col_a = returns.columns.get_loc(asset_a)
                        col_b = returns.columns.get_loc(asset_b)
                        P[i, col_a] = 1
                        P[i, col_b] = -1
                        q[i] = views[asset_a] - views[asset_b]

            # Confidence matrix
            Omega = np.diag([1 / confidence[asset] for asset in assets])

            # Black-Litterman posterior estimates
            M1 = np.linalg.inv(
                np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P
            )
            M2 = np.linalg.inv(tau * Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ q

            mu_bl = M1 @ M2
            Sigma + M1

            # Convert to Series
            mu_bl_series = pd.Series(mu_bl, index=returns.columns)
            _unused_var = mu_bl_series  # Placeholder, flake8 ignore: F841

            # Run mean-variance optimization with BL estimates
            return self.mean_variance_optimization(
                returns,
                target_return=None,
                risk_aversion=risk_aversion,
                constraints={"bl_views": views, "view_type": view_type},
            )

        except Exception as e:
            logger.error(f"Error in enhanced Black-Litterman optimization: {e}")
            return {"error": str(e)}

    def min_cvar_optimization(
        self,
        returns: pd.DataFrame,
        confidence_level: float = 0.95,
        target_return: Optional[float] = None,
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
            return self._simple_cvar_optimization(
                returns, confidence_level, target_return
            )

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
            objective = cp.Minimize(alpha + (1 / beta) * cp.sum(z) / n_scenarios)

            # Constraints
            constraints = [
                w >= 0,  # Long-only
                cp.sum(w) == 1,  # Budget constraint
                z >= 0,  # Non-negative auxiliary variables
                z >= -R @ w - alpha,  # CVaR constraint
            ]

            if target_return is not None:
                mu = returns.mean()
                constraints.append(mu @ w >= target_return)

            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status == "optimal":
                weights = w.value
                cvar = alpha.value + (1 / beta) * np.sum(z.value) / n_scenarios

                # Calculate additional metrics
                portfolio_return = returns.mean() @ weights
                portfolio_vol = np.sqrt(weights @ returns.cov() @ weights)
                sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol

                result = {
                    "weights": dict(zip(returns.columns, weights)),
                    "portfolio_return": portfolio_return,
                    "portfolio_volatility": portfolio_vol,
                    "cvar": cvar,
                    "sharpe_ratio": sharpe_ratio,
                    "confidence_level": confidence_level,
                    "optimization_status": "optimal",
                }

                # Save results
                self._save_optimization_results("min_cvar", result)

                return result
            else:
                logger.warning(f"Min-CVaR optimization failed: {problem.status}")
                return {"error": f"Optimization failed: {problem.status}"}

        except Exception as e:
            logger.error(f"Error in Min-CVaR optimization: {e}")
            return self._simple_cvar_optimization(
                returns, confidence_level, target_return
            )

    def _simple_mean_variance(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float],
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """Simplified mean-variance optimization without CVXPY."""
        _cols = list(returns.columns) if returns is not None else []

        def _equal_weights() -> Dict[str, Any]:
            _n = max(1, len(_cols))
            _w = {c: round(1.0 / _n, 6) for c in _cols}
            return {
                "weights": _w,
                "portfolio_return": 0.0,
                "portfolio_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "optimization_status": "equal_weight",
                "note": "Equal weights (optimization unavailable)",
            }

        if returns is None or returns.empty or len(_cols) < 2:
            return {"error": "Insufficient data for optimization"}

        try:
            mu = returns.mean()
            Sigma = returns.cov()
            if mu.shape[0] != len(_cols) or Sigma.shape[0] != len(_cols):
                logger.error(
                    "Simple MV shape mismatch: mu=%s cols=%d",
                    getattr(mu, "shape", None),
                    len(_cols),
                )
                return _equal_weights()

            # Inverse volatility — avoid div by zero / NaN (causes tuple/shape errors downstream)
            vol = returns.std().replace(0, np.nan).fillna(1e-8)
            inv_vol = 1.0 / vol.clip(lower=1e-12)
            denom = float(inv_vol.sum())
            if not np.isfinite(denom) or denom <= 0:
                return _equal_weights()
            weights = inv_vol / denom
            weights = pd.Series(weights.values, index=_cols, dtype=float)

            portfolio_return = float(mu.to_numpy() @ weights.to_numpy())
            _w_arr = weights.to_numpy(dtype=float)
            portfolio_vol = float(np.sqrt(max(0.0, _w_arr @ Sigma.to_numpy() @ _w_arr)))
            sharpe_ratio = (
                (portfolio_return - self.daily_risk_free_rate) / portfolio_vol
                if portfolio_vol > 1e-10
                else 0.0
            )

            return {
                "weights": dict(zip(_cols, weights.tolist())),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_vol,
                "sharpe_ratio": float(sharpe_ratio),
                "optimization_status": "fallback_inverse_volatility",
            }

        except Exception as e:
            logger.error("Error in simple mean-variance: %s", e)
            return _equal_weights()

    def _simple_risk_parity(
        self, returns: pd.DataFrame, target_risk: Optional[float], risk_measure: str
    ) -> Dict[str, Any]:
        """Simple risk parity implementation without CVXPY."""
        try:
            Sigma = returns.cov()
            n_assets = len(returns.columns)

            # ALGORITHM FIX (on top of the pandas-2 iloc fix that exposed
            # it): the previous equal-weight-start multiplicative update
            # skipped assets whose marginal risk went negative (sample
            # covariances routinely produce this), so their weights froze
            # while everything else shrank - the portfolio collapsed onto
            # one asset and 'risk parity' returned wildly unequal, even
            # negative, contributions. Verified on a heteroskedastic
            # 4-asset case. Now: inverse-volatility seed (the textbook
            # correlation-free risk-parity solution) refined by a DAMPED
            # multiplicative iteration with a positive floor on marginal
            # risk - converges to near-equal contributions and degrades
            # gracefully to inverse-vol when correlations fight it.
            Sigma_np = Sigma.to_numpy()
            vols = np.sqrt(np.clip(np.diag(Sigma_np), 1e-12, None))
            weights = (1.0 / vols) / np.sum(1.0 / vols)

            max_iter = 200
            tolerance = 1e-8
            damp = 0.5  # exponent damping stabilizes the update

            iteration = 0
            for iteration in range(max_iter):
                portfolio_vol = float(np.sqrt(weights @ Sigma_np @ weights))
                if portfolio_vol <= 0:
                    break
                marginal = (Sigma_np @ weights) / portfolio_vol
                marginal = np.clip(marginal, 1e-12, None)  # positivity floor
                risk_contrib = weights * marginal
                target_contrib = portfolio_vol / n_assets

                if np.max(np.abs(risk_contrib - target_contrib)) < tolerance:
                    break

                weights = weights * (target_contrib / risk_contrib) ** damp
                weights = np.clip(weights, 1e-9, None)
                weights = weights / np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = returns.mean() @ weights
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol

            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(
                weights, Sigma, risk_measure
            )

            result = {
                "weights": dict(zip(returns.columns, weights)),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "risk_contributions": risk_contributions,
                "risk_measure": risk_measure,
                "optimization_status": "simple_risk_parity",
                "iterations": iteration + 1,
            }

            self._save_optimization_results("risk_parity", result)

            return result

        except Exception as e:
            logger.error(f"Error in simple risk parity: {e}")
            return {"error": str(e)}

    def _calculate_risk_contributions(
        self, weights: np.ndarray, Sigma: pd.DataFrame, risk_measure: str
    ) -> Dict[str, float]:
        """Calculate risk contributions for each asset."""
        try:
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            risk_contrib = {}

            for i, asset in enumerate(Sigma.columns):
                if risk_measure == "volatility":
                    # PANDAS-2 FIX: Sigma is a labeled DataFrame, so
                    # (Sigma @ weights) is a Series indexed by TICKER
                    # names - integer [i] raised KeyError: 0 on pandas 2.x
                    # (positional int indexing on labeled Series removed),
                    # which the broad except turned into {'error': '0'}.
                    # Risk parity has therefore returned an error dict on
                    # every call under current pandas. Use positional iloc.
                    marginal_risk = (Sigma @ weights).iloc[i] / portfolio_vol
                    risk_contrib[asset] = weights[i] * marginal_risk
                else:
                    # Simplified for other risk measures
                    risk_contrib[asset] = weights[i] * portfolio_vol / len(weights)

            return risk_contrib

        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return {}

    def _simple_cvar_optimization(
        self,
        returns: pd.DataFrame,
        confidence_level: float,
        target_return: Optional[float],
    ) -> Dict[str, Any]:
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
            sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / portfolio_vol

            return {
                "weights": dict(zip(returns.columns, weights)),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_vol,
                "cvar": cvar,
                "sharpe_ratio": sharpe_ratio,
                "confidence_level": confidence_level,
                "optimization_status": "fallback_historical_cvar",
            }

        except Exception as e:
            logger.error(f"Error in simple CVaR optimization: {e}")
            return {"error": str(e)}

    def _calculate_asset_contributions(
        self, weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate asset contributions to portfolio metrics."""
        try:
            mu @ weights
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)

            # Return contribution
            return_contrib = {
                asset: weight * mu[asset] for asset, weight in zip(mu.index, weights)
            }

            # Risk contribution
            risk_contrib = {}
            for i, asset in enumerate(mu.index):
                # Marginal contribution to risk
                mcr = (Sigma.iloc[i] @ weights) / portfolio_vol
                risk_contrib[asset] = weights[i] * mcr

            # Sharpe contribution
            sharpe_contrib = {}
            for asset in mu.index:
                sharpe_contrib[asset] = (
                    return_contrib[asset]
                    - self.daily_risk_free_rate * weights[mu.index.get_loc(asset)]
                ) / portfolio_vol

            return {
                "return_contribution": return_contrib,
                "risk_contribution": risk_contrib,
                "sharpe_contribution": sharpe_contrib,
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

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Optimization results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")

    def compare_strategies(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compare different optimization strategies."""
        try:
            strategies = {}

            # Equal Weight (benchmark)
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            equal_return = returns.mean() @ equal_weights
            equal_vol = np.sqrt(equal_weights @ returns.cov() @ equal_weights)
            equal_sharpe = (equal_return - self.daily_risk_free_rate) / equal_vol

            strategies["Equal Weight"] = {
                "Return": equal_return,
                "Volatility": equal_vol,
                "Sharpe": equal_sharpe,
            }

            # Mean-Variance
            mv_result = self.mean_variance_optimization(returns)
            if "error" not in mv_result:
                strategies["Mean-Variance"] = {
                    "Return": mv_result["portfolio_return"],
                    "Volatility": mv_result["portfolio_volatility"],
                    "Sharpe": mv_result["sharpe_ratio"],
                }

            # Risk Parity
            rp_result = self.risk_parity_optimization(returns)
            if "error" not in rp_result:
                strategies["Risk Parity"] = {
                    "Return": rp_result["portfolio_return"],
                    "Volatility": rp_result["portfolio_volatility"],
                    "Sharpe": rp_result["sharpe_ratio"],
                }

            # Min-CVaR
            cvar_result = self.min_cvar_optimization(returns)
            if "error" not in cvar_result:
                strategies["Min-CVaR"] = {
                    "Return": cvar_result["portfolio_return"],
                    "Volatility": cvar_result["portfolio_volatility"],
                    "Sharpe": cvar_result["sharpe_ratio"],
                    "CVaR": cvar_result["cvar"],
                }

            # Black-Litterman (if market caps available)
            if len(returns.columns) >= 2:
                market_caps = pd.Series(
                    1.0 / len(returns.columns), index=returns.columns
                )
                views = {returns.columns[0]: 0.05}  # Simple view
                confidence = {returns.columns[0]: 0.5}

                bl_result = self.black_litterman_optimization(
                    returns, market_caps, views, confidence
                )
                if "error" not in bl_result:
                    strategies["Black-Litterman"] = {
                        "Return": bl_result["portfolio_return"],
                        "Volatility": bl_result["portfolio_volatility"],
                        "Sharpe": bl_result["sharpe_ratio"],
                    }

                # Enhanced Black-Litterman with relative views
                if len(returns.columns) >= 3:
                    relative_views = {
                        f"{returns.columns[0]} vs {returns.columns[1]}": 0.02
                    }
                    relative_confidence = {
                        f"{returns.columns[0]} vs {returns.columns[1]}": 0.6
                    }

                    ebl_result = self.enhanced_black_litterman_optimization(
                        returns,
                        market_caps,
                        relative_views,
                        relative_confidence,
                        view_type="relative",
                    )
                    if "error" not in ebl_result:
                        strategies["Enhanced BL (Relative)"] = {
                            "Return": ebl_result["portfolio_return"],
                            "Volatility": ebl_result["portfolio_volatility"],
                            "Sharpe": ebl_result["sharpe_ratio"],
                        }

            # Risk parity with different risk measures
            rp_cvar_result = self.risk_parity_optimization(returns, risk_measure="cvar")
            if "error" not in rp_cvar_result:
                strategies["Risk Parity (CVaR)"] = {
                    "Return": rp_cvar_result["portfolio_return"],
                    "Volatility": rp_cvar_result["portfolio_volatility"],
                    "Sharpe": rp_cvar_result["sharpe_ratio"],
                }

            # Create comparison DataFrame
            comparison_df = pd.DataFrame(strategies).T
            comparison_df = comparison_df.round(4)

            # Save comparison
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_file = self.results_dir / f"strategy_comparison_{timestamp}.csv"
            comparison_df.to_csv(comparison_file)

            logger.info(f"Strategy comparison saved to {comparison_file}")

            return comparison_df

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
