"""
Risk & Performance Utilities
==============================
INTEGRATION NOTES:
- Drop into: utils/risk_metrics.py
- Wire into: pages/4_Trade.py risk section
- Call pattern:
    from utils.risk_metrics import (
        calculate_var, calculate_cvar,
        kelly_criterion, PerformanceMetrics
    )

Dependencies: numpy, pandas, scipy (all in requirements)

Related: ``trading/risk/risk_metrics.py`` provides Plotly/rolling-dashboard metrics;
keep this module as the canonical VaR/Kelly/``PerformanceMetrics`` stack.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# VALUE AT RISK & CVAR
# ─────────────────────────────────────────

def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
    horizon_days: int = 1,
    portfolio_value: float = 10000.0,
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Daily return series
        confidence: Confidence level (0.95 = 95%)
        method: 'historical', 'parametric', or 'cornish_fisher'
        horizon_days: Forecast horizon in days
        portfolio_value: Portfolio value in dollars

    Returns:
        Dict with VaR in both % and dollar terms
    """
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            return {"error": "Insufficient data (need 30+ observations)"}

        alpha = 1 - confidence

        if method == "historical":
            var_pct = float(np.percentile(returns_clean, alpha * 100))

        elif method == "parametric":
            from scipy import stats
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            var_pct = float(stats.norm.ppf(alpha, mu, sigma))

        elif method == "cornish_fisher":
            # Cornish-Fisher expansion accounts for skewness and kurtosis
            from scipy import stats
            z = stats.norm.ppf(alpha)
            skew = float(returns_clean.skew())
            kurt = float(returns_clean.kurtosis())
            z_cf = (
                z
                + (z**2 - 1) * skew / 6
                + (z**3 - 3*z) * kurt / 24
                - (2*z**3 - 5*z) * skew**2 / 36
            )
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            var_pct = float(mu + z_cf * sigma)
        else:
            return {"error": f"Unknown method: {method}"}

        # Scale to horizon
        var_pct_horizon = var_pct * np.sqrt(horizon_days)
        var_dollar = abs(var_pct_horizon) * portfolio_value

        return {
            "var_pct": round(var_pct_horizon * 100, 3),
            "var_dollar": round(var_dollar, 2),
            "confidence": confidence,
            "horizon_days": horizon_days,
            "method": method,
            "interpretation": (
                f"With {confidence:.0%} confidence, maximum expected loss "
                f"over {horizon_days} day(s) is "
                f"${var_dollar:.2f} ({abs(var_pct_horizon)*100:.2f}%)"
            ),
        }

    except Exception as e:
        logger.warning("VaR calculation failed: %s", e)
        return {"error": str(e)}


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 10000.0,
) -> Dict[str, Any]:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    CVaR = expected loss given that loss exceeds VaR.
    More conservative and coherent risk measure than VaR.
    """
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            return {"error": "Insufficient data"}

        alpha = 1 - confidence
        var_threshold = np.percentile(returns_clean, alpha * 100)

        # CVaR = mean of returns below VaR threshold
        tail_returns = returns_clean[returns_clean <= var_threshold]
        cvar_pct = float(tail_returns.mean()) if len(tail_returns) > 0 else var_threshold

        cvar_pct_horizon = cvar_pct * np.sqrt(horizon_days)
        cvar_dollar = abs(cvar_pct_horizon) * portfolio_value

        var_result = calculate_var(
            returns, confidence, "historical",
            horizon_days, portfolio_value
        )
        var_dollar = var_result.get("var_dollar", 0)

        return {
            "cvar_pct": round(cvar_pct_horizon * 100, 3),
            "cvar_dollar": round(cvar_dollar, 2),
            "var_dollar": var_dollar,
            "tail_observations": len(tail_returns),
            "confidence": confidence,
            "horizon_days": horizon_days,
            "excess_over_var": round(cvar_dollar - var_dollar, 2),
            "interpretation": (
                f"Expected loss in worst {(1-confidence)*100:.0f}% of scenarios: "
                f"${cvar_dollar:.2f} ({abs(cvar_pct_horizon)*100:.2f}%)"
            ),
        }

    except Exception as e:
        logger.warning("CVaR calculation failed: %s", e)
        return {"error": str(e)}


# ─────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────

def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
    max_position: float = 0.20,
) -> Dict[str, Any]:
    """
    Calculate Kelly Criterion optimal position size.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average win as decimal (e.g., 0.05 for 5%)
        avg_loss: Average loss as decimal (positive number)
        fraction: Kelly fraction to use (0.25 = quarter Kelly, safer)
        max_position: Maximum allowed position size

    Returns:
        Dict with optimal position size and risk metrics
    """
    try:
        if avg_loss <= 0:
            return {"error": "avg_loss must be positive"}
        if not 0 < win_rate < 1:
            return {"error": "win_rate must be between 0 and 1"}

        lose_rate = 1 - win_rate
        odds = avg_win / avg_loss

        # Full Kelly: f* = (bp - q) / b
        # where b = odds, p = win_rate, q = lose_rate
        full_kelly = (odds * win_rate - lose_rate) / odds

        if full_kelly <= 0:
            return {
                "full_kelly": round(full_kelly, 4),
                "fractional_kelly": 0.0,
                "recommended_size": 0.0,
                "edge": round(full_kelly, 4),
                "interpretation": (
                    "Negative edge — do not take this trade"
                ),
                "is_positive_edge": False,
            }

        fractional_kelly = full_kelly * fraction
        recommended = min(fractional_kelly, max_position)

        # Expected value
        ev = win_rate * avg_win - lose_rate * avg_loss
        edge_ratio = ev / avg_loss  # edge per unit risk

        return {
            "full_kelly": round(full_kelly, 4),
            "fractional_kelly": round(fractional_kelly, 4),
            "recommended_size": round(recommended, 4),
            "recommended_pct": round(recommended * 100, 2),
            "win_rate": win_rate,
            "odds": round(odds, 3),
            "expected_value_per_trade": round(ev, 4),
            "edge_ratio": round(edge_ratio, 3),
            "fraction_used": fraction,
            "is_positive_edge": True,
            "interpretation": (
                f"Optimal position: {recommended*100:.1f}% of portfolio "
                f"({fraction:.0%} Kelly). "
                f"Expected edge: {ev*100:.2f}% per trade."
            ),
        }

    except Exception as e:
        logger.warning("Kelly Criterion calculation failed: %s", e)
        return {"error": str(e)}


def kelly_from_returns(
    returns: pd.Series,
    fraction: float = 0.25,
    max_position: float = 0.20,
) -> Dict[str, Any]:
    """
    Calculate Kelly Criterion from a return series.
    Estimates win rate, avg win, avg loss from historical data.
    """
    try:
        returns_clean = returns.dropna()
        wins = returns_clean[returns_clean > 0]
        losses = returns_clean[returns_clean < 0]

        if len(wins) == 0 or len(losses) == 0:
            return {"error": "Insufficient win/loss data"}

        win_rate = len(wins) / len(returns_clean)
        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))

        result = kelly_criterion(
            win_rate, avg_win, avg_loss, fraction, max_position
        )
        result["data_points"] = len(returns_clean)
        result["win_count"] = len(wins)
        result["loss_count"] = len(losses)
        return result

    except Exception as e:
        logger.warning("Kelly from returns failed: %s", e)
        return {"error": str(e)}


# ─────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a return series."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility_annual: float
    downside_deviation: float
    win_rate: float
    profit_factor: float
    beta: float
    alpha: float
    information_ratio: float
    var_95: float
    cvar_95: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Total Return": f"{self.total_return*100:.2f}%",
            "Annualized Return": f"{self.annualized_return*100:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown*100:.2f}%",
            "Max DD Duration (days)": self.max_drawdown_duration,
            "Annual Volatility": f"{self.volatility_annual*100:.2f}%",
            "Win Rate": f"{self.win_rate*100:.1f}%",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Beta": f"{self.beta:.3f}",
            "Alpha (annual)": f"{self.alpha*100:.2f}%",
            "Information Ratio": f"{self.information_ratio:.3f}",
            "VaR (95%)": f"{self.var_95*100:.2f}%",
            "CVaR (95%)": f"{self.cvar_95*100:.2f}%",
        }

    def grade(self) -> str:
        """Grade overall performance A-F."""
        score = 0
        if self.sharpe_ratio > 2.0: score += 3
        elif self.sharpe_ratio > 1.0: score += 2
        elif self.sharpe_ratio > 0.5: score += 1

        if self.max_drawdown > -0.10: score += 2
        elif self.max_drawdown > -0.20: score += 1

        if self.win_rate > 0.6: score += 2
        elif self.win_rate > 0.5: score += 1

        if self.calmar_ratio > 1.0: score += 2
        elif self.calmar_ratio > 0.5: score += 1

        if score >= 8: return "A"
        if score >= 6: return "B"
        if score >= 4: return "C"
        if score >= 2: return "D"
        return "F"


def compute_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        returns: Daily return series
        benchmark_returns: Benchmark (e.g., SPY) return series
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year

    Returns:
        PerformanceMetrics dataclass
    """
    r = returns.dropna()
    daily_rf = risk_free_rate / trading_days

    # Total and annualized return
    total_return = float((1 + r).prod() - 1)
    n_years = len(r) / trading_days
    annualized_return = float(
        (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    )

    # Volatility
    vol_daily = float(r.std())
    vol_annual = vol_daily * np.sqrt(trading_days)

    # Sharpe ratio
    excess_returns = r - daily_rf
    sharpe = float(
        excess_returns.mean()
        / (excess_returns.std() + 1e-10)
        * np.sqrt(trading_days)
    )

    # Sortino ratio (MAR = risk-free rate variant)
    # Downside deviation = std of returns below rf (excess vs daily_rf)
    # Note: some implementations use MAR=0 (negative returns only). This uses MAR=rf.
    downside = r[r < daily_rf] - daily_rf
    downside_dev = float(
        np.sqrt(np.mean(downside ** 2)) * np.sqrt(trading_days)
    ) if len(downside) > 0 else vol_annual
    sortino = float(
        (annualized_return - risk_free_rate) / downside_dev
    ) if downside_dev > 0 else 0.0

    # Max drawdown
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Max drawdown duration
    in_drawdown = drawdown < 0
    dd_duration = 0
    max_dd_duration = 0
    for val in in_drawdown:
        if val:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Calmar ratio
    calmar = float(
        annualized_return / abs(max_dd)
    ) if max_dd != 0 else 0.0

    # Win rate and profit factor
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / len(r)) if len(r) > 0 else 0.5
    profit_factor = float(
        wins.sum() / abs(losses.sum())
    ) if len(losses) > 0 and losses.sum() != 0 else 1.0

    # Beta and Alpha vs benchmark
    beta = 0.0
    alpha = annualized_return - risk_free_rate
    info_ratio = 0.0

    if benchmark_returns is not None:
        try:
            b = benchmark_returns.dropna()
            aligned = pd.concat([r, b], axis=1).dropna()
            if len(aligned) > 10 and aligned.shape[1] == 2:
                cov_matrix = aligned.cov()
                beta = float(
                    cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
                )
                bench_annual = float(
                    (1 + aligned.iloc[:, 1]).prod() ** (
                        trading_days / len(aligned)
                    ) - 1
                )
                alpha = annualized_return - (
                    risk_free_rate + beta * (bench_annual - risk_free_rate)
                )

                # Information ratio
                active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                if active_returns.std() > 0:
                    info_ratio = float(
                        active_returns.mean() / active_returns.std()
                        * np.sqrt(trading_days)
                    )
        except Exception as e:
            logger.debug("Beta/alpha calculation failed: %s", e)

    # VaR and CVaR
    var_result = calculate_var(r, 0.95)
    cvar_result = calculate_cvar(r, 0.95)
    var_95 = var_result.get("var_pct", 0) / 100 if "error" not in var_result else 0.0
    cvar_95 = cvar_result.get("cvar_pct", 0) / 100 if "error" not in cvar_result else 0.0

    return PerformanceMetrics(
        total_return=round(total_return, 4),
        annualized_return=round(annualized_return, 4),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown=round(max_dd, 4),
        max_drawdown_duration=max_dd_duration,
        volatility_annual=round(vol_annual, 4),
        downside_deviation=round(downside_dev, 4),
        win_rate=round(win_rate, 3),
        profit_factor=round(profit_factor, 3),
        beta=round(beta, 3),
        alpha=round(alpha, 4),
        information_ratio=round(info_ratio, 3),
        var_95=round(var_95, 4),
        cvar_95=round(cvar_95, 4),
    )


def render_risk_metrics_streamlit(
    returns: pd.Series,
    symbol: str = "",
    portfolio_value: float = 10000.0,
    benchmark_returns: Optional[pd.Series] = None,
) -> None:
    """Render full risk metrics dashboard in Streamlit."""
    try:
        import streamlit as st

        metrics = compute_performance_metrics(
            returns, benchmark_returns
        )
        grade = metrics.grade()
        grade_color = {
            "A": "🟢", "B": "🔵",
            "C": "🟡", "D": "🟠", "F": "🔴"
        }.get(grade, "⚪")

        st.markdown(
            f"### {grade_color} Performance Grade: {grade}"
            f"{f' — {symbol}' if symbol else ''}"
        )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Total Return",
            f"{metrics.total_return*100:.1f}%"
        )
        col2.metric(
            "Sharpe Ratio",
            f"{metrics.sharpe_ratio:.2f}",
            delta="Good" if metrics.sharpe_ratio > 1 else "Low",
        )
        col3.metric(
            "Max Drawdown",
            f"{metrics.max_drawdown*100:.1f}%",
            delta_color="inverse",
        )
        col4.metric("Win Rate", f"{metrics.win_rate*100:.1f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sortino", f"{metrics.sortino_ratio:.2f}")
        col2.metric("Calmar", f"{metrics.calmar_ratio:.2f}")
        col3.metric(
            "Annual Vol",
            f"{metrics.volatility_annual*100:.1f}%"
        )
        col4.metric(
            "Profit Factor",
            f"{metrics.profit_factor:.2f}"
        )

        # VaR/CVaR
        st.markdown("---")
        st.markdown("**Risk Metrics**")
        var_result = calculate_var(
            returns, 0.95,
            portfolio_value=portfolio_value
        )
        cvar_result = calculate_cvar(
            returns, 0.95,
            portfolio_value=portfolio_value
        )
        kelly_result = kelly_from_returns(returns)

        col1, col2, col3 = st.columns(3)
        if "error" not in var_result:
            col1.metric(
                "VaR (95%, 1-day)",
                f"${var_result['var_dollar']:,.0f}",
                f"{var_result['var_pct']:.2f}%",
                delta_color="inverse",
            )
        if "error" not in cvar_result:
            col2.metric(
                "CVaR (95%, 1-day)",
                f"${cvar_result['cvar_dollar']:,.0f}",
                f"{cvar_result['cvar_pct']:.2f}%",
                delta_color="inverse",
            )
        if "error" not in kelly_result and kelly_result.get("is_positive_edge"):
            col3.metric(
                "Kelly Position Size",
                f"{kelly_result['recommended_pct']:.1f}%",
                "of portfolio",
            )

        # Full metrics table
        with st.expander("Full Metrics"):
            metrics_dict = metrics.to_dict()
            df = pd.DataFrame(
                list(metrics_dict.items()),
                columns=["Metric", "Value"]
            )
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        try:
            import streamlit as st
            st.caption(f"Risk metrics unavailable: {e}")
        except Exception:
            pass
