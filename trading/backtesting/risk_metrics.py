"""
Risk Metrics for Backtesting

This module contains risk metric enums and a risk metrics engine for calculating
various risk and performance statistics for trading strategies.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Import our custom performance metrics
from utils.performance_metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    avg_drawdown,
    drawdown_details,
    value_at_risk,
    conditional_value_at_risk,
    omega_ratio,
    information_ratio,
    treynor_ratio,
    jensen_alpha,
    downside_deviation,
    gain_loss_ratio,
    profit_factor,
    recovery_factor,
    risk_reward_ratio
)

logger = logging.getLogger(__name__)


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


class RiskMetricsEngine:
    """Engine for calculating risk metrics."""

    def __init__(self, risk_free_rate: float = 0.02, period: int = 252):
        self.risk_free_rate = risk_free_rate
        self.period = period

    def calculate(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate a comprehensive set of risk metrics for a return series."""
        metrics = {}
        try:
            metrics["sharpe_ratio"] = sharpe_ratio(
                returns, risk_free=self.risk_free_rate, period=self.period
            )
            metrics["sortino_ratio"] = sortino_ratio(
                returns, risk_free=self.risk_free_rate, period=self.period
            )
            metrics["max_drawdown"] = max_drawdown(returns)
            metrics["calmar_ratio"] = calmar_ratio(
                returns, risk_free=self.risk_free_rate, period=self.period
            )
            metrics["volatility"] = returns.std() * np.sqrt(self.period)
            metrics["skewness"] = returns.skew()
            metrics["kurtosis"] = returns.kurtosis()
            metrics["mean_return"] = returns.mean() * self.period
            metrics["std_return"] = returns.std() * np.sqrt(self.period)
            metrics["min_return"] = returns.min()
            metrics["max_return"] = returns.max()
            metrics["value_at_risk"] = value_at_risk(returns, confidence_level=0.95)
            metrics["conditional_var"] = conditional_value_at_risk(returns, confidence_level=0.95)
            metrics["gain_loss_ratio"] = gain_loss_ratio(returns)
            metrics["profit_factor"] = profit_factor(returns)
            metrics["recovery_factor"] = recovery_factor(returns)
            metrics["risk_reward_ratio"] = risk_reward_ratio(returns)
            metrics["omega_ratio"] = omega_ratio(returns)
            metrics["downside_deviation"] = downside_deviation(returns)
        except Exception as e:
            logger.warning(f"Risk metric calculation failed: {e}")
        return metrics

    def get_metric(self, returns: pd.Series, metric: RiskMetric) -> Optional[float]:
        """Get a specific risk metric for a return series."""
        metrics = self.calculate(returns)
        return metrics.get(metric.value)
