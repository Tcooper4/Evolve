"""
Risk Metrics for Backtesting

This module contains risk metric enums and a risk metrics engine for calculating
various risk and performance statistics for trading strategies.
"""

from enum import Enum
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import empyrical, with fallback
try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    class EmpyricalFallback:
        @staticmethod
        def sharpe_ratio(returns, risk_free=0.0, period=252):
            excess_returns = returns - risk_free
            if len(excess_returns) == 0 or excess_returns.std() == 0:
                return 0.0
            return (excess_returns.mean() * period) / (excess_returns.std() * np.sqrt(period))
        @staticmethod
        def sortino_ratio(returns, risk_free=0.0, period=252):
            excess_returns = returns - risk_free
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return (excess_returns.mean() * period) / (downside_returns.std() * np.sqrt(period))
        @staticmethod
        def max_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        @staticmethod
        def calmar_ratio(returns, risk_free=0.0, period=252):
            max_dd = EmpyricalFallback.max_drawdown(returns)
            if max_dd == 0:
                return 0.0
            annual_return = (1 + returns.mean()) ** period - 1
            return annual_return / abs(max_dd)
    ep = EmpyricalFallback()

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
            metrics['sharpe_ratio'] = ep.sharpe_ratio(returns, risk_free=self.risk_free_rate, period=self.period)
            metrics['sortino_ratio'] = ep.sortino_ratio(returns, risk_free=self.risk_free_rate, period=self.period)
            metrics['max_drawdown'] = ep.max_drawdown(returns)
            metrics['calmar_ratio'] = ep.calmar_ratio(returns, risk_free=self.risk_free_rate, period=self.period)
            metrics['volatility'] = returns.std() * np.sqrt(self.period)
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            metrics['mean_return'] = returns.mean() * self.period
            metrics['std_return'] = returns.std() * np.sqrt(self.period)
            metrics['min_return'] = returns.min()
            metrics['max_return'] = returns.max()
            metrics['value_at_risk'] = np.percentile(returns, 5)
            metrics['conditional_var'] = returns[returns <= metrics['value_at_risk']].mean() if len(returns[returns <= metrics['value_at_risk']]) > 0 else np.nan
        except Exception as e:
            logger.warning(f"Risk metric calculation failed: {e}")
        return metrics

    def get_metric(self, returns: pd.Series, metric: RiskMetric) -> Optional[float]:
        """Get a specific risk metric for a return series."""
        metrics = self.calculate(returns)
        return metrics.get(metric.value) 