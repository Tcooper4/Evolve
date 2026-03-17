"""Advanced Risk Analytics for Evolve Trading Platform.

This module provides comprehensive risk metrics including VaR, CVaR,
max drawdown, expected shortfall, and other advanced risk measures.
"""

import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics structure."""

    # Basic risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int

    # Expected shortfall
    expected_shortfall_95: float
    expected_shortfall_99: float

    # Additional risk metrics
    downside_deviation: float
    upside_potential: float
    gain_loss_ratio: float
    win_rate: float

    # Tail risk metrics
    tail_risk_95: float
    tail_risk_99: float
    kurtosis: float
    skewness: float

    # Stress test metrics
    stress_test_1sd: float
    stress_test_2sd: float
    stress_test_3sd: float

    # Calculation metadata
    calculation_date: str
    lookback_period: int
    confidence_levels: List[float]


class AdvancedRiskAnalyzer:
    """Advanced risk analysis engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk analyzer."""
        self.config = config or {}
        self.risk_history = []

    def calculate_comprehensive_risk(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        confidence_levels: List[float] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""

        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        # Basic statistics
        mean_return = returns.mean()
        volatility = returns.std()

        # Basic risk ratios
        sharpe_ratio = mean_return / volatility * np.sqrt(252) if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0
            else 0.001
        )
        sortino_ratio = (
            mean_return / downside_deviation * np.sqrt(252)
            if downside_deviation > 0
            else 0
        )

        # Calmar ratio
        max_dd = self.calculate_max_drawdown(returns)
        calmar_ratio = mean_return * 252 / abs(max_dd) if max_dd != 0 else 0

        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Drawdown analysis
        drawdown_metrics = self.calculate_drawdown_metrics(returns)

        # Expected shortfall
        expected_shortfall_95 = self.calculate_expected_shortfall(returns, 0.95)
        expected_shortfall_99 = self.calculate_expected_shortfall(returns, 0.99)

        # Additional metrics
        upside_returns = returns[returns > 0]
        upside_potential = upside_returns.mean() if len(upside_returns) > 0 else 0

        gain_loss_ratio = abs(upside_potential / cvar_95) if cvar_95 != 0 else 0
        win_rate = len(upside_returns) / len(returns) if len(returns) > 0 else 0

        # Tail risk metrics
        tail_risk_95 = self.calculate_tail_risk(returns, 0.95)
        tail_risk_99 = self.calculate_tail_risk(returns, 0.99)

        # Distribution metrics
        kurtosis = returns.kurtosis()
        skewness = returns.skew()

        # Stress test metrics
        stress_metrics = self.calculate_stress_test_metrics(returns)

        risk_metrics = RiskMetrics(
            volatility=volatility * np.sqrt(252),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            avg_drawdown=drawdown_metrics["avg_drawdown"],
            drawdown_duration=drawdown_metrics["max_duration"],
            expected_shortfall_95=expected_shortfall_95,
            expected_shortfall_99=expected_shortfall_99,
            downside_deviation=downside_deviation,
            upside_potential=upside_potential,
            gain_loss_ratio=gain_loss_ratio,
            win_rate=win_rate,
            tail_risk_95=tail_risk_95,
            tail_risk_99=tail_risk_99,
            kurtosis=kurtosis,
            skewness=skewness,
            stress_test_1sd=stress_metrics["stress_1sd"],
            stress_test_2sd=stress_metrics["stress_2sd"],
            stress_test_3sd=stress_metrics["stress_3sd"],
            calculation_date=datetime.now().isoformat(),
            lookback_period=len(returns),
            confidence_levels=confidence_levels,
        )

        self.risk_history.append(risk_metrics)
        return risk_metrics

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(
                    {
                        "start": start_idx,
                        "end": i,
                        "duration": i - start_idx,
                        "depth": drawdown[start_idx:i].min(),
                    }
                )

        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_periods.append(
                {
                    "start": start_idx,
                    "end": len(drawdown) - 1,
                    "duration": len(drawdown) - 1 - start_idx,
                    "depth": drawdown[start_idx:].min(),
                }
            )

        if drawdown_periods:
            avg_drawdown = np.mean([dd["depth"] for dd in drawdown_periods])
            max_duration = max([dd["duration"] for dd in drawdown_periods])
        else:
            avg_drawdown = 0
            max_duration = 0

        return {
            "max_drawdown": drawdown.min(),
            "avg_drawdown": avg_drawdown,
            "max_duration": max_duration,
            "drawdown_periods": drawdown_periods,
        }

    def calculate_expected_shortfall(
        self, returns: pd.Series, confidence_level: float
    ) -> float:
        """Calculate expected shortfall (conditional VaR)."""
        var_level = (1 - confidence_level) * 100
        var = np.percentile(returns, var_level)
        return returns[returns <= var].mean()

    def calculate_tail_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate tail risk using simple percentile-based approach."""
        var_level = (1 - confidence_level) * 100
        tail_returns = returns[returns <= np.percentile(returns, var_level)]

        if len(tail_returns) > 0:
            return tail_returns.std() * np.sqrt(252)
        else:
            return 0

    def calculate_stress_test_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic stress test metrics."""
        mean_return = returns.mean()
        volatility = returns.std()

        return {
            "stress_1sd": mean_return - volatility,
            "stress_2sd": mean_return - 2 * volatility,
            "stress_3sd": mean_return - 3 * volatility,
        }

