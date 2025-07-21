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
        """Calculate tail risk using extreme value theory."""
        # Simple tail risk calculation
        var_level = (1 - confidence_level) * 100
        tail_returns = returns[returns <= np.percentile(returns, var_level)]

        if len(tail_returns) > 0:
            # Use generalized Pareto distribution approximation
            return tail_returns.std() * np.sqrt(252)
        else:
            return 0

    def calculate_stress_test_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate stress test metrics."""
        mean_return = returns.mean()
        volatility = returns.std()

        # Stress scenarios (1, 2, 3 standard deviations)
        stress_1sd = mean_return - volatility
        stress_2sd = mean_return - 2 * volatility
        stress_3sd = mean_return - 3 * volatility

        return {
            "stress_1sd": stress_1sd,
            "stress_2sd": stress_2sd,
            "stress_3sd": stress_3sd,
        }

    def calculate_portfolio_risk(
        self, portfolio_returns: pd.Series, weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""

        # Portfolio risk metrics
        portfolio_risk = self.calculate_comprehensive_risk(portfolio_returns)

        # Component VaR (if weights provided)
        component_var = {}
        if weights:
            for asset, weight in weights.items():
                # Simplified component VaR calculation
                component_var[asset] = weight * portfolio_risk.var_95

        return {
            "portfolio_risk": portfolio_risk,
            "component_var": component_var,
            "diversification_ratio": self.calculate_diversification_ratio(
                portfolio_returns, weights
            ),
        }

    def calculate_diversification_ratio(
        self, portfolio_returns: pd.Series, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate diversification ratio."""
        if weights is None:
            return 1.0

        # Simplified diversification ratio
        weighted_vol = sum(abs(weight) for weight in weights.values())
        portfolio_vol = portfolio_returns.std()

        return portfolio_vol / weighted_vol if weighted_vol > 0 else 1.0

    def calculate_regime_risk(
        self, returns: pd.Series, regime_indicator: pd.Series
    ) -> Dict[str, RiskMetrics]:
        """Calculate risk metrics for different market regimes."""
        regime_risks = {}

        unique_regimes = regime_indicator.unique()

        for regime in unique_regimes:
            regime_returns = returns[regime_indicator == regime]
            if len(regime_returns) > 10:  # Minimum observations
                regime_risks[regime] = self.calculate_comprehensive_risk(regime_returns)

        return regime_risks

    def calculate_rolling_risk(
        self, returns: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling risk metrics."""
        rolling_metrics = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window: i]
            metrics = self.calculate_comprehensive_risk(window_returns)

            rolling_metrics.append(
                {
                    "date": returns.index[i - 1],
                    "volatility": metrics.volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "var_95": metrics.var_95,
                    "max_drawdown": metrics.max_drawdown,
                }
            )

        return pd.DataFrame(rolling_metrics)

    def generate_risk_report(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Generate comprehensive risk report."""

        # Risk classification
        risk_level = self._classify_risk_level(risk_metrics)

        # Risk alerts
        alerts = self._generate_risk_alerts(risk_metrics)

        report = {
            "summary": {
                "risk_level": risk_level,
                "overall_risk_score": self._calculate_risk_score(risk_metrics),
                "key_metrics": {
                    "volatility": f"{risk_metrics.volatility:.2%}",
                    "sharpe_ratio": f"{risk_metrics.sharpe_ratio:.2f}",
                    "max_drawdown": f"{risk_metrics.max_drawdown:.2%}",
                    "var_95": f"{risk_metrics.var_95:.2%}",
                },
            },
            "detailed_metrics": asdict(risk_metrics),
            "alerts": alerts,
            "recommendations": self._generate_risk_recommendations(risk_metrics),
            "report_date": datetime.now().isoformat(),
        }

        return report

    def _classify_risk_level(self, metrics: RiskMetrics) -> str:
        """Classify overall risk level."""
        risk_score = self._calculate_risk_score(metrics)

        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Very High"

    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-1)."""
        # Normalize and weight different risk factors
        volatility_score = min(1.0, metrics.volatility / 0.5)  # 50% annual vol = max
        drawdown_score = min(1.0, abs(metrics.max_drawdown) / 0.3)  # 30% drawdown = max
        var_score = min(1.0, abs(metrics.var_95) / 0.1)  # 10% daily VaR = max

        # Weighted average
        risk_score = volatility_score * 0.3 + drawdown_score * 0.4 + var_score * 0.3

        return min(1.0, risk_score)

    def _generate_risk_alerts(self, metrics: RiskMetrics) -> List[Dict]:
        """Generate risk alerts based on metrics."""
        alerts = []

        # Volatility alert
        if metrics.volatility > 0.3:
            alerts.append(
                {
                    "type": "high_volatility",
                    "severity": "warning",
                    "message": f"High volatility detected: {metrics.volatility:.2%}",
                    "threshold": "30%",
                }
            )

        # Drawdown alert
        if abs(metrics.max_drawdown) > 0.2:
            alerts.append(
                {
                    "type": "large_drawdown",
                    "severity": "critical",
                    "message": f"Large drawdown detected: {metrics.max_drawdown:.2%}",
                    "threshold": "20%",
                }
            )

        # VaR alert
        if abs(metrics.var_95) > 0.05:
            alerts.append(
                {
                    "type": "high_var",
                    "severity": "warning",
                    "message": f"High VaR detected: {metrics.var_95:.2%}",
                    "threshold": "5%",
                }
            )

        # Sharpe ratio alert
        if metrics.sharpe_ratio < 0.5:
            alerts.append(
                {
                    "type": "low_sharpe",
                    "severity": "info",
                    "message": f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}",
                    "threshold": "0.5",
                }
            )

        return alerts

    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        if metrics.volatility > 0.3:
            recommendations.append(
                "Consider reducing position sizes to manage high volatility"
            )

        if abs(metrics.max_drawdown) > 0.2:
            recommendations.append("Implement stop-loss mechanisms to limit drawdowns")

        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Review strategy to improve risk-adjusted returns")

        if metrics.kurtosis > 5:
            recommendations.append("Consider tail risk hedging due to fat tails")

        if metrics.skewness < -1:
            recommendations.append(
                "Strategy shows negative skewness - consider asymmetric risk management"
            )

        return recommendations

    def save_risk_report(
        self, report: Dict[str, Any], filepath: str = "reports/risk_analysis.json"
    ) -> bool:
        """Save risk report to file."""
        try:
            import json

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Saved risk report to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving risk report: {e}")
            return False


# Global risk analyzer instance
risk_analyzer = AdvancedRiskAnalyzer()


def get_risk_analyzer() -> AdvancedRiskAnalyzer:
    """Get the global risk analyzer instance."""
    return risk_analyzer
