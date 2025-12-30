"""Risk & Tail Exposure Engine for Evolve Trading Platform.

This module provides comprehensive risk analysis including VaR, CVaR,
drawdown analysis, and regime-based risk metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import our custom performance metrics
from utils.performance_metrics import (
    avg_drawdown,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_details,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""

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

    # Tail risk metrics
    tail_dependence: float
    expected_shortfall: float
    tail_risk_ratio: float

    # Regime-specific metrics
    bull_market_risk: float
    bear_market_risk: float
    neutral_market_risk: float

    # Stress test results
    stress_test_loss: float
    scenario_analysis: Dict[str, float]


@dataclass
class RegimeAnalysis:
    """Market regime analysis results."""

    regime: str  # "bull", "bear", "neutral", "crisis"
    start_date: datetime
    end_date: datetime
    duration: int
    return_mean: float
    return_vol: float
    var_95: float
    max_drawdown: float
    risk_score: float


@dataclass
class TailRiskReport:
    """Comprehensive tail risk report."""

    portfolio_metrics: RiskMetrics
    regime_analysis: List[RegimeAnalysis]
    stress_scenarios: Dict[str, float]
    risk_decomposition: Dict[str, float]
    recommendations: List[str]
    report_date: datetime


class TailRiskEngine:
    """Comprehensive tail risk analysis engine."""

    def __init__(
        self,
        confidence_levels: List[float] = None,
        lookback_period: int = 252,
        regime_threshold: float = 0.1,
    ):
        """Initialize tail risk engine.

        Args:
            confidence_levels: VaR confidence levels
            lookback_period: Lookback period for calculations
            regime_threshold: Threshold for regime classification
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.lookback_period = lookback_period
        self.regime_threshold = regime_threshold

        # Risk parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.annualization_factor = 252  # Trading days per year

        # Regime classification parameters
        self.bull_threshold = 0.05  # 5% positive return threshold
        self.bear_threshold = -0.05  # -5% negative return threshold
        self.crisis_threshold = -0.10  # -10% crisis threshold

        logger.info(
            f"Initialized Tail Risk Engine with {len(self.confidence_levels)} confidence levels"
        )

    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics.

        Args:
            returns: Portfolio returns series

        Returns:
            Risk metrics
        """
        try:
            # Clean returns
            returns_clean = returns.dropna()

            if len(returns_clean) < 30:
                raise ValueError("Insufficient data for risk calculation")

            # Basic risk metrics
            volatility = returns_clean.std() * np.sqrt(self.annualization_factor)
            sharpe_ratio = self._calculate_sharpe_ratio(returns_clean)
            sortino_ratio = self._calculate_sortino_ratio(returns_clean)
            calmar_ratio = self._calculate_calmar_ratio(returns_clean)

            # VaR and CVaR
            var_95, var_99 = self._calculate_var(returns_clean)
            cvar_95, cvar_99 = self._calculate_cvar(returns_clean)

            # Drawdown metrics
            (
                max_drawdown,
                avg_drawdown,
                drawdown_duration,
            ) = self._calculate_drawdown_metrics(returns_clean)

            # Tail risk metrics
            tail_dependence = self._calculate_tail_dependence(returns_clean)
            expected_shortfall = self._calculate_expected_shortfall(returns_clean)
            tail_risk_ratio = self._calculate_tail_risk_ratio(returns_clean)

            # Regime-specific metrics
            regime_metrics = self._calculate_regime_risk(returns_clean)

            # Stress test
            stress_test_loss = self._run_stress_test(returns_clean)
            scenario_analysis = self._run_scenario_analysis(returns_clean)

            return RiskMetrics(
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                avg_drawdown=avg_drawdown,
                drawdown_duration=drawdown_duration,
                tail_dependence=tail_dependence,
                expected_shortfall=expected_shortfall,
                tail_risk_ratio=tail_risk_ratio,
                bull_market_risk=regime_metrics.get("bull", 0.0),
                bear_market_risk=regime_metrics.get("bear", 0.0),
                neutral_market_risk=regime_metrics.get("neutral", 0.0),
                stress_test_loss=stress_test_loss,
                scenario_analysis=scenario_analysis,
            )

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        return sharpe_ratio(
            returns, risk_free=self.risk_free_rate, period=self.annualization_factor
        )

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        return sortino_ratio(
            returns, risk_free=self.risk_free_rate, period=self.annualization_factor
        )

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        return calmar_ratio(
            returns, risk_free=self.risk_free_rate, period=self.annualization_factor
        )

    def _calculate_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk."""
        var_95 = value_at_risk(returns, confidence_level=0.95)
        var_99 = value_at_risk(returns, confidence_level=0.99)
        return var_95, var_99

    def _calculate_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        cvar_95 = conditional_value_at_risk(returns, confidence_level=0.95)
        cvar_99 = conditional_value_at_risk(returns, confidence_level=0.99)
        return cvar_95, cvar_99

    def _calculate_drawdown_metrics(
        self, returns: pd.Series
    ) -> Tuple[float, float, int]:
        """Calculate drawdown metrics."""
        max_dd = abs(max_drawdown(returns))
        avg_dd = abs(avg_drawdown(returns))

        # Get drawdown details for duration
        dd_details = drawdown_details(returns)
        dd_duration = dd_details["days"].max() if not dd_details.empty else 0

        return max_dd, avg_dd, dd_duration

    def _calculate_tail_dependence(self, returns: pd.Series) -> float:
        """Calculate tail dependence coefficient."""
        # Simplified tail dependence calculation
        # In practice, you'd use copula-based methods

        # Calculate extreme events (bottom 5%)
        extreme_threshold = np.percentile(returns, 5)
        extreme_events = returns <= extreme_threshold

        # Calculate tail dependence as correlation of extreme events
        if extreme_events.sum() > 1:
            # Use rolling correlation of extreme events
            rolling_extreme = extreme_events.rolling(window=20).mean()
            tail_dependence = rolling_extreme.corr(returns)
            return abs(tail_dependence) if not np.isnan(tail_dependence) else 0.0
        else:
            return 0.0

    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate expected shortfall beyond VaR."""
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95]

        if len(tail_returns) > 0:
            return tail_returns.mean()
        else:
            return var_95

    def _calculate_tail_risk_ratio(self, returns: pd.Series) -> float:
        """Calculate tail risk ratio."""
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95]

        if len(tail_returns) > 0 and returns.std() > 0:
            return abs(tail_returns.mean() / returns.std())
        else:
            return 0.0

    def _calculate_regime_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate regime-specific risk metrics."""
        # Classify market regimes
        regimes = self._classify_market_regimes(returns)

        regime_metrics = {}

        for regime in ["bull", "bear", "neutral"]:
            regime_returns = returns[regimes == regime]

            if len(regime_returns) > 10:
                # Calculate regime-specific VaR
                regime_var = np.percentile(regime_returns, 5)
                regime_metrics[regime] = abs(regime_var)
            else:
                regime_metrics[regime] = 0.0

        return regime_metrics

    def _classify_market_regimes(self, returns: pd.Series) -> pd.Series:
        """Classify market regimes based on returns."""
        # Use rolling window to classify regimes
        rolling_mean = returns.rolling(window=20).mean()
        rolling_vol = returns.rolling(window=20).std()

        regimes = pd.Series("neutral", index=returns.index)

        # Bull market: high positive returns, low volatility
        bull_condition = (rolling_mean > self.bull_threshold) & (
            rolling_vol < rolling_vol.quantile(0.7)
        )
        regimes[bull_condition] = "bull"

        # Bear market: negative returns, high volatility
        bear_condition = (rolling_mean < self.bear_threshold) & (
            rolling_vol > rolling_vol.quantile(0.7)
        )
        regimes[bear_condition] = "bear"

        # Crisis: extreme negative returns
        crisis_condition = rolling_mean < self.crisis_threshold
        regimes[crisis_condition] = "crisis"

        return regimes

    def _run_stress_test(self, returns: pd.Series) -> float:
        """Run stress test scenarios."""
        # Historical stress test using worst periods
        worst_periods = [
            returns.rolling(window=5).sum().min(),  # Worst 5-day period
            returns.rolling(window=10).sum().min(),  # Worst 10-day period
            returns.rolling(window=20).sum().min(),  # Worst 20-day period
        ]

        return min(worst_periods)

    def _run_scenario_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """Run scenario analysis."""
        scenarios = {
            "market_crash": -0.20,  # 20% market crash
            "high_volatility": returns.std() * 2,  # 2x volatility
            "interest_rate_shock": -0.05,  # 5% interest rate shock
            "liquidity_crisis": -0.15,  # 15% liquidity crisis
            "black_swan": -0.30,  # 30% black swan event
        }

        return scenarios

    def analyze_regimes(self, returns: pd.Series) -> List[RegimeAnalysis]:
        """Analyze market regimes over time."""
        regimes = self._classify_market_regimes(returns)

        regime_analysis = []
        current_regime = None
        regime_start = None

        for date, regime in regimes.items():
            if regime != current_regime:
                # End previous regime
                if current_regime and regime_start:
                    regime_end = date - pd.Timedelta(days=1)
                    regime_returns = returns[regime_start:regime_end]

                    if len(regime_returns) > 5:  # Minimum regime duration
                        analysis = RegimeAnalysis(
                            regime=current_regime,
                            start_date=regime_start,
                            end_date=regime_end,
                            duration=len(regime_returns),
                            return_mean=regime_returns.mean(),
                            return_vol=regime_returns.std(),
                            var_95=np.percentile(regime_returns, 5),
                            max_drawdown=self._calculate_drawdown_metrics(
                                regime_returns
                            )[0],
                            risk_score=self._calculate_regime_risk_score(
                                regime_returns
                            ),
                        )
                        regime_analysis.append(analysis)

                # Start new regime
                current_regime = regime
                regime_start = date

        return regime_analysis

    def _calculate_regime_risk_score(self, returns: pd.Series) -> float:
        """Calculate risk score for a regime."""
        if len(returns) < 5:
            return 0.0

        # Combine multiple risk factors
        volatility = returns.std()
        var_95 = np.percentile(returns, 5)
        max_dd = self._calculate_drawdown_metrics(returns)[0]

        # Normalize and combine
        risk_score = (abs(var_95) + max_dd + volatility) / 3
        return risk_score

    def create_drawdown_heatmap(
        self, returns: pd.DataFrame, symbols: List[str]
    ) -> plt.Figure:
        """Create drawdown heatmap by market regime."""
        try:
            # Calculate drawdowns for each symbol
            drawdowns = {}
            regimes = {}

            for symbol in symbols:
                if symbol in returns.columns:
                    symbol_returns = returns[symbol].dropna()
                    if len(symbol_returns) > 30:
                        # Calculate drawdown
                        cumulative = (1 + symbol_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max

                        # Classify regimes
                        symbol_regimes = self._classify_market_regimes(symbol_returns)

                        drawdowns[symbol] = drawdown
                        regimes[symbol] = symbol_regimes

            # Create heatmap data
            regime_drawdowns = {}
            for regime in ["bull", "bear", "neutral", "crisis"]:
                regime_drawdowns[regime] = {}

                for symbol in symbols:
                    if symbol in drawdowns and symbol in regimes:
                        regime_mask = regimes[symbol] == regime
                        if regime_mask.sum() > 0:
                            regime_dd = drawdowns[symbol][regime_mask]
                            regime_drawdowns[regime][symbol] = (
                                abs(regime_dd.min()) if len(regime_dd) > 0 else 0.0
                            )
                        else:
                            regime_drawdowns[regime][symbol] = 0.0

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data for heatmap
            heatmap_data = []
            symbols_list = []

            for regime in ["bull", "bear", "neutral", "crisis"]:
                for symbol in symbols:
                    if symbol in regime_drawdowns[regime]:
                        heatmap_data.append(regime_drawdowns[regime][symbol])
                        symbols_list.append(f"{symbol}_{regime}")

            # Reshape data for heatmap
            n_symbols = len(symbols)
            n_regimes = 4
            heatmap_matrix = np.array(heatmap_data).reshape(n_regimes, n_symbols)

            # Create heatmap
            sns.heatmap(
                heatmap_matrix,
                xticklabels=symbols,
                yticklabels=["Bull", "Bear", "Neutral", "Crisis"],
                annot=True,
                fmt=".3f",
                cmap="RdYlGn_r",
                ax=ax,
            )

            ax.set_title("Drawdown Heatmap by Market Regime")
            ax.set_xlabel("Symbols")
            ax.set_ylabel("Market Regimes")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating drawdown heatmap: {e}")
            return fig

    def generate_risk_report(
        self, returns: pd.Series, portfolio_name: str = "Portfolio"
    ) -> TailRiskReport:
        """Generate comprehensive risk report.

        Args:
            returns: Portfolio returns
            portfolio_name: Name of the portfolio

        Returns:
            Comprehensive risk report
        """
        try:
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(returns)

            # Analyze regimes
            regime_analysis = self.analyze_regimes(returns)

            # Run stress scenarios
            stress_scenarios = self._run_scenario_analysis(returns)

            # Risk decomposition
            risk_decomposition = self._decompose_risk(returns)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_metrics, regime_analysis
            )

            return TailRiskReport(
                portfolio_metrics=risk_metrics,
                regime_analysis=regime_analysis,
                stress_scenarios=stress_scenarios,
                risk_decomposition=risk_decomposition,
                recommendations=recommendations,
                report_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise

    def _decompose_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Decompose risk into components."""
        # Simplified risk decomposition
        volatility = returns.std()

        # Decompose into systematic and idiosyncratic risk
        # In practice, you'd use factor models

        decomposition = {
            "total_risk": volatility,
            "systematic_risk": volatility * 0.7,  # Assume 70% systematic
            "idiosyncratic_risk": volatility * 0.3,  # Assume 30% idiosyncratic
            "tail_risk": abs(np.percentile(returns, 5)),
            "liquidity_risk": volatility * 0.1,  # Assume 10% liquidity risk
            "concentration_risk": volatility * 0.05,  # Assume 5% concentration risk
        }

        return decomposition

    def _generate_recommendations(
        self, risk_metrics: RiskMetrics, regime_analysis: List[RegimeAnalysis]
    ) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        # VaR-based recommendations
        if risk_metrics.var_95 < -0.05:  # VaR > 5%
            recommendations.append(
                "Consider reducing position sizes to lower VaR exposure"
            )

        if risk_metrics.max_drawdown > 0.20:  # Max drawdown > 20%
            recommendations.append("Implement stricter stop-loss mechanisms")

        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Review strategy risk-return profile")

        # Regime-based recommendations
        bear_regimes = [r for r in regime_analysis if r.regime == "bear"]
        if bear_regimes:
            avg_bear_risk = np.mean([r.risk_score for r in bear_regimes])
            if avg_bear_risk > 0.1:
                recommendations.append(
                    "Consider hedging strategies for bear market protection"
                )

        # Tail risk recommendations
        if risk_metrics.tail_risk_ratio > 2.0:
            recommendations.append("Implement tail risk hedging strategies")

        if risk_metrics.stress_test_loss < -0.15:
            recommendations.append(
                "Review stress test scenarios and adjust risk limits"
            )

        # General recommendations
        recommendations.extend(
            [
                "Monitor regime changes and adjust strategy accordingly",
                "Regularly review and update risk limits",
                "Consider diversification to reduce concentration risk",
            ]
        )

        return recommendations[:5]  # Return top 5 recommendations


def calculate_portfolio_risk(
    returns: pd.Series, confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate portfolio risk metrics.

    Args:
        returns: Portfolio returns
        confidence_level: VaR confidence level

    Returns:
        Risk metrics dictionary
    """
    try:
        engine = TailRiskEngine()
        risk_metrics = engine.calculate_risk_metrics(returns)

        return {
            "volatility": risk_metrics.volatility,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "var": getattr(
                risk_metrics, f"var_{int(confidence_level * 100)}", risk_metrics.var_95
            ),
            "cvar": getattr(
                risk_metrics,
                f"cvar_{int(confidence_level * 100)}",
                risk_metrics.cvar_95,
            ),
            "max_drawdown": risk_metrics.max_drawdown,
            "tail_risk_ratio": risk_metrics.tail_risk_ratio,
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {e}")
        return {}


def analyze_tail_risk(returns: pd.Series) -> TailRiskReport:
    """Analyze tail risk for a portfolio.

    Args:
        returns: Portfolio returns

    Returns:
        Tail risk report
    """
    try:
        engine = TailRiskEngine()
        return engine.generate_risk_report(returns)

    except Exception as e:
        logger.error(f"Error analyzing tail risk: {e}")
        raise
