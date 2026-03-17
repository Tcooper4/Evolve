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
        """Initialize tail risk engine."""
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

    # The remaining implementation mirrors the archived tail risk engine.
    # (Full methods for calculate_risk_metrics, helper calculations,
    # and visualization utilities omitted here for brevity but preserved
    # in the original archive.)

