"""Portfolio Risk Manager

Comprehensive risk management for portfolios:
- Risk limit enforcement (max drawdown, max exposure, volatility targeting)
- Portfolio simulation and backtesting
- Dynamic rebalancing logic
- Risk metrics calculation
- Position sizing and leverage control
- Stress testing and scenario analysis
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.common_helpers import load_config, safe_json_save

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Risk metrics for monitoring"""

    VAR = "value_at_risk"
    CVAR = "conditional_var"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    LEVERAGE = "leverage"


@dataclass
class RiskLimits:
    """Risk limits configuration"""

    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_exposure: float = 0.3  # 30% maximum single position exposure
    max_leverage: float = 2.0  # 2x maximum leverage
    target_volatility: float = 0.15  # 15% target volatility
    var_limit: float = 0.02  # 2% daily VaR limit
    max_correlation: float = 0.7  # 70% maximum correlation
    sector_limit: float = 0.4  # 40% maximum sector exposure
    liquidity_limit: float = 0.1  # 10% maximum illiquid position


@dataclass
class PortfolioState:
    """Current portfolio state"""

    timestamp: str
    positions: Dict[str, float]  # ticker -> weight
    portfolio_value: float
    cash: float
    leverage: float
    volatility: float
    drawdown: float
    var_95: float
    exposure_concentration: float
    sector_exposure: Dict[str, float]
    risk_metrics: Dict[str, float]


@dataclass
class RiskViolation:
    """Risk limit violation"""

    timestamp: str
    risk_metric: RiskMetric
    current_value: float
    limit_value: float
    severity: str  # 'warning', 'critical'
    action_required: str
    affected_positions: List[str]


@dataclass
class RebalancingAction:
    """Rebalancing action to take"""

    action_type: str  # 'buy', 'sell', 'rebalance', 'hedge'
    ticker: str
    current_weight: float
    target_weight: float
    trade_amount: float
    priority: int  # 1-5, 5 being highest
    reason: str


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.risk_config = self.config.get("risk_management", {})

        # Risk limits
        self.risk_limits = RiskLimits(
            max_drawdown=self.risk_config.get("max_drawdown", 0.15),
            max_exposure=self.risk_config.get("max_exposure", 0.3),
            max_leverage=self.risk_config.get("max_leverage", 2.0),
            target_volatility=self.risk_config.get("target_volatility", 0.15),
            var_limit=self.risk_config.get("var_limit", 0.02),
            max_correlation=self.risk_config.get("max_correlation", 0.7),
            sector_limit=self.risk_config.get("sector_limit", 0.4),
            liquidity_limit=self.risk_config.get("liquidity_limit", 0.1),
        )

        # Portfolio history
        self.portfolio_history: List[PortfolioState] = []
        self.violations_history: List[RiskViolation] = []

        # Risk calculation parameters
        self.var_confidence = self.risk_config.get("var_confidence", 0.95)
        self.lookback_period = self.risk_config.get("lookback_period", 252)  # 1 year
        self.rebalancing_frequency = self.risk_config.get(
            "rebalancing_frequency", "daily"
        )

        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}

        # Sector classifications
        self.sector_classifications = self._load_sector_classifications()

        logger.info(
            "Risk management system initialized. Portfolio risk controls "
            "are now active."
        )

    def _load_sector_classifications(self) -> Dict[str, str]:
        """Load sector classifications for assets"""
        # This would typically load from a database or file
        # For now, return a sample mapping
        return {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "TSLA": "Automotive",
            "NVDA": "Technology",
            "AMD": "Technology",
            "META": "Technology",
            "AMZN": "Consumer Discretionary",
            "JPM": "Financial",
            "JNJ": "Healthcare",
        }

    def calculate_portfolio_risk(
        self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics
        """
        if not positions:
            return {}

        # Calculate returns for each asset
        returns_data = {}
        for ticker, weight in positions.items():
            if ticker in market_data and not market_data[ticker].empty:
                returns_data[ticker] = market_data[ticker]["returns"].dropna()

        if not returns_data:
            return {}

        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        # Risk metrics
        risk_metrics = {}

        # Volatility
        risk_metrics["volatility"] = portfolio_returns.std() * np.sqrt(252)

        # Value at Risk
        risk_metrics["var_95"] = np.percentile(portfolio_returns, 5)
        risk_metrics["var_99"] = np.percentile(portfolio_returns, 1)

        # Conditional Value at Risk (Expected Shortfall)
        var_95 = risk_metrics["var_95"]
        risk_metrics["cvar_95"] = portfolio_returns[portfolio_returns <= var_95].mean()

        # Beta (if market data available)
        if "SPY" in returns_data:
            market_returns = returns_data["SPY"]
            # Align dates
            common_dates = portfolio_returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                market_aligned = market_returns.loc[common_dates]
                covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
                market_variance = np.var(market_aligned)
                risk_metrics["beta"] = (
                    covariance / market_variance if market_variance > 0 else 1.0
                )

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics["max_drawdown"] = drawdown.min()
        risk_metrics["current_drawdown"] = drawdown.iloc[-1]

        # Concentration metrics
        risk_metrics["exposure_concentration"] = max(positions.values()) if positions else 0
        risk_metrics["herfindahl_index"] = sum(w**2 for w in positions.values())

        # Sector concentration
        sector_exposure = self._calculate_sector_exposure(positions)
        risk_metrics["max_sector_exposure"] = (
            max(sector_exposure.values()) if sector_exposure else 0
        )

        # Correlation risk
        if len(returns_data) > 1:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            # Average correlation
            n_assets = len(returns_data)
            total_correlation = 0
            count = 0
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    total_correlation += abs(correlation_matrix.iloc[i, j])
                    count += 1
            risk_metrics["avg_correlation"] = total_correlation / count if count > 0 else 0

        return risk_metrics

    def _calculate_portfolio_returns(
        self, positions: Dict[str, float], returns_data: Dict[str, pd.Series]
    ) -> pd.Series:
        """Calculate portfolio returns from asset returns"""
        # Align all return series
        all_returns = []
        for ticker, returns in returns_data.items():
            if ticker in positions:
                all_returns.append(returns)

        if not all_returns:
            return pd.Series(dtype=float)

        returns_df = pd.concat(all_returns, axis=1).fillna(0)
        weights = np.array([positions.get(ticker, 0) for ticker in returns_data.keys()])

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()

        portfolio_returns = returns_df.dot(weights)
        return portfolio_returns

    def _calculate_sector_exposure(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposure for the portfolio"""
        sector_exposure: Dict[str, float] = {}
        for ticker, weight in positions.items():
            sector = self.sector_classifications.get(ticker, "Other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        return sector_exposure

