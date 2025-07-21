"""
Market Analyzer for financial data analysis.

This module provides comprehensive market analysis capabilities including
technical indicators, regime detection, and pattern recognition.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from trading.logs.logger import log_metrics  # Using our centralized logging system


class MarketAnalysisError(Exception):
    """Custom exception for market analysis errors."""


class MarketAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if "trend_threshold" in self.config and not isinstance(
            self.config["trend_threshold"], (int, float)
        ):
            raise ValueError("trend_threshold must be a number")
        if "volatility_window" in self.config and (
            not isinstance(self.config["volatility_window"], int)
            or self.config["volatility_window"] <= 0
        ):
            raise ValueError("volatility_window must be a positive integer")
        if "correlation_threshold" in self.config and (
            not isinstance(self.config["correlation_threshold"], (int, float))
            or not -1 <= self.config["correlation_threshold"] <= 1
        ):
            raise ValueError("correlation_threshold must be between -1 and 1")

    def detect_market_regime(
        self,
        data: pd.DataFrame,
        regime_type: str = "trend",
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Detect market regime with parameterization for different types.

        Args:
            data: Input DataFrame with price data
            regime_type: Type of regime to detect ('trend', 'volatility', 'correlation')
            market_data: Optional market data for correlation analysis

        Returns:
            Dictionary with regime analysis results
        """
        try:
            if "Close" not in data.columns:
                raise KeyError("Missing 'Close' column in input data.")

            if regime_type == "trend":
                return self._detect_trend_regime(data)
            elif regime_type == "volatility":
                return self._detect_volatility_regime(data)
            elif regime_type == "correlation":
                if market_data is None:
                    raise ValueError(
                        "Market data required for correlation regime detection"
                    )
                return self._detect_correlation_regime(data, market_data)
            else:
                raise ValueError(f"Unknown regime type: {regime_type}")

        except Exception as e:
            self.logger.error(f"Error detecting {regime_type} regime: {e}")
            raise

    def _detect_trend_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend-based market regime."""
        ma_short = data["Close"].rolling(window=20).mean()
        ma_long = data["Close"].rolling(window=50).mean()
        trend_strength = (ma_short - ma_long) / ma_long

        if len(trend_strength.dropna()) == 0:
            raise ValueError("Insufficient data for trend analysis")

        current_trend = (
            "up"
            if trend_strength.iloc[-1] > self.config.get("trend_threshold", 0)
            else "down"
        )
        trend_changes = np.diff(np.signbit(trend_strength.dropna()))

        trend_duration = len(data)
        if len(np.where(trend_changes)[0]) > 0:
            trend_duration = len(data) - np.where(trend_changes)[0][-1]

        result = {
            "regime_type": "trend",
            "regime": current_trend,
            "strength": float(trend_strength.iloc[-1]),
            "duration": int(trend_duration),
            "ma_short": float(ma_short.iloc[-1]),
            "ma_long": float(ma_long.iloc[-1]),
        }

        log_metrics("trend_regime", result)
        return result

    def _detect_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect volatility-based market regime."""
        returns = data["Close"].pct_change()
        current_volatility = returns.std() * np.sqrt(252)
        historical_volatility = returns.rolling(
            window=self.config.get("volatility_window", 252)
        ).std() * np.sqrt(252)

        hv_non_null = historical_volatility.dropna()
        if len(hv_non_null) < 2:
            raise ValueError("Insufficient data for volatility analysis")

        prev_volatility = hv_non_null.iloc[-2]
        volatility_trend = (
            "increasing" if current_volatility > prev_volatility else "decreasing"
        )

        vol_min = historical_volatility.min()
        vol_max = historical_volatility.max()
        volatility_rank = (
            (current_volatility - vol_min) / (vol_max - vol_min)
            if vol_max > vol_min
            else 0.5
        )

        # Determine volatility regime
        if volatility_rank > 0.7:
            regime = "high_volatility"
        elif volatility_rank < 0.3:
            regime = "low_volatility"
        else:
            regime = "normal_volatility"

        result = {
            "regime_type": "volatility",
            "regime": regime,
            "current_volatility": float(current_volatility),
            "volatility_rank": float(volatility_rank),
            "volatility_trend": volatility_trend,
            "historical_volatility": float(historical_volatility.iloc[-1]),
        }

        log_metrics("volatility_regime", result)
        return result

    def _detect_correlation_regime(
        self, data: pd.DataFrame, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect correlation-based market regime."""
        if "Close" not in market_data.columns:
            raise KeyError("Missing 'Close' column in market data.")

        returns = data["Close"].pct_change()
        market_returns = market_data["Close"].pct_change()
        correlation = returns.corr(market_returns)
        rolling_correlation = returns.rolling(window=252).corr(market_returns)

        rc_non_null = rolling_correlation.dropna()
        if len(rc_non_null) < 2:
            correlation_trend = "unknown"
        else:
            correlation_trend = (
                "increasing" if correlation > rc_non_null.iloc[-2] else "decreasing"
            )

        # Determine correlation regime
        if correlation > 0.7:
            regime = "high_correlation"
        elif correlation < 0.3:
            regime = "low_correlation"
        else:
            regime = "moderate_correlation"

        result = {
            "regime_type": "correlation",
            "regime": regime,
            "correlation": float(correlation),
            "correlation_trend": correlation_trend,
            "rolling_correlation": (
                float(rolling_correlation.dropna().iloc[-1])
                if len(rolling_correlation.dropna()) > 0
                else None
            ),
        }

        log_metrics("correlation_regime", result)
        return result

    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend using moving averages (deprecated, use detect_market_regime)."""
        return self.detect_market_regime(data, "trend")

    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility (deprecated, use detect_market_regime)."""
        return self.detect_market_regime(data, "volatility")

    def analyze_correlation(
        self, data: pd.DataFrame, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze correlation with market data (deprecated, use detect_market_regime)."""
        return self.detect_market_regime(data, "correlation", market_data)

    def analyze_market_conditions(
        self, data: pd.DataFrame, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze overall market conditions using unified regime detection."""
        try:
            trend = self.detect_market_regime(data, "trend")
            volatility = self.detect_market_regime(data, "volatility")
            correlation = self.detect_market_regime(data, "correlation", market_data)

            combined = {
                "trend": trend,
                "volatility": volatility,
                "correlation": correlation,
                "timestamp": datetime.utcnow().isoformat(),
            }

            log_metrics("market_conditions", combined)
            return combined

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            raise
