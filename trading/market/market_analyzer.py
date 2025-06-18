"""
Market Analyzer for financial data analysis.

This module provides comprehensive market analysis capabilities including
technical indicators, regime detection, and pattern recognition.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from trading.logs.logger import log_metrics  # Using our centralized logging system

class MarketAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'trend_threshold' in self.config and not isinstance(self.config['trend_threshold'], (int, float)):
            raise ValueError("trend_threshold must be a number")
        if 'volatility_window' in self.config and (not isinstance(self.config['volatility_window'], int) or self.config['volatility_window'] <= 0):
            raise ValueError("volatility_window must be a positive integer")
        if 'correlation_threshold' in self.config and (not isinstance(self.config['correlation_threshold'], (int, float)) or not -1 <= self.config['correlation_threshold'] <= 1):
            raise ValueError("correlation_threshold must be between -1 and 1")

    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend using moving averages."""
        try:
            if 'Close' not in data.columns:
                raise KeyError("Missing 'Close' column in input data.")
            
            ma_short = data['Close'].rolling(window=20).mean()
            ma_long = data['Close'].rolling(window=50).mean()
            trend_strength = (ma_short - ma_long) / ma_long
            
            # Safe index access with validation
            if len(trend_strength.dropna()) == 0:
                raise ValueError("Insufficient data for trend analysis")
            
            current_trend = 'up' if trend_strength.iloc[-1] > self.config.get('trend_threshold', 0) else 'down'
            trend_changes = np.diff(np.signbit(trend_strength.dropna()))
            
            # Safe trend duration calculation
            trend_duration = len(data)
            if len(np.where(trend_changes)[0]) > 0:
                trend_duration = len(data) - np.where(trend_changes)[0][-1]
            
            result = {
                'trend_direction': current_trend,
                'trend_strength': float(trend_strength.iloc[-1]),  # Convert to native Python type
                'trend_duration': int(trend_duration),
                'ma_short': float(ma_short.iloc[-1]),
                'ma_long': float(ma_long.iloc[-1])
            }
            
            # Log metrics using our centralized system
            log_metrics("trend", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            raise

    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility."""
        try:
            if 'Close' not in data.columns:
                raise KeyError("Missing 'Close' column in input data.")
            
            returns = data['Close'].pct_change()
            current_volatility = returns.std() * np.sqrt(252)
            historical_volatility = returns.rolling(window=self.config.get('volatility_window', 252)).std() * np.sqrt(252)
            
            # Safe index access with validation
            hv_non_null = historical_volatility.dropna()
            if len(hv_non_null) < 2:
                raise ValueError("Insufficient data for volatility analysis")
            
            prev_volatility = hv_non_null.iloc[-2]
            volatility_trend = 'increasing' if current_volatility > prev_volatility else 'decreasing'
            
            # Safe volatility rank calculation
            vol_min = historical_volatility.min()
            vol_max = historical_volatility.max()
            volatility_rank = (current_volatility - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
            
            result = {
                'current_volatility': float(current_volatility),
                'volatility_rank': float(volatility_rank),
                'volatility_trend': volatility_trend,
                'historical_volatility': float(historical_volatility.iloc[-1])
            }
            
            log_metrics("volatility", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            raise

    def analyze_correlation(self, data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation with market data."""
        try:
            # Validate input data
            for df in [data, market_data]:
                if 'Close' not in df.columns:
                    raise KeyError("Missing 'Close' column in one of the input datasets.")
            
            returns = data['Close'].pct_change()
            market_returns = market_data['Close'].pct_change()
            correlation = returns.corr(market_returns)
            rolling_correlation = returns.rolling(window=252).corr(market_returns)
            
            # Safe correlation trend calculation
            rc_non_null = rolling_correlation.dropna()
            if len(rc_non_null) < 2:
                correlation_trend = 'unknown'
            else:
                correlation_trend = 'increasing' if correlation > rc_non_null.iloc[-2] else 'decreasing'
            
            result = {
                'correlation': float(correlation),
                'correlation_trend': correlation_trend,
                'rolling_correlation': float(rolling_correlation.dropna().iloc[-1]) if len(rolling_correlation.dropna()) > 0 else None
            }
            
            log_metrics("correlation", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation: {e}")
            raise

    def analyze_market_conditions(self, data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market conditions."""
        try:
            trend = self.analyze_trend(data)
            volatility = self.analyze_volatility(data)
            correlation = self.analyze_correlation(data, market_data)
            
            combined = {
                "trend": trend,
                "volatility": volatility,
                "correlation": correlation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            log_metrics("market_conditions", combined)
            return combined
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            raise 