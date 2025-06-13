import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime

class MarketAnalyzer:
    """Class for analyzing market data and conditions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the market analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'trend_threshold' in self.config:
            if not isinstance(self.config['trend_threshold'], (int, float)):
                raise ValueError("trend_threshold must be a number")
                
        if 'volatility_window' in self.config:
            if not isinstance(self.config['volatility_window'], int) or self.config['volatility_window'] <= 0:
                raise ValueError("volatility_window must be a positive integer")
                
        if 'correlation_threshold' in self.config:
            if not isinstance(self.config['correlation_threshold'], (int, float)) or not -1 <= self.config['correlation_threshold'] <= 1:
                raise ValueError("correlation_threshold must be between -1 and 1")
                
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market trend.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Calculate moving averages
            ma_short = data['Close'].rolling(window=20).mean()
            ma_long = data['Close'].rolling(window=50).mean()
            
            # Calculate trend strength
            trend_strength = (ma_short - ma_long) / ma_long
            
            # Determine trend direction
            current_trend = 'up' if trend_strength.iloc[-1] > 0 else 'down'
            
            # Calculate trend duration
            trend_changes = np.diff(np.signbit(trend_strength))
            trend_duration = len(data) - np.where(trend_changes)[0][-1] if len(np.where(trend_changes)[0]) > 0 else len(data)
            
            return {
                'trend_direction': current_trend,
                'trend_strength': trend_strength.iloc[-1],
                'trend_duration': trend_duration,
                'ma_short': ma_short.iloc[-1],
                'ma_long': ma_long.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            raise
            
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volatility analysis results
        """
        try:
            # Calculate returns
            returns = data['Close'].pct_change()
            
            # Calculate volatility metrics
            current_volatility = returns.std() * np.sqrt(252)  # Annualized
            historical_volatility = returns.rolling(window=252).std() * np.sqrt(252)
            volatility_rank = (current_volatility - historical_volatility.min()) / (historical_volatility.max() - historical_volatility.min())
            
            # Calculate volatility trend
            hv_non_null = historical_volatility.dropna()
            if len(hv_non_null) > 1:
                prev_volatility = hv_non_null.iloc[-2]
                volatility_trend = 'increasing' if current_volatility > prev_volatility else 'decreasing'
            else:
                volatility_trend = 'unknown'
            
            return {
                'current_volatility': current_volatility,
                'volatility_rank': volatility_rank.iloc[-1],
                'volatility_trend': volatility_trend,
                'historical_volatility': historical_volatility.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            raise
            
    def analyze_correlation(self, data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation with market.
        
        Args:
            data: DataFrame with OHLCV data
            market_data: DataFrame with market OHLCV data
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            # Calculate returns
            returns = data['Close'].pct_change()
            market_returns = market_data['Close'].pct_change()
            
            # Calculate correlation
            correlation = returns.corr(market_returns)
            
            # Calculate rolling correlation
            rolling_correlation = returns.rolling(window=252).corr(market_returns)
            
            # Determine correlation trend
            correlation_trend = 'increasing' if correlation > rolling_correlation.iloc[-2] else 'decreasing'
            
            return {
                'correlation': correlation,
                'correlation_trend': correlation_trend,
                'rolling_correlation': rolling_correlation.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation: {str(e)}")
            raise
            
    def analyze_market_conditions(self, data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market conditions.
        
        Args:
            data: DataFrame with OHLCV data
            market_data: DataFrame with market OHLCV data
            
        Returns:
            Dictionary with market conditions analysis
        """
        try:
            # Get individual analyses
            trend_analysis = self.analyze_trend(data)
            volatility_analysis = self.analyze_volatility(data)
            correlation_analysis = self.analyze_correlation(data, market_data)
            
            # Combine analyses
            market_conditions = {
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'correlation': correlation_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            raise 