"""
Market Indicators Module

This module provides technical analysis indicators for market data, with both CPU and GPU implementations.
It includes robust error handling, logging, and performance monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import torch
from scipy import stats
from numba import jit
import warnings
import talib
from trading.logs.logger import log_metrics

class MarketIndicators:
    """Class for calculating technical analysis indicators with GPU/CPU support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the market indicators calculator.
        
        Args:
            config: Configuration dictionary with indicator parameters
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
        
        # Initialize performance tracking
        self.performance_metrics = {
            'total_calculations': 0,
            'gpu_calculations': 0,
            'cpu_calculations': 0,
            'errors': 0
        }
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU for calculations")
            
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ['rsi_window', 'macd_fast', 'macd_slow', 'macd_signal']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        if not isinstance(self.config['rsi_window'], int) or self.config['rsi_window'] <= 0:
            raise ValueError("rsi_window must be a positive integer")
        
        if not isinstance(self.config['macd_fast'], int) or self.config['macd_fast'] <= 0:
            raise ValueError("macd_fast must be a positive integer")
        
        if not isinstance(self.config['macd_slow'], int) or self.config['macd_slow'] <= 0:
            raise ValueError("macd_slow must be a positive integer")
        
        if not isinstance(self.config['macd_signal'], int) or self.config['macd_signal'] <= 0:
            raise ValueError("macd_signal must be a positive integer")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data is empty")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")
            
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order")
            
        # Check for missing values
        if data.isnull().any().any():
            warnings.warn("Data contains missing values. Consider handling them before calculation.")
            
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=np.number)).any().any():
            raise ValueError("Data contains infinite values")
            
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi_fast(prices: np.ndarray, period: int) -> np.ndarray:
        """Fast RSI calculation using Numba.
        
        Args:
            prices: Array of closing prices
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1.+rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)
            
        return rsi
            
    def calculate_rsi(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with 'Close' prices
            window: RSI window period (defaults to config value)
            
        Returns:
            Series containing RSI values
        """
        try:
            window = window or self.config['rsi_window']
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            rsi = talib.RSI(data['Close'].values, timeperiod=window)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'RSI',
                    'window': window,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return pd.Series(rsi, index=data.index)
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with 'Close' prices
            
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            macd, signal, hist = talib.MACD(
                data['Close'].values,
                fastperiod=self.config['macd_fast'],
                slowperiod=self.config['macd_slow'],
                signalperiod=self.config['macd_signal']
            )
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'MACD',
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {
                'macd': pd.Series(macd, index=data.index),
                'signal': pd.Series(signal, index=data.index),
                'histogram': pd.Series(hist, index=data.index)
            }
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA).
        
        Args:
            data: DataFrame with 'Close' prices
            window: SMA window period
            
        Returns:
            Series containing SMA values
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            sma = talib.SMA(data['Close'].values, timeperiod=window)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'SMA',
                    'window': window,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return pd.Series(sma, index=data.index)
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating SMA: {str(e)}")
            raise

    def calculate_ema(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA).
        
        Args:
            data: DataFrame with 'Close' prices
            window: EMA window period
            
        Returns:
            Series containing EMA values
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            ema = talib.EMA(data['Close'].values, timeperiod=window)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'EMA',
                    'window': window,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return pd.Series(ema, index=data.index)
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating EMA: {str(e)}")
            raise

    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with 'Close' prices
            window: BB window period
            num_std: Number of standard deviations
            
        Returns:
            Dictionary containing upper band, middle band, and lower band
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            upper, middle, lower = talib.BBANDS(
                data['Close'].values,
                timeperiod=window,
                nbdevup=num_std,
                nbdevdn=num_std
            )
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'Bollinger Bands',
                    'window': window,
                    'num_std': num_std,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {
                'upper': pd.Series(upper, index=data.index),
                'middle': pd.Series(middle, index=data.index),
                'lower': pd.Series(lower, index=data.index)
            }
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, Dict[str, pd.Series]]]:
        """Calculate all technical indicators for the given data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            
            # Calculate indicators
            indicators = {
                'rsi': self.calculate_rsi(data),
                'macd': self.calculate_macd(data),
                'sma_20': self.calculate_sma(data, 20),
                'sma_50': self.calculate_sma(data, 50),
                'ema_20': self.calculate_ema(data, 20),
                'ema_50': self.calculate_ema(data, 50),
                'bollinger_bands': self.calculate_bollinger_bands(data)
            }
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'All Indicators',
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return indicators
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating all indicators: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, int]:
        """Get performance metrics for indicator calculations."""
        return self.performance_metrics.copy()

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14,
                           d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLCV data
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            DataFrame with %K and %D values
        """
        try:
            self._validate_data(data)
            
            if self.device.type == 'cuda':
                # GPU implementation
                high = torch.tensor(data['High'].values, device=self.device)
                low = torch.tensor(data['Low'].values, device=self.device)
                close = torch.tensor(data['Close'].values, device=self.device)
                
                # Calculate %K
                k = torch.zeros_like(close)
                for i in range(k_period-1, len(close)):
                    period_high = high[i-k_period+1:i+1].max()
                    period_low = low[i-k_period+1:i+1].min()
                    k[i] = 100 * ((close[i] - period_low) / (period_high - period_low))
                    
                # Calculate %D
                d = torch.zeros_like(k)
                for i in range(d_period-1, len(k)):
                    d[i] = k[i-d_period+1:i+1].mean()
                    
                return pd.DataFrame({
                    '%K': k.cpu().numpy(),
                    '%D': d.cpu().numpy()
                }, index=data.index)
            else:
                # CPU implementation
                low_min = data['Low'].rolling(window=k_period).min()
                high_max = data['High'].rolling(window=k_period).max()
                k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
                d = k.rolling(window=d_period).mean()
                
                return pd.DataFrame({
                    '%K': k,
                    '%D': d
                })
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            raise
            
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLCV data
            period: ATR period (default: 14)
            
        Returns:
            Series with ATR values
        """
        try:
            self._validate_data(data)
            
            if self.device.type == 'cuda':
                # GPU implementation
                high = torch.tensor(data['High'].values, device=self.device)
                low = torch.tensor(data['Low'].values, device=self.device)
                close = torch.tensor(data['Close'].values, device=self.device)
                
                # Calculate True Range
                high_low = high - low
                high_close = torch.abs(high - torch.roll(close, 1))
                low_close = torch.abs(low - torch.roll(close, 1))
                
                true_range = torch.maximum(
                    torch.maximum(high_low, high_close),
                    low_close
                )
                
                # Calculate ATR
                atr = torch.zeros_like(true_range)
                atr[period-1] = true_range[:period].mean()
                
                for i in range(period, len(true_range)):
                    atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
                    
                return pd.Series(atr.cpu().numpy(), index=data.index)
            else:
                # CPU implementation
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                
                atr = true_range.rolling(window=period).mean()
                
                return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            raise
            
    def calculate_ichimoku(self, data: pd.DataFrame, 
                          tenkan_period: int = 9,
                          kijun_period: int = 26,
                          senkou_span_b_period: int = 52,
                          displacement: int = 26) -> pd.DataFrame:
        """Calculate Ichimoku Cloud.
        
        Args:
            data: DataFrame with OHLCV data
            tenkan_period: Tenkan-sen period (default: 9)
            kijun_period: Kijun-sen period (default: 26)
            senkou_span_b_period: Senkou Span B period (default: 52)
            displacement: Displacement period (default: 26)
            
        Returns:
            DataFrame with Ichimoku components
        """
        try:
            self._validate_data(data)
            
            if self.device.type == 'cuda':
                # GPU implementation
                high = torch.tensor(data['High'].values, device=self.device)
                low = torch.tensor(data['Low'].values, device=self.device)
                close = torch.tensor(data['Close'].values, device=self.device)
                
                # Calculate Tenkan-sen (Conversion Line)
                tenkan_high = torch.zeros_like(close)
                tenkan_low = torch.zeros_like(close)
                for i in range(tenkan_period-1, len(close)):
                    tenkan_high[i] = high[i-tenkan_period+1:i+1].max()
                    tenkan_low[i] = low[i-tenkan_period+1:i+1].min()
                tenkan_sen = (tenkan_high + tenkan_low) / 2
                
                # Calculate Kijun-sen (Base Line)
                kijun_high = torch.zeros_like(close)
                kijun_low = torch.zeros_like(close)
                for i in range(kijun_period-1, len(close)):
                    kijun_high[i] = high[i-kijun_period+1:i+1].max()
                    kijun_low[i] = low[i-kijun_period+1:i+1].min()
                kijun_sen = (kijun_high + kijun_low) / 2
                
                # Calculate Senkou Span A (Leading Span A)
                senkou_span_a = (tenkan_sen + kijun_sen) / 2
                
                # Calculate Senkou Span B (Leading Span B)
                senkou_span_b = torch.zeros_like(close)
                for i in range(senkou_span_b_period-1, len(close)):
                    period_high = high[i-senkou_span_b_period+1:i+1].max()
                    period_low = low[i-senkou_span_b_period+1:i+1].min()
                    senkou_span_b[i] = (period_high + period_low) / 2
                    
                # Displace Senkou Spans
                senkou_span_a = torch.roll(senkou_span_a, displacement)
                senkou_span_b = torch.roll(senkou_span_b, displacement)
                
                return pd.DataFrame({
                    'Tenkan-sen': tenkan_sen.cpu().numpy(),
                    'Kijun-sen': kijun_sen.cpu().numpy(),
                    'Senkou Span A': senkou_span_a.cpu().numpy(),
                    'Senkou Span B': senkou_span_b.cpu().numpy(),
                    'Chikou Span': close.cpu().numpy()
                }, index=data.index)
            else:
                # CPU implementation
                # Calculate Tenkan-sen (Conversion Line)
                tenkan_high = data['High'].rolling(window=tenkan_period).max()
                tenkan_low = data['Low'].rolling(window=tenkan_period).min()
                tenkan_sen = (tenkan_high + tenkan_low) / 2
                
                # Calculate Kijun-sen (Base Line)
                kijun_high = data['High'].rolling(window=kijun_period).max()
                kijun_low = data['Low'].rolling(window=kijun_period).min()
                kijun_sen = (kijun_high + kijun_low) / 2
                
                # Calculate Senkou Span A (Leading Span A)
                senkou_span_a = (tenkan_sen + kijun_sen) / 2
                
                # Calculate Senkou Span B (Leading Span B)
                senkou_span_b_high = data['High'].rolling(window=senkou_span_b_period).max()
                senkou_span_b_low = data['Low'].rolling(window=senkou_span_b_period).min()
                senkou_span_b = (senkou_span_b_high + senkou_span_b_low) / 2
                
                # Displace Senkou Spans
                senkou_span_a = senkou_span_a.shift(displacement)
                senkou_span_b = senkou_span_b.shift(displacement)
                
                return pd.DataFrame({
                    'Tenkan-sen': tenkan_sen,
                    'Kijun-sen': kijun_sen,
                    'Senkou Span A': senkou_span_a,
                    'Senkou Span B': senkou_span_b,
                    'Chikou Span': data['Close']
                })
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            raise 