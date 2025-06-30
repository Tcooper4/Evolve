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
import warnings

# Workaround for pandas_ta numpy compatibility issue
try:
    # Patch numpy import issue in pandas_ta
    import numpy
    if not hasattr(numpy, 'NaN'):
        numpy.NaN = numpy.nan
    
    import pandas_ta as ta
except ImportError as e:
    warnings.warn(f"pandas_ta import failed: {e}. Technical indicators may not be available.")
    ta = None

from trading.logs.logger import log_metrics

# Safe numba import with fallback
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ numba not installed – JIT optimization will be skipped.")
    
    # Create a dummy jit decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return {'success': True, 'result': {'success': True, 'result': func, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        return decorator

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
            
                return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
            
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
            
        return {'success': True, 'result': rsi, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
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
            rsi = ta.rsi(data['Close'], length=window)
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
            
            return {'success': True, 'result': rsi, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
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
            macd = ta.macd(
                data['Close'],
                fast=self.config['macd_fast'],
                slow=self.config['macd_slow'],
                signal=self.config['macd_signal']
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
            
            return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                'macd': macd[f'MACD_{self.config["macd_fast"]}_{self.config["macd_slow"]}_{self.config["macd_signal"]}'],
                'signal': macd[f'MACDs_{self.config["macd_fast"]}_{self.config["macd_slow"]}_{self.config["macd_signal"]}'],
                'histogram': macd[f'MACDh_{self.config["macd_fast"]}_{self.config["macd_slow"]}_{self.config["macd_signal"]}']
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
            sma = ta.sma(data['Close'], length=window)
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
            
            return {'success': True, 'result': sma, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
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
            ema = ta.ema(data['Close'], length=window)
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
            
            return {'success': True, 'result': ema, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating EMA: {str(e)}")
            raise

    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with 'Close' prices
            window: Window period for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            Dictionary containing upper, middle, and lower bands
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must contain 'Close' column")
            
            start_time = datetime.now()
            bb = ta.bbands(data['Close'], length=window, std=num_std)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'Bollinger Bands',
                    'window': window,
                    'std': num_std,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                'upper': bb[f'BBU_{window}_{num_std}'],
                'middle': bb[f'BBM_{window}_{num_std}'],
                'lower': bb[f'BBL_{window}_{num_std}']
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
            
            return {'success': True, 'result': indicators, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error calculating all indicators: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict[str, int]:
        """Get performance metrics for indicator calculations."""
        return {'success': True, 'result': self.performance_metrics.copy(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

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
            
            start_time = datetime.now()
            stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=k_period, d=d_period)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'Stochastic',
                    'k_period': k_period,
                    'd_period': d_period,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {'success': True, 'result': stoch, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
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
            
            start_time = datetime.now()
            atr = ta.atr(data['High'], data['Low'], data['Close'], length=period)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'ATR',
                    'period': period,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {'success': True, 'result': atr, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
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
            DataFrame with Ichimoku Cloud components
        """
        try:
            self._validate_data(data)
            
            start_time = datetime.now()
            ichimoku = ta.ichimoku(
                data['High'],
                data['Low'],
                tenkan=tenkan_period,
                kijun=kijun_period,
                senkou=senkou_span_b_period,
                displacement=displacement
            )
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_metrics['total_calculations'] += 1
            self.performance_metrics['cpu_calculations'] += 1
            
            if self.config.get('metrics_enabled', False):
                log_metrics("indicator_calculation", {
                    'indicator': 'Ichimoku',
                    'tenkan_period': tenkan_period,
                    'kijun_period': kijun_period,
                    'senkou_span_b_period': senkou_span_b_period,
                    'displacement': displacement,
                    'calculation_time': calculation_time,
                    'data_points': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return {'success': True, 'result': ichimoku, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            raise 