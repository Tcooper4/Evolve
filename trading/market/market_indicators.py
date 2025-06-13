import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import torch
from scipy import stats
from numba import jit
import warnings

class MarketIndicators:
    """Class for calculating market indicators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the market indicators calculator.
        
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
            
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU for calculations")
            
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'rsi_period' in self.config:
            if not isinstance(self.config['rsi_period'], int) or self.config['rsi_period'] <= 0:
                raise ValueError("rsi_period must be a positive integer")
                
        if 'macd_fast' in self.config:
            if not isinstance(self.config['macd_fast'], int) or self.config['macd_fast'] <= 0:
                raise ValueError("macd_fast must be a positive integer")
                
        if 'macd_slow' in self.config:
            if not isinstance(self.config['macd_slow'], int) or self.config['macd_slow'] <= 0:
                raise ValueError("macd_slow must be a positive integer")
                
        if 'macd_signal' in self.config:
            if not isinstance(self.config['macd_signal'], int) or self.config['macd_signal'] <= 0:
                raise ValueError("macd_signal must be a positive integer")
                
        if 'bb_period' in self.config:
            if not isinstance(self.config['bb_period'], int) or self.config['bb_period'] <= 0:
                raise ValueError("bb_period must be a positive integer")
                
        if 'bb_std' in self.config:
            if not isinstance(self.config['bb_std'], (int, float)) or self.config['bb_std'] <= 0:
                raise ValueError("bb_std must be a positive number")
                
        if 'use_gpu' in self.config:
            if not isinstance(self.config['use_gpu'], bool):
                raise ValueError("use_gpu must be a boolean")
                
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
            
    def calculate_rsi(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with OHLCV data
            period: RSI period (default: from config or 14)
            
        Returns:
            Series with RSI values
        """
        try:
            self._validate_data(data)
            period = period or self.config.get('rsi_period', 14)
            
            if self.device.type == 'cuda':
                # GPU implementation
                prices = torch.tensor(data['Close'].values, device=self.device)
                deltas = torch.diff(prices)
                gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
                losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
                
                avg_gains = torch.zeros_like(prices)
                avg_losses = torch.zeros_like(prices)
                
                # Initialize first values
                avg_gains[period] = gains[:period].mean()
                avg_losses[period] = losses[:period].mean()
                
                # Calculate subsequent values
                for i in range(period + 1, len(prices)):
                    avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
                    avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
                    
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))
                return pd.Series(rsi.cpu().numpy(), index=data.index)
            else:
                # CPU implementation with Numba
                return pd.Series(
                    self._calculate_rsi_fast(data['Close'].values, period),
                    index=data.index
                )
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            raise
            
    def calculate_macd(self, data: pd.DataFrame, fast_period: Optional[int] = None,
                      slow_period: Optional[int] = None, signal_period: Optional[int] = None) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with OHLCV data
            fast_period: Fast period (default: from config or 12)
            slow_period: Slow period (default: from config or 26)
            signal_period: Signal period (default: from config or 9)
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        try:
            self._validate_data(data)
            
            fast_period = fast_period or self.config.get('macd_fast', 12)
            slow_period = slow_period or self.config.get('macd_slow', 26)
            signal_period = signal_period or self.config.get('macd_signal', 9)
            
            if self.device.type == 'cuda':
                # GPU implementation
                prices = torch.tensor(data['Close'].values, device=self.device)
                
                # Calculate MACD line
                exp1 = torch.zeros_like(prices)
                exp2 = torch.zeros_like(prices)
                
                # Initialize first values
                exp1[0] = prices[0]
                exp2[0] = prices[0]
                
                # Calculate subsequent values
                for i in range(1, len(prices)):
                    exp1[i] = (prices[i] * (2/(fast_period+1)) + 
                              exp1[i-1] * (1 - 2/(fast_period+1)))
                    exp2[i] = (prices[i] * (2/(slow_period+1)) + 
                              exp2[i-1] * (1 - 2/(slow_period+1)))
                    
                macd = exp1 - exp2
                
                # Calculate signal line
                signal = torch.zeros_like(macd)
                signal[0] = macd[0]
                
                for i in range(1, len(macd)):
                    signal[i] = (macd[i] * (2/(signal_period+1)) + 
                               signal[i-1] * (1 - 2/(signal_period+1)))
                    
                histogram = macd - signal
                
                return pd.DataFrame({
                    'MACD': macd.cpu().numpy(),
                    'Signal': signal.cpu().numpy(),
                    'Histogram': histogram.cpu().numpy()
                }, index=data.index)
            else:
                # CPU implementation
                exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
                exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=signal_period, adjust=False).mean()
                histogram = macd - signal
                
                return pd.DataFrame({
                    'MACD': macd,
                    'Signal': signal,
                    'Histogram': histogram
                })
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            raise
            
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: Optional[int] = None,
                                std_dev: Optional[float] = None) -> pd.DataFrame:
        """Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            period: Moving average period (default: from config or 20)
            std_dev: Number of standard deviations (default: from config or 2.0)
            
        Returns:
            DataFrame with Middle, Upper, and Lower bands
        """
        try:
            self._validate_data(data)
            
            period = period or self.config.get('bb_period', 20)
            std_dev = std_dev or self.config.get('bb_std', 2.0)
            
            if self.device.type == 'cuda':
                # GPU implementation
                prices = torch.tensor(data['Close'].values, device=self.device)
                
                # Calculate middle band (SMA)
                middle = torch.zeros_like(prices)
                for i in range(period-1, len(prices)):
                    middle[i] = prices[i-period+1:i+1].mean()
                    
                # Calculate standard deviation
                std = torch.zeros_like(prices)
                for i in range(period-1, len(prices)):
                    std[i] = prices[i-period+1:i+1].std()
                    
                # Calculate bands
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
                return pd.DataFrame({
                    'Middle': middle.cpu().numpy(),
                    'Upper': upper.cpu().numpy(),
                    'Lower': lower.cpu().numpy()
                }, index=data.index)
            else:
                # CPU implementation
                middle = data['Close'].rolling(window=period).mean()
                std = data['Close'].rolling(window=period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
                return pd.DataFrame({
                    'Middle': middle,
                    'Upper': upper,
                    'Lower': lower
                })
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise
            
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
            
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Calculate all market indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all indicator values
        """
        try:
            self._validate_data(data)
            
            indicators = {}
            
            # Calculate basic indicators
            indicators['RSI'] = self.calculate_rsi(data)
            indicators['MACD'] = self.calculate_macd(data)
            indicators['Bollinger Bands'] = self.calculate_bollinger_bands(data)
            indicators['Stochastic'] = self.calculate_stochastic(data)
            indicators['ATR'] = self.calculate_atr(data)
            indicators['Ichimoku'] = self.calculate_ichimoku(data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating all indicators: {str(e)}")
            raise

    def calculate_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            data (pd.DataFrame): The market data.
            window (int): The window size for the SMA.

        Returns:
            pd.Series: The calculated SMA.
        """
        return data['Close'].rolling(window=window).mean()

    def calculate_ema(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            data (pd.DataFrame): The market data.
            window (int): The window size for the EMA.

        Returns:
            pd.Series: The calculated EMA.
        """
        return data['Close'].ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            data (pd.DataFrame): The market data.
            window (int): The window size for the RSI.

        Returns:
            pd.Series: The calculated RSI.
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)) 