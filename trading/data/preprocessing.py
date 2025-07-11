import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = Path("memory/logs/preprocessing.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

class DataPreprocessor:
    """Data preprocessing class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        self._validate_config()
        self.scaling_method = self.config.get('scaling_method', 'standard')
        self.missing_value_method = self.config.get('missing_value_method', 'ffill')
        self.outlier_method = self.config.get('outlier_method', 'zscore')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.scaler = None
        self.is_fitted = False
        self._feature_stats = {}

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'scaling_method' in self.config:
            if self.config['scaling_method'] not in ['standard', 'minmax']:
                raise ValueError("scaling_method must be 'standard' or 'minmax'")
                
        if 'missing_value_method' in self.config:
            if self.config['missing_value_method'] not in ['ffill', 'bfill', 'interpolate']:
                raise ValueError("missing_value_method must be 'ffill', 'bfill', or 'interpolate'")
                
        if 'outlier_method' in self.config:
            if self.config['outlier_method'] not in ['zscore', 'iqr']:
                raise ValueError("outlier_method must be 'zscore' or 'iqr'")
                
        if 'outlier_threshold' in self.config:
            if not isinstance(self.config['outlier_threshold'], (int, float)) or self.config['outlier_threshold'] <= 0:
                raise ValueError("outlier_threshold must be a positive number")
                
    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data is None:
            raise ValueError("Input data cannot be None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data is empty")
        # Check for required columns with flexible validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            for col in missing_columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else 100.0
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Data index must be convertible to DatetimeIndex: {e}")
        if not data.index.is_monotonic_increasing:
            logger.warning("Data index not sorted, sorting in ascending order")
            data.sort_index(inplace=True)
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected, replacing with NaN")
            data = data.replace([np.inf, -np.inf], np.nan)

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        if data.empty:
            raise ValueError("Input data is empty")
            
        result = data.copy()
        
        # Check for columns with all NaN values
        all_nan_cols = result.columns[result.isna().all()].tolist()
        if all_nan_cols:
            raise ValueError(f"Columns with all NaN values: {all_nan_cols}")
        
        # Handle missing values based on method
        if self.missing_value_method == 'ffill':
            result = result.ffill()
        elif self.missing_value_method == 'bfill':
            result = result.bfill()
        elif self.missing_value_method == 'interpolate':
            # Check if we have enough data points for interpolation
            if len(result) < 2:
                raise ValueError("Need at least 2 data points for interpolation")
            result = result.interpolate(method='linear')
        
        # Fill any remaining NaN values with forward fill
        result = result.ffill().bfill()
        
        # Verify no NaN values remain
        if result.isna().any().any():
            raise ValueError("Unable to handle all missing values")
        
        return result

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        if data.empty:
            raise ValueError("Input data is empty")
            
        result = data.copy()
        
        for col in result.select_dtypes(include=[np.number]).columns:
            if self.outlier_method == 'zscore':
                # Handle case where std=0
                std = result[col].std()
                if std == 0:
                    continue
                
                z_scores = np.abs((result[col] - result[col].mean()) / std)
                outliers = z_scores > self.outlier_threshold
                
                if outliers.any():
                    # Replace outliers with mean
                    result.loc[outliers, col] = result[col].mean()
            
            elif self.outlier_method == 'iqr':
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                
                if outliers.any():
                    # Replace outliers with bounds
                    result.loc[result[col] < lower_bound, col] = lower_bound
                    result.loc[result[col] > upper_bound, col] = upper_bound
        
        return result
        
    def _compute_feature_stats(self, data: pd.DataFrame) -> None:
        """Compute feature statistics."""
        self._feature_stats = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        self._validate_input(data)
        result = self._handle_missing_values(data)
        result = self._handle_outliers(result)
        return result
        
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data using the specified scaling method."""
        self._validate_input(data)
        
        if not self.is_fitted:
            self.fit(data)
        
        return self.transform(data)
        
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit the preprocessor to the data."""
        self._validate_input(data)
        self.scaler = DataScaler(self.config)
        self.scaler.fit(data)
        self.is_fitted = True
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_input(data)
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.scaler.transform(data)
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the data."""
        self._validate_input(data)
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return self.scaler.inverse_transform(data)
        
    def get_params(self) -> Dict[str, Any]:
        """Get preprocessor parameters."""
        return self.config.copy()
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality metrics."""
        try:
            self._validate_input(data)
            
            metrics = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'duplicate_percentage': (data.duplicated().sum() / len(data) * 100),
                'data_types': data.dtypes.to_dict(),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'date_range': {
                    'start': data.index.min().isoformat() if not data.empty else None,
                    'end': data.index.max().isoformat() if not data.empty else None,
                    'duration_days': (data.index.max() - data.index.min()).days if len(data) > 1 else 0
                },
                'outliers_detected': 0,
                'infinite_values': np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            }
            
            # Detect outliers using IQR method
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                metrics['outliers_detected'] += outliers
            
            logger.info(f"Data quality metrics calculated: {metrics['total_rows']} rows, "
                       f"{metrics['missing_percentage']:.1f}% missing values")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {e}")
            return {'error': str(e)}
        
    def set_params(self, **params) -> 'DataPreprocessor':
        """Set preprocessor parameters."""
        self.config.update(params)
        self._validate_config()
        return self

class FeatureEngineering:
    """Feature engineering class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineering with configuration."""
        self.config = config or {}
        self._validate_config()
        
        # Initialize parameters with defaults
        self.ma_windows = self.config.get('ma_windows', [5, 10, 20, 50, 200])
        self.rsi_window = self.config.get('rsi_window', 14)
        self.macd_params = self.config.get('macd_params', {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        })
        self.bb_window = self.config.get('bb_window', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.fourier_periods = self.config.get('fourier_periods', [5, 10, 20])
        self.lag_periods = self.config.get('lag_periods', [1, 2, 3, 5, 10])
        self.volume_ma_windows = self.config.get('volume_ma_windows', [5, 10, 20])

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate moving average windows
        if 'ma_windows' in self.config:
            if not isinstance(self.config['ma_windows'], list):
                raise ValueError("ma_windows must be a list")
            if not all(isinstance(x, int) and x > 0 for x in self.config['ma_windows']):
                raise ValueError("ma_windows must contain positive integers")
            if not all(x < y for x, y in zip(self.config['ma_windows'], self.config['ma_windows'][1:])):
                raise ValueError("ma_windows must be in ascending order")

        # Validate RSI window
        if 'rsi_window' in self.config:
            if not isinstance(self.config['rsi_window'], int) or self.config['rsi_window'] <= 0:
                raise ValueError("rsi_window must be a positive integer")

        # Validate MACD parameters
        if 'macd_params' in self.config:
            required_keys = {'fast_period', 'slow_period', 'signal_period'}
            if not all(key in self.config['macd_params'] for key in required_keys):
                raise ValueError("macd_params must contain fast_period, slow_period, and signal_period")
            if not all(isinstance(self.config['macd_params'][key], int) and self.config['macd_params'][key] > 0 
                      for key in required_keys):
                raise ValueError("MACD periods must be positive integers")
            if self.config['macd_params']['fast_period'] >= self.config['macd_params']['slow_period']:
                raise ValueError("fast_period must be less than slow_period")

        # Validate Bollinger Bands parameters
        if 'bb_window' in self.config:
            if not isinstance(self.config['bb_window'], int) or self.config['bb_window'] <= 0:
                raise ValueError("bb_window must be a positive integer")
        if 'bb_std' in self.config:
            if not isinstance(self.config['bb_std'], (int, float)) or self.config['bb_std'] <= 0:
                raise ValueError("bb_std must be a positive number")

        # Validate Fourier periods
        if 'fourier_periods' in self.config:
            if not isinstance(self.config['fourier_periods'], list):
                raise ValueError("fourier_periods must be a list")
            if not all(isinstance(x, int) and x > 0 for x in self.config['fourier_periods']):
                raise ValueError("fourier_periods must contain positive integers")

        # Validate lag periods
        if 'lag_periods' in self.config:
            if not isinstance(self.config['lag_periods'], list):
                raise ValueError("lag_periods must be a list")
            if not all(isinstance(x, int) and x > 0 for x in self.config['lag_periods']):
                raise ValueError("lag_periods must contain positive integers")

        # Validate volume MA windows
        if 'volume_ma_windows' in self.config:
            if not isinstance(self.config['volume_ma_windows'], list):
                raise ValueError("volume_ma_windows must be a list")
            if not all(isinstance(x, int) and x > 0 for x in self.config['volume_ma_windows']):
                raise ValueError("volume_ma_windows must contain positive integers")

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data is None:
            raise ValueError("Input data cannot be None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data is empty")
        # Check for required columns with flexible validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            for col in missing_columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else 100.0
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Data index must be convertible to DatetimeIndex: {e}")
        if not data.index.is_monotonic_increasing:
            logger.warning("Data index not sorted, sorting in ascending order")
            data.sort_index(inplace=True)
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected, replacing with NaN")
            data = data.replace([np.inf, -np.inf], np.nan)
        # Check for sufficient data points
        min_required = max(
            max(self.ma_windows),
            self.rsi_window,
            self.macd_params['slow_period'] + self.macd_params['signal_period'],
            self.bb_window,
            max(self.fourier_periods),
            max(self.lag_periods),
            max(self.volume_ma_windows)
        )
        if len(data) < min_required:
            raise ValueError(f"Need at least {min_required} data points for feature calculation")

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        self._validate_input(data)
        result = pd.DataFrame(index=data.index)
        
        for window in self.ma_windows:
            if window > len(data):
                continue
            result[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            result[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
        
        return result

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI."""
        self._validate_input(data)
        
        if len(data) < self.rsi_window + 1:
            raise ValueError(f"Need at least {self.rsi_window + 1} data points for RSI calculation")
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({'RSI': rsi}, index=data.index)

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        self._validate_input(data)
        
        min_required = self.macd_params['slow_period'] + self.macd_params['signal_period']
        if len(data) < min_required:
            raise ValueError(f"Need at least {min_required} data points for MACD calculation")
        
        exp1 = data['Close'].ewm(span=self.macd_params['fast_period'], adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.macd_params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_params['signal_period'], adjust=False).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Histogram': histogram
        }, index=data.index)

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        self._validate_input(data)
        
        if len(data) < self.bb_window:
            raise ValueError(f"Need at least {self.bb_window} data points for Bollinger Bands calculation")
        
        middle_band = data['Close'].rolling(window=self.bb_window).mean()
        std = data['Close'].rolling(window=self.bb_window).std()
        upper_band = middle_band + (std * self.bb_std)
        lower_band = middle_band - (std * self.bb_std)
        bandwidth = (upper_band - lower_band) / middle_band
        
        return pd.DataFrame({
            'BB_Middle': middle_band,
            'BB_Upper': upper_band,
            'BB_Lower': lower_band,
            'BB_Bandwidth': bandwidth
        }, index=data.index)

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators."""
        self._validate_input(data)
        result = pd.DataFrame(index=data.index)
        
        # Volume moving averages
        for window in self.volume_ma_windows:
            if window > len(data):
                continue
            result[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
        
        # Volume trend
        result['Volume_Trend'] = data['Volume'].pct_change()
        
        # Volume Price Trend (VPT)
        result['VPT'] = (data['Volume'] * (data['Close'] - data['Close'].shift(1)) / 
                        data['Close'].shift(1)).cumsum()
        
        # On-Balance Volume (OBV)
        result['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
        
        return result

    def calculate_fourier_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fourier transform features."""
        self._validate_input(data)
        result = pd.DataFrame(index=data.index)
        
        for period in self.fourier_periods:
            if period > len(data):
                continue
            # Calculate Fourier transform
            fft = np.fft.fft(data['Close'].values)
            # Get the magnitude of the first period
            magnitude = np.abs(fft[period])
            result[f'Fourier_{period}'] = magnitude
        
        return result

    def calculate_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate lag features."""
        self._validate_input(data)
        result = pd.DataFrame(index=data.index)
        
        for period in self.lag_periods:
            if period > len(data):
                continue
            result[f'Close_Lag_{period}'] = data['Close'].shift(period)
        
        return result

    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        self._validate_input(data)
        result = pd.DataFrame(index=data.index)
        
        # Rate of Change (ROC)
        result['ROC'] = data['Close'].pct_change(periods=10) * 100
        
        # Stochastic Oscillator
        if len(data) >= 14:
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            result['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            result['Stoch_D'] = result['Stoch_K'].rolling(window=3).mean()
        
        return result

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features."""
        self._validate_input(data)
        
        features = []
        
        # Calculate all features
        features.append(self.calculate_moving_averages(data))
        features.append(self.calculate_rsi(data))
        features.append(self.calculate_macd(data))
        features.append(self.calculate_bollinger_bands(data))
        features.append(self.calculate_volume_indicators(data))
        features.append(self.calculate_fourier_features(data))
        features.append(self.calculate_lag_features(data))
        features.append(self.calculate_momentum_indicators(data))
        
        # Combine all features
        result = pd.concat(features, axis=1)
        
        # Fill NaN values with 0
        result = result.fillna(0)
        
        return result

    def get_feature_list(self) -> List[str]:
        """Get list of engineered features."""
        features = []
        
        # Moving averages
        for window in self.ma_windows:
            features.extend([f'SMA_{window}', f'EMA_{window}'])
        
        # RSI
        features.append('RSI')
        
        # MACD
        features.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
        
        # Bollinger Bands
        features.extend(['BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Bandwidth'])
        
        # Volume indicators
        for window in self.volume_ma_windows:
            features.append(f'Volume_MA_{window}')
        features.extend(['Volume_Trend', 'VPT', 'OBV'])
        
        # Fourier features
        for period in self.fourier_periods:
            features.append(f'Fourier_{period}')
        
        # Lag features
        for period in self.lag_periods:
            features.append(f'Close_Lag_{period}')
        
        # Momentum indicators
        features.extend(['ROC', 'Stoch_K', 'Stoch_D'])
        
        return features

    def get_params(self) -> Dict[str, Any]:
        """Get feature engineering parameters."""
        return {
            'ma_windows': self.ma_windows,
            'rsi_window': self.rsi_window,
            'macd_params': self.macd_params,
            'bb_window': self.bb_window,
            'bb_std': self.bb_std,
            'fourier_periods': self.fourier_periods,
            'lag_periods': self.lag_periods,
            'volume_ma_windows': self.volume_ma_windows
        }

    def set_params(self, **params) -> 'FeatureEngineering':
        """Set feature engineering parameters."""
        self.config.update(params)
        self._validate_config()
        
        # Update parameters
        if 'ma_windows' in params:
            self.ma_windows = params['ma_windows']
        if 'rsi_window' in params:
            self.rsi_window = params['rsi_window']
        if 'macd_params' in params:
            self.macd_params = params['macd_params']
        if 'bb_window' in params:
            self.bb_window = params['bb_window']
        if 'bb_std' in params:
            self.bb_std = params['bb_std']
        if 'fourier_periods' in params:
            self.fourier_periods = params['fourier_periods']
        if 'lag_periods' in params:
            self.lag_periods = params['lag_periods']
        if 'volume_ma_windows' in params:
            self.volume_ma_windows = params['volume_ma_windows']
        
        return self

class DataValidator:
    """Data validation class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration."""
        self.config = config or {}
        self._validate_config()
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.min_price = self.config.get('min_price', 0.0)
        self.max_price = self.config.get('max_price', float('inf'))

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'outlier_threshold' in self.config:
            if not isinstance(self.config['outlier_threshold'], (int, float)) or self.config['outlier_threshold'] <= 0:
                raise ValueError("outlier_threshold must be a positive number")
        
        if 'min_price' in self.config:
            if not isinstance(self.config['min_price'], (int, float)) or self.config['min_price'] < 0:
                raise ValueError("min_price must be a non-negative number")
        
        if 'max_price' in self.config:
            if not isinstance(self.config['max_price'], (int, float)) or self.config['max_price'] <= 0:
                raise ValueError("max_price must be a positive number")
            if 'min_price' in self.config and self.config['max_price'] <= self.config['min_price']:
                raise ValueError("max_price must be greater than min_price")

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")
        
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order")
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            raise ValueError("Duplicate timestamps found in data")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the data."""
        self._validate_input(data)
        
        # Check for non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Non-numeric columns found: {non_numeric}")
        
        # Check for infinite values
        inf_cols = data.columns[data.isin([np.inf, -np.inf]).any()].tolist()
        if inf_cols:
            raise ValueError(f"Infinite values found in columns: {inf_cols}")
        
        # Check for negative prices
        neg_price_cols = data.columns[data[['Open', 'High', 'Low', 'Close']].lt(0).any()].tolist()
        if neg_price_cols:
            raise ValueError(f"Negative prices found in columns: {neg_price_cols}")
        
        # Check price relationships
        if (data['High'] < data['Low']).any():
            raise ValueError("High price is less than Low price")
        if (data['Open'] > data['High']).any():
            raise ValueError("Open price is greater than High price")
        if (data['Open'] < data['Low']).any():
            raise ValueError("Open price is less than Low price")
        if (data['Close'] > data['High']).any():
            raise ValueError("Close price is greater than High price")
        if (data['Close'] < data['Low']).any():
            raise ValueError("Close price is less than Low price")
        
        # Check price bounds
        if (data[['Open', 'High', 'Low', 'Close']] < self.min_price).any().any():
            raise ValueError(f"Prices below minimum threshold {self.min_price}")
        if (data[['Open', 'High', 'Low', 'Close']] > self.max_price).any().any():
            raise ValueError(f"Prices above maximum threshold {self.max_price}")
        
        # Check for negative volume
        if (data['Volume'] < 0).any():
            raise ValueError("Negative volume found")
        
        # Check data frequency consistency
        if len(data) > 1:
            freq = pd.infer_freq(data.index)
            if freq is None:
                raise ValueError("Inconsistent data frequency")
        
        return True

    def check_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for outliers in the data."""
        self._validate_input(data)
        
        result = pd.DataFrame(index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Calculate z-scores
            mean = data[col].mean()
            std = data[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((data[col] - mean) / std)
            result[f'{col}_outlier'] = z_scores > self.outlier_threshold
        
        return result

    def check_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for missing values in the data."""
        self._validate_input(data)
        
        result = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            result[f'{col}_missing'] = data[col].isna()
        
        return result

    def check_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Check data types of columns."""
        self._validate_input(data)
        
        return {col: str(dtype) for col, dtype in data.dtypes.items()}

    def check_date_index(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check date index properties."""
        self._validate_input(data)
        
        return {
            'is_datetime': isinstance(data.index, pd.DatetimeIndex),
            'is_sorted': data.index.is_monotonic_increasing,
            'has_duplicates': data.index.duplicated().any(),
            'frequency': pd.infer_freq(data.index),
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'total_days': (data.index.max() - data.index.min()).days
        }

    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters."""
        return {
            'outlier_threshold': self.outlier_threshold,
            'min_price': self.min_price,
            'max_price': self.max_price
        }

    def set_params(self, **params) -> 'DataValidator':
        """Set validator parameters."""
        self.config.update(params)
        self._validate_config()
        
        if 'outlier_threshold' in params:
            self.outlier_threshold = params['outlier_threshold']
        if 'min_price' in params:
            self.min_price = params['min_price']
        if 'max_price' in params:
            self.max_price = params['max_price']
        
        return self

class DataScaler:
    """Data scaling class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scaler with configuration."""
        self.config = config or {}
        self._validate_config()
        self.scaling_method = self.config.get('scaling_method', 'standard')
        self.outlier_method = self.config.get('outlier_method', 'zscore')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.enable_outlier_clipping = self.config.get('enable_outlier_clipping', True)
        self.enable_zscore_standardization = self.config.get('enable_zscore_standardization', True)
        self._feature_stats = {}
        self.is_fitted = False

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'scaling_method' in self.config:
            if self.config['scaling_method'] not in ['standard', 'minmax', 'robust']:
                raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
                
        if 'outlier_method' in self.config:
            if self.config['outlier_method'] not in ['zscore', 'iqr', 'mad']:
                raise ValueError("outlier_method must be 'zscore', 'iqr', or 'mad'")
                
        if 'outlier_threshold' in self.config:
            if not isinstance(self.config['outlier_threshold'], (int, float)) or self.config['outlier_threshold'] <= 0:
                raise ValueError("outlier_threshold must be a positive number")

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
            
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configurable method."""
        if not self.enable_outlier_clipping:
            return data
            
        result = data.copy()
        
        for col in result.select_dtypes(include=[np.number]).columns:
            if self.outlier_method == 'zscore':
                result = self._clip_outliers_zscore(result, col)
            elif self.outlier_method == 'iqr':
                result = self._clip_outliers_iqr(result, col)
            elif self.outlier_method == 'mad':
                result = self._clip_outliers_mad(result, col)
        
        return result

    def _clip_outliers_zscore(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clip outliers using z-score method."""
        mean_val = data[column].mean()
        std_val = data[column].std()
        
        if std_val == 0:
            return data
            
        z_scores = np.abs((data[column] - mean_val) / std_val)
        outliers = z_scores > self.outlier_threshold
        
        if outliers.any():
            # Clip outliers to threshold
            upper_bound = mean_val + self.outlier_threshold * std_val
            lower_bound = mean_val - self.outlier_threshold * std_val
            data.loc[data[column] > upper_bound, column] = upper_bound
            data.loc[data[column] < lower_bound, column] = lower_bound
            
        return data

    def _clip_outliers_iqr(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clip outliers using IQR method."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return data
            
        upper_bound = Q3 + self.outlier_threshold * IQR
        lower_bound = Q1 - self.outlier_threshold * IQR
        
        data.loc[data[column] > upper_bound, column] = upper_bound
        data.loc[data[column] < lower_bound, column] = lower_bound
        
        return data

    def _clip_outliers_mad(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clip outliers using Median Absolute Deviation method."""
        median_val = data[column].median()
        mad = np.median(np.abs(data[column] - median_val))
        
        if mad == 0:
            return data
            
        upper_bound = median_val + self.outlier_threshold * mad
        lower_bound = median_val - self.outlier_threshold * mad
        
        data.loc[data[column] > upper_bound, column] = upper_bound
        data.loc[data[column] < lower_bound, column] = lower_bound
        
        return data

    def _standardize_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score standardization."""
        if not self.enable_zscore_standardization:
            return data
            
        result = data.copy()
        
        for col in result.select_dtypes(include=[np.number]).columns:
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                mean_val = stats['mean']
                std_val = stats['std']
                
                if std_val == 0:
                    std_val = 1.0  # Avoid division by zero
                    
                result[col] = (result[col] - mean_val) / std_val
        
        return result

    def fit(self, data: pd.DataFrame) -> 'DataScaler':
        """Fit the scaler to the data."""
        self._validate_input(data)
        
        # Handle outliers first
        data_cleaned = self._handle_outliers(data)
        
        self._feature_stats = {}
        for col in data_cleaned.select_dtypes(include=[np.number]).columns:
            # Handle case where std=0
            std = data_cleaned[col].std()
            if std == 0:
                std = 1.0  # Use 1.0 as default std to avoid division by zero
            
            self._feature_stats[col] = {
                'mean': data_cleaned[col].mean(),
                'std': std,
                'min': data_cleaned[col].min(),
                'max': data_cleaned[col].max(),
                'median': data_cleaned[col].median(),
                'q1': data_cleaned[col].quantile(0.25),
                'q3': data_cleaned[col].quantile(0.75)
            }
        
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_input(data)
        
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Handle outliers
        result = self._handle_outliers(data)
        
        # Apply scaling
        for col in result.select_dtypes(include=[np.number]).columns:
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                
                if self.scaling_method == 'standard':
                    # Z-score standardization
                    std = stats['std']
                    if std == 0:
                        std = 1.0
                    result[col] = (result[col] - stats['mean']) / std
                elif self.scaling_method == 'minmax':
                    # Min-max scaling
                    if stats['max'] == stats['min']:
                        result[col] = 0.0
                    else:
                        result[col] = (result[col] - stats['min']) / (stats['max'] - stats['min'])
                elif self.scaling_method == 'robust':
                    # Robust scaling using median and IQR
                    iqr = stats['q3'] - stats['q1']
                    if iqr == 0:
                        result[col] = 0.0
                    else:
                        result[col] = (result[col] - stats['median']) / iqr
        
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the data."""
        self._validate_input(data)
        
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        result = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                
                if self.scaling_method == 'standard':
                    std = stats['std']
                    if std == 0:
                        std = 1.0
                    result[col] = data[col] * std + stats['mean']
                elif self.scaling_method == 'minmax':
                    if stats['max'] == stats['min']:
                        result[col] = stats['min']
                    else:
                        result[col] = data[col] * (stats['max'] - stats['min']) + stats['min']
                elif self.scaling_method == 'robust':
                    iqr = stats['q3'] - stats['q1']
                    if iqr == 0:
                        result[col] = stats['median']
                    else:
                        result[col] = data[col] * iqr + stats['median']
        
        return result

    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters."""
        return self.config.copy()

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get feature statistics."""
        return self._feature_stats.copy()

def remove_outliers(df: pd.DataFrame, method: str = 'iqr', columns: Optional[list] = None) -> pd.DataFrame:
    """Remove outliers using IQR or Z-score method.
    
    Args:
        df: Input DataFrame
        method: 'iqr' or 'zscore'
        columns: List of columns to process (default: all numeric)
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    logger.info(f"Removing outliers using {method} method for columns: {columns}")
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        else:  # zscore
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores < 3
        
        outliers = (~mask).sum()
        if outliers > 0:
            logger.info(f"Removed {outliers} outliers from {col}")
            df = df[mask]
    
    return df

def apply_agent_transformations(df: pd.DataFrame, agent_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Apply custom transformations defined by agents.
    
    Args:
        df: Input DataFrame
        agent_config: Optional configuration dict for agent-specific transformations
        
    Returns:
        Transformed DataFrame
    """
    df = df.copy()
    logger.info("Applying agent transformations")
    
    if agent_config is None:
        return df
        
    # Example agent transformations (extend as needed)
    if 'custom_scaling' in agent_config:
        scale_factor = agent_config['custom_scaling']
        logger.info(f"Applying custom scaling factor: {scale_factor}")
        df = df * scale_factor
        
    if 'custom_filters' in agent_config:
        for col, condition in agent_config['custom_filters'].items():
            if col in df.columns:
                logger.info(f"Applying custom filter to {col}")
                mask = _parse_safe_condition(df[col], condition)
                df = df[mask]
    
    return df

def _parse_safe_condition(series: pd.Series, condition: str) -> pd.Series:
    """Safely parse and apply a condition to a pandas Series.
    
    Args:
        series: Pandas Series to apply condition to
        condition: String condition (e.g., "> 100", "<= 50", "== 0")
        
    Returns:
        Boolean mask for the condition
        
    Raises:
        ValueError: If condition format is invalid or unsafe
    """
    import re
    
    # Strip whitespace and normalize
    condition = condition.strip()
    
    # Define safe operators and their corresponding pandas operations
    safe_operators = {
        '>': 'gt',
        '>=': 'ge', 
        '<': 'lt',
        '<=': 'le',
        '==': 'eq',
        '!=': 'ne',
        'in': 'isin',
        'not in': 'not_na'  # Simplified for not in
    }
    
    # Find the operator in the condition
    operator_found = None
    operator_value = None
    
    for op, pandas_op in safe_operators.items():
        if op in condition:
            # Split on the operator
            parts = condition.split(op, 1)
            if len(parts) == 2:
                operator_found = pandas_op
                operator_value = parts[1].strip()
                break
    
    if operator_found is None:
        raise ValueError(f"Unsafe or invalid condition: {condition}")
    
    try:
        # Parse the value safely
        if operator_found in ['isin']:
            # Handle 'in' operator - expect a list-like structure
            if operator_value.startswith('[') and operator_value.endswith(']'):
                # Parse as literal list
                import ast
                value_list = ast.literal_eval(operator_value)
                return series.isin(value_list)
            else:
                raise ValueError(f"Invalid 'in' condition format: {condition}")
        elif operator_found == 'not_na':
            # Handle 'not in' - simplified as not null
            return series.notna()
        else:
            # Handle numeric comparisons
            if operator_value.lower() in ['true', 'false']:
                value = operator_value.lower() == 'true'
            elif operator_value.lower() == 'null' or operator_value.lower() == 'none':
                value = None
            else:
                # Try to parse as number
                try:
                    value = float(operator_value)
                    # If it's an integer, convert back
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    # Try as string
                    value = operator_value.strip('"\'')
            
            # Apply the condition using pandas methods
            if operator_found == 'gt':
                return series > value
            elif operator_found == 'ge':
                return series >= value
            elif operator_found == 'lt':
                return series < value
            elif operator_found == 'le':
                return series <= value
            elif operator_found == 'eq':
                return series == value
            elif operator_found == 'ne':
                return series != value
            else:
                raise ValueError(f"Unsupported operator: {operator_found}")
                
    except Exception as e:
        raise ValueError(f"Error parsing condition '{condition}': {str(e)}")

def preprocess_data(
    df: pd.DataFrame,
    remove_outliers_method: Optional[str] = 'iqr',
    agent_config: Optional[Dict[str, Any]] = None,
    transformations: Optional[list[Callable]] = None
) -> pd.DataFrame:
    """Main preprocessing function with logging and agent support.
    
    Args:
        df: Input DataFrame
        remove_outliers_method: Method for outlier removal ('iqr' or 'zscore')
        agent_config: Optional configuration for agent transformations
        transformations: Optional list of custom transformation functions
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting data preprocessing")
    df = df.copy()
    
    # Log initial state
    logger.info(f"Input shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove outliers if requested
    if remove_outliers_method:
        df = remove_outliers(df, method=remove_outliers_method)
        logger.info(f"Shape after outlier removal: {df.shape}")
    
    # Apply standard transformations
    if transformations:
        for transform in transformations:
            logger.info(f"Applying transformation: {transform.__name__}")
            df = transform(df)
    
    # Apply agent transformations
    df = apply_agent_transformations(df, agent_config)
    
    # Log final state
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Final missing values:\n{df.isnull().sum()}")
    
    return df