"""
Data transformation utilities for the trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class DataTransformer:
    """Data transformation utility class."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize data transformer.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
        self.pca = None
        self.is_fitted = False
        
    def _create_scaler(self):
        """Create the appropriate scaler."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {self.scaler_type}, using StandardScaler")
            return StandardScaler()
    
    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit the transformer and transform the data."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit and transform
        transformed_data = data.copy()
        transformed_data[columns] = self.scaler.fit_transform(data[columns])
        self.is_fitted = True
        
        return transformed_data
    
    def transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        transformed_data = data.copy()
        transformed_data[columns] = self.scaler.transform(data[columns])
        
        return transformed_data
    
    def inverse_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Inverse transform data."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        inverse_data = data.copy()
        inverse_data[columns] = self.scaler.inverse_transform(data[columns])
        
        return inverse_data
    
    def apply_pca(self, data: pd.DataFrame, n_components: Optional[int] = None, 
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply PCA transformation."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if n_components is None:
            n_components = min(len(columns), len(data))
        
        self.pca = PCA(n_components=n_components)
        pca_data = self.pca.fit_transform(data[columns])
        
        # Create new DataFrame with PCA components
        pca_columns = [f'pca_{i}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            result = pd.concat([data[non_numeric_cols], pca_df], axis=1)
        else:
            result = pca_df
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from PCA."""
        if self.pca is None:
            return {}
        
        return {f'pca_{i}': importance for i, importance in enumerate(self.pca.explained_variance_ratio_)}

def resample_data(data: pd.DataFrame, timeframe: str, agg_method: str = 'ohlc') -> pd.DataFrame:
    """Resample time series data to different timeframe."""
    try:
        if agg_method == 'ohlc':
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif agg_method == 'mean':
            resampled = data.resample(timeframe).mean()
        else:
            resampled = data.resample(timeframe).agg(agg_method)
        
        return resampled.dropna()
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        return data

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators."""
    try:
        result = data.copy()
        
        # Simple Moving Averages
        result['sma_20'] = data['close'].rolling(window=20).mean()
        result['sma_50'] = data['close'].rolling(window=50).mean()
        result['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        result['ema_12'] = data['close'].ewm(span=12).mean()
        result['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Bollinger Bands
        result['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
        result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        result['volume_sma'] = data['volume'].rolling(window=20).mean()
        result['volume_ratio'] = data['volume'] / result['volume_sma']
        
        return result
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return data

def split_data(data: pd.DataFrame, train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    try:
        total_rows = len(data)
        train_end = int(total_rows * train_ratio)
        val_end = int(total_rows * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return data, data, data

def prepare_forecast_data(data: pd.DataFrame, target_column: str = 'close',
                         sequence_length: int = 60, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for time series forecasting."""
    try:
        # Create sequences
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[target_column].values[i:i + sequence_length])
            y.append(data[target_column].values[i + sequence_length:i + sequence_length + forecast_horizon])
        
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error preparing forecast data: {e}")
        return np.array([]), np.array([]) 