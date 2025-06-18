"""Data utilities for validation and preprocessing.

This module provides utilities for validating and preprocessing financial data,
including data quality checks, feature engineering, and data transformation.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class DataValidator:
    """Validator for financial data quality."""
    
    def __init__(self, min_data_points: int = 100):
        """Initialize the validator.
        
        Args:
            min_data_points: Minimum number of data points required
        """
        self.min_data_points = min_data_points
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        check_missing: bool = True,
        check_duplicates: bool = True,
        check_outliers: bool = True
    ) -> Tuple[bool, List[str]]:
        """Validate a DataFrame for data quality.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            check_missing: Whether to check for missing values
            check_duplicates: Whether to check for duplicates
            check_outliers: Whether to check for outliers
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data points: {len(df)} < {self.min_data_points}")
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check missing values
        if check_missing:
            missing = df.isnull().sum()
            if missing.any():
                issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check duplicates
        if check_duplicates:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate rows")
        
        # Check outliers
        if check_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    issues.append(f"Found {outliers} outliers in column {col}")
        
        return len(issues) == 0, issues

class DataPreprocessor:
    """Preprocessor for financial data."""
    
    def __init__(
        self,
        scale_features: bool = True,
        handle_missing: bool = True,
        remove_outliers: bool = False
    ):
        """Initialize the preprocessor.
        
        Args:
            scale_features: Whether to scale features
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
        """
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Preprocess the data.
        
        Args:
            df: DataFrame to preprocess
            target_column: Name of target column
            feature_columns: List of feature column names
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        if self.handle_missing:
            df = self._handle_missing_values(df)
        
        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Scale features
        if self.scale_features:
            df = self._scale_features(df, target_column, feature_columns)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with handled missing values
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Handle categorical columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= 3]
        
        return df
    
    def _scale_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        feature_columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Scale features in the data.
        
        Args:
            df: DataFrame to process
            target_column: Name of target column
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with scaled features
        """
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column:
                feature_columns.remove(target_column)
        
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df

def resample_data(
    df: pd.DataFrame,
    freq: str,
    agg_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Resample time series data.
    
    Args:
        df: DataFrame to resample
        freq: Resampling frequency (e.g., '1D', '1H')
        agg_dict: Dictionary of column names and aggregation functions
        
    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    return df.resample(freq).agg(agg_dict)

def calculate_technical_indicators(
    df: pd.DataFrame,
    indicators: List[str],
    params: Optional[Dict[str, Dict[str, int]]] = None
) -> pd.DataFrame:
    """Calculate technical indicators.
    
    Args:
        df: DataFrame with price data
        indicators: List of indicators to calculate
        params: Dictionary of indicator parameters
        
    Returns:
        DataFrame with added indicators
    """
    if params is None:
        params = {
            'sma': {'window': 20},
            'ema': {'window': 20},
            'rsi': {'window': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }
    
    df = df.copy()
    
    for indicator in indicators:
        if indicator == 'sma':
            window = params['sma']['window']
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        elif indicator == 'ema':
            window = params['ema']['window']
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        elif indicator == 'rsi':
            window = params['rsi']['window']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        elif indicator == 'macd':
            fast = params['macd']['fast']
            slow = params['macd']['slow']
            signal = params['macd']['signal']
            
            exp1 = df['close'].ewm(span=fast).mean()
            exp2 = df['close'].ewm(span=slow).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame to split
        target_column: Name of target column
        test_size: Proportion of data for test set
        validation_size: Proportion of data for validation set
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))
    
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]
    
    return train_df, val_df, test_df 