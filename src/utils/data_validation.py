"""
Data validation and preprocessing utilities for market data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class DataValidator:
    """Class for validating and preprocessing market data."""
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate if the dataframe has the required structure and data quality.
        
        Args:
            df: Input dataframe with market data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df.empty:
            return False, "DataFrame is empty"
            
        # Check required columns
        missing_cols = [col for col in DataValidator.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        # Check for NaN values
        if df[DataValidator.REQUIRED_COLUMNS].isna().any().any():
            return False, "DataFrame contains NaN values"
            
        # Validate price relationships
        if not DataValidator._validate_price_relationships(df):
            return False, "Invalid price relationships detected"
            
        # Validate data types
        if not DataValidator._validate_data_types(df):
            return False, "Invalid data types detected"
            
        return True, "Data validation successful"
    
    @staticmethod
    def _validate_price_relationships(df: pd.DataFrame) -> bool:
        """Validate that high >= open/close >= low."""
        return (
            (df['high'] >= df['open']).all() and
            (df['high'] >= df['close']).all() and
            (df['low'] <= df['open']).all() and
            (df['low'] <= df['close']).all()
        )
    
    @staticmethod
    def _validate_data_types(df: pd.DataFrame) -> bool:
        """Validate that all required columns are numeric."""
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in DataValidator.REQUIRED_COLUMNS)
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the market data for analysis.
        
        Args:
            df: Input dataframe with market data
            
        Returns:
            Preprocessed dataframe
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Sort by index if it's datetime
        if isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed = df_processed.sort_index()
        
        # Calculate returns
        df_processed['returns'] = df_processed['close'].pct_change()
        
        # Calculate log returns
        df_processed['log_returns'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        df_processed['volatility'] = df_processed['returns'].rolling(window=20).std()
        
        # Calculate price ranges
        df_processed['daily_range'] = df_processed['high'] - df_processed['low']
        df_processed['daily_range_pct'] = df_processed['daily_range'] / df_processed['close']
        
        # Calculate volume changes
        df_processed['volume_change'] = df_processed['volume'].pct_change()
        
        # Remove the first row which will have NaN values
        df_processed = df_processed.iloc[1:]
        
        return df_processed
    
    @staticmethod
    def handle_missing_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing data in the dataframe.
        
        Args:
            df: Input dataframe with market data
            method: Method to handle missing data ('ffill', 'bfill', 'interpolate')
            
        Returns:
            DataFrame with missing data handled
        """
        df_cleaned = df.copy()
        
        if method == 'ffill':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif method == 'bfill':
            df_cleaned = df_cleaned.fillna(method='bfill')
        elif method == 'interpolate':
            df_cleaned = df_cleaned.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return df_cleaned
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], n_std: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns using standard deviation method.
        
        Args:
            df: Input dataframe with market data
            columns: List of columns to check for outliers
            n_std: Number of standard deviations to use as threshold
            
        Returns:
            DataFrame with outliers removed
        """
        df_cleaned = df.copy()
        
        for col in columns:
            if col not in df_cleaned.columns:
                continue
                
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            threshold = n_std * std
            
            # Replace outliers with NaN
            df_cleaned.loc[abs(df_cleaned[col] - mean) > threshold, col] = np.nan
            
        return df_cleaned
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize specified columns in the dataframe.
        
        Args:
            df: Input dataframe with market data
            columns: List of columns to normalize
            method: Normalization method ('zscore' or 'minmax')
            
        Returns:
            DataFrame with normalized columns
        """
        df_normalized = df.copy()
        
        for col in columns:
            if col not in df_normalized.columns:
                continue
                
            if method == 'zscore':
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                df_normalized[col] = (df_normalized[col] - mean) / std
            elif method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
        return df_normalized 