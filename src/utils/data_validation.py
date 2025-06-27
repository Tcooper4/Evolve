"""
Data validation and preprocessing utilities for market data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating and preprocessing market data."""
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results = {}
        self.validation_timestamp = None
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate if the dataframe has the required structure and data quality.
        
        Args:
            df: Input dataframe with market data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        self.validation_timestamp = datetime.now()
        self.validation_results = {
            "timestamp": self.validation_timestamp.isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "summary": {}
        }
        
        logger.info(f"🔍 Starting data validation for DataFrame with shape {df.shape}")
        
        # Check if dataframe is empty
        if df.empty:
            self._add_error("DataFrame is empty")
            return False, "DataFrame is empty"
        
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            self._add_error(f"Missing required columns: {missing_cols}")
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for NaN values
        nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        total_nans = nan_counts.sum()
        self.validation_results["summary"]["total_nans"] = total_nans
        self.validation_results["summary"]["nan_counts"] = nan_counts.to_dict()
        
        if total_nans > 0:
            self._add_warning(f"Found {total_nans} NaN values across required columns")
            logger.warning(f"⚠️ NaN counts: {nan_counts.to_dict()}")
        
        # Validate price relationships
        if not self._validate_price_relationships(df):
            self._add_error("Invalid price relationships detected")
            return False, "Invalid price relationships detected"
        
        # Validate data types
        if not self._validate_data_types(df):
            self._add_error("Invalid data types detected")
            return False, "Invalid data types detected"
        
        # Additional quality checks
        self._validate_data_quality(df)
        
        # Generate summary
        self._generate_validation_summary(df)
        
        is_valid = len(self.validation_results["errors"]) == 0
        error_message = "; ".join(self.validation_results["errors"]) if not is_valid else "Data validation successful"
        
        if is_valid:
            logger.info("✅ Data validation completed successfully")
        else:
            logger.error(f"❌ Data validation failed: {error_message}")
        
        return is_valid, error_message
    
    def _validate_price_relationships(self, df: pd.DataFrame) -> bool:
        """Validate that high >= open/close >= low."""
        try:
            # Check high >= open
            high_open_valid = (df['high'] >= df['open']).all()
            self.validation_results["checks"]["high_open_valid"] = high_open_valid
            
            # Check high >= close
            high_close_valid = (df['high'] >= df['close']).all()
            self.validation_results["checks"]["high_close_valid"] = high_close_valid
            
            # Check low <= open
            low_open_valid = (df['low'] <= df['open']).all()
            self.validation_results["checks"]["low_open_valid"] = low_open_valid
            
            # Check low <= close
            low_close_valid = (df['low'] <= df['close']).all()
            self.validation_results["checks"]["low_close_valid"] = low_close_valid
            
            return all([high_open_valid, high_close_valid, low_open_valid, low_close_valid])
            
        except Exception as e:
            logger.error(f"Error validating price relationships: {str(e)}")
            return False
    
    def _validate_data_types(self, df: pd.DataFrame) -> bool:
        """Validate that all required columns are numeric."""
        try:
            for col in self.REQUIRED_COLUMNS:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                self.validation_results["checks"][f"{col}_numeric"] = is_numeric
                
                if not is_numeric:
                    logger.warning(f"⚠️ Column {col} is not numeric: {df[col].dtype}")
            
            return all(self.validation_results["checks"].values())
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            return False
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Perform additional data quality checks."""
        try:
            # Check for zero or negative prices
            zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any().any()
            self.validation_results["checks"]["no_zero_prices"] = not zero_prices
            if zero_prices:
                self._add_warning("Found zero or negative prices")
            
            # Check for zero volume
            zero_volume = (df['volume'] <= 0).any()
            self.validation_results["checks"]["no_zero_volume"] = not zero_volume
            if zero_volume:
                self._add_warning("Found zero or negative volume")
            
            # Check for extreme price movements
            price_changes = df['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.5).sum()  # >50% moves
            self.validation_results["summary"]["extreme_moves"] = extreme_moves
            if extreme_moves > 0:
                self._add_warning(f"Found {extreme_moves} extreme price movements (>50%)")
            
            # Check for duplicate timestamps
            if isinstance(df.index, pd.DatetimeIndex):
                duplicates = df.index.duplicated().sum()
                self.validation_results["summary"]["duplicate_timestamps"] = duplicates
                if duplicates > 0:
                    self._add_warning(f"Found {duplicates} duplicate timestamps")
            
            # Check data range
            date_range = None
            if isinstance(df.index, pd.DatetimeIndex):
                date_range = (df.index.min(), df.index.max())
                self.validation_results["summary"]["date_range"] = {
                    "start": date_range[0].isoformat(),
                    "end": date_range[1].isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {str(e)}")
    
    def _add_error(self, message: str) -> None:
        """Add an error message to validation results."""
        self.validation_results["errors"].append(message)
        logger.error(f"❌ Validation error: {message}")
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message to validation results."""
        self.validation_results["warnings"].append(message)
        logger.warning(f"⚠️ Validation warning: {message}")
    
    def _generate_validation_summary(self, df: pd.DataFrame) -> None:
        """Generate validation summary statistics."""
        self.validation_results["summary"].update({
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "validation_timestamp": self.validation_timestamp.isoformat()
        })
    
    def get_validation_results(self) -> Dict[str, Any]:
        """
        Get detailed validation results.
        
        Returns:
            Dictionary with validation results and statistics
        """
        return self.validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results for UI display.
        
        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {"status": "No validation performed"}
        
        return {
            "status": "failed" if self.validation_results["errors"] else "passed",
            "timestamp": self.validation_results["timestamp"],
            "total_checks": len(self.validation_results["checks"]),
            "passed_checks": sum(self.validation_results["checks"].values()),
            "warnings": len(self.validation_results["warnings"]),
            "errors": len(self.validation_results["errors"]),
            "summary": self.validation_results["summary"]
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the market data for analysis.
        
        Args:
            df: Input dataframe with market data
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("⚙️ Starting data preprocessing")
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Sort by index if it's datetime
        if isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed = df_processed.sort_index()
            logger.info("📅 Sorted data by datetime index")
        
        # Calculate returns
        if 'close' in df_processed.columns:
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['log_returns'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
            logger.info("📊 Calculated returns and log returns")
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        if 'returns' in df_processed.columns:
            df_processed['volatility'] = df_processed['returns'].rolling(window=20).std()
            logger.info("📈 Calculated volatility")
        
        # Calculate price ranges
        if all(col in df_processed.columns for col in ['high', 'low', 'close']):
            df_processed['daily_range'] = df_processed['high'] - df_processed['low']
            df_processed['daily_range_pct'] = df_processed['daily_range'] / df_processed['close']
            logger.info("📊 Calculated daily price ranges")
        
        # Calculate volume changes
        if 'volume' in df_processed.columns:
            df_processed['volume_change'] = df_processed['volume'].pct_change()
            logger.info("📊 Calculated volume changes")
        
        # Remove the first row which will have NaN values
        df_processed = df_processed.iloc[1:]
        
        logger.info(f"✅ Data preprocessing completed. Final shape: {df_processed.shape}")
        return df_processed
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing data in the dataframe.
        
        Args:
            df: Input dataframe with market data
            method: Method to handle missing data ('ffill', 'bfill', 'interpolate')
            
        Returns:
            DataFrame with missing data handled
        """
        logger.info(f"🔧 Handling missing data with method: {method}")
        
        df_cleaned = df.copy()
        
        if method == 'ffill':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif method == 'bfill':
            df_cleaned = df_cleaned.fillna(method='bfill')
        elif method == 'interpolate':
            df_cleaned = df_cleaned.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fill any remaining NaNs with forward fill
        df_cleaned = df_cleaned.fillna(method='ffill')
        
        logger.info("✅ Missing data handling completed")
        return df_cleaned
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], n_std: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns using standard deviation method.
        
        Args:
            df: Input dataframe with market data
            columns: List of columns to check for outliers
            n_std: Number of standard deviations to use as threshold
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"🔧 Removing outliers from {columns} with {n_std} std threshold")
        
        df_cleaned = df.copy()
        outliers_removed = 0
        
        for col in columns:
            if col not in df_cleaned.columns:
                continue
                
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            threshold = n_std * std
            
            # Count outliers before removal
            col_outliers = (abs(df_cleaned[col] - mean) > threshold).sum()
            outliers_removed += col_outliers
            
            # Replace outliers with NaN
            df_cleaned.loc[abs(df_cleaned[col] - mean) > threshold, col] = np.nan
        
        logger.info(f"✅ Removed {outliers_removed} outliers")
        return df_cleaned
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str], method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize specified columns in the dataframe.
        
        Args:
            df: Input dataframe with market data
            columns: List of columns to normalize
            method: Normalization method ('zscore' or 'minmax')
            
        Returns:
            DataFrame with normalized columns
        """
        logger.info(f"🔧 Normalizing {columns} with method: {method}")
        
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
        
        logger.info("✅ Data normalization completed")
        return df_normalized


def validate_data_for_training(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation for training data.
    
    Args:
        df: Input dataframe with market data
        
    Returns:
        Tuple of (is_valid, validation_summary)
    """
    validator = DataValidator()
    is_valid, error_message = validator.validate_dataframe(df)
    validation_summary = validator.get_validation_summary()
    
    return is_valid, validation_summary


def validate_data_for_forecasting(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation for forecasting data.
    
    Args:
        df: Input dataframe with market data
        
    Returns:
        Tuple of (is_valid, validation_summary)
    """
    validator = DataValidator()
    is_valid, error_message = validator.validate_dataframe(df)
    validation_summary = validator.get_validation_summary()
    
    # Additional checks for forecasting
    if is_valid:
        # Check if we have enough recent data
        if isinstance(df.index, pd.DatetimeIndex):
            days_of_data = (df.index.max() - df.index.min()).days
            if days_of_data < 30:
                validation_summary["warnings"].append("Limited historical data for forecasting")
        
        # Check for recent data gaps
        if isinstance(df.index, pd.DatetimeIndex):
            recent_data = df.tail(10)
            if recent_data.isna().any().any():
                validation_summary["warnings"].append("Recent data contains missing values")
    
    return is_valid, validation_summary 