"""
Forecast Formatter

Formats and normalizes forecast data to prevent downstream failures.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ForecastFormatter:
    """Formats forecast data with proper normalization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the forecast formatter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.normalize_timezone = self.config.get("normalize_timezone", True)
        self.remove_timezone = self.config.get("remove_timezone", True)
        self.sort_index = self.config.get("sort_index", True)
        
    def normalize_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize datetime index to prevent downstream failures.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with normalized index
        """
        if df is None or df.empty:
            return df
            
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, attempting conversion")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Failed to convert index to datetime: {e}")
                return df
                
        # Remove timezone info if requested
        if self.remove_timezone and df.index.tz is not None:
            logger.info("Removing timezone information from index")
            df.index = df.index.tz_localize(None)
            
        # Normalize timezone if requested
        elif self.normalize_timezone and df.index.tz is not None:
            logger.info("Normalizing timezone to UTC")
            df.index = df.index.tz_convert('UTC')
            
        # Sort index if requested
        if self.sort_index:
            df = df.sort_index()
            
        logger.info(f"Normalized datetime index: {df.index.dtype}")
        return df
        
    def format_forecast_data(
        self, 
        forecast_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Format forecast data with proper normalization.
        
        Args:
            forecast_data: Raw forecast data
            
        Returns:
            Formatted DataFrame
        """
        if isinstance(forecast_data, np.ndarray):
            # Convert numpy array to DataFrame
            df = pd.DataFrame(forecast_data)
            logger.info("Converted numpy array to DataFrame")
        elif isinstance(forecast_data, dict):
            # Extract forecast from dictionary
            if "forecast" in forecast_data:
                forecast = forecast_data["forecast"]
                if isinstance(forecast, pd.DataFrame):
                    df = forecast
                elif isinstance(forecast, np.ndarray):
                    df = pd.DataFrame(forecast)
                else:
                    df = pd.DataFrame([forecast])
            else:
                logger.warning("No 'forecast' key found in dictionary")
                df = pd.DataFrame(forecast_data)
        elif isinstance(forecast_data, pd.DataFrame):
            df = forecast_data
        else:
            logger.error(f"Unsupported forecast data type: {type(forecast_data)}")
            return pd.DataFrame()
            
        # Normalize datetime index
        df = self.normalize_datetime_index(df)
        
        # Ensure numeric data
        df = self._ensure_numeric_data(df)
        
        return df
        
    def _ensure_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame contains numeric data.
        
        Args:
            df: DataFrame to check
            
        Returns:
            DataFrame with numeric data
        """
        # Convert non-numeric columns to numeric where possible
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {e}")
                    
        return df
        
    def format_confidence_intervals(
        self, 
        forecast: pd.DataFrame, 
        lower: Union[pd.Series, np.ndarray], 
        upper: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Format confidence intervals with proper normalization.
        
        Args:
            forecast: Forecast DataFrame
            lower: Lower confidence bounds
            upper: Upper confidence bounds
            
        Returns:
            DataFrame with confidence intervals
        """
        # Ensure all inputs are DataFrames
        if not isinstance(forecast, pd.DataFrame):
            forecast = pd.DataFrame(forecast)
        if not isinstance(lower, pd.Series):
            lower = pd.Series(lower)
        if not isinstance(upper, pd.Series):
            upper = pd.Series(upper)
            
        # Normalize datetime indexes
        forecast = self.normalize_datetime_index(forecast)
        lower.index = forecast.index
        upper.index = forecast.index
        
        # Create confidence interval DataFrame
        ci_df = pd.DataFrame({
            'forecast': forecast.iloc[:, 0] if len(forecast.columns) > 0 else forecast.iloc[:, 0],
            'lower': lower,
            'upper': upper
        })
        
        return ci_df
        
    def validate_forecast_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate forecast DataFrame format.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation_result["valid"] = False
            validation_result["errors"].append("DataFrame is empty")
            
        # Check for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            validation_result["warnings"].append("Index is not DatetimeIndex")
            
        # Check for timezone info
        if df.index.tz is not None:
            validation_result["warnings"].append("Index has timezone information")
            
        # Check for numeric data
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
                
        if non_numeric_cols:
            validation_result["warnings"].append(f"Non-numeric columns: {non_numeric_cols}")
            
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            validation_result["warnings"].append(f"Found {nan_count} NaN values")
            
        return validation_result 