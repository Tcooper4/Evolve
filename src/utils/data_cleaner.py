"""
Data Cleaner

Cleans and preprocesses data with selective NaN handling and context-aware operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    critical_columns: List[str]
    optional_columns: List[str]
    fill_method: str  # 'ffill', 'bfill', 'interpolate', 'mean', 'median', 'drop'
    max_missing_percentage: float
    remove_outliers: bool
    outlier_threshold: float
    normalize_columns: List[str]
    log_cleaning_actions: bool


class DataCleaner:
    """Cleans and preprocesses data with selective NaN handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cleaning_config = self._initialize_cleaning_config()
        self.logger = logging.getLogger(__name__)
        self.cleaning_history = []
        
    def _initialize_cleaning_config(self) -> CleaningConfig:
        """Initialize cleaning configuration."""
        return CleaningConfig(
            critical_columns=self.config.get("critical_columns", ["Close", "target"]),
            optional_columns=self.config.get("optional_columns", ["Volume", "Open", "High", "Low"]),
            fill_method=self.config.get("fill_method", "ffill"),
            max_missing_percentage=self.config.get("max_missing_percentage", 30.0),
            remove_outliers=self.config.get("remove_outliers", False),
            outlier_threshold=self.config.get("outlier_threshold", 3.0),
            normalize_columns=self.config.get("normalize_columns", []),
            log_cleaning_actions=self.config.get("log_cleaning_actions", True)
        )
        
    def clean_data(
        self, 
        df: pd.DataFrame,
        context: Optional[str] = None,
        critical_columns: Optional[List[str]] = None,
        fill_method: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data with selective NaN handling.
        
        Args:
            df: Input DataFrame
            context: Context for cleaning (e.g., 'training', 'prediction', 'backtest')
            critical_columns: Columns that must not have NaNs
            fill_method: Method to fill NaNs ('ffill', 'bfill', 'interpolate', 'mean', 'median', 'drop')
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning metadata)
        """
        if df.empty:
            self.logger.warning("Input DataFrame is empty")
            return df, {"cleaning_applied": False, "reason": "empty_dataframe"}
            
        original_shape = df.shape
        cleaning_metadata = {
            "original_shape": original_shape,
            "context": context,
            "cleaning_steps": [],
            "rows_removed": 0,
            "columns_cleaned": [],
            "nan_counts_before": df.isna().sum().to_dict(),
            "nan_counts_after": {}
        }
        
        cleaned_df = df.copy()
        
        # Use provided critical columns or default
        if critical_columns is None:
            critical_columns = self.cleaning_config.critical_columns
            
        # Use provided fill method or default
        if fill_method is None:
            fill_method = self.cleaning_config.fill_method
            
        # Step 1: Handle critical columns with selective dropna
        cleaned_df, critical_metadata = self._handle_critical_columns(
            cleaned_df, critical_columns, context
        )
        cleaning_metadata["cleaning_steps"].append("critical_columns_handling")
        cleaning_metadata["rows_removed"] += critical_metadata["rows_removed"]
        
        # Step 2: Handle optional columns with fill methods
        cleaned_df, optional_metadata = self._handle_optional_columns(
            cleaned_df, fill_method
        )
        cleaning_metadata["cleaning_steps"].append("optional_columns_handling")
        cleaning_metadata["columns_cleaned"].extend(optional_metadata["columns_cleaned"])
        
        # Step 3: Remove outliers if configured
        if self.cleaning_config.remove_outliers:
            cleaned_df, outlier_metadata = self._remove_outliers(cleaned_df)
            cleaning_metadata["cleaning_steps"].append("outlier_removal")
            cleaning_metadata["rows_removed"] += outlier_metadata["rows_removed"]
            
        # Step 4: Normalize columns if configured
        if self.cleaning_config.normalize_columns:
            cleaned_df, normalize_metadata = self._normalize_columns(cleaned_df)
            cleaning_metadata["cleaning_steps"].append("normalization")
            
        # Update final metadata
        cleaning_metadata.update({
            "final_shape": cleaned_df.shape,
            "nan_counts_after": cleaned_df.isna().sum().to_dict(),
            "total_rows_removed": original_shape[0] - cleaned_df.shape[0],
            "total_columns_removed": original_shape[1] - cleaned_df.shape[1]
        })
        
        # Log cleaning actions if enabled
        if self.cleaning_config.log_cleaning_actions:
            self._log_cleaning_summary(cleaning_metadata)
            
        # Store in history
        self.cleaning_history.append(cleaning_metadata)
        
        return cleaned_df, cleaning_metadata
        
    def _handle_critical_columns(
        self, 
        df: pd.DataFrame, 
        critical_columns: List[str],
        context: Optional[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle critical columns with selective NaN removal."""
        metadata = {"rows_removed": 0, "columns_checked": []}
        
        # Find critical columns that exist in the DataFrame
        existing_critical = [col for col in critical_columns if col in df.columns]
        metadata["columns_checked"] = existing_critical
        
        if not existing_critical:
            self.logger.warning("No critical columns found in DataFrame")
            return df, metadata
            
        # Use selective dropna for critical columns
        original_len = len(df)
        cleaned_df = df.dropna(subset=existing_critical)
        rows_removed = original_len - len(cleaned_df)
        
        if rows_removed > 0:
            self.logger.info(f"Removed {rows_removed} rows due to NaN values in critical columns: {existing_critical}")
            metadata["rows_removed"] = rows_removed
            
        return cleaned_df, metadata
        
    def _handle_optional_columns(
        self, 
        df: pd.DataFrame, 
        fill_method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle optional columns with fill methods."""
        metadata = {"columns_cleaned": []}
        
        # Find optional columns that exist in the DataFrame
        existing_optional = [col for col in self.cleaning_config.optional_columns if col in df.columns]
        
        for column in existing_optional:
            if df[column].isna().sum() > 0:
                cleaned_series = self._fill_nan_values(df[column], fill_method)
                df[column] = cleaned_series
                metadata["columns_cleaned"].append(column)
                
        return df, metadata
        
    def _fill_nan_values(self, series: pd.Series, method: str) -> pd.Series:
        """Fill NaN values using specified method."""
        if method == "ffill":
            return series.fillna(method='ffill').fillna(method='bfill')
        elif method == "bfill":
            return series.fillna(method='bfill').fillna(method='ffill')
        elif method == "interpolate":
            return series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        elif method == "mean":
            return series.fillna(series.mean())
        elif method == "median":
            return series.fillna(series.median())
        elif method == "drop":
            # This would be handled in critical columns
            return series
        else:
            self.logger.warning(f"Unknown fill method: {method}, using ffill")
            return series.fillna(method='ffill').fillna(method='bfill')
            
    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove outliers from numeric columns."""
        metadata = {"rows_removed": 0, "columns_processed": []}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in df.columns and df[column].isna().sum() < len(df):
                # Calculate z-scores
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > self.cleaning_config.outlier_threshold
                
                if outliers.sum() > 0:
                    original_len = len(df)
                    df = df[~outliers]
                    rows_removed = original_len - len(df)
                    metadata["rows_removed"] += rows_removed
                    metadata["columns_processed"].append(column)
                    
                    self.logger.info(f"Removed {outliers.sum()} outliers from column '{column}'")
                    
        return df, metadata
        
    def _normalize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize specified columns."""
        metadata = {"columns_normalized": []}
        
        for column in self.cleaning_config.normalize_columns:
            if column in df.columns and df[column].dtype in ['float64', 'int64']:
                # Min-max normalization
                min_val = df[column].min()
                max_val = df[column].max()
                
                if max_val > min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
                    metadata["columns_normalized"].append(column)
                    
        return df, metadata
        
    def _log_cleaning_summary(self, metadata: Dict[str, Any]):
        """Log a summary of cleaning actions."""
        self.logger.info("Data cleaning summary:")
        self.logger.info(f"  - Original shape: {metadata['original_shape']}")
        self.logger.info(f"  - Final shape: {metadata['final_shape']}")
        self.logger.info(f"  - Rows removed: {metadata['total_rows_removed']}")
        self.logger.info(f"  - Columns removed: {metadata['total_columns_removed']}")
        self.logger.info(f"  - Cleaning steps: {metadata['cleaning_steps']}")
        
        if metadata['columns_cleaned']:
            self.logger.info(f"  - Columns cleaned: {metadata['columns_cleaned']}")
            
    def clean_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean data specifically for training context."""
        return self.clean_data(
            df, 
            context="training",
            critical_columns=["Close", "target"],
            fill_method="ffill"
        )
        
    def clean_for_prediction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean data specifically for prediction context."""
        return self.clean_data(
            df, 
            context="prediction",
            critical_columns=["Close"],
            fill_method="ffill"
        )
        
    def clean_for_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean data specifically for backtesting context."""
        return self.clean_data(
            df, 
            context="backtest",
            critical_columns=["Close", "Open", "High", "Low"],
            fill_method="interpolate"
        )
        
    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """Get cleaning history."""
        return self.cleaning_history.copy()
        
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        if not self.cleaning_history:
            return {"total_cleanings": 0}
            
        total_cleanings = len(self.cleaning_history)
        total_rows_removed = sum(h["total_rows_removed"] for h in self.cleaning_history)
        total_columns_removed = sum(h["total_columns_removed"] for h in self.cleaning_history)
        
        return {
            "total_cleanings": total_cleanings,
            "total_rows_removed": total_rows_removed,
            "total_columns_removed": total_columns_removed,
            "average_rows_removed": total_rows_removed / total_cleanings if total_cleanings > 0 else 0,
            "average_columns_removed": total_columns_removed / total_cleanings if total_cleanings > 0 else 0
        }
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update cleaning configuration."""
        self.config.update(new_config)
        self.cleaning_config = self._initialize_cleaning_config()
        self.logger.info(f"Updated cleaning config: {new_config}")
        
    def reset_history(self):
        """Reset cleaning history."""
        self.cleaning_history.clear()
        self.logger.info("Cleaning history reset") 