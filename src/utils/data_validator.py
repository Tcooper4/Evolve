"""
Data Validator

Validates data quality and structure with support for multi-index DataFrames.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Validates data quality and structure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 100)
        self.max_missing_percentage = self.config.get("max_missing_percentage", 50.0)
        self.zero_std_threshold = self.config.get("zero_std_threshold", 1e-10)
        self.support_multi_index = self.config.get("support_multi_index", True)
        self.logger = logging.getLogger(__name__)
        
    def validate_dataframe(
        self, 
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        check_missing: bool = True,
        check_duplicates: bool = True,
        check_outliers: bool = True,
        check_zero_std: bool = True,
        handle_multi_index: bool = True
    ) -> ValidationResult:
        """
        Validate a DataFrame for data quality.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            check_missing: Whether to check for missing values
            check_duplicates: Whether to check for duplicates
            check_outliers: Whether to check for outliers
            check_zero_std: Whether to check for zero standard deviation columns
            handle_multi_index: Whether to handle multi-index DataFrames
            
        Returns:
            ValidationResult with validation results
        """
        issues = []
        warnings = []
        metadata = {
            "original_shape": df.shape,
            "original_index_type": str(type(df.index)),
            "multi_index_detected": False,
            "index_reset_applied": False
        }
        
        # Handle multi-index if detected
        if handle_multi_index and self._is_multi_index(df):
            metadata["multi_index_detected"] = True
            if self.support_multi_index:
                warnings.append("Multi-index detected, using reset_index() fallback")
                df = df.reset_index()
                metadata["index_reset_applied"] = True
                metadata["new_shape"] = df.shape
            else:
                issues.append("Multi-index DataFrame not supported")
                return ValidationResult(
                    is_valid=False,
                    issues=issues,
                    warnings=warnings,
                    metadata=metadata
                )
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                metadata=metadata
            )
        
        # Check data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data points: {len(df)} < {self.min_data_points}")
            
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
                
        # Check for missing values
        if check_missing:
            missing_issues = self._check_missing_values(df)
            issues.extend(missing_issues["issues"])
            warnings.extend(missing_issues["warnings"])
            
        # Check for duplicates
        if check_duplicates:
            duplicate_issues = self._check_duplicates(df)
            issues.extend(duplicate_issues["issues"])
            warnings.extend(duplicate_issues["warnings"])
            
        # Check for outliers
        if check_outliers:
            outlier_issues = self._check_outliers(df)
            issues.extend(outlier_issues["issues"])
            warnings.extend(outlier_issues["warnings"])
            
        # Check for zero standard deviation columns
        if check_zero_std:
            zero_std_issues = self._check_zero_std_columns(df)
            issues.extend(zero_std_issues["issues"])
            warnings.extend(zero_std_issues["warnings"])
            
        # Check for 100% null columns
        null_issues = self._check_null_columns(df)
        issues.extend(null_issues["issues"])
        warnings.extend(null_issues["warnings"])
        
        # Update metadata
        metadata.update({
            "final_shape": df.shape,
            "total_issues": len(issues),
            "total_warnings": len(warnings),
            "columns_analyzed": list(df.columns),
            "data_types": df.dtypes.to_dict()
        })
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            metadata=metadata
        )
        
    def _is_multi_index(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has a multi-index."""
        return isinstance(df.index, pd.MultiIndex)
        
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for missing values in the DataFrame."""
        issues = []
        warnings = []
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            if missing_count == len(df):
                issues.append(f"Column '{column}' is 100% null")
            elif missing_percentage > self.max_missing_percentage:
                issues.append(f"Column '{column}' has {missing_percentage:.1f}% missing values (>{self.max_missing_percentage}%)")
            elif missing_count > 0:
                warnings.append(f"Column '{column}' has {missing_count} missing values ({missing_percentage:.1f}%)")
                
        return {"issues": issues, "warnings": warnings}
        
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for duplicate rows."""
        issues = []
        warnings = []
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 10:
                issues.append(f"High duplicate rate: {duplicate_percentage:.1f}% ({duplicate_count} rows)")
            else:
                warnings.append(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)")
                
        return {"issues": issues, "warnings": warnings}
        
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for outliers in numeric columns."""
        issues = []
        warnings = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].isna().all():
                continue
                
            # Remove NaN values for outlier detection
            clean_data = df[column].dropna()
            if len(clean_data) < 10:
                continue
                
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
                
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_percentage = (len(outliers) / len(clean_data)) * 100
            
            if outlier_percentage > 20:
                issues.append(f"Column '{column}' has {outlier_percentage:.1f}% outliers")
            elif outlier_percentage > 5:
                warnings.append(f"Column '{column}' has {outlier_percentage:.1f}% outliers")
                
        return {"issues": issues, "warnings": warnings}
        
    def _check_zero_std_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for columns with zero or near-zero standard deviation."""
        issues = []
        warnings = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].isna().all():
                continue
                
            std_value = df[column].std()
            if std_value <= self.zero_std_threshold:
                issues.append(f"Column '{column}' has zero standard deviation (std={std_value})")
            elif std_value < 0.01:
                warnings.append(f"Column '{column}' has very low standard deviation (std={std_value})")
                
        return {"issues": issues, "warnings": warnings}
        
    def _check_null_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for columns that are 100% null."""
        issues = []
        warnings = []
        
        for column in df.columns:
            null_count = df[column].isna().sum()
            if null_count == len(df):
                issues.append(f"Column '{column}' is 100% null")
            elif null_count == len(df) - 1:
                warnings.append(f"Column '{column}' has only one non-null value")
                
        return {"issues": issues, "warnings": warnings}
        
    def validate_ohlcv_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLCV data specifically."""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        
        result = self.validate_dataframe(
            df=df,
            required_columns=required_columns,
            check_missing=True,
            check_duplicates=True,
            check_outliers=True,
            check_zero_std=True,
            handle_multi_index=True
        )
        
        # Additional OHLCV-specific checks
        if result.is_valid:
            ohlcv_issues = self._check_ohlcv_consistency(df)
            result.issues.extend(ohlcv_issues["issues"])
            result.warnings.extend(ohlcv_issues["warnings"])
            result.is_valid = len(result.issues) == 0
            
        return result
        
    def _check_ohlcv_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check OHLCV data consistency."""
        issues = []
        warnings = []
        
        # Check High >= Low
        invalid_high_low = (df["High"] < df["Low"]).sum()
        if invalid_high_low > 0:
            issues.append(f"Found {invalid_high_low} rows where High < Low")
            
        # Check Open and Close are within High-Low range
        invalid_open = ((df["Open"] > df["High"]) | (df["Open"] < df["Low"])).sum()
        invalid_close = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
        
        if invalid_open > 0:
            issues.append(f"Found {invalid_open} rows where Open is outside High-Low range")
        if invalid_close > 0:
            issues.append(f"Found {invalid_close} rows where Close is outside High-Low range")
            
        # Check for negative prices
        negative_prices = (df[["Open", "High", "Low", "Close"]] < 0).any(axis=1).sum()
        if negative_prices > 0:
            issues.append(f"Found {negative_prices} rows with negative prices")
            
        # Check for negative volume
        negative_volume = (df["Volume"] < 0).sum()
        if negative_volume > 0:
            issues.append(f"Found {negative_volume} rows with negative volume")
            
        return {"issues": issues, "warnings": warnings}
        
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a formatted validation summary."""
        summary = f"Data Validation Summary:\n"
        summary += f"  - Valid: {result.is_valid}\n"
        summary += f"  - Issues: {len(result.issues)}\n"
        summary += f"  - Warnings: {len(result.warnings)}\n"
        
        if result.issues:
            summary += f"\nIssues:\n"
            for issue in result.issues:
                summary += f"  - {issue}\n"
                
        if result.warnings:
            summary += f"\nWarnings:\n"
            for warning in result.warnings:
                summary += f"  - {warning}\n"
                
        return summary
        
    def fix_common_issues(self, df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Attempt to fix common data issues."""
        fixed_df = df.copy()
        
        for issue in result.issues:
            if "100% null" in issue:
                # Remove 100% null columns
                column_name = issue.split("'")[1]
                if column_name in fixed_df.columns:
                    fixed_df = fixed_df.drop(columns=[column_name])
                    self.logger.info(f"Dropped 100% null column: {column_name}")
                    
            elif "zero standard deviation" in issue:
                # Remove zero std columns
                column_name = issue.split("'")[1]
                if column_name in fixed_df.columns:
                    fixed_df = fixed_df.drop(columns=[column_name])
                    self.logger.info(f"Dropped zero std column: {column_name}")
                    
        # Remove duplicate rows
        if any("duplicate" in issue for issue in result.issues):
            original_len = len(fixed_df)
            fixed_df = fixed_df.drop_duplicates()
            dropped_count = original_len - len(fixed_df)
            if dropped_count > 0:
                self.logger.info(f"Dropped {dropped_count} duplicate rows")
                
        return fixed_df 