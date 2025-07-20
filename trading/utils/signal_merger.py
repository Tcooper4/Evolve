"""
Signal Merger

Merges signal DataFrames with index validation to ensure compatibility.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SignalMerger:
    """Merges signal DataFrames with validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the signal merger.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.merge_method = self.config.get("merge_method", "outer")
        self.validate_indexes = self.config.get("validate_indexes", True)

    def merge_signals(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Merge two signal DataFrames with index validation and UTC localization.
        """
        if df1 is None or df2 is None:
            raise ValueError("Cannot merge None DataFrames")
        if df1.empty and df2.empty:
            logger.warning("Both DataFrames are empty")
            return pd.DataFrame()
        # Ensure DateTimeIndex is timezone-aware and set to UTC
        for df in [df1, df2]:
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    try:
                        df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                    except Exception:
                        df.index = df.index.tz_localize('UTC', errors='coerce')
                else:
                    df.index = df.index.tz_convert('UTC')
        # Validate indexes if requested
        if validate and self.validate_indexes:
            if not df1.index.equals(df2.index):
                logger.error("Indexes do not match between DataFrames")
                logger.error(f"df1 index: {df1.index[:5]}...")
                logger.error(f"df2 index: {df2.index[:5]}...")
                raise ValueError("Indexes must match for signal merging")
        # Perform merge
        try:
            merged_df = pd.merge(
                df1,
                df2,
                left_index=True,
                right_index=True,
                how=self.merge_method
            )
            logger.info(f"Successfully merged signals: {df1.shape} + {df2.shape} -> {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Error merging signals: {e}")
            raise

    def merge_multiple_signals(
        self,
        dataframes: List[pd.DataFrame],
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Merge multiple signal DataFrames with UTC localization.
        """
        if not dataframes:
            logger.warning("No DataFrames provided for merging")
            return pd.DataFrame()
        if len(dataframes) == 1:
            return dataframes[0]
        # Ensure all indexes are UTC
        for i, df in enumerate(dataframes):
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    try:
                        df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                    except Exception:
                        df.index = df.index.tz_localize('UTC', errors='coerce')
                else:
                    df.index = df.index.tz_convert('UTC')
        # Validate all indexes match if requested
        if validate and self.validate_indexes:
            reference_index = dataframes[0].index
            for i, df in enumerate(dataframes[1:], 1):
                if not df.index.equals(reference_index):
                    logger.error(f"Index mismatch between DataFrame 0 and DataFrame {i}")
                    raise ValueError(f"All DataFrames must have matching indexes")
        # Merge all DataFrames
        try:
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = self.merge_signals(merged_df, df, validate=False)
            logger.info(f"Successfully merged {len(dataframes)} DataFrames: {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Error merging multiple signals: {e}")
            raise

    def validate_merge_compatibility(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate compatibility for merging two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "compatible": True,
            "indexes_match": False,
            "common_columns": [],
            "warnings": []
        }

        # Check index compatibility
        if df1.index.equals(df2.index):
            validation_result["indexes_match"] = True
        else:
            validation_result["compatible"] = False
            validation_result["warnings"].append("Indexes do not match")

        # Check for common columns
        common_cols = set(df1.columns) & set(df2.columns)
        validation_result["common_columns"] = list(common_cols)

        if common_cols:
            validation_result["warnings"].append(f"Common columns found: {common_cols}")

        return validation_result

    def get_merge_summary(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get summary of merge operation.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Dictionary with merge summary
        """
        summary = {
            "df1_shape": df1.shape if df1 is not None else (0, 0),
            "df2_shape": df2.shape if df2 is not None else (0, 0),
            "df1_columns": list(df1.columns) if df1 is not None else [],
            "df2_columns": list(df2.columns) if df2 is not None else [],
            "common_columns": [],
            "index_overlap": 0
        }

        if df1 is not None and df2 is not None:
            summary["common_columns"] = list(set(df1.columns) & set(df2.columns))
            summary["index_overlap"] = len(df1.index.intersection(df2.index))

        return summary
