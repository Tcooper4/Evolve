"""
Safe JSON saving utilities to prevent accidental data loss.

This module provides utilities for safely saving JSON data with protection
against overwriting files with empty data that could wipe valid historical records.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_save(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2,
    default: Optional[callable] = None,
    backup_existing: bool = True,
    min_data_size: int = 1,
) -> Dict[str, Any]:
    """
    Safely save JSON data with protection against overwriting with empty data.

    Args:
        data: Data to save (will be converted to JSON)
        filepath: Path to save the JSON file
        indent: JSON indentation (default: 2)
        default: Function to handle non-serializable objects
        backup_existing: Whether to backup existing file before overwriting
        min_data_size: Minimum size of data to consider it non-empty (default: 1)

    Returns:
        Dictionary with operation result
    """
    try:
        filepath = Path(filepath)

        # Check if data is effectively empty
        if not data:
            logger.warning(f"Skipped saving empty data to prevent wiping valid history: {filepath}")
            return {
                "success": False,
                "error": "Empty data detected - skipping save to prevent data loss",
                "filepath": str(filepath),
            }

        # Additional check for empty dictionaries/lists
        if isinstance(data, (dict, list)) and len(data) < min_data_size:
            logger.warning(f"Data too small ({len(data)} items) - skipping save to prevent data loss: {filepath}")
            return {
                "success": False,
                "error": f"Data too small ({len(data)} items) - skipping save to prevent data loss",
                "filepath": str(filepath),
            }

        # Create backup if requested and file exists
        if backup_existing and filepath.exists():
            backup_path = filepath.with_suffix(f"{filepath.suffix}.backup")
            try:
                import shutil

                shutil.copy2(filepath, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save the data
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=default)

        logger.debug(f"Successfully saved JSON data to: {filepath}")
        return {
            "success": True,
            "filepath": str(filepath),
            "data_size": len(data) if hasattr(data, "__len__") else "unknown",
        }

    except Exception as e:
        logger.error(f"Failed to save JSON data to {filepath}: {e}")
        return {"success": False, "error": str(e), "filepath": str(filepath)}


def safe_json_save_with_validation(
    data: Any, filepath: Union[str, Path], validation_func: Optional[callable] = None, **kwargs
) -> Dict[str, Any]:
    """
    Save JSON data with additional validation.

    Args:
        data: Data to save
        filepath: Path to save the JSON file
        validation_func: Optional function to validate data before saving
        **kwargs: Additional arguments for safe_json_save

    Returns:
        Dictionary with operation result
    """
    try:
        # Run validation if provided
        if validation_func:
            validation_result = validation_func(data)
            if not validation_result.get("valid", True):
                logger.warning(f"Data validation failed: {validation_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f'Validation failed: {validation_result.get("error", "Unknown error")}',
                    "filepath": str(filepath),
                }

        return safe_json_save(data, filepath, **kwargs)

    except Exception as e:
        logger.error(f"Failed to save JSON data with validation to {filepath}: {e}")
        return {"success": False, "error": str(e), "filepath": str(filepath)}


def validate_historical_data(data: Any) -> Dict[str, Any]:
    """
    Validate that data contains meaningful historical information.

    Args:
        data: Data to validate

    Returns:
        Dictionary with validation result
    """
    try:
        if not data:
            return {"valid": False, "error": "Data is empty"}

        if isinstance(data, dict):
            # Check for common historical data patterns
            if "timestamp" in data or "history" in data or "metrics" in data:
                return {"valid": True}

            # Check if dict has meaningful content
            if len(data) == 0:
                return {"valid": False, "error": "Dictionary is empty"}

        elif isinstance(data, list):
            # Check if list has meaningful content
            if len(data) == 0:
                return {"valid": False, "error": "List is empty"}

            # Check first few items for structure
            for item in data[:3]:
                if isinstance(item, dict) and ("timestamp" in item or "date" in item):
                    return {"valid": True}

        return {"valid": True}

    except Exception as e:
        return {"valid": False, "error": f"Validation error: {e}"}


# Convenience function for historical data
def safe_save_historical_data(data: Any, filepath: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Safely save historical data with validation.

    Args:
        data: Historical data to save
        filepath: Path to save the JSON file
        **kwargs: Additional arguments for safe_json_save

    Returns:
        Dictionary with operation result
    """
    return safe_json_save_with_validation(data, filepath, validation_func=validate_historical_data, **kwargs)
