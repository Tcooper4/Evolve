"""
Example demonstrating safe JSON saving to prevent data loss.

This script shows how to use the safe_json_saver utility to protect against
accidentally overwriting important historical data with empty or invalid data.
"""

import sys
from pathlib import Path

from utils.safe_json_saver import safe_json_save, safe_save_historical_data

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))


def example_basic_safe_saving():
    """Example of basic safe JSON saving."""
    print("=== Basic Safe JSON Saving Example ===")

    # Example 1: Saving valid data
    valid_data = {
        "timestamp": "2024-01-15T10:30:00",
        "metrics": {"sharpe_ratio": 1.25, "max_drawdown": 0.15, "win_rate": 0.65},
        "history": [
            {"date": "2024-01-14", "return": 0.02},
            {"date": "2024-01-15", "return": 0.01},
        ],
    }

    result = safe_json_save(valid_data, "examples/valid_data.json")
    print(f"Valid data save result: {result}")

    # Example 2: Attempting to save empty data (will be prevented)
    empty_data = {}

    result = safe_json_save(empty_data, "examples/empty_data.json")
    print(f"Empty data save result: {result}")

    # Example 3: Attempting to save None data (will be prevented)
    result = safe_json_save(None, "examples/none_data.json")
    print(f"None data save result: {result}")


def example_historical_data_saving():
    """Example of saving historical data with validation."""
    print("\n=== Historical Data Saving Example ===")

    # Example 1: Valid historical data
    historical_data = {
        "performance_history": [
            {
                "timestamp": "2024-01-15T10:30:00",
                "metric_name": "sharpe_ratio",
                "value": 1.25,
                "context": {"market_regime": "bull"},
                "metadata": {"strategy": "momentum"},
            },
            {
                "timestamp": "2024-01-15T11:30:00",
                "metric_name": "sharpe_ratio",
                "value": 1.30,
                "context": {"market_regime": "bull"},
                "metadata": {"strategy": "momentum"},
            },
        ],
        "trends": {
            "sharpe_ratio": {
                "trend_direction": "improving",
                "trend_strength": 0.8,
                "period_days": 30,
            }
        },
    }

    result = safe_save_historical_data(
        historical_data, "examples/performance_history.json"
    )
    print(f"Historical data save result: {result}")

    # Example 2: Invalid historical data (empty list)
    invalid_data = {"performance_history": [], "trends": {}}

    result = safe_save_historical_data(invalid_data, "examples/invalid_history.json")
    print(f"Invalid historical data save result: {result}")


def example_custom_validation():
    """Example of using custom validation functions."""
    print("\n=== Custom Validation Example ===")

    def validate_trading_data(data):
        """Custom validation for trading data."""
        if not data:
            return {"valid": False, "error": "Data is empty"}

        if isinstance(data, dict):
            # Check for required trading fields
            required_fields = ["symbol", "price", "volume"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}",
                }

            # Check for reasonable values
            if data.get("price", 0) <= 0:
                return {"valid": False, "error": "Price must be positive"}

            if data.get("volume", 0) < 0:
                return {"valid": False, "error": "Volume cannot be negative"}

        return {"valid": True}

    # Valid trading data
    valid_trading_data = {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000,
        "timestamp": "2024-01-15T10:30:00",
    }

    from utils.safe_json_saver import safe_json_save_with_validation

    result = safe_json_save_with_validation(
        valid_trading_data,
        "examples/trading_data.json",
        validation_func=validate_trading_data,
    )
    print(f"Valid trading data save result: {result}")

    # Invalid trading data (missing required fields)
    invalid_trading_data = {
        "symbol": "AAPL",
        "price": 150.25,
        # Missing volume field
    }

    result = safe_json_save_with_validation(
        invalid_trading_data,
        "examples/invalid_trading_data.json",
        validation_func=validate_trading_data,
    )
    print(f"Invalid trading data save result: {result}")


def example_backup_protection():
    """Example showing backup protection feature."""
    print("\n=== Backup Protection Example ===")

    # Create initial data
    initial_data = {"version": 1, "data": "important information"}
    result = safe_json_save(
        initial_data, "examples/backup_test.json", backup_existing=True
    )
    print(f"Initial save result: {result}")

    # Update data (this will create a backup of the previous version)
    updated_data = {"version": 2, "data": "updated information"}
    result = safe_json_save(
        updated_data, "examples/backup_test.json", backup_existing=True
    )
    print(f"Updated save result: {result}")

    # Check if backup was created
    backup_file = Path("examples/backup_test.json.backup")
    if backup_file.exists():
        print(f"✓ Backup file created: {backup_file}")
        with open(backup_file, "r") as f:
            import json

            backup_content = json.load(f)
            print(f"  Backup contains: {backup_content}")
    else:
        print("✗ No backup file found")


def example_minimum_data_size():
    """Example showing minimum data size protection."""
    print("\n=== Minimum Data Size Example ===")

    # Data that's too small (less than minimum size)
    small_data = {"metric": "test"}

    result = safe_json_save(small_data, "examples/small_data.json", min_data_size=5)
    print(f"Small data save result: {result}")

    # Data that meets minimum size requirement
    adequate_data = {
        "metric1": "value1",
        "metric2": "value2",
        "metric3": "value3",
        "metric4": "value4",
        "metric5": "value5",
    }

    result = safe_json_save(
        adequate_data, "examples/adequate_data.json", min_data_size=5
    )
    print(f"Adequate data save result: {result}")


if __name__ == "__main__":
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)

    print("Safe JSON Saving Examples")
    print("=" * 50)

    example_basic_safe_saving()
    example_historical_data_saving()
    example_custom_validation()
    example_backup_protection()
    example_minimum_data_size()

    print("\n" + "=" * 50)
    print("Examples completed! Check the 'examples/' directory for generated files.")
    print("\nKey benefits of safe JSON saving:")
    print("1. Prevents accidental overwriting with empty data")
    print("2. Validates data before saving")
    print("3. Creates automatic backups")
    print("4. Configurable minimum data size requirements")
    print("5. Detailed error reporting and logging")
