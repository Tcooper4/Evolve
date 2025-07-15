import pytest
import pandas as pd
import numpy as np
from src.utils.data_validator import DataValidator, ValidationResult

def test_empty_dataframe():
    validator = DataValidator()
    df = pd.DataFrame()
    result = validator.validate_dataframe(df)
    assert not result.is_valid
    assert "DataFrame is empty" in result.issues

def test_zero_std_column():
    validator = DataValidator()
    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1],
        'B': [2, 3, 4, 5, 6]
    })
    result = validator.validate_dataframe(df)
    assert any("zero standard deviation" in issue for issue in result.issues)

def test_all_null_column():
    validator = DataValidator()
    df = pd.DataFrame({
        'A': [np.nan, np.nan, np.nan, np.nan],
        'B': [1, 2, 3, 4]
    })
    result = validator.validate_dataframe(df)
    assert any("100% null" in issue for issue in result.issues)

def test_mixed_null_and_zero_std():
    validator = DataValidator()
    df = pd.DataFrame({
        'A': [np.nan, np.nan, np.nan, np.nan],
        'B': [1, 1, 1, 1],
        'C': [1, 2, 3, 4]
    })
    result = validator.validate_dataframe(df)
    assert any("100% null" in issue for issue in result.issues)
    assert any("zero standard deviation" in issue for issue in result.issues)

def test_multi_index_reset():
    validator = DataValidator()
    arrays = [
        ["A", "A", "B", "B"],
        [1, 2, 1, 2]
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("ticker", "date"))
    df = pd.DataFrame({"Close": [1, 2, 3, 4]}, index=index)
    result = validator.validate_dataframe(df, required_columns=["Close"])
    assert result.metadata["multi_index_detected"]
    assert result.metadata["index_reset_applied"]
    assert result.is_valid 