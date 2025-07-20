"""
Unit tests for strategy validation and error handling.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from utils.strategy_utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    validate_signal_schema,
    calculate_risk_metrics
)


class TestStrategyValidation:
    """Test strategy validation with edge cases."""

    def test_calculate_returns_empty_dataframe(self):
        """Test calculate_returns with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_returns(empty_df)
        assert result.empty

    def test_calculate_returns_missing_close_column(self):
        """Test calculate_returns with missing 'Close' column."""
        df = pd.DataFrame({'Open': [100, 101, 102], 'High': [105, 106, 107]})
        result = calculate_returns(df)
        assert result.empty

    def test_calculate_sharpe_ratio_empty_series(self):
        """Test calculate_sharpe_ratio with empty Series."""
        empty_series = pd.Series()
        result = calculate_sharpe_ratio(empty_series)
        assert result == 0.0

    def test_calculate_max_drawdown_empty_series(self):
        """Test calculate_max_drawdown with empty Series."""
        empty_series = pd.Series()
        result = calculate_max_drawdown(empty_series)
        assert result == 0.0

    def test_validate_signal_schema_empty_dataframe(self):
        """Test validate_signal_schema with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = validate_signal_schema(empty_df)
        assert result is False

    def test_validate_signal_schema_missing_close_column(self):
        """Test validate_signal_schema with missing 'Close' column."""
        df = pd.DataFrame({'Open': [100, 101], 'SignalType': ['buy', 'sell']})
        result = validate_signal_schema(df)
        assert result is False

    def test_validate_signal_schema_missing_signal_type(self):
        """Test validate_signal_schema with missing 'SignalType' column."""
        df = pd.DataFrame({'Close': [100, 101], 'Open': [99, 100]})
        result = validate_signal_schema(df)
        assert result is False

    def test_calculate_risk_metrics_empty_series(self):
        """Test calculate_risk_metrics with empty Series."""
        empty_series = pd.Series()
        result = calculate_risk_metrics(empty_series)
        assert result == {}

    def test_calculate_risk_metrics_valid_data(self):
        """Test calculate_risk_metrics with valid data."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result

    def test_calculate_returns_valid_data(self):
        """Test calculate_returns with valid data."""
        prices = pd.Series([100, 101, 102, 101, 103])
        result = calculate_returns(prices)
        assert len(result) == len(prices)
        assert not result.empty

    def test_calculate_sharpe_ratio_valid_data(self):
        """Test calculate_sharpe_ratio with valid data."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        result = calculate_sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_calculate_max_drawdown_valid_data(self):
        """Test calculate_max_drawdown with valid data."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        result = calculate_max_drawdown(returns)
        assert isinstance(result, float)

    def test_validate_signal_schema_valid_data(self):
        """Test validate_signal_schema with valid data."""
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'SignalType': ['buy', 'sell', 'hold']
        })
        result = validate_signal_schema(df)
        assert result is True

    @patch('utils.strategy_utils.logger')
    def test_logger_called_on_error(self, mock_logger):
        """Test that logger.error is called when strategy fails."""
        empty_df = pd.DataFrame()
        calculate_returns(empty_df)
        mock_logger.error.assert_called()
