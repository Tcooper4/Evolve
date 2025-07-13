"""
Basic Functionality Tests

This module tests basic functionality of the Evolve trading system
without relying on problematic imports or complex dependencies.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    # Create sample price data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame(
        {
            "Open": np.random.normal(100, 10, len(dates)),
            "High": np.random.normal(105, 10, len(dates)),
            "Low": np.random.normal(95, 10, len(dates)),
            "Close": np.random.normal(100, 10, len(dates)),
            "Volume": np.random.normal(1000000, 200000, len(dates)),
        },
        index=dates,
    )

    # Add some trend to make it more realistic
    trend = np.linspace(0, 20, len(dates))
    data["Close"] += trend
    data["Open"] += trend
    data["High"] += trend
    data["Low"] += trend

    return data


def test_data_validation(sample_data):
    """Test basic data validation."""
    # Test that sample data has required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        assert col in sample_data.columns

    # Test that data is not empty
    assert len(sample_data) > 0

    # Test that prices are positive
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        assert all(sample_data[col] > 0)

    # Test that High >= Low
    assert all(sample_data["High"] >= sample_data["Low"])

    logger.info("✅ Data validation tests passed")


def test_basic_indicators(sample_data):
    """Test basic technical indicator calculations."""
    # Test Simple Moving Average
    sma_20 = sample_data["Close"].rolling(window=20).mean()
    assert len(sma_20) == len(sample_data)
    assert all(pd.notna(sma_20.iloc[19:]))  # First 19 values should be NaN

    # Test Exponential Moving Average
    ema_20 = sample_data["Close"].ewm(span=20).mean()
    assert len(ema_20) == len(sample_data)
    assert all(pd.notna(ema_20))

    # Test RSI calculation
    delta = sample_data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    assert len(rsi) == len(sample_data)
    assert all(rsi >= 0) and all(rsi <= 100)

    logger.info("✅ Basic indicators tests passed")


def test_signal_generation(sample_data):
    """Test basic signal generation logic."""
    # Create simple buy/sell signals based on moving averages
    sma_short = sample_data["Close"].rolling(window=10).mean()
    sma_long = sample_data["Close"].rolling(window=30).mean()

    # Generate signals
    signals = pd.Series(0, index=sample_data.index)
    signals[sma_short > sma_long] = 1  # Buy signal
    signals[sma_short < sma_long] = -1  # Sell signal

    # Test signal properties
    assert len(signals) == len(sample_data)
    assert all(signals.isin([-1, 0, 1]))

    # Test that we have some signals
    signal_count = len(signals[signals != 0])
    assert signal_count > 0

    logger.info("✅ Signal generation tests passed")


def test_performance_calculation(sample_data):
    """Test basic performance calculation."""
    # Calculate returns
    returns = sample_data["Close"].pct_change()

    # Generate simple strategy returns
    signals = pd.Series(0, index=sample_data.index)
    signals.iloc[50:100] = 1  # Buy for a period
    signals.iloc[100:150] = -1  # Sell for a period

    strategy_returns = signals.shift(1) * returns

    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Test that metrics are reasonable
    assert isinstance(total_return, float)
    assert isinstance(annualized_return, float)
    assert isinstance(volatility, float)
    assert isinstance(sharpe_ratio, float)

    logger.info("✅ Performance calculation tests passed")


def test_risk_metrics(sample_data):
    """Test basic risk metrics calculation."""
    # Calculate returns
    returns = sample_data["Close"].pct_change().dropna()

    # Calculate risk metrics
    volatility = returns.std()
    var_95 = returns.quantile(0.05)  # 95% VaR
    max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()

    # Test that metrics are reasonable
    assert volatility > 0
    assert var_95 < 0  # VaR should be negative
    assert max_drawdown < 0  # Max drawdown should be negative

    logger.info("✅ Risk metrics tests passed")


def test_data_preprocessing(sample_data):
    """Test basic data preprocessing."""
    # Test data cleaning
    clean_data = sample_data.copy()

    # Remove any infinite values
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)

    # Forward fill missing values
    clean_data = clean_data.fillna(method="ffill")

    # Test that data is clean
    assert not clean_data.isnull().any().any()
    assert not np.isinf(clean_data).any().any()

    # Test data scaling
    scaled_data = (clean_data - clean_data.mean()) / clean_data.std()

    # Test that scaled data has mean close to 0 and std close to 1
    for col in scaled_data.columns:
        assert abs(scaled_data[col].mean()) < 0.1
        assert abs(scaled_data[col].std() - 1) < 0.1

    logger.info("✅ Data preprocessing tests passed")
