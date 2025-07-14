"""Pytest configuration and shared fixtures for the trading system tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add all necessary directories to Python path
for directory in ["core", "trading", "models", "utils", "optimizer", "market_analysis"]:
    dir_path = project_root / directory
    if dir_path.exists():
        sys.path.insert(0, str(dir_path))


@pytest.fixture
def sample_price_data():
    """Generate realistic sample price data for testing."""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns with volatility
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame(
        {
            "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": np.random.normal(1000000, 100000, 100),
        },
        index=dates,
    )

    # Ensure High >= Low and High >= Close >= Low
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture
def sample_price_data_with_indicators(sample_price_data):
    """Generate sample price data with pandas-ta indicators for testing."""
    import pandas_ta as ta

    df = sample_price_data.copy()

    # Add RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # Add MACD
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)

    # Add Bollinger Bands
    bb = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # Add SMA
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    # Add EMA
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    return df


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    mock_data = pd.DataFrame(
        {
            "Open": np.random.normal(100, 2, 30),
            "High": np.random.normal(102, 2, 30),
            "Low": np.random.normal(98, 2, 30),
            "Close": np.random.normal(100, 2, 30),
            "Volume": np.random.normal(1000000, 100000, 30),
        },
        index=dates,
    )
    return mock_data


@pytest.fixture
def mock_alpha_vantage_response():
    """Mock Alpha Vantage API response."""
    return {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2024-01-01",
            "4. Output Size": "Compact",
            "5. Time Zone": "US/Eastern",
        },
        "Time Series (Daily)": {
            "2024-01-01": {
                "1. open": "100.00",
                "2. high": "102.00",
                "3. low": "98.00",
                "4. close": "101.00",
                "5. volume": "1000000",
            }
        },
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Based on the market analysis, I recommend a BUY signal for AAPL with 75% confidence."
                }
            }
        ]
    }


@pytest.fixture
def strategy_config():
    """Shared strategy configuration for testing."""
    return {
        "sma": {"short_window": 20, "long_window": 50},
        "rsi": {"window": 14, "overbought": 70, "oversold": 30},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"window": 20, "num_std": 2},
    }


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    mock = Mock()
    mock.process_goal.return_value = {
        "action": "buy",
        "confidence": 0.8,
        "reasoning": "Test reasoning",
        "strategy": "RSI",
        "parameters": {"period": 14},
    }
    mock.learn_from_performance.return_value = True
    mock.adapt_goals.return_value = True
    return mock


@pytest.fixture
def mock_router():
    """Create a mock router for testing."""
    mock = Mock()
    mock.route_intent.return_value = {
        "intent": "buy",
        "entity": "AAPL",
        "confidence": 0.9,
        "strategy": "RSI",
    }
    mock.detect_intent.return_value = {
        "intent": "buy",
        "confidence": 0.85,
        "entities": ["AAPL"],
    }
    return mock


@pytest.fixture
def mock_performance_logger():
    """Create a mock performance logger for testing."""
    mock = Mock()
    mock.log_performance.return_value = {
        "accuracy": 0.8,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.1,
        "total_return": 0.25,
        "win_rate": 0.65,
    }
    mock.get_performance_history.return_value = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "accuracy": np.random.uniform(0.6, 0.9, 10),
            "sharpe_ratio": np.random.uniform(0.5, 2.0, 10),
        }
    )
    return mock


@pytest.fixture
def mock_forecaster():
    """Create a mock forecaster for testing."""
    mock = Mock()
    mock.predict.return_value = pd.DataFrame(
        {
            "forecast": np.random.normal(100, 2, 30),
            "lower_bound": np.random.normal(98, 2, 30),
            "upper_bound": np.random.normal(102, 2, 30),
        },
        index=pd.date_range(start="2024-04-01", periods=30, freq="D"),
    )
    mock.train.return_value = {"accuracy": 0.85, "mse": 0.02}
    return mock


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "Open": np.random.normal(100, 2, 50),
            "High": np.random.normal(102, 2, 50),
            "Low": np.random.normal(98, 2, 50),
            "Close": np.random.normal(100, 2, 50),
            "Volume": np.random.normal(1000000, 100000, 50),
        },
        index=dates,
    )


@pytest.fixture
def mock_strategy_settings():
    """Mock strategy settings for testing."""
    return {
        "rsi": {"optimal_period": 14, "buy_threshold": 30, "sell_threshold": 70},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"window": 20, "std": 2},
    }


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "api: mark test as requiring external API")
    config.addinivalue_line("markers", "unit: mark test as a unit test")


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Automatically mock external APIs for all tests."""
    with patch("yfinance.download") as mock_yf, patch(
        "requests.get"
    ) as mock_requests, patch("openai.ChatCompletion.create") as mock_openai:
        # Mock yfinance
        mock_yf.return_value = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [98, 99, 100],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        # Mock requests
        mock_requests.return_value.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "100.00",
                    "2. high": "102.00",
                    "3. low": "98.00",
                    "4. close": "101.00",
                    "5. volume": "1000000",
                }
            }
        }

        # Mock OpenAI
        mock_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Based on analysis, BUY signal with 75% confidence."
                    }
                }
            ]
        }

        yield
