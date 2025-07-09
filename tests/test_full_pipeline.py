#!/usr/bin/env python3
"""
Comprehensive Test for Full Trading Pipeline (Pytest Version)

This script tests the complete trading pipeline:
Prompt → Forecast → Strategy → Backtest → Report → Trade

Usage:
    pytest tests/test_full_pipeline.py
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.llm.agent import get_prompt_agent
from models.forecast_router import ForecastRouter
from trading.backtesting.backtester import Backtester
from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
from trading.data.providers.fallback_provider import get_fallback_provider
from trading.ui.components import create_system_metrics_panel

@pytest.fixture
def fallback_provider():
    return get_fallback_provider()

@pytest.fixture
def forecast_router():
    return ForecastRouter()

@pytest.fixture
def prompt_agent():
    return get_prompt_agent()

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    volumes = np.random.uniform(1000000, 5000000, len(dates))
    return pd.DataFrame({'close': prices, 'volume': volumes}, index=dates)

def test_data_provider(fallback_provider):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    data = fallback_provider.get_historical_data('AAPL', start_date, end_date, '1d')
    assert data is not None and not data.empty, "Failed to retrieve historical data"
    price = fallback_provider.get_live_price('AAPL')
    assert price is not None and price > 0, "Failed to retrieve live price"

def test_forecast_router(forecast_router, sample_data):
    models = ['arima', 'lstm', 'xgboost']
    for model in models:
        result = forecast_router.get_forecast(data=sample_data, horizon=15, model_type=model)
        assert result and 'forecast' in result, f"{model.upper()} forecast failed"

def test_backtest_engine(sample_data):
    # Create multi-level DataFrame for backtester
    dates = sample_data.index
    data = pd.DataFrame(index=dates)
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        volumes = np.random.uniform(1000000, 5000000, len(dates))
        signals = np.random.choice([-1, 0, 1], size=len(dates), p=[0.3, 0.4, 0.3])
        data[(symbol, 'Close')] = prices
        data[(symbol, 'Volume')] = volumes
        data[(symbol, 'Signal')] = signals
    backtester = Backtester(data)
    strategies = ['RSI Mean Reversion', 'Bollinger Bands', 'Moving Average Crossover']
    for strategy in strategies:
        equity_curve, trade_log, metrics = backtester.run_backtest([strategy])
        assert metrics, f"{strategy} backtest failed - no metrics"

def test_optimizer():
    optimizer = SelfTuningOptimizer()
    assert optimizer is not None, "Failed to initialize SelfTuningOptimizer"

def test_prompt_agent(prompt_agent):
    prompt = "Forecast AAPL for the next 7 days."
    result = prompt_agent.process_prompt(prompt)
    assert result is not None, "Prompt agent failed to process prompt"

def test_system_metrics_panel():
    panel = create_system_metrics_panel()
    assert panel is not None, "Failed to create system metrics panel" 