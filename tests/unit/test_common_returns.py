import pandas as pd
import numpy as np
from trading.utils.common import calculate_returns, calculate_sharpe_ratio


def test_calculate_returns_log():
    prices = pd.Series([1.0, 1.1, 1.21])
    expected = np.log(prices / prices.shift(1))
    pd.testing.assert_series_equal(calculate_returns(prices, method="log"), expected)


def test_calculate_returns_simple():
    prices = pd.Series([1.0, 1.1, 1.21])
    expected = prices.pct_change()
    pd.testing.assert_series_equal(calculate_returns(prices, method="simple"), expected)


def test_calculate_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.005])
    sr = calculate_sharpe_ratio(returns)
    assert isinstance(sr, float)
