import pandas as pd
from trading.portfolio.portfolio_manager import PortfolioManager


def test_calculate_sharpe_ratio():
    pm = PortfolioManager()
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    ratio = pm.calculate_sharpe_ratio(returns)
    assert isinstance(ratio, float)
    assert ratio != 0.0
