import pandas as pd
import numpy as np
from trading.market.market_analyzer import MarketAnalyzer


def test_volatility_trend_unknown_with_insufficient_history():
    analyzer = MarketAnalyzer()
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = pd.DataFrame({'Close': np.random.rand(10) + 100}, index=dates)
    result = analyzer.analyze_volatility(data)
    assert result['volatility_trend'] == 'unknown'


def test_volatility_trend_calculated_with_sufficient_history():
    analyzer = MarketAnalyzer()
    dates = pd.date_range(start='2024-01-01', periods=260, freq='D')
    data = pd.DataFrame({'Close': np.random.rand(260) + 100}, index=dates)
    result = analyzer.analyze_volatility(data)
    assert result['volatility_trend'] in ['increasing', 'decreasing']
