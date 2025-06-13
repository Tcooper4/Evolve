import pandas as pd
from trading.analysis.market_analyzer import MarketAnalyzer


def _make_df(length: int) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=length, freq="D")
    values = range(1, length + 1)
    data = {
        "Open": values,
        "High": [v + 1 for v in values],
        "Low": [v - 1 for v in values],
        "Close": values,
        "Volume": [100] * length,
    }
    return pd.DataFrame(data, index=dates)


def test_market_summary_single_row():
    ma = MarketAnalyzer()
    ma.data["TEST"] = _make_df(1)
    ma.indicators["TEST"] = ma.data["TEST"]  # minimal indicators
    summary = ma.get_market_summary("TEST")
    assert summary is not None
    assert summary["price"]["daily_change"] is None
    assert summary["price"]["weekly_change"] is None
    assert summary["price"]["monthly_change"] is None


def test_market_summary_short_data():
    ma = MarketAnalyzer()
    ma.data["TEST"] = _make_df(3)
    ma.indicators["TEST"] = ma.data["TEST"]
    summary = ma.get_market_summary("TEST")
    assert summary["price"]["daily_change"] == 1
    assert summary["price"]["weekly_change"] == 2
    assert summary["price"]["monthly_change"] == 2


def test_market_summary_medium_data():
    ma = MarketAnalyzer()
    ma.data["TEST"] = _make_df(10)
    ma.indicators["TEST"] = ma.data["TEST"]
    summary = ma.get_market_summary("TEST")
    assert summary["price"]["daily_change"] == 1
    assert summary["price"]["weekly_change"] == 5
    assert summary["price"]["monthly_change"] == 9
