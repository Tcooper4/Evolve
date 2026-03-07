"""
Analysis module for market data processing and analysis.
"""

from trading.market.market_analyzer import MarketAnalyzer

# Live market events monitor
from trading.analysis.market_monitor import scan_watchlist, DEFAULT_WATCHLIST
from trading.analysis.chart_builder import build_event_chart
from trading.analysis.event_news_fetcher import fetch_news_around_event
from trading.analysis.news_ranker import rank_news_by_relevance

__all__ = [
    "MarketAnalyzer",
    "scan_watchlist",
    "DEFAULT_WATCHLIST",
    "build_event_chart",
    "fetch_news_around_event",
    "rank_news_by_relevance",
]
