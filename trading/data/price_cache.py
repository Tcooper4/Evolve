# -*- coding: utf-8 -*-
"""
Price and market data cache layer using yfinance with Streamlit cache_data.
Used by Dashboard, Analyze, Scanner, and other pages for consistent TTL and no duplicate fetches.
"""

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=15, show_spinner=False)
def get_quote(ticker: str) -> dict:
    """Live quote — refreshes every 15 seconds."""
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "price": getattr(info, "last_price", None),
            "prev_close": getattr(info, "previous_close", None),
            "volume": getattr(info, "last_volume", None),
            "market_cap": getattr(info, "market_cap", None),
        }
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history. 60s TTL for intraday freshness."""
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Fundamentals — 5 min TTL."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def get_news(ticker: str) -> list:
    """News headlines — 15 min TTL."""
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def batch_quotes(tickers: list) -> pd.DataFrame:
    """Batch OHLCV for scanner — 5 min TTL."""
    if not tickers:
        return pd.DataFrame()
    try:
        return yf.download(
            tickers,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()
