# -*- coding: utf-8 -*-
"""
Price and market data cache layer using yfinance with Streamlit cache_data.
Used by Dashboard, Analyze, Scanner, and other pages for consistent TTL and no duplicate fetches.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

# Best-effort Yahoo session refresh — reduces stale-crumb HTTP 401s on Cloud.
try:
    _yf_utils = getattr(yf, "utils", None)
    if _yf_utils is not None and hasattr(_yf_utils, "get_json"):
        _yf_utils.get_json(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            proxy=None,
        )
except Exception:
    pass


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
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if isinstance(df, pd.DataFrame) and len(df.index) > 0:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_convert(None)
        # tz_localize(None) is naive-only; tz_convert(None) strips tz-aware indexes
        return df
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
    """News headlines — 15 min TTL. Preserves url/link for UI deep links."""
    try:
        raw = yf.Ticker(ticker).news or []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or {}
            if not isinstance(content, dict):
                content = {}
            url = (
                item.get("url")
                or item.get("link")
                or item.get("href")
                or content.get("url")
                or content.get("link")
                or content.get("href")
                or content.get("clickThroughUrl")
                or ""
            )
            title = (
                item.get("title")
                or item.get("headline")
                or content.get("title")
                or content.get("summary")
                or ""
            )
            title = str(title or "").strip()
            if title and "({'url'" in title:
                title = title.split("({")[0].strip()
            elif title and "({'" in title:
                title = title.split("({")[0].strip()
            if title.startswith("{"):
                title = ""
            merged = dict(item)
            merged["url"] = str(url or "").strip()
            merged["title"] = title
            out.append(merged)
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Macro / index series — 1h TTL bucket for SPY, VIX, yields, etc."""
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        return pd.DataFrame()


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
