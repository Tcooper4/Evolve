"""
Links significant volume/price events to contemporaneous news.
Used for the news-annotated candlestick chart.
"""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from trading.utils.data_manager import disk_cache_get, disk_cache_set

logger = logging.getLogger(__name__)


def detect_significant_candles(
    hist: pd.DataFrame,
    volume_threshold: float = 2.0,
    price_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Returns rows where volume > N×avg AND |price change| > threshold.
    Adds columns: volume_ratio, price_change_pct, is_significant, candle_type.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    df = hist.copy()

    # Normalize column names (work with either Close/Volume or lowercase)
    cols_lower = [c.lower() for c in df.columns]
    col_map = {c: c.lower() for c in df.columns}
    df.columns = cols_lower

    # If we lost Close due to mismatch, fall back to original
    if "close" not in df.columns and "Close" in hist.columns:
        df = hist.copy()
        df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]

    if "Close" in df.columns and "close" not in df.columns:
        df["close"] = df["Close"]
    if "Volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["Volume"]

    if "close" not in df.columns or "volume" not in df.columns:
        return df

    df["price_change_pct"] = df["close"].pct_change()
    rolling_vol = df["volume"].rolling(20, min_periods=5).mean()
    df["volume_ratio"] = df["volume"] / rolling_vol.clip(lower=1)
    # Flag abnormal volume: (2× vol AND ≥2% move) OR extreme 3× volume alone
    vol_and_move = (df["volume_ratio"] >= volume_threshold) & (
        df["price_change_pct"].abs() >= price_threshold
    )
    vol_extreme = df["volume_ratio"] >= max(volume_threshold * 1.5, 3.0)
    df["is_significant"] = vol_and_move | vol_extreme
    df["candle_type"] = np.where(
        df["price_change_pct"] > 0,
        "bullish",
        np.where(df["price_change_pct"] < 0, "bearish", "neutral"),
    )
    return df


@lru_cache(maxsize=64)
def get_news_for_date(symbol: str, date_str: str, n_articles: int = 5) -> List[Dict]:
    """
    Fetch news articles from around a specific date.
    date_str: ISO format 'YYYY-MM-DD'.
    """
    cache_key = f"news_date:v2:{symbol}:{date_str}:{n_articles}"
    cached = disk_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from trading.data.news_aggregator import get_news

        articles = list(get_news(symbol, max_items=max(n_articles, 8)) or [])
        # Twitter/X dated search when bearer is set (fast breaking context)
        try:
            from trading.data.twitter_headlines import get_twitter_symbol_headlines

            tw = get_twitter_symbol_headlines(
                symbol, max_items=n_articles, since_date=date_str
            )
            if tw:
                articles.extend(tw)
        except Exception as _tw_e:
            logger.debug("Twitter date overlay skipped: %s", _tw_e)

        if not articles:
            return []

        target = datetime.strptime(date_str, "%Y-%m-%d")
        window_start = target - timedelta(days=1)
        window_end = target + timedelta(days=2)

        relevant: List[Dict] = []
        for a in articles:
            try:
                pub_raw = a.get("published", "") or ""
                if not pub_raw:
                    # Dated Twitter search already windowed — keep if source is twitter
                    if a.get("source_type") == "twitter":
                        relevant.append(a)
                    continue
                s = pub_raw.replace("Z", "+00:00")
                pub_dt = datetime.fromisoformat(s.split("+")[0])
                # drop tz for comparison with naive window
                if getattr(pub_dt, "tzinfo", None) is not None:
                    pub_dt = pub_dt.replace(tzinfo=None)
                if window_start <= pub_dt <= window_end:
                    relevant.append(a)
            except Exception:
                continue

        # If no time-filtered results, return top articles anyway
        if relevant:
            result = relevant[:n_articles]
        else:
            result = articles[:3]
        disk_cache_set(cache_key, result, ttl=3600)  # 1 hour
        return result
    except Exception as e:  # pragma: no cover
        logger.debug("News for date failed for %s on %s: %s", symbol, date_str, e)
        return []


def build_chart_annotations(
    df: pd.DataFrame,
    symbol: str,
    max_annotations: int = 10,
) -> List[Dict]:
    """
    For each significant candle, build a Plotly annotation dict with
    news headlines as hover text.

    Returns list of dicts for use in go.Figure().add_trace(...) as markers.
    """
    if df is None or df.empty or "is_significant" not in df.columns:
        return []

    sig = df[df["is_significant"]].copy()
    if sig.empty:
        return []

    # Limit to top N most significant by volume ratio
    sig = sig.nlargest(max_annotations, "volume_ratio")

    annotations: List[Dict] = []
    for date, row in sig.iterrows():
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        news = get_news_for_date(symbol, date_str, n_articles=4)

        # Build hover text
        if news:
            news_lines = [
                f"• {a.get('title', '')[:80]} [{a.get('source', '')}]"
                for a in news[:4]
            ]
            hover = (
                f"<b>{date_str}</b><br>"
                f"Vol: {row['volume_ratio']:.1f}x avg<br>"
                f"Move: {row['price_change_pct'] * 100:+.1f}%<br><br>"
                + "<br>".join(news_lines)
            )
        else:
            hover = (
                f"<b>{date_str}</b><br>"
                f"Vol: {row['volume_ratio']:.1f}x avg<br>"
                f"Move: {row['price_change_pct'] * 100:+.1f}%<br>"
                "No news found for this date"
            )

        high_val = row.get("High") or row.get("high") or row.get("close")
        try:
            price_val = float(high_val) * 1.005 if high_val is not None else float(
                row.get("close", 0)
            )
        except Exception:
            price_val = float(row.get("close", 0) or 0)

        color = "#00FF88" if row.get("candle_type") == "bullish" else "#FF4444"

        annotations.append(
            {
                "date": date_str,
                "price": price_val,
                "text": "📰",
                "hover": hover,
                "color": color,
                "volume_ratio": float(row["volume_ratio"]),
                "price_change_pct": float(row["price_change_pct"]),
                "news": news,
            }
        )

    return annotations

