"""
Live market events monitor: scans a watchlist for significant volume/price spikes.
Used by the Home page to show featured events and a scrollable event feed.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST = [
    "AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "SPY",
    "QQQ", "JPM", "BAC", "GS", "AMD", "NFLX", "UBER", "PLTR", "ARM",
]

MARKET_CAP_RANK = {
    "AAPL": 1, "MSFT": 2, "NVDA": 3, "GOOGL": 4, "AMZN": 5,
    "META": 6, "TSLA": 7, "SPY": 8, "QQQ": 9, "JPM": 10,
    "BAC": 11, "GS": 12, "AMD": 13, "NFLX": 14, "UBER": 15,
    "PLTR": 16, "ARM": 17,
}


def scan_watchlist(
    watchlist: list = None,
    volume_multiplier: float = 3.0,
    min_price_move_pct: float = 2.0,
    interval: str = "5m",
    period: str = "1d",
) -> list[dict]:
    """
    Scan watchlist for significant candles.
    Returns list of spike dicts sorted by impact score descending.
    Impact score = volume_ratio * price_move_pct * (1 / market_cap_rank)
    Only returns spikes from the last 10 minutes.
    """
    if watchlist is None:
        watchlist = DEFAULT_WATCHLIST
    spikes = []
    cutoff = datetime.now() - timedelta(minutes=10)

    for symbol in watchlist:
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
            if df is None or len(df) < 25:
                continue
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df["avg_volume"] = df["Volume"].rolling(20).mean()
            df["volume_ratio"] = df["Volume"] / df["avg_volume"]
            df["price_move_pct"] = (df["Close"] - df["Open"]).abs() / df["Open"] * 100
            df["direction"] = np.where(df["Close"] >= df["Open"], "up", "down")
            # Only look at recent candles
            cutoff_ts = pd.Timestamp(cutoff)
            if df.index.tz is not None:
                cutoff_ts = cutoff_ts.tz_localize(df.index.tz) if cutoff_ts.tz is None else cutoff_ts
            recent = df[df.index >= cutoff_ts]
            significant = recent[
                (recent["volume_ratio"] >= volume_multiplier)
                & (recent["price_move_pct"] >= min_price_move_pct)
            ]
            for ts, row in significant.iterrows():
                cap_rank = MARKET_CAP_RANK.get(symbol, 99)
                impact = row["volume_ratio"] * row["price_move_pct"] * (1 / cap_rank)
                spikes.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "direction": row["direction"],
                    "price_move_pct": row["price_move_pct"],
                    "volume_ratio": row["volume_ratio"],
                    "close": row["Close"],
                    "impact_score": impact,
                    "df": df.copy(),
                })
        except Exception as e:
            logger.warning("MarketMonitor: failed to scan %s: %s", symbol, e)
            continue

    return sorted(spikes, key=lambda x: x["impact_score"], reverse=True)
