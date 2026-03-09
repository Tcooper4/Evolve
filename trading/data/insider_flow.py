"""Insider transaction pipeline using yfinance SEC Form 4 data."""

from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=128)
def get_insider_flow(symbol: str, days_back: int = 90) -> dict:
    """Summarize insider transactions over a recent window."""
    try:
        insiders = yf.Ticker(symbol).insider_transactions

        if insiders is None or getattr(insiders, "empty", True):
            return {
                "symbol": symbol,
                "transactions": [],
                "net_shares_90d": 0,
                "buy_count": 0,
                "sell_count": 0,
                "signal": "NO_ACTIVITY",
                "largest_transaction": None,
            }

        # Ensure DatetimeIndex is timezone-naive for comparisons
        if isinstance(insiders.index, pd.DatetimeIndex):
            if getattr(insiders.index, "tz", None) is not None:
                insiders.index = insiders.index.tz_localize(None)

        cutoff = datetime.today() - timedelta(days=days_back)
        recent = insiders[insiders.index >= cutoff]

        transactions = []
        net = 0
        buys = 0
        sells = 0

        for date, row in recent.iterrows():
            shares = int(row.get("Shares") or 0)
            txn = str(row.get("Transaction") or "").upper()

            is_buy = any(k in txn for k in ["BUY", "PURCHASE", "ACQUI"])
            is_sell = any(k in txn for k in ["SELL", "SALE", "DISPOS"])

            if is_buy:
                net += shares
                buys += 1
            elif is_sell:
                net -= shares
                sells += 1

            val = row.get("Value")
            try:
                val_float = float(val) if val is not None else None
            except Exception:
                val_float = None

            transactions.append(
                {
                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                    "insider": row.get("Insider", "?"),
                    "title": row.get("Position", ""),
                    "transaction_type": txn,
                    "shares": shares,
                    "value": val_float,
                    "is_buy": is_buy,
                }
            )

        transactions.sort(key=lambda x: x["date"], reverse=True)

        total_txns = buys + sells
        if total_txns == 0:
            signal = "NO_ACTIVITY"
        elif buys > sells * 2:
            signal = "INSIDER_BUYING"
        elif sells > buys * 2:
            signal = "INSIDER_SELLING"
        else:
            signal = "MIXED"

        largest = max(
            transactions,
            key=lambda x: abs(x.get("value") or 0),
            default=None,
        )

        return {
            "symbol": symbol,
            "transactions": transactions[:10],
            "net_shares_90d": net,
            "buy_count": buys,
            "sell_count": sells,
            "signal": signal,
            "largest_transaction": largest,
        }

    except Exception as e:  # pragma: no cover - defensive fallback
        return {
            "symbol": symbol,
            "transactions": [],
            "net_shares_90d": 0,
            "buy_count": 0,
            "sell_count": 0,
            "signal": "NO_ACTIVITY",
            "largest_transaction": None,
            "error": str(e),
        }

