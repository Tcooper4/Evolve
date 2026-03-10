"""Insider transaction pipeline using yfinance SEC Form 4 data."""

from datetime import datetime, timedelta
from functools import lru_cache
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


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

        # Normalize index to a naive DatetimeIndex for safe comparisons
        try:
            idx = insiders.index
            if not isinstance(idx, pd.DatetimeIndex):
                # Some yfinance versions return integer/epoch-style index; convert explicitly
                insiders.index = pd.to_datetime(idx, errors="coerce", utc=True)
                idx = insiders.index
            if getattr(idx, "tz", None) is not None:
                insiders.index = idx.tz_convert(None)
                idx = insiders.index
        except Exception as e:
            logger.debug("Could not normalize insider_transactions index for %s: %s", symbol, e)

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        try:
            recent = insiders[insiders.index >= cutoff]
        except Exception as e:
            # If comparison fails due to mixed types, log and skip date filtering
            logger.debug(
                "Date filtering insider_transactions failed for %s (using full dataset): %s",
                symbol,
                e,
            )
            recent = insiders

        transactions = []
        net = 0
        buys = 0
        sells = 0

        for date, row in recent.iterrows():
            try:
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
            except Exception:
                # Skip malformed rows but continue processing others
                logger.debug("Skipping malformed insider transaction row for %s", symbol)

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

