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


def get_insider_cluster_signal(
    symbol: str,
    days_back: int = 90,
    cluster_window_days: int = 30,
) -> dict:
    """
    Detects insider cluster buying —
    multiple insiders buying in the
    same time window.

    Academic basis: Seyhun (1988) and
    Lakonishok & Lee (2001) show cluster
    insider buying (3+ insiders, same
    month) predicts 4-6% abnormal
    returns over 6 months vs ~1% for
    single insider buys.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral_cluster(sym)

    try:
        t = yf.Ticker(sym)
        insiders = t.insider_transactions
        if insiders is None or getattr(insiders, "empty", True):
            return _neutral_cluster(sym)

        try:
            if "Start Date" in insiders.columns:
                insiders = insiders.copy()
                insiders["_date"] = pd.to_datetime(
                    insiders["Start Date"],
                    errors="coerce",
                    utc=True,
                ).dt.tz_convert(None)
            else:
                idx = insiders.index
                if not isinstance(idx, pd.DatetimeIndex):
                    insiders.index = pd.to_datetime(
                        idx,
                        errors="coerce",
                        utc=True,
                    )
                if getattr(insiders.index, "tz", None):
                    insiders.index = insiders.index.tz_convert(None)
                insiders["_date"] = insiders.index
        except Exception:
            return _neutral_cluster(sym)

        _cutoff = datetime.utcnow() - timedelta(days=days_back)
        _recent = insiders[insiders["_date"] >= _cutoff].copy()

        if _recent.empty:
            result = _neutral_cluster(sym)
            result["success"] = True
            return result

        def _is_buy(row) -> bool:
            _txt = str(row.get("Transaction", "") or "").lower()
            _text = str(row.get("Text", "") or "").lower()
            return (
                "purchase" in _txt
                or "buy" in _txt
                or "acquisition" in _txt
                or "purchase" in _text
            )

        def _is_sell(row) -> bool:
            _txt = str(row.get("Transaction", "") or "").lower()
            return "sale" in _txt or "sell" in _txt

        _recent["_is_buy"] = _recent.apply(_is_buy, axis=1)
        _recent["_is_sell"] = _recent.apply(_is_sell, axis=1)
        _recent["_insider"] = _recent.get("Insider", _recent.index).astype(str)

        _max_cluster_buys = 0
        _max_cluster_sells = 0
        _best_buy_window_buyers = []
        _best_sell_window_sellers = []

        _dates = sorted(_recent["_date"].dropna())
        for _start in _dates:
            _end = _start + timedelta(days=cluster_window_days)
            _window = _recent[
                (_recent["_date"] >= _start) & (_recent["_date"] < _end)
            ]
            _buyers = _window[_window["_is_buy"]]["_insider"].unique().tolist()
            _sellers = _window[_window["_is_sell"]]["_insider"].unique().tolist()

            if len(_buyers) > _max_cluster_buys:
                _max_cluster_buys = len(_buyers)
                _best_buy_window_buyers = _buyers

            if len(_sellers) > _max_cluster_sells:
                _max_cluster_sells = len(_sellers)
                _best_sell_window_sellers = _sellers

        _signal = "NEUTRAL"
        _strength = 5.0

        if _max_cluster_buys >= 3:
            _signal = "STRONG_BUY"
            _strength = min(9.0, 6.0 + _max_cluster_buys * 0.5)
        elif _max_cluster_buys == 2:
            _signal = "BUY"
            _strength = 7.0
        elif _max_cluster_sells >= 3:
            _signal = "SELL"
            _strength = max(2.0, 4.0 - _max_cluster_sells * 0.3)

        _net = _max_cluster_buys - _max_cluster_sells * 0.5

        return {
            "symbol": sym,
            "cluster_buy_count": _max_cluster_buys,
            "cluster_sell_count": _max_cluster_sells,
            "cluster_signal": _signal,
            "cluster_strength": round(_strength, 1),
            "recent_buyers": _best_buy_window_buyers[:5],
            "recent_sellers": _best_sell_window_sellers[:3],
            "net_sentiment": round(_net, 1),
            "success": True,
        }
    except Exception as e:
        logger.debug("Insider cluster failed %s: %s", sym, e)
        return _neutral_cluster(sym)


def _neutral_cluster(sym: str) -> dict:
    return {
        "symbol": sym,
        "cluster_buy_count": 0,
        "cluster_sell_count": 0,
        "cluster_signal": "NEUTRAL",
        "cluster_strength": 5.0,
        "recent_buyers": [],
        "recent_sellers": [],
        "net_sentiment": 0.0,
        "success": False,
    }

