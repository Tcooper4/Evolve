"""
Earnings reaction tracker.
Measures historical price reactions to earnings announcements.
Helps answer: "How does AAPL typically move after earnings?"
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_reaction_cache: dict = {}
_REACTION_TTL_SECONDS = 3600  # 1 hour

def get_earnings_reactions(symbol: str, num_quarters: int = 8) -> Dict[str, Any]:
    import time

    _key = f"{symbol}_{num_quarters}"
    _now = time.time()
    if _key in _reaction_cache:
        _entry = _reaction_cache[_key]
        if _now - _entry["ts"] < _REACTION_TTL_SECONDS:
            return _entry["data"]
    result = _compute_earnings_reactions(symbol, num_quarters)
    _reaction_cache[_key] = {"data": result, "ts": _now}
    return result


def _compute_earnings_reactions(symbol: str, num_quarters: int = 8) -> Dict[str, Any]:
    """
    Fetch historical earnings dates and measure the price reaction
    in the 1-day, 3-day, and 5-day windows after each announcement.

    Returns:
        reactions: list of dicts per quarter:
            date, eps_actual, eps_estimate, surprise_pct,
            move_1d, move_3d, move_5d, direction
        avg_move_1d: float — mean abs 1-day move across quarters
        avg_move_3d: float — mean abs 3-day move
        beat_rate: float — % of quarters where EPS beat estimate
        positive_reaction_rate: float — % of beats with positive 1d move
        typical_range: tuple — (low_pct, high_pct) of 1d moves
        next_earnings: dict from get_upcoming_earnings or None
    """
    try:
        ticker = yf.Ticker(symbol)

        hist_earnings = getattr(ticker, "earnings_history", None)
        if hist_earnings is None or (
            hasattr(hist_earnings, "empty") and hist_earnings.empty
        ):
            return _empty_reaction(symbol, "No earnings history available")

        if not hasattr(hist_earnings, "index"):
            return _empty_reaction(symbol, "No earnings history available")

        price_hist = ticker.history(period="5y", interval="1d")
        if price_hist.empty:
            return _empty_reaction(symbol, "No price history available")

        if price_hist.index.tz is not None:
            price_hist.index = price_hist.index.tz_localize(None)
        if "Close" not in price_hist.columns and "close" in price_hist.columns:
            price_hist = price_hist.rename(columns={"close": "Close"})

        reactions = []
        earnings_dates = (
            hist_earnings.index[-num_quarters:]
            if len(hist_earnings) >= num_quarters
            else hist_earnings.index
        )

        for date in earnings_dates:
            try:
                if hasattr(date, "tz_localize"):
                    date = date.tz_localize(None)
                elif hasattr(date, "tz") and date.tz is not None:
                    date = date.replace(tzinfo=None)
                date_ts = pd.Timestamp(date)

                future_dates = price_hist.index[price_hist.index >= date_ts]
                if len(future_dates) < 6:
                    continue

                d0 = future_dates[0]
                d0_price = float(price_hist.loc[d0, "Close"])

                past_dates = price_hist.index[price_hist.index < date_ts]
                if len(past_dates) == 0:
                    continue
                d_minus1_price = float(price_hist.loc[past_dates[-1], "Close"])

                move_1d = (
                    (float(price_hist.loc[future_dates[1], "Close"]) / d_minus1_price - 1)
                    * 100
                    if len(future_dates) > 1
                    else None
                )
                move_3d = (
                    (float(price_hist.loc[future_dates[3], "Close"]) / d_minus1_price - 1)
                    * 100
                    if len(future_dates) > 3
                    else None
                )
                move_5d = (
                    (float(price_hist.loc[future_dates[5], "Close"]) / d_minus1_price - 1)
                    * 100
                    if len(future_dates) > 5
                    else None
                )

                row = hist_earnings.loc[date]
                if hasattr(row, "to_dict"):
                    row = row.to_dict()
                eps_actual = float(row.get("epsActual", 0) or 0)
                eps_est = float(row.get("epsEstimate", 0) or row.get("epsEstimate", 0) or 0)
                surprise = (
                    float((eps_actual - eps_est) / abs(eps_est) * 100) if eps_est != 0 else 0.0
                )

                reactions.append(
                    {
                        "date": str(date_ts.date()),
                        "eps_actual": round(eps_actual, 2),
                        "eps_estimate": round(eps_est, 2),
                        "surprise_pct": round(surprise, 1),
                        "move_1d": round(move_1d, 2) if move_1d is not None else None,
                        "move_3d": round(move_3d, 2) if move_3d is not None else None,
                        "move_5d": round(move_5d, 2) if move_5d is not None else None,
                        "direction": "UP" if (move_1d or 0) > 0 else "DOWN",
                        "beat": eps_actual > eps_est,
                    }
                )
            except Exception as e:
                logger.debug("Skipping earnings date %s: %s", date, e)
                continue

        if not reactions:
            return _empty_reaction(
                symbol, "Could not compute reactions from available data"
            )

        moves_1d = [r["move_1d"] for r in reactions if r["move_1d"] is not None]
        moves_3d = [r["move_3d"] for r in reactions if r["move_3d"] is not None]

        avg_move_1d = float(np.mean([abs(m) for m in moves_1d])) if moves_1d else 0.0
        avg_move_3d = float(np.mean([abs(m) for m in moves_3d])) if moves_3d else 0.0
        beat_rate = float(np.mean([r["beat"] for r in reactions])) * 100
        pos_reaction = [r for r in reactions if r["beat"] and (r["move_1d"] or 0) > 0]
        beats = [r for r in reactions if r["beat"]]
        pos_rate = len(pos_reaction) / len(beats) * 100 if beats else 50.0

        typical_low = (
            float(np.percentile(moves_1d, 25))
            if len(moves_1d) >= 4
            else min(moves_1d or [0])
        )
        typical_high = (
            float(np.percentile(moves_1d, 75))
            if len(moves_1d) >= 4
            else max(moves_1d or [0])
        )

        next_earnings = None
        try:
            from trading.data.earnings_calendar import get_upcoming_earnings

            next_earnings = get_upcoming_earnings(symbol)
        except Exception:
            pass

        return {
            "symbol": symbol,
            "reactions": reactions,
            "avg_move_1d": round(avg_move_1d, 2),
            "avg_move_3d": round(avg_move_3d, 2),
            "beat_rate": round(beat_rate, 1),
            "positive_reaction_rate": round(pos_rate, 1),
            "typical_range": (round(typical_low, 2), round(typical_high, 2)),
            "num_quarters": len(reactions),
            "next_earnings": next_earnings,
            "error": None,
        }

    except Exception as e:
        logger.error("Earnings reaction failed for %s: %s", symbol, e)
        return _empty_reaction(symbol, str(e))


def _empty_reaction(symbol: str, reason: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "reactions": [],
        "avg_move_1d": 0.0,
        "avg_move_3d": 0.0,
        "beat_rate": 0.0,
        "positive_reaction_rate": 0.0,
        "typical_range": (0.0, 0.0),
        "num_quarters": 0,
        "next_earnings": None,
        "error": reason,
    }
