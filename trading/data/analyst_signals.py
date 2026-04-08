"""
Analyst signals pipeline.
Tracks analyst rating changes, price target revisions, and EPS estimate
momentum via yfinance.
No API key required.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

_ANALYST_CACHE: Dict[str, Any] = {}
_ANALYST_CACHE_TS: Dict[str, float] = {}
_ANALYST_TTL = 3600  # 1 hour


def get_analyst_signals(symbol: str) -> Dict[str, Any]:
    """
    Returns analyst consensus data:
    - recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
    - n_analysts: number of analysts
    - target_mean: mean price target
    - target_high: high price target
    - target_low: low price target
    - upside_pct: % to mean target from current price
    - eps_current_year: current year EPS estimate
    - eps_next_year: next year estimate
    - eps_growth_pct: implied growth
    - signal_strength: float 0-10
    - signal: BUY / NEUTRAL / SELL
    - source: "yfinance_analyst"
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral_analyst(sym)

    _now = time.time()
    if sym in _ANALYST_CACHE and _now - _ANALYST_CACHE_TS.get(sym, 0) < _ANALYST_TTL:
        return _ANALYST_CACHE[sym]

    try:
        import yfinance as yf

        t = yf.Ticker(sym)
        info = t.info or {}

        rec = str(info.get("recommendationKey", "") or "").lower()
        n = info.get("numberOfAnalystOpinions", 0) or 0
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")

        upside = None
        if target_mean is not None and current is not None:
            try:
                upside = (float(target_mean) - float(current)) / float(current) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                upside = None

        eps_cur = info.get("epsCurrentYear")
        eps_next = info.get("forwardEps") or info.get("epsForward")
        eps_growth = None
        if eps_cur is not None and eps_next is not None:
            try:
                ec = float(eps_cur)
                en = float(eps_next)
                if ec != 0:
                    eps_growth = (en - ec) / abs(ec) * 100
            except (TypeError, ValueError):
                eps_growth = None

        strength = 5.0
        signal = "NEUTRAL"

        rec_map = {
            "strong_buy": (8.5, "BUY"),
            "buy": (7.0, "BUY"),
            "hold": (5.0, "NEUTRAL"),
            "underperform": (3.0, "SELL"),
            "sell": (2.0, "SELL"),
            "strong_sell": (1.0, "SELL"),
        }
        if rec in rec_map:
            strength, signal = rec_map[rec]

        if upside is not None:
            if upside > 20:
                strength = min(10.0, strength + 1.0)
            elif upside < -10:
                strength = max(0.0, strength - 1.5)
                signal = "SELL"

        if eps_growth is not None:
            if eps_growth > 15:
                strength = min(10.0, strength + 0.5)
            elif eps_growth < -10:
                strength = max(0.0, strength - 0.5)

        result = {
            "symbol": sym,
            "recommendation": rec,
            "n_analysts": n,
            "target_mean": target_mean,
            "target_high": target_high,
            "target_low": target_low,
            "upside_pct": upside,
            "eps_current_year": eps_cur,
            "eps_next_year": eps_next,
            "eps_growth_pct": eps_growth,
            "signal_strength": round(strength, 2),
            "signal": signal,
            "source": "yfinance_analyst",
            "success": True,
        }

        _ANALYST_CACHE[sym] = result
        _ANALYST_CACHE_TS[sym] = _now
        return result

    except Exception as e:
        logger.debug("Analyst signals failed for %s: %s", sym, e)
        return _neutral_analyst(sym)


def _neutral_analyst(sym: str) -> Dict[str, Any]:
    return {
        "symbol": sym,
        "recommendation": "hold",
        "n_analysts": 0,
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "upside_pct": None,
        "eps_growth_pct": None,
        "signal_strength": 5.0,
        "signal": "NEUTRAL",
        "source": "unavailable",
        "success": False,
    }
