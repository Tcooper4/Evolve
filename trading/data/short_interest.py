"""Short interest and squeeze risk pipeline."""

import time

import yfinance as yf

# BUG FIX: this used @lru_cache with no TTL, so short-interest data was
# frozen at first fetch for the LIFETIME of the process. Fine for a
# short-lived local run; on a hosted site that runs for weeks it meant
# permanently stale squeeze scores. Replaced with the same TTL cache
# pattern the sibling data modules use (6h - the underlying FINRA data
# updates biweekly, but a long-lived process must still refresh).
_SI_CACHE: dict = {}
_SI_TS: dict = {}
_SI_TTL = 6 * 3600.0


def get_short_interest(symbol: str) -> dict:
    """Return short interest metrics and a simple squeeze risk score."""
    _now = time.time()
    if symbol in _SI_CACHE and _now - _SI_TS.get(symbol, 0) < _SI_TTL:
        return dict(_SI_CACHE[symbol])
    try:
        info = yf.Ticker(symbol).info or {}

        short_ratio = info.get("shortRatio")
        short_pct = info.get("shortPercentOfFloat") or 0

        # Compute simple squeeze score: blend of days-to-cover and % of float short
        # Cap each component at 50 so combined max is 100.
        score = min(
            100,
            min(50, (short_ratio or 0) * 5)
            + min(50, (short_pct * 100 * 2) if short_pct else 0),
        )

        result = {
            "symbol": symbol,
            "short_ratio": short_ratio,
            "short_pct_float": round(short_pct * 100, 2) if short_pct else None,
            "shares_short": info.get("sharesShort"),
            "shares_float": info.get("floatShares"),
            "short_squeeze_score": round(score, 1),
            "signal": (
                "HIGH_SHORT"
                if score >= 60
                else ("MODERATE" if score >= 30 else "LOW_SHORT")
            ),
        }
        _SI_CACHE[symbol] = dict(result)
        _SI_TS[symbol] = _now
        return result

    except Exception as e:  # pragma: no cover - defensive fallback
        return {
            "symbol": symbol,
            "short_ratio": None,
            "short_pct_float": None,
            "shares_short": None,
            "shares_float": None,
            "short_squeeze_score": 0,
            "signal": "UNKNOWN",
            "error": str(e),
        }

