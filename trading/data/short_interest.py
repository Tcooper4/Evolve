"""Short interest and squeeze risk pipeline."""

from functools import lru_cache

import yfinance as yf


@lru_cache(maxsize=128)
def get_short_interest(symbol: str) -> dict:
    """Return short interest metrics and a simple squeeze risk score."""
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

        return {
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

