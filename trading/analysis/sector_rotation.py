"""
Sector rotation model.
Compares sector ETF performance vs SPY
on 20/60/120 day rolling basis to
identify which sectors have institutional
momentum.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

_ROTATION_CACHE: Dict[str, Any] = {}
_ROTATION_TS: float = 0.0
_ROTATION_TTL = 3600  # 1 hour


def get_sector_rotation() -> Dict[str, Any]:
    """
    Returns sector momentum rankings
    vs SPY for 20d, 60d, 120d periods.
    """
    global _ROTATION_CACHE, _ROTATION_TS
    _now = time.time()
    if _ROTATION_CACHE and _now - _ROTATION_TS < _ROTATION_TTL:
        return _ROTATION_CACHE

    try:
        from trading.data.price_cache import get_history

        results: Dict[str, Any] = {}
        _spy = get_history("SPY", period="1y")
        if _spy.empty:
            return {"error": "SPY unavailable", "sectors": {}}

        _spy_close = (
            _spy["Close"] if "Close" in _spy.columns else _spy["close"]
        )

        for sector, etf in SECTOR_ETFS.items():
            try:
                _df = get_history(etf, period="1y")
                if _df.empty:
                    continue
                _close = (
                    _df["Close"]
                    if "Close" in _df.columns
                    else _df["close"]
                )

                _common = _close.index.intersection(_spy_close.index)
                if len(_common) < 21:
                    continue
                _sc = _close.loc[_common]
                _sp = _spy_close.loc[_common]

                _rel: Dict[str, float] = {}
                for days, label in [
                    (20, "1m"),
                    (60, "3m"),
                    (120, "6m"),
                ]:
                    if len(_sc) > days:
                        _etf_ret = (
                            float(_sc.iloc[-1]) - float(_sc.iloc[-days - 1])
                        ) / float(_sc.iloc[-days - 1])
                        _spy_ret = (
                            float(_sp.iloc[-1]) - float(_sp.iloc[-days - 1])
                        ) / float(_sp.iloc[-days - 1])
                        _rel[label] = round((_etf_ret - _spy_ret) * 100, 2)

                _score = (
                    _rel.get("1m", 0) * 0.5
                    + _rel.get("3m", 0) * 0.3
                    + _rel.get("6m", 0) * 0.2
                )
                results[sector] = {
                    "etf": etf,
                    "relative_returns": _rel,
                    "composite_score": round(_score, 2),
                    "trend": (
                        "outperforming"
                        if _score > 2
                        else "underperforming"
                        if _score < -2
                        else "neutral"
                    ),
                }
            except Exception as _se:
                logger.debug("Sector %s failed: %s", sector, _se)

        _ranked = sorted(
            results.items(),
            key=lambda x: x[1].get("composite_score", 0),
            reverse=True,
        )

        output: Dict[str, Any] = {
            "sectors": results,
            "top_sectors": [s for s, _ in _ranked[:3]],
            "weak_sectors": [s for s, _ in _ranked[-3:]],
            "ranked": [s for s, _ in _ranked],
            "error": None,
        }
        _ROTATION_CACHE = output
        _ROTATION_TS = _now
        return output

    except Exception as e:
        logger.warning("Sector rotation failed: %s", e)
        return {
            "error": str(e),
            "sectors": {},
        }


def get_sector_signal_for_ticker(sector: str) -> Dict[str, Any]:
    """
    Returns sector rotation context
    for a specific ticker's sector.
    Used in AI score to adjust
    fundamental score.
    """
    if not sector:
        return {}
    try:
        rotation = get_sector_rotation()
        if rotation.get("error"):
            return {}
        sec_data = rotation.get("sectors", {}).get(sector, {})
        return {
            "sector": sector,
            "trend": sec_data.get("trend", "neutral"),
            "composite_score": sec_data.get("composite_score", 0),
            "relative_1m": sec_data.get("relative_returns", {}).get("1m", 0),
            "top_sectors": rotation.get("top_sectors", []),
        }
    except Exception:
        return {}
