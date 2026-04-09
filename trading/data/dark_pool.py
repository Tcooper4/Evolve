"""
Dark pool / block trade detection.
Uses FINRA OTC Transparency API
(public, no API key required).
Data is delayed ~1 week but sufficient
for institutional flow signals.

Endpoints used:
  https://api.finra.org/data/group/
  otcMarket/name/weeklySummary
"""

import json
import logging
import time
import urllib.request
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DP_CACHE: Dict[str, Any] = {}
_DP_TS: Dict[str, float] = {}
_DP_TTL = 21600  # 6 hours

FINRA_HEADERS = {
    "User-Agent": (
        "Evolve Trading Research "
        "research@evolve-trading.com"
    ),
    "Accept": "application/json",
}


def get_dark_pool_activity(symbol: str) -> Dict[str, Any]:
    """
    Fetches dark pool / OTC block trade
    data for a symbol from FINRA.

    Returns:
        symbol: str
        otc_volume: int — total OTC/dark
            pool shares traded (weekly)
        total_volume: int — total reported
            volume for comparison
        dark_pool_pct: float — OTC as %
            of total volume
        block_trades: int — number of
            block-sized prints
        signal: str — ACCUMULATION /
            NEUTRAL / DISTRIBUTION
        signal_strength: float 0-10
        source: str
        success: bool
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral_dp(sym)

    _now = time.time()
    if (
        sym in _DP_CACHE
        and _now - _DP_TS.get(sym, 0) < _DP_TTL
    ):
        return dict(_DP_CACHE[sym])

    try:
        _url = (
            "https://api.finra.org/data/"
            "group/otcMarket/name/"
            "weeklySummary"
            f"?compareFilters="
            f"issueSymbolIdentifier"
            f"%3D{sym}"
            "&limit=4"
            "&fields=issueSymbolIdentifier"
            ",totalWeeklyShareQuantity"
            ",totalWeeklyTradeCount"
            ",lastUpdateDate"
        )
        _req = urllib.request.Request(_url, headers=FINRA_HEADERS)
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _raw = json.loads(
                _r.read().decode("utf-8", errors="replace")
            )

        if isinstance(_raw, dict):
            _data = _raw.get("data") or _raw.get("items") or []
        elif isinstance(_raw, list):
            _data = _raw
        else:
            _data = []

        if not _data:
            return _neutral_dp(sym)

        _total_otc = 0
        _total_trades = 0
        for _row in _data:
            if not isinstance(_row, dict):
                continue
            _total_otc += int(
                _row.get("totalWeeklyShareQuantity", 0) or 0
            )
            _total_trades += int(
                _row.get("totalWeeklyTradeCount", 0) or 0
            )

        if _total_otc == 0:
            result = _neutral_dp(sym)
            result["success"] = True
            result["source"] = "finra"
            _DP_CACHE[sym] = result
            _DP_TS[sym] = _now
            return result

        _exchange_vol = 0
        try:
            import yfinance as yf

            _hist = yf.Ticker(sym).history(period="1mo")
            if not _hist.empty and "Volume" in _hist.columns:
                _exchange_vol = int(_hist["Volume"].sum())
        except Exception:
            pass

        _total_vol = max(
            _total_otc,
            _exchange_vol + _total_otc,
        )
        _dp_pct = (
            _total_otc / _total_vol * 100 if _total_vol > 0 else 0.0
        )

        _signal = "NEUTRAL"
        _strength = 5.0

        if _dp_pct > 45:
            _signal = "ACCUMULATION"
            _strength = min(
                8.5,
                5.0 + (_dp_pct - 45) / 10,
            )
        elif _dp_pct > 30:
            _signal = "ACCUMULATION"
            _strength = 6.5
        elif _dp_pct < 10:
            _signal = "DISTRIBUTION"
            _strength = 4.0

        result = {
            "symbol": sym,
            "otc_volume": _total_otc,
            "total_volume": _total_vol,
            "dark_pool_pct": round(_dp_pct, 1),
            "block_trades": _total_trades,
            "signal": _signal,
            "signal_strength": round(_strength, 1),
            "source": "finra_otc",
            "success": True,
        }
        _DP_CACHE[sym] = result
        _DP_TS[sym] = _now
        return result

    except Exception as e:
        logger.debug(
            "Dark pool fetch failed for %s: %s",
            sym,
            e,
        )
        return _neutral_dp(sym)


def _neutral_dp(sym: str) -> Dict[str, Any]:
    return {
        "symbol": sym,
        "otc_volume": 0,
        "total_volume": 0,
        "dark_pool_pct": 0.0,
        "block_trades": 0,
        "signal": "NEUTRAL",
        "signal_strength": 5.0,
        "source": "unavailable",
        "success": False,
    }
