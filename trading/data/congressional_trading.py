"""
Congressional trading signals.
Tracks STOCK Act disclosures from
House and Senate members via the
official disclosure search APIs.
No API key required — public data.
"""

import json
import logging
import time
import urllib.request
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CONGRESS_CACHE: Dict[str, Any] = {}
_CONGRESS_TS: Dict[str, float] = {}
_CONGRESS_TTL = 14400  # 4 hours

HEADERS = {
    "User-Agent": (
        "Evolve Trading Research "
        "contact@evolve-trading.com"
    ),
    "Accept": "application/json",
}


def _fetch_house_disclosures(
    symbol: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch House STOCK Act filings."""
    try:
        _url = (
            "https://efts.house.gov/"
            "LATEST/search-index"
            f"?q=%22{symbol}%22"
            "&dateRange=custom"
            "&fromDate=2024-01-01"
            "&index=ptr"
        )
        _req = urllib.request.Request(_url, headers=HEADERS)
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _data = json.loads(
                _r.read().decode("utf-8", errors="replace")
            )
        _hits = _data.get("hits", {}).get("hits", [])
        results: List[Dict[str, Any]] = []
        for h in _hits[:limit]:
            _s = h.get("_source", {})
            results.append(
                {
                    "source": "house",
                    "filer": _s.get("filer_name", ""),
                    "ticker": symbol,
                    "type": _s.get("asset_description", ""),
                    "transaction": _s.get("type", ""),
                    "amount": _s.get("amount", ""),
                    "date": _s.get("transaction_date", ""),
                    "filed": _s.get("filing_date", ""),
                }
            )
        return results
    except Exception as e:
        logger.debug(
            "House disclosures failed for %s: %s",
            symbol,
            e,
        )
        return []


def _fetch_senate_disclosures(
    symbol: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch Senate STOCK Act filings."""
    try:
        _url = (
            "https://efts.house.gov/"
            "LATEST/search-index"
            f"?q=%22{symbol}%22"
            "&index=ptr&chamber=S"
        )
        _req = urllib.request.Request(_url, headers=HEADERS)
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _data = json.loads(
                _r.read().decode("utf-8", errors="replace")
            )
        _hits = _data.get("hits", {}).get("hits", [])
        results: List[Dict[str, Any]] = []
        for h in _hits[:limit]:
            _s = h.get("_source", {})
            results.append(
                {
                    "source": "senate",
                    "filer": _s.get("filer_name", ""),
                    "ticker": symbol,
                    "type": _s.get("asset_description", ""),
                    "transaction": _s.get("type", ""),
                    "amount": _s.get("amount", ""),
                    "date": _s.get("transaction_date", ""),
                    "filed": _s.get("filing_date", ""),
                }
            )
        return results
    except Exception as e:
        logger.debug(
            "Senate disclosures failed for %s: %s",
            symbol,
            e,
        )
        return []


def get_congressional_trades(symbol: str) -> Dict[str, Any]:
    """
    Returns recent congressional trades
    for a symbol with signal strength.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral(sym)

    _now = time.time()
    if (
        sym in _CONGRESS_CACHE
        and _now - _CONGRESS_TS.get(sym, 0) < _CONGRESS_TTL
    ):
        return _CONGRESS_CACHE[sym]

    try:
        _house = _fetch_house_disclosures(sym)
        _senate = _fetch_senate_disclosures(sym)
        _all = _house + _senate

        if not _all:
            result = _neutral(sym)
            result["success"] = True
            result["message"] = (
                "No recent congressional activity"
            )
        else:
            _buys = sum(
                1
                for t in _all
                if "purchase" in t.get("transaction", "").lower()
                or "buy" in t.get("transaction", "").lower()
            )
            _sells = sum(
                1
                for t in _all
                if "sale" in t.get("transaction", "").lower()
                or "sell" in t.get("transaction", "").lower()
            )
            _total = len(_all)
            _signal = "NEUTRAL"
            _strength = 5.0
            if _buys > _sells * 1.5:
                _signal = "BUY"
                _strength = min(8.0, 5.0 + _buys * 0.5)
            elif _sells > _buys * 1.5:
                _signal = "SELL"
                _strength = max(2.0, 5.0 - _sells * 0.5)

            result = {
                "symbol": sym,
                "trades": _all,
                "total_trades": _total,
                "buys": _buys,
                "sells": _sells,
                "signal": _signal,
                "signal_strength": round(_strength, 1),
                "source": "congressional_disclosure",
                "success": True,
            }

        _CONGRESS_CACHE[sym] = result
        _CONGRESS_TS[sym] = _now
        return result

    except Exception as e:
        logger.debug(
            "Congressional trades failed for %s: %s",
            sym,
            e,
        )
        return _neutral(sym)


def _neutral(sym: str) -> Dict[str, Any]:
    return {
        "symbol": sym,
        "trades": [],
        "total_trades": 0,
        "buys": 0,
        "sells": 0,
        "signal": "NEUTRAL",
        "signal_strength": 5.0,
        "source": "unavailable",
        "success": False,
    }
