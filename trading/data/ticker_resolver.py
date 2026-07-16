"""
Smart ticker resolution.
Tries multiple symbol formats to find
valid price data from yfinance.
Handles indices (^), crypto (-USD),
futures (=F), forex (=X), common
aliases, frequent typos, and near-miss
matches against liquid names.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# True aliases where name != symbol
# Keep this list small — only genuine
# mismatches that auto-probing can't
# catch
_ALIASES = {
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "GSPC": "^GSPC",
    "SPXW": "^GSPC",
    "XSP": "^XSP",
    "DJI": "^DJI",
    "DJIA": "^DJI",
    "DOW": "^DJI",
    "NDX": "^NDX",
    "COMP": "^IXIC",
    "NASDAQ": "^IXIC",
    "RUT": "^RUT",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "TYX": "^TYX",
    "IRX": "^IRX",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "GOLD": "GC=F",
    "XAUUSD": "GC=F",
    "OIL": "CL=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "NATGAS": "NG=F",
    "SILVER": "SI=F",
    "XAGUSD": "SI=F",
    "COPPER": "HG=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    "ZB": "ZB=F",
    "ZN": "ZN=F",
    "ZF": "ZF=F",
}

# Frequent fat-finger typos (edit distance 1 / transposition) for mega-caps.
# Applied in normalize_ticker so the dashboard search bar stays fast.
_COMMON_TYPOS = {
    "APPL": "AAPL",
    "AAPPL": "AAPL",
    "APAL": "AAPL",
    "NVDIA": "NVDA",
    "NVDAA": "NVDA",
    "TSLAA": "TSLA",
    "TSAL": "TSLA",
    "GOOGLL": "GOOGL",
    "AMZNN": "AMZN",
    "MSFTT": "MSFT",
    "METAA": "META",
    "NFLXX": "NFLX",
    "QQQQ": "QQQ",
    "SPYY": "SPY",
}

# Suffixes to probe automatically
# if exact match fails
_PROBE_SUFFIXES = [
    lambda s: f"^{s}",  # index
    lambda s: f"{s}-USD",  # crypto
    lambda s: f"{s}=F",  # futures
    lambda s: f"{s}=X",  # forex
]


def _liquid_universe() -> List[str]:
    """Liquid names for near-miss suggestions (no network)."""
    try:
        from trading.analysis.market_scanner import DEFAULT_UNIVERSE

        return [str(s).upper() for s in DEFAULT_UNIVERSE]
    except Exception as e:
        logger.debug("liquid universe fallback: %s", e)
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            "SPY", "QQQ", "IWM", "AMD", "NFLX", "JPM", "V",
        ]


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance; short-circuit for |len| gap > 1."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 99
    # DP for small strings only
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return int(prev[lb])


def suggest_ticker(
    symbol: str,
    *,
    candidates: Optional[Sequence[str]] = None,
    max_distance: int = 1,
) -> Optional[str]:
    """Best near-miss ticker (distance ≤ max_distance), or None."""
    sym = (symbol or "").strip().upper()
    if not sym or len(sym) < 2:
        return None
    pool = list(candidates) if candidates is not None else _liquid_universe()
    best: Optional[str] = None
    best_d = max_distance + 1
    for cand in pool:
        c = str(cand).strip().upper()
        if not c or c == sym:
            continue
        d = _edit_distance(sym, c)
        if d < best_d:
            best_d = d
            best = c
            if d == 0:
                break
    if best is not None and best_d <= max_distance:
        return best
    return None


def normalize_ticker(symbol: str) -> str:
    """
    Fast normalization without yfinance
    validation. Alias + common-typo lookup.
    Use for search bars and inputs
    where latency matters.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return sym
    if sym in _ALIASES:
        return _ALIASES[sym]
    if sym in _COMMON_TYPOS:
        return _COMMON_TYPOS[sym]
    return sym


def _has_live_price(candidate: str) -> bool:
    try:
        import yfinance as yf

        info = yf.Ticker(candidate).fast_info
        return bool(
            hasattr(info, "last_price")
            and info.last_price
            and float(info.last_price) > 0
        )
    except Exception:
        return False


def resolve_ticker(
    symbol: str,
    validate: bool = True,
) -> str:
    """
    Resolve a user-supplied ticker to
    a valid yfinance symbol.

    Steps:
    1. Normalize (strip, uppercase, aliases, common typos)
    2. If validate=True, probe yfinance
       with suffixes until data found
    3. Near-miss against liquid universe (edit distance ≤ 1)
    4. Return best match or original

    Args:
        symbol: Raw user input
        validate: Whether to probe
            yfinance to confirm data
            exists (adds ~0.5s latency)

    Returns:
        Resolved yfinance symbol string
    """
    raw = (symbol or "").strip().upper()
    if not raw:
        return raw

    sym = normalize_ticker(raw)

    # Already structured (index/crypto/fx/futures)
    if (
        sym.startswith("^")
        or "=" in sym
        or "-" in sym
    ):
        return sym

    if not validate:
        return sym

    # Probe normalized symbol first
    if _has_live_price(sym):
        return sym

    # Probe with suffixes on the normalized form
    for suffix_fn in _PROBE_SUFFIXES:
        candidate = suffix_fn(sym)
        if _has_live_price(candidate):
            logger.debug("Ticker probe: %s -> %s", raw, candidate)
            return candidate

    # Near-miss (e.g. unknown typo not in _COMMON_TYPOS)
    suggestion = suggest_ticker(sym)
    if suggestion and _has_live_price(suggestion):
        logger.info(
            "Ticker near-miss: %s -> %s",
            raw,
            suggestion,
        )
        return suggestion

    # Also try near-miss on the raw pre-normalize form if different
    if raw != sym:
        suggestion = suggest_ticker(raw)
        if suggestion and _has_live_price(suggestion):
            logger.info(
                "Ticker near-miss: %s -> %s",
                raw,
                suggestion,
            )
            return suggestion

    return sym


__all__ = [
    "normalize_ticker",
    "resolve_ticker",
    "suggest_ticker",
]
