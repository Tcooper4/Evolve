"""
Smart ticker resolution.
Tries multiple symbol formats to find
valid price data from yfinance.
Handles indices (^), crypto (-USD),
futures (=F), forex (=X), and common
aliases automatically.
"""

import logging

logger = logging.getLogger(__name__)

# True aliases where name != symbol
# Keep this list small — only genuine
# mismatches that auto-probing can't
# catch
_ALIASES = {
    "SPX": "^GSPC",
    "SPXW": "^GSPC",
    "XSP": "^XSP",
    "DJI": "^DJI",
    "DJIA": "^DJI",
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

# Suffixes to probe automatically
# if exact match fails
_PROBE_SUFFIXES = [
    lambda s: f"^{s}",  # index
    lambda s: f"{s}-USD",  # crypto
    lambda s: f"{s}=F",  # futures
    lambda s: f"{s}=X",  # forex
]


def resolve_ticker(
    symbol: str,
    validate: bool = True,
) -> str:
    """
    Resolve a user-supplied ticker to
    a valid yfinance symbol.

    Steps:
    1. Normalize (strip, uppercase)
    2. Check alias table
    3. If validate=True, probe yfinance
       with suffixes until data found
    4. Return best match or original

    Args:
        symbol: Raw user input
        validate: Whether to probe
            yfinance to confirm data
            exists (adds ~0.5s latency)

    Returns:
        Resolved yfinance symbol string
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return sym

    # Step 1: Check alias table
    if sym in _ALIASES:
        resolved = _ALIASES[sym]
        logger.debug(
            "Ticker alias: %s -> %s",
            sym,
            resolved,
        )
        return resolved

    # Step 2: If already has ^ or =
    # or -, return as-is
    if (
        sym.startswith("^")
        or "=" in sym
        or "-" in sym
    ):
        return sym

    # Step 3: If validate=False,
    # return as-is (trust the user)
    if not validate:
        return sym

    # Step 4: Probe yfinance with
    # the original symbol first
    try:
        import yfinance as yf

        _t = yf.Ticker(sym)
        _info = _t.fast_info
        # fast_info raises or returns
        # empty if ticker invalid
        if (
            hasattr(_info, "last_price")
            and _info.last_price
            and _info.last_price > 0
        ):
            return sym
    except Exception:
        pass

    # Step 5: Probe with suffixes
    for _suffix_fn in _PROBE_SUFFIXES:
        _candidate = _suffix_fn(sym)
        try:
            import yfinance as yf

            _t = yf.Ticker(_candidate)
            _info = _t.fast_info
            if (
                hasattr(_info, "last_price")
                and _info.last_price
                and _info.last_price > 0
            ):
                logger.debug(
                    "Ticker probe: %s -> %s",
                    sym,
                    _candidate,
                )
                return _candidate
        except Exception:
            continue

    # Step 6: Return original if
    # nothing worked — let yfinance
    # produce its own error
    return sym


def normalize_ticker(symbol: str) -> str:
    """
    Fast normalization without yfinance
    validation. Alias lookup only.
    Use for search bars and inputs
    where latency matters.
    """
    sym = (symbol or "").strip().upper()
    return _ALIASES.get(sym, sym)
