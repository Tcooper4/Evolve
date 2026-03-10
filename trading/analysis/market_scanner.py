"""
Market scanner — screens a stock universe by technical filters
and ranks results by AI Score.

Designed to run on demand (not continuously) to avoid rate limits.
Uses yfinance batch download for efficiency.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default universe — large/mid cap liquid names
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "XOM", "UNH", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "MRK", "CVX",
    "ABBV", "ORCL", "BAC", "COST", "PEP", "KO", "CSCO", "CRM", "MCD", "NKE",
    "TMO", "ABT", "ACN", "LIN", "DHR", "TXN", "INTC", "QCOM", "AMD", "NFLX",
    "ADBE", "NOW", "INTU", "ISRG", "AMGN", "PLD", "CB", "SPGI", "GS", "BLK",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "HYG", "EEM",
]

SCAN_FILTERS = {
    "momentum": "Price > SMA20 AND 20d return > 2%",
    "oversold": "RSI < 35 AND Price > SMA50 (dip buyers)",
    "breakout": "Price within 2% of 52w high AND volume > 1.5x avg",
    "high_short": "Short squeeze score > 50",
    "insider_buying": "Insider buying signal in last 90d",
    "top_ai_score": "AI Score >= 7.0",
}


def scan_market(
    filters: List[str] = None,
    universe: List[str] = None,
    max_results: int = 20,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run market scan.

    Args:
        filters: list of filter keys from SCAN_FILTERS, or None for all
        universe: list of tickers, or None for DEFAULT_UNIVERSE
        max_results: cap on returned rows
        progress_callback: callable(completed, total) for UI progress bar

    Returns:
        dict with:
            results: list of dicts (one per passing stock, sorted by ai_score)
            filters_applied: list of filter names
            scanned: int — total tickers checked
            passed: int — tickers passing filters
            scan_time_s: float
            error: str | None
    """
    t0 = time.time()

    if universe is None:
        universe = DEFAULT_UNIVERSE
    if filters is None:
        filters = ["top_ai_score"]

    # Batch download price data (much faster than per-ticker)
    try:
        raw = yf.download(
            universe,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        return {
            "results": [],
            "filters_applied": filters,
            "scanned": 0,
            "passed": 0,
            "scan_time_s": 0,
            "error": f"Download failed: {e}",
        }

    # Normalize: single-ticker download returns plain columns; multi returns MultiIndex
    if len(universe) == 1:
        raw = pd.DataFrame(raw)
        if raw.columns.nlevels > 1:
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)

    results = []
    total = len(universe)

    for i, symbol in enumerate(universe):
        try:
            if progress_callback:
                try:
                    progress_callback(i + 1, total)
                except Exception:
                    pass

            # Extract per-ticker data from batch download
            if len(universe) == 1:
                hist = raw.copy()
            else:
                if hasattr(raw.columns, "get_level_values") and raw.columns.nlevels >= 2:
                    if symbol not in raw.columns.get_level_values(0):
                        continue
                    hist = raw[symbol].copy()
                    if isinstance(hist, pd.Series):
                        continue
                    hist = hist.dropna(how="all")
                else:
                    continue

            if hist.empty or len(hist) < 20:
                continue

            # Normalize column names to Capitalized
            hist = hist.copy()
            cols = {c: c if c == c.capitalize() else c.capitalize() for c in hist.columns}
            for k, v in list(cols.items()):
                if k != v and v not in hist.columns:
                    hist = hist.rename(columns={k: v})
            if "Close" not in hist.columns and "close" in hist.columns:
                hist = hist.rename(columns={"close": "Close"})
            if "Volume" not in hist.columns and "volume" in hist.columns:
                hist = hist.rename(columns={"volume": "Volume"})

            close = hist["Close"].values.astype(float)
            volume = hist["Volume"].values.astype(float) if "Volume" in hist.columns else None
            last_price = float(close[-1])

            # Compute indicators needed for filters
            sma20 = float(np.mean(close[-20:])) if len(close) >= 20 else None
            sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else None
            rsi = _rsi(close, 14)
            ret_20d = float((close[-1] / close[-20] - 1) * 100) if len(close) >= 20 else 0.0
            high_52w = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
            pct_from_high = float((last_price / high_52w - 1) * 100)

            avg_vol = float(np.mean(volume[-20:])) if volume is not None and len(volume) >= 20 else None
            vol_ratio = float(volume[-1] / avg_vol) if avg_vol and avg_vol > 0 else 1.0

            # Apply filters
            passes = True
            for f in filters:
                if f == "momentum":
                    if not (sma20 and last_price > sma20 and ret_20d > 2):
                        passes = False
                        break
                elif f == "oversold":
                    if not (rsi is not None and rsi < 35 and sma50 and last_price > sma50):
                        passes = False
                        break
                elif f == "breakout":
                    if not (pct_from_high > -2 and vol_ratio > 1.5):
                        passes = False
                        break
                elif f == "high_short":
                    try:
                        from trading.data.short_interest import get_short_interest
                        si = get_short_interest(symbol)
                        if si.get("short_squeeze_score", 0) <= 50:
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                elif f == "insider_buying":
                    try:
                        from trading.data.insider_flow import get_insider_flow
                        ins = get_insider_flow(symbol)
                        if ins.get("signal") != "INSIDER_BUYING":
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                elif f == "top_ai_score":
                    pass  # scored below after filter pass

            if not passes:
                continue

            # Compute AI Score for passing stocks
            try:
                from trading.analysis.ai_score import compute_ai_score
                ai = compute_ai_score(symbol, hist)
                ai_score = ai.get("overall_score", 5.0)
                ai_grade = ai.get("grade", "C")
            except Exception:
                ai_score = 5.0
                ai_grade = "C"

            if "top_ai_score" in filters and ai_score < 7.0:
                continue

            results.append({
                "symbol": symbol,
                "price": round(last_price, 2),
                "change_20d": round(ret_20d, 2),
                "rsi": round(rsi, 1) if rsi is not None else None,
                "vs_sma20": round((last_price / sma20 - 1) * 100, 2) if sma20 else None,
                "pct_from_52w_high": round(pct_from_high, 2),
                "volume_ratio": round(vol_ratio, 2),
                "ai_score": ai_score,
                "ai_grade": ai_grade,
            })

        except Exception as e:
            logger.debug("Scanner: %s failed: %s", symbol, e)
            continue

    # Sort by AI Score descending
    results.sort(key=lambda x: x["ai_score"], reverse=True)

    return {
        "results": results[:max_results],
        "filters_applied": filters,
        "scanned": total,
        "passed": len(results),
        "scan_time_s": round(time.time() - t0, 1),
        "error": None,
    }


def _rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    d = np.diff(prices)
    g = np.where(d > 0, d, 0.0)
    l_ = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[-period:])
    al = np.mean(l_[-period:])
    if al == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))


def get_available_filters() -> Dict[str, str]:
    return SCAN_FILTERS.copy()
