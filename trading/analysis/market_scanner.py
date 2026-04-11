"""
Market scanner — screens a stock universe by technical filters
and ranks results by Quick Score (batch data only; no per-ticker LLM).

Designed to run on demand (not continuously) to avoid rate limits.
Uses yfinance batch download for efficiency. Full AI Score is available
in Analyze → Signal breakdown when you select a symbol.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

UNIVERSE_FILES = {
    "sp100": "data/universes/sp100.json",
    "sp500": "data/universes/sp500.json",
    "nasdaq100": "data/universes/nasdaq100.json",
    "russell1000": "data/universes/russell1000.json",
    "russell3000": "data/universes/russell3000.json",
}


def _get_universe(label: str) -> List[str]:
    """
    Resolve a universe label to a ticker list.
    Known JSON universes load from data/universes/; otherwise DEFAULT_UNIVERSE.
    """
    key = (label or "default").lower().strip()
    if key in UNIVERSE_FILES:
        path = Path(UNIVERSE_FILES[key])
        if path.exists():
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    out = [str(x).strip().upper() for x in data if x]
                    if out:
                        return out
            except Exception as e:
                logger.warning("Universe load failed for %s: %s", key, e)
    return list(DEFAULT_UNIVERSE)

SCAN_FILTERS = {
    "momentum": "Price > SMA20 AND 20d return > 2%",
    "oversold": "RSI < 35 AND Price > SMA50 (dip buyers)",
    "breakout": "Price within 2% of 52w high AND volume > 1.5x avg",
    "high_short": "Short squeeze score > 50",
    "insider_buying": "Insider buying signal in last 90d",
    "quick_technical": (
        "Quick Score ⚡ ≥ threshold (technical estimate, instant)"
    ),
    "high_short_score": (
        "Short Quick Score ⬇️ ≥ threshold "
        "(bearish technical estimate, instant)"
    ),
}

_SCAN_AI_MAX_WORKERS = 8


def _score_ticker_ai(
    symbol: str, hist: pd.DataFrame
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Worker: AI score for one symbol (hist pre-sliced, thread-owned copy)."""
    try:
        from trading.analysis.ai_score import compute_ai_score

        ai = compute_ai_score(symbol, hist)
        return symbol, ai
    except Exception as e:
        logger.debug("Scanner AI score failed for %s: %s", symbol, e)
        return symbol, None


def scan_market(
    filters: List[str] = None,
    universe: List[str] = None,
    max_results: int = 20,
    min_quick_score: float = 6.0,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run market scan.

    Args:
        filters: list of filter keys from SCAN_FILTERS, or None for all
        universe: list of tickers, or None for DEFAULT_UNIVERSE
        max_results: cap on returned rows
        progress_callback: callable(completed, total, phase=...) for UI;
            phase is "filter" (per ticker) or "ai" (scoring); optional kwarg.

    Returns:
        dict with:
            results: list of dicts (one per passing stock, sorted by quick_score)
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
        filters = ["quick_technical"]

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
        # Normalize timezone to prevent tz-naive/tz-aware join errors
        if isinstance(raw, pd.DataFrame) and len(raw.index) > 0:
            if hasattr(raw.index, "tz") and raw.index.tz is not None:
                raw = raw.copy()
                raw.index = raw.index.tz_convert(None)
        # tz_localize(None) is naive-only; tz_convert(None) strips tz-aware indexes
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
    pending: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = []
    # UI progress: phase="filter" (per ticker), then phase="ai" (pool scoring).

    def _emit_progress(done: int, tot: int, phase: str) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(done, tot, phase=phase)
        except TypeError:
            try:
                progress_callback(done, tot)
            except Exception:
                pass
        except Exception:
            pass

    for _scan_i, symbol in enumerate(universe):
        try:
            _emit_progress(_scan_i + 1, total, "filter")
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

            vs_sma20_pct = (
                round((last_price / sma20 - 1) * 100, 2) if sma20 else None
            )
            vs_sma50_pct = (
                round((last_price / sma50 - 1) * 100, 2)
                if sma50
                else None
            )
            _vol5 = None
            if len(close) >= 6:
                _rel5 = np.diff(close[-6:]) / np.maximum(
                    close[-6:-1], 1e-12
                )
                _vol5 = float(np.std(_rel5))
            _vol20 = None
            if len(close) >= 21:
                _rel20 = np.diff(close[-21:]) / np.maximum(
                    close[-21:-1], 1e-12
                )
                _vol20 = float(np.std(_rel20))
            vol_expansion = (
                float(_vol5 / _vol20)
                if (
                    _vol5 is not None
                    and _vol20 is not None
                    and _vol20 > 0
                )
                else 1.0
            )
            qs_pre = _quick_score(
                rsi,
                ret_20d,
                vs_sma20_pct,
                vol_ratio,
                pct_from_high,
                vs_sma50=vs_sma50_pct,
                vol_expansion=vol_expansion,
            )
            _sqs = _short_quick_score(
                rsi,
                ret_20d,
                vs_sma20_pct,
                vol_ratio,
                pct_from_high,
                vs_sma50=vs_sma50_pct,
            )

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
                elif f == "quick_technical":
                    if qs_pre < float(min_quick_score):
                        passes = False
                        break
                elif f == "high_short_score":
                    if _sqs < float(min_quick_score):
                        passes = False
                        break

            if not passes:
                continue

            partial = {
                "symbol": symbol,
                "price": round(last_price, 2),
                "change_20d": round(ret_20d, 2),
                "rsi": round(rsi, 1) if rsi is not None else None,
                "vs_sma20": vs_sma20_pct,
                "pct_from_52w_high": round(pct_from_high, 2),
                "volume_ratio": round(vol_ratio, 2),
                "quick_score": qs_pre,
                "short_quick_score": _sqs,
            }
            pending.append((symbol, hist.copy(), partial))

        except Exception as e:
            logger.debug("Scanner: %s failed: %s", symbol, e)
            continue

    # Empty filters => full parallel AI scoring (briefing phase 2, agent tools).
    if len(filters) == 0:
        ai_by_symbol: Dict[str, Any] = {}
        n_pend = len(pending)
        _emit_progress(0, max(1, n_pend), "ai")
        if pending:
            done_ai = 0
            with ThreadPoolExecutor(max_workers=_SCAN_AI_MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(_score_ticker_ai, sym, h): sym
                    for sym, h, _partial in pending
                }
                for fut in as_completed(future_map):
                    sym, ai = fut.result()
                    ai_by_symbol[sym] = ai
                    done_ai += 1
                    _emit_progress(done_ai, n_pend, "ai")
        else:
            _emit_progress(1, 1, "ai")

        for symbol, _hist, partial in pending:
            ai = ai_by_symbol.get(symbol)
            if ai:
                ai_score = float(ai.get("overall_score") or 5.0)
                ai_grade = ai.get("grade", "C")
                signals = ai.get("signals") or []
            else:
                ai_score = 5.0
                ai_grade = "C"
                signals = []
            row = {
                **partial,
                "ai_score": ai_score,
                "ai_grade": ai_grade,
                "signals": signals,
            }
            results.append(row)

        results.sort(key=lambda x: x.get("ai_score") or 0.0, reverse=True)
        return {
            "results": results[:max_results],
            "filters_applied": filters,
            "scanned": total,
            "passed": len(results),
            "scan_time_s": round(time.time() - t0, 1),
            "error": None,
        }

    # Quick Score only (no bulk AI) — Scanner tab / briefing phase 1.
    n_pend = len(pending)
    _emit_progress(0, max(1, n_pend), "ai")
    _emit_progress(n_pend, max(1, n_pend), "ai")

    for symbol, _hist, partial in pending:
        row = {
            **partial,
            "ai_score": None,
            "ai_grade": None,
            "signals": [],
        }
        results.append(row)

    _sort_key = (
        "short_quick_score"
        if filters and "high_short_score" in filters
        else "quick_score"
    )
    results.sort(
        key=lambda x: float(x.get(_sort_key) or 0.0),
        reverse=True,
    )

    return {
        "results": results[:max_results],
        "filters_applied": filters,
        "scanned": total,
        "passed": len(results),
        "scan_time_s": round(time.time() - t0, 1),
        "error": None,
    }


def _rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
    """Wilder RSI (smoothed averages) — aligns with standard charting tools."""
    if len(prices) < period + 1:
        return None
    d = np.diff(prices)
    g = np.where(d > 0, d, 0.0)
    l_ = np.where(d < 0, -d, 0.0)
    # Wilder smoothing (true RSI)
    ag = float(np.mean(g[:period]))
    al = float(np.mean(l_[:period]))
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l_[i]) / period
    if al == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))


def _quick_score(
    rsi: Optional[float],
    ret_20d: float,
    vs_sma20: Optional[float],
    vol_ratio: float,
    pct_from_high: float,
    vs_sma50: Optional[float] = None,
    vol_expansion: float = 1.0,
) -> float:
    """
    Fast technical pre-score (0-10).
    Computed from batch-downloaded data only — no extra API calls.
    Used as a fast alternative to the full AI Score for large universe scans.

    Components (equal weight, 7 total when extras provided):
      RSI position    (0-10)
      20d momentum    (0-10)
      SMA20 position  (0-10)
      Volume ratio    (0-10)
      52w high prox   (0-10)
      SMA50 position  (0-10)
      Vol expansion   (0-10) — short vs medium vol of returns
    """
    scores = []

    # RSI: sweet spot 40-60 neutral,
    # 30-40 oversold opportunity,
    # below 30 very oversold
    if rsi is not None:
        if rsi < 30:
            scores.append(8.0)
        elif rsi < 40:
            scores.append(7.0)
        elif rsi < 60:
            scores.append(6.0)
        elif rsi < 70:
            scores.append(5.0)
        else:
            scores.append(3.0)

    # 20d momentum: positive = good
    if ret_20d > 10:
        scores.append(9.0)
    elif ret_20d > 5:
        scores.append(8.0)
    elif ret_20d > 2:
        scores.append(7.0)
    elif ret_20d > 0:
        scores.append(6.0)
    elif ret_20d > -5:
        scores.append(4.0)
    else:
        scores.append(2.0)

    # Price vs SMA20: above = bullish
    if vs_sma20 is not None:
        if vs_sma20 > 5:
            scores.append(8.0)
        elif vs_sma20 > 2:
            scores.append(7.0)
        elif vs_sma20 > 0:
            scores.append(6.0)
        elif vs_sma20 > -2:
            scores.append(5.0)
        else:
            scores.append(3.0)

    # Volume ratio: higher = more interest/conviction
    if vol_ratio > 2.0:
        scores.append(9.0)
    elif vol_ratio > 1.5:
        scores.append(7.0)
    elif vol_ratio > 1.0:
        scores.append(6.0)
    elif vol_ratio > 0.5:
        scores.append(5.0)
    else:
        scores.append(3.0)

    # 52w high proximity: near high = strength, far from high = weakness
    if pct_from_high > -5:
        scores.append(8.0)
    elif pct_from_high > -10:
        scores.append(7.0)
    elif pct_from_high > -20:
        scores.append(6.0)
    elif pct_from_high > -35:
        scores.append(4.0)
    else:
        scores.append(2.0)

    if vs_sma50 is not None:
        if vs_sma50 > 5:
            scores.append(8.0)
        elif vs_sma50 > 2:
            scores.append(7.0)
        elif vs_sma50 > 0:
            scores.append(6.0)
        elif vs_sma50 > -3:
            scores.append(5.0)
        else:
            scores.append(3.0)

    if ret_20d > 2 and vol_expansion > 1.2:
        scores.append(7.5)
    elif ret_20d < -2 and vol_expansion > 1.2:
        scores.append(3.0)
    else:
        scores.append(5.5)

    if not scores:
        return 5.0
    return round(sum(scores) / len(scores), 1)


def _short_quick_score(
    rsi: Optional[float],
    ret_20d: float,
    vs_sma20: Optional[float],
    vol_ratio: float,
    pct_from_high: float,
    vs_sma50: Optional[float] = None,
) -> float:
    """
    Fast bearish pre-score (0-10).
    Higher = stronger short candidate (inverse of long quick score).
    """
    scores = []

    if rsi is not None:
        if rsi > 75:
            scores.append(9.0)
        elif rsi > 65:
            scores.append(7.0)
        elif rsi > 50:
            scores.append(5.5)
        elif rsi > 35:
            scores.append(4.0)
        else:
            scores.append(2.0)

    if ret_20d < -10:
        scores.append(9.0)
    elif ret_20d < -5:
        scores.append(8.0)
    elif ret_20d < -2:
        scores.append(7.0)
    elif ret_20d < 0:
        scores.append(5.5)
    elif ret_20d < 5:
        scores.append(3.5)
    else:
        scores.append(2.0)

    if vs_sma20 is not None:
        if vs_sma20 < -5:
            scores.append(8.5)
        elif vs_sma20 < -2:
            scores.append(7.0)
        elif vs_sma20 < 0:
            scores.append(5.5)
        elif vs_sma20 < 2:
            scores.append(4.0)
        else:
            scores.append(2.5)

    if ret_20d < 0 and vol_ratio > 1.5:
        scores.append(8.0)
    elif ret_20d > 0 and vol_ratio > 1.5:
        scores.append(3.0)
    else:
        scores.append(5.0)

    if pct_from_high < -30:
        scores.append(7.5)
    elif pct_from_high < -20:
        scores.append(6.5)
    elif pct_from_high < -10:
        scores.append(5.5)
    elif pct_from_high < -5:
        scores.append(4.0)
    else:
        scores.append(2.5)

    if vs_sma50 is not None:
        if vs_sma50 < -5:
            scores.append(8.5)
        elif vs_sma50 < -2:
            scores.append(7.0)
        elif vs_sma50 < 0:
            scores.append(5.5)
        else:
            scores.append(3.0)

    if not scores:
        return 5.0
    return round(sum(scores) / len(scores), 1)


def get_available_filters() -> Dict[str, str]:
    return SCAN_FILTERS.copy()
