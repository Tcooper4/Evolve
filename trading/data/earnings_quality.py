"""
Earnings quality signals.

1. Earnings revision momentum:
   Detects whether EPS estimates are
   being revised up or down vs the
   earnings surprise trend.
   Academic basis: Hawkins, Chamberlin
   & Daniel (1984) — analyst revision
   momentum persists 60-90 days.

2. Accruals anomaly:
   High accounting accruals (earnings
   driven by non-cash items) predict
   underperformance.
   Academic basis: Sloan (1996).
   Formula: Accruals = (Net Income -
   Operating Cash Flow) / Total Assets
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_EQ_CACHE: Dict[str, Any] = {}
_EQ_TS: Dict[str, float] = {}
_EQ_TTL = 14400  # 4 hours


def get_earnings_quality(symbol: str) -> Dict[str, Any]:
    """
    Returns:
        symbol: str
        revision_signal: UP / DOWN / NEUTRAL
        revision_score: float — magnitude
            of revision trend 0-10
        accruals_ratio: float — net income
            minus operating cash flow
            divided by total assets
        accruals_signal: HIGH / NORMAL / LOW
        beat_rate: float — % of recent
            quarters where EPS beat
        beat_streak: int — consecutive
            beats (positive) or misses
            (negative)
        composite_score: float 0-10
            (higher = better quality)
        signal: QUALITY / NEUTRAL / CONCERN
        success: bool
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral_eq(sym)

    _now = time.time()
    if sym in _EQ_CACHE and _now - _EQ_TS.get(sym, 0) < _EQ_TTL:
        return dict(_EQ_CACHE[sym])

    try:
        import numpy as np
        import yfinance as yf

        t = yf.Ticker(sym)

        revision_signal = "NEUTRAL"
        revision_score = 5.0
        beat_rate = 0.5
        beat_streak = 0

        try:
            _hist = getattr(t, "earnings_history", None)
            if _hist is not None and not getattr(_hist, "empty", True):
                if len(_hist) >= 3:
                    try:
                        _hist = _hist.sort_index(ascending=False)
                    except Exception:
                        _hist = _hist.iloc[::-1]

                    _beats = []
                    _surprises = []
                    for _, row in _hist.iterrows():
                        _act = row.get("epsActual")
                        _est = row.get("epsEstimate")
                        if (
                            _act is not None
                            and _est is not None
                            and float(_est) != 0
                        ):
                            _beat = float(_act) > float(_est)
                            _beats.append(_beat)
                            _surp = (
                                (float(_act) - float(_est))
                                / abs(float(_est))
                                * 100
                            )
                            _surprises.append(_surp)

                    if _beats:
                        beat_rate = sum(_beats) / len(_beats)
                        _streak = 0
                        for _b in _beats:
                            if _b:
                                _streak += 1
                            else:
                                break
                        if _streak == 0:
                            for _b in _beats:
                                if not _b:
                                    _streak -= 1
                                else:
                                    break
                        beat_streak = _streak

                    if len(_surprises) >= 3:
                        _recent = float(np.mean(_surprises[:2]))
                        _older = float(np.mean(_surprises[2:]))
                        _trend = _recent - _older
                        if _trend > 5:
                            revision_signal = "UP"
                            revision_score = min(
                                8.5,
                                6.0 + _trend / 10,
                            )
                        elif _trend < -5:
                            revision_signal = "DOWN"
                            revision_score = max(
                                1.5,
                                4.0 + _trend / 10,
                            )
        except Exception as _re:
            logger.debug(
                "Earnings history failed %s: %s",
                sym,
                _re,
            )

        accruals_ratio = 0.0
        accruals_signal = "NORMAL"

        try:
            _cf = t.cashflow
            _inc = t.income_stmt
            _bal = t.balance_sheet

            if (
                _cf is not None
                and not _cf.empty
                and _inc is not None
                and not _inc.empty
                and _bal is not None
                and not _bal.empty
            ):
                _ocf = None
                for _k in [
                    "Operating Cash Flow",
                    "Total Cash From Operating Activities",
                    "Cash From Operations",
                ]:
                    if _k in _cf.index:
                        _ocf = float(_cf.loc[_k].iloc[0])
                        break

                _ni = None
                for _k in [
                    "Net Income",
                    "Net Income Common Stockholders",
                ]:
                    if _k in _inc.index:
                        _ni = float(_inc.loc[_k].iloc[0])
                        break

                _ta = None
                for _k in [
                    "Total Assets",
                ]:
                    if _k in _bal.index:
                        _ta = float(_bal.loc[_k].iloc[0])
                        break

                if (
                    _ocf is not None
                    and _ni is not None
                    and _ta is not None
                    and abs(_ta) > 0
                ):
                    accruals_ratio = (_ni - _ocf) / _ta
                    if accruals_ratio > 0.08:
                        accruals_signal = "HIGH"
                    elif accruals_ratio < -0.05:
                        accruals_signal = "LOW"
        except Exception as _ae:
            logger.debug(
                "Accruals calc failed %s: %s",
                sym,
                _ae,
            )

        _score = 5.0
        _score += (beat_rate - 0.5) * 4
        if beat_streak >= 3:
            _score += 1.0
        elif beat_streak <= -2:
            _score -= 1.5
        if revision_signal == "UP":
            _score += 1.0
        elif revision_signal == "DOWN":
            _score -= 1.5
        if accruals_signal == "HIGH":
            _score -= 1.0
        elif accruals_signal == "LOW":
            _score += 0.5
        _score = max(0.0, min(10.0, _score))

        if _score >= 7.0:
            _signal = "QUALITY"
        elif _score <= 4.0:
            _signal = "CONCERN"
        else:
            _signal = "NEUTRAL"

        result = {
            "symbol": sym,
            "revision_signal": revision_signal,
            "revision_score": round(revision_score, 2),
            "accruals_ratio": round(accruals_ratio, 4),
            "accruals_signal": accruals_signal,
            "beat_rate": round(beat_rate, 3),
            "beat_streak": beat_streak,
            "composite_score": round(_score, 2),
            "signal": _signal,
            "success": True,
        }
        _EQ_CACHE[sym] = result
        _EQ_TS[sym] = _now
        return result

    except Exception as e:
        logger.debug(
            "Earnings quality failed %s: %s",
            sym,
            e,
        )
        return _neutral_eq(sym)


def _neutral_eq(sym: str) -> Dict[str, Any]:
    return {
        "symbol": sym,
        "revision_signal": "NEUTRAL",
        "revision_score": 5.0,
        "accruals_ratio": 0.0,
        "accruals_signal": "NORMAL",
        "beat_rate": 0.5,
        "beat_streak": 0,
        "composite_score": 5.0,
        "signal": "NEUTRAL",
        "success": False,
    }


_BREADTH_CACHE: Dict[str, Any] = {}
_BREADTH_TS: float = 0.0
_BREADTH_TTL = 86400.0


def get_revision_breadth(
    sample_size: int = 150,
) -> Dict[str, Any]:
    """
    EPS revision breadth across an S&P 500 sample.
    Cached 24 hours in-process.
    """
    import random
    import time as _time
    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )

    global _BREADTH_CACHE, _BREADTH_TS

    _now = _time.time()
    if _BREADTH_CACHE and _now - _BREADTH_TS < _BREADTH_TTL:
        return dict(_BREADTH_CACHE)

    try:
        from trading.analysis.market_scanner import _get_universe

        uni = _get_universe("sp500")
        random.seed(42)
        sample: List[str] = random.sample(
            uni,
            min(sample_size, len(uni)),
        )

        results: List[str] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {
                ex.submit(get_earnings_quality, sym): sym
                for sym in sample
            }
            for fut in as_completed(futs):
                try:
                    r = fut.result(timeout=30)
                    if r.get("success"):
                        results.append(
                            r.get(
                                "revision_signal",
                                "NEUTRAL",
                            )
                        )
                except Exception:
                    pass

        if not results:
            return _neutral_breadth()

        n = len(results)
        n_up = results.count("UP")
        n_down = results.count("DOWN")
        pct_up = round(n_up / n * 100, 1)
        pct_down = round(n_down / n * 100, 1)
        pct_neut = round(100.0 - pct_up - pct_down, 1)
        score = max(
            0.0,
            min(
                10.0,
                5.0 + (n_up - n_down) / n * 5.0,
            ),
        )
        signal = (
            "POSITIVE" if pct_up > pct_down + 10
            else "NEGATIVE" if pct_down > pct_up + 10
            else "NEUTRAL"
        )
        result = {
            "pct_up": pct_up,
            "pct_down": pct_down,
            "pct_neutral": pct_neut,
            "breadth_score": round(score, 2),
            "signal": signal,
            "sample_size": n,
            "description": (
                f"{pct_up:.0f}% of {n} sampled stocks have upward "
                f"EPS revisions, {pct_down:.0f}% downward"
            ),
            "success": True,
        }
        _BREADTH_CACHE = result
        _BREADTH_TS = _now
        return result
    except Exception as e:
        logger.debug("Revision breadth failed: %s", e)
        return _neutral_breadth()


def _neutral_breadth() -> Dict[str, Any]:
    return {
        "pct_up": 33.3,
        "pct_down": 33.3,
        "pct_neutral": 33.3,
        "breadth_score": 5.0,
        "signal": "NEUTRAL",
        "sample_size": 0,
        "description": "Revision breadth unavailable",
        "success": False,
    }
