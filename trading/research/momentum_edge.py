# -*- coding: utf-8 -*-
"""Equity / ETF momentum edge test (signal-edge program — momentum family).

---------------------------------------------------------------------------
PREDECLARED TRIAL SET (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Two formation windows only — not a wide grid:

  1) formation="12_1"  lookback=252, skip=21
       Classic Jegadeesh & Titman (1993) 12-month-minus-1-month window
       (also the Moskowitz, Ooi & Pedersen 2012 TSMOM formation length).

  2) formation="3_1"   lookback=63,  skip=21
       Shorter ~3-month-minus-1-month window for a monthly-rebalance
       retail horizon (same skip-month to avoid short-term reversal
       contamination; not tuned after peeking at DSR).

Fixed for all trials:
  hold = 21 trading days (one month — literature default hold band)
  step_size = 21 (monthly decision dates; not daily overlapping bets)
  modes:
    - ETF basket SPY/QQQ/IWM: time-series momentum (sign of formation return)
    - Equity basket (10 liquid names, predeclared): cross-sectional demeaned
      formation return (relative strength within the basket)

Why momentum matters here
-------------------------
Vol-selling structures already tested have negative / "steamroller" skew.
Momentum is documented to show *positive* skew and often holds up better
in stress — a potential diversifier, not another copy of short-premium.
This module reports OOS observation skew and high-vol-window behavior
explicitly alongside the DSR gate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trading.research.signal_edge_harness import (
    DISCLOSURE as HARNESS_DISCLOSURE,
    TargetSpec,
    TrialSpec,
    observation_stats,
    run_signal_edge_oos,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

ETF_BASKET: Tuple[str, ...] = ("SPY", "QQQ", "IWM")
# Modest liquid cross-section — locked before run (do not expand after peeking).
EQUITY_BASKET: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "JNJ", "XOM", "WMT",
)

PERIOD = "5y"  # need ~1y warm-up for 12_1 formation
HOLD_DAYS = 21
STEP_SIZE = 21
SKIP_DAYS = 21

PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = (
    TrialSpec(
        params={
            "formation": "12_1",
            "lookback": 252,
            "skip": SKIP_DAYS,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
        },
        label="form_12_1",
    ),
    TrialSpec(
        params={
            "formation": "3_1",
            "lookback": 63,
            "skip": SKIP_DAYS,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
        },
        label="form_3_1",
    ),
)

TRIAL_JUSTIFICATION = (
    "Two trials only: formation 12_1 (lookback=252, skip=21) citing "
    "Jegadeesh & Titman (1993) and Moskowitz, Ooi & Pedersen (2012) TSMOM; "
    "formation 3_1 (lookback=63, skip=21) as the shorter retail monthly-"
    "rebalance analogue with the same skip-month. Hold/step=21 fixed. "
    "Not expanded after inspecting DSR."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " Momentum research: ETF time-series and small-basket cross-sectional "
    "relative strength. Monthly decision dates. Skew / stress diagnostics "
    "are situational awareness for diversification vs vol-selling — not a "
    "second live gate. recommend_live never auto-wires."
)

_XS_PANEL_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}


def _normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cm = {str(c).lower(): c for c in df.columns}
    if "close" in cm and cm["close"] != "Close":
        df = df.rename(columns={cm["close"]: "Close"})
    return df.sort_index()


def _close(history: pd.DataFrame) -> pd.Series:
    hist = _normalize_ohlcv(history)
    return pd.to_numeric(hist["Close"], errors="coerce").dropna()


def formation_return(
    close: pd.Series,
    *,
    lookback: int,
    skip: int,
) -> pd.Series:
    """R from t-lookback to t-skip (classic skip-month formation)."""
    lb = int(lookback)
    sk = int(skip)
    if lb <= sk:
        raise ValueError("lookback must exceed skip")
    # close[t-sk] / close[t-lb] - 1, stamped at t
    return (close.shift(sk) / close.shift(lb)) - 1.0


def tsmom_signal_series(
    history: pd.DataFrame,
    *,
    lookback: int,
    skip: int,
    step_size: int = STEP_SIZE,
    min_bars: Optional[int] = None,
) -> pd.Series:
    """Time-series momentum: sign(formation return) on monthly step dates."""
    close = _close(history)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    form = formation_return(close, lookback=lookback, skip=skip)
    start = int(min_bars) if min_bars is not None else int(lookback) + 5
    step = max(1, int(step_size))
    for i in range(start, len(close), step):
        v = form.iloc[i]
        if not np.isfinite(v) or v == 0.0:
            continue
        out.iloc[i] = float(np.sign(v))
    return out


def build_xs_strength_panel(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    lookback: int,
    skip: int,
) -> pd.DataFrame:
    """Cross-sectional demeaned formation returns (relative strength)."""
    key = (
        int(lookback),
        int(skip),
        tuple(sorted(prices_by_symbol.keys())),
        tuple(
            (s, len(h), str(h.index[-1]) if len(h) else "")
            for s, h in sorted(prices_by_symbol.items())
        ),
    )
    if key in _XS_PANEL_CACHE:
        return _XS_PANEL_CACHE[key]

    forms = {}
    for sym, hist in prices_by_symbol.items():
        close = _close(hist)
        forms[str(sym).upper()] = formation_return(
            close, lookback=lookback, skip=skip
        )
    panel = pd.DataFrame(forms).sort_index()
    # Demean cross-sectionally each day (relative strength)
    demeaned = panel.sub(panel.mean(axis=1), axis=0)
    _XS_PANEL_CACHE[key] = demeaned
    return demeaned


def xs_signal_series(
    symbol: str,
    history: pd.DataFrame,
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    lookback: int,
    skip: int,
    step_size: int = STEP_SIZE,
) -> pd.Series:
    """Sign of demeaned formation return within the predeclared basket."""
    panel = build_xs_strength_panel(
        prices_by_symbol, lookback=lookback, skip=skip
    )
    sym = str(symbol).upper()
    out = pd.Series(np.nan, index=history.index, dtype=float)
    if sym not in panel.columns:
        return out
    strength = panel[sym].reindex(_close(history).index)
    start = int(lookback) + 5
    step = max(1, int(step_size))
    idx = strength.index
    for i in range(start, len(idx), step):
        v = strength.iloc[i]
        if not np.isfinite(v) or v == 0.0:
            continue
        # Map to history index position
        dt = idx[i]
        if dt in out.index:
            out.loc[dt] = float(np.sign(v))
    return out


def make_tsmom_signal_fn():
    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        return tsmom_signal_series(
            history,
            lookback=int(params.get("lookback", 252)),
            skip=int(params.get("skip", SKIP_DAYS)),
            step_size=int(params.get("step_size", STEP_SIZE)),
        )

    return _fn


def make_xs_signal_fn(prices_by_symbol: Mapping[str, pd.DataFrame]):
    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        return xs_signal_series(
            symbol,
            history,
            prices_by_symbol,
            lookback=int(params.get("lookback", 252)),
            skip=int(params.get("skip", SKIP_DAYS)),
            step_size=int(params.get("step_size", STEP_SIZE)),
        )

    return _fn


def load_prices(
    symbols: Sequence[str],
    period: str = PERIOD,
) -> Dict[str, pd.DataFrame]:
    from trading.data.price_cache import get_history

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            hist = get_history(str(sym), period=period)
            if hist is None or hist.empty or len(hist) < 280:
                logger.warning("skip %s: insufficient history", sym)
                continue
            out[str(sym).upper()] = _normalize_ohlcv(hist)
        except Exception as e:
            logger.warning("load %s failed: %s", sym, e)
    return out


def _obs_skew(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return None
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / sd) ** 3))


def _stress_split_stats(
    payoffs: np.ndarray,
    dates: Sequence[pd.Timestamp],
    spy_close: pd.Series,
    hold: int,
) -> Dict[str, Any]:
    """Split OOS payoffs by whether the hold window had elevated SPY vol."""
    if payoffs.size == 0 or len(dates) != payoffs.size:
        return {"high_vol": None, "low_vol": None, "note": "insufficient"}

    spy = spy_close.sort_index()
    rets = spy.pct_change()
    # Realized vol over next ``hold`` days from each decision date
    high_flags = []
    for d in dates:
        try:
            loc = spy.index.get_indexer([pd.Timestamp(d)], method="ffill")[0]
        except Exception:
            high_flags.append(False)
            continue
        if loc < 0 or loc + hold >= len(rets):
            high_flags.append(False)
            continue
        window = rets.iloc[loc + 1 : loc + 1 + hold].dropna()
        rv = float(window.std(ddof=1)) if len(window) > 2 else np.nan
        high_flags.append(bool(np.isfinite(rv) and rv >= float(rets.std(ddof=1))))

    high_flags_a = np.asarray(high_flags, dtype=bool)
    # Align lengths if some dates failed
    n = min(len(high_flags_a), payoffs.size)
    high_flags_a = high_flags_a[:n]
    pay = payoffs[:n]
    hi = pay[high_flags_a]
    lo = pay[~high_flags_a]
    return {
        "high_vol": observation_stats(hi),
        "low_vol": observation_stats(lo),
        "high_vol_skew": _obs_skew(hi),
        "low_vol_skew": _obs_skew(lo),
        "n_high_vol": int(hi.size),
        "n_low_vol": int(lo.size),
    }


def _diversifier_note(
    oos_skew: Optional[float],
    stress: Mapping[str, Any],
) -> str:
    """Compare risk shape to vol-selling (negative / steamroller skew)."""
    parts = [
        "Vol-selling structures previously tested are characteristically "
        "negative-skew / steamroller. Momentum is hypothesized to show "
        "positive skew and better stress behavior (diversifier)."
    ]
    if oos_skew is None:
        parts.append("OOS skew unavailable (too few observations).")
    elif oos_skew > 0.25:
        parts.append(
            f"OOS payoff skew={oos_skew:.3f} is POSITIVE — risk shape differs "
            "from vol-selling in the expected direction."
        )
    elif oos_skew < -0.25:
        parts.append(
            f"OOS payoff skew={oos_skew:.3f} is NEGATIVE — does NOT exhibit "
            "the classic momentum positive-skew diversifier property here."
        )
    else:
        parts.append(
            f"OOS payoff skew={oos_skew:.3f} is near zero — weak evidence of "
            "a diversifying skew profile."
        )

    hi = (stress or {}).get("high_vol") or {}
    lo = (stress or {}).get("low_vol") or {}
    hs = hi.get("sharpe") if isinstance(hi, dict) else None
    ls = lo.get("sharpe") if isinstance(lo, dict) else None
    if hs is not None and ls is not None:
        if float(hs) >= float(ls):
            parts.append(
                f"High-vol-window OOS Sharpe ({hs}) ≥ low-vol ({ls}) — "
                "consistent with stress-resilient momentum behavior."
            )
        else:
            parts.append(
                f"High-vol-window OOS Sharpe ({hs}) < low-vol ({ls}) — "
                "does not show stress outperformance in this sample."
            )
    return " ".join(parts)


def run_momentum_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Full momentum report → ``data/momentum_oos_real.json``."""
    if clear_cache:
        _XS_PANEL_CACHE.clear()

    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "momentum",
        "disclosure": DISCLOSURE,
        "period": period,
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Formation windows and baskets fixed before this run; not "
            "selected after inspecting DSR."
        ),
        "hold_days": HOLD_DAYS,
        "step_size": STEP_SIZE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sections": {},
        "recommend_live": False,
        "error": None,
    }

    sections = (
        ("etf_tsmom", ETF_BASKET, "tsmom"),
        ("equity_xs", EQUITY_BASKET, "xs"),
    )

    any_ok = False
    any_live = False

    for section, basket, mode in sections:
        print(f"=== {section} ===", flush=True)
        prices = load_prices(basket, period=period)
        if len(prices) < 2:
            report["sections"][section] = {
                "success": False,
                "error": "insufficient history",
                "basket": list(basket),
            }
            continue

        if mode == "tsmom":
            sfn = make_tsmom_signal_fn()
        else:
            sfn = make_xs_signal_fn(prices)

        out = run_signal_edge_oos(
            sfn,
            prices,
            signal_name=f"momentum_{section}",
            trials=PREDECLARED_TRIALS,
            trial_justification=TRIAL_JUSTIFICATION,
            target=TargetSpec(kind="forward_return", horizon=HOLD_DAYS),
            universe=list(prices.keys()),
            purge_bars=HOLD_DAYS,
            disclosure=DISCLOSURE,
            min_train_obs=12,
            min_test_obs=10,
            extra={"mode": mode, "basket": list(prices.keys())},
        )

        # Skew / stress diagnostics on champion OOS path
        diversifier: Dict[str, Any] = {}
        if out.get("success") and out.get("champion"):
            champ_params = dict((out.get("champion") or {}).get("params") or {})
            any_hist = next(iter(prices.values()))
            all_dates = pd.to_datetime(_close(any_hist).index).tz_localize(None)
            n = len(all_dates)
            from trading.research.signal_edge_harness import purged_split_indices

            _te, ts = purged_split_indices(n, HOLD_DAYS, train_frac=0.60)
            test_dates = set(all_dates[ts:])

            from trading.research.signal_edge_harness import observation_payoffs

            target = TargetSpec(kind="forward_return", horizon=HOLD_DAYS)
            dated: List[Tuple[pd.Timestamp, float]] = []
            for sym, hist in prices.items():
                try:
                    sig = sfn(sym, hist, champ_params)
                except Exception:
                    continue
                pay = observation_payoffs(pd.Series(sig), hist, target).dropna()
                for dt, val in pay.items():
                    if dt in test_dates:
                        dated.append((pd.Timestamp(dt), float(val)))
            if dated:
                dated.sort(key=lambda x: x[0])
                dts = [d for d, _ in dated]
                pays = np.array([v for _, v in dated], dtype=float)
                oos_skew = _obs_skew(pays)
                spy_hist = prices.get("SPY")
                if spy_hist is None:
                    spy_hist = next(iter(prices.values()))
                spy = _close(spy_hist)
                stress = _stress_split_stats(pays, dts, spy, HOLD_DAYS)
                diversifier = {
                    "oos_skew": oos_skew,
                    "oos_stats_diagnostic": observation_stats(pays),
                    "stress_split": stress,
                    "note": _diversifier_note(oos_skew, stress),
                }

        section_out = dict(out)
        section_out["basket"] = list(prices.keys())
        section_out["mode"] = mode
        section_out["diversifier_vs_vol_selling"] = diversifier
        report["sections"][section] = section_out

        if out.get("success"):
            any_ok = True
        if out.get("recommend_live"):
            any_live = True

        dsr = out.get("deflated_sharpe") or {}
        print(
            {
                "section": section,
                "champion": out.get("champion"),
                "oos": (out.get("test") or {}).get("stats"),
                "dsr": dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
                "recommend_live": out.get("recommend_live"),
                "diversifier": diversifier.get("note"),
            },
            flush=True,
        )

    report["success"] = any_ok
    report["recommend_live"] = any_live
    report["note"] = (
        "At least one momentum section cleared OOS+DSR — still research-only; "
        "not auto-wired. Check diversifier_vs_vol_selling for skew/stress."
        if any_live
        else (
            "Null / not significant on momentum OOS+DSR across predeclared "
            "formations — leave research-only (acceptable). Skew/stress "
            "diagnostics still reported when observations exist."
        )
    )

    path = Path(out_path) if out_path else ROOT / "data" / "momentum_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    print(f"wrote {path}", flush=True)
    return report


__all__ = [
    "ETF_BASKET",
    "EQUITY_BASKET",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "DISCLOSURE",
    "formation_return",
    "tsmom_signal_series",
    "build_xs_strength_panel",
    "make_tsmom_signal_fn",
    "make_xs_signal_fn",
    "run_momentum_oos_real",
]
