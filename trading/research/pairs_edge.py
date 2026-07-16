# -*- coding: utf-8 -*-
"""Engle–Granger z-score pairs edge (signal-edge program — pairs family).

---------------------------------------------------------------------------
PREDECLARED (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Fixed pair basket (Gatev-style — no O(n²) discovery on train then trade
those pairs OOS):

  SPY/QQQ, IWM/SPY, GLD/SLV, TLT/HYG

Four trials only (lookback × entry z). Exit / rolling hedge / corr gate
fixed at engine defaults (not swept):

  1) lb60_z2   lookback=60,  z_entry=2.0
  2) lb120_z2  lookback=120, z_entry=2.0
  3) lb252_z2  lookback=252, z_entry=2.0   (engine default lookback)
  4) lb120_z25 lookback=120, z_entry=2.5   (conservative entry)

Fixed for all trials:
  z_exit unused (discrete hold-to-horizon, same as momentum)
  rolling_window = 60 (engine default hedge OLS window)
  z_window = 20 (engine hardcoded spread z rolling window)
  min_correlation = 0.7, max_p_value = 0.05 (engine cointegration gate)
  hold = 10 trading days, step_size = 5 (weekly decisions)
  period = 5y

Signal geometry (mean reversion, matches PairsTradingEngine):
  z > +z_entry → signal = -1  (short leg1 / long leg2)
  z < -z_entry → signal = +1  (long leg1 / short leg2)
  else / failed coint gate → NaN

Payoff path: synthetic Close = cumulative NAV of the return-spread book
  r_spread[t] = r1[t] - beta[t-1] * r2[t]
so harness ``signal * forward_return`` is dollar-neutral pairs P&L.
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
    run_signal_edge_oos,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Fixed ex-ante pairs — do not expand / swap after peeking at DSR.
PREDECLARED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("SPY", "QQQ"),
    ("IWM", "SPY"),
    ("GLD", "SLV"),
    ("TLT", "HYG"),
)

PERIOD = "5y"
HOLD_DAYS = 10
STEP_SIZE = 5
ROLLING_HEDGE = 60
Z_WINDOW = 20
MIN_CORR = 0.7
MAX_P_VALUE = 0.05

PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = (
    TrialSpec(
        params={
            "lookback_period": 60,
            "z_score_threshold": 2.0,
            "rolling_window": ROLLING_HEDGE,
            "z_window": Z_WINDOW,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
            "min_correlation": MIN_CORR,
            "max_p_value": MAX_P_VALUE,
        },
        label="lb60_z2",
    ),
    TrialSpec(
        params={
            "lookback_period": 120,
            "z_score_threshold": 2.0,
            "rolling_window": ROLLING_HEDGE,
            "z_window": Z_WINDOW,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
            "min_correlation": MIN_CORR,
            "max_p_value": MAX_P_VALUE,
        },
        label="lb120_z2",
    ),
    TrialSpec(
        params={
            "lookback_period": 252,
            "z_score_threshold": 2.0,
            "rolling_window": ROLLING_HEDGE,
            "z_window": Z_WINDOW,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
            "min_correlation": MIN_CORR,
            "max_p_value": MAX_P_VALUE,
        },
        label="lb252_z2",
    ),
    TrialSpec(
        params={
            "lookback_period": 120,
            "z_score_threshold": 2.5,
            "rolling_window": ROLLING_HEDGE,
            "z_window": Z_WINDOW,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
            "min_correlation": MIN_CORR,
            "max_p_value": MAX_P_VALUE,
        },
        label="lb120_z25",
    ),
)

TRIAL_JUSTIFICATION = (
    "Four trials only on a fixed ex-ante ETF pair basket "
    "(SPY/QQQ, IWM/SPY, GLD/SLV, TLT/HYG) — no in-sample pair discovery. "
    "Lookbacks {60, 120, 252} × z_entry {2.0, 2.5} with 252/2.0 matching "
    "PairsTradingEngine defaults and 2.5 as the conservative entry; "
    "rolling hedge=60, z_window=20, corr≥0.7, p<0.05 locked to engine. "
    "Hold=10 / step=5 for short-horizon mean-reversion. "
    "Not expanded after inspecting DSR."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " Pairs research: Engle–Granger gate + z-score mean reversion on "
    "predeclared ETF pairs. Synthetic NAV Close carries the return-spread "
    "book so harness forward_return payoffs equal hedged pairs P&L. "
    "recommend_live never auto-wires."
)

_PAIR_FRAME_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_COINT_CACHE: Dict[Tuple[Any, ...], bool] = {}


def pair_key(leg1: str, leg2: str) -> str:
    return f"{str(leg1).upper()}_{str(leg2).upper()}"


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


def rolling_hedge_beta(
    leg1: pd.Series,
    leg2: pd.Series,
    window: int,
) -> pd.Series:
    """Causal OLS slope (cov/var demeaned) over ``window`` bars ending at t."""
    w = max(5, int(window))
    # Rolling cov(leg1,leg2)/var(leg2) ≡ OLS without needing a loop.
    cov = leg1.rolling(w, min_periods=w).cov(leg2)
    var = leg2.rolling(w, min_periods=w).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta


def build_pair_nav_frame(
    leg1_close: pd.Series,
    leg2_close: pd.Series,
    *,
    rolling_window: int = ROLLING_HEDGE,
) -> pd.DataFrame:
    """Synthetic Close = NAV of long-leg1 / short-beta-leg2 return book."""
    s1 = pd.to_numeric(leg1_close, errors="coerce").dropna()
    s2 = pd.to_numeric(leg2_close, errors="coerce").dropna()
    idx = s1.index.intersection(s2.index)
    s1 = s1.loc[idx].sort_index()
    s2 = s2.loc[idx].sort_index()
    if len(s1) < int(rolling_window) + 30:
        return pd.DataFrame()

    beta = rolling_hedge_beta(s1, s2, rolling_window)
    r1 = s1.pct_change()
    r2 = s2.pct_change()
    # Lag beta so day's return uses yesterday's hedge (no same-bar look-ahead).
    r_spread = r1 - beta.shift(1) * r2
    r_spread = r_spread.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    nav = (1.0 + r_spread).cumprod() * 100.0
    spread = s1 - beta * s2
    return pd.DataFrame(
        {
            "Close": nav,
            "Leg1": s1,
            "Leg2": s2,
            "Beta": beta,
            "Spread": spread,
        },
        index=s1.index,
    )


def _coint_gate(
    leg1: pd.Series,
    leg2: pd.Series,
    *,
    lookback: int,
    min_corr: float,
    max_p: float,
    cache_i: Optional[int] = None,
) -> bool:
    """Engle–Granger + correlation gate on the trailing lookback window."""
    lb = int(lookback)
    if len(leg1) < lb or len(leg2) < lb:
        return False
    a = leg1.iloc[-lb:]
    b = leg2.iloc[-lb:]
    if a.isna().any() or b.isna().any():
        return False

    cache_key: Optional[Tuple[Any, ...]] = None
    if cache_i is not None:
        cache_key = (
            int(cache_i),
            lb,
            round(float(min_corr), 4),
            round(float(max_p), 4),
            round(float(a.iloc[-1]), 6),
            round(float(b.iloc[-1]), 6),
            round(float(a.iloc[0]), 6),
            round(float(b.iloc[0]), 6),
        )
        hit = _COINT_CACHE.get(cache_key)
        if hit is not None:
            return hit

    ok = False
    try:
        corr = float(a.corr(b))
        if not np.isfinite(corr) or abs(corr) < float(min_corr):
            ok = False
        else:
            from statsmodels.tsa.stattools import coint

            _stat, p_value, _crit = coint(a, b)
            ok = bool(np.isfinite(p_value) and float(p_value) < float(max_p))
    except Exception as e:
        logger.debug("coint gate failed: %s", e)
        ok = False

    if cache_key is not None:
        _COINT_CACHE[cache_key] = ok
    return ok


def pairs_signal_series(
    history: pd.DataFrame,
    *,
    lookback_period: int,
    z_score_threshold: float,
    z_window: int = Z_WINDOW,
    step_size: int = STEP_SIZE,
    min_correlation: float = MIN_CORR,
    max_p_value: float = MAX_P_VALUE,
) -> pd.Series:
    """Mean-reversion signal on step dates; NaN when flat / gate fails."""
    if history is None or history.empty:
        return pd.Series(dtype=float)
    cm = {str(c).lower(): c for c in history.columns}
    if "spread" not in cm or "leg1" not in cm or "leg2" not in cm:
        return pd.Series(np.nan, index=history.index, dtype=float)

    spread = pd.to_numeric(history[cm["spread"]], errors="coerce")
    leg1 = pd.to_numeric(history[cm["leg1"]], errors="coerce")
    leg2 = pd.to_numeric(history[cm["leg2"]], errors="coerce")
    out = pd.Series(np.nan, index=history.index, dtype=float)

    z_w = max(5, int(z_window))
    mu = spread.rolling(z_w, min_periods=z_w).mean()
    sd = spread.rolling(z_w, min_periods=z_w).std()
    z = (spread - mu) / sd.replace(0.0, np.nan)

    lb = int(lookback_period)
    start = max(lb, z_w, ROLLING_HEDGE) + 5
    step = max(1, int(step_size))
    z_entry = float(z_score_threshold)

    for i in range(start, len(history), step):
        zv = z.iloc[i]
        if not np.isfinite(zv) or abs(float(zv)) < z_entry:
            continue
        # Causal coint gate on trailing window ending at i (inclusive).
        if not _coint_gate(
            leg1.iloc[: i + 1],
            leg2.iloc[: i + 1],
            lookback=lb,
            min_corr=min_correlation,
            max_p=max_p_value,
            cache_i=i,
        ):
            continue
        # Mean reversion: fade the z-score.
        out.iloc[i] = -1.0 if float(zv) > 0 else 1.0
    return out


def make_pairs_signal_fn():
    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        return pairs_signal_series(
            history,
            lookback_period=int(params.get("lookback_period", 252)),
            z_score_threshold=float(params.get("z_score_threshold", 2.0)),
            z_window=int(params.get("z_window", Z_WINDOW)),
            step_size=int(params.get("step_size", STEP_SIZE)),
            min_correlation=float(params.get("min_correlation", MIN_CORR)),
            max_p_value=float(params.get("max_p_value", MAX_P_VALUE)),
        )

    return _fn


def load_leg_prices(
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


def build_pair_universe(
    leg_prices: Mapping[str, pd.DataFrame],
    pairs: Sequence[Tuple[str, str]] = PREDECLARED_PAIRS,
    *,
    rolling_window: int = ROLLING_HEDGE,
) -> Dict[str, pd.DataFrame]:
    """Map pair_key → synthetic NAV frame for harness universe."""
    frames: Dict[str, pd.DataFrame] = {}
    for a, b in pairs:
        sa, sb = str(a).upper(), str(b).upper()
        if sa not in leg_prices or sb not in leg_prices:
            logger.warning("skip pair %s/%s: missing legs", sa, sb)
            continue
        key = (
            sa,
            sb,
            int(rolling_window),
            len(leg_prices[sa]),
            len(leg_prices[sb]),
            str(leg_prices[sa].index[-1]),
            str(leg_prices[sb].index[-1]),
        )
        if key in _PAIR_FRAME_CACHE:
            frames[pair_key(sa, sb)] = _PAIR_FRAME_CACHE[key]
            continue
        frame = build_pair_nav_frame(
            _close(leg_prices[sa]),
            _close(leg_prices[sb]),
            rolling_window=rolling_window,
        )
        if frame.empty or len(frame) < 300:
            logger.warning("skip pair %s/%s: thin aligned history", sa, sb)
            continue
        _PAIR_FRAME_CACHE[key] = frame
        frames[pair_key(sa, sb)] = frame
    return frames


def run_pairs_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Full pairs report → ``data/pairs_oos_real.json``."""
    if clear_cache:
        _PAIR_FRAME_CACHE.clear()
        _COINT_CACHE.clear()

    print("PREDECLARED PAIRS:", list(PREDECLARED_PAIRS), flush=True)
    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "pairs_eg_zscore",
        "disclosure": DISCLOSURE,
        "period": period,
        "predeclared_pairs": [list(p) for p in PREDECLARED_PAIRS],
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Pair basket and trial grid fixed before this run; not selected "
            "after inspecting DSR. No post-hoc pair discovery."
        ),
        "hold_days": HOLD_DAYS,
        "step_size": STEP_SIZE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recommend_live": False,
        "error": None,
    }

    needed: List[str] = []
    for a, b in PREDECLARED_PAIRS:
        needed.extend([a, b])
    legs = load_leg_prices(sorted(set(needed)), period=period)
    pair_prices = build_pair_universe(legs, PREDECLARED_PAIRS)

    if len(pair_prices) < 2:
        report["error"] = "insufficient pair history"
        report["legs_loaded"] = list(legs.keys())
        path = Path(out_path) if out_path else ROOT / "data" / "pairs_oos_real.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["wrote"] = str(path)
        return report

    sfn = make_pairs_signal_fn()
    out = run_signal_edge_oos(
        sfn,
        pair_prices,
        signal_name="pairs_eg_zscore",
        trials=PREDECLARED_TRIALS,
        trial_justification=TRIAL_JUSTIFICATION,
        target=TargetSpec(kind="forward_return", horizon=HOLD_DAYS),
        universe=list(pair_prices.keys()),
        purge_bars=HOLD_DAYS,
        disclosure=DISCLOSURE,
        min_train_obs=20,
        min_test_obs=10,
        extra={
            "pairs": list(pair_prices.keys()),
            "engine_defaults_locked": {
                "rolling_window": ROLLING_HEDGE,
                "z_window": Z_WINDOW,
                "min_correlation": MIN_CORR,
                "max_p_value": MAX_P_VALUE,
            },
        },
        out_path=None,  # write enriched artifact below
    )

    # Fold harness report into top-level pairs artifact (keep pair metadata).
    report.update(out)
    report["predeclared_pairs"] = [list(p) for p in PREDECLARED_PAIRS]
    report["legs_loaded"] = list(legs.keys())
    report["pairs_evaluated"] = list(pair_prices.keys())
    report["hold_days"] = HOLD_DAYS
    report["step_size"] = STEP_SIZE
    report["period"] = period
    report["ordering_note"] = (
        "Pair basket and trial grid fixed before this run; not selected "
        "after inspecting DSR. No post-hoc pair discovery."
    )

    path = Path(out_path) if out_path else ROOT / "data" / "pairs_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)

    dsr = out.get("deflated_sharpe") or {}
    print(
        "champion:",
        (out.get("champion") or {}).get("label"),
        "train_n:",
        ((out.get("champion") or {}).get("train_stats") or {}).get("n"),
        "OOS Sharpe:",
        ((out.get("test") or {}) or {}).get("stats", {}).get("sharpe")
        if isinstance(out.get("test"), dict)
        else None,
        "DSR:",
        dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
        "recommend_live:",
        out.get("recommend_live"),
        flush=True,
    )
    print("note:", out.get("note"), flush=True)
    print("wrote", report["wrote"], flush=True)
    return report


__all__ = [
    "PREDECLARED_PAIRS",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "HOLD_DAYS",
    "STEP_SIZE",
    "pair_key",
    "rolling_hedge_beta",
    "build_pair_nav_frame",
    "pairs_signal_series",
    "make_pairs_signal_fn",
    "load_leg_prices",
    "build_pair_universe",
    "run_pairs_oos_real",
]
