# -*- coding: utf-8 -*-
"""Risk-parity vs 1/N vs conditional-vol overlay — signal-edge Phase 2.

---------------------------------------------------------------------------
PREDECLARED (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Three construction methods only on a fixed ETF basket
(SPY / QQQ / IWM / EFA / TLT — same as conditional_vol_sizing default):

  1) equal_weight     — classic 1/N (DeMiguel, Garlappi & Uppal 2009)
  2) risk_parity      — PortfolioOptimizer._simple_risk_parity on a causal
                        252d covariance window (CVXPY path is non-DCP /
                        unstable; simple inverse-vol seed is the production
                        fallback and the honest research path)
  3) conditional_vol  — 1/N base returns × conditional_vol_multiplier
                        (high-vol cut to 0.50) on the portfolio NAV path

Fixed for all trials:
  lookback_cov = 252, rebalance / hold / step = 21, purge = 21,
  train_frac = 0.60, period = 5y

Harness adaptation (pairs_edge precedent): synthetic Close = portfolio NAV;
signal = +1 on monthly step dates so payoff = forward period return.
Max drawdown is reported as a post-harness diagnostic (not in harness stats).

Honest null (risk parity does not beat 1/N on OOS Sharpe/DSR) is acceptable.
"""

from __future__ import annotations

import json
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
    purged_split_indices,
    run_signal_edge_oos,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

BASKET: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "TLT")
PERIOD = "5y"
LOOKBACK_COV = 252
HOLD_DAYS = 21
STEP_SIZE = 21

PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = (
    TrialSpec(
        params={
            "method": "equal_weight",
            "lookback": LOOKBACK_COV,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
        },
        label="equal_weight",
    ),
    TrialSpec(
        params={
            "method": "risk_parity",
            "lookback": LOOKBACK_COV,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
        },
        label="risk_parity",
    ),
    TrialSpec(
        params={
            "method": "conditional_vol",
            "lookback": LOOKBACK_COV,
            "hold": HOLD_DAYS,
            "step_size": STEP_SIZE,
        },
        label="conditional_vol",
    ),
)

TRIAL_JUSTIFICATION = (
    "Three construction methods only on fixed ETF basket "
    "SPY/QQQ/IWM/EFA/TLT: equal_weight (1/N baseline), risk_parity "
    "(causal 252d covariance via PortfolioOptimizer simple risk-parity "
    "fallback — the production path when CVXPY fails DCP), and "
    "conditional_vol (1/N returns × high-vol exposure cut 0.50 from "
    "conditional_vol_sizing). Monthly rebalance hold/step/purge=21. "
    "Not expanded after inspecting DSR."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " Portfolio-construction research: synthetic NAV Close carries "
    "rebalanced basket P&L so harness forward_return payoffs equal period "
    "portfolio returns. Max drawdown is a diagnostic outside the harness. "
    "recommend_live never auto-wires."
)

_NAV_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}


def portfolio_key(method: str) -> str:
    return f"PORTFOLIO_{str(method).strip().lower()}"


def _normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cm = {str(c).lower(): c for c in df.columns}
    if "close" in cm and cm["close"] != "Close":
        df = df.rename(columns={cm["close"]: "Close"})
    return df.sort_index()


def _close(history: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(_normalize_ohlcv(history)["Close"], errors="coerce").dropna()


def load_basket_prices(
    symbols: Sequence[str] = BASKET,
    period: str = PERIOD,
) -> Dict[str, pd.DataFrame]:
    from trading.data.price_cache import get_history

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            hist = get_history(str(sym), period=period)
            if hist is None or hist.empty or len(hist) < LOOKBACK_COV + 40:
                logger.warning("skip %s: insufficient history", sym)
                continue
            out[str(sym).upper()] = _normalize_ohlcv(hist)
        except Exception as e:
            logger.warning("load %s failed: %s", sym, e)
    return out


def aligned_close_panel(
    prices: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    cols = {}
    for sym, hist in prices.items():
        cols[str(sym).upper()] = _close(hist)
    panel = pd.DataFrame(cols).sort_index().dropna(how="any")
    return panel


def equal_weights(columns: Sequence[str]) -> Dict[str, float]:
    cols = [str(c) for c in columns]
    if not cols:
        return {}
    w = 1.0 / len(cols)
    return {c: w for c in cols}


def risk_parity_weights(returns_window: pd.DataFrame) -> Dict[str, float]:
    """Causal risk-parity weights via the simple (stable) optimizer path."""
    from trading.optimization.portfolio_optimizer import PortfolioOptimizer

    if returns_window is None or returns_window.empty or returns_window.shape[1] < 2:
        return equal_weights(list(returns_window.columns) if returns_window is not None else [])

    opt = PortfolioOptimizer()
    # Use the documented production fallback directly — CVXPY risk-parity
    # objective is non-DCP and routinely fails into this path anyway.
    result = opt._simple_risk_parity(returns_window, None, "volatility")
    if not isinstance(result, dict) or result.get("error") or "weights" not in result:
        return equal_weights(list(returns_window.columns))
    weights = result["weights"]
    if not isinstance(weights, dict) or not weights:
        return equal_weights(list(returns_window.columns))
    # Normalize defensively
    total = float(sum(float(v) for v in weights.values()))
    if total <= 0:
        return equal_weights(list(returns_window.columns))
    return {str(k): float(v) / total for k, v in weights.items()}


def _max_drawdown(equity: np.ndarray) -> Optional[float]:
    arr = np.asarray(equity, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak > 0, peak, np.nan)
    m = float(np.nanmin(dd))
    return None if not np.isfinite(m) else round(m, 6)


def build_portfolio_nav(
    close_panel: pd.DataFrame,
    *,
    method: str,
    lookback: int = LOOKBACK_COV,
    step_size: int = STEP_SIZE,
) -> pd.DataFrame:
    """Synthetic Close = cumulative NAV of the chosen construction method."""
    method_l = str(method).strip().lower()
    panel = close_panel.sort_index()
    if panel.empty or panel.shape[1] < 2:
        return pd.DataFrame()

    rets = panel.pct_change()
    n = len(panel)
    lb = max(30, int(lookback))
    step = max(1, int(step_size))
    daily = np.zeros(n, dtype=float)
    weights = equal_weights(list(panel.columns))

    for i in range(1, n):
        if i >= lb and ((i - lb) % step == 0 or i == lb):
            window = rets.iloc[i - lb : i].dropna(how="any")
            if len(window) >= max(40, lb // 3):
                if method_l == "risk_parity":
                    weights = risk_parity_weights(window)
                else:
                    # equal_weight and conditional_vol share 1/N base weights
                    weights = equal_weights(list(panel.columns))

        r_row = rets.iloc[i]
        port_r = 0.0
        for col, w in weights.items():
            v = r_row.get(col)
            if v is not None and np.isfinite(v):
                port_r += float(w) * float(v)
        daily[i] = port_r

    if method_l == "conditional_vol":
        from trading.portfolio.conditional_vol_sizing import (
            conditional_vol_multiplier,
        )

        # Causal overlay: scale each day's 1/N return by vol multiplier
        # computed on the *unscaled* EW NAV path up to t (no look-ahead).
        ew_nav = np.cumprod(1.0 + daily) * 100.0
        ew_series = pd.Series(ew_nav, index=panel.index)
        scaled = np.zeros(n, dtype=float)
        for i in range(1, n):
            if i < lb:
                scaled[i] = daily[i]
                continue
            info = conditional_vol_multiplier(ew_series.iloc[: i + 1])
            m = float(info.get("multiplier") or 1.0)
            scaled[i] = daily[i] * m
        daily = scaled

    nav = np.cumprod(1.0 + daily) * 100.0
    # Flat until warm-up so early forward returns aren't garbage
    nav[:lb] = 100.0
    return pd.DataFrame({"Close": nav}, index=panel.index)


def build_portfolio_universe(
    prices: Mapping[str, pd.DataFrame],
    *,
    lookback: int = LOOKBACK_COV,
    step_size: int = STEP_SIZE,
) -> Dict[str, pd.DataFrame]:
    panel = aligned_close_panel(prices)
    if panel.empty or len(panel) < LOOKBACK_COV + 60:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for trial in PREDECLARED_TRIALS:
        method = str(trial.params.get("method"))
        key = (
            method,
            int(lookback),
            int(step_size),
            tuple(panel.columns),
            len(panel),
            str(panel.index[-1]),
        )
        if key in _NAV_CACHE:
            out[portfolio_key(method)] = _NAV_CACHE[key]
            continue
        frame = build_portfolio_nav(
            panel, method=method, lookback=lookback, step_size=step_size
        )
        if frame.empty:
            continue
        _NAV_CACHE[key] = frame
        out[portfolio_key(method)] = frame
    return out


def portfolio_signal_series(
    symbol: str,
    history: pd.DataFrame,
    params: Mapping[str, Any],
) -> pd.Series:
    """+1 on monthly step dates only for the matching PORTFOLIO_* symbol."""
    method = str(params.get("method", "")).strip().lower()
    out = pd.Series(np.nan, index=history.index, dtype=float)
    if portfolio_key(method).upper() != str(symbol).upper():
        return out
    step = max(1, int(params.get("step_size", STEP_SIZE)))
    lb = max(30, int(params.get("lookback", LOOKBACK_COV)))
    for i in range(lb, len(history), step):
        out.iloc[i] = 1.0
    return out


def make_portfolio_signal_fn():
    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        return portfolio_signal_series(symbol, history, params)

    return _fn


def run_risk_parity_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Full construction-method report → ``data/risk_parity_oos_real.json``."""
    if clear_cache:
        _NAV_CACHE.clear()

    print("PREDECLARED BASKET:", list(BASKET), flush=True)
    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "portfolio_construction",
        "disclosure": DISCLOSURE,
        "period": period,
        "basket": list(BASKET),
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Methods and basket fixed before this run; not selected after "
            "inspecting DSR. Honest null vs 1/N is acceptable."
        ),
        "hold_days": HOLD_DAYS,
        "step_size": STEP_SIZE,
        "lookback_cov": LOOKBACK_COV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recommend_live": False,
        "error": None,
    }

    prices = load_basket_prices(BASKET, period=period)
    if len(prices) < 3:
        report["error"] = "insufficient basket history"
        path = Path(out_path) if out_path else ROOT / "data" / "risk_parity_oos_real.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["wrote"] = str(path)
        return report

    universe = build_portfolio_universe(prices)
    if len(universe) < 3:
        report["error"] = "failed to build portfolio NAVs"
        path = Path(out_path) if out_path else ROOT / "data" / "risk_parity_oos_real.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["wrote"] = str(path)
        return report

    sfn = make_portfolio_signal_fn()
    out = run_signal_edge_oos(
        sfn,
        universe,
        signal_name="portfolio_construction",
        trials=PREDECLARED_TRIALS,
        trial_justification=TRIAL_JUSTIFICATION,
        target=TargetSpec(kind="forward_return", horizon=HOLD_DAYS),
        universe=list(universe.keys()),
        purge_bars=HOLD_DAYS,
        disclosure=DISCLOSURE,
        min_train_obs=12,
        min_test_obs=10,
        extra={
            "basket": list(prices.keys()),
            "methods": [t.label for t in PREDECLARED_TRIALS],
            "harness_note": (
                "Synthetic NAV per method; signal=+1 on monthly steps; "
                "payoff = period portfolio return."
            ),
        },
        out_path=None,
    )

    # Max-drawdown diagnostics on each method's OOS NAV path
    panel = aligned_close_panel(prices)
    any_nav = next(iter(universe.values()))
    dates = pd.to_datetime(any_nav.index).tz_localize(None)
    n = len(dates)
    _te, ts = purged_split_indices(n, HOLD_DAYS, train_frac=0.60)
    test_dates = set(dates[ts:])

    method_dd: Dict[str, Any] = {}
    for method_key, nav_df in universe.items():
        close = pd.to_numeric(nav_df["Close"], errors="coerce")
        mask = close.index.isin(test_dates)
        eq = close.loc[mask].to_numpy(dtype=float)
        method_dd[method_key] = {
            "oos_max_drawdown": _max_drawdown(eq),
            "oos_n_bars": int(np.isfinite(eq).sum()),
            "oos_end_nav": float(eq[-1]) if eq.size else None,
            "oos_start_nav": float(eq[0]) if eq.size else None,
        }

    report.update(out)
    report["basket"] = list(prices.keys())
    report["methods_evaluated"] = list(universe.keys())
    report["oos_max_drawdown_by_method"] = method_dd
    report["hold_days"] = HOLD_DAYS
    report["step_size"] = STEP_SIZE
    report["lookback_cov"] = LOOKBACK_COV
    report["period"] = period
    report["ordering_note"] = (
        "Methods and basket fixed before this run; not selected after "
        "inspecting DSR. Honest null vs 1/N is acceptable."
    )

    # Per-method OOS observation stats (for apples-to-apples table)
    from trading.research.signal_edge_harness import observation_payoffs

    per_method: Dict[str, Any] = {}
    target = TargetSpec(kind="forward_return", horizon=HOLD_DAYS)
    for trial in PREDECLARED_TRIALS:
        method = str(trial.params.get("method"))
        key = portfolio_key(method)
        hist = universe.get(key)
        if hist is None:
            continue
        sig = sfn(key, hist, trial.params)
        pay = observation_payoffs(pd.Series(sig), hist, target).dropna()
        oos_pay = pay[pay.index.isin(test_dates)].to_numpy(dtype=float)
        per_method[method] = {
            "oos_stats": observation_stats(oos_pay),
            "oos_max_drawdown": (method_dd.get(key) or {}).get("oos_max_drawdown"),
        }
    report["per_method_oos"] = per_method

    path = Path(out_path) if out_path else ROOT / "data" / "risk_parity_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)

    dsr = out.get("deflated_sharpe") or {}
    print(
        "champion:",
        (out.get("champion") or {}).get("label"),
        "OOS Sharpe:",
        ((out.get("test") or {}).get("stats") or {}).get("sharpe"),
        "DSR:",
        dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
        "recommend_live:",
        out.get("recommend_live"),
        flush=True,
    )
    print("per_method_oos:", json.dumps(per_method, indent=2), flush=True)
    print("wrote", report["wrote"], flush=True)
    return report


__all__ = [
    "BASKET",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "HOLD_DAYS",
    "STEP_SIZE",
    "LOOKBACK_COV",
    "portfolio_key",
    "equal_weights",
    "risk_parity_weights",
    "build_portfolio_nav",
    "build_portfolio_universe",
    "portfolio_signal_series",
    "make_portfolio_signal_fn",
    "load_basket_prices",
    "run_risk_parity_oos_real",
]
