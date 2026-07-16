# -*- coding: utf-8 -*-
"""Base forecast-ensemble absolute edge test (signal-edge program Phase 1).

---------------------------------------------------------------------------
PREDECLARED TRIAL SET (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
The ARIMA/XGBoost/Ridge/CatBoost/Prophet consensus has been the baseline
every other model had to beat; its own absolute quality was never checked.

Trials (N=4) span the two axes already used in Evolve research/live code —
horizon and voter set — without a wide grid:

  horizon ∈ {5, 7} × models ∈ {core3, base5}

  - horizon=5 — matches ``routing_validation.REAL_OOS_ENSEMBLE`` / GNN baseline
  - horizon=7 — matches live ``get_consensus_forecast`` default
  - core3 — arima/xgboost/ridge (``REAL_OOS_ENSEMBLE``)
  - base5 — full ``DEFAULT_ENSEMBLE`` price voters

Fixed for all trials (not swept):
  step_size=21   # ~monthly decision dates (not daily refits; compute budget)
  min_train=180  # bars of history before first call
  universe=SPY/QQQ/IWM
  period=2y      # same window as routing_validation.run_real_ticker_oos

Direction = live router band: BULLISH/BEARISH/NEUTRAL from consensus price
vs last close (±0.5%). Neutral days contribute no observation.

Targets (separate harness runs, same trials — not cherry-picked after):
  - hit_miss vs zero          → directional accuracy
  - hit_miss vs persistence   → beat naive random-walk / last-day direction
  - forward_return with BH-excess weights → beat buy-and-hold on call days

Null is a complete, important finding.
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

BASKET: Tuple[str, ...] = ("SPY", "QQQ", "IWM")
PERIOD = "2y"
STEP_SIZE = 21
MIN_TRAIN_BARS = 180

MODEL_SETS: Dict[str, Tuple[str, ...]] = {
    "core3": ("arima", "xgboost", "ridge"),
    "base5": ("arima", "xgboost", "ridge", "catboost", "prophet"),
}

# --- locked BEFORE real-data runs ---
PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = (
    TrialSpec(
        params={"horizon": 5, "models": "core3", "step_size": STEP_SIZE},
        label="h5_core3",
    ),
    TrialSpec(
        params={"horizon": 5, "models": "base5", "step_size": STEP_SIZE},
        label="h5_base5",
    ),
    TrialSpec(
        params={"horizon": 7, "models": "core3", "step_size": STEP_SIZE},
        label="h7_core3",
    ),
    TrialSpec(
        params={"horizon": 7, "models": "base5", "step_size": STEP_SIZE},
        label="h7_base5",
    ),
)

TRIAL_JUSTIFICATION = (
    "Four trials only: horizons {5, 7} × voter sets {core3, base5}. "
    "horizon=5 matches existing real-ticker / GNN baseline OOS; horizon=7 "
    "matches live consensus default; core3 is REAL_OOS_ENSEMBLE "
    "(arima/xgboost/ridge); base5 is DEFAULT_ENSEMBLE (adds catboost/"
    "prophet). step_size=21 fixed (monthly decision dates) to keep causal "
    "refits tractable — not a free parameter. period=2y matches "
    "routing_validation.run_real_ticker_oos. Locked before any real-data "
    "DSR evaluation. Each horizon is scored with a matching forward target "
    "horizon so the 5d and 7d cells are not cross-contaminated."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " BASE ensemble Phase 1: ForecastRouter.get_consensus_forecast "
    "directional band (±0.5% vs last close) on SPY/QQQ/IWM. Expanding "
    "causal context only; monthly step_size — denser sampling would be a "
    "different (costlier) experiment, not a post-hoc re-grid of this one."
)

# Process-local cache: (symbol, horizon, models_key, step) -> direction Series
_DIRECTION_CACHE: Dict[Tuple[Any, ...], pd.Series] = {}


def direction_from_consensus(fc: Mapping[str, Any]) -> float:
    """Map router consensus payload to +1 / -1 / 0."""
    if not fc or fc.get("error"):
        return 0.0
    d = str(fc.get("direction") or "NEUTRAL").upper()
    if d == "BULLISH":
        return 1.0
    if d == "BEARISH":
        return -1.0
    return 0.0


def _normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cm = {str(c).lower(): c for c in df.columns}
    if "close" in cm and cm["close"] != "Close":
        df = df.rename(columns={cm["close"]: "Close"})
    return df.sort_index()


def compute_direction_series(
    symbol: str,
    history: pd.DataFrame,
    *,
    horizon: int,
    models: Sequence[str],
    step_size: int = STEP_SIZE,
    min_train: int = MIN_TRAIN_BARS,
    forecast_fn=None,
) -> pd.Series:
    """Causal expanding-window consensus directions on step dates only."""
    hist = _normalize_ohlcv(history)
    out = pd.Series(np.nan, index=hist.index, dtype=float)
    if len(hist) < min_train + int(horizon) + 5:
        return out

    if forecast_fn is None:
        from trading.models.forecast_router import get_router_singleton

        router = get_router_singleton()

        def forecast_fn(ctx: pd.DataFrame, h: int, mods: Sequence[str]):
            return router.get_consensus_forecast(
                data=ctx,
                horizon=int(h),
                models=list(mods),
                symbol=symbol,
                model_configs={"arima": {"fast_mode": True}},
            )

    models_l = list(models)
    step = max(1, int(step_size))
    for i in range(int(min_train), len(hist), step):
        ctx = hist.iloc[: i + 1]
        try:
            fc = forecast_fn(ctx, int(horizon), models_l)
            out.iloc[i] = float(direction_from_consensus(fc))
        except Exception as e:
            logger.debug("consensus failed %s @%s: %s", symbol, hist.index[i], e)
            out.iloc[i] = np.nan
    # Neutrals are valid "no trade" — leave as 0.0; harness hit_miss skips 0
    return out


def _cached_directions(
    symbol: str,
    history: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    forecast_fn=None,
) -> pd.Series:
    models_key = str(params.get("models", "core3"))
    horizon = int(params.get("horizon", 5))
    step = int(params.get("step_size", STEP_SIZE))
    key = (str(symbol).upper(), horizon, models_key, step, id(history))
    # Prefer content-stable key without id when possible
    key2 = (
        str(symbol).upper(),
        horizon,
        models_key,
        step,
        len(history),
        str(history.index[0]) if len(history) else "",
        str(history.index[-1]) if len(history) else "",
    )
    if key2 in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[key2]
    models = MODEL_SETS.get(models_key, MODEL_SETS["core3"])
    series = compute_direction_series(
        symbol,
        history,
        horizon=horizon,
        models=models,
        step_size=step,
        forecast_fn=forecast_fn,
    )
    _DIRECTION_CACHE[key2] = series
    return series


def make_direction_signal_fn(forecast_fn=None):
    """Signal = router direction ∈ {+1, -1, 0}; NaN on non-decision dates."""

    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        d = _cached_directions(symbol, history, params, forecast_fn=forecast_fn)
        # Non-step dates stay NaN; zeros (NEUTRAL) stay 0 for hit_miss skip
        return d

    return _fn


def make_bh_excess_weight_fn(forecast_fn=None):
    """Weight so (weight * fwd) = (pos - 1) * fwd on directional call days.

    Bullish → excess 0 vs BH; bearish → excess -2*fwd (beats BH iff market falls).
    Neutrals / non-calls → NaN.
    """

    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        d = _cached_directions(symbol, history, params, forecast_fn=forecast_fn)
        pos = d.copy()
        w = pd.Series(np.nan, index=pos.index, dtype=float)
        mask = pos.notna() & (pos != 0.0)
        w.loc[mask] = pos.loc[mask] - 1.0
        return w

    return _fn


def load_basket_prices(
    symbols: Sequence[str] = BASKET,
    period: str = PERIOD,
) -> Dict[str, pd.DataFrame]:
    from trading.data.price_cache import get_history

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            hist = get_history(str(sym), period=period)
            if hist is None or hist.empty or len(hist) < MIN_TRAIN_BARS + 40:
                logger.warning("skip %s: insufficient history", sym)
                continue
            out[str(sym).upper()] = _normalize_ohlcv(hist)
        except Exception as e:
            logger.warning("load %s failed: %s", sym, e)
    return out


def run_base_ensemble_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    symbols: Sequence[str] = BASKET,
    forecast_fn=None,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Full Phase-1 report → ``data/base_ensemble_oos_real.json``."""
    if clear_cache:
        _DIRECTION_CACHE.clear()

    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    prices = load_basket_prices(symbols, period=period)
    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "base_ensemble_consensus",
        "disclosure": DISCLOSURE,
        "basket": list(prices.keys()),
        "period": period,
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Trial set and justification fixed before this run; not selected "
            "after inspecting DSR."
        ),
        "step_size": STEP_SIZE,
        "min_train_bars": MIN_TRAIN_BARS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": {},
        "recommend_live": False,
        "error": None,
    }
    if len(prices) < 2:
        report["error"] = "need >=2 symbols with history"
        return report

    dir_fn = make_direction_signal_fn(forecast_fn=forecast_fn)
    bh_fn = make_bh_excess_weight_fn(forecast_fn=forecast_fn)

    trials_h5 = [t for t in PREDECLARED_TRIALS if int(t.params["horizon"]) == 5]
    trials_h7 = [t for t in PREDECLARED_TRIALS if int(t.params["horizon"]) == 7]

    any_ok = False
    recommend_any = False

    section_defs = (
        (
            "directional_vs_zero",
            dir_fn,
            "hit_miss",
            "zero",
            "hit_miss vs zero: does sign(consensus) match sign(fwd return)?",
        ),
        (
            "vs_persistence",
            dir_fn,
            "hit_miss",
            "persistence",
            "hit_miss vs persistence (last 1d return as random-walk benchmark).",
        ),
        (
            "vs_buy_hold",
            bh_fn,
            "forward_return",
            None,
            "Excess vs buy-and-hold on directional-call days: (pos-1)*fwd. "
            "Bullish ties BH; bearish beats BH iff market falls.",
        ),
    )

    for section, sfn, kind, bench, note in section_defs:
        print(f"=== target section: {section} ===", flush=True)
        section_out: Dict[str, Any] = {"note": note, "by_horizon": {}}
        for h, tlist in ((5, trials_h5), (7, trials_h7)):
            if not tlist:
                continue
            if kind == "forward_return":
                target = TargetSpec(kind="forward_return", horizon=h)
            else:
                target = TargetSpec(
                    kind="hit_miss",
                    horizon=h,
                    benchmark=bench,  # type: ignore[arg-type]
                )
            out = run_signal_edge_oos(
                sfn,
                prices,
                signal_name=f"base_ensemble_{section}_h{h}",
                trials=tlist,
                trial_justification=TRIAL_JUSTIFICATION,
                target=target,
                universe=list(prices.keys()),
                purge_bars=h,
                disclosure=DISCLOSURE,
                min_train_obs=12,
                min_test_obs=10,
                extra={"step_size": STEP_SIZE, "period": period, "section": section},
            )
            section_out["by_horizon"][str(h)] = out
            if out.get("success"):
                any_ok = True
            if out.get("recommend_live"):
                recommend_any = True
            dsr = out.get("deflated_sharpe") or {}
            print(
                {
                    "section": section,
                    "horizon": h,
                    "n_trials": out.get("n_trials"),
                    "champion": out.get("champion"),
                    "oos": (out.get("test") or {}).get("stats"),
                    "dsr": dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
                    "recommend_live": out.get("recommend_live"),
                    "error": out.get("error"),
                    "note": out.get("note"),
                },
                flush=True,
            )

        section_out["recommend_live"] = any(
            bool((section_out["by_horizon"][k] or {}).get("recommend_live"))
            for k in section_out["by_horizon"]
        )
        report["targets"][section] = section_out

    report["success"] = any_ok
    report["recommend_live"] = recommend_any
    report["note"] = (
        "At least one BASE-ensemble target/horizon cleared OOS+DSR — "
        "still research-only; not auto-wired."
        if recommend_any
        else (
            "Null / not significant on BASE ensemble absolute edge across "
            "predeclared trials — leave research-only. Prior comparisons "
            "that treated this ensemble as a proven baseline were relative "
            "to an unproven absolute edge (important finding)."
            if any_ok
            else "BASE ensemble OOS did not complete successfully."
        )
    )

    path = Path(out_path) if out_path else ROOT / "data" / "base_ensemble_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    print(f"wrote {path}", flush=True)
    return report


__all__ = [
    "BASKET",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "DISCLOSURE",
    "MODEL_SETS",
    "direction_from_consensus",
    "compute_direction_series",
    "make_direction_signal_fn",
    "make_bh_excess_weight_fn",
    "load_basket_prices",
    "run_base_ensemble_oos_real",
]
