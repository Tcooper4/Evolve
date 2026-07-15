# -*- coding: utf-8 -*-
"""Real-data purged OOS: GNN vs default ensemble (arima/xgboost/ridge).

Research harness only — never flips live routing/GNN always-on flags.
Null result (GNN does not beat the simpler baseline under DSR) is an
acceptable, expected outcome.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Same equity basket used for conditional-vol / options-VIX real OOS
# (TLT omitted here — GNN needs correlated equity cross-section; 4 ETFs).
DEFAULT_GNN_BASKET: List[str] = ["SPY", "QQQ", "IWM", "EFA"]
FOCUS_SYMBOL = "SPY"
BASELINE_ENSEMBLE: List[str] = ["arima", "xgboost", "ridge"]

# Small, documented search — every combo is a DSR trial.
GNN_PARAM_GRID: List[Dict[str, Any]] = [
    {"hidden_size": 32, "seq_length": 20, "correlation_threshold": 0.3},
    {"hidden_size": 32, "seq_length": 20, "correlation_threshold": 0.5},
    {"hidden_size": 32, "seq_length": 30, "correlation_threshold": 0.3},
    {"hidden_size": 32, "seq_length": 30, "correlation_threshold": 0.5},
    {"hidden_size": 64, "seq_length": 20, "correlation_threshold": 0.3},
    {"hidden_size": 64, "seq_length": 20, "correlation_threshold": 0.5},
    {"hidden_size": 64, "seq_length": 30, "correlation_threshold": 0.3},
    {"hidden_size": 64, "seq_length": 30, "correlation_threshold": 0.5},
]

DEFAULT_MIN_DA_IMPROVEMENT = 0.05
DEFAULT_DSR_BAR = 0.95


def load_multi_asset_closes(
    symbols: Sequence[str],
    period: str = "2y",
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Aligned close columns, one per symbol. Requires ≥3 assets."""
    from trading.data.price_cache import get_history

    series: Dict[str, pd.Series] = {}
    for sym in symbols:
        s = str(sym).strip().upper()
        if not s:
            continue
        try:
            hist = get_history(s, period=period)
        except Exception as e:
            logger.warning("GNN OOS load %s failed: %s", s, e)
            continue
        if hist is None or hist.empty:
            continue
        cm = {str(c).lower(): c for c in hist.columns}
        close = pd.to_numeric(hist[cm.get("close", hist.columns[0])], errors="coerce")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        series[s] = close.dropna()
    if len(series) < 3:
        return None, f"need ≥3 assets with history, got {list(series)}"
    df = pd.DataFrame(series).dropna(how="any")
    if len(df) < 200:
        return None, f"aligned history too short ({len(df)} rows)"
    return df, None


def _directional_hits(
    anchor: float,
    preds: np.ndarray,
    actuals: np.ndarray,
) -> List[float]:
    hits: List[float] = []
    prev = float(anchor)
    for j, (p, a) in enumerate(zip(preds, actuals)):
        if j > 0:
            prev = float(actuals[j - 1])
        if not np.isfinite(prev) or prev == 0:
            continue
        if not np.isfinite(p) or not np.isfinite(a):
            continue
        hits.append(
            1.0 if np.sign(float(p) - prev) == np.sign(float(a) - prev) else 0.0
        )
    return hits


def purged_baseline_da(
    single_asset_ohlcv: pd.DataFrame,
    *,
    models: Sequence[str] = tuple(BASELINE_ENSEMBLE),
    horizon: int = 5,
    train_window: int = 180,
    test_window: int = 5,
    step_size: int = 42,
) -> Dict[str, Any]:
    """Reuse ``purged_ensemble_oos`` on the focus symbol."""
    from trading.models.routing_validation import purged_ensemble_oos

    return purged_ensemble_oos(
        single_asset_ohlcv,
        models=list(models),
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        purge=horizon,
    )


def purged_gnn_da(
    multi_close: pd.DataFrame,
    *,
    focus: str = FOCUS_SYMBOL,
    hidden_size: int = 64,
    seq_length: int = 30,
    correlation_threshold: float = 0.5,
    epochs: int = 20,
    horizon: int = 5,
    train_window: int = 180,
    test_window: int = 5,
    step_size: int = 42,
) -> Dict[str, Any]:
    """Walk-forward purged OOS for one GNN hyperparameter set.

    Purge gap = horizon — same meaning as WalkForwardValidator / purged_ensemble_oos.
    """
    from trading.models.advanced.gnn.gnn_model import GNNForecaster

    if focus not in multi_close.columns:
        return {"error": f"focus {focus} missing", "directional_accuracy": None}

    purge_i = int(horizon)
    n = len(multi_close)
    need = train_window + purge_i + test_window
    if n < need:
        return {
            "error": f"insufficient data ({n} < {need})",
            "directional_accuracy": None,
            "n_windows": 0,
        }

    n_assets = multi_close.shape[1]
    das: List[float] = []
    window_scores: List[float] = []
    start = train_window
    windows = 0
    errors = 0

    while start + purge_i + test_window <= n:
        train = multi_close.iloc[0:start].copy()
        test = multi_close.iloc[start + purge_i: start + purge_i + test_window].copy()
        try:
            gnn = GNNForecaster(
                num_assets=n_assets,
                hidden_size=int(hidden_size),
                num_layers=2,
                seq_length=int(seq_length),
                learning_rate=0.001,
                correlation_threshold=float(correlation_threshold),
            )
            gnn.fit(train, epochs=int(epochs), batch_size=16)
            fc = gnn.forecast(train, horizon=horizon, target_asset=focus)
            pred = np.asarray(fc.get("forecast") or fc.get("predictions") or [], dtype=float).ravel()
            actual = test[focus].to_numpy(dtype=float)[:horizon]
            m = min(len(pred), len(actual), horizon)
            if m < 1:
                errors += 1
                start += step_size
                continue
            anchor = float(train[focus].iloc[-1])
            hits = _directional_hits(anchor, pred[:m], actual[:m])
            if hits:
                das.extend(hits)
                window_scores.append(float(np.mean(hits)))
                windows += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            logger.debug("GNN purged window failed @%s: %s", start, e)
        start += step_size

    if not das:
        return {
            "error": "no scored steps",
            "directional_accuracy": None,
            "n_windows": 0,
            "n_errors": errors,
            "purge": purge_i,
        }

    return {
        "directional_accuracy": float(np.mean(das)),
        "n_steps": len(das),
        "n_windows": windows,
        "n_errors": errors,
        "purge": purge_i,
        "horizon": horizon,
        "window_scores": [round(x, 4) for x in window_scores],
        "params": {
            "hidden_size": hidden_size,
            "seq_length": seq_length,
            "correlation_threshold": correlation_threshold,
            "epochs": epochs,
        },
    }


def run_gnn_oos_real(
    symbols: Optional[Sequence[str]] = None,
    *,
    focus: str = FOCUS_SYMBOL,
    period: str = "2y",
    out_path: str = "data/gnn_oos_real.json",
    horizon: int = 5,
    train_window: int = 180,
    test_window: int = 5,
    step_size: int = 42,
    epochs: int = 20,
    param_grid: Optional[Sequence[Dict[str, Any]]] = None,
    min_da_improvement: float = DEFAULT_MIN_DA_IMPROVEMENT,
    dsr_bar: float = DEFAULT_DSR_BAR,
) -> Dict[str, Any]:
    """Full research pass → JSON. Never recommends live without DSR + DA bar."""
    from trading.data.price_cache import get_history
    from trading.optimization.deflated_sharpe import deflated_sharpe_ratio

    basket = list(symbols or DEFAULT_GNN_BASKET)
    multi, err = load_multi_asset_closes(basket, period=period)
    report: Dict[str, Any] = {
        "success": False,
        "basket": basket,
        "focus": focus,
        "baseline_ensemble": list(BASELINE_ENSEMBLE),
        "n_trials": 0,
        "recommend_live": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if err or multi is None:
        report["error"] = err or "multi-asset load failed"
        _write(report, out_path)
        return report

    # Baseline: OHLCV for focus symbol (same loader as other OOS)
    ohlcv = get_history(focus, period=period)
    if ohlcv is None or ohlcv.empty:
        report["error"] = f"no OHLCV for focus {focus}"
        _write(report, out_path)
        return report
    if getattr(ohlcv.index, "tz", None) is not None:
        ohlcv = ohlcv.copy()
        ohlcv.index = ohlcv.index.tz_localize(None)

    baseline = purged_baseline_da(
        ohlcv,
        models=BASELINE_ENSEMBLE,
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    )
    report["baseline"] = {
        "directional_accuracy": baseline.get("directional_accuracy"),
        "mape": baseline.get("mape"),
        "n_steps": baseline.get("n_steps"),
        "n_windows": baseline.get("n_windows"),
        "purge": baseline.get("purge"),
        "error": baseline.get("error"),
    }

    grid = list(param_grid or GNN_PARAM_GRID)
    trials: List[Dict[str, Any]] = []
    trial_das: List[float] = []

    for params in grid:
        logger.info("GNN OOS trial params=%s", params)
        res = purged_gnn_da(
            multi,
            focus=focus,
            epochs=epochs,
            horizon=horizon,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            **params,
        )
        da = res.get("directional_accuracy")
        trials.append({**params, **{k: v for k, v in res.items() if k != "window_scores"},
                       "window_scores": res.get("window_scores")})
        if da is not None and np.isfinite(float(da)):
            trial_das.append(float(da))

    report["n_trials"] = len(grid)
    report["trials"] = trials
    report["aligned_rows"] = len(multi)
    report["meta"] = {
        "period": period,
        "horizon": horizon,
        "train_window": train_window,
        "test_window": test_window,
        "step_size": step_size,
        "purge": horizon,
        "epochs": epochs,
        "method": (
            "Purged walk-forward (purge=horizon) matching "
            "WalkForwardValidator / purged_ensemble_oos. "
            "GNN fits multi-asset closes; baseline is single-asset "
            f"{BASELINE_ENSEMBLE} consensus on {focus}."
        ),
    }

    viable = [t for t in trials if t.get("directional_accuracy") is not None]
    if not viable:
        report["success"] = True
        report["note"] = "No viable GNN trial windows — null result."
        report["champion"] = None
        report["deflated_sharpe"] = None
        _write(report, out_path)
        return report

    champion = max(viable, key=lambda t: float(t["directional_accuracy"]))
    base_da = baseline.get("directional_accuracy")
    champ_da = float(champion["directional_accuracy"])
    delta = (champ_da - float(base_da)) if base_da is not None else None

    dsr = None
    n_obs = int(champion.get("n_steps") or 0)
    if trial_das and n_obs > 1:
        # Treat per-config mean DA as trial scores (selection-bias correction).
        # Map DA in [0,1] to a Sharpe-like scale centered at 0.5 for DSR.
        scores = [d - 0.5 for d in trial_das]
        obs = champ_da - 0.5
        try:
            dsr = deflated_sharpe_ratio(obs, scores, n_obs=n_obs)
        except Exception as e:
            logger.debug("DSR failed: %s", e)

    dsr_val = None
    if isinstance(dsr, dict):
        dsr_val = dsr.get("deflated_sharpe")

    clears = bool(
        delta is not None
        and delta > float(min_da_improvement)
        and dsr_val is not None
        and float(dsr_val) >= float(dsr_bar)
        and n_obs >= 10
    )

    report["success"] = True
    report["champion"] = {
        "params": {
            "hidden_size": champion.get("hidden_size"),
            "seq_length": champion.get("seq_length"),
            "correlation_threshold": champion.get("correlation_threshold"),
        },
        "directional_accuracy": champ_da,
        "n_steps": champion.get("n_steps"),
        "n_windows": champion.get("n_windows"),
    }
    report["delta_da_vs_baseline"] = (
        round(float(delta), 4) if delta is not None else None
    )
    report["deflated_sharpe"] = dsr
    report["min_da_improvement"] = min_da_improvement
    report["dsr_bar"] = dsr_bar
    report["recommend_live"] = False  # research harness — never auto-wire
    report["would_clear_research_bar"] = clears
    if clears:
        report["note"] = (
            "Champion cleared research DA+DSR bar on this pass — still "
            "research-only; recommend_live remains false pending review."
        )
    else:
        report["note"] = (
            "Null / not significant vs baseline under DA margin + DSR — "
            "leave GNN research-only (acceptable expected outcome)."
        )

    _write(report, out_path)
    return report


def _write(report: Dict[str, Any], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
        errors="replace",
    )


__all__ = [
    "DEFAULT_GNN_BASKET",
    "FOCUS_SYMBOL",
    "BASELINE_ENSEMBLE",
    "GNN_PARAM_GRID",
    "load_multi_asset_closes",
    "purged_baseline_da",
    "purged_gnn_da",
    "run_gnn_oos_real",
]
