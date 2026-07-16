# -*- coding: utf-8 -*-
"""Scanner screening-criteria edge test (signal-edge program Phase 5).

---------------------------------------------------------------------------
PREDECLARED TRIAL SET (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Default Scanner filter is ``quick_technical`` (``scan_market`` default).
We test the code default threshold vs the UI default threshold only —

  min_quick_score ∈ {6.0, 6.5}
  horizon ∈ {5, 10, 20}     # ~1 / 2 / 4 weeks

N_trials = 6. Each horizon is scored with a matching forward target;
DSR within a horizon uses the 2 threshold trials.

Fixed (not swept):
  filters = ["quick_technical"]   # actual default screening criteria
  universe = DEFAULT_UNIVERSE     # same membership as the Scanner default
  period = 2y (+ warm-up via get_history)
  control seed = 42

Control
-------
Same-universe random draw matched to daily pass *count* (not vs SPY /
whole market — the universe is already curated). recommend_live requires
real to clear the gate AND shuffle control to fail it (or underperform).

Causality
---------
Price/Quick-Score path only (no high_short / insider_buying — those are
not point-in-time in the live scanner). Indicators use history <= T.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trading.analysis.market_scanner import (
    DEFAULT_UNIVERSE,
    _quick_score,
    _rsi,
)
from trading.research.signal_edge_harness import (
    DISCLOSURE as HARNESS_DISCLOSURE,
    TargetSpec,
    TrialSpec,
    run_signal_edge_oos,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

PERIOD = "2y"
MIN_BARS = 60
WARMUP_BARS = 252
CONTROL_SEED = 42
FILTERS = ("quick_technical",)

PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = tuple(
    TrialSpec(
        params={
            "filters": "quick_technical",
            "min_quick_score": qs,
            "horizon": h,
        },
        label=f"qs{qs}_h{h}",
    )
    for qs in (6.0, 6.5)
    for h in (5, 10, 20)
)

TRIAL_JUSTIFICATION = (
    "Six trials only: min_quick_score {6.0, 6.5} × hold horizons {5, 10, 20}. "
    "Filter locked to quick_technical (scan_market default). 6.0 is the code "
    "default threshold; 6.5 is the Scanner UI default slider. Horizons cover "
    "~1–4 weeks after a pass — aligned with the scanner's 20d lookbacks. "
    "Universe = DEFAULT_UNIVERSE; control = same-universe count-matched "
    "random (seed=42). Alt-data filters excluded (not causal as-of T). "
    "Locked before any real-data DSR evaluation."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " Scanner Phase 5: causal quick_technical re-screen of DEFAULT_UNIVERSE "
    "vs same-universe random control. Live scan_market is last-bar only; "
    "this module reimplements the price Quick-Score path as-of each T. "
    "recommend_live never auto-wires."
)

_PASS_CACHE: Dict[Tuple[Any, ...], pd.Series] = {}
_PANEL_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}


def _normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cm = {str(c).lower(): c for c in df.columns}
    ren = {}
    for std in ("open", "high", "low", "close", "volume"):
        if std in cm and str(cm[std]) != std.title():
            ren[cm[std]] = std.title()
    if ren:
        df = df.rename(columns=ren)
    return df.sort_index()


def passes_quick_technical_at_end(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    *,
    min_quick_score: float = 6.0,
) -> bool:
    """Same Quick-Score gate as ``scan_market`` for filter quick_technical."""
    if close is None or len(close) < 20:
        return False
    last_price = float(close[-1])
    sma20 = float(np.mean(close[-20:])) if len(close) >= 20 else None
    sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else None
    rsi = _rsi(close, 14)
    ret_20d = float((close[-1] / close[-20] - 1) * 100) if len(close) >= 20 else 0.0
    high_52w = (
        float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    )
    pct_from_high = float((last_price / high_52w - 1) * 100)
    avg_vol = (
        float(np.mean(volume[-20:]))
        if volume is not None and len(volume) >= 20
        else None
    )
    vol_ratio = float(volume[-1] / avg_vol) if avg_vol and avg_vol > 0 else 1.0
    vs_sma20_pct = (
        round((last_price / sma20 - 1) * 100, 2) if sma20 else None
    )
    vs_sma50_pct = (
        round((last_price / sma50 - 1) * 100, 2) if sma50 else None
    )
    _vol5 = None
    if len(close) >= 6:
        _rel5 = np.diff(close[-6:]) / np.maximum(close[-6:-1], 1e-12)
        _vol5 = float(np.std(_rel5))
    _vol20 = None
    if len(close) >= 21:
        _rel20 = np.diff(close[-21:]) / np.maximum(close[-21:-1], 1e-12)
        _vol20 = float(np.std(_rel20))
    vol_expansion = (
        float(_vol5 / _vol20)
        if (_vol5 is not None and _vol20 is not None and _vol20 > 0)
        else 1.0
    )
    qs = _quick_score(
        rsi,
        ret_20d,
        vs_sma20_pct,
        vol_ratio,
        pct_from_high,
        vs_sma50=vs_sma50_pct,
        vol_expansion=vol_expansion,
    )
    return bool(qs >= float(min_quick_score))


def compute_pass_series(
    history: pd.DataFrame,
    *,
    min_quick_score: float = 6.0,
    min_bars: int = MIN_BARS,
) -> pd.Series:
    """Causal daily 0/1 pass flags (NaN before warm-up)."""
    hist = _normalize_ohlcv(history)
    out = pd.Series(np.nan, index=hist.index, dtype=float)
    if "Close" not in hist.columns:
        return out
    close_all = hist["Close"].to_numpy(dtype=float)
    vol_all = (
        hist["Volume"].to_numpy(dtype=float)
        if "Volume" in hist.columns
        else None
    )
    start = max(min_bars, 20)
    for i in range(start, len(hist)):
        close = close_all[: i + 1]
        volume = vol_all[: i + 1] if vol_all is not None else None
        out.iloc[i] = (
            1.0
            if passes_quick_technical_at_end(
                close, volume, min_quick_score=min_quick_score
            )
            else 0.0
        )
    return out


def build_pass_panel(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    min_quick_score: float,
) -> pd.DataFrame:
    """Wide panel: index=date, columns=symbol, values 0/1/NaN."""
    key = (
        float(min_quick_score),
        tuple(sorted(prices_by_symbol.keys())),
        tuple(
            (s, len(h), str(h.index[0]) if len(h) else "", str(h.index[-1]) if len(h) else "")
            for s, h in sorted(prices_by_symbol.items())
        ),
    )
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]

    cols = {}
    for sym, hist in prices_by_symbol.items():
        cols[str(sym).upper()] = compute_pass_series(
            hist, min_quick_score=float(min_quick_score)
        )
    panel = pd.DataFrame(cols).sort_index()
    _PANEL_CACHE[key] = panel
    return panel


def control_panel_from_real(
    real: pd.DataFrame,
    *,
    seed: int = CONTROL_SEED,
) -> pd.DataFrame:
    """Per date: keep pass count, reassign randomly across columns."""
    rng = np.random.default_rng(int(seed))
    out = pd.DataFrame(np.nan, index=real.index, columns=real.columns)
    symbols = list(real.columns)
    for dt, row in real.iterrows():
        vals = row.to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if not valid.any():
            continue
        n_pass = int(np.nansum(vals == 1.0))
        out.loc[dt, symbols] = 0.0
        if n_pass <= 0:
            continue
        cand = [s for s, ok in zip(symbols, valid) if ok]
        if not cand:
            continue
        k = min(n_pass, len(cand))
        chosen = rng.choice(cand, size=k, replace=False)
        out.loc[dt, chosen] = 1.0
    return out


def make_scanner_signal_fn(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    control: bool = False,
    seed: int = CONTROL_SEED,
):
    """Harness SignalFn: +1 on pass days, NaN otherwise (0 drops via NaN)."""

    panels: Dict[float, pd.DataFrame] = {}

    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        qs = float(params.get("min_quick_score", 6.0))
        if qs not in panels:
            real = build_pass_panel(prices_by_symbol, min_quick_score=qs)
            panels[qs] = (
                control_panel_from_real(real, seed=seed) if control else real
            )
        panel = panels[qs]
        sym = str(symbol).upper()
        if sym not in panel.columns:
            return pd.Series(np.nan, index=history.index, dtype=float)
        s = panel[sym].reindex(history.index)
        # Encode pass as +1 exposure; non-pass → NaN (no observation)
        out = s.where(s == 1.0)
        return out

    return _fn


def load_universe_prices(
    symbols: Optional[Sequence[str]] = None,
    period: str = PERIOD,
) -> Dict[str, pd.DataFrame]:
    from trading.data.price_cache import get_history

    syms = list(symbols) if symbols is not None else list(DEFAULT_UNIVERSE)
    out: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        try:
            hist = get_history(str(sym), period=period)
            if hist is None or hist.empty or len(hist) < MIN_BARS:
                continue
            out[str(sym).upper()] = _normalize_ohlcv(hist)
        except Exception as e:
            logger.debug("load %s failed: %s", sym, e)
    return out


def run_scanner_criteria_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    symbols: Optional[Sequence[str]] = None,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Phase-5 report → ``data/scanner_criteria_oos_real.json``."""
    if clear_cache:
        _PASS_CACHE.clear()
        _PANEL_CACHE.clear()

    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    prices = load_universe_prices(symbols, period=period)
    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "scanner_quick_technical",
        "disclosure": DISCLOSURE,
        "filters": list(FILTERS),
        "universe_size": len(prices),
        "universe": sorted(prices.keys()),
        "period": period,
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Thresholds, horizons, universe, and control seed fixed before "
            "this run; not selected after inspecting DSR."
        ),
        "control_seed": CONTROL_SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "real": {},
        "random_control": {},
        "recommend_live": False,
        "error": None,
    }
    if len(prices) < 10:
        report["error"] = "insufficient universe history loaded"
        return report

    # Pass-rate census (qs=6.0) for honesty
    panel60 = build_pass_panel(prices, min_quick_score=6.0)
    panel65 = build_pass_panel(prices, min_quick_score=6.5)
    report["pass_census"] = {
        "qs_6.0": {
            "n_pass_events": int((panel60 == 1.0).sum().sum()),
            "mean_daily_pass_rate": float(
                (panel60 == 1.0).sum(axis=1).mean()
            )
            if len(panel60)
            else 0.0,
        },
        "qs_6.5": {
            "n_pass_events": int((panel65 == 1.0).sum().sum()),
            "mean_daily_pass_rate": float(
                (panel65 == 1.0).sum(axis=1).mean()
            )
            if len(panel65)
            else 0.0,
        },
    }
    print("PASS_CENSUS:", report["pass_census"], flush=True)

    real_fn = make_scanner_signal_fn(prices, control=False)
    ctrl_fn = make_scanner_signal_fn(prices, control=True, seed=CONTROL_SEED)

    any_ok = False
    real_live = False
    ctrl_live = False

    for tag, sfn, bucket in (
        ("real", real_fn, report["real"]),
        ("random_control", ctrl_fn, report["random_control"]),
    ):
        print(f"=== {tag} ===", flush=True)
        for h in (5, 10, 20):
            tlist = [
                t for t in PREDECLARED_TRIALS if int(t.params["horizon"]) == h
            ]
            out = run_signal_edge_oos(
                sfn,
                prices,
                signal_name=f"scanner_{tag}_h{h}",
                trials=tlist,
                trial_justification=TRIAL_JUSTIFICATION,
                target=TargetSpec(kind="forward_return", horizon=h),
                universe=list(prices.keys()),
                purge_bars=h,
                disclosure=DISCLOSURE,
                min_train_obs=30,
                min_test_obs=10,
                extra={"arm": tag, "horizon": h, "filters": list(FILTERS)},
            )
            bucket[str(h)] = out
            if out.get("success"):
                any_ok = True
            if out.get("recommend_live"):
                if tag == "real":
                    real_live = True
                else:
                    ctrl_live = True
            dsr = out.get("deflated_sharpe") or {}
            print(
                {
                    "arm": tag,
                    "horizon": h,
                    "champion": out.get("champion"),
                    "oos": (out.get("test") or {}).get("stats"),
                    "dsr": dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
                    "recommend_live": out.get("recommend_live"),
                    "note": out.get("note"),
                },
                flush=True,
            )

    report["success"] = any_ok
    report["real_clears_gate"] = real_live
    report["control_clears_gate"] = ctrl_live
    report["recommend_live"] = bool(real_live and not ctrl_live)
    if real_live and ctrl_live:
        report["note"] = (
            "Real scanner passes cleared DSR but so did the same-universe "
            "random control — screening timing not isolated from universe "
            "drift; leave research-only."
        )
    elif report["recommend_live"]:
        report["note"] = (
            "Scanner quick_technical cleared OOS+DSR and beat random "
            "control — still research-only; not auto-wired."
        )
    else:
        report["note"] = (
            "Null / not significant on scanner-criteria OOS+DSR (or failed "
            "to beat same-universe random control) — leave research-only "
            "(acceptable)."
        )

    path = Path(out_path) if out_path else ROOT / "data" / "scanner_criteria_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    print(f"wrote {path}", flush=True)
    return report


__all__ = [
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "DISCLOSURE",
    "CONTROL_SEED",
    "passes_quick_technical_at_end",
    "compute_pass_series",
    "build_pass_panel",
    "control_panel_from_real",
    "make_scanner_signal_fn",
    "load_universe_prices",
    "run_scanner_criteria_oos_real",
]
