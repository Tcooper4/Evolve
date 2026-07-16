# -*- coding: utf-8 -*-
"""Chart-pattern edge test (signal-edge program Phase 2).

---------------------------------------------------------------------------
PREDECLARED TRIAL SET (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Edwards & Magee / Bulkowski-style classics — not every detector name:

  pattern_set ∈ {classic_reversal, triangles}
    classic_reversal: Head & Shoulders, Inverse H&S, Double Top, Double Bottom
    triangles:        Ascending / Descending Triangle

  horizon ∈ {5, 10, 20}   # ~1 / 2 / 4 trading weeks post-completion

That is N_trials = 6 (2 × 3). Each horizon is scored with a matching
forward target so cells are not cross-contaminated; DSR within a horizon
uses the 2 pattern-set trials.

Fixed (not swept):
  min_confidence = 0.6   # matches ChartPatternDetector._generate_signals floor
  peak_confirm_lag = 7   # matches _find_peaks(..., window=7) confirmation
  scan_step = 5          # weekly causal rescan (compute budget; not daily)
  universe = SPY/QQQ/IWM
  period = 2y

Control
-------
Same signed events with fire *dates* randomly shuffled (seed locked) —
isolates whether the PATTERN timing carries information beyond market
drift / sampling. recommend_live requires the real series to clear the
standard gate AND the shuffle control to fail it (or underperform).

Causality
---------
``detect_all()`` on a full series is a UI snapshot with peak lookahead.
This module only scores expanding slices and emits at completion + lag.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

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
MIN_BARS = 80
PEAK_CONFIRM_LAG = 7
SCAN_STEP = 5
MIN_CONFIDENCE = 0.6
SHUFFLE_SEED = 42

PATTERN_SETS: Dict[str, Set[str]] = {
    "classic_reversal": {
        "Head and Shoulders",
        "Inverse Head and Shoulders",
        "Double Top",
        "Double Bottom",
    },
    "triangles": {
        "Ascending Triangle",
        "Descending Triangle",
    },
}

PREDECLARED_TRIALS: Tuple[TrialSpec, ...] = tuple(
    TrialSpec(
        params={
            "pattern_set": pset,
            "horizon": h,
            "min_confidence": MIN_CONFIDENCE,
            "peak_lag": PEAK_CONFIRM_LAG,
            "scan_step": SCAN_STEP,
        },
        label=f"{pset}_h{h}",
    )
    for pset in ("classic_reversal", "triangles")
    for h in (5, 10, 20)
)

TRIAL_JUSTIFICATION = (
    "Six trials only: pattern_set {classic_reversal, triangles} × "
    "hold horizons {5, 10, 20}. classic_reversal = H&S / inverse H&S / "
    "double top / double bottom (most-cited TA reversals); triangles = "
    "ascending/descending (classic breakout structures). Horizons cover "
    "~1–4 weeks post-completion. min_confidence=0.6, peak_lag=7, "
    "scan_step=5 fixed (detector signal floor + peak-window confirmation + "
    "weekly causal rescan). Shuffle control seed=42 locked. Not expanded "
    "after inspecting DSR."
)

DISCLOSURE = (
    HARNESS_DISCLOSURE
    + " Chart-pattern Phase 2: causal expanding-window ChartPatternDetector "
    "with peak_confirm_lag=7. Full-series detect_all() is NOT used for "
    "scoring. Shuffle-date control included. recommend_live never auto-wires."
)

# Cache: (symbol, pattern_set, min_conf, peak_lag, scan_step, hist fingerprint) -> Series
_SIGNAL_CACHE: Dict[Tuple[Any, ...], pd.Series] = {}


def _normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cm = {str(c).lower(): c for c in df.columns}
    ren = {}
    for std in ("open", "high", "low", "close", "volume"):
        if std in cm and cm[std] != std.title():
            ren[cm[std]] = std.title()
    if ren:
        df = df.rename(columns=ren)
    return df.sort_index()


def _hist_fingerprint(history: pd.DataFrame) -> Tuple[Any, ...]:
    if history is None or len(history) == 0:
        return (0, "", "")
    return (len(history), str(history.index[0]), str(history.index[-1]))


def compute_pattern_signal_series(
    symbol: str,
    history: pd.DataFrame,
    *,
    pattern_names: Set[str],
    min_confidence: float = MIN_CONFIDENCE,
    peak_lag: int = PEAK_CONFIRM_LAG,
    scan_step: int = SCAN_STEP,
    min_bars: int = MIN_BARS,
) -> pd.Series:
    """Causal +1/−1 series: expanding detect, emit at completion + peak_lag."""
    from trading.analysis.chart_pattern_detector import ChartPatternDetector

    hist = _normalize_ohlcv(history)
    out = pd.Series(np.nan, index=hist.index, dtype=float)
    n = len(hist)
    if n < min_bars + peak_lag + 5:
        return out

    step = max(1, int(scan_step))
    lag = max(0, int(peak_lag))
    names = set(pattern_names)

    for t in range(min_bars, n, step):
        slice_df = hist.iloc[: t + 1]
        try:
            det = ChartPatternDetector(str(symbol), slice_df)
            det.detect_all()
            patterns = list(det._patterns or [])
        except Exception as e:
            logger.debug("pattern detect failed %s @%s: %s", symbol, t, e)
            continue

        score = 0.0
        for p in patterns:
            if p.name not in names:
                continue
            if float(p.confidence or 0) < float(min_confidence):
                continue
            # Newly completed near the slice end (within this scan step)
            if int(p.end_idx) < t - step:
                continue
            if str(p.pattern_type).lower() == "bullish":
                score += 1.0
            elif str(p.pattern_type).lower() == "bearish":
                score -= 1.0

        if score == 0.0:
            continue
        emit_i = min(t + lag, n - 1)
        prev = out.iloc[emit_i]
        combined = float(np.sign(score) if not np.isfinite(prev) else np.sign(prev + np.sign(score)))
        out.iloc[emit_i] = combined if combined != 0 else np.sign(score)

    return out


def shuffle_signal_dates(
    signal: pd.Series,
    *,
    horizon: int,
    seed: int = SHUFFLE_SEED,
) -> pd.Series:
    """Keep signed events; reassign fire dates uniformly among valid bars."""
    s = pd.to_numeric(signal, errors="coerce")
    events = s.dropna()
    events = events[events != 0]
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if events.empty:
        return out

    n = len(s)
    h = max(1, int(horizon))
    # Valid emit indices: room for forward target
    valid = np.arange(0, max(0, n - h), dtype=int)
    if len(valid) == 0:
        return out

    rng = np.random.default_rng(int(seed))
    # Sample without replacement when possible
    k = min(len(events), len(valid))
    chosen = rng.choice(valid, size=k, replace=(k > len(valid)))
    signs = events.to_numpy(dtype=float)[:k]
    # If more events than slots, truncate; if fewer, use all
    for idx, sign in zip(chosen, signs):
        out.iloc[int(idx)] = float(np.sign(sign))
    return out


def make_pattern_signal_fn(*, shuffle: bool = False, shuffle_seed: int = SHUFFLE_SEED):
    """Harness SignalFn. ``params`` must include pattern_set + horizon."""

    def _fn(symbol: str, history: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        pset = str(params.get("pattern_set", "classic_reversal"))
        names = PATTERN_SETS.get(pset, PATTERN_SETS["classic_reversal"])
        min_conf = float(params.get("min_confidence", MIN_CONFIDENCE))
        peak_lag = int(params.get("peak_lag", PEAK_CONFIRM_LAG))
        scan_step = int(params.get("scan_step", SCAN_STEP))
        horizon = int(params.get("horizon", 5))
        fp = _hist_fingerprint(history)
        key = (str(symbol).upper(), pset, min_conf, peak_lag, scan_step, fp, bool(shuffle), int(shuffle_seed), horizon if shuffle else 0)

        if key in _SIGNAL_CACHE:
            return _SIGNAL_CACHE[key]

        real = compute_pattern_signal_series(
            symbol,
            history,
            pattern_names=names,
            min_confidence=min_conf,
            peak_lag=peak_lag,
            scan_step=scan_step,
        )
        series = (
            shuffle_signal_dates(real, horizon=horizon, seed=shuffle_seed)
            if shuffle
            else real
        )
        _SIGNAL_CACHE[key] = series
        return series

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
            if hist is None or hist.empty or len(hist) < MIN_BARS + 40:
                logger.warning("skip %s: insufficient history", sym)
                continue
            out[str(sym).upper()] = _normalize_ohlcv(hist)
        except Exception as e:
            logger.warning("load %s failed: %s", sym, e)
    return out


def _section_clears(out: Mapping[str, Any]) -> bool:
    return bool(out.get("recommend_live"))


def run_chart_patterns_oos_real(
    *,
    out_path: Optional[str] = None,
    period: str = PERIOD,
    symbols: Sequence[str] = BASKET,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """Phase-2 report → ``data/chart_patterns_oos_real.json``."""
    if clear_cache:
        _SIGNAL_CACHE.clear()

    print("PREDECLARED TRIALS:", [t.as_dict() for t in PREDECLARED_TRIALS], flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)
    print("SHUFFLE_SEED:", SHUFFLE_SEED, flush=True)

    prices = load_basket_prices(symbols, period=period)
    report: Dict[str, Any] = {
        "success": False,
        "signal_name": "chart_patterns",
        "disclosure": DISCLOSURE,
        "basket": list(prices.keys()),
        "period": period,
        "predeclared_trials": [t.as_dict() for t in PREDECLARED_TRIALS],
        "n_trials": len(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "ordering_note": (
            "Trial set, pattern families, horizons, and shuffle seed were "
            "fixed before this run; not selected after inspecting DSR."
        ),
        "peak_confirm_lag": PEAK_CONFIRM_LAG,
        "scan_step": SCAN_STEP,
        "min_confidence": MIN_CONFIDENCE,
        "shuffle_seed": SHUFFLE_SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "real": {},
        "shuffle_control": {},
        "recommend_live": False,
        "error": None,
    }
    if len(prices) < 2:
        report["error"] = "need >=2 symbols with history"
        return report

    # Honest event census (same causal settings as trials) before DSR.
    event_counts: Dict[str, Any] = {}
    for pset, names in PATTERN_SETS.items():
        per: Dict[str, int] = {}
        total = 0
        for sym, hist in prices.items():
            series = compute_pattern_signal_series(
                sym,
                hist,
                pattern_names=names,
                min_confidence=MIN_CONFIDENCE,
                peak_lag=PEAK_CONFIRM_LAG,
                scan_step=SCAN_STEP,
            )
            n_ev = int((series.dropna() != 0).sum())
            per[sym] = n_ev
            total += n_ev
        event_counts[pset] = {"per_symbol": per, "total": total}
    report["event_counts"] = event_counts
    print("EVENT_COUNTS:", event_counts, flush=True)

    real_fn = make_pattern_signal_fn(shuffle=False)
    shuffle_fn = make_pattern_signal_fn(shuffle=True, shuffle_seed=SHUFFLE_SEED)

    any_ok = False
    real_any_live = False
    shuffle_any_live = False

    for tag, sfn, bucket in (
        ("real", real_fn, report["real"]),
        ("shuffle_control", shuffle_fn, report["shuffle_control"]),
    ):
        print(f"=== {tag} ===", flush=True)
        for h in (5, 10, 20):
            tlist = [
                t for t in PREDECLARED_TRIALS if int(t.params["horizon"]) == h
            ]
            out = run_signal_edge_oos(
                sfn,
                prices,
                signal_name=f"chart_patterns_{tag}_h{h}",
                trials=tlist,
                trial_justification=TRIAL_JUSTIFICATION,
                target=TargetSpec(kind="hit_miss", horizon=h, benchmark="zero"),
                universe=list(prices.keys()),
                purge_bars=h,
                disclosure=DISCLOSURE,
                min_train_obs=8,
                min_test_obs=10,
                extra={
                    "arm": tag,
                    "horizon": h,
                    "peak_lag": PEAK_CONFIRM_LAG,
                    "scan_step": SCAN_STEP,
                },
            )
            bucket[str(h)] = out
            if out.get("success"):
                any_ok = True
            if _section_clears(out):
                if tag == "real":
                    real_any_live = True
                else:
                    shuffle_any_live = True
            dsr = out.get("deflated_sharpe") or {}
            print(
                {
                    "arm": tag,
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

    # Live only if real clears and shuffle does not (pattern timing matters)
    report["success"] = any_ok
    report["real_clears_gate"] = real_any_live
    report["shuffle_clears_gate"] = shuffle_any_live
    report["recommend_live"] = bool(real_any_live and not shuffle_any_live)
    if real_any_live and shuffle_any_live:
        report["note"] = (
            "Real series cleared DSR gate but so did the shuffled-date "
            "control — pattern *timing* is not isolated from drift; "
            "leave research-only (do not treat as pattern edge)."
        )
    elif report["recommend_live"]:
        report["note"] = (
            "Real pattern signals cleared OOS+DSR and shuffle control did "
            "not — still research-only; not auto-wired."
        )
    else:
        n_rev = int((event_counts.get("classic_reversal") or {}).get("total") or 0)
        n_tri = int((event_counts.get("triangles") or {}).get("total") or 0)
        if n_rev + n_tri < 20:
            report["status"] = "insufficient_events"
            report["note"] = (
                f"Insufficient causal pattern fires under the locked protocol "
                f"(classic_reversal={n_rev}, triangles={n_tri} on {period} "
                f"{list(prices.keys())} with weekly scan + peak_lag={PEAK_CONFIRM_LAG}). "
                "OOS+DSR is underpowered — defer meaningful edge claims until "
                "more events accumulate (longer history / denser scan would be a "
                "new predeclared experiment, not a post-hoc re-grid). "
                "recommend_live=false."
            )
        else:
            report["status"] = "tested_null"
            report["note"] = (
                "Null / not significant on chart-pattern OOS+DSR (or failed to "
                "beat shuffle control) — leave research-only (acceptable)."
            )

    path = Path(out_path) if out_path else ROOT / "data" / "chart_patterns_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    print(f"wrote {path}", flush=True)
    return report


__all__ = [
    "BASKET",
    "PATTERN_SETS",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "DISCLOSURE",
    "SHUFFLE_SEED",
    "compute_pattern_signal_series",
    "shuffle_signal_dates",
    "make_pattern_signal_fn",
    "load_basket_prices",
    "run_chart_patterns_oos_real",
]
