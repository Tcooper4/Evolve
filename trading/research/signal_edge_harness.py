# -*- coding: utf-8 -*-
"""Reusable signal-edge-test harness (Phase 0 of the Evolve edge program).

===========================================================================
ONGOING PROGRAM — not a one-shot
===========================================================================
This module is the **shared infrastructure** for a systematic program that
tests every signal-generating component in Evolve for *real*, DSR-honest
edge (not just mechanical correctness).

Future signal areas (base ensemble, chart patterns, GEX, AI Score, scanner,
and whatever follows) MUST go through ``run_signal_edge_oos`` (or the
lower-level ``evaluate_trials_purged_oos``) so every artifact lands in the
same ``data/*_oos_real.json`` shape. Do **not** reimplement purge / DSR /
``recommend_live`` gates per signal.

Adding a new signal test (checklist)
------------------------------------
1. Pre-declare the trial set + written justification *before* any run
   (same discipline as ``trading.backtesting.pead_strategy``).
2. Implement a causal ``signal_fn(symbol, history, params) -> Series``.
3. Choose a ``TargetSpec`` (``forward_return`` | ``forward_realized_vol`` |
   ``hit_miss``).
4. Call ``run_signal_edge_oos(...)``; write the returned dict to
   ``data/<name>_oos_real.json``.
5. Null is a complete answer — do not re-grid the same window to chase DSR.
6. Never ship default-on / live behavior without clearing the gate below.

Gate (identical to options / PEAD research)
-------------------------------------------
``recommend_live`` requires ALL of:
  - DSR >= 0.95 (via ``trading.optimization.deflated_sharpe``)
  - OOS observed Sharpe > 0
  - OOS n >= 10

Reuse (do not fork)
-------------------
- DSR: ``trading.optimization.deflated_sharpe.deflated_sharpe_ratio``
- Purge gap: same semantics as
  ``WalkForwardValidator.run(..., purge=horizon)`` /
  ``options_strategy_backtest._purged_split_indices``
- Trial justification convention: PEAD module docstring style
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Program constants (do not weaken per-signal)
# ---------------------------------------------------------------------------
DSR_LIVE_THRESHOLD = 0.95
MIN_OOS_OBS = 10
FAILED_TRIAL_SCORE = -999.0
DEFAULT_TRAIN_FRAC = 0.60
DEFAULT_RECOMMEND_LIVE = False

TargetKind = Literal["forward_return", "forward_realized_vol", "hit_miss"]
HitBenchmark = Literal["zero", "buy_hold", "persistence"]

DISCLOSURE = (
    "Signal-edge research harness: purged train/test + predeclared trials + "
    "Deflated Sharpe gate. Per-observation Sharpe is not annualized. "
    "Null OOS is expected and acceptable. recommend_live never auto-wires "
    "product defaults."
)

# Causal signal: may only use rows with index <= as-of for each stamped score.
# Returns a float Series indexed like ``history`` (NaN = no call that day).
SignalFn = Callable[[str, pd.DataFrame, Mapping[str, Any]], pd.Series]


@dataclass(frozen=True)
class TargetSpec:
    """How to turn post-T prices into an observation metric."""

    kind: TargetKind = "forward_return"
    horizon: int = 5
    # hit_miss only: compare signal direction vs forward move vs benchmark.
    benchmark: HitBenchmark = "zero"
    # Realized-vol window ending at T+horizon (inclusive of forward path).
    vol_window: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be >= 1")


@dataclass(frozen=True)
class TrialSpec:
    """One predeclared parameter dict (locked before any DSR peek)."""

    params: Mapping[str, Any]
    label: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.params)
        if self.label is not None:
            out = {"label": self.label, **out}
        return out


@dataclass
class SignalEdgeConfig:
    signal_name: str
    trials: Sequence[TrialSpec]
    trial_justification: str
    target: TargetSpec
    universe: Sequence[str]
    purge_bars: Optional[int] = None  # default = target.horizon
    train_frac: float = DEFAULT_TRAIN_FRAC
    min_train_obs: int = 30
    min_test_obs: int = MIN_OOS_OBS
    disclosure: str = DISCLOSURE
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Purge (generalized WalkForwardValidator / options bar split)
# ---------------------------------------------------------------------------
def purged_split_indices(
    n: int,
    purge: int,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> Tuple[int, int]:
    """Train/test indices with the WalkForwardValidator embargo semantics.

    ``train`` uses ``[:train_end]``; bars ``[train_end:test_start)`` are
    purged; ``test`` uses ``[test_start:]``.

    Identical to ``options_strategy_backtest._purged_split_indices`` and to
    ``WalkForwardValidator.run``'s ``start_idx`` / ``start_idx + purge``.
    """
    try:
        purge_i = max(0, int(purge))
    except Exception:
        purge_i = 0
    n_i = max(0, int(n))
    if n_i <= 1:
        return 0, 0
    start_idx = max(1, min(n_i - 1, int(n_i * float(train_frac))))
    train_end = start_idx
    test_start = start_idx + purge_i
    return train_end, test_start


# ---------------------------------------------------------------------------
# Stats + gate
# ---------------------------------------------------------------------------
def observation_stats(values: np.ndarray) -> Dict[str, Any]:
    """Per-observation stats (same shape as options/PEAD ``stats`` blocks)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "win_rate": None,
            "avg_pnl": None,
            "total_pnl": 0.0,
            "sharpe": None,
            "n": 0,
        }
    wins = float(np.mean(arr > 0))
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if sd > 1e-12:
        sharpe = mu / sd
    elif arr.size > 1 and abs(mu) > 1e-12:
        # Degenerate perfect (or perfectly bad) sample: |Sharpe| → ∞.
        # Cap so trial ranking / DSR stay numeric.
        sharpe = 10.0 if mu > 0 else -10.0
    else:
        sharpe = None
    return {
        "win_rate": round(wins, 4),
        "avg_pnl": round(mu, 6),
        "total_pnl": round(float(np.sum(arr)), 6),
        "sharpe": None if sharpe is None else round(float(sharpe), 4),
        "n": int(arr.size),
    }


def recommend_live_gate(
    dsr: Optional[Mapping[str, Any]],
    observed_sharpe: Optional[float],
    n_obs: int,
    *,
    dsr_threshold: float = DSR_LIVE_THRESHOLD,
    min_obs: int = MIN_OOS_OBS,
) -> bool:
    """Standard research live-candidate gate (never auto-wires product)."""
    return bool(
        dsr
        and float(dsr.get("deflated_sharpe") or 0) >= float(dsr_threshold)
        and (observed_sharpe is not None and float(observed_sharpe) > 0)
        and int(n_obs) >= int(min_obs)
    )


def compute_deflated_sharpe(
    observed_sr: Optional[float],
    trial_scores: Sequence[float],
    n_obs: int,
) -> Optional[Dict[str, Any]]:
    """Thin wrapper — always import DSR from deflated_sharpe.py."""
    if observed_sr is None or n_obs <= 1:
        return None
    try:
        from trading.optimization.deflated_sharpe import deflated_sharpe_ratio

        scores = [float(s) for s in trial_scores if s is not None and float(s) > -998]
        return deflated_sharpe_ratio(float(observed_sr), scores, n_obs=int(n_obs))
    except Exception as e:
        logger.debug("DSR failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Targets (harness-owned — prevents callers from mixing future into signal)
# ---------------------------------------------------------------------------
def _close_series(history: pd.DataFrame) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype=float)
    cm = {str(c).lower(): c for c in history.columns}
    col = cm.get("close") or cm.get("adj close") or history.columns[0]
    s = pd.to_numeric(history[col], errors="coerce")
    idx = pd.to_datetime(s.index).tz_localize(None)
    s = pd.Series(s.values, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """R_t = close[t+h]/close[t] - 1 (aligned to signal date t)."""
    h = int(horizon)
    fut = close.shift(-h)
    return (fut / close) - 1.0


def forward_realized_vol(close: pd.Series, horizon: int, vol_window: Optional[int] = None) -> pd.Series:
    """Realized vol over the forward path starting *after* t (no look-ahead in window end).

    Uses log-returns from t+1 .. t+horizon (or vol_window if set, capped by horizon).
    Stamped at t so it is a valid post-signal target.
    """
    h = int(horizon)
    w = int(vol_window) if vol_window is not None else h
    w = max(1, min(w, h))
    log_r = np.log(close / close.shift(1))
    # At date t, realized vol of the next ``w`` returns = std of log_r[t+1 : t+w]
    # Equivalent: rolling std of log_r ending at t+w, then shift back by w.
    rolled = log_r.rolling(w).std()
    return rolled.shift(-w)


def hit_miss_payoff(
    signal: pd.Series,
    close: pd.Series,
    horizon: int,
    benchmark: HitBenchmark = "zero",
) -> pd.Series:
    """+1 / -1 when directional call matches forward move vs benchmark."""
    fwd = forward_returns(close, horizon)
    sig = pd.to_numeric(signal, errors="coerce")
    if benchmark == "zero":
        bench = pd.Series(0.0, index=fwd.index)
    elif benchmark == "buy_hold":
        # Beat flat long: excess over the same forward return is tautological;
        # hit = sign(signal) matches sign(fwd) AND |fwd| useful — use sign match
        # vs zero for long/short call, reported separately as vs BH in extras.
        bench = pd.Series(0.0, index=fwd.index)
    elif benchmark == "persistence":
        # Random-walk / persistence: tomorrow continues yesterday's 1d move.
        bench = close.pct_change().fillna(0.0)
    else:
        raise ValueError(f"unknown hit_miss benchmark: {benchmark}")

    edge = fwd - bench
    pos = np.sign(sig.to_numpy(dtype=float))
    move = np.sign(edge.to_numpy(dtype=float))
    out = np.where(
        np.isfinite(pos) & np.isfinite(move) & (pos != 0) & (move != 0),
        np.where(pos == move, 1.0, -1.0),
        np.nan,
    )
    return pd.Series(out, index=fwd.index)


def observation_payoffs(
    signal: pd.Series,
    history: pd.DataFrame,
    target: TargetSpec,
) -> pd.Series:
    """Map causal signal + post-T prices → per-date observation values."""
    close = _close_series(history)
    sig = pd.to_numeric(signal, errors="coerce").reindex(close.index)

    if target.kind == "forward_return":
        fwd = forward_returns(close, target.horizon)
        # Continuous exposure: signal strength * forward return (sign carries direction).
        return (sig * fwd).astype(float)

    if target.kind == "forward_realized_vol":
        rv = forward_realized_vol(close, target.horizon, target.vol_window)
        # High signal → expect high vol: payoff = (z-scored later) raw product.
        return (sig * rv).astype(float)

    if target.kind == "hit_miss":
        return hit_miss_payoff(sig, close, target.horizon, target.benchmark)

    raise ValueError(f"unknown target kind: {target.kind}")


# ---------------------------------------------------------------------------
# Core OOS evaluation
# ---------------------------------------------------------------------------
def _stack_payoffs(
    signal_fn: SignalFn,
    prices_by_symbol: Mapping[str, pd.DataFrame],
    params: Mapping[str, Any],
    target: TargetSpec,
) -> pd.Series:
    """Concatenate per-symbol observation payoffs; index = MultiIndex(symbol, date)."""
    chunks: List[pd.Series] = []
    for sym, hist in prices_by_symbol.items():
        if hist is None or getattr(hist, "empty", True):
            continue
        try:
            raw = signal_fn(str(sym), hist, params)
        except Exception as e:
            logger.debug("signal_fn failed for %s: %s", sym, e)
            continue
        if raw is None:
            continue
        pay = observation_payoffs(pd.Series(raw), hist, target).dropna()
        if pay.empty:
            continue
        pay = pay.copy()
        pay.index = pd.MultiIndex.from_product(
            [[str(sym).upper()], pay.index], names=["symbol", "date"]
        )
        chunks.append(pay)
    if not chunks:
        return pd.Series(dtype=float)
    return pd.concat(chunks).sort_index()


def evaluate_trials_purged_oos(
    signal_fn: SignalFn,
    prices_by_symbol: Mapping[str, pd.DataFrame],
    config: SignalEdgeConfig,
) -> Dict[str, Any]:
    """Run predeclared trials through purge + DSR + recommend_live gate.

    Observations are ordered by date (pooled across the universe). Purge is
    in **calendar positions of the unique sorted date index**, matching the
    bar-embargo pattern used by WalkForwardValidator.
    """
    trials = list(config.trials)
    target = config.target
    purge = int(config.purge_bars) if config.purge_bars is not None else int(target.horizon)

    out: Dict[str, Any] = {
        "success": False,
        "signal_name": config.signal_name,
        "disclosure": config.disclosure,
        "target": {
            "kind": target.kind,
            "horizon": int(target.horizon),
            "benchmark": target.benchmark if target.kind == "hit_miss" else None,
            "vol_window": target.vol_window,
        },
        "universe": [str(s).upper() for s in config.universe],
        "predeclared_trials": [t.as_dict() for t in trials],
        "trial_justification": config.trial_justification,
        "ordering_note": (
            "Trial set and justification were fixed before this run; "
            "not selected after inspecting DSR."
        ),
        "purge_bars": purge,
        "train_frac": float(config.train_frac),
        "n_trials": len(trials),
        "recommend_live": DEFAULT_RECOMMEND_LIVE,
        "deflated_sharpe": None,
        "champion": None,
        "test": None,
        "trials": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "note": None,
    }
    if config.extra:
        out["extra"] = dict(config.extra)

    if not trials:
        out["error"] = "predeclared_trials required"
        return out
    if not config.trial_justification or not str(config.trial_justification).strip():
        out["error"] = "trial_justification required (lock rationale before running)"
        return out

    # Build date universe from any symbol history
    all_dates: Optional[pd.DatetimeIndex] = None
    for hist in prices_by_symbol.values():
        if hist is None or getattr(hist, "empty", True):
            continue
        idx = pd.to_datetime(_close_series(hist).index).tz_localize(None)
        all_dates = idx if all_dates is None else all_dates.union(idx)
    if all_dates is None or len(all_dates) < 40:
        out["error"] = "insufficient price history across universe"
        return out
    all_dates = all_dates.sort_values().unique()
    n_dates = len(all_dates)
    train_end, test_start = purged_split_indices(
        n_dates, purge, train_frac=float(config.train_frac)
    )
    if train_end < 20 or test_start >= n_dates - 5:
        out["error"] = "not enough history for purged train/test split"
        return out

    train_dates = set(all_dates[:train_end])
    test_dates = set(all_dates[test_start:])
    out["train_end_idx"] = int(train_end)
    out["test_start_idx"] = int(test_start)
    out["n_common_dates"] = int(n_dates)
    out["date_range"] = {
        "start": str(all_dates[0].date()),
        "end": str(all_dates[-1].date()),
        "train_end": str(all_dates[train_end - 1].date()) if train_end else None,
        "test_start": str(all_dates[test_start].date())
        if test_start < n_dates
        else None,
    }

    trial_rows: List[Dict[str, Any]] = []
    scores: List[float] = []

    for trial in trials:
        params = dict(trial.params)
        pay = _stack_payoffs(signal_fn, prices_by_symbol, params, target)
        if pay.empty:
            st = observation_stats(np.array([]))
            score = FAILED_TRIAL_SCORE
            n_tr = 0
        else:
            dates = pay.index.get_level_values("date")
            train_pay = pay[dates.isin(train_dates)].to_numpy(dtype=float)
            st = observation_stats(train_pay)
            score = (
                float(st["sharpe"])
                if st.get("sharpe") is not None
                else FAILED_TRIAL_SCORE
            )
            n_tr = int(st.get("n") or 0)
        row = {
            **trial.as_dict(),
            "train_stats": st,
            "n_obs": n_tr,
            "score": score,
        }
        trial_rows.append(row)
        scores.append(score)

    out["trials"] = trial_rows
    best = max(trial_rows, key=lambda r: float(r["score"])) if trial_rows else None
    if best is None or float(best["score"]) <= FAILED_TRIAL_SCORE + 1:
        out["success"] = True
        out["note"] = "No viable train-window observations — null result."
        return out

    champ_params = {
        k: v
        for k, v in best.items()
        if k not in ("train_stats", "n_obs", "score", "label")
    }
    out["champion"] = {
        "params": champ_params,
        "label": best.get("label"),
        "train_stats": best.get("train_stats"),
    }

    if int(best.get("n_obs") or 0) < int(config.min_train_obs):
        out["success"] = True
        out["note"] = (
            f"Champion train n={best.get('n_obs')} < min_train_obs="
            f"{config.min_train_obs} — null / underpowered."
        )
        return out

    test_pay = _stack_payoffs(signal_fn, prices_by_symbol, champ_params, target)
    if test_pay.empty:
        test_stats = observation_stats(np.array([]))
    else:
        dates = test_pay.index.get_level_values("date")
        test_stats = observation_stats(
            test_pay[dates.isin(test_dates)].to_numpy(dtype=float)
        )

    n_obs = int(test_stats.get("n") or 0)
    obs_sr = test_stats.get("sharpe")
    dsr = compute_deflated_sharpe(obs_sr, scores, n_obs)
    recommend = recommend_live_gate(dsr, obs_sr, n_obs)

    if n_obs < int(config.min_test_obs) and not recommend:
        # Still report numbers; gate already false when n < 10
        pass

    out.update(
        {
            "success": True,
            "champion": {
                "params": champ_params,
                "label": best.get("label"),
                "train_stats": best.get("train_stats"),
            },
            "test": {"n_obs": n_obs, "stats": test_stats},
            "deflated_sharpe": dsr,
            "recommend_live": recommend,
            "note": (
                "Champion cleared DSR>=0.95 on OOS — still research; not auto-wired."
                if recommend
                else (
                    "Null / not significant on OOS+DSR — leave research-only "
                    "(acceptable expected outcome)."
                )
            ),
        }
    )
    return out


def run_signal_edge_oos(
    signal_fn: SignalFn,
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    signal_name: str,
    trials: Sequence[Union[TrialSpec, Mapping[str, Any]]],
    trial_justification: str,
    target: TargetSpec,
    universe: Optional[Sequence[str]] = None,
    purge_bars: Optional[int] = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    min_train_obs: int = 30,
    min_test_obs: int = MIN_OOS_OBS,
    disclosure: str = DISCLOSURE,
    extra: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Public entry — standardized purge + DSR + JSON-shaped result.

    Parameters
    ----------
    signal_fn
        Causal scorer: ``(symbol, history, params) -> Series``.
    prices_by_symbol
        OHLCV (or at least Close) panels keyed by symbol.
    trials
        Predeclared ``TrialSpec`` or raw param dicts — locked before run.
    trial_justification
        Required written rationale (PEAD-style).
    target
        ``TargetSpec`` for forward_return / forward_realized_vol / hit_miss.
    min_train_obs / min_test_obs
        Floor for underpowered nulls; live gate still requires ``MIN_OOS_OBS``.
    """
    trial_objs: List[TrialSpec] = []
    for t in trials:
        if isinstance(t, TrialSpec):
            trial_objs.append(t)
        else:
            trial_objs.append(TrialSpec(params=dict(t)))

    univ = list(universe) if universe is not None else list(prices_by_symbol.keys())
    cfg = SignalEdgeConfig(
        signal_name=signal_name,
        trials=trial_objs,
        trial_justification=trial_justification,
        target=target,
        universe=univ,
        purge_bars=purge_bars,
        train_frac=train_frac,
        min_train_obs=int(min_train_obs),
        min_test_obs=int(min_test_obs),
        disclosure=disclosure,
        extra=dict(extra or {}),
    )
    # Restrict price map to requested universe when provided
    prices = {
        str(s).upper(): prices_by_symbol[s]
        for s in univ
        if s in prices_by_symbol
    }
    # Also try uppercase keys
    if not prices:
        prices = {
            str(k).upper(): v
            for k, v in prices_by_symbol.items()
            if str(k).upper() in {str(u).upper() for u in univ}
        }

    result = evaluate_trials_purged_oos(signal_fn, prices, cfg)

    if out_path:
        try:
            import json
            from pathlib import Path

            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8",
            )
            result["wrote"] = str(path)
        except Exception as e:
            result["write_error"] = str(e)

    return result


__all__ = [
    "DISCLOSURE",
    "DSR_LIVE_THRESHOLD",
    "MIN_OOS_OBS",
    "FAILED_TRIAL_SCORE",
    "TargetSpec",
    "TrialSpec",
    "SignalEdgeConfig",
    "purged_split_indices",
    "observation_stats",
    "recommend_live_gate",
    "compute_deflated_sharpe",
    "forward_returns",
    "forward_realized_vol",
    "hit_miss_payoff",
    "observation_payoffs",
    "evaluate_trials_purged_oos",
    "run_signal_edge_oos",
]
