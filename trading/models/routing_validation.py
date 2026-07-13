# -*- coding: utf-8 -*-
"""FFORMA-lite routing research + live forecast policy.

Phase 3: per-symbol OOS validation of feature-based rules (research only).
Phase 4: live path applies registry *eligibility* only. Feature-routing rules
stay off until a rule clears the Phase-3 broad-win gate — none have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trading.models.forecast_router import get_series_features
from trading.optimization.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)

# Same family as Analyze/Backtest fast path (price voters; GARCH left out
# of the routing study because it is a volatility model).
DEFAULT_ENSEMBLE: List[str] = [
    "arima", "xgboost", "ridge", "catboost", "prophet",
]

# Router ids ↔ registry PascalCase names (filter_eligible_models uses registry)
ROUTER_TO_REGISTRY: Dict[str, str] = {
    "arima": "ARIMA",
    "xgboost": "XGBoost",
    "ridge": "Ridge",
    "catboost": "CatBoost",
    "prophet": "Prophet",
    "garch": "GARCH",
    "lstm": "LSTM",
    "tcn": "TCN",
    "hybrid": "Hybrid",
    "transformer": "Transformer",
    "ensemble": "Ensemble",
    "gnn": "GNN",
    "n-beats": "N-BEATS",
    "n-hits": "N-HiTS",
    "patchtst": "PatchTST",
    "tft": "TFT",
}
REGISTRY_TO_ROUTER: Dict[str, str] = {v: k for k, v in ROUTER_TO_REGISTRY.items()}

# Phase 3 produced zero always-on candidates. Do not flip this without a
# fresh validate_routing_universe() broad-win on real OOS evidence.
LIVE_FEATURE_ROUTING_ENABLED = False
LIVE_ALWAYS_ON_RULES: List[str] = []

# self_tune-style absolute margin on directional accuracy (higher=better)
DEFAULT_MIN_IMPROVEMENT = 0.05

ForecastFn = Callable[[pd.DataFrame, int, Sequence[str]], np.ndarray]


def live_forecast_policy(
    data: pd.DataFrame,
    requested_models: Optional[Sequence[str]] = None,
    *,
    n_assets: int = 1,
    apply_feature_rules: bool = LIVE_FEATURE_ROUTING_ENABLED,
) -> Dict[str, Any]:
    """Build the live model list: eligibility ∩ request; rules off by default.

    Returns dict with:
      models: list[str] router ids to run
      features: Phase-1 series features
      excluded_ineligible: requested models dropped by registry metadata
      feature_routing_applied: bool (False until Phase-3 broad win)
      always_on_rules: list (empty — none validated)
      justification: plain-language honesty string for the UI
    """
    from trading.models.model_registry import filter_eligible_models

    requested = [
        str(m).strip().lower()
        for m in (requested_models or DEFAULT_ENSEMBLE)
        if m and str(m).strip()
    ]
    features = get_series_features(data)
    n_points = int(features.get("data_length") or len(data) or 0)

    eligible_reg = filter_eligible_models(
        features=features,
        available_data_points=n_points,
        n_assets=int(n_assets),
    )
    eligible_router = {
        REGISTRY_TO_ROUTER[n]
        for n in eligible_reg
        if n in REGISTRY_TO_ROUTER
    }
    # Also accept registry names that already match lowercase keys
    for n in eligible_reg:
        eligible_router.add(n.lower().replace(" ", "-"))

    models = [m for m in requested if m in eligible_router]
    excluded = [m for m in requested if m not in eligible_router]
    # Never return empty if something was requested — fall back to ridge/arima
    # among eligible, else raw requested (router will fail soft per model).
    if not models:
        for fb in ("ridge", "arima", "xgboost"):
            if fb in eligible_router:
                models = [fb]
                break
        if not models:
            models = list(requested[:3]) or list(DEFAULT_ENSEMBLE[:3])

    feature_routing_applied = False
    active_rule = None
    if apply_feature_rules and LIVE_ALWAYS_ON_RULES:
        # Reserved for Phase-3 winners only — currently unreachable.
        for rule in CANDIDATE_RULES:
            if rule.rule_id not in LIVE_ALWAYS_ON_RULES:
                continue
            chosen = rule.select(features)
            if chosen:
                models = [m for m in chosen if m in eligible_router] or models
                feature_routing_applied = True
                active_rule = rule.rule_id
                break

    justification = (
        "Eligibility filter only (min data / asset count). "
        "No feature-routing rule is live — Phase 3 found no broad OOS winner "
        f"(always_on={LIVE_ALWAYS_ON_RULES or []})."
    )
    if excluded:
        justification += f" Dropped ineligible: {', '.join(excluded)}."

    return {
        "models": models,
        "features": features,
        "excluded_ineligible": excluded,
        "feature_routing_applied": feature_routing_applied,
        "active_rule": active_rule,
        "always_on_rules": list(LIVE_ALWAYS_ON_RULES),
        "justification": justification,
    }


@dataclass(frozen=True)
class RoutingRule:
    """Feature gate → model subset. Descriptive id only — not live policy."""

    rule_id: str
    description: str
    # (features) -> model list when the rule fires; None = use default
    select: Callable[[Dict[str, Any]], Optional[List[str]]]


def _rule_arima_when_noisy(features: Dict[str, Any]) -> Optional[List[str]]:
    if (
        float(features.get("trend_strength") or 0.0) < 0.30
        and float(features.get("noise_entropy") or 0.0) > 0.55
    ):
        return ["arima", "ridge"]
    return None


def _rule_xgb_when_trendy(features: Dict[str, Any]) -> Optional[List[str]]:
    if float(features.get("trend_strength") or 0.0) > 0.50:
        return ["xgboost", "ridge"]
    return None


def _rule_prophet_when_seasonal(features: Dict[str, Any]) -> Optional[List[str]]:
    if float(features.get("seasonality_strength") or 0.0) > 0.30:
        return ["prophet", "arima"]
    return None


CANDIDATE_RULES: List[RoutingRule] = [
    RoutingRule(
        "arima_when_noisy",
        "Prefer ARIMA+Ridge when trend_strength low and noise_entropy high",
        _rule_arima_when_noisy,
    ),
    RoutingRule(
        "xgb_when_trendy",
        "Prefer XGBoost+Ridge when trend_strength high",
        _rule_xgb_when_trendy,
    ),
    RoutingRule(
        "prophet_when_seasonal",
        "Prefer Prophet+ARIMA when seasonality_strength high",
        _rule_prophet_when_seasonal,
    ),
]


def models_for_rule(
    rule: Optional[RoutingRule],
    features: Dict[str, Any],
    default_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve the model list for one decision point (inclusion only)."""
    base = list(default_models or DEFAULT_ENSEMBLE)
    if rule is None:
        return base
    chosen = rule.select(features)
    return list(chosen) if chosen else base


def _close_col(df: pd.DataFrame) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    return col_map.get("close") or list(df.columns)[0]


def _default_consensus_forecast(
    train: pd.DataFrame,
    horizon: int,
    models: Sequence[str],
) -> np.ndarray:
    """Live router consensus — slow; used only when no forecast_fn injected."""
    from trading.models.forecast_router import get_router_singleton

    router = get_router_singleton()
    fc = router.get_consensus_forecast(
        data=train,
        horizon=int(horizon),
        models=list(models),
        model_configs={"arima": {"fast_mode": True}},
    )
    path = fc.get("consensus_forecast") or fc.get("forecast") or []
    arr = np.asarray(path, dtype=float).ravel()
    return arr


def purged_ensemble_oos(
    data: pd.DataFrame,
    *,
    models: Optional[Sequence[str]] = None,
    rule: Optional[RoutingRule] = None,
    default_models: Optional[Sequence[str]] = None,
    horizon: int = 7,
    train_window: int = 120,
    test_window: int = 40,
    step_size: int = 20,
    purge: Optional[int] = None,
    forecast_fn: Optional[ForecastFn] = None,
) -> Dict[str, Any]:
    """Walk-forward OOS for a fixed model list with train/test purge gap.

    If ``rule`` is set, models are chosen each step from
    ``get_series_features(train_ctx)`` (causal). Otherwise ``models`` is used
    as a fixed list.

    ``purge`` defaults to ``horizon`` (regime-paper baseline). Metric:
    mean directional accuracy (higher better) and mean MAPE.
    """
    if data is None or getattr(data, "empty", True):
        return {"error": "no data", "directional_accuracy": None, "mape": None}

    purge_i = int(horizon if purge is None else purge)
    purge_i = max(0, purge_i)
    predict = forecast_fn or _default_consensus_forecast
    close = _close_col(data)
    n = len(data)
    need = train_window + purge_i + test_window
    if n < need:
        return {
            "error": f"insufficient data ({n} < {need})",
            "directional_accuracy": None,
            "mape": None,
            "n_windows": 0,
            "purge": purge_i,
        }

    base = list(default_models or models or DEFAULT_ENSEMBLE)
    fixed = None if rule is not None else list(models or base)

    das: List[float] = []
    mapes: List[float] = []
    window_scores: List[float] = []
    models_used_hist: List[str] = []
    n_rule_fires = 0
    start = train_window
    windows = 0

    while start + purge_i + test_window <= n:
        # Expanding train (matches WalkForwardValidator default)
        train = data.iloc[0:start].copy()
        test = data.iloc[start + purge_i: start + purge_i + test_window].copy()

        for i in range(0, len(test), horizon):
            ctx = (
                pd.concat([train, test.iloc[:i]]) if i > 0 else train
            )
            if len(ctx) < 30:
                continue
            if rule is not None:
                feats = get_series_features(ctx)
                raw = models_for_rule(rule, feats, base)
                base_set = set(base)
                step_models = [m for m in raw if m in base_set] or list(base)
                if step_models != base:
                    n_rule_fires += 1
            else:
                step_models = list(fixed or base)
            models_used_hist.append(",".join(step_models))
            try:
                pred = np.asarray(
                    predict(ctx, horizon, step_models), dtype=float
                ).ravel()
            except Exception as e:
                logger.debug("purged_ensemble_oos forecast failed: %s", e)
                continue
            actual = test.iloc[i: i + horizon][close].to_numpy(dtype=float)
            m = min(len(pred), len(actual))
            if m < 1:
                continue
            pred, actual = pred[:m], actual[:m]
            step_hits: List[float] = []
            for j in range(m):
                anchor = float(ctx.iloc[-1][close]) if j == 0 else float(actual[j - 1])
                if not np.isfinite(anchor) or anchor == 0:
                    continue
                hit = 1.0 if np.sign(pred[j] - anchor) == np.sign(actual[j] - anchor) else 0.0
                das.append(hit)
                step_hits.append(hit)
                if actual[j] != 0:
                    mapes.append(abs((pred[j] - actual[j]) / actual[j]) * 100.0)
            if step_hits:
                window_scores.append(float(np.mean(step_hits)))
            windows += 1

        start += step_size

    if not das:
        return {
            "error": "no scored steps",
            "directional_accuracy": None,
            "mape": None,
            "n_windows": 0,
            "purge": purge_i,
            "n_rule_fires": n_rule_fires,
        }

    return {
        "directional_accuracy": float(np.mean(das)),
        "mape": float(np.mean(mapes)) if mapes else None,
        "n_steps": len(das),
        "n_windows": windows,
        "purge": purge_i,
        "horizon": int(horizon),
        "models": fixed if fixed is not None else "rule-adaptive",
        "models_used_hist_tail": models_used_hist[-5:],
        "n_rule_fires": n_rule_fires,
        "window_scores": window_scores,
    }


@dataclass
class SymbolRuleResult:
    symbol: str
    rule_id: str
    baseline_da: Optional[float]
    challenger_da: Optional[float]
    delta_da: Optional[float]
    helped: bool
    hurt: bool
    adopted_for_symbol: bool
    reason: str
    features: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    challenger: Dict[str, Any] = field(default_factory=dict)


def evaluate_symbol_rules(
    symbol: str,
    data: pd.DataFrame,
    *,
    rules: Optional[Sequence[RoutingRule]] = None,
    default_models: Optional[Sequence[str]] = None,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    forecast_fn: Optional[ForecastFn] = None,
    horizon: int = 5,
    train_window: int = 80,
    test_window: int = 30,
    step_size: int = 15,
    causal_features: bool = True,
) -> List[SymbolRuleResult]:
    """Champion (default ensemble) vs each rule on one symbol, purged OOS.

    When ``causal_features`` is True (default), each walk-forward step
    recomputes ``get_series_features(train_ctx)`` before applying the rule.
    """
    rules = list(rules or CANDIDATE_RULES)
    default_models = list(default_models or DEFAULT_ENSEMBLE)
    features = get_series_features(data)

    baseline = purged_ensemble_oos(
        data,
        models=default_models,
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        purge=horizon,
        forecast_fn=forecast_fn,
    )
    base_da = baseline.get("directional_accuracy")

    out: List[SymbolRuleResult] = []
    for rule in rules:
        if causal_features:
            chall = purged_ensemble_oos(
                data,
                rule=rule,
                default_models=default_models,
                horizon=horizon,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                purge=horizon,
                forecast_fn=forecast_fn,
            )
            fired = int(chall.get("n_rule_fires") or 0) > 0
        else:
            models = models_for_rule(rule, features, default_models)
            fired = models != default_models
            chall = (
                baseline
                if not fired
                else purged_ensemble_oos(
                    data,
                    models=models,
                    horizon=horizon,
                    train_window=train_window,
                    test_window=test_window,
                    step_size=step_size,
                    purge=horizon,
                    forecast_fn=forecast_fn,
                )
            )
        chall_da = chall.get("directional_accuracy")
        delta = None
        if base_da is not None and chall_da is not None:
            delta = float(chall_da) - float(base_da)

        helped = bool(
            delta is not None and delta > abs(min_improvement)
        )
        hurt = bool(
            delta is not None and delta < -abs(min_improvement)
        )
        adopted = helped  # per-symbol only; not global always-on
        if not fired:
            reason = "rule did not fire on any purged window (identical to baseline)"
            adopted = False
            helped = False
            hurt = False
        elif base_da is None or chall_da is None:
            reason = "insufficient OOS scores"
            adopted = False
        elif helped:
            reason = (
                f"challenger beat baseline DA by {delta:.3f} "
                f"(margin {min_improvement})"
            )
        elif hurt:
            reason = (
                f"challenger worse than baseline DA by {delta:.3f} "
                f"(beyond margin {min_improvement})"
            )
        else:
            reason = (
                f"delta_da={delta:.3f} within ±{min_improvement} — keep baseline"
            )

        out.append(
            SymbolRuleResult(
                symbol=symbol,
                rule_id=rule.rule_id,
                baseline_da=base_da,
                challenger_da=chall_da,
                delta_da=delta,
                helped=helped,
                hurt=hurt,
                adopted_for_symbol=adopted,
                reason=reason,
                features=dict(features),
                baseline=dict(baseline),
                challenger=dict(chall),
            )
        )
    return out


def summarize_universe(
    results: Sequence[SymbolRuleResult],
    *,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    min_help_fraction: float = 0.67,
) -> Dict[str, Any]:
    """Aggregate helped/hurt; recommend always_on only on a broad win.

    Broad win = helped (with margin) on ≥ ``min_help_fraction`` of symbols
    where the rule fired, AND mean delta_da > min_improvement.
    """
    by_rule: Dict[str, List[SymbolRuleResult]] = {}
    for r in results:
        by_rule.setdefault(r.rule_id, []).append(r)

    rule_summaries = []
    always_on_candidates: List[str] = []

    trial_deltas: List[float] = []
    for rule_id, rows in by_rule.items():
        fired = [
            r for r in rows
            if "did not fire" not in r.reason and r.delta_da is not None
        ]
        helped_n = sum(1 for r in fired if r.helped)
        hurt_n = sum(1 for r in fired if r.hurt)
        deltas = [float(r.delta_da) for r in fired if r.delta_da is not None]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        help_frac = (helped_n / len(fired)) if fired else 0.0
        trial_deltas.extend(deltas)

        broad_win = bool(
            fired
            and help_frac >= min_help_fraction
            and mean_delta > abs(min_improvement)
        )
        if broad_win:
            always_on_candidates.append(rule_id)

        # DSR treating per-symbol delta_da as trial scores (honest noise check)
        dsr = None
        if len(deltas) >= 2:
            # Map DA deltas into a Sharpe-like proxy: mean/std of deltas
            mu = float(np.mean(deltas))
            sig = float(np.std(deltas, ddof=1)) or 1e-12
            observed = mu / sig
            dsr = deflated_sharpe_ratio(
                observed_sr=observed,
                trial_scores=deltas,
                n_obs=len(deltas),
            )

        rule_summaries.append({
            "rule_id": rule_id,
            "n_symbols": len(rows),
            "n_fired": len(fired),
            "helped": helped_n,
            "hurt": hurt_n,
            "neutral": max(0, len(fired) - helped_n - hurt_n),
            "help_fraction": round(help_frac, 3),
            "mean_delta_da": round(mean_delta, 4),
            "broad_win": broad_win,
            "recommend_always_on": broad_win,
            "dsr": dsr,
            "per_symbol": [
                {
                    "symbol": r.symbol,
                    "delta_da": r.delta_da,
                    "helped": r.helped,
                    "hurt": r.hurt,
                    "reason": r.reason,
                }
                for r in rows
            ],
        })

    return {
        "min_improvement": min_improvement,
        "min_help_fraction": min_help_fraction,
        "always_on_candidates": always_on_candidates,
        "note": (
            "No rule ships as live always-on unless recommend_always_on "
            "is true. Mixed helped/hurt across symbols is the expected "
            "outcome (regime paper)."
            if not always_on_candidates
            else "One or more rules cleared the broad-win gate — still "
            "requires Phase 4 review before live wiring."
        ),
        "rules": rule_summaries,
    }


def validate_routing_universe(
    series: Dict[str, pd.DataFrame],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run evaluate_symbol_rules for each symbol and summarize."""
    all_rows: List[SymbolRuleResult] = []
    for sym, df in series.items():
        all_rows.extend(evaluate_symbol_rules(sym, df, **kwargs))
    summary = summarize_universe(
        all_rows,
        min_improvement=float(kwargs.get("min_improvement", DEFAULT_MIN_IMPROVEMENT)),
    )
    return {
        "success": True,
        "n_symbols": len(series),
        "results": [
            {
                "symbol": r.symbol,
                "rule_id": r.rule_id,
                "baseline_da": r.baseline_da,
                "challenger_da": r.challenger_da,
                "delta_da": r.delta_da,
                "helped": r.helped,
                "hurt": r.hurt,
                "adopted_for_symbol": r.adopted_for_symbol,
                "reason": r.reason,
            }
            for r in all_rows
        ],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Synthetic universe for offline / unit demonstration of MIXED outcomes
# ---------------------------------------------------------------------------

def make_synthetic_universe(n: int = 160, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Three hand-shaped series: noisy, trendy, seasonal."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    def _ohlc(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(1e6, 3e6, n).astype(float),
        }, index=idx)

    t = np.arange(n, dtype=float)
    noisy = 100.0 + rng.normal(0.0, 2.0, n).cumsum() * 0.15
    trendy = 100.0 * np.exp(0.0012 * t + rng.normal(0, 0.001, n))
    seasonal = 100.0 + 4.0 * np.sin(2 * np.pi * t / 5.0) + rng.normal(0, 0.15, n)
    return {
        "NOISY": _ohlc(noisy),
        "TRENDY": _ohlc(trendy),
        "SEASONAL": _ohlc(seasonal),
    }


def stylized_forecast_fn(
    train: pd.DataFrame,
    horizon: int,
    models: Sequence[str],
) -> np.ndarray:
    """Oracle-ish predictor: accuracy depends on which models are selected.

    Used only for offline Phase-3 demos / tests — not live trading.
    - arima/ridge subsets track last price (good on noise)
    - xgboost subsets extrapolate recent drift (good on trend)
    - prophet subsets replay lag-5 seasonal step (good on seasonal)
    - full ensemble averages the three → middling everywhere
    """
    close = train[_close_col(train)].to_numpy(dtype=float)
    last = float(close[-1])
    h = int(horizon)
    models_l = {m.lower() for m in models}

    # last-value / mild mean reversion
    arima_path = np.full(h, last, dtype=float)
    # drift from last 10 returns
    if len(close) >= 11:
        drift = float(np.mean(np.diff(close[-11:])))
    else:
        drift = 0.0
    xgb_path = last + drift * np.arange(1, h + 1, dtype=float)
    # weekly seasonal continuation from last 5-step move
    if len(close) >= 6:
        season_step = float(close[-1] - close[-6]) / 5.0
    else:
        season_step = 0.0
    prophet_path = last + season_step * np.arange(1, h + 1, dtype=float)

    paths = []
    if models_l <= {"arima", "ridge"}:
        paths.append(arima_path)
    elif "xgboost" in models_l and "prophet" not in models_l:
        paths.append(xgb_path)
    elif "prophet" in models_l:
        paths.append(prophet_path)
    else:
        # default ensemble: average of the three styles
        paths = [arima_path, xgb_path, prophet_path]
    return np.mean(np.vstack(paths), axis=0)


# Hybrid-core voters for tractable real-ticker OOS (documented in report).
REAL_OOS_ENSEMBLE: List[str] = ["arima", "xgboost", "ridge"]
DEFAULT_REAL_SYMBOLS: List[str] = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]


def load_symbol_frames(
    symbols: Sequence[str],
    period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV via price_cache; skip tickers with insufficient history."""
    from trading.data.price_cache import get_history

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        s = str(sym).strip().upper()
        if not s:
            continue
        try:
            hist = get_history(s, period=period)
        except Exception as e:
            logger.warning("load_symbol_frames %s failed: %s", s, e)
            continue
        if hist is None or getattr(hist, "empty", True) or len(hist) < 200:
            logger.warning("load_symbol_frames %s: insufficient history", s)
            continue
        out[s] = hist
    return out


def run_real_ticker_oos(
    symbols: Optional[Sequence[str]] = None,
    *,
    period: str = "2y",
    models: Optional[Sequence[str]] = None,
    out_path: str = "data/routing_oos_real.json",
    horizon: int = 5,
    train_window: int = 180,
    test_window: int = 5,
    step_size: int = 42,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> Dict[str, Any]:
    """Purged causal OOS on real tickers; never flips live always-on flags.

    Uses one forecast per outer window (``test_window == horizon``) and the
    hybrid-core ensemble by default so the pass finishes in reasonable time.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    syms = list(symbols or DEFAULT_REAL_SYMBOLS)
    ensemble = list(models or REAL_OOS_ENSEMBLE)
    series = load_symbol_frames(syms, period=period)
    if not series:
        return {"success": False, "error": "no symbol history loaded"}

    logger.info(
        "real ticker OOS: %d symbols, models=%s, horizon=%d purge=%d",
        len(series), ensemble, horizon, horizon,
    )
    report = validate_routing_universe(
        series,
        default_models=ensemble,
        min_improvement=min_improvement,
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        forecast_fn=None,  # live router consensus
        # validate_routing_universe -> evaluate_symbol_rules causal default
    )
    report["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols_requested": syms,
        "symbols_loaded": list(series.keys()),
        "ensemble": ensemble,
        "period": period,
        "horizon": horizon,
        "purge": horizon,
        "train_window": train_window,
        "test_window": test_window,
        "step_size": step_size,
        "min_improvement": min_improvement,
        "causal_features": True,
        "live_routing_enabled": LIVE_FEATURE_ROUTING_ENABLED,
        "note": (
            "Research pass only. LIVE_ALWAYS_ON_RULES is not modified; "
            "enable a rule only after broad_win on this (or a larger) universe."
        ),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["meta"]["out_path"] = str(path)
    return report


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")
    _rep = run_real_ticker_oos()
    _sum = (_rep or {}).get("summary") or {}
    print("always_on_candidates:", _sum.get("always_on_candidates"))
    print("note:", _sum.get("note"))
    for _rule in _sum.get("rules") or []:
        print(
            f"{_rule['rule_id']}: helped={_rule['helped']} hurt={_rule['hurt']} "
            f"neutral={_rule['neutral']} mean_dDA={_rule['mean_delta_da']} "
            f"broad={_rule['broad_win']}"
        )
        for _p in _rule.get("per_symbol") or []:
            print(
                f"  {_p['symbol']}: dDA={_p['delta_da']} "
                f"helped={_p['helped']} hurt={_p['hurt']} | {_p['reason'][:80]}"
            )
