"""Bridge between the strategy registry and the optimization cluster.

The optimization cluster (grid search / genetic / PSO / Bayesian via
``StrategyOptimizer``) was verified working end-to-end by the audit but was
never wired into the live app. This module supplies the missing piece: a
robust, parameterized strategy runner and an objective-function factory so
any registered strategy can be optimized against real OHLCV data.

Why the registry's own parameter path isn't used directly
---------------------------------------------------------
``StrategyRegistry.execute_strategy(name, data, parameters)`` has three
verified-by-execution incompatibilities across the built-in strategies:

* ``RSIStrategy.set_parameters(**kwargs)`` takes keyword args, but the
  registry calls ``set_parameters(parameters)`` positionally -> TypeError.
* ``ATRStrategy``/``CCIStrategy.generate_signals(data)`` accept no
  ``**kwargs``, but the registry forwards ``**parameters`` -> TypeError.
* ``ATRStrategy``/``CCIStrategy`` require lowercase OHLCV column names and
  raise on the capitalized columns yfinance returns, so they fail in the
  live Backtest tab today.

:func:`run_strategy` sidesteps all three: it normalizes columns, constructs
a *fresh* strategy instance per call (no state leaks into the registry's
singletons), and applies parameters by writing them onto the strategy's
config/attributes directly - which every built-in strategy supports.

Backtest convention
-------------------
Signals are treated as *position intents*: a nonzero signal opens/flips a
position which is then **held until the next opposite signal** ("hold"
mode). Positions act with a one-bar delay (signal on bar t -> position on
bar t+1) so there is no lookahead. Per-side transaction costs (bps) are
charged on position changes, which also stops optimizers from favoring
pathologically hyperactive parameter sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy_param_spaces import (
    ParamSpec,
    constraint_violations,
    get_default_params,
    get_param_specs,
    to_optimizer_space,
)

logger = logging.getLogger(__name__)

# Large finite penalty for invalid/failed parameter sets. Finite (not inf)
# for the same reason the audit fixed the early-stopping sentinel: skopt's
# Gaussian Process cannot fit non-finite objective values.
INVALID_PENALTY = 1e9

METRIC_CHOICES = {
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "calmar_ratio": "Calmar ratio",
    "total_return": "Total return",
    "max_drawdown": "Max drawdown (minimize)",
}

# Metrics where bigger is better (objective returns their negative).
_MAXIMIZE = {"sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return"}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy with exactly one, lowercase set of OHLCV columns.

    Built-in strategies disagree on casing: ATR/CCI strictly require
    lowercase and raise on yfinance's Capitalized columns, while the others
    lowercase internally (and therefore *break* if both casings are present
    at once - duplicate columns after their internal ``str.lower()`` make
    ``df["close"]`` return a DataFrame). Verified by execution: canonical
    lowercase-only is the one representation all six built-ins accept.
    """
    lower_map: Dict[str, Any] = {}
    for c in df.columns:
        key = str(c).lower()
        # First occurrence wins if a frame somehow has both casings.
        lower_map.setdefault(key, c)
    cols = {}
    for name in ("open", "high", "low", "close", "volume"):
        src = lower_map.get(name)
        if src is not None:
            cols[name] = df[src]
    out = pd.DataFrame(cols, index=df.index)
    return out


def apply_parameters(strategy: Any, params: Dict[str, Any]) -> None:
    """Write parameters onto a strategy instance, wherever it keeps them.

    Order of precedence per parameter: a matching field on
    ``strategy.config`` (MACD/Bollinger/SMA/ATR/CCI), else a matching
    attribute on the strategy itself (RSI). Unknown names are logged and
    skipped rather than raised, so a shared space can carry extras.
    """
    cfg = getattr(strategy, "config", None)
    for name, value in params.items():
        if cfg is not None and hasattr(cfg, name):
            setattr(cfg, name, value)
        elif hasattr(strategy, name):
            setattr(strategy, name, value)
        else:
            logger.warning(
                "%s has no parameter %r; skipping",
                type(strategy).__name__, name,
            )
    # Invalidate any cached artifacts from a prior run.
    for stale in ("signals", "positions", "smoothed_signals"):
        if hasattr(strategy, stale):
            setattr(strategy, stale, None)


def run_strategy(
    strategy_name: str,
    data: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run a registered strategy on OHLCV data with explicit parameters.

    A fresh instance is constructed per call so parameterized runs never
    mutate the registry's shared singletons.

    Returns the strategy's signals DataFrame (must contain a ``signal``
    column; every built-in strategy produces one).
    """
    from trading.strategies.registry import get_strategy_registry

    registry = get_strategy_registry()
    prototype = registry.get_strategy(strategy_name)
    if prototype is None:
        raise ValueError(f"Strategy '{strategy_name}' not found in registry")
    strategy = type(prototype)()  # all built-ins have zero-arg constructors
    if params:
        apply_parameters(strategy, params)
    return strategy.generate_signals(normalize_ohlcv(data))


def signals_to_returns(
    signals: pd.DataFrame,
    close: pd.Series,
    signal_mode: str = "hold",
    cost_bps: float = 5.0,
) -> pd.Series:
    """Convert a signals DataFrame into a net daily strategy return series.

    Args:
        signals: DataFrame with a ``signal`` column (-1/0/+1 events or
            per-bar stances).
        close: Close price series (same or superset index).
        signal_mode: ``"hold"`` (default) keeps the last nonzero signal as
            the position until an opposite signal; ``"raw"`` uses the
            signal column verbatim as the per-bar position.
        cost_bps: One-way transaction cost in basis points, charged on
            each unit of position change.
    """
    close = close.astype(float)
    if signals is None or signals.empty or "signal" not in signals.columns:
        return pd.Series(0.0, index=close.index)

    sig = signals["signal"].reindex(close.index).fillna(0.0).astype(float)
    if signal_mode == "hold":
        position = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        position = sig
    position = position.clip(-1.0, 1.0)

    # One-bar execution delay: no lookahead.
    lagged = position.shift(1).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)
    gross = lagged * asset_returns

    turnover = lagged.diff().abs().fillna(lagged.abs())
    costs = turnover * (cost_bps / 10_000.0)
    return gross - costs


def evaluate_params(
    strategy_name: str,
    data: pd.DataFrame,
    params: Dict[str, Any],
    cost_bps: float = 5.0,
) -> Dict[str, float]:
    """Run strategy with params and return full performance metrics."""
    from utils.risk_metrics import compute_performance_metrics

    signals = run_strategy(strategy_name, data, params)
    ndata = normalize_ohlcv(data)
    returns = signals_to_returns(signals, ndata["close"], cost_bps=cost_bps)
    if not (returns != 0).any():
        # A strategy that never traded has no meaningful risk metrics.
        # Without this guard, compute_performance_metrics divides the
        # risk-free drag by ~zero volatility and reports Sharpe values in
        # the tens of millions (observed: -31,497,039 for zero trades).
        zero = {k: 0.0 for k in (
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown",
            "volatility_annual", "win_rate", "profit_factor",
        )}
        zero.update({"active_bars": 0, "signal_events": 0})
        return zero
    metrics = compute_performance_metrics(returns)
    n_trades = int((returns != 0).sum())
    position = signals["signal"].reindex(ndata.index)
    out = {
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "volatility_annual": metrics.volatility_annual,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "active_bars": n_trades,
        "signal_events": int((position.fillna(0) != 0).sum()),
    }
    return out


def make_objective(
    strategy_name: str,
    metric: str = "sharpe_ratio",
    cost_bps: float = 5.0,
    min_signal_events: int = 3,
) -> Callable[[Dict[str, Any], pd.DataFrame], float]:
    """Build an ``objective(params, data) -> float`` for the optimizer
    cluster (minimization convention; ratio/return metrics are negated).

    Parameter sets that violate cross-parameter constraints, produce fewer
    than ``min_signal_events`` signals, or crash the strategy score
    :data:`INVALID_PENALTY` so every optimization method treats them as
    "very bad" without breaking (finite for skopt's GP).
    """
    if metric not in METRIC_CHOICES:
        raise ValueError(
            f"Unknown metric {metric!r}; choose from {sorted(METRIC_CHOICES)}"
        )
    specs = {s.name: s for s in get_param_specs(strategy_name)}

    def objective(params: Dict[str, Any], data: pd.DataFrame) -> float:
        # Cast optimizer-proposed floats back to declared types.
        cast = {
            k: (specs[k].cast(v) if k in specs else v) for k, v in params.items()
        }
        if constraint_violations(strategy_name, cast):
            return INVALID_PENALTY
        try:
            results = evaluate_params(strategy_name, data, cast, cost_bps)
        except Exception as e:  # a bad region shouldn't kill the search
            logger.debug("Objective failed for %s %s: %s", strategy_name, cast, e)
            return INVALID_PENALTY
        if results["signal_events"] < min_signal_events:
            return INVALID_PENALTY
        value = results[metric]
        if not np.isfinite(value):
            return INVALID_PENALTY
        if metric == "max_drawdown":
            # Stored as a negative/most-negative-is-worst quantity; minimize
            # its magnitude.
            return abs(value)
        return -value if metric in _MAXIMIZE else value

    return objective


@dataclass
class StrategyOptimizationRun:
    """Everything the UI needs from one optimization run."""

    strategy_name: str
    method: str
    metric: str
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    baseline_params: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    best_score: float
    n_evaluations: int
    optimization_time: float
    convergence_history: List[float] = field(default_factory=list)
    # Out-of-sample validation (populated by optimize_strategy_validated).
    oos_best_metrics: Optional[Dict[str, float]] = None
    oos_baseline_metrics: Optional[Dict[str, float]] = None
    train_range: Optional[str] = None
    test_range: Optional[str] = None


def optimize_strategy_validated(
    strategy_name: str,
    data: pd.DataFrame,
    train_fraction: float = 0.75,
    **kwargs: Any,
) -> StrategyOptimizationRun:
    """Optimize on a training window, then evaluate the winning parameters
    on a held-out test window the search never saw.

    In-sample optimization results systematically overstate performance
    (the search selects for whatever fit the sample, signal or noise).
    This wrapper splits the history chronologically, runs
    :func:`optimize_strategy` on the first ``train_fraction``, then
    re-evaluates both the optimized and the default parameters on the
    remainder. The out-of-sample comparison is the number that deserves
    trust; a large train->test degradation is the classic overfit
    signature and the UI surfaces it as such.
    """
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    ndata = normalize_ohlcv(data)
    if len(ndata) < 120:
        raise ValueError(
            "Need at least 120 bars for a meaningful train/test split"
        )
    split = int(len(ndata) * train_fraction)
    train, test = data.iloc[:split], data.iloc[split:]

    cost_bps = kwargs.get("cost_bps", 5.0)
    run = optimize_strategy(strategy_name, train, **kwargs)

    run.oos_baseline_metrics = evaluate_params(
        strategy_name, test, run.baseline_params, cost_bps
    )
    run.oos_best_metrics = (
        evaluate_params(strategy_name, test, run.best_params, cost_bps)
        if run.best_params
        else dict(run.oos_baseline_metrics)
    )
    run.train_range = f"{ndata.index[0].date()} → {ndata.index[split - 1].date()}"
    run.test_range = f"{ndata.index[split].date()} → {ndata.index[-1].date()}"
    return run


def optimize_strategy(
    strategy_name: str,
    data: pd.DataFrame,
    method: str = "grid_search",
    metric: str = "sharpe_ratio",
    selected_params: Optional[List[str]] = None,
    max_evaluations: int = 150,
    cost_bps: float = 5.0,
    **method_kwargs: Any,
) -> StrategyOptimizationRun:
    """High-level entry point: optimize a registered strategy's parameters.

    Args:
        strategy_name: Name as registered (e.g. ``"RSIStrategy"``).
        data: OHLCV DataFrame (any column casing).
        method: ``grid_search`` | ``genetic`` | ``pso`` | ``bayesian``.
        metric: Key of :data:`METRIC_CHOICES` to optimize.
        selected_params: Restrict tuning to a subset of the strategy's
            canonical space.
        max_evaluations: Evaluation budget passed through to the cluster.
        cost_bps: Per-side transaction cost applied inside the objective.
        **method_kwargs: Extra per-method knobs (population_size, etc.).
    """
    from .strategy_optimizer import StrategyOptimizer

    space = to_optimizer_space(strategy_name, selected_params)
    if not space:
        raise ValueError(
            f"No tunable parameter space is defined for '{strategy_name}'. "
            "Add one to trading/optimization/strategy_param_spaces.py."
        )
    objective = make_objective(strategy_name, metric=metric, cost_bps=cost_bps)

    optimizer = StrategyOptimizer()
    if method == "grid_search":
        # Grid search subsamples via max_points; align it with the budget.
        method_kwargs.setdefault("max_points", max_evaluations)
    elif method == "pso":
        # BUG FIX (verified by execution): without an explicit schedule,
        # PSO ran its internal default (thousands of evaluations) and the
        # budget wrapper merely returned the penalty for everything past
        # max_evaluations - a 30-eval request burned 3000 evaluations, most
        # of them worthless penalty stubs polluting the convergence
        # history. Derive particles x iterations from the budget so the
        # budget is the actual amount of work done.
        n_particles = method_kwargs.setdefault(
            "n_particles", int(min(20, max(6, max_evaluations // 8)))
        )
        method_kwargs.setdefault(
            "n_iterations", int(max(2, max_evaluations // max(n_particles, 1)))
        )
    elif method == "genetic":
        # Same budget-derivation as PSO, for the same reason.
        population = method_kwargs.setdefault(
            "population_size", int(min(24, max(8, max_evaluations // 6)))
        )
        method_kwargs.setdefault(
            "n_generations", int(max(2, max_evaluations // max(population, 1)))
        )
    elif method == "bayesian":
        method_kwargs.setdefault("n_calls", int(max_evaluations))
    # Strategy backtest objectives are noisy; the cluster's default
    # patience of 5-10 kills searches almost immediately. Let a run use a
    # meaningful share of its budget before giving up.
    method_kwargs.setdefault(
        "early_stopping",
        {"patience": max(25, max_evaluations // 3), "min_delta": 1e-4},
    )
    result = optimizer.optimize(
        objective,
        space,
        data,
        method=method,
        max_evaluations=max_evaluations,
        **method_kwargs,
    )

    specs = {s.name: s for s in get_param_specs(strategy_name)}
    raw_best = result.best_params or {}
    best_params = {
        k: (specs[k].cast(v) if k in specs else v) for k, v in raw_best.items()
    }

    baseline_params = get_default_params(strategy_name)
    baseline_metrics = evaluate_params(
        strategy_name, data, baseline_params, cost_bps
    )
    best_metrics = (
        evaluate_params(strategy_name, data, best_params, cost_bps)
        if best_params
        else dict(baseline_metrics)
    )

    return StrategyOptimizationRun(
        strategy_name=strategy_name,
        method=method,
        metric=metric,
        best_params=best_params,
        best_metrics=best_metrics,
        baseline_params=baseline_params,
        baseline_metrics=baseline_metrics,
        best_score=float(result.best_score),
        n_evaluations=len(result.all_scores or []),
        optimization_time=float(getattr(result, "optimization_time", 0.0)),
        convergence_history=list(result.convergence_history or []),
    )
