"""Canonical tunable-parameter spaces for Evolve's built-in strategies.

Single source of truth consumed by:

* the Strategy Optimizer UI (pages/5_Backtest.py "Optimizer" tab), which
  converts these into the ``param_space`` format the optimization cluster
  (grid search / genetic / PSO / Bayesian) expects;
* ``SelfTuningOptimizer`` (via :func:`get_self_tuning_config`), which needs
  ``parameter_bounds`` and ``parameter_steps`` per strategy and was
  previously inert because neither was ever configured anywhere;
* anything else that needs to know which parameters of a strategy are
  meaningfully tunable and over what ranges.

Design notes
------------
* Ranges are deliberately conservative and practitioner-sane (e.g. RSI
  period 5-30, not 2-500). The point of optimization here is refinement
  within a defensible region, not curve-fitting to noise.
* Only parameters that change *signal generation* are exposed. Data-quality
  filters (``min_volume``, ``min_price``) are excluded on purpose: letting
  an optimizer tune those is a classic way to overfit by silently changing
  the tradable universe instead of the strategy.
* Keys match the strategy names as registered by
  ``trading.strategies.registry`` (class names: ``RSIStrategy`` etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "ParamSpec",
    "STRATEGY_PARAM_SPACES",
    "get_param_specs",
    "get_default_params",
    "to_optimizer_space",
    "get_self_tuning_config",
    "constraint_violations",
]


@dataclass(frozen=True)
class ParamSpec:
    """Specification of one tunable strategy parameter."""

    name: str
    kind: str  # "int" or "float"
    low: float
    high: float
    step: float
    default: float
    label: str = ""
    help: str = ""

    def grid_values(self) -> List[Any]:
        """Explicit grid over [low, high] at ``step`` resolution."""
        values: List[Any] = []
        v = self.low
        # Use an epsilon so float accumulation doesn't drop the top value.
        eps = self.step * 1e-6
        while v <= self.high + eps:
            values.append(int(round(v)) if self.kind == "int" else round(v, 6))
            v += self.step
        # De-duplicate while preserving order (int rounding can collide).
        seen = set()
        out = []
        for x in values:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def cast(self, value: Any) -> Any:
        """Cast a raw optimizer-proposed value to this parameter's type."""
        if self.kind == "int":
            return int(round(float(value)))
        return float(value)


# ---------------------------------------------------------------------------
# Canonical spaces
# ---------------------------------------------------------------------------

STRATEGY_PARAM_SPACES: Dict[str, List[ParamSpec]] = {
    "RSIStrategy": [
        ParamSpec("rsi_period", "int", 5, 30, 1, 14, "RSI period",
                  "Lookback for RSI; shorter reacts faster, noisier."),
        ParamSpec("oversold_threshold", "float", 15, 40, 1, 30, "Oversold",
                  "RSI level below which a long signal fires."),
        ParamSpec("overbought_threshold", "float", 60, 85, 1, 70, "Overbought",
                  "RSI level above which a short/exit signal fires."),
    ],
    "MACDStrategy": [
        ParamSpec("fast_period", "int", 5, 20, 1, 12, "Fast EMA"),
        ParamSpec("slow_period", "int", 18, 50, 1, 26, "Slow EMA"),
        ParamSpec("signal_period", "int", 5, 15, 1, 9, "Signal EMA"),
    ],
    "BollingerStrategy": [
        ParamSpec("window", "int", 10, 50, 1, 20, "Window"),
        ParamSpec("num_std", "float", 1.0, 3.0, 0.25, 2.0, "Std bands"),
    ],
    "SMAStrategy": [
        ParamSpec("short_window", "int", 5, 40, 1, 20, "Short SMA"),
        ParamSpec("long_window", "int", 30, 150, 5, 50, "Long SMA"),
        ParamSpec("confirmation_periods", "int", 1, 5, 1, 3, "Confirmation bars"),
    ],
    "ATRStrategy": [
        ParamSpec("period", "int", 7, 30, 1, 14, "ATR period"),
        ParamSpec("multiplier", "float", 1.0, 4.0, 0.25, 2.0, "Band multiplier"),
        ParamSpec("volatility_threshold", "float", 0.005, 0.05, 0.005, 0.02,
                  "Volatility filter"),
    ],
    "CCIStrategy": [
        ParamSpec("period", "int", 10, 40, 1, 20, "CCI period"),
        ParamSpec("oversold_threshold", "float", -200, -50, 10, -100, "Oversold"),
        ParamSpec("overbought_threshold", "float", 50, 200, 10, 100, "Overbought"),
    ],
}

# Cross-parameter validity constraints, per strategy. Each entry is
# (description, callable(params) -> bool) where True means VIOLATED.
_CONSTRAINTS: Dict[str, List] = {
    "RSIStrategy": [
        ("oversold_threshold must be below overbought_threshold",
         lambda p: p.get("oversold_threshold", 30) >= p.get("overbought_threshold", 70)),
    ],
    "MACDStrategy": [
        ("fast_period must be below slow_period",
         lambda p: p.get("fast_period", 12) >= p.get("slow_period", 26)),
    ],
    "SMAStrategy": [
        ("short_window must be below long_window",
         lambda p: p.get("short_window", 20) >= p.get("long_window", 50)),
    ],
    "CCIStrategy": [
        ("oversold_threshold must be below overbought_threshold",
         lambda p: p.get("oversold_threshold", -100) >= p.get("overbought_threshold", 100)),
    ],
}


def get_param_specs(strategy_name: str) -> List[ParamSpec]:
    """Param specs for a strategy, or [] if it has no registered space."""
    return list(STRATEGY_PARAM_SPACES.get(strategy_name, []))


def get_default_params(strategy_name: str) -> Dict[str, Any]:
    """Default parameter values for a strategy."""
    return {
        s.name: (int(s.default) if s.kind == "int" else float(s.default))
        for s in get_param_specs(strategy_name)
    }


def constraint_violations(strategy_name: str, params: Dict[str, Any]) -> List[str]:
    """Return human-readable descriptions of violated constraints (empty = valid)."""
    out = []
    for desc, is_violated in _CONSTRAINTS.get(strategy_name, []):
        try:
            if is_violated(params):
                out.append(desc)
        except Exception:
            # A malformed params dict shouldn't crash validation.
            out.append(f"could not evaluate constraint: {desc}")
    return out


def to_optimizer_space(
    strategy_name: str, selected: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    """Convert to the explicit-grid ``param_space`` format the optimization
    cluster accepts (dict of param -> list of candidate values).

    Explicit value lists are used rather than {"start","end"} dicts because
    they are the one representation every method in the cluster (grid,
    genetic, PSO, Bayesian/skopt Categorical) handles identically, and they
    preserve int-ness for integer parameters.

    Args:
        strategy_name: Registered strategy name.
        selected: Optionally restrict to a subset of parameter names.
    """
    space: Dict[str, List[Any]] = {}
    for spec in get_param_specs(strategy_name):
        if selected is not None and spec.name not in selected:
            continue
        space[spec.name] = spec.grid_values()
    return space


def get_self_tuning_config(**overrides: Any) -> Dict[str, Any]:
    """Build a ready-to-use config for ``SelfTuningOptimizer``.

    Before this existed, SelfTuningOptimizer was constructed with no config
    anywhere in the live app, so ``parameter_bounds`` was always empty and
    ``optimize_strategy()`` unconditionally returned None. This wires the
    canonical spaces through so the self-tuning path actually runs.
    """
    bounds: Dict[str, Dict[str, tuple]] = {}
    steps: Dict[str, Dict[str, float]] = {}
    for strategy, specs in STRATEGY_PARAM_SPACES.items():
        bounds[strategy] = {s.name: (s.low, s.high) for s in specs}
        steps[strategy] = {s.name: s.step for s in specs}
    cfg: Dict[str, Any] = {
        "parameter_bounds": bounds,
        "parameter_steps": steps,
    }
    cfg.update(overrides)
    return cfg
