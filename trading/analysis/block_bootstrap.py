# -*- coding: utf-8 -*-
"""Stationary block bootstrap (Politis-Romano) for return paths.

Why offer block bootstrap
-------------------------
i.i.d. daily resampling destroys volatility clustering. Stationary
bootstrap (Politis & Romano, 1994) keeps local dependence via geometric-
length blocks (end probability p=1/L; otherwise take the next circular
observation).

Evolve comparison (honest — default gating)
-------------------------------------------
Multi-seed replications on GARCH(1,1) (omega=1e-6, alpha=0.08, beta=0.90)
and real 2y ``get_history`` series (SPY/QQQ/IWM/AAPL) found that
``iid_p5 - block_p5`` medians at 63d were typically **negative**: i.i.d.
often produced *lower* (more stressed) p5 capital than stationary block,
with the gap growing for L=15/30. Likely cause: i.i.d. can stack crash
days that never occurred consecutively. An earlier single-seed anecdote
suggesting *more* block downside did **not** hold in this broader check.

Therefore ``DEFAULT_METHOD = "iid"`` remains — we do not quietly soften
reported tails. Pass ``method="stationary_block"`` when you want
dependence-faithful paths; the API ``note`` discloses method + this
tradeoff.

Mean block length L
-------------------
From squared-return lag-1 ACF (vol-clustering channel):

    rho = corr(r_t^2, r_{t-1}^2)  clipped to [0, 0.99]
    L   = 1 / (1 - rho)          clamped to [MIN_MEAN_BLOCK, MAX_MEAN_BLOCK]

Derived from series dependence (Kelly sample-size style), not a guess.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

MIN_MEAN_BLOCK = 5
MAX_MEAN_BLOCK = 63  # ~one quarter of trading days
# Kept as iid after real+synthetic comparison (see module docstring).
DEFAULT_METHOD: Literal["iid", "stationary_block"] = "iid"

BootstrapMethod = Literal["iid", "stationary_block"]


def _as_1d(returns: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def estimate_mean_block_length(
    returns: Union[np.ndarray, Sequence[float]],
    *,
    min_block: int = MIN_MEAN_BLOCK,
    max_block: int = MAX_MEAN_BLOCK,
) -> Dict[str, Any]:
    """Mean geometric block length from squared-return lag-1 persistence.

    Returns dict with ``L``, ``rho_sq``, ``reason``.
    """
    arr = _as_1d(returns)
    out: Dict[str, Any] = {
        "L": int(min_block),
        "rho_sq": 0.0,
        "reason": "insufficient history — using floor mean block length",
    }
    if arr.size < 40:
        return out
    sq = arr * arr
    if float(np.std(sq)) < 1e-18:
        out["reason"] = "near-zero squared-return variance — using floor L"
        return out
    x0, x1 = sq[:-1], sq[1:]
    # Pearson corr of consecutive squared returns
    try:
        c = float(np.corrcoef(x0, x1)[0, 1])
    except Exception as e:
        logger.debug("block length corr failed: %s", e)
        return out
    if not np.isfinite(c):
        return out
    rho = float(np.clip(c, 0.0, 0.99))
    raw = 1.0 / max(1.0 - rho, 1e-6)
    L = int(np.clip(round(raw), min_block, max_block))
    out.update({
        "L": L,
        "rho_sq": rho,
        "raw_L": float(raw),
        "reason": (
            f"L={L} from squared-return lag-1 ACF rho={rho:.3f} "
            f"(memory 1/(1-rho)={raw:.1f}d, clamped to "
            f"[{min_block},{max_block}])"
        ),
    })
    return out


def stationary_block_path(
    returns: Union[np.ndarray, Sequence[float]],
    horizon: int,
    *,
    mean_block_length: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """One path of length ``horizon`` via Politis–Romano stationary bootstrap."""
    arr = _as_1d(returns)
    n = int(arr.size)
    h = int(max(1, horizon))
    if n == 0:
        return np.zeros(h, dtype=float)
    L = max(float(mean_block_length), 1.0)
    p_end = min(1.0, 1.0 / L)
    gen = rng or np.random.default_rng()
    out = np.empty(h, dtype=float)
    # Start index uniform; wrap cyclically
    idx = int(gen.integers(0, n))
    for t in range(h):
        out[t] = arr[idx]
        if gen.random() < p_end:
            idx = int(gen.integers(0, n))
        else:
            idx = (idx + 1) % n
    return out


def iid_path(
    returns: Union[np.ndarray, Sequence[float]],
    horizon: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    arr = _as_1d(returns)
    h = int(max(1, horizon))
    if arr.size == 0:
        return np.zeros(h, dtype=float)
    gen = rng or np.random.default_rng()
    return gen.choice(arr, size=h, replace=True)


def simulate_equity_paths(
    returns: Union[np.ndarray, Sequence[float]],
    *,
    n_simulations: int,
    horizon: int,
    initial_capital: float = 10_000.0,
    method: BootstrapMethod = "iid",
    mean_block_length: Optional[float] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Bootstrap equity paths; return p5/p50/p95 finals + metadata."""
    arr = _as_1d(returns)
    if arr.size < 10:
        return {"success": False, "error": "Need more return history"}

    n_sims = max(50, min(int(n_simulations), 1500))
    h = max(5, min(int(horizon), 252))
    capital = float(initial_capital)
    gen = np.random.default_rng(seed)

    block_meta: Dict[str, Any] = {}
    L = mean_block_length
    if method == "stationary_block":
        if L is None:
            block_meta = estimate_mean_block_length(arr)
            L = float(block_meta["L"])
        else:
            L = float(L)
            block_meta = {
                "L": L,
                "rho_sq": None,
                "reason": f"explicit mean block length L={L}",
            }

    paths = np.empty((n_sims, h), dtype=float)
    for i in range(n_sims):
        if method == "stationary_block":
            draws = stationary_block_path(arr, h, mean_block_length=float(L), rng=gen)
        else:
            draws = iid_path(arr, h, rng=gen)
        paths[i] = capital * np.cumprod(1.0 + draws)

    finals = paths[:, -1]
    p5, p50, p95 = np.percentile(finals, [5, 50, 95])
    mean_path = paths.mean(axis=0)

    if method == "stationary_block":
        note = (
            f"Stationary block bootstrap (Politis-Romano geometric blocks, "
            f"mean L={float(L):.0f}). Preserves vol clustering that i.i.d. "
            f"destroys. {block_meta.get('reason', '')} "
            f"On Evolve's GARCH + real-symbol comparison, block p5 was often "
            f"*higher* (less stressed) than i.i.d. at 63d — i.i.d. can stack "
            f"independent crash days. Opt-in method; default remains iid. "
            f"Not a price forecast."
        ).strip()
    else:
        note = (
            "i.i.d. bootstrap of historical daily returns (default). "
            "Destroys volatility clustering; can fabricate consecutive crash "
            "stacks that never occurred — often *more* stressed p5 than "
            "stationary block in Evolve's comparison. "
            "Pass method='stationary_block' for dependence-faithful paths. "
            "Not a price forecast."
        )

    return {
        "success": True,
        "method": method,
        "n_simulations": n_sims,
        "horizon_days": h,
        "initial_capital": capital,
        "final_p5": float(p5),
        "final_p50": float(p50),
        "final_p95": float(p95),
        "mean_path": mean_path,
        "paths": paths,
        "block": block_meta,
        "note": note,
    }


def compare_tail_risk(
    returns: Union[np.ndarray, Sequence[float]],
    *,
    horizons: Sequence[int] = (5, 10, 21, 63),
    mean_blocks: Sequence[float] = (15.0, 30.0),
    n_simulations: int = 800,
    initial_capital: float = 10_000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """i.i.d. vs stationary-block p5 shortfall (capital units + ppt)."""
    arr = _as_1d(returns)
    rows: List[Dict[str, Any]] = []
    for h in horizons:
        iid = simulate_equity_paths(
            arr,
            n_simulations=n_simulations,
            horizon=int(h),
            initial_capital=initial_capital,
            method="iid",
            seed=seed,
        )
        for L in mean_blocks:
            blk = simulate_equity_paths(
                arr,
                n_simulations=n_simulations,
                horizon=int(h),
                initial_capital=initial_capital,
                method="stationary_block",
                mean_block_length=float(L),
                seed=seed + 1,
            )
            # Positive delta_p5_ppt ⇒ block shows *more* downside (lower p5)
            delta = float(iid["final_p5"]) - float(blk["final_p5"])
            rows.append({
                "horizon": int(h),
                "mean_block": float(L),
                "iid_p5": float(iid["final_p5"]),
                "block_p5": float(blk["final_p5"]),
                "delta_p5_capital": delta,
                "delta_p5_ppt": 100.0 * delta / float(initial_capital),
            })
    # Auto L row at 63d
    auto = estimate_mean_block_length(arr)
    iid63 = simulate_equity_paths(
        arr, n_simulations=n_simulations, horizon=63,
        initial_capital=initial_capital, method="iid", seed=seed,
    )
    blk63 = simulate_equity_paths(
        arr, n_simulations=n_simulations, horizon=63,
        initial_capital=initial_capital, method="stationary_block",
        mean_block_length=float(auto["L"]), seed=seed + 1,
    )
    delta_auto = float(iid63["final_p5"]) - float(blk63["final_p5"])
    return {
        "rows": rows,
        "auto_block": auto,
        "auto_63d": {
            "iid_p5": float(iid63["final_p5"]),
            "block_p5": float(blk63["final_p5"]),
            "delta_p5_ppt": 100.0 * delta_auto / float(initial_capital),
            "L": auto["L"],
        },
    }


def simulate_garch11(
    n: int = 1500,
    *,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    seed: int = 20260714,
) -> np.ndarray:
    """GARCH(1,1) returns for stress comparisons (variance clustering)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    r = np.zeros(n)
    v = omega / max(1.0 - alpha - beta, 1e-6)
    for t in range(n):
        v = omega + alpha * (r[t - 1] ** 2 if t else 0.0) + beta * v
        r[t] = float(np.sqrt(max(v, 1e-18)) * eps[t])
    return r


__all__ = [
    "MIN_MEAN_BLOCK",
    "MAX_MEAN_BLOCK",
    "DEFAULT_METHOD",
    "estimate_mean_block_length",
    "stationary_block_path",
    "iid_path",
    "simulate_equity_paths",
    "compare_tail_risk",
    "simulate_garch11",
]
