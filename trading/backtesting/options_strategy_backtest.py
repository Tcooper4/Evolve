# -*- coding: utf-8 -*-
"""Defined-risk options STRUCTURE backtest (Black-Scholes + VIX proxy).

CRITICAL SCOPING — not negotiable
---------------------------------
Free option chains do **not** supply historical strike-level bid/ask
(see ``gex_snapshot_logger`` / ``options_cost_model``). This module does
**not** attempt a real historical options-fill replay.

Method (academically standard for structural research; cf. arXiv
2501.12397-style iron-condor control work): price each leg with
Black-Scholes using (1) real historical underlying closes and (2) real
historical VIX as an IV *proxy* for that underlying, plus a constant
risk-free rate. Transaction costs use
``options_cost_model.modeled_half_spread_fraction`` on every leg entry
and exit — exactly, no parallel cost logic.

This answers: "does this STRUCTURE have merit under realistic modeled
costs?" — not "what was my exact fill on date D."

Every public result includes ``DISCLOSURE`` below. Parameter sweeps go
through train/test + purge + Deflated Sharpe; nothing clears a live
default-on flag without that bar (null is acceptable).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DISCLOSURE = (
    "Modeled via Black-Scholes on real underlying + VIX history, not real "
    "historical option quotes — informative about strategy structure and "
    "cost realism, not a precise historical fill replay."
)

RISK_FREE_DEFAULT = 0.04
StrategyName = Literal["iron_condor", "put_credit_spread", "call_credit_spread"]

# Research-language defaults (30–45 DTE, 0.15–0.30Δ shorts, 50% profit,
# exit inside 7–10 DTE). Not Evolve-OOS-validated winners.
DEFAULT_DTE = 37
DEFAULT_SHORT_DELTA = 0.20
DEFAULT_WING_PCT = 0.05
DEFAULT_PROFIT_TAKE = 0.50
DEFAULT_MAX_LOSS_MULT = 2.0
DEFAULT_EXIT_DTE_FLOOR = 8
DEFAULT_REENTRY_GAP_DAYS = 5
DEFAULT_RECOMMEND_LIVE = False


@dataclass
class OptionsStructureParams:
    strategy: StrategyName = "iron_condor"
    dte: int = DEFAULT_DTE
    short_delta: float = DEFAULT_SHORT_DELTA
    wing_pct: float = DEFAULT_WING_PCT
    profit_take: float = DEFAULT_PROFIT_TAKE
    max_loss_mult: float = DEFAULT_MAX_LOSS_MULT
    exit_dte_floor: int = DEFAULT_EXIT_DTE_FLOOR
    reentry_gap_days: int = DEFAULT_REENTRY_GAP_DAYS
    risk_free: float = RISK_FREE_DEFAULT
    multiplier: float = 100.0


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    iv: float,
    *,
    is_call: bool,
    rate: float = RISK_FREE_DEFAULT,
    dividend: float = 0.0,
) -> float:
    """European BS price. Hand-checkable vs textbook ATM examples."""
    S, K, T, sig = float(spot), float(strike), float(time_years), float(iv)
    if S <= 0 or K <= 0 or sig <= 0:
        return 0.0
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    from scipy.stats import norm

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (rate - dividend + 0.5 * sig * sig) * T) / (
        sig * sqrt_t
    )
    d2 = d1 - sig * sqrt_t
    if is_call:
        return float(
            S * math.exp(-dividend * T) * norm.cdf(d1)
            - K * math.exp(-rate * T) * norm.cdf(d2)
        )
    return float(
        K * math.exp(-rate * T) * norm.cdf(-d2)
        - S * math.exp(-dividend * T) * norm.cdf(-d1)
    )


def black_scholes_delta(
    spot: float,
    strike: float,
    time_years: float,
    iv: float,
    *,
    is_call: bool,
    rate: float = RISK_FREE_DEFAULT,
    dividend: float = 0.0,
) -> float:
    S, K, T, sig = float(spot), float(strike), float(time_years), float(iv)
    if S <= 0 or K <= 0 or sig <= 0 or T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    from scipy.stats import norm

    d1 = (
        math.log(S / K) + (rate - dividend + 0.5 * sig * sig) * T
    ) / (sig * math.sqrt(T))
    if is_call:
        return float(math.exp(-dividend * T) * norm.cdf(d1))
    return float(-math.exp(-dividend * T) * norm.cdf(-d1))


def strike_for_target_delta(
    spot: float,
    time_years: float,
    iv: float,
    target_abs_delta: float,
    *,
    is_call: bool,
    rate: float = RISK_FREE_DEFAULT,
) -> float:
    """Grid search strike with |Δ| ≈ target (OTM for credits)."""
    tgt = abs(float(target_abs_delta))
    best_k, best_err = float(spot), 1e9
    for pct in np.linspace(0.80, 1.20, 81):
        k = float(spot) * float(pct)
        if is_call and k < spot:
            continue
        if (not is_call) and k > spot:
            continue
        d = black_scholes_delta(
            spot, k, time_years, iv, is_call=is_call, rate=rate
        )
        err = abs(abs(d) - tgt)
        if err < best_err:
            best_err, best_k = err, k
    return round(best_k, 2)


def _leg_mid(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    *,
    is_call: bool,
    rate: float,
) -> float:
    return black_scholes_price(
        spot, strike, t_years, iv, is_call=is_call, rate=rate
    )


def _half_spread_frac(spot: float, strike: float, dte: float) -> float:
    from trading.backtesting.options_cost_model import modeled_half_spread_fraction

    out = modeled_half_spread_fraction(spot=spot, strike=strike, dte=float(dte))
    return float(out["half_spread_fraction"])


def _trade_fill(mid: float, half_frac: float, *, buying: bool) -> float:
    m = max(float(mid), 0.0)
    h = max(float(half_frac), 0.0)
    if buying:
        return m * (1.0 + h)
    return max(0.0, m * (1.0 - h))


@dataclass
class Leg:
    name: str
    strike: float
    is_call: bool
    qty: float  # +1 long, -1 short


def build_structure_legs(
    params: OptionsStructureParams,
    spot: float,
    iv: float,
) -> List[Leg]:
    T = max(int(params.dte), 1) / 365.0
    sd = abs(float(params.short_delta))
    wing = max(float(params.wing_pct), 0.01) * float(spot)
    legs: List[Leg] = []

    if params.strategy in ("iron_condor", "put_credit_spread"):
        sp = strike_for_target_delta(
            spot, T, iv, sd, is_call=False, rate=params.risk_free
        )
        lp = round(sp - wing, 2)
        if lp <= 0:
            lp = round(sp * 0.95, 2)
        legs.append(Leg("short_put", sp, False, -1.0))
        legs.append(Leg("long_put", lp, False, +1.0))

    if params.strategy in ("iron_condor", "call_credit_spread"):
        sc = strike_for_target_delta(
            spot, T, iv, sd, is_call=True, rate=params.risk_free
        )
        lc = round(sc + wing, 2)
        legs.append(Leg("short_call", sc, True, -1.0))
        legs.append(Leg("long_call", lc, True, +1.0))

    return legs


def mark_structure(
    legs: Sequence[Leg],
    spot: float,
    iv: float,
    dte: float,
    *,
    rate: float,
    apply_costs: bool,
    closing: bool,
) -> Dict[str, Any]:
    dte_f = max(float(dte), 0.0)
    t_years = dte_f / 365.0
    mid_credit = 0.0
    fill_credit = 0.0
    leg_rows = []
    for leg in legs:
        mid = _leg_mid(
            spot, leg.strike, t_years, iv, is_call=leg.is_call, rate=rate
        )
        hs = _half_spread_frac(spot, leg.strike, dte_f) if apply_costs else 0.0
        mid_credit += (-leg.qty) * mid
        if closing:
            buying = leg.qty < 0
            fill = _trade_fill(mid, hs, buying=buying)
            fill_credit += -fill if buying else +fill
        else:
            buying = leg.qty > 0
            fill = _trade_fill(mid, hs, buying=buying)
            fill_credit += -fill if buying else +fill
        leg_rows.append({
            "name": leg.name,
            "strike": leg.strike,
            "is_call": leg.is_call,
            "qty": leg.qty,
            "mid": round(mid, 4),
            "half_spread_frac": round(hs, 4),
        })
    return {
        "mid_credit": float(mid_credit),
        "fill_credit": float(fill_credit),
        "legs": leg_rows,
    }


def _aligned_spot_vix(
    symbol: str,
    period: str = "2y",
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    try:
        from trading.data.price_cache import get_history
        from trading.portfolio.options_vix_sizing import fetch_vix_history

        hist = get_history(symbol, period=period)
        if hist is None or hist.empty:
            return None, None, "no underlying history"
        cm = {str(c).lower(): c for c in hist.columns}
        close = pd.to_numeric(hist[cm.get("close", hist.columns[0])], errors="coerce")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.dropna()
        vix = fetch_vix_history(period=period)
        if vix is None or vix.empty:
            return None, None, "no VIX history"
        vix = vix.copy()
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        df = pd.DataFrame({"spot": close, "vix": vix}).dropna()
        if len(df) < 80:
            return None, None, "insufficient aligned spot/VIX history"
        return df["spot"], df["vix"] / 100.0, None
    except Exception as e:
        return None, None, str(e)


def _trade_stats(pnls: np.ndarray) -> Dict[str, Any]:
    if pnls.size == 0:
        return {
            "win_rate": None,
            "avg_pnl": None,
            "total_pnl": 0.0,
            "sharpe": None,
            "n": 0,
        }
    wins = float(np.mean(pnls > 0))
    mu = float(np.mean(pnls))
    sd = float(np.std(pnls, ddof=1)) if pnls.size > 1 else 0.0
    sharpe = (mu / sd) if sd > 1e-12 else None
    return {
        "win_rate": round(wins, 4),
        "avg_pnl": round(mu, 2),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "n": int(pnls.size),
    }


def simulate_structure_trades(
    spot: pd.Series,
    iv: pd.Series,
    params: OptionsStructureParams,
    *,
    apply_costs: bool = True,
) -> Dict[str, Any]:
    dates = list(spot.index)
    trades: List[Dict[str, Any]] = []
    i = 0
    max_i = len(dates) - int(params.dte) - 2
    while i < max_i:
        d0 = dates[i]
        s0 = float(spot.loc[d0])
        iv0 = float(iv.loc[d0])
        if s0 <= 0 or iv0 <= 0:
            i += 1
            continue
        legs = build_structure_legs(params, s0, iv0)
        if not legs:
            i += 1
            continue
        entry = mark_structure(
            legs, s0, iv0, float(params.dte),
            rate=params.risk_free, apply_costs=apply_costs, closing=False,
        )
        entry_credit = entry["fill_credit"]
        if entry_credit <= 0:
            i += max(1, params.reentry_gap_days)
            continue

        exit_i = None
        exit_reason = "dte_floor"
        pnl_credit_units = None
        j_end = min(i + int(params.dte), len(dates) - 1)
        for j in range(i + 1, j_end + 1):
            dte_left = int(params.dte) - (j - i)
            sj = float(spot.iloc[j])
            ivj = float(iv.iloc[j])
            mark = mark_structure(
                legs, sj, ivj, float(max(dte_left, 0)),
                rate=params.risk_free, apply_costs=apply_costs, closing=True,
            )
            pnl = entry_credit + mark["fill_credit"]
            if pnl >= float(params.profit_take) * entry_credit:
                exit_i, exit_reason, pnl_credit_units = j, "profit_take", pnl
                break
            if pnl <= -float(params.max_loss_mult) * entry_credit:
                exit_i, exit_reason, pnl_credit_units = j, "max_loss", pnl
                break
            if dte_left <= int(params.exit_dte_floor):
                exit_i, exit_reason, pnl_credit_units = j, "dte_floor", pnl
                break

        if exit_i is None:
            j = j_end
            dte_left = max(int(params.dte) - (j - i), 0)
            mark = mark_structure(
                legs, float(spot.iloc[j]), float(iv.iloc[j]), float(dte_left),
                rate=params.risk_free, apply_costs=apply_costs, closing=True,
            )
            pnl_credit_units = entry_credit + mark["fill_credit"]
            exit_i, exit_reason = j, "expiry_window"

        pnl_dollars = float(pnl_credit_units) * float(params.multiplier)
        trades.append({
            "entry_date": str(pd.Timestamp(d0).date()),
            "exit_date": str(pd.Timestamp(dates[exit_i]).date()),
            "entry_credit": round(entry_credit, 4),
            "pnl_per_share": round(float(pnl_credit_units), 4),
            "pnl_dollars": round(pnl_dollars, 2),
            "exit_reason": exit_reason,
            "strikes": {lg.name: lg.strike for lg in legs},
            "entry_spot": round(s0, 4),
            "exit_spot": round(float(spot.iloc[exit_i]), 4),
        })
        i = exit_i + max(1, int(params.reentry_gap_days))

    pnls = np.array([t["pnl_dollars"] for t in trades], dtype=float)
    return {
        "success": True,
        "n_trades": len(trades),
        "trades": trades,
        "stats": _trade_stats(pnls),
        "params": asdict(params),
        "disclosure": DISCLOSURE,
        "recommend_live": DEFAULT_RECOMMEND_LIVE,
    }


def _purged_split_indices(n: int, purge: int, train_frac: float = 0.60) -> Tuple[int, int]:
    """Train/test indices with the same purge gap as WalkForwardValidator.run.

    train ends at ``start_idx``; test starts at ``start_idx + purge``
    (bars in between are embargoed). See
    ``trading.validation.walk_forward_utils.WalkForwardValidator``.
    """
    try:
        purge_i = max(0, int(purge))
    except Exception:
        purge_i = 0
    start_idx = max(1, min(n - 1, int(n * float(train_frac))))
    train_end = start_idx
    test_start = start_idx + purge_i
    return train_end, test_start


def _sweep_oos(
    spot: pd.Series,
    iv: pd.Series,
    base: OptionsStructureParams,
    *,
    apply_costs: bool,
    symbol: str,
    period: str,
) -> Dict[str, Any]:
    n = len(spot)
    # Embargo length = entry DTE (structure horizon); same purge= meaning
    # as WalkForwardValidator (skip bars between train end and test start).
    purge = max(int(base.dte), 5)
    train_end, test_start = _purged_split_indices(n, purge, train_frac=0.60)
    if train_end < 60 or test_start >= n - 20:
        return {
            "success": False,
            "symbol": symbol,
            "error": "not enough history for purged train/test split",
            "disclosure": DISCLOSURE,
            "recommend_live": False,
        }

    spot_tr, iv_tr = spot.iloc[:train_end], iv.iloc[:train_end]
    spot_te, iv_te = spot.iloc[test_start:], iv.iloc[test_start:]

    grid_delta = [0.15, 0.20, 0.25, 0.30]
    grid_wing = [0.04, 0.05, 0.06]
    trials: List[Dict[str, Any]] = []
    scores: List[float] = []

    for sd in grid_delta:
        for wp in grid_wing:
            p = OptionsStructureParams(
                **{**asdict(base), "short_delta": sd, "wing_pct": wp}
            )
            sim = simulate_structure_trades(
                spot_tr, iv_tr, p, apply_costs=apply_costs
            )
            sh = (sim.get("stats") or {}).get("sharpe")
            score = float(sh) if sh is not None else -999.0
            trials.append({
                "short_delta": sd,
                "wing_pct": wp,
                "train_stats": sim.get("stats"),
                "n_trades": sim.get("n_trades"),
                "score": score,
            })
            scores.append(score)

    best = max(trials, key=lambda t: t["score"]) if trials else None
    if best is None or best["score"] <= -998:
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "method": "sweep_oos",
            "disclosure": DISCLOSURE,
            "recommend_live": False,
            "note": "No viable train-window trades — null result.",
            "trials": trials,
            "purge_days": purge,
        }

    champ = OptionsStructureParams(
        **{
            **asdict(base),
            "short_delta": best["short_delta"],
            "wing_pct": best["wing_pct"],
        }
    )
    test_sim = simulate_structure_trades(
        spot_te, iv_te, champ, apply_costs=apply_costs
    )
    test_stats = test_sim.get("stats") or {}
    n_obs = int(test_stats.get("n") or 0)
    obs_sr = test_stats.get("sharpe")
    dsr = None
    if obs_sr is not None and n_obs > 1:
        try:
            from trading.optimization.deflated_sharpe import deflated_sharpe_ratio

            dsr = deflated_sharpe_ratio(
                float(obs_sr),
                [s for s in scores if s > -998],
                n_obs=n_obs,
            )
        except Exception as e:
            logger.debug("DSR failed: %s", e)

    recommend = bool(
        dsr
        and float(dsr.get("deflated_sharpe") or 0) >= 0.95
        and (obs_sr is not None and float(obs_sr) > 0)
        and n_obs >= 10
    )

    return {
        "success": True,
        "symbol": symbol,
        "period": period,
        "method": "sweep_oos",
        "disclosure": DISCLOSURE,
        "iv_proxy": "VIX_close/100",
        "purge_days": purge,
        "train_end_idx": train_end,
        "test_start_idx": test_start,
        "n_trials": len(trials),
        "champion": {
            "short_delta": best["short_delta"],
            "wing_pct": best["wing_pct"],
            "train_stats": best["train_stats"],
        },
        "test": test_sim,
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
        "trials": trials,
    }


def run_options_structure_backtest(
    symbol: str,
    *,
    strategy: StrategyName = "iron_condor",
    period: str = "2y",
    dte: int = DEFAULT_DTE,
    short_delta: float = DEFAULT_SHORT_DELTA,
    wing_pct: float = DEFAULT_WING_PCT,
    profit_take: float = DEFAULT_PROFIT_TAKE,
    max_loss_mult: float = DEFAULT_MAX_LOSS_MULT,
    exit_dte_floor: int = DEFAULT_EXIT_DTE_FLOOR,
    sweep: bool = False,
    apply_costs: bool = True,
) -> Dict[str, Any]:
    """Public entry. ``sweep=True`` → purged train/test + DSR."""
    sym = (symbol or "").strip().upper()
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "disclosure": DISCLOSURE,
        "recommend_live": DEFAULT_RECOMMEND_LIVE,
        "error": None,
    }
    if not sym:
        out["error"] = "symbol required"
        return out

    spot, iv, err = _aligned_spot_vix(sym, period=period)
    if err or spot is None or iv is None:
        out["error"] = err or "history unavailable"
        return out

    base = OptionsStructureParams(
        strategy=strategy,  # type: ignore[arg-type]
        dte=int(dte),
        short_delta=float(short_delta),
        wing_pct=float(wing_pct),
        profit_take=float(profit_take),
        max_loss_mult=float(max_loss_mult),
        exit_dte_floor=int(exit_dte_floor),
    )

    if not sweep:
        sim = simulate_structure_trades(spot, iv, base, apply_costs=apply_costs)
        sim["symbol"] = sym
        sim["period"] = period
        sim["method"] = "fixed_params"
        sim["iv_proxy"] = "VIX_close/100"
        return sim

    return _sweep_oos(
        spot, iv, base, apply_costs=apply_costs, symbol=sym, period=period
    )


__all__ = [
    "DISCLOSURE",
    "DEFAULT_RECOMMEND_LIVE",
    "OptionsStructureParams",
    "black_scholes_price",
    "black_scholes_delta",
    "strike_for_target_delta",
    "build_structure_legs",
    "mark_structure",
    "simulate_structure_trades",
    "run_options_structure_backtest",
]
