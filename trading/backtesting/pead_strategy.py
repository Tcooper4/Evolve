# -*- coding: utf-8 -*-
"""Post-earnings announcement drift (PEAD) — equity long book research.

Reuses ``trading.data.earnings_reaction`` for surprise detection and the
BMO/AMC timing inference (do not reinvent). Holding periods are multi-week
closes after ``reaction_date``, not the module's native 1d/3d/5d stats.

---------------------------------------------------------------------------
PREDECLARED TRIAL SET (locked BEFORE any real-data OOS / DSR run)
---------------------------------------------------------------------------
Literature on PEAD (Bernard & Thomas–style post-earnings drift; subsequent
reviews) finds drift lasting months but **concentrated in the early weeks
after the announcement**. Retail execution also cannot always enter at the
instant of the print. We therefore fix exactly **four** theory-motivated
variants — not a wide grid chosen after peeking at DSR:

  hold_days ∈ {20, 40}   # ~4 and ~8 trading weeks
  entry_lag ∈ {0, 1}     # 0 = close of reaction_date; 1 = next session

That is N_trials = 4. Choosing this set after seeing which subset clears
DSR would reintroduce the selection bias DSR exists to catch — do not.

Short side
----------
Academic PEAD often shorts negative surprises. At retail scale that side
is contaminated by borrow cost, hard-to-borrow names, and asymmetric
coverage. This harness is **long-only on positive EPS surprise**
(``surprise_pct > 0``). Negative surprises are skipped (avoid), not
shorted. That limitation is reported in every result payload.

Costs
-----
Retail equity one-way from ``get_retail_cost_config``:
fee 10 bps + spread 5 bps + slip 2 bps = 17 bps; round-trip subtracted
from each trade's return. Not the options cost model.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DISCLOSURE = (
    "PEAD research harness: long-only on positive EPS surprise using "
    "earnings_reaction BMO/AMC timing; multi-week close-to-close holds with "
    "retail equity round-trip costs. Not a forecast; null OOS is expected "
    "and acceptable. Short-negative side intentionally omitted (retail "
    "borrow / hard-to-borrow asymmetry)."
)

DEFAULT_RECOMMEND_LIVE = False

# --- locked BEFORE real-data runs (see module docstring) ---
PREDECLARED_HOLD_DAYS: Tuple[int, ...] = (20, 40)
PREDECLARED_ENTRY_LAGS: Tuple[int, ...] = (0, 1)
PREDECLARED_TRIALS: Tuple[Dict[str, int], ...] = tuple(
    {"hold_days": h, "entry_lag": lag}
    for h in PREDECLARED_HOLD_DAYS
    for lag in PREDECLARED_ENTRY_LAGS
)
TRIAL_JUSTIFICATION = (
    "Four trials only: hold 20 vs 40 trading days (~4 / ~8 weeks) × entry "
    "lag 0 vs 1 session. Anchored on PEAD literature that drift concentrates "
    "early after the announcement while sometimes extending further, plus "
    "practical retail entry lag. Locked before any real-data DSR evaluation."
)

# Liquid large/mid names (ml_score_trainer + scanner-style adds) for event power.
LARGE_CAP_BASKET: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ",
    "XOM", "BRK-B", "UNH", "V", "MA", "HD", "PG", "ABBV", "MRK", "LLY", "PEP",
    "AVGO", "COST", "WMT", "BAC", "KO", "CVX", "CRM", "AMD", "ORCL", "ADBE",
]

# Smaller / typically less sell-side-covered midcaps (literature crowding
# check). Kept separate — not mixed into the primary large-cap OOS.
MID_LESS_COVERED_BASKET: List[str] = [
    "DECK", "ETSY", "POOL", "WSM", "RH", "TXRH", "WING", "CROX", "BLD", "TOL",
    "ZBRA", "FIVE", "MHK", "LEG", "SNA", "GPC", "PHM", "DHI", "NVR", "MAS",
]


def equity_round_trip_cost_fraction() -> float:
    """Retail equity RT fraction (fees + spread + slippage), both legs."""
    from trading.backtesting.cost_model import get_retail_cost_config

    cfg = get_retail_cost_config()
    one_way = float(cfg.fee_rate) + float(cfg.spread_rate) + float(cfg.slippage_rate)
    return 2.0 * one_way


@dataclass
class PeadParams:
    hold_days: int = 20
    entry_lag: int = 0  # trading sessions after reaction_date


def _normalize_hist(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    cm = {str(c).lower(): c for c in h.columns}
    if "close" in cm and "Close" not in h.columns:
        h = h.rename(columns={cm["close"]: "Close"})
    if "open" in cm and "Open" not in h.columns:
        h = h.rename(columns={cm["open"]: "Open"})
    h.index = pd.to_datetime(h.index).normalize()
    return h


def simulate_pead_trade_return(
    closes: pd.Series,
    reaction_date: pd.Timestamp,
    *,
    hold_days: int,
    entry_lag: int = 0,
    cost_rt: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Long one event: entry at close(reaction + entry_lag), exit hold_days later.

    Returns dict with pnl_return (net of costs) or None if history insufficient.
    Hand-checkable: flat costs only reduce return by cost_rt.
    """
    if cost_rt is None:
        cost_rt = equity_round_trip_cost_fraction()
    idx = list(closes.index)
    try:
        rx = pd.Timestamp(reaction_date).normalize()
    except Exception:
        return None
    # Find reaction bar (exact or first on/after)
    pos = None
    for i, t in enumerate(idx):
        if pd.Timestamp(t).normalize() >= rx:
            pos = i
            break
    if pos is None:
        return None
    entry_i = pos + int(entry_lag)
    exit_i = entry_i + int(hold_days)
    if entry_i < 0 or exit_i >= len(idx):
        return None
    entry_px = float(closes.iloc[entry_i])
    exit_px = float(closes.iloc[exit_i])
    if entry_px <= 0 or exit_px <= 0:
        return None
    raw = exit_px / entry_px - 1.0
    net = raw - float(cost_rt)
    return {
        "entry_date": str(pd.Timestamp(idx[entry_i]).date()),
        "exit_date": str(pd.Timestamp(idx[exit_i]).date()),
        "entry_px": round(entry_px, 4),
        "exit_px": round(exit_px, 4),
        "raw_return": round(raw, 6),
        "cost_rt": float(cost_rt),
        "pnl_return": round(net, 6),
        "hold_days": int(hold_days),
        "entry_lag": int(entry_lag),
    }


def collect_positive_surprise_events(
    symbols: Sequence[str],
    *,
    num_quarters: int = 20,
) -> List[Dict[str, Any]]:
    """Pool long-only PEAD candidates using earnings_reaction for each name."""
    from trading.data.earnings_reaction import get_earnings_reactions

    events: List[Dict[str, Any]] = []
    for sym in symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            block = get_earnings_reactions(s, num_quarters=num_quarters) or {}
        except Exception as e:
            logger.debug("PEAD earnings fetch %s: %s", s, e)
            continue
        for r in block.get("reactions") or []:
            try:
                surprise = float(r.get("surprise_pct") or 0.0)
            except Exception:
                continue
            if surprise <= 0:
                continue  # long-only; skip non-positive surprises
            rx = r.get("reaction_date") or r.get("date")
            if not rx:
                continue
            events.append({
                "symbol": s,
                "earnings_date": r.get("date"),
                "reaction_date": str(rx),
                "surprise_pct": surprise,
                "timing": r.get("timing"),
                "beat": bool(r.get("beat", surprise > 0)),
            })
    events.sort(key=lambda e: str(e.get("reaction_date") or ""))
    return events


def _load_closes(symbol: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="5y", interval="1d")
        if hist is None or hist.empty:
            return None
        hist = _normalize_hist(hist)
        return pd.to_numeric(hist["Close"], errors="coerce").dropna()
    except Exception as e:
        logger.debug("PEAD price %s: %s", symbol, e)
        return None


def simulate_event_trades(
    events: Sequence[Dict[str, Any]],
    params: PeadParams,
    *,
    closes_by_symbol: Optional[Dict[str, pd.Series]] = None,
    cost_rt: Optional[float] = None,
) -> Dict[str, Any]:
    """Simulate all events under one PeadParams; returns trade list + stats."""
    if cost_rt is None:
        cost_rt = equity_round_trip_cost_fraction()
    cache = dict(closes_by_symbol or {})
    trades: List[Dict[str, Any]] = []
    for ev in events:
        sym = str(ev.get("symbol") or "")
        if sym not in cache:
            cache[sym] = _load_closes(sym)  # type: ignore[assignment]
        closes = cache.get(sym)
        if closes is None or len(closes) < 10:
            continue
        sim = simulate_pead_trade_return(
            closes,
            pd.Timestamp(ev["reaction_date"]),
            hold_days=int(params.hold_days),
            entry_lag=int(params.entry_lag),
            cost_rt=cost_rt,
        )
        if sim is None:
            continue
        row = dict(sim)
        row.update({
            "symbol": sym,
            "surprise_pct": ev.get("surprise_pct"),
            "timing": ev.get("timing"),
            "earnings_date": ev.get("earnings_date"),
        })
        trades.append(row)

    pnls = np.array([t["pnl_return"] for t in trades], dtype=float)
    stats = _trade_stats(pnls)
    return {
        "success": True,
        "n_trades": len(trades),
        "trades": trades,
        "stats": stats,
        "params": asdict(params),
        "cost_rt": float(cost_rt),
        "disclosure": DISCLOSURE,
        "recommend_live": DEFAULT_RECOMMEND_LIVE,
        "_closes_cache": cache,  # for reuse inside sweep; stripped before JSON
    }


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
    # Per-trade Sharpe (not annualized) — same convention as options structure BT
    sharpe = (mu / sd) if sd > 1e-12 else None
    return {
        "win_rate": round(wins, 4),
        "avg_pnl": round(mu, 6),
        "total_pnl": round(float(np.sum(pnls)), 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "n": int(pnls.size),
    }


def _purged_event_split(
    events: Sequence[Dict[str, Any]],
    purge_days: int,
    train_frac: float = 0.60,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Train/test on sorted events with a calendar embargo after train end.

    Unlike bar-index purge (daily options BT), PEAD events are sparse —
    ``purge_days`` is trading days after the last train ``reaction_date``,
    not a count of events to skip.
    """
    n = len(events)
    train_end = max(1, min(n - 1, int(n * float(train_frac))))
    train = list(events[:train_end])
    if not train:
        return [], [], 0
    last_rx = pd.Timestamp(train[-1]["reaction_date"]).normalize()
    try:
        cutoff = last_rx + pd.tseries.offsets.BDay(max(0, int(purge_days)))
    except Exception:
        cutoff = last_rx + pd.Timedelta(days=max(0, int(purge_days)))
    test = [
        e
        for e in events[train_end:]
        if pd.Timestamp(e["reaction_date"]).normalize() >= cutoff
    ]
    return train, test, train_end


def run_pead_oos(
    symbols: Sequence[str],
    *,
    num_quarters: int = 20,
    events: Optional[List[Dict[str, Any]]] = None,
    closes_by_symbol: Optional[Dict[str, pd.Series]] = None,
    label: str = "large_cap",
) -> Dict[str, Any]:
    """Pooled purged OOS over the predeclared 4-trial PEAD grid + DSR gate."""
    out: Dict[str, Any] = {
        "success": False,
        "label": label,
        "universe": [str(s).upper() for s in symbols],
        "disclosure": DISCLOSURE,
        "recommend_live": False,
        "predeclared_trials": list(PREDECLARED_TRIALS),
        "trial_justification": TRIAL_JUSTIFICATION,
        "short_side_policy": "long_only_positive_surprise",
        "short_side_note": (
            "Negative surprises skipped (avoid), not shorted — retail borrow "
            "costs and hard-to-borrow asymmetry make the academic short side "
            "unclean at this scale."
        ),
        "cost_rt": equity_round_trip_cost_fraction(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    evs = events if events is not None else collect_positive_surprise_events(
        symbols, num_quarters=num_quarters
    )
    out["n_events_positive_surprise"] = len(evs)
    if len(evs) < 30:
        out["error"] = (
            f"insufficient positive-surprise events for OOS ({len(evs)} < 30)"
        )
        out["success"] = True  # ran; null data
        out["note"] = "Null — not enough pooled observations."
        return out

    # Purge sized to longest hold in the predeclared set (calendar days)
    purge = max(PREDECLARED_HOLD_DAYS)
    train_ev, test_ev, train_end = _purged_event_split(
        evs, purge, train_frac=0.60
    )
    if len(train_ev) < 15 or len(test_ev) < 8:
        out["error"] = (
            f"not enough events after calendar purge "
            f"(train={len(train_ev)}, test={len(test_ev)}, purge_bdays={purge})"
        )
        out["success"] = True
        out["train_end_idx"] = train_end
        out["purge_days"] = purge
        out["n_train_events"] = len(train_ev)
        out["n_test_events"] = len(test_ev)
        return out

    out["purge_days"] = purge
    out["train_end_idx"] = train_end
    out["n_train_events"] = len(train_ev)
    out["n_test_events"] = len(test_ev)

    cache: Dict[str, pd.Series] = dict(closes_by_symbol or {})
    trials: List[Dict[str, Any]] = []
    scores: List[float] = []

    for spec in PREDECLARED_TRIALS:
        params = PeadParams(
            hold_days=int(spec["hold_days"]),
            entry_lag=int(spec["entry_lag"]),
        )
        sim = simulate_event_trades(
            train_ev, params, closes_by_symbol=cache, cost_rt=out["cost_rt"]
        )
        cache = sim.pop("_closes_cache", cache)
        sh = (sim.get("stats") or {}).get("sharpe")
        score = float(sh) if sh is not None else -999.0
        trials.append({
            "hold_days": params.hold_days,
            "entry_lag": params.entry_lag,
            "train_stats": sim.get("stats"),
            "n_trades": sim.get("n_trades"),
            "score": score,
        })
        scores.append(score)

    best = max(trials, key=lambda t: t["score"]) if trials else None
    if best is None or best["score"] <= -998:
        out["success"] = True
        out["trials"] = trials
        out["note"] = "No viable train-window trades — null result."
        return out

    champ = PeadParams(
        hold_days=int(best["hold_days"]), entry_lag=int(best["entry_lag"])
    )
    test_sim = simulate_event_trades(
        test_ev, champ, closes_by_symbol=cache, cost_rt=out["cost_rt"]
    )
    test_sim.pop("_closes_cache", None)
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
            logger.debug("PEAD DSR failed: %s", e)

    recommend = bool(
        dsr
        and float(dsr.get("deflated_sharpe") or 0) >= 0.95
        and (obs_sr is not None and float(obs_sr) > 0)
        and n_obs >= 10
    )

    # Drop bulky trade lists from archive JSON
    test_compact = {
        k: v for k, v in test_sim.items() if k != "trades"
    }
    test_compact["n_trades"] = test_sim.get("n_trades")

    out.update({
        "success": True,
        "n_trials": len(trials),
        "champion": {
            "hold_days": champ.hold_days,
            "entry_lag": champ.entry_lag,
            "train_stats": best["train_stats"],
        },
        "test": test_compact,
        "deflated_sharpe": dsr,
        "recommend_live": recommend,
        "trials": trials,
        "note": (
            "Champion cleared DSR>=0.95 on OOS — still research; not auto-wired."
            if recommend
            else (
                "Null / not significant on OOS+DSR — leave research-only "
                "(acceptable expected outcome)."
            )
        ),
    })
    return out


def run_pead_oos_real(
    *,
    out_path: Optional[str] = None,
    run_midcap: bool = True,
) -> Dict[str, Any]:
    """Full Phase-1 real-data report (large-cap primary + optional midcap)."""
    report: Dict[str, Any] = {
        "success": True,
        "disclosure": DISCLOSURE,
        "trial_justification": TRIAL_JUSTIFICATION,
        "predeclared_trials": list(PREDECLARED_TRIALS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Trial grid and baskets were fixed in source before this run; "
            "not selected after inspecting DSR."
        ),
        "large_cap": None,
        "mid_less_covered": None,
        "recommend_live": False,
    }

    large = run_pead_oos(LARGE_CAP_BASKET, label="large_cap")
    report["large_cap"] = large

    mid = None
    if run_midcap:
        mid = run_pead_oos(MID_LESS_COVERED_BASKET, label="mid_less_covered")
        report["mid_less_covered"] = mid

    report["recommend_live"] = bool(
        (large or {}).get("recommend_live")
        or (mid or {}).get("recommend_live")
    )
    report["note"] = (
        "At least one basket cleared the live bar — still research-only."
        if report["recommend_live"]
        else (
            "Null / not significant on PEAD OOS+DSR for reported baskets — "
            "leave research-only (acceptable expected outcome)."
        )
    )
    report["success"] = bool(
        (large or {}).get("success") or (mid or {}).get("success")
    )

    if out_path:
        import json
        from pathlib import Path

        Path(out_path).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    return report


__all__ = [
    "DISCLOSURE",
    "PREDECLARED_TRIALS",
    "TRIAL_JUSTIFICATION",
    "LARGE_CAP_BASKET",
    "MID_LESS_COVERED_BASKET",
    "PeadParams",
    "equity_round_trip_cost_fraction",
    "simulate_pead_trade_return",
    "collect_positive_surprise_events",
    "simulate_event_trades",
    "run_pead_oos",
    "run_pead_oos_real",
]
