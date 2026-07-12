# -*- coding: utf-8 -*-
"""Self-tuning loop: the honest version of "models optimize themselves."

Design principles (why this is NOT a magic self-improving black box):

* **Adopt only what proves itself out-of-sample.** Each cycle re-optimizes
  a strategy's parameters on the older 75% of recent history and adopts
  the new parameters ONLY if they beat the currently-adopted parameters
  on the held-out 25% the optimizer never saw. Without that gate, a
  self-tuning loop is an overfitting machine that gets more confident
  and less correct with every cycle.
* **Everything is recorded.** Every cycle writes to the leaderboard and
  a tuning journal (data/self_tune.json): what was tried, what won, what
  the OOS numbers were, and whether adoption happened. Adaptation you
  can't audit is drift, not learning.
* **Champion/challenger, never hot-swap.** Current parameters stay the
  champion until a challenger beats them where it counts. Ties and
  marginal wins (< min_improvement) keep the champion - churn is a cost.

Run one cycle manually::

    python -m trading.services.self_tune --symbols SPY QQQ

or on a schedule (cron/systemd timer), or via the ``retune_strategies``
chat/MCP tool. Adopted parameters are stored per (strategy, symbol) in
``data/self_tune.json`` and read back by ``get_adopted_params``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STORE_PATH = Path("data/self_tune.json")

DEFAULT_STRATEGIES = ["RSIStrategy", "MACDStrategy", "BollingerStrategy",
                      "SMAStrategy"]


def _load_store() -> Dict[str, Any]:
    try:
        if STORE_PATH.exists():
            return json.loads(STORE_PATH.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("self_tune store unreadable: %s", e)
    return {"adopted": {}, "journal": []}


def _save_store(store: Dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, default=str))


def get_adopted_params(strategy: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Currently-adopted (champion) parameters, or None for defaults."""
    entry = _load_store()["adopted"].get(f"{strategy}:{symbol}")
    return dict(entry["params"]) if entry else None


def clear_adopted_params(strategy: str, symbol: str) -> bool:
    """Remove adopted params for (strategy, symbol). Returns True if removed."""
    store = _load_store()
    key = f"{strategy}:{symbol}"
    if key not in store.get("adopted", {}):
        return False
    del store["adopted"][key]
    journal = list(store.get("journal") or [])
    journal.append({
        "strategy": strategy,
        "symbol": symbol,
        "adopted": False,
        "source": "clear",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": {},
    })
    store["journal"] = journal[-200:]
    _save_store(store)
    return True


def adopt_params(
    strategy: str,
    symbol: str,
    params: Dict[str, Any],
    *,
    oos_metrics: Optional[Dict[str, Any]] = None,
    source: str = "manual_optimize",
) -> Dict[str, Any]:
    """Persist tuned parameters for (strategy, symbol) so later backtests reuse them."""
    store = _load_store()
    key = f"{strategy}:{symbol}"
    entry = {
        "params": dict(params or {}),
        "oos_metrics": oos_metrics or {},
        "source": source,
        "adopted_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    store["adopted"][key] = entry
    journal = list(store.get("journal") or [])
    journal.append({
        "strategy": strategy,
        "symbol": symbol,
        "adopted": True,
        "source": source,
        "timestamp": entry["adopted_at"],
        "params": entry["params"],
    })
    store["journal"] = journal[-200:]
    _save_store(store)
    return entry


def _oos_metric(strategy: str, df, params: Optional[Dict[str, Any]],
                train_fraction: float, metric: str) -> Optional[float]:
    """Evaluate params on the held-out tail only."""
    from trading.optimization.strategy_backtest_objective import evaluate_params

    split = int(len(df) * train_fraction)
    test = df.iloc[split:]
    if len(test) < 40:
        return None
    result = evaluate_params(strategy, test, params or {})
    if isinstance(result, dict):
        if "error" in result:
            return None
        return result.get(metric)
    return float(result) if result is not None else None


def tune_one(
    strategy: str,
    symbol: str,
    df=None,
    method: str = "pso",
    metric: str = "sharpe_ratio",
    max_evaluations: int = 60,
    train_fraction: float = 0.75,
    min_improvement: float = 0.05,
) -> Dict[str, Any]:
    """One champion/challenger cycle for (strategy, symbol).

    Returns a journal entry dict with champion/challenger OOS numbers and
    whether adoption happened. Pass ``df`` (OHLCV) to run offline/on test
    data; otherwise fetches ``2y`` of daily history.
    """
    import trading.strategies  # noqa: F401 - registry discovery

    if df is None:
        import yfinance as yf

        df = yf.Ticker(symbol).history(period="2y", interval="1d")
        if df is None or df.empty:
            return {"success": False, "strategy": strategy, "symbol": symbol,
                    "error": "no price data"}
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)

    # Challenger: optimize on the TRAIN window only
    try:
        from trading.optimization.strategy_backtest_objective import (
            optimize_strategy,
        )

        split = int(len(df) * train_fraction)
        train = df.iloc[:split]
        opt = optimize_strategy(strategy, train, method=method, metric=metric,
                                max_evaluations=max_evaluations)
        # optimize_strategy returns a StrategyOptimizationRun dataclass
        challenger = dict(getattr(opt, "best_params", None) or {})
    except Exception as e:  # noqa: BLE001
        return {"success": False, "strategy": strategy, "symbol": symbol,
                "error": f"optimization failed: {e}"}

    champion = get_adopted_params(strategy, symbol)  # None => defaults
    champ_oos = _oos_metric(strategy, df, champion, train_fraction, metric)
    chall_oos = _oos_metric(strategy, df, challenger, train_fraction, metric)

    adopted = False
    reason = "challenger did not beat champion out-of-sample"
    if chall_oos is not None and (
        champ_oos is None or chall_oos > champ_oos + abs(min_improvement)
    ):
        adopted = True
        reason = "challenger beat champion on held-out data"

    entry = {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "metric": metric,
        "champion_params": champion,
        "champion_oos": champ_oos,
        "challenger_params": challenger,
        "challenger_oos": chall_oos,
        "adopted": adopted,
        "reason": reason,
    }

    store = _load_store()
    if adopted:
        store["adopted"][f"{strategy}:{symbol}"] = {
            "params": challenger,
            "oos_metric": chall_oos,
            "metric": metric,
            "adopted_at": entry["timestamp"],
        }
    store["journal"].append(entry)
    store["journal"] = store["journal"][-500:]  # bounded audit trail
    _save_store(store)

    # Leaderboard record (adaptation you can audit)
    try:
        from trading.agents.agent_leaderboard import AgentLeaderboard

        AgentLeaderboard().update_performance(
            agent_name=f"{strategy}:{symbol}",
            sharpe_ratio=float(chall_oos if adopted else (champ_oos or 0.0)),
            max_drawdown=0.0,
            win_rate=0.0,
            total_return=0.0,
            extra_metrics={"self_tune": True, "adopted": adopted},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("leaderboard record skipped: %s", e)

    return entry


def tune_all(
    symbols: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One full cycle across symbols x strategies. Never raises."""
    results = []
    for sym in symbols or ["SPY"]:
        for strat in strategies or DEFAULT_STRATEGIES:
            try:
                results.append(tune_one(strat, sym, **kwargs))
            except Exception as e:  # noqa: BLE001
                results.append({"success": False, "strategy": strat,
                                "symbol": sym, "error": str(e)})
    adopted = [r for r in results if r.get("adopted")]
    return {
        "success": True,
        "cycles": len(results),
        "adoptions": len(adopted),
        "results": results,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run one self-tuning cycle")
    ap.add_argument("--symbols", nargs="+", default=["SPY"])
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    ap.add_argument("--method", default="pso")
    args = ap.parse_args()
    out = tune_all(symbols=args.symbols, strategies=args.strategies,
                   method=args.method)
    print(json.dumps({k: v for k, v in out.items() if k != "results"},
                     indent=2))
    for r in out["results"]:
        print(f"  {r.get('strategy')}:{r.get('symbol')} "
              f"adopted={r.get('adopted')} oos={r.get('challenger_oos')}")


if __name__ == "__main__":
    main()
