#!/usr/bin/env python
"""One-shot Optuna tune for forecast models; saves params under models/best_params/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optuna_tune")


def _features(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Simple return-prediction features (next-day return target)."""
    df = pd.DataFrame({"close": close.astype(float)})
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["sma_10"] = df["close"].rolling(10).mean() / df["close"] - 1
    df["sma_20"] = df["close"].rolling(20).mean() / df["close"] - 1
    df["target"] = df["ret_1"].shift(-1)
    df = df.dropna()
    y = df["target"]
    X = df.drop(columns=["target", "close"])
    return X, y


def main() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from trading.data.price_cache import get_history
    from trading.optimization.optuna_optimizer import HyperparameterOptimizer

    hist = get_history("SPY", period="2y")
    if hist is None or hist.empty:
        raise SystemExit("no SPY history")
    cm = {str(c).lower(): c for c in hist.columns}
    close = hist[cm.get("close", hist.columns[0])]
    X, y = _features(close)
    logger.info("rows=%s features=%s", len(X), list(X.columns))

    opt = HyperparameterOptimizer(backend="optuna", study_name="evolve_forecast_tune")
    # Keep trials modest for interactive/CI time
    result = opt.optimize_xgboost(X, y, n_trials=30)
    best = result.get("best_params") or {}
    score = result.get("best_score")
    out_dir = Path("models/best_params")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": "SPY",
        "model": "xgboost",
        "best_params": best,
        "best_score": score,
        "n_trials": 30,
        "note": "Tuned on next-day return RMSE via Optuna",
    }
    path = out_dir / "xgboost_consensus_best.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved %s params=%s score=%s", path, best, score)

    # Also write a ridge alpha sweep (cheap)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    best_alpha, best_rmse = 1.0, 1e9
    tscv = TimeSeriesSplit(n_splits=3)
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        rmses = []
        for tr, te in tscv.split(X):
            m = Ridge(alpha=alpha, max_iter=2000)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
        rmse = float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha
    ridge_path = out_dir / "ridge_consensus_best.json"
    ridge_path.write_text(
        json.dumps({"best_params": {"alpha": best_alpha}, "best_score": best_rmse}, indent=2),
        encoding="utf-8",
    )
    logger.info("ridge best alpha=%s rmse=%s", best_alpha, best_rmse)


if __name__ == "__main__":
    main()
