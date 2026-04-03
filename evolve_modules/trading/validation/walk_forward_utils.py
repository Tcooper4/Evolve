"""
Walk-Forward Validation Utilities
==================================
INTEGRATION NOTES:
- Drop into: trading/validation/walk_forward_utils.py (replace existing stub)
- Wire into: pages/5_Backtest.py Walk-Forward tab
- Call pattern:
    from trading.validation.walk_forward_utils import WalkForwardValidator
    wfv = WalkForwardValidator(model_name="xgboost", symbol="AAPL")
    results = wfv.run(data, train_window=252, test_window=63, step=21)
    summary = wfv.get_summary()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: List[float]
    actuals: List[float]
    mae: float
    mse: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    model_name: str
    symbol: str
    windows: List[WalkForwardWindow] = field(default_factory=list)
    model_performance: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.windows:
            return pd.DataFrame()
        rows = []
        for w in self.windows:
            rows.append({
                "window": w.window_index,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "mae": round(w.mae, 4),
                "mse": round(w.mse, 4),
                "mape": round(w.mape, 2),
                "directional_accuracy": round(w.directional_accuracy, 3),
                "sharpe_ratio": round(w.sharpe_ratio, 3),
                "max_drawdown": round(w.max_drawdown, 4),
            })
        return pd.DataFrame(rows)


class WalkForwardValidator:
    """
    Proper walk-forward validator for time series forecasting models.

    Implements expanding or rolling window validation with full
    performance metrics per window and aggregated summary statistics.

    Args:
        model_name: Name of model to validate (must be in forecast_router)
        symbol: Ticker symbol
        window_type: 'expanding' (default) or 'rolling'
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        symbol: str = "AAPL",
        window_type: str = "expanding",
    ):
        self.model_name = model_name
        self.symbol = symbol
        self.window_type = window_type
        self.results: Optional[WalkForwardResult] = None
        self._windows: List[WalkForwardWindow] = []

    def run(
        self,
        data: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        horizon: int = 7,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            train_window: Minimum training window in days
            test_window: Test window size in days
            step_size: Days to advance each step
            horizon: Forecast horizon (days ahead)

        Returns:
            WalkForwardResult with per-window and aggregate metrics
        """
        self._windows = []

        if data is None or len(data) < train_window + test_window:
            logger.warning(
                "WalkForwardValidator: insufficient data "
                "(%d rows, need %d)",
                len(data) if data is not None else 0,
                train_window + test_window,
            )
            return WalkForwardResult(
                model_name=self.model_name,
                symbol=self.symbol,
                model_performance={"error": "Insufficient data"},
            )

        # Normalize column names
        _col_map = {c.lower(): c for c in data.columns}
        close_col = _col_map.get("close", list(
            data.select_dtypes(include="number").columns
        )[0])

        total_rows = len(data)
        window_idx = 0
        start_idx = train_window

        while start_idx + test_window <= total_rows:
            try:
                # Define train/test splits
                if self.window_type == "rolling":
                    train_start_idx = start_idx - train_window
                else:  # expanding
                    train_start_idx = 0

                train_end_idx = start_idx
                test_start_idx = start_idx
                test_end_idx = min(start_idx + test_window, total_rows)

                train_data = data.iloc[train_start_idx:train_end_idx].copy()
                test_data = data.iloc[test_start_idx:test_end_idx].copy()

                # Generate forecasts using forecast router
                preds, actuals = self._generate_forecasts(
                    train_data, test_data, close_col, horizon
                )

                if preds is None or len(preds) == 0:
                    logger.warning(
                        "WalkForwardValidator: window %d forecast failed",
                        window_idx,
                    )
                    start_idx += step_size
                    window_idx += 1
                    continue

                # Calculate metrics
                metrics = self._calculate_metrics(preds, actuals)

                window = WalkForwardWindow(
                    window_index=window_idx,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx - 1],
                    predictions=preds,
                    actuals=actuals,
                    **metrics,
                )
                self._windows.append(window)
                logger.info(
                    "WalkForwardValidator: window %d complete "
                    "— MAPE=%.2f%% DA=%.1f%%",
                    window_idx,
                    metrics["mape"],
                    metrics["directional_accuracy"] * 100,
                )

            except Exception as e:
                logger.warning(
                    "WalkForwardValidator: window %d error: %s",
                    window_idx, e,
                )

            start_idx += step_size
            window_idx += 1

        self.results = WalkForwardResult(
            model_name=self.model_name,
            symbol=self.symbol,
            windows=self._windows,
            model_performance=self._aggregate_performance(),
        )
        return self.results

    def _generate_forecasts(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        close_col: str,
        horizon: int,
    ) -> Tuple[List[float], List[float]]:
        """Generate forecasts for a single window."""
        try:
            from trading.models.forecast_router import ForecastRouter
            router = ForecastRouter()

            # Fit on train, predict on test
            preds = []
            actuals = []

            # Step through test data one horizon at a time
            for i in range(0, len(test_data), horizon):
                context = pd.concat([
                    train_data,
                    test_data.iloc[:i]
                ]) if i > 0 else train_data

                if len(context) < 30:
                    continue

                result = router.forecast(
                    data=context,
                    model=self.model_name,
                    horizon=horizon,
                )

                if result is None:
                    continue

                forecast_vals = result.get(
                    "forecast",
                    result.get("predictions", [])
                )
                if not forecast_vals:
                    continue

                # Actual values for this window
                actual_slice = test_data.iloc[
                    i:i + horizon
                ][close_col].values.tolist()

                min_len = min(len(forecast_vals), len(actual_slice))
                preds.extend(forecast_vals[:min_len])
                actuals.extend(actual_slice[:min_len])

            return preds, actuals

        except Exception as e:
            logger.warning(
                "WalkForwardValidator: forecast generation failed: %s", e
            )
            return [], []

    def _calculate_metrics(
        self,
        predictions: List[float],
        actuals: List[float],
    ) -> Dict[str, float]:
        """Calculate performance metrics for a window."""
        if not predictions or not actuals:
            return {
                "mae": float("nan"),
                "mse": float("nan"),
                "mape": float("nan"),
                "directional_accuracy": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
            }

        preds = np.array(predictions)
        acts = np.array(actuals)
        min_len = min(len(preds), len(acts))
        preds = preds[:min_len]
        acts = acts[:min_len]

        # Error metrics
        mae = float(np.mean(np.abs(preds - acts)))
        mse = float(np.mean((preds - acts) ** 2))

        # MAPE — avoid division by zero
        nonzero = acts != 0
        mape = float(
            np.mean(np.abs((preds[nonzero] - acts[nonzero]) / acts[nonzero])) * 100
        ) if nonzero.any() else float("nan")

        # Directional accuracy
        if len(acts) > 1:
            actual_dirs = np.sign(np.diff(acts))
            pred_dirs = np.sign(np.diff(preds))
            da = float(np.mean(actual_dirs == pred_dirs))
        else:
            da = float("nan")

        # Returns-based metrics
        if len(acts) > 1:
            returns = np.diff(acts) / acts[:-1]
            pred_returns = np.diff(preds) / preds[:-1]

            # Sharpe (annualized, assuming daily)
            if returns.std() > 0:
                sharpe = float(
                    returns.mean() / returns.std() * np.sqrt(252)
                )
            else:
                sharpe = 0.0

            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = float(drawdown.min())
        else:
            sharpe = float("nan")
            max_dd = float("nan")

        return {
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "directional_accuracy": da,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    def _aggregate_performance(self) -> Dict[str, Any]:
        """Aggregate metrics across all windows."""
        if not self._windows:
            return {"error": "No windows completed"}

        maes = [w.mae for w in self._windows if not np.isnan(w.mae)]
        mapes = [w.mape for w in self._windows if not np.isnan(w.mape)]
        das = [w.directional_accuracy for w in self._windows
               if not np.isnan(w.directional_accuracy)]
        sharpes = [w.sharpe_ratio for w in self._windows
                   if not np.isnan(w.sharpe_ratio)]
        drawdowns = [w.max_drawdown for w in self._windows
                     if not np.isnan(w.max_drawdown)]

        return {
            "model": self.model_name,
            "symbol": self.symbol,
            "n_windows": len(self._windows),
            "mean_mae": round(np.mean(maes), 4) if maes else None,
            "mean_mape": round(np.mean(mapes), 2) if mapes else None,
            "mean_directional_accuracy": round(np.mean(das), 3) if das else None,
            "mean_sharpe_ratio": round(np.mean(sharpes), 3) if sharpes else None,
            "mean_max_drawdown": round(np.mean(drawdowns), 4) if drawdowns else None,
            "consistency_score": self._consistency_score(das),
            "window_type": self.window_type,
        }

    def _consistency_score(self, directional_accuracies: List[float]) -> float:
        """
        Score 0-10 measuring how consistent the model is across windows.
        High score = model works consistently, not just in some windows.
        """
        if not directional_accuracies or len(directional_accuracies) < 2:
            return 5.0
        mean_da = np.mean(directional_accuracies)
        std_da = np.std(directional_accuracies)
        # Penalize high variance
        raw = (mean_da - 0.5) * 10 - std_da * 5
        return round(float(np.clip(raw + 5, 0, 10)), 2)

    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary of results."""
        if self.results is None:
            return {"error": "Run validate() first"}
        return self.results.model_performance

    def get_dataframe(self) -> pd.DataFrame:
        """Get per-window results as DataFrame for display."""
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()
