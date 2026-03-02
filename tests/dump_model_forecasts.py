import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# Ensure the project root (containing the trading package) is on sys.path so
# we can import tests.model_smoke_test in the same way as the main smoke file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import model_smoke_test as smoke


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Re-run the smoke-test style forecasts and print final horizon values."""
    results: dict[str, dict[str, object]] = {}

    # Use a single synthetic OHLCV sample for all models to keep runtime bounded
    base_df: pd.DataFrame = smoke.make_ohlcv()

    # XGBoost
    try:
        from trading.models.xgboost_model import XGBoostModel

        df = base_df.copy()
        model = XGBoostModel({})
        model.train(df)
        preds = model.predict(df)
        fc = np.asarray(preds, dtype="float64").ravel()[-7:]
        results["XGBoostModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["XGBoostModel"] = {"status": f"ERROR: {e}", "values": []}

    # ARIMA
    try:
        from trading.models.arima_model import ARIMAModel

        df = base_df.copy()
        series = df["close"]
        model = ARIMAModel({"use_auto_arima": True})
        model.fit(series)
        res = model.forecast(series, horizon=7)
        raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
        fc = np.asarray(raw, dtype="float64").ravel()
        results["ARIMAModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["ARIMAModel"] = {"status": f"ERROR: {e}", "values": []}

    # LSTMForecaster (optional)
    try:
        from trading.models.lstm_model import LSTMForecaster

        df = base_df.copy()
        config = {
            "input_size": df.shape[1],
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 30,
            "feature_columns": list(df.columns),
            "target_column": "close",
            "max_sequence_length": 60,
            "max_batch_size": 128,
            "max_epochs": 10,
        }
        model = LSTMForecaster(config)
        model.fit(
            X=df[config["feature_columns"]],
            y=df["close"],
            epochs=3,
            batch_size=32,
        )
        res = model.forecast(df, horizon=7)
        raw = res.get("forecast", res) if isinstance(res, dict) else res
        fc = np.asarray(raw, dtype="float64").ravel()
        results["LSTMForecaster"] = {"status": "PASS", "values": fc.tolist()}
    except ImportError as e:
        results["LSTMForecaster"] = {"status": f"SKIPPED (import error: {e})", "values": []}
    except Exception as e:
        results["LSTMForecaster"] = {"status": f"ERROR: {e}", "values": []}

    # HybridModel with ARIMA + XGBoost
    try:
        from trading.forecasting.hybrid_model import HybridModel
        from trading.models.arima_model import ARIMAModel
        from trading.models.xgboost_model import XGBoostModel

        df = base_df.copy()
        submodels = {
            "arima": ARIMAModel({"use_auto_arima": True}),
            "xgboost": XGBoostModel({}),
        }
        submodels["arima"].fit(df["close"])
        submodels["xgboost"].fit(df, df["close"])
        hybrid = HybridModel(submodels)
        hybrid.fit(df)
        preds = hybrid.predict(df)
        fc = np.asarray(preds, dtype="float64").ravel()[-7:]
        results["HybridModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["HybridModel"] = {"status": f"ERROR: {e}", "values": []}

    # ProphetModel (optional)
    try:
        from trading.models.prophet_model import ProphetModel, PROPHET_AVAILABLE

        if PROPHET_AVAILABLE:
            df = base_df.copy()
            df_prophet = pd.DataFrame({"date": df.index, "close": df["close"].values})
            config = {"date_column": "date", "target_column": "close"}
            model = ProphetModel(config)
            model.fit(df_prophet)
            res = model.forecast(df_prophet, horizon=7)
            raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
            fc = np.asarray(raw, dtype="float64").ravel()
            results["ProphetModel"] = {"status": "PASS", "values": fc.tolist()}
        else:
            results["ProphetModel"] = {"status": "SKIPPED (Prophet not installed)", "values": []}
    except ImportError as e:
        results["ProphetModel"] = {"status": f"SKIPPED (import error: {e})", "values": []}
    except Exception as e:
        results["ProphetModel"] = {"status": f"ERROR: {e}", "values": []}

    # CatBoostModel (optional)
    try:
        from trading.models.catboost_model import CatBoostModel

        df = base_df.copy()
        config = {
            "feature_columns": list(df.columns),
            "target_column": "close",
            "catboost_params": {
                "iterations": 50,
                "depth": 4,
                "learning_rate": 0.1,
                "verbose": False,
            },
        }
        model = CatBoostModel(config)
        model.fit(df)
        res = model.forecast(df, horizon=7)
        raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
        fc = np.asarray(raw, dtype="float64").ravel()
        results["CatBoostModel"] = {"status": "PASS", "values": fc.tolist()}
    except ImportError as e:
        results["CatBoostModel"] = {"status": f"SKIPPED (import error: {e})", "values": []}
    except Exception as e:
        results["CatBoostModel"] = {"status": f"ERROR: {e}", "values": []}

    # RidgeModel
    try:
        from trading.models.ridge_model import RidgeModel

        df = base_df.copy()
        model = RidgeModel({})
        model.train(df)
        res = model.forecast(df, horizon=7)
        raw = res.get("predictions", res.get("forecast", [])) if isinstance(res, dict) else res
        fc = np.asarray(raw, dtype="float64").ravel()
        results["RidgeModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["RidgeModel"] = {"status": f"ERROR: {e}", "values": []}

    # TCNModel (optional)
    try:
        from trading.models.tcn_model import TCNModel

        df = base_df.copy()
        config = {
            "feature_columns": ["close", "volume"],
            "target_column": "close",
            "sequence_length": 20,
        }
        model = TCNModel(config)
        model.fit(df, epochs=2, batch_size=32, learning_rate=0.001)
        res = model.forecast(df, horizon=7)
        if isinstance(res, dict):
            inner = res.get("result", {})
            raw = inner.get("forecast", inner.get("values", []))
        else:
            raw = res
        fc = np.asarray(raw, dtype="float64").ravel()
        results["TCNModel"] = {"status": "PASS", "values": fc.tolist()}
    except ImportError as e:
        results["TCNModel"] = {"status": f"SKIPPED (import error: {e})", "values": []}
    except Exception as e:
        results["TCNModel"] = {"status": f"ERROR: {e}", "values": []}

    # EnsembleModel with Ridge + XGBoost
    try:
        from trading.models.ensemble_model import EnsembleModel

        df = base_df.copy()
        config = {
            "models": [
                {"name": "Ridge", "class_path": "trading.models.ridge_model.RidgeModel"},
                {"name": "XGBoost", "class_path": "trading.models.xgboost_model.XGBoostModel"},
            ],
            "voting_method": "mse",
            "weight_window": 20,
        }
        model = EnsembleModel(config)
        model.fit(df)
        preds = model.predict(df)
        fc = np.asarray(preds, dtype="float64").ravel()[-7:]
        results["EnsembleModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["EnsembleModel"] = {"status": f"ERROR: {e}", "values": []}

    # GARCHModel
    try:
        from trading.models.garch_model import GARCHModel, ARCH_AVAILABLE

        df = base_df.copy()
        model = GARCHModel({})
        if not ARCH_AVAILABLE:
            # Match smoke test behaviour – expect RuntimeError when arch is missing
            try:
                model.fit(df)
            except RuntimeError as err:
                msg = str(err)
                if "GARCH requires arch package" in msg:
                    results["GARCHModel"] = {
                        "status": "SKIPPED (arch not installed)",
                        "values": [],
                    }
                else:
                    results["GARCHModel"] = {"status": f"ERROR: {err}", "values": []}
            else:
                results["GARCHModel"] = {
                    "status": "ERROR: expected RuntimeError when arch is missing",
                    "values": [],
                }
        else:
            model.fit(df)
            res = model.forecast(df, horizon=7)
            raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
            fc = np.asarray(raw, dtype="float64").ravel()
            results["GARCHModel"] = {"status": "PASS", "values": fc.tolist()}
    except Exception as e:
        results["GARCHModel"] = {"status": f"ERROR: {e}", "values": []}

    # Pretty-print results in a simple, parseable format
    print(f"# Smoke test forecast dump ({datetime.utcnow().isoformat()}Z)")
    for name, payload in results.items():
        status = payload.get("status", "UNKNOWN")
        values = payload.get("values", [])
        print(f"{name} | {status} | {values}")


if __name__ == "__main__":
    main()

