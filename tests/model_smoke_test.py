import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# Ensure the project root (containing the trading package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_ohlcv(n: int = 250, start_price: float = 150.0) -> pd.DataFrame:
    """Create a synthetic AAPL-like OHLCV DataFrame with lowercase yfinance columns."""
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    rets = np.random.normal(0.0005, 0.02, size=n)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + np.random.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(np.random.normal(0.0, 0.003, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(np.random.normal(0.0, 0.003, size=n)))
    volume = np.random.randint(100_000, 1_000_000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _assert_price_range(name: str, values: np.ndarray, low: float = 50.0, high: float = 1000.0) -> None:
    if values.size == 0:
        raise AssertionError(f"{name}: forecast array is empty")
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{name}: forecast contains non-finite values")
    if not np.all((values >= low) & (values <= high)):
        raise AssertionError(
            f"{name}: forecast values out of range {low}-{high}: "
            f"min={values.min():.2f}, max={values.max():.2f}"
        )


def test_xgboost():
    from trading.models.xgboost_model import XGBoostModel

    logger.info("Testing XGBoostModel...")
    df = make_ohlcv()
    model = XGBoostModel({})
    # XGBoostModel.train() is the primary training entry point
    model.train(df)
    # Use predict() directly to avoid joblib argument inspection issues on the
    # cached forecast wrapper in this standalone test context.
    preds = model.predict(df)
    fc = np.asarray(preds, dtype="float64").ravel()[-7:]
    logger.info("XGBoostModel forecast: %s", fc)
    _assert_price_range("XGBoostModel", fc)


def test_arima():
    from trading.models.arima_model import ARIMAModel

    logger.info("Testing ARIMAModel...")
    df = make_ohlcv()
    series = df["close"]
    model = ARIMAModel({"use_auto_arima": True})
    model.fit(series)
    res = model.forecast(series, horizon=7)
    # safe_forecast may return either the dict or the underlying data; normalize here
    if isinstance(res, dict):
        raw = res.get("forecast", res.get("values", []))
    else:
        raw = res
    fc = np.asarray(raw, dtype="float64").ravel()
    logger.info("ARIMAModel forecast: %s", fc)
    _assert_price_range("ARIMAModel", fc)


def test_lstm():
    try:
        from trading.models.lstm_model import LSTMForecaster
    except ImportError as e:
        logger.warning("Skipping LSTMForecaster (import error): %s", e)
        return

    logger.info("Testing LSTMForecaster...")
    df = make_ohlcv()
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
    # Train quickly with reduced epochs for smoke test
    model.fit(
        X=df[config["feature_columns"]],
        y=df["close"],
        epochs=3,
        batch_size=32,
    )
    res = model.forecast(df, horizon=7)
    fc = np.asarray(res.get("forecast", res), dtype="float64").ravel()
    logger.info("LSTMForecaster forecast: %s", fc)
    _assert_price_range("LSTMForecaster", fc)


def test_hybrid():
    """Test HybridModel with ARIMA + XGBoost as submodels."""
    from trading.forecasting.hybrid_model import HybridModel
    from trading.models.arima_model import ARIMAModel
    from trading.models.xgboost_model import XGBoostModel

    logger.info("Testing HybridModel (ARIMA + XGBoost)...")
    df = make_ohlcv()
    submodels = {
        "arima": ARIMAModel({"use_auto_arima": True}),
        "xgboost": XGBoostModel({}),
    }
    # Fit submodels individually
    submodels["arima"].fit(df["close"])
    submodels["xgboost"].fit(df, df["close"])

    hybrid = HybridModel(submodels)
    hybrid.fit(df)
    preds = hybrid.predict(df)
    preds = np.asarray(preds, dtype="float64").ravel()
    logger.info("HybridModel forecast: %s", preds[-7:])
    # Use last 7 as the comparable horizon
    _assert_price_range("HybridModel", preds[-7:])


def test_prophet():
    try:
        from trading.models.prophet_model import ProphetModel, PROPHET_AVAILABLE
    except ImportError as e:
        logger.warning("Skipping ProphetModel (import error): %s", e)
        return

    if not PROPHET_AVAILABLE:
        logger.warning("Skipping ProphetModel (Prophet not installed).")
        return

    logger.info("Testing ProphetModel...")
    df = make_ohlcv()
    # ProphetModel expects a date + target column; reuse index as date
    df_prophet = pd.DataFrame({"date": df.index, "close": df["close"].values})
    config = {"date_column": "date", "target_column": "close"}
    model = ProphetModel(config)
    model.fit(df_prophet)
    res = model.forecast(df_prophet, horizon=7)
    raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
    fc = np.asarray(raw, dtype="float64").ravel()
    logger.info("ProphetModel forecast: %s", fc)
    _assert_price_range("ProphetModel", fc)


def test_catboost():
    try:
        from trading.models.catboost_model import CatBoostModel
    except ImportError as e:
        logger.warning("Skipping CatBoostModel (import error): %s", e)
        return

    logger.info("Testing CatBoostModel...")
    df = make_ohlcv()
    config = {
        "feature_columns": list(df.columns),
        "target_column": "close",
        "catboost_params": {"iterations": 50, "depth": 4, "learning_rate": 0.1, "verbose": False},
    }
    model = CatBoostModel(config)
    model.fit(df)
    res = model.forecast(df, horizon=7)
    raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
    fc = np.asarray(raw, dtype="float64").ravel()
    logger.info("CatBoostModel forecast: %s", fc)
    _assert_price_range("CatBoostModel", fc)


def test_ridge():
    from trading.models.ridge_model import RidgeModel

    logger.info("Testing RidgeModel...")
    df = make_ohlcv()
    model = RidgeModel({})
    model.train(df)
    res = model.forecast(df, horizon=7)
    raw = res.get("predictions", res.get("forecast", [])) if isinstance(res, dict) else res
    fc = np.asarray(raw, dtype="float64").ravel()
    logger.info("RidgeModel forecast: %s", fc)
    _assert_price_range("RidgeModel", fc)


def test_tcn():
    try:
        from trading.models.tcn_model import TCNModel
    except ImportError as e:
        logger.warning("Skipping TCNModel (import error): %s", e)
        return

    logger.info("Testing TCNModel...")
    df = make_ohlcv()
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
    logger.info("TCNModel forecast: %s", fc)
    _assert_price_range("TCNModel", fc)


def test_ensemble():
    from trading.models.ensemble_model import EnsembleModel

    logger.info("Testing EnsembleModel...")
    df = make_ohlcv()
    # Use Ridge + XGBoost submodels that share the (data)->fit/predict interface
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
    logger.info("EnsembleModel forecast: %s", fc)
    _assert_price_range("EnsembleModel", fc)


def test_garch():
    from trading.models.garch_model import GARCHModel, ARCH_AVAILABLE

    logger.info("Testing GARCHModel...")
    df = make_ohlcv()
    model = GARCHModel({})
    if not ARCH_AVAILABLE:
        try:
            model.fit(df)
        except RuntimeError as e:
            msg = str(e)
            logger.info("GARCHModel raised RuntimeError as expected: %s", msg)
            if "GARCH requires arch package" not in msg:
                raise
        else:
            raise AssertionError("GARCHModel.fit() did not raise RuntimeError when arch is missing")
        return

    # When arch is installed, this should behave like other models
    model.fit(df)
    res = model.forecast(df, horizon=7)
    raw = res.get("forecast", res.get("values", [])) if isinstance(res, dict) else res
    fc = np.asarray(raw, dtype="float64").ravel()
    logger.info("GARCHModel forecast: %s", fc)
    _assert_price_range("GARCHModel", fc)


def main():
    start = datetime.now()
    logger.info("Running model smoke tests...")
    test_xgboost()
    test_arima()
    test_lstm()
    test_hybrid()
    test_prophet()
    test_catboost()
    test_ridge()
    test_tcn()
    test_ensemble()
    test_garch()
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("All smoke tests completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()

