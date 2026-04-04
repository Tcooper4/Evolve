"""Forecast router for managing and selecting forecasting models.

This module provides a router for selecting and managing different forecasting models
based on data characteristics, performance history, and user preferences. It supports
multiple model types including statistical (ARIMA), deep learning (LSTM), and
tree-based (XGBoost) models.

The router implements a dynamic model selection strategy that considers:
1. Data characteristics (time series length, seasonality, trend)
2. Historical model performance
3. Computational requirements
4. User preferences and constraints

Example:
    ```python
    router = ForecastRouter()
    forecast = router.get_forecast(
        data=time_series_data,
        horizon=30,
        model_type='auto'  # Let router select best model
    )
    ```
"""

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.models.forecast_features import add_forecast_features, prepare_forecast_data
from trading.models.arima_model import ARIMAModel
from trading.models.lstm_model import LSTMModel
from trading.models.ridge_model import RidgeModel
from trading.models.xgboost_model import XGBoostModel
logger = logging.getLogger(__name__)

# Make Prophet import optional
try:
    from trading.models.prophet_model import ProphetModel

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    ProphetModel = None

# Try to import Autoformer from NeuralForecast
try:
    from trading.models.neuralforecast_models import AutoformerModel, NEURALFORECAST_AVAILABLE
    AUTOFORMER_AVAILABLE = NEURALFORECAST_AVAILABLE
except ImportError:
    AUTOFORMER_AVAILABLE = False
    AutoformerModel = None

# Try to import Transformer model
try:
    from trading.models.advanced.transformer.time_series_transformer import (
        TransformerForecaster,
    )

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TransformerForecaster = None

# Stem (from filename) -> (module_path, class_name) for discovery
_DISCOVERY_CLASS_MAP = {
    "arima": ("trading.models.arima_model", "ARIMAModel"),
    "base": ("trading.models.base_model", "BaseModel"),
    "catboost": ("trading.models.catboost_model", "CatBoostModel"),
    "ensemble": ("trading.models.ensemble_model", "EnsembleModel"),
    "garch": ("trading.models.garch_model", "GARCHModel"),
    "lstm": ("trading.models.lstm_model", "LSTMModel"),
    "prophet": ("trading.models.prophet_model", "ProphetModel"),
    "ridge": ("trading.models.ridge_model", "RidgeModel"),
    "tcn": ("trading.models.tcn_model", "TCNModel"),
    "xgboost": ("trading.models.xgboost_model", "XGBoostModel"),
}
# Aliases for config class_path that use wrong casing (e.g. ArimaModel -> ARIMAModel)
_CLASS_NAME_ALIASES = {
    "ArimaModel": "ARIMAModel",
    "LstmModel": "LSTMModel",
    "XgboostModel": "XGBoostModel",
    "CatboostModel": "CatBoostModel",
    "GarchModel": "GARCHModel",
    "TcnModel": "TCNModel",
}

try:
    from trading.system_resilience import handle_model_failure as _resilience_handle_failure
except Exception:
    _resilience_handle_failure = None  # type: ignore


class ForecastRouter:
    """Router for managing and selecting forecasting models.

    This class implements a dynamic model selection strategy based on data
    characteristics and historical performance. It supports multiple model types
    and provides a unified interface for forecasting.

    Attributes:
        model_registry (Dict[str, Any]): Registry of available models
        performance_history (pd.DataFrame): Historical performance metrics
        model_weights (Dict[str, float]): Weights for model selection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the forecast router.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_registry = {}
        self._load_model_registry()

        # Add Prophet only if available
        if PROPHET_AVAILABLE:
            self.model_registry["prophet"] = ProphetModel

        # Add Transformer only if available
        if TRANSFORMER_AVAILABLE:
            self.model_registry["transformer"] = TransformerForecaster

        self.performance_history = pd.DataFrame()
        self.model_weights = self._initialize_weights()
        self._last_price_used = 1.0
        self._last_stationarity_meta: Dict[str, Any] = {
            "is_stationary": True,
            "note": None,
        }

    @staticmethod
    @lru_cache(maxsize=64)
    def _cached_model_key(
        symbol: str,
        start: str,
        end: str,
        model_name: str,
    ) -> Tuple[str, str, str, str]:
        """Lightweight cache key helper for trained model instances."""
        return (symbol or "", start or "", end or "", model_name or "")

    def _get_cached_trained_model(
        self,
        data: pd.DataFrame,
        selected_model: str,
        run_walk_forward: bool,
        **kwargs: Any,
    ):
        """Return a trained model instance, cached by (symbol, date-range, model).

        This prevents 3–5x redundant retraining across Quick Forecast,
        EnsembleModel, HybridModel, and consensus calls for the same
        ticker/date-range/model combination.
        """
        # Derive a simple key from the primary price column and index range
        symbol = str(kwargs.get("symbol") or kwargs.get("ticker") or "")
        if not symbol and isinstance(data, pd.DataFrame) and "symbol" in data.columns:
            try:
                symbol = str(data["symbol"].iloc[0])
            except Exception:
                symbol = ""
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
            start = data.index[0].isoformat()
            end = data.index[-1].isoformat()
        else:
            start = ""
            end = ""

        _cache_model = selected_model
        if selected_model == "arima" and kwargs.get("fast_mode"):
            _cache_model = "arima|fast"

        key = self._cached_model_key(symbol, start, end, _cache_model)

        # Attach a simple in-memory cache dict on the instance
        cache: Dict[Tuple[str, str, str, str], Any] = getattr(
            self, "_trained_model_cache", {}
        )

        if key in cache:
            return cache[key]

        # Instantiate and fit a fresh model, then cache it
        model_class = self.model_registry[selected_model]
        config = self._get_model_defaults(selected_model)
        config.update(kwargs)

        if selected_model == "lstm":
            config["input_dim"] = (
                data.shape[1] if hasattr(data, "shape") else 1
            )
            config["output_dim"] = 1

        if selected_model == "hybrid":
            from trading.forecasting.hybrid_model import HybridModel

            arima_sub = ARIMAModel(
                {
                    "order": (5, 1, 0),
                    "use_auto_arima": True,
                    "target_column": "close",
                }
            )
            ridge_sub = RidgeModel(
                {
                    "alpha": 1.0,
                    "target_column": "close",
                }
            )
            hybrid_models = {"arima": arima_sub, "ridge": ridge_sub}
            try:
                _arima = ARIMAModel(
                    {
                        "order": (5, 1, 0),
                        "use_auto_arima": True,
                        "target_column": config.get("target_column", "close"),
                    }
                )
                _ridge = RidgeModel(
                    {
                        "alpha": 1.0,
                        "max_iter": 1000,
                        "target_column": config.get("target_column", "close"),
                    }
                )
                hybrid_models = {"arima": _arima, "ridge": _ridge}
            except Exception as _e:
                logger.warning(f"Hybrid sub-model injection failed: {_e}")
            model = HybridModel(hybrid_models)
        elif selected_model == "gnn":
            from trading.models.advanced.gnn.gnn_model import GNNForecaster

            # GNNForecaster expects explicit kwargs, not a config dict.
            # Infer number of assets from the data columns as a safe default.
            _n_assets = (
                data.shape[1]
                if isinstance(data, pd.DataFrame) and data.shape[1] > 1
                else 1
            )
            # Clamp to a reasonable range
            _n_assets = max(3, min(20, _n_assets))

            model = GNNForecaster(
                num_assets=_n_assets,
                hidden_size=64,
                num_layers=2,
                seq_length=30,
                learning_rate=0.001,
            )
        else:
            model = model_class(config)

        fit_args: Dict[str, Any] = {}
        if selected_model == "lstm":
            fit_args = {
                "epochs": config.get("epochs", 50),
                "batch_size": config.get("batch_size", 32),
                "learning_rate": config.get("learning_rate", 0.001),
            }

        try:
            if selected_model == "lstm":
                model.train_model(data, data.iloc[:, 0], **fit_args)
            else:
                model.fit(data)
            logger.info(f"Successfully fitted {selected_model}")
        except Exception as e:
            logger.error(f"Failed to fit {selected_model}: {e}")
            if _resilience_handle_failure is not None:
                try:
                    _resilience_handle_failure(
                        selected_model,
                        e,
                        {"stage": "fit", "symbol": symbol, "start": start, "end": end},
                    )
                except Exception:
                    pass
            # Fallback to a simpler model (non-cached) if training fails
            fallback_model = self._get_fallback_model(selected_model)
            logger.info(f"Trying fallback model: {fallback_model}")
            model_class = self.model_registry[fallback_model]
            fallback_config = self._get_model_defaults(fallback_model)
            if fallback_model == "lstm":
                fallback_config["input_dim"] = (
                    data.shape[1] if hasattr(data, "shape") else 1
                )
                fallback_config["output_dim"] = 1
            model = model_class(fallback_config)
            model.fit(data)

        # Store in cache on success
        cache[key] = model
        self._trained_model_cache = cache
        return model

    def _load_model_registry(self):
        """Dynamically load model registry from config or discovery."""
        try:
            config_models = self.config.get("models", {})
            if config_models:
                self._load_models_from_config(config_models)
            else:
                from trading.models.model_registry import get_registry as get_model_registry

                registry = get_model_registry()
                # ModelRegistry exposes .registry (property) or ._models
                raw_registry = (
                    getattr(registry, "registry", None)
                    or getattr(registry, "_models", None)
                    or {}
                )

                # Deduplicate and normalize: use lowercase keys as canonical form
                deduped: Dict[str, Any] = {}
                for name, model_cls in raw_registry.items():
                    key = str(name).strip().lower()
                    # First writer wins; log if we see a duplicate alias
                    if key in deduped:
                        logger.debug(
                            "Deduplicating model registry alias '%s' -> '%s'",
                            name,
                            key,
                        )
                        continue
                    deduped[key] = model_cls

                self.model_registry = deduped

                if not self.model_registry:
                    self._load_default_models()

            self._discover_available_models()

        except Exception as e:
            logger.warning(f"Error loading model registry: {e}")
            self._load_default_models()

    def _load_models_from_config(self, config_models: Dict[str, Any]):
        """Load models from configuration."""
        for model_name, model_config in config_models.items():
            try:
                if model_config.get("enabled", True):
                    model_class = self._get_model_class(model_config.get("class_path"))
                    if model_class:
                        self.model_registry[model_name] = model_class
                        logger.info(f"Loaded model {model_name} from config")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name} from config: {e}")

    def _load_default_models(self):
        """Load default models as fallback."""
        default_models = {
            "arima": ARIMAModel,
            "lstm": LSTMModel,
            "xgboost": XGBoostModel,
            "xgb": XGBoostModel,  # Alias
        }

        # Add Autoformer if available
        if AUTOFORMER_AVAILABLE and AutoformerModel is not None:
            default_models["autoformer"] = AutoformerModel

        # Add Transformer if available
        if TRANSFORMER_AVAILABLE:
            default_models["transformer"] = TransformerForecaster

        for model_name, model_class in default_models.items():
            try:
                self.model_registry[model_name] = model_class
                logger.info(f"Loaded default model {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load default model {model_name}: {e}")

    def _discover_available_models(self):
        """Discover additional models dynamically using trading.models paths."""
        try:
            models_dir = Path(__file__).parent
            for model_file in models_dir.glob("*_model.py"):
                model_name = model_file.stem.replace("_model", "")
                if model_name not in self.model_registry and model_name in _DISCOVERY_CLASS_MAP:
                    module_path, class_name = _DISCOVERY_CLASS_MAP[model_name]
                    try:
                        model_class = self._get_model_class(f"{module_path}.{class_name}")
                        if model_class:
                            self.model_registry[model_name] = model_class
                            logger.info(f"Discovered model {model_name}")
                    except Exception as e:
                        logger.debug(f"Failed to discover model {model_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering models: {e}")

    def get_available_models(self) -> list:
        """Return list of available model names (display-ready, e.g. LSTM, XGBoost)."""
        name_map = {
            "lstm": "LSTM",
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "prophet": "Prophet",
            "arima": "ARIMA",
            "transformer": "Transformer",
            "ensemble": "Ensemble",
            "tcn": "TCN",
            "catboost": "CatBoost",
            "ridge": "Ridge",
            "garch": "GARCH",
            "autoformer": "Autoformer",
        }
        display_names = {name_map.get(k, k.capitalize()) for k in self.model_registry.keys()}
        return sorted(display_names)

    def _get_model_class(self, class_path: str):
        """Get model class from class path string. Normalizes models.* to trading.models.*."""
        try:
            if not class_path:
                return None
            # Use trading.models.* so we don't trigger top-level models/__init__ (which imports missing forecast_router)
            if class_path.startswith("models.") and not class_path.startswith("trading.models."):
                class_path = "trading.models." + class_path[7:]
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
            class_name = _CLASS_NAME_ALIASES.get(class_name, class_name)
            return getattr(module, class_name, None)
        except Exception as e:
            logger.warning(f"Failed to load model class {class_path}: {e}")
            return None

    def register_model(
        self, name: str, model_class, config: Optional[Dict[str, Any]] = None
    ):
        """Register a new model dynamically.

        Args:
            name: Model name
            model_class: Model class
            config: Optional model configuration
        """
        try:
            self.model_registry[name] = model_class
            if config:
                # Store model-specific config
                if not hasattr(self, "model_configs"):
                    self.model_configs = {}
                self.model_configs[name] = config
            logger.info(f"Registered model {name}")
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")

    def unregister_model(self, name: str):
        """Unregister a model.

        Args:
            name: Model name to unregister
        """
        try:
            if name in self.model_registry:
                del self.model_registry[name]
                if hasattr(self, "model_configs") and name in self.model_configs:
                    del self.model_configs[name]
                logger.info(f"Unregistered model {name}")
        except Exception as e:
            logger.error(f"Failed to unregister model {name}: {e}")

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize model selection weights.

        Returns:
            Dictionary of model weights
        """
        return {model: 1.0 for model in self.model_registry.keys()}

    def _analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series data characteristics.

        Args:
            data: Time series data to analyze

        Returns:
            Dictionary of data characteristics
        """
        characteristics = {
            "length": len(data),
            "has_seasonality": self._check_seasonality(data),
            "has_trend": self._check_trend(data),
            "volatility": data.std().mean(),
            "missing_values": data.isnull().sum().sum(),
        }
        return characteristics

    def _check_seasonality(self, data: pd.DataFrame) -> bool:
        """Check for seasonality in time series.

        Args:
            data: Time series data

        Returns:
            True if seasonality is detected
        """
        # TODO: implement seasonality detection
        # Currently defaults to False so auto selection uses XGBoost/Ridge path
        return False

    def _check_trend(self, data: pd.DataFrame) -> bool:
        """Check for trend in time series.

        Args:
            data: Time series data

        Returns:
            True if trend is detected
        """
        # TODO: implement trend detection
        # Currently defaults to False so auto selection uses XGBoost/Ridge path
        return False

    def _prepare_data_safely(self, data: pd.DataFrame, normalize_close: bool = False) -> pd.DataFrame:
        """Prepare data: add forecast features. Do not normalize close for non-neural models (they predict raw prices)."""
        if data is None or data.empty:
            self._last_price_used = 1.0
            return pd.DataFrame()
        prepared, last_price = prepare_forecast_data(data, normalize_close=normalize_close)
        self._last_price_used = last_price
        logger.info("Forecast data prepared: last_price=%.2f, normalize_close=%s", self._last_price_used, normalize_close)
        return prepared

    def _select_model_with_fallback(
        self, data: pd.DataFrame, model_type: Optional[str] = None
    ) -> str:
        """Select model with fallback; never return a key not in registry."""
        selected = self._select_model(data, model_type)
        if selected in self.model_registry:
            return selected
        fallback = self._get_fallback_model(selected)
        return fallback if fallback in self.model_registry else list(self.model_registry.keys())[0]

    def _get_fallback_model(self, model_type: str) -> str:
        """Return fallback model name when given model fails."""
        fallbacks = {"transformer": "lstm", "lstm": "arima", "xgboost": "arima", "prophet": "arima"}
        return fallbacks.get(model_type, "arima" if "arima" in self.model_registry else list(self.model_registry.keys())[0])

    def _get_fallback_result(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Return a minimal fallback forecast result."""
        last = 0.0
        if data is not None and not data.empty:
            numeric = data.select_dtypes(include=[np.number])
            if not numeric.empty and len(numeric) > 0:
                last_val = numeric.iloc[-1].mean()
                last = float(last_val) if pd.notna(last_val) else 0.0
            else:
                last = 0.0
        return {
            "model": "fallback",
            "forecast": np.full(horizon, last),
            "confidence": 0.0,
            "metadata": {},
            "warnings": ["Fallback forecast used"],
            "validation_mape": None,
            "in_sample_mape": None,
            "last_actual_price": last,
            "confidence_label": "Unknown",
        }

    def _get_model_defaults(self, model_name: str) -> Dict[str, Any]:
        """Return default config kwargs for a model (for instantiation)."""
        defaults = {
            "arima": {"order": (5, 1, 0), "use_auto_arima": True, "target_column": "close"},
            "lstm": {"target_column": "close", "sequence_length": 60, "hidden_dim": 64, "num_layers": 2, "dropout": 0.2, "epochs": 50, "batch_size": 32, "learning_rate": 0.001},
            "xgboost": {"target_column": "close", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            "prophet": {"date_column": "ds", "target_column": "close", "prophet_params": {}},
            "ridge": {"target_column": "close", "alpha": 1.0, "max_iter": 1000},
            "garch": {"p": 1, "q": 1, "target_column": "close"},
            "ensemble": {"models": [], "voting_method": "mse", "weight_window": 20},
            "hybrid": {
                "models": [
                    {"name": "ARIMA", "target_column": "close"},
                    {"name": "Ridge", "target_column": "close"},
                    {"name": "XGBoost", "target_column": "close"},
                ],
            },
            "tcn": {"target_column": "close", "feature_columns": ["close", "volume"], "sequence_length": 20},
            "catboost": {"target_column": "close", "iterations": 500, "depth": 6},
        }
        return defaults.get(model_name, {"target_column": "close"}).copy()

    def _generate_simple_forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate a simple constant/last-value forecast."""
        if data is None or data.empty or horizon <= 0:
            return np.array([])
        # Use an explicit close/price column when possible (avoid volume/features blowing up the fallback).
        for col in ("Close", "close", "price", "Price"):
            if col in data.columns:
                try:
                    s = pd.to_numeric(data[col], errors="coerce").dropna()
                    last = float(s.iloc[-1]) if len(s) > 0 and not pd.isna(s.iloc[-1]) else None
                    if last is not None and np.isfinite(last):
                        return np.full(horizon, last, dtype="float64")
                except Exception as _e:
                    logger.warning(
                        "ForecastRouter: simple forecast parse failed for column '%s': %s",
                        col,
                        _e,
                    )

        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return np.zeros(horizon, dtype="float64")
        last = float(numeric.iloc[-1].dropna().iloc[-1]) if not numeric.iloc[-1].dropna().empty else 0.0
        return np.full(horizon, last, dtype="float64")

    def _get_confidence(self, model: Any, model_name: str) -> float:
        """Return confidence score for the model."""
        try:
            if hasattr(model, "get_confidence"):
                c = model.get_confidence()
                return float(c.get("confidence", 0.5)) if isinstance(c, dict) else float(c)
        except Exception as _e:
            logger.warning(
                "ForecastRouter: get_confidence failed for model '%s': %s",
                model_name,
                _e,
            )
        return 0.5

    def _get_metadata(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Return metadata dict for the model."""
        try:
            if hasattr(model, "get_model_info"):
                return model.get_model_info() or {}
        except Exception as _e:
            logger.warning(
                "ForecastRouter: get_model_info failed for model '%s': %s",
                model_name,
                _e,
            )
        return {"model": model_name}

    def _get_warnings(self, data: pd.DataFrame, model_name: str) -> list:
        """Return list of warnings for the forecast."""
        return []

    @staticmethod
    def _compute_mape(actual: np.ndarray, pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error; return 0 if undefined."""
        actual = np.asarray(actual, dtype="float64").ravel()
        pred = np.asarray(pred, dtype="float64").ravel()
        n = min(len(actual), len(pred))
        if n == 0:
            return 0.0
        actual, pred = actual[:n], pred[:n]
        mask = np.isfinite(actual) & (np.abs(actual) > 1e-12)
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)

    def _run_walk_forward_validation(
        self,
        prepared_data: pd.DataFrame,
        selected_model: str,
        horizon: int,
    ) -> float:
        """Train on first 200 rows, predict next 50, return validation MAPE (%). Uses 80/20 if len < 250."""
        if prepared_data is None or len(prepared_data) < 60:
            return float("nan")
        n = len(prepared_data)
        if n >= 250:
            train_size, val_size = 200, 50
        else:
            train_size = int(0.8 * n)
            val_size = min(50, n - train_size)
        if val_size < 5:
            return float("nan")
        train_df = prepared_data.iloc[:train_size]
        val_df = prepared_data.iloc[train_size: train_size + val_size]
        close_col = "close" if "close" in prepared_data.columns else prepared_data.columns[0]
        try:
            model_class = self.model_registry.get(selected_model)
            if not model_class:
                return float("nan")
            config = self._get_model_defaults(selected_model)
            if selected_model == "lstm":
                config["input_dim"] = train_df.shape[1] if hasattr(train_df, "shape") else 1
                config["output_dim"] = 1
            model = model_class(config)
            if selected_model == "lstm":
                model.train_model(
                    train_df, train_df[close_col],
                    epochs=config.get("epochs", 30), batch_size=32
                )
            else:
                model.fit(train_df)
            if hasattr(model, "forecast"):
                fc = model.forecast(train_df, horizon=val_size)
            else:
                fc = model.predict(val_df) if hasattr(model, "predict") else None
            if fc is None:
                return float("nan")
            pred = np.asarray(fc, dtype="float64").ravel()[:val_size]
            actual = val_df[close_col].values[: len(pred)]
            return self._compute_mape(actual, pred)
        except Exception as e:
            logger.debug("Walk-forward validation failed: %s", e)
            return float("nan")

    def _in_sample_mape(self, model: Any, prepared_data: pd.DataFrame, selected_model: str) -> Optional[float]:
        """Compute in-sample MAPE (%) if model supports predict on training data."""
        if prepared_data is None or len(prepared_data) < 10:
            return None
        # Skip MAPE check for models whose predict() operates in normalized space
        _skip_mape_models = {'ridge', 'catboost', 'tcn', 'hybrid'}
        if selected_model in _skip_mape_models:
            return None
        close_col = "close" if "close" in prepared_data.columns else prepared_data.columns[0]
        try:
            if hasattr(model, "predict"):
                pred = model.predict(prepared_data)
            else:
                return None
            if pred is None:
                return None
            pred = np.asarray(pred, dtype="float64").ravel()
            actual = prepared_data[close_col].values
            n = min(len(actual), len(pred))
            if n < 5:
                return None
            return self._compute_mape(actual[:n], pred[:n])
        except Exception:
            return None

    def _log_performance(self, model_name: str, forecast: Any, data: pd.DataFrame) -> None:
        """Log performance metrics."""
        logger.debug(f"Model {model_name} forecast length {getattr(forecast, '__len__', lambda: 0)()}")

    def _select_model(
        self, data: pd.DataFrame, model_type: Optional[str] = None
    ) -> str:
        """Select appropriate model based on data and preferences.

        Args:
            data: Time series data
            model_type: Optional preferred model type

        Returns:
            Selected model type
        """
        self._last_stationarity_meta = {"is_stationary": True, "note": None}
        try:
            from trading.analysis.econometric_diagnostics import (
                run_stationarity_tests,
            )

            _sr = run_stationarity_tests(data)
            self._last_stationarity_meta["is_stationary"] = bool(
                _sr.get("is_stationary", True)
            )
            if not self._last_stationarity_meta["is_stationary"]:
                self._last_stationarity_meta["note"] = (
                    "Price level non-stationary (ADF); ARIMA preferred; "
                    "Ridge assumes stationary structure — interpret with care."
                )
        except Exception as _se:
            logger.debug("Stationarity probe skipped: %s", _se)

        # If the user explicitly asked for a model, honor it (no heuristic override).
        if model_type is not None:
            mt = str(model_type).lower()
            # Common aliases
            if mt in ("xgb",):
                mt = "xgboost"
            if mt in self.model_registry:
                return mt
            if mt != "auto":
                logger.warning("Requested model '%s' not in registry; falling back.", model_type)

        _auto = model_type is None or str(model_type).lower() in (
            "auto",
            "",
        )
        if (
            _auto
            and not self._last_stationarity_meta.get("is_stationary", True)
            and "arima" in self.model_registry
        ):
            return "arima"

        # Auto-selection: pick a reasonable default when model_type is None/"auto"/invalid.
        if PROPHET_AVAILABLE and "prophet" in self.model_registry:
            default_model = "prophet"
        else:
            default_model = "arima" if "arima" in self.model_registry else list(self.model_registry.keys())[0]

        try:
            from trading.strategies.adaptive_selector import get_adaptive_selector

            _col_map = {c.lower(): c for c in data.columns}
            _cc = _col_map.get("close", data.columns[0])
            _price = pd.to_numeric(data[_cc], errors="coerce").dropna()
            if len(_price) >= 40:
                _sel = get_adaptive_selector()
                _mc = _sel.analyze_market_conditions(_price)
                _avail_disp = []
                _disp_to_key = {
                    "LSTM": "lstm",
                    "XGBoost": "xgboost",
                    "ARIMA": "arima",
                    "Prophet": "prophet",
                    "Transformer": "transformer",
                }
                for disp, key in _disp_to_key.items():
                    if key in self.model_registry:
                        _avail_disp.append(disp)
                if _avail_disp:
                    _picked = _sel.model_selector.select_optimal_model(
                        _mc, _avail_disp
                    )
                    _key = _disp_to_key.get(_picked)
                    if _key is None:
                        _key = str(_picked).lower()
                    if _key in self.model_registry:
                        return _key
        except Exception as _ae:
            logger.debug("AdaptiveSelector override skipped: %s", _ae)

        # Heuristic selection (only in auto mode)
        try:
            characteristics = self._analyze_data(data)
            if characteristics.get("length", 0) < 100 and "arima" in self.model_registry:
                return "arima"
            if characteristics.get("has_seasonality") and "prophet" in self.model_registry:
                return "prophet"
            # Prefer tree models for longer, non-seasonal series when available
            if "xgboost" in self.model_registry:
                return "xgboost"
            if "ridge" in self.model_registry:
                if self._last_stationarity_meta.get("is_stationary", True):
                    return "ridge"
        except Exception as _e:
            logger.warning(
                "ForecastRouter: data analysis failed during auto-selection: %s",
                _e,
            )

        return default_model

    def get_forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        model_type: Optional[str] = None,
        run_walk_forward: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get forecast for time series data.

        Args:
            data: Time series data
            horizon: Forecast horizon in periods
            model_type: Optional preferred model type
            run_walk_forward: If True, run walk-forward validation for MAPE (slower). Use False for Quick Forecast.
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing forecast results

        Raises:
            ValueError: If data is invalid or model type is not supported
        """
        try:
            # Accept common alias used throughout the app/scripts
            # (keeps backwards compatibility with callers passing model_name=...)
            if model_type is None:
                model_type = kwargs.get("model_name") or kwargs.get("model")

            if data is None or data.empty:
                _sym = kwargs.get("symbol") or kwargs.get("ticker") or ""
                logger.error(
                    "Cannot forecast: price history unavailable for symbol %s",
                    _sym or "(unknown)",
                )
                return {
                    "error": "Insufficient price data",
                    "symbol": _sym,
                    "forecasts": [],
                    "forecast": np.array([], dtype="float64"),
                    "model": None,
                    "confidence": 0.0,
                    "metadata": {},
                    "warnings": ["Insufficient price history for forecast"],
                    "validation_mape": None,
                    "in_sample_mape": None,
                    "last_actual_price": None,
                    "confidence_label": "Unknown",
                }

            if horizon is None or horizon <= 0:
                logger.warning(
                    f"Invalid horizon {horizon}, using default horizon of 30"
                )
                horizon = 30

            # Validate data structure
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Data must be pandas DataFrame, got {type(data)}")
                raise ValueError("Data must be pandas DataFrame")

            # Ensure we have at least one numeric price column (yfinance uses "Close")
            price_cols = [c for c in ("close", "Close", "price", "Price") if c in data.columns]
            if not price_cols:
                numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
                if not numeric_columns:
                    raise ValueError("No numeric columns found in data")
                logger.warning("No explicit price column found; using %s as price proxy", numeric_columns[0])

            # Prepare data with defensive checks
            prepared_data = self._prepare_data_safely(data)

            # Select model with fallback logic
            selected_model = self._select_model_with_fallback(prepared_data, model_type)
            logger.info(f"Selected model: {selected_model}")

            # Ensure config is defined for all code paths that use it (fit_args, logger)
            config = self._get_model_defaults(selected_model)

            # Initialize / retrieve a cached trained model instance based on (symbol, date_range, model)
            model = self._get_cached_trained_model(
                data=prepared_data,
                selected_model=selected_model,
                run_walk_forward=run_walk_forward,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("run_walk_forward", "model_name", "model")
                },
            )
            fit_args = (
                {
                    "epochs": config.get("epochs", 50),
                    "batch_size": config.get("batch_size", 32),
                    "learning_rate": config.get("learning_rate", 0.001),
                }
                if selected_model == "lstm"
                else {}
            )

            logger.info(f"Initializing {selected_model} with config: {config}")

            # Walk-forward validation (optional; skip for Quick Forecast to keep it fast)
            validation_mape = float("nan")
            if run_walk_forward and len(prepared_data) >= 60:
                validation_mape = self._run_walk_forward_validation(
                    prepared_data, selected_model, horizon
                )

            # At this point, model is already trained (from cache or freshly fitted)

            # Generate forecast with error handling
            try:
                # Prefer explicit forecast/forecast_with_uncertainty APIs, fall back to predict(data)
                if hasattr(model, "forecast_with_uncertainty"):
                    result = model.forecast_with_uncertainty(
                        prepared_data, horizon=horizon
                    )
                elif hasattr(model, "forecast"):
                    # Common signature: forecast(data, horizon=...)
                    # Some decorator/caching wrappers are picky about kwargs; try kwargs then positional.
                    try:
                        result = model.forecast(prepared_data, horizon=horizon)
                    except TypeError:
                        result = model.forecast(prepared_data, horizon)
                elif hasattr(model, "predict"):
                    # Some models expose only predict(data) → array
                    preds = model.predict(prepared_data)
                    preds = np.asarray(preds)
                    if preds.size >= horizon:
                        preds = preds[-horizon:]
                    result = {"forecast": preds}
                else:
                    result = {"forecast": self._generate_simple_forecast(prepared_data, horizon)}

                # Normalize to dict with "forecast" key
                if isinstance(result, dict):
                    forecast_values = result.get("forecast")
                else:
                    forecast_values = result

                # Ensure we have a 1D numpy array of floats; denormalize from normalized price space
                if forecast_values is None:
                    raise ValueError("Model returned no forecast values")
                forecast_array = np.asarray(forecast_values, dtype="float64").ravel()
                if forecast_array.size == 0:
                    raise ValueError("Model returned empty forecast array")
                last_price = getattr(self, "_last_price_used", 1.0)
                # Denormalize exactly once unless model explicitly says it already returned raw prices
                already_denormalized = False
                if isinstance(result, dict):
                    already_denormalized = bool(
                        result.get("denormalized", False) or result.get("already_denormalized", False)
                    )
                if (not already_denormalized) and last_price and last_price != 1.0:
                    forecast_array = forecast_array * last_price

                # FINAL PRICE SPACE GUARD — non-negotiable
                # Enforce that outputs are in price space and roughly anchored to last close.
                try:
                    last_close = None
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        _col_map = {c.lower(): c for c in data.columns}
                        _num_cols = list(
                            data.select_dtypes(include=[np.number]).columns
                        )
                        _tgt_col = _col_map.get(
                            "close",
                            _num_cols[0] if _num_cols else None,
                        )
                        if _tgt_col is not None and _tgt_col in data.columns:
                            last_close = float(data[_tgt_col].iloc[-1])
                        elif "price" in data.columns:
                            last_close = float(data["price"].iloc[-1])
                    if last_close is not None and np.isfinite(last_close) and last_close > 10:
                        fa = np.asarray(forecast_array, dtype="float64").ravel()
                        if fa.size:
                            # If model produced ratios/returns, convert to prices
                            if np.nanmax(fa) < 10:
                                prices = [last_close]
                                for v in fa:
                                    r = float(v)
                                    next_p = (
                                        prices[-1] * r
                                        if 0.5 < r < 2.0
                                        else prices[-1] * (1.0 + r)
                                    )
                                    prices.append(float(next_p))
                                forecast_array = np.asarray(prices[1:], dtype="float64")
                            # Explosion guard
                            elif np.nanmax(fa) > last_close * 10:
                                forecast_array = np.full(len(fa), last_close, dtype="float64")

                            # Anchor first step within 10% of last close (shift series)
                            fa2 = np.asarray(forecast_array, dtype="float64").ravel()
                            if fa2.size:
                                gap_pct = abs(float(fa2[0]) - last_close) / (last_close or 1e-8)
                                if gap_pct > 0.10:
                                    offset = last_close - float(fa2[0])
                                    forecast_array = fa2 + offset
                        # Extra hard clamp: if anything is still wildly off, flatten to last_close.
                        fa3 = np.asarray(forecast_array, dtype="float64").ravel()
                        if fa3.size:
                            if (np.nanmax(fa3) > last_close * 5.0) or (np.nanmin(fa3) < last_close * 0.2):
                                forecast_array = np.full(len(fa3), last_close, dtype="float64")
                except Exception as _e:
                    logger.warning(
                        "ForecastRouter: final price space guard failed: %s",
                        _e,
                    )

                logger.info(f"Successfully generated forecast with {selected_model}")
            except Exception as e:
                logger.error(f"Failed to generate forecast with {selected_model}: {e}")
                if _resilience_handle_failure is not None:
                    try:
                        _resilience_handle_failure(
                            selected_model,
                            e,
                            {
                                "stage": "forecast",
                                "symbol": kwargs.get("symbol") or kwargs.get("ticker"),
                            },
                        )
                    except Exception:
                        pass
                # Return simple forecast as fallback (in normalized space; denorm once)
                forecast_array = self._generate_simple_forecast(prepared_data, horizon)
                lp = getattr(self, "_last_price_used", 1.0)
                if lp and lp != 1.0:
                    forecast_array = np.asarray(forecast_array, dtype="float64") * lp
                result = {"forecast": forecast_array}

            # In-sample MAPE and poor-fit warning (suppress extreme/unstable values for UI)
            warnings_list = list(self._get_warnings(data, selected_model))
            in_sample_mape = self._in_sample_mape(model, prepared_data, selected_model)
            _skip_mape_models = {'ridge', 'catboost', 'tcn', 'hybrid'}
            if in_sample_mape is not None and np.isfinite(in_sample_mape) and selected_model not in _skip_mape_models:
                if in_sample_mape > 50.0:
                    # Very high MAPE is often an artifact of normalization mismatch; log only.
                    logger.warning(
                        "Model %s has very high in-sample MAPE (%.1f%%); treating as diagnostic only.",
                        selected_model,
                        in_sample_mape,
                    )
                elif in_sample_mape > 15.0:
                    msg = (
                        f"Model {selected_model} has poor in-sample fit (MAPE={in_sample_mape:.1f}%). "
                        "Consider more training data or different hyperparameters."
                    )
                    logger.debug(
                        "Model %s MAPE=%.1f%%",
                        selected_model,
                        in_sample_mape,
                    )
                    warnings_list.append(msg)

            last_actual_price = getattr(self, "_last_price_used", 1.0)
            # Confidence label from validation or in-sample MAPE (Low = MAPE < 5%, Medium = 5-10%, High = > 10%)
            vmape = validation_mape if np.isfinite(validation_mape) else None
            if vmape is None and in_sample_mape is not None and np.isfinite(in_sample_mape):
                vmape = in_sample_mape
            if vmape is not None:
                if vmape < 5.0:
                    confidence_label = "Low"
                elif vmape < 10.0:
                    confidence_label = "Medium"
                else:
                    confidence_label = "High"
            else:
                confidence_label = "Unknown"

            # Log performance
            self._log_performance(selected_model, forecast_array, data)

            _meta = self._get_metadata(model, selected_model)
            try:
                _sn = getattr(self, "_last_stationarity_meta", None) or {}
                if isinstance(_meta, dict) and _sn.get("note"):
                    _meta = {**_meta, "stationarity_note": _sn.get("note")}
            except Exception as _me:
                logger.debug("metadata stationarity merge skipped: %s", _me)

            return {
                "model": selected_model,
                "forecast": forecast_array,
                "confidence": self._get_confidence(model, selected_model),
                "metadata": _meta,
                "warnings": warnings_list,
                "validation_mape": validation_mape if np.isfinite(validation_mape) else None,
                "in_sample_mape": in_sample_mape,
                "last_actual_price": last_actual_price,
                "confidence_label": confidence_label,
            }

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Forecast error: {str(e)}")
            # Return fallback result instead of raising
            fallback_result = self._get_fallback_result(data, horizon)
            return fallback_result
        except Exception as e:
            logger.error(f"Unexpected forecast error: {str(e)}")
            # Return fallback result for unexpected errors
            fallback_result = self._get_fallback_result(data, horizon)
            return fallback_result

    def select_best_model(
        self,
        data: pd.DataFrame,
        horizon: int,
        symbol: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train each available model on first 80% of data, evaluate on last 20%.
        Return the model name with lowest validation MAPE and a table of all scores.
        Cache result in MemoryStore with key "best_model_{symbol}" when symbol is provided.
        """
        if data is None or data.empty or len(data) < 60:
            return "arima", {"scores": [], "best": "arima", "reason": "Insufficient data"}
        prepared_data = self._prepare_data_safely(data)
        n = len(prepared_data)
        train_size = int(0.8 * n)
        val_size = n - train_size
        if val_size < 5:
            return "arima", {"scores": [], "best": "arima", "reason": "Validation split too small"}
        train_df = prepared_data.iloc[:train_size]
        val_df = prepared_data.iloc[train_size:]
        close_col = "close" if "close" in prepared_data.columns else prepared_data.columns[0]
        actual_val = val_df[close_col].values

        scores: List[Dict[str, Any]] = []
        best_name = "arima"
        best_mape = float("inf")

        for model_name in list(self.model_registry.keys()):
            if model_name == "base":
                continue
            try:
                model_class = self.model_registry[model_name]
                config = self._get_model_defaults(model_name)
                if model_name == "lstm":
                    config["input_dim"] = train_df.shape[1] if hasattr(train_df, "shape") else 1
                    config["output_dim"] = 1
                model = model_class(config)
                if model_name == "lstm":
                    model.train_model(
                        train_df, train_df[close_col],
                        epochs=config.get("epochs", 30), batch_size=32
                    )
                else:
                    model.fit(train_df)
                if hasattr(model, "forecast"):
                    fc = model.forecast(train_df, horizon=val_size)
                else:
                    fc = model.predict(val_df) if hasattr(model, "predict") else None
                if fc is None:
                    scores.append({"model": model_name, "validation_mape": None, "error": "No forecast"})
                    continue
                # Normalize dict outputs from models (TCN/Ridge/CatBoost/etc.)
                if isinstance(fc, dict):
                    raw_vals = fc.get("predictions")
                    if raw_vals is None or (hasattr(raw_vals, "__len__") and len(raw_vals) == 0):
                        raw_vals = fc.get("forecast")
                    if raw_vals is None or (hasattr(raw_vals, "__len__") and len(raw_vals) == 0):
                        raw_vals = fc.get("values")
                    if raw_vals is None or (hasattr(raw_vals, "__len__") and len(raw_vals) == 0):
                        raw_vals = fc.get("forecast_values")
                else:
                    raw_vals = fc
                pred = np.asarray(raw_vals, dtype="float64").ravel()[:val_size]
                mape_val = self._compute_mape(actual_val[: len(pred)], pred)
                scores.append({"model": model_name, "validation_mape": round(mape_val, 2)})
                if np.isfinite(mape_val) and mape_val < best_mape:
                    best_mape = mape_val
                    best_name = model_name
            except Exception as e:
                logger.debug("select_best_model %s failed: %s", model_name, e)
                scores.append({"model": model_name, "validation_mape": None, "error": str(e)})

        result = {"scores": scores, "best": best_name, "validation_mape": best_mape if np.isfinite(best_mape) else None}
        if symbol:
            try:
                from trading.memory import get_memory_store
                get_memory_store().upsert_preference(f"best_model_{symbol}", {"model": best_name, "validation_mape": result["validation_mape"], "scores": scores})
            except Exception as e:
                logger.debug("Could not cache best model: %s", e)
        return best_name, result

    def get_consensus_forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 7,
        models: Optional[List[str]] = None,
        symbol: Optional[str] = None,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple models and build a consensus forecast.

        Returns a dict containing:
        - consensus_forecast: mean path across models
        - upper_bound / lower_bound: mean ± std across models (model agreement
          band; NOT a calibrated prediction interval)
        - band_type: "model_disagreement"
        - consensus_price: day-horizon consensus price
        - price_targets: per-model day-horizon prices
        - direction: BULLISH / BEARISH / NEUTRAL
        - conviction: HIGH / MEDIUM / LOW / INSUFFICIENT
        - models_used / models_failed
        - last_price: last observed close
        """
        if data is None or data.empty:
            return {"error": "No data provided", "models_failed": []}

        # Clean and normalize input data for all models
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception as _e:
                logger.warning(
                    "forecast_router: index datetime coercion failed: %s",
                    _e,
                )
        # Drop NaT rows produced by coercion
        nat_mask = df.index.isna()
        if nat_mask.any():
            logger.warning(
                "forecast_router: %d NaT values in "
                "index after coercion — dropping",
                nat_mask.sum()
            )
            df = df[~nat_mask]
        if len(df) == 0:
            logger.error(
                "forecast_router: all rows dropped "
                "after NaT filter — cannot forecast"
            )
            return {
                "error": "All data dropped after datetime coercion",
                "consensus_forecast": [],
                "models_failed": [],
                "models_used": [],
                "direction": "NEUTRAL",
                "conviction": "INSUFFICIENT",
                "last_price": None,
                "consensus_price": None,
                "consensus_7d_change_pct": None,
            }
        df = df.sort_index()
        if isinstance(df.index, pd.DatetimeIndex) and getattr(df.index, "tz", None) is not None:
            try:
                df.index = df.index.tz_convert(None)
            except Exception as _e:
                # Best effort; if tz conversion fails, continue with original index
                logger.warning(
                    "forecast_router: tz_convert failed; continuing with original index: %s",
                    _e,
                )

        # Ensure we have a Close column
        if "Close" not in df.columns and "close" in df.columns:
            df["Close"] = df["close"]
        if "Close" not in df.columns:
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                return {"error": "No numeric price column found", "models_failed": []}
            df["Close"] = numeric.iloc[:, 0]

        df = df.dropna(subset=["Close"])
        if df.empty:
            return {"error": "No price data after cleaning", "models_failed": []}

        last_price_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if last_price_series.empty:
            return {"error": "No valid closing prices", "models_failed": []}
        last_price = float(last_price_series.iloc[-1])

        try:
            horizon_int = int(horizon)
        except Exception:
            horizon_int = 7
        if horizon_int <= 0:
            horizon_int = 7

        if models is None:
            models = [
                "arima",
                "xgboost",
                "ridge",
                "catboost",
                "prophet",
                "lstm",
                "tcn",
                "transformer",
                "garch",
                "ensemble",
            ]

        _SLOW_MODELS = frozenset(
            {"lstm", "tcn", "transformer", "garch", "ensemble"}
        )

        def _consensus_timeout_s(model_name: str) -> float:
            n = str(model_name).lower()
            if n == "ensemble":
                return 180.0
            if n in _SLOW_MODELS:
                return 90.0
            return 120.0

        all_forecasts: List[np.ndarray] = []
        used: List[str] = []
        failed: List[str] = []
        price_targets: Dict[str, float] = {}

        for name in models:
            try:

                def _run_one_forecast():
                    _per = dict((model_configs or {}).get(name) or {})
                    return self.get_forecast(
                        df,
                        model_type=name,
                        horizon=horizon_int,
                        run_walk_forward=False,
                        symbol=symbol,
                        **_per,
                    )

                _to = _consensus_timeout_s(name)
                with ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(_run_one_forecast)
                    try:
                        result = _fut.result(timeout=_to)
                    except FuturesTimeout:
                        failed.append(f"{name}(timeout {_to:.0f}s)")
                        continue
                # Exclude models with unreliable fit (wild MAPE) from consensus
                MAX_RELIABLE_MAPE = 100.0
                _vmape = (result or {}).get("validation_mape")
                _ismape = (result or {}).get("in_sample_mape")
                _mape_vals = []
                for _mv in (_vmape, _ismape):
                    if _mv is not None and np.isfinite(float(_mv)):
                        _mape_vals.append(float(_mv))
                if _mape_vals:
                    _worst_mape = max(_mape_vals)
                    if _worst_mape > MAX_RELIABLE_MAPE:
                        logger.warning(
                            "Excluding %s from consensus: MAPE=%.1f%% exceeds "
                            "reliability threshold (%.0f%%)",
                            name,
                            _worst_mape,
                            MAX_RELIABLE_MAPE,
                        )
                        failed.append(f"{name}(MAPE {_worst_mape:.1f}%)")
                        continue
                fc_result = (result or {}).get("forecast", [])
                fc = np.asarray(fc_result, dtype="float64").ravel()
                if fc.size < horizon_int:
                    raise ValueError("forecast shorter than horizon")
                # Exclude pathological level jumps vs last close (continuity sanity)
                if last_price > 0 and fc.size > 0:
                    _cont = abs(float(fc[0]) - last_price) / last_price
                    if _cont > 5.0:
                        logger.warning(
                            "Excluding %s from consensus: first-step gap vs "
                            "last price = %.1f%% (>500%% threshold)",
                            name,
                            _cont * 100.0,
                        )
                        failed.append(f"{name}(continuity gap {_cont*100:.0f}%)")
                        continue

                # Sanity: require some finite values within a broad multiple of last price
                finite = fc[np.isfinite(fc)]
                if finite.size == 0:
                    raise ValueError("no finite forecast values")
                lo, hi = float(finite.min()), float(finite.max())
                if last_price > 0:
                    if hi < last_price * 0.3 or lo > last_price * 3.0:
                        raise ValueError("forecast outside sanity band")

                # Cap forecast to realistic range: max ±15% from last price
                try:
                    _fc_arr = np.asarray(fc_result, dtype="float64").ravel()
                    if last_price is not None and last_price > 0 and _fc_arr.size > 0:
                        _max_move = 0.15
                        _upper_cap = last_price * (1 + _max_move)
                        _lower_cap = last_price * (1 - _max_move)
                        _clipped = np.clip(_fc_arr, _lower_cap, _upper_cap)
                        _orig_end = float(_fc_arr[-1])
                        _new_end = float(_clipped[-1])
                        if abs(_orig_end - _new_end) > 0.01:
                            logger.warning(
                                "Model %s forecast capped: original end %.2f → %.2f (±15%% from %.2f)",
                                name,
                                _orig_end,
                                _new_end,
                                last_price,
                            )
                        fc = _clipped
                    # If capping failed, fc remains as previously validated
                except Exception as _e:
                    logger.warning(
                        "ForecastRouter: forecast capping failed for model '%s' (continuing): %s",
                        name,
                        _e,
                    )

                target_idx = horizon_int - 1
                price_targets[name] = float(fc[target_idx])
                all_forecasts.append(fc[:horizon_int])
                used.append(name)
            except Exception as e:  # pragma: no cover - defensive guard
                failed.append(f"{name}({e})")

        # Debug: per-model std/mean/size (page can store in session_state)
        forecast_debug: Dict[str, Dict[str, float]] = {}
        for name, fc in zip(used, all_forecasts):
            arr = np.asarray(fc)
            forecast_debug[name] = {
                "std": float(arr.std()) if arr.size > 0 else 0.0,
                "mean": float(arr.mean()) if arr.size > 0 else 0.0,
                "size": int(arr.size),
            }

        # Only include non-flat forecasts in consensus (std > 0.001)
        valid_forecasts: List[np.ndarray] = []
        valid_used: List[str] = []
        for fc, name in zip(all_forecasts, used):
            arr = np.asarray(fc)
            if arr.size > 0 and float(arr.std()) > 0.001:
                valid_forecasts.append(arr)
                valid_used.append(name)

        if not valid_forecasts:
            # Fallback: try a simple ARIMA model instead of returning an empty forecast
            try:
                from trading.models.arima_model import ARIMAModel

                fb = ARIMAModel({"target_column": "close"})
                fb.fit(df)
                fb_result = fb.forecast(df, horizon=horizon_int)
                fb_fc = np.asarray(
                    (fb_result or {}).get("forecast", []), dtype="float64"
                ).ravel()
                if fb_fc.size >= horizon_int:
                    return {
                        "consensus_forecast": fb_fc[:horizon_int].tolist(),
                        "upper_bound": None,
                        "lower_bound": None,
                        "band_type": "model_disagreement",
                        "model_agreement": 0.0,
                        "conviction": "INSUFFICIENT",
                        "direction": "NEUTRAL",
                        "models_used": ["arima_fallback"],
                        "models_failed": failed,
                        "last_price": last_price,
                        "consensus_7d_change_pct": None,
                        "consensus_price": float(fb_fc[horizon_int - 1]),
                        "price_targets": {},
                        "forecast_debug": forecast_debug,
                    }
            except Exception as _e:
                logger.warning(
                    "ForecastRouter: consensus fallback ARIMA forecast failed: %s",
                    _e,
                )

            return {
                "error": "All model forecasts were flat (no variation). Check data or model config.",
                "models_failed": failed,
                "forecast_debug": forecast_debug,
            }

        # --- Outlier exclusion across valid forecasts ---
        models_excluded: Dict[str, Dict[str, float]] = {}
        vf_map: Dict[str, np.ndarray] = {
            name: np.asarray(fc) for name, fc in zip(valid_used, valid_forecasts)
        }
        if len(vf_map) >= 3:
            means = np.array([float(np.mean(fc)) for fc in vf_map.values()])
            group_mean = float(np.mean(means))
            group_std = float(np.std(means))

            filtered: Dict[str, np.ndarray] = {}
            for model_name, fc in vf_map.items():
                m = float(np.mean(fc))
                if group_std > 0 and abs(m - group_mean) > 2.5 * group_std:
                    deviation_std = abs(m - group_mean) / group_std
                    models_excluded[model_name] = {
                        "mean": round(m, 2),
                        "deviation_std": round(float(deviation_std), 1),
                    }
                    logger.warning(
                        "Excluding %s from consensus: mean=%.2f is %.1f std from group",
                        model_name,
                        m,
                        deviation_std,
                    )
                else:
                    filtered[model_name] = fc

            # Use filtered forecasts only if at least two models remain
            if len(filtered) >= 2:
                vf_map = filtered
            else:
                models_excluded = {}

        # Rebuild valid_forecasts/valid_used from (possibly) filtered vf_map
        valid_used = list(vf_map.keys())

        # Secondary cap: Ridge must be within ±5% of
        # median of other surviving models. Prevents
        # Ridge skewing consensus even after 2.5σ exclusion.
        if "ridge" in vf_map and len(vf_map) >= 2:
            _other_means = [
                float(np.mean(fc))
                for name, fc in vf_map.items()
                if name != "ridge"
            ]
            if _other_means:
                _median_other = float(np.median(_other_means))
                _ridge_mean = float(np.mean(vf_map["ridge"]))
                _cap = 0.05
                _lo = _median_other * (1 - _cap)
                _hi = _median_other * (1 + _cap)
                if _ridge_mean < _lo or _ridge_mean > _hi:
                    logger.warning(
                        "Ridge secondary cap triggered: "
                        "mean=%.2f outside ±5%% of "
                        "median=%.2f — excluding",
                        _ridge_mean, _median_other,
                    )
                    vf_map.pop("ridge")

        # Rebuild after possible Ridge removal
        valid_used = list(vf_map.keys())
        valid_forecasts = [vf_map[name] for name in valid_used]

        price_targets = {
            k: v for k, v in price_targets.items()
            if k in valid_used
        }

        stacked = np.stack(valid_forecasts)
        consensus = stacked.mean(axis=0)
        # Per-timestep std across models — dispersion of model paths, not
        # empirical forecast error; used for agreement bands only.
        std = stacked.std(axis=0)

        # Consensus price at horizon
        consensus_price = float(consensus[horizon_int - 1])
        change_pct = (
            float((consensus_price / last_price - 1) * 100.0) if last_price else 0.0
        )

        # Direction from consensus price (not per-model votes which include excluded outliers)
        if last_price > 0:
            if consensus_price > last_price * 1.005:
                direction = "BULLISH"
            elif consensus_price < last_price * 0.995:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        else:
            direction = "NEUTRAL"

        # Keep per-model directions for price_targets display only (valid_used = post-exclusion)
        model_dirs: Dict[str, int] = {}
        for name in valid_used:
            target = price_targets.get(name)
            if (target is None or
                    not np.isfinite(target) or
                    last_price <= 0):
                d = 0
            else:
                if target > last_price * 1.005:
                    d = 1
                elif target < last_price * 0.995:
                    d = -1
                else:
                    d = 0
            model_dirs[name] = d

        # Conviction from consensus magnitude and model agreement among valid_used
        _agreeing = sum(
            1 for d in model_dirs.values()
            if (d == 1 and direction == "BULLISH")
            or (d == -1 and direction == "BEARISH"))
        _total_valid = max(len(valid_used), 1)
        _agreement_ratio = _agreeing / _total_valid

        if (abs(change_pct) >= 2.0 and
                _agreement_ratio >= 0.6):
            conviction = "HIGH"
        elif (abs(change_pct) >= 0.5 and
                _agreement_ratio >= 0.4):
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        model_agreement = round(_agreement_ratio, 2)

        out: Dict[str, Any] = {
            "consensus_forecast": consensus.tolist(),
            "upper_bound": (consensus + std).tolist(),
            "lower_bound": (consensus - std).tolist(),
            "band_type": "model_disagreement",
            "model_agreement": model_agreement,
            "conviction": conviction,
            "direction": direction,
            "models_used": valid_used,
            "models_failed": failed,
            "models_excluded": models_excluded,
            "last_price": last_price,
            "consensus_7d_change_pct": round(change_pct, 2),
            "consensus_price": consensus_price,
            "price_targets": price_targets,
            "forecast_debug": forecast_debug,
        }
        try:
            import streamlit as st

            sym = (symbol or "").strip().upper()
            if sym:
                wf = st.session_state.get(f"wf_{sym}")
                if isinstance(wf, dict) and not wf.get("error"):
                    hr = wf.get("mean_directional_accuracy")
                    if hr is None:
                        hr = wf.get("hit_rate_5pct")
                    if hr is not None:
                        hit = float(hr)
                        if hit > 0.55:
                            out["walk_forward_confidence"] = "high"
                        elif hit >= 0.45:
                            out["walk_forward_confidence"] = "medium"
                        else:
                            out["walk_forward_confidence"] = "low"
                            out.setdefault(
                                "walk_forward_warnings", []
                            ).append(
                                "Walk-forward hit rate is low (~"
                                f"{hit:.0%}); treat this consensus with caution."
                            )
        except Exception as _wf_e:
            logger.debug("walk-forward session merge skipped: %s", _wf_e)
        return out

