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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.models.forecast_features import add_forecast_features, prepare_forecast_data
from trading.models.arima_model import ARIMAModel
from trading.models.lstm_model import LSTMModel
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
        TimeSeriesTransformer,
    )

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TimeSeriesTransformer = None

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
            self.model_registry["transformer"] = TimeSeriesTransformer

        self.performance_history = pd.DataFrame()
        self.model_weights = self._initialize_weights()
        self._last_price_used = 1.0

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
                self.model_registry = getattr(registry, "registry", None) or getattr(registry, "_models", None) or {}

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
            default_models["transformer"] = TimeSeriesTransformer

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
        # Implement seasonality detection
        return False

    def _check_trend(self, data: pd.DataFrame) -> bool:
        """Check for trend in time series.

        Args:
            data: Time series data

        Returns:
            True if trend is detected
        """
        # Implement trend detection
        return False

    def _prepare_data_safely(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data: add forecast features and normalize target by raw last price. Store raw last_price for single denorm."""
        if data is None or data.empty:
            self._last_price_used = 1.0
            return pd.DataFrame()
        prepared, last_price = prepare_forecast_data(data)
        self._last_price_used = last_price  # raw close, used exactly once to denormalize forecast
        logger.info("Forecast denorm: last_price=%.2f (raw close, multiply forecast once)", self._last_price_used)
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
            if not numeric.empty:
                last = float(numeric.iloc[-1].mean())
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
            "tcn": {"target_column": "close", "feature_columns": ["close", "volume"], "sequence_length": 20},
            "catboost": {"target_column": "close", "iterations": 500, "depth": 6},
        }
        return defaults.get(model_name, {"target_column": "close"}).copy()

    def _generate_simple_forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate a simple constant/last-value forecast."""
        if data is None or data.empty or horizon <= 0:
            return np.array([])
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return np.zeros(horizon)
        last = float(numeric.iloc[-1].mean())
        return np.full(horizon, last)

    def _get_confidence(self, model: Any, model_name: str) -> float:
        """Return confidence score for the model."""
        try:
            if hasattr(model, "get_confidence"):
                c = model.get_confidence()
                return float(c.get("confidence", 0.5)) if isinstance(c, dict) else float(c)
        except Exception:
            pass
        return 0.5

    def _get_metadata(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Return metadata dict for the model."""
        try:
            if hasattr(model, "get_model_info"):
                return model.get_model_info() or {}
        except Exception:
            pass
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

    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate minimal fallback DataFrame for missing data."""
        return pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

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
        # Handle None or invalid model_name with fallback
        if model_type is None:
            logger.warning("No model_name specified, defaulting to Prophet forecaster")
            if PROPHET_AVAILABLE:
                return "prophet"
            else:
                logger.warning("Prophet not available, falling back to ARIMA")
                return "arima"

        # Check if model exists in registry
        if model_type not in self.model_registry:
            logger.warning(
                f"Model '{model_type}' not found in registry, defaulting to Prophet forecaster"
            )
            if PROPHET_AVAILABLE:
                return "prophet"
            else:
                logger.warning("Prophet not available, falling back to ARIMA")
                return "arima"

        # Alias matching for xgboost/xgb
        if model_type.lower() in ["xgboost", "xgb"]:
            return "xgboost"

        # Analyze data characteristics
        characteristics = self._analyze_data(data)

        # Apply selection rules
        if characteristics["length"] < 100:
            return "arima"  # Better for short series
        elif characteristics["has_seasonality"] and PROPHET_AVAILABLE:
            return "prophet"  # Better for seasonal data
        elif characteristics["has_trend"]:
            return "autoformer"  # Better for trending data
        else:
            return "lstm"

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
            # Defensive checks for input parameters
            if data is None or data.empty:
                logger.warning("Empty or None data provided, using fallback data")
                data = self._generate_fallback_data()

            if horizon is None or horizon <= 0:
                logger.warning(
                    f"Invalid horizon {horizon}, using default horizon of 30"
                )
                horizon = 30

            # Validate data structure
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Data must be pandas DataFrame, got {type(data)}")
                raise ValueError("Data must be pandas DataFrame")

            # Check for required columns
            required_columns = (
                ["close", "volume"] if "close" in data.columns else ["price"]
            )
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                logger.warning(
                    f"Missing columns {missing_columns}, using available columns"
                )
                # Use first numeric column as target
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    target_col = numeric_columns[0]
                    logger.info(f"Using {target_col} as target column")
                else:
                    raise ValueError("No numeric columns found in data")

            # Prepare data with defensive checks
            prepared_data = self._prepare_data_safely(data)

            # Select model with fallback logic
            selected_model = self._select_model_with_fallback(prepared_data, model_type)
            logger.info(f"Selected model: {selected_model}")

            # Initialize model with single config dict (all constructors accept config: Dict)
            model_class = self.model_registry[selected_model]
            config = self._get_model_defaults(selected_model)
            kwargs_for_model = {k: v for k, v in kwargs.items() if k != "run_walk_forward"}
            config.update(kwargs_for_model)
            if selected_model == "lstm":
                config["input_dim"] = (
                    prepared_data.shape[1] if hasattr(prepared_data, "shape") else 1
                )
                config["output_dim"] = 1
            model = model_class(config)
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

            # Fit model with error handling
            try:
                if selected_model == "lstm":
                    model.train_model(
                        prepared_data, prepared_data.iloc[:, 0], **fit_args
                    )
                else:
                    model.fit(prepared_data)
                logger.info(f"Successfully fitted {selected_model}")
            except Exception as e:
                logger.error(f"Failed to fit {selected_model}: {e}")
                # Try fallback model
                fallback_model = self._get_fallback_model(selected_model)
                logger.info(f"Trying fallback model: {fallback_model}")
                model_class = self.model_registry[fallback_model]
                fallback_config = self._get_model_defaults(fallback_model)
                if fallback_model == "lstm":
                    fallback_config["input_dim"] = (
                        prepared_data.shape[1] if hasattr(prepared_data, "shape") else 1
                    )
                    fallback_config["output_dim"] = 1
                model = model_class(fallback_config)
                model.fit(prepared_data)
                selected_model = fallback_model

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

                logger.info(f"Successfully generated forecast with {selected_model}")
            except Exception as e:
                logger.error(f"Failed to generate forecast with {selected_model}: {e}")
                # Return simple forecast as fallback (in normalized space; denorm once)
                forecast_array = self._generate_simple_forecast(prepared_data, horizon)
                lp = getattr(self, "_last_price_used", 1.0)
                if lp and lp != 1.0:
                    forecast_array = np.asarray(forecast_array, dtype="float64") * lp
                result = {"forecast": forecast_array}

            # In-sample MAPE and poor-fit warning
            warnings_list = list(self._get_warnings(data, selected_model))
            in_sample_mape = self._in_sample_mape(model, prepared_data, selected_model)
            if in_sample_mape is not None and in_sample_mape > 15.0:
                msg = (
                    f"Model {selected_model} has poor in-sample fit (MAPE={in_sample_mape:.1f}%). "
                    "Consider more training data or different hyperparameters."
                )
                logger.warning(msg)
                warnings_list.append(msg)

            last_actual_price = getattr(self, "_last_price_used", 1.0)
            # Confidence label: Low = MAPE < 5%, Medium = 5-10%, High = > 10%
            vmape = validation_mape if np.isfinite(validation_mape) else None
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

            return {
                "model": selected_model,
                "forecast": forecast_array,
                "confidence": self._get_confidence(model, selected_model),
                "metadata": self._get_metadata(model, selected_model),
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
                    raw_vals = (
                        fc.get("predictions")
                        or fc.get("forecast")
                        or fc.get("values")
                        or fc.get("forecast_values")
                    )
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
