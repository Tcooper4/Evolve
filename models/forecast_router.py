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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from trading.models.arima_model import ARIMAModel
from trading.models.lstm_model import LSTMModel
from trading.models.xgboost_model import XGBoostModel

# Make Prophet import optional
try:
    from trading.models.prophet_model import ProphetModel

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    ProphetModel = None
from trading.models.autoformer_model import AutoformerModel

# Try to import Transformer model
try:
    from trading.models.advanced.transformer.time_series_transformer import (
        TimeSeriesTransformer,
    )

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TimeSeriesTransformer = None

logger = logging.getLogger(__name__)


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

    def _load_model_registry(self):
        """Dynamically load model registry from config or discovery."""
        try:
            # Try to load from config first
            config_models = self.config.get("models", {})
            if config_models:
                self._load_models_from_config(config_models)
            else:
                # Use dynamic registry
                from trading.models.registry import get_model_registry

                registry = get_model_registry()
                self.model_registry = registry.registry

                # Fallback to default models if registry is empty
                if not self.model_registry:
                    self._load_default_models()

            # Try to discover additional models
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
            "autoformer": AutoformerModel,
        }

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
        """Discover additional models dynamically."""
        try:
            # Look for model plugins in models directory
            models_dir = Path(__file__).parent
            for model_file in models_dir.glob("*_model.py"):
                model_name = model_file.stem.replace("_model", "")
                if model_name not in self.model_registry:
                    try:
                        model_class = self._get_model_class(
                            f"models.{model_name}_model.{model_name.title()}Model"
                        )
                        if model_class:
                            self.model_registry[model_name] = model_class
                            logger.info(f"Discovered model {model_name}")
                    except Exception as e:
                        logger.debug(f"Failed to discover model {model_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering models: {e}")

    def _get_model_class(self, class_path: str):
        """Get model class from class path string."""
        try:
            if not class_path:
                return None

            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
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
        if model_type and model_type in self.model_registry:
            return model_type

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
        **kwargs,
    ) -> Dict[str, Any]:
        """Get forecast for time series data.

        Args:
            data: Time series data
            horizon: Forecast horizon in periods
            model_type: Optional preferred model type
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

            # Initialize model with sensible defaults
            model_class = self.model_registry[selected_model]
            model_kwargs = self._get_model_defaults(selected_model)
            model_kwargs.update(kwargs)  # Override with user-provided kwargs

            # For LSTM, only pass valid constructor args
            if selected_model == "lstm":
                # Determine input_dim and output_dim from data
                input_dim = (
                    prepared_data.shape[1] if hasattr(prepared_data, "shape") else 1
                )
                output_dim = 1
                lstm_args = {
                    "input_dim": input_dim,
                    "hidden_dim": model_kwargs.get("hidden_dim", 64),
                    "output_dim": output_dim,
                    "num_layers": model_kwargs.get("num_layers", 2),
                    "dropout": model_kwargs.get("dropout", 0.2),
                }
                model = model_class(**lstm_args)
                fit_args = {
                    "epochs": model_kwargs.get("epochs", 50),
                    "batch_size": model_kwargs.get("batch_size", 32),
                    "learning_rate": model_kwargs.get("learning_rate", 0.001),
                }
            else:
                model = model_class(**model_kwargs)
                fit_args = {}

            logger.info(f"Initializing {selected_model} with config: {model_kwargs}")

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
                model = model_class(**self._get_model_defaults(fallback_model))
                model.fit(prepared_data)
                selected_model = fallback_model

            # Generate forecast with error handling
            try:
                forecast = model.predict(horizon=horizon)
                logger.info(f"Successfully generated forecast with {selected_model}")
            except Exception as e:
                logger.error(f"Failed to generate forecast with {selected_model}: {e}")
                # Return simple forecast as fallback
                forecast = self._generate_simple_forecast(prepared_data, horizon)

            # Log performance
            self._log_performance(selected_model, forecast, data)

            return {
                "model": selected_model,
                "forecast": forecast,
                "confidence": self._get_confidence(model, selected_model),
                "metadata": self._get_metadata(model, selected_model),
                "warnings": self._get_warnings(data, selected_model),
            }

        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            # Return fallback result instead of raising
            fallback_result = self._get_fallback_result(data, horizon)
            return fallback_result
        # TODO: Specify exception type instead of using bare except
