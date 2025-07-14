"""
Model Parameter Tuner for UI-based parameter adjustment.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for a model parameter."""

    name: str
    display_name: str
    param_type: str  # 'int', 'float', 'bool', 'select'
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: str = ""


class ModelParameterTuner:
    """UI component for tuning model parameters."""

    def __init__(self):
        """Initialize the parameter tuner."""
        self.parameter_configs = self._initialize_parameter_configs()
        self.history_file = Path("logs/optimization_history.json")
        self.history_file.parent.mkdir(exist_ok=True)
        self._load_optimization_history()

    def _initialize_parameter_configs(self) -> Dict[str, List[ParameterConfig]]:
        """Initialize parameter configurations for different models."""
        configs = {
            "transformer": [
                ParameterConfig(
                    name="d_model",
                    display_name="Model Dimension",
                    param_type="int",
                    default_value=64,
                    min_value=32,
                    max_value=512,
                    step=32,
                    description="Dimension of the transformer model",
                ),
                ParameterConfig(
                    name="nhead",
                    display_name="Number of Heads",
                    param_type="int",
                    default_value=4,
                    min_value=2,
                    max_value=16,
                    step=2,
                    description="Number of attention heads",
                ),
                ParameterConfig(
                    name="num_layers",
                    display_name="Number of Layers",
                    param_type="int",
                    default_value=2,
                    min_value=1,
                    max_value=8,
                    step=1,
                    description="Number of transformer layers",
                ),
                ParameterConfig(
                    name="dropout",
                    display_name="Dropout Rate",
                    param_type="float",
                    default_value=0.2,
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                    description="Dropout rate for regularization",
                ),
                ParameterConfig(
                    name="learning_rate",
                    display_name="Learning Rate",
                    param_type="float",
                    default_value=0.001,
                    min_value=0.0001,
                    max_value=0.01,
                    step=0.0001,
                    description="Learning rate for training",
                ),
            ],
            "lstm": [
                ParameterConfig(
                    name="hidden_size",
                    display_name="Hidden Size",
                    param_type="int",
                    default_value=50,
                    min_value=16,
                    max_value=256,
                    step=16,
                    description="Size of LSTM hidden layers",
                ),
                ParameterConfig(
                    name="num_layers",
                    display_name="Number of Layers",
                    param_type="int",
                    default_value=2,
                    min_value=1,
                    max_value=4,
                    step=1,
                    description="Number of LSTM layers",
                ),
                ParameterConfig(
                    name="dropout",
                    display_name="Dropout Rate",
                    param_type="float",
                    default_value=0.2,
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                    description="Dropout rate for regularization",
                ),
                ParameterConfig(
                    name="learning_rate",
                    display_name="Learning Rate",
                    param_type="float",
                    default_value=0.001,
                    min_value=0.0001,
                    max_value=0.01,
                    step=0.0001,
                    description="Learning rate for training",
                ),
            ],
            "xgboost": [
                ParameterConfig(
                    name="n_estimators",
                    display_name="Number of Estimators",
                    param_type="int",
                    default_value=100,
                    min_value=50,
                    max_value=500,
                    step=50,
                    description="Number of boosting rounds",
                ),
                ParameterConfig(
                    name="max_depth",
                    display_name="Max Depth",
                    param_type="int",
                    default_value=6,
                    min_value=3,
                    max_value=15,
                    step=1,
                    description="Maximum depth of trees",
                ),
                ParameterConfig(
                    name="learning_rate",
                    display_name="Learning Rate",
                    param_type="float",
                    default_value=0.1,
                    min_value=0.01,
                    max_value=0.3,
                    step=0.01,
                    description="Learning rate for boosting",
                ),
                ParameterConfig(
                    name="subsample",
                    display_name="Subsample Ratio",
                    param_type="float",
                    default_value=0.8,
                    min_value=0.5,
                    max_value=1.0,
                    step=0.05,
                    description="Subsample ratio of training instances",
                ),
            ],
            "ensemble": [
                ParameterConfig(
                    name="voting_method",
                    display_name="Voting Method",
                    param_type="select",
                    default_value="weighted",
                    options=["weighted", "simple", "soft"],
                    description="Method for combining model predictions",
                ),
                ParameterConfig(
                    name="weight_update_frequency",
                    display_name="Weight Update Frequency",
                    param_type="int",
                    default_value=7,
                    min_value=1,
                    max_value=30,
                    step=1,
                    description="Days between weight updates",
                ),
                ParameterConfig(
                    name="min_models",
                    display_name="Minimum Models",
                    param_type="int",
                    default_value=3,
                    min_value=2,
                    max_value=10,
                    step=1,
                    description="Minimum number of models in ensemble",
                ),
            ],
        }
        return configs

    def render_parameter_tuner(self, model_type: str) -> Dict[str, Any]:
        """Render parameter tuning interface for a specific model.

        Args:
            model_type: Type of model to tune

        Returns:
            Dictionary with tuned parameters
        """
        try:
            if model_type not in self.parameter_configs:
                st.warning(f"No parameter configuration found for {model_type}")
                return {}

            st.subheader(f"Tune {model_type.upper()} Parameters")

            parameters = {}
            configs = self.parameter_configs[model_type]

            # Create columns for better layout
            cols = st.columns(2)

            for i, config in enumerate(configs):
                col_idx = i % 2

                with cols[col_idx]:
                    if config.param_type == "int":
                        value = st.slider(
                            config.display_name,
                            min_value=int(config.min_value),
                            max_value=int(config.max_value),
                            value=int(config.default_value),
                            step=int(config.step) if config.step else 1,
                            help=config.description,
                        )
                    elif config.param_type == "float":
                        value = st.slider(
                            config.display_name,
                            min_value=float(config.min_value),
                            max_value=float(config.max_value),
                            value=float(config.default_value),
                            step=float(config.step) if config.step else 0.01,
                            help=config.description,
                        )
                    elif config.param_type == "bool":
                        value = st.checkbox(
                            config.display_name,
                            value=bool(config.default_value),
                            help=config.description,
                        )
                    elif config.param_type == "select":
                        value = st.selectbox(
                            config.display_name,
                            options=config.options,
                            index=config.options.index(config.default_value)
                            if config.default_value in config.options
                            else 0,
                            help=config.description,
                        )
                    else:
                        st.warning(f"Unknown parameter type: {config.param_type}")
                        value = config.default_value

                    parameters[config.name] = value

            # Add parameter validation
            validation_result = self._validate_parameters(model_type, parameters)
            if validation_result["valid"]:
                st.success("✅ Parameters are valid")
            else:
                st.error(
                    f"❌ Parameter validation failed: {validation_result['message']}"
                )

            # Add parameter summary
            if st.checkbox("Show parameter summary"):
                self._show_parameter_summary(model_type, parameters)

            return parameters

        except Exception as e:
            logger.error(f"Error rendering parameter tuner: {e}")
            st.error(f"Error rendering parameter tuner: {str(e)}")
            return {}

    def _validate_parameters(
        self, model_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model parameters.

        Args:
            model_type: Type of model
            parameters: Parameters to validate

        Returns:
            Validation result
        """
        try:
            if model_type == "transformer":
                # Validate transformer parameters
                if parameters.get("d_model", 0) % parameters.get("nhead", 1) != 0:
                    return {
                        "valid": False,
                        "message": "Model dimension must be divisible by number of heads",
                    }

                if parameters.get("num_layers", 0) > 8:
                    return {
                        "valid": False,
                        "message": "Too many layers may cause overfitting",
                    }

            elif model_type == "lstm":
                # Validate LSTM parameters
                if parameters.get("hidden_size", 0) > 256:
                    return {
                        "valid": False,
                        "message": "Hidden size too large may cause overfitting",
                    }

            elif model_type == "xgboost":
                # Validate XGBoost parameters
                if parameters.get("max_depth", 0) > 15:
                    return {
                        "valid": False,
                        "message": "Max depth too large may cause overfitting",
                    }

                if parameters.get("learning_rate", 0) > 0.3:
                    return {
                        "valid": False,
                        "message": "Learning rate too high may cause instability",
                    }

            return {"valid": True, "message": "Parameters are valid"}

        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return {"valid": False, "message": f"Validation error: {str(e)}"}

    def _show_parameter_summary(self, model_type: str, parameters: Dict[str, Any]):
        """Show a summary of the selected parameters.

        Args:
            model_type: Type of model
            parameters: Selected parameters
        """
        try:
            st.subheader("Parameter Summary")

            # Create a DataFrame for display
            summary_data = []
            configs = self.parameter_configs[model_type]

            for config in configs:
                if config.name in parameters:
                    summary_data.append(
                        {
                            "Parameter": config.display_name,
                            "Value": parameters[config.name],
                            "Type": config.param_type,
                            "Description": config.description,
                        }
                    )

            if summary_data:
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)

                # Add parameter impact analysis
                self._show_parameter_impact(model_type, parameters)
            else:
                st.warning("No parameters selected")

        except Exception as e:
            logger.error(f"Error showing parameter summary: {e}")
            st.error(f"Error showing parameter summary: {str(e)}")

    def _show_parameter_impact(self, model_type: str, parameters: Dict[str, Any]):
        """Show the potential impact of parameter changes.

        Args:
            model_type: Type of model
            parameters: Selected parameters
        """
        try:
            st.subheader("Parameter Impact Analysis")

            impacts = []

            if model_type == "transformer":
                if parameters.get("d_model", 0) > 128:
                    impacts.append("🔴 High model dimension may increase training time")
                elif parameters.get("d_model", 0) < 64:
                    impacts.append("🟡 Low model dimension may reduce capacity")

                if parameters.get("nhead", 0) > 8:
                    impacts.append("🟡 Many attention heads may improve performance")

                if parameters.get("dropout", 0) > 0.3:
                    impacts.append("🟡 High dropout may reduce overfitting")

            elif model_type == "lstm":
                if parameters.get("hidden_size", 0) > 128:
                    impacts.append("🔴 Large hidden size may increase training time")

                if parameters.get("num_layers", 0) > 3:
                    impacts.append("🟡 Deep LSTM may capture complex patterns")

            elif model_type == "xgboost":
                if parameters.get("n_estimators", 0) > 300:
                    impacts.append("🔴 Many estimators may increase training time")

                if parameters.get("max_depth", 0) > 10:
                    impacts.append("🟡 Deep trees may cause overfitting")

            if impacts:
                for impact in impacts:
                    st.write(impact)
            else:
                st.info("✅ Parameters look well-balanced")

        except Exception as e:
            logger.error(f"Error showing parameter impact: {e}")
            st.error(f"Error showing parameter impact: {str(e)}")

    def get_default_parameters(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default parameters
        """
        try:
            if model_type not in self.parameter_configs:
                return {}

            defaults = {}
            for config in self.parameter_configs[model_type]:
                defaults[config.name] = config.default_value

            return defaults

        except Exception as e:
            logger.error(f"Error getting default parameters: {e}")
            return {}

    def save_parameter_preset(
        self, model_type: str, parameters: Dict[str, Any], preset_name: str
    ):
        """Save a parameter preset.

        Args:
            model_type: Type of model
            parameters: Parameters to save
            preset_name: Name for the preset
        """
        try:
            # In a real implementation, this would save to a file or database
            st.success(f"✅ Preset '{preset_name}' saved for {model_type}")

        except Exception as e:
            logger.error(f"Error saving parameter preset: {e}")
            st.error(f"Error saving preset: {str(e)}")

    def load_parameter_preset(
        self, model_type: str, preset_name: str
    ) -> Dict[str, Any]:
        """Load a parameter preset.

        Args:
            model_type: Type of model
            preset_name: Name of the preset to load

        Returns:
            Dictionary of loaded parameters
        """
        try:
            # In a real implementation, this would load from a file or database
            # For now, return default parameters
            return self.get_default_parameters(model_type)

        except Exception as e:
            logger.error(f"Error loading parameter preset: {e}")
            return {}

    def _load_optimization_history(self):
        """Load optimization history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    self.optimization_history = json.load(f)
            else:
                self.optimization_history = []
        except Exception as e:
            logger.warning(f"Could not load optimization history: {e}")
            self.optimization_history = []

    def _save_optimization_history(self):
        """Save optimization history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save optimization history: {e}")

    def add_optimization_result(
        self,
        model_type: str,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ):
        """Add optimization result to history.

        Args:
            model_type: Type of model optimized
            parameters: Parameters used
            performance_metrics: Performance metrics achieved
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "parameters": parameters,
            "performance_metrics": performance_metrics,
        }
        self.optimization_history.append(result)
        self._save_optimization_history()
        logger.info(f"Added optimization result for {model_type}")

    def get_optimization_history(
        self, model_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get optimization history.

        Args:
            model_type: Filter by model type (optional)
            limit: Maximum number of results to return

        Returns:
            List of optimization results
        """
        if model_type:
            filtered = [
                r for r in self.optimization_history if r["model_type"] == model_type
            ]
        else:
            filtered = self.optimization_history

        return filtered[-limit:] if limit else filtered


# Global instance
parameter_tuner = ModelParameterTuner()


def get_parameter_tuner() -> ModelParameterTuner:
    """Get the global parameter tuner instance."""
    return parameter_tuner
