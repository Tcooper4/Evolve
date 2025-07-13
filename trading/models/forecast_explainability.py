"""Intelligent Forecast Explainability.

This module provides comprehensive explainability for model forecasts including
confidence intervals, forecast vs actual plots, and SHAP feature importance.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


@dataclass
class ForecastExplanation:
    """Forecast explanation result."""

    forecast_id: str
    symbol: str
    forecast_date: datetime
    forecast_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    feature_importance: Dict[str, float]
    model_metadata: Dict[str, Any]
    explanation_text: str
    timestamp: datetime


@dataclass
class ForecastAccuracy:
    """Forecast accuracy metrics."""

    forecast_id: str
    symbol: str
    actual_value: float
    forecast_value: float
    error: float
    error_pct: float
    within_confidence: bool
    accuracy_score: float
    timestamp: datetime


class ForecastExplainability:
    """Intelligent forecast explainability engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize forecast explainability engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Confidence interval settings
        self.confidence_levels = self.config.get("confidence_levels", [0.68, 0.95, 0.99])
        self.default_confidence = self.config.get("default_confidence", 0.95)

        # SHAP settings
        self.shap_config = self.config.get(
            "shap_config",
            {"max_features": 20, "background_samples": 100, "explainer_type": "tree"},  # tree, linear, kernel
        )

        # Explanation templates
        self.explanation_templates = {
            "bullish": "The model predicts a {change_pct:.1f}% increase in {symbol} over {horizon} days with {confidence:.1%} confidence. Key drivers include {top_features}.",
            "bearish": "The model predicts a {change_pct:.1f}% decrease in {symbol} over {horizon} days with {confidence:.1%} confidence. Key drivers include {top_features}.",
            "neutral": "The model predicts minimal change in {symbol} over {horizon} days with {confidence:.1%} confidence. Key drivers include {top_features}.",
        }

        # Performance tracking
        self.forecast_history = []
        self.accuracy_history = []
        self.explanation_history = []

        logger.info("Forecast Explainability Engine initialized")

        self.init_status = {
            "success": True,
            "message": "ForecastExplainability initialized successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def explain_forecast(
        self,
        forecast_id: str,
        symbol: str,
        forecast_value: float,
        model: Any,
        features: pd.DataFrame,
        target_history: Optional[pd.Series] = None,
        horizon: int = 15,
    ) -> Dict[str, Any]:
        """Generate comprehensive forecast explanation.

        Args:
            forecast_id: Unique forecast identifier
            symbol: Trading symbol
            forecast_value: Forecasted value
            model: Trained model
            features: Feature data used for prediction
            target_history: Historical target values
            horizon: Forecast horizon in days

        Returns:
            Dictionary containing explanation and status
        """
        try:
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                model, features, forecast_value, self.default_confidence
            )

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, features)

            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                symbol, forecast_value, target_history, feature_importance, horizon
            )

            # Create explanation object
            explanation = ForecastExplanation(
                forecast_id=forecast_id,
                symbol=symbol,
                forecast_date=datetime.now(),
                forecast_value=forecast_value,
                confidence_interval_lower=confidence_intervals["lower"],
                confidence_interval_upper=confidence_intervals["upper"],
                confidence_level=self.default_confidence,
                feature_importance=feature_importance,
                model_metadata=self._extract_model_metadata(model),
                explanation_text=explanation_text,
                timestamp=datetime.now(),
            )

            # Store in history
            self.explanation_history.append(explanation)

            logger.info(f"Generated explanation for forecast {forecast_id}")

            return {
                "success": True,
                "explanation": explanation,
                "message": f"Explanation generated for forecast {forecast_id}",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error explaining forecast {forecast_id}: {e}")
            default_explanation = self._create_default_explanation(forecast_id, symbol, forecast_value)
            return {
                "success": False,
                "explanation": default_explanation,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_confidence_intervals(
        self, model: Any, features: pd.DataFrame, forecast_value: float, confidence_level: float
    ) -> Dict[str, float]:
        """Calculate confidence intervals for forecast.

        Args:
            model: Trained model
            features: Feature data
            forecast_value: Point forecast
            confidence_level: Confidence level (0-1)

        Returns:
            Confidence interval bounds
        """
        try:
            # Method 1: Bootstrap confidence intervals
            if hasattr(model, "predict"):
                bootstrap_predictions = []

                # Generate bootstrap samples
                n_bootstrap = 1000
                for _ in range(n_bootstrap):
                    # Bootstrap sample of features
                    bootstrap_idx = np.random.choice(len(features), size=len(features), replace=True)
                    bootstrap_features = features.iloc[bootstrap_idx]

                    # Add noise to features
                    noise = np.random.normal(0, 0.01, bootstrap_features.shape)
                    bootstrap_features_noisy = bootstrap_features + noise

                    # Make prediction
                    try:
                        pred = model.predict(bootstrap_features_noisy.iloc[-1:])[0]
                        bootstrap_predictions.append(pred)
                    except (ValueError, TypeError, IndexError) as e:
                        logger.debug(f"Bootstrap prediction failed: {e}")
                        bootstrap_predictions.append(forecast_value)

                if bootstrap_predictions:
                    # Calculate percentiles
                    alpha = 1 - confidence_level
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100

                    lower_bound = np.percentile(bootstrap_predictions, lower_percentile)
                    upper_bound = np.percentile(bootstrap_predictions, upper_percentile)

                    return {"lower": lower_bound, "upper": upper_bound}
            # Fallback
            return {"lower": forecast_value * 0.95, "upper": forecast_value * 1.05}
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {"lower": forecast_value * 0.95, "upper": forecast_value * 1.05}

    def _calculate_feature_importance(
        self, model: Any, features: pd.DataFrame, method: str = "shap"
    ) -> Dict[str, float]:
        """Calculate feature importance using SHAP, LIME, or model-specific methods."""
        try:
            feature_importance = {}
            if method.lower() == "lime":
                # LIME explainer
                try:
                    explainer = LimeTabularExplainer(
                        training_data=features.values, feature_names=features.columns.tolist(), mode="regression"
                    )
                    exp = explainer.explain_instance(
                        features.iloc[-1].values, model.predict, num_features=min(20, features.shape[1])
                    )
                    for feature, importance in exp.as_list():
                        feature_importance[feature] = abs(importance)
                    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                    return feature_importance
                except Exception as e:
                    logger.warning(f"LIME calculation failed: {e}")
            elif method.lower() == "shap":
                if SHAP_AVAILABLE and hasattr(model, "predict"):
                    try:
                        # Model type detection
                        model_name = type(model).__name__.lower()
                        if (
                            "xgboost" in model_name
                            or "randomforest" in model_name
                            or hasattr(model, "feature_importances_")
                        ):
                            explainer = shap.TreeExplainer(model)
                        elif "linear" in model_name or hasattr(model, "coef_"):
                            explainer = shap.LinearExplainer(model, features)
                        else:
                            explainer = shap.KernelExplainer(model.predict, features.sample(min(100, len(features))))
                        shap_values = explainer.shap_values(features.iloc[-1:])
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        feature_names = features.columns
                        for i, feature in enumerate(feature_names):
                            if i < len(shap_values[0]):
                                importance = abs(shap_values[0][i])
                                feature_importance[feature] = float(importance)
                        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                        max_features = self.shap_config.get("max_features", 20)
                        feature_importance = dict(list(feature_importance.items())[:max_features])
                        return feature_importance
                    except Exception as e:
                        logger.warning(f"SHAP calculation failed: {e}")
            # Method 2: Model-specific feature importance
            if hasattr(model, "feature_importances_"):
                # Tree-based models (Random Forest, XGBoost, etc.)
                feature_names = features.columns
                importances = model.feature_importances_

                for i, feature in enumerate(feature_names):
                    if i < len(importances):
                        feature_importance[feature] = float(importances[i])

                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

                return feature_importance

            elif hasattr(model, "coef_"):
                # Linear models
                feature_names = features.columns
                coefficients = model.coef_

                for i, feature in enumerate(feature_names):
                    if i < len(coefficients):
                        feature_importance[feature] = float(abs(coefficients[i]))

                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

                return feature_importance

            # Method 3: Correlation-based importance
            else:
                # Calculate correlation with target (if available)
                if "target" in features.columns:
                    correlations = features.corr()["target"].abs()
                    correlations = correlations.drop("target")

                    for feature, corr in correlations.items():
                        feature_importance[feature] = float(corr)

                    # Sort by importance
                    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

                    return feature_importance

            # Default: equal importance
            feature_names = features.columns
            equal_importance = 1.0 / len(feature_names)
            for feature in feature_names:
                feature_importance[feature] = equal_importance

            return feature_importance

        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}

    def _generate_explanation_text(
        self,
        symbol: str,
        forecast_value: float,
        target_history: Optional[pd.Series],
        feature_importance: Dict[str, float],
        horizon: int,
    ) -> str:
        """Generate human-readable explanation text.

        Args:
            symbol: Trading symbol
            forecast_value: Forecasted value
            target_history: Historical target values
            feature_importance: Feature importance dictionary
            horizon: Forecast horizon

        Returns:
            Explanation text
        """
        try:
            # Calculate change percentage
            if target_history is not None and len(target_history) > 0:
                current_value = target_history.iloc[-1]
                change_pct = ((forecast_value - current_value) / current_value) * 100
            else:
                change_pct = 0.0

            # Determine forecast direction
            if change_pct > 2.0:
                direction = "bullish"
            elif change_pct < -2.0:
                direction = "bearish"
            else:
                direction = "neutral"

            # Get top features
            top_features = list(feature_importance.keys())[:3]
            top_features_text = ", ".join(top_features)

            # Generate explanation
            template = self.explanation_templates.get(direction, self.explanation_templates["neutral"])

            explanation = template.format(
                symbol=symbol,
                change_pct=abs(change_pct),
                horizon=horizon,
                confidence=self.default_confidence,
                top_features=top_features_text,
            )

            # Add feature importance details
            if feature_importance:
                top_feature = list(feature_importance.keys())[0]
                top_importance = feature_importance[top_feature]
                explanation += f" The most important factor is {top_feature} (importance: {top_importance:.3f})."

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation text: {e}")
            return f"Forecast for {symbol}: {forecast_value:.2f} over {horizon} days."

    def _extract_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from model.

        Args:
            model: Trained model

        Returns:
            Model metadata
        """
        try:
            metadata = {"model_type": type(model).__name__, "timestamp": datetime.now()}

            # Extract model-specific attributes
            if hasattr(model, "n_estimators"):
                metadata["n_estimators"] = model.n_estimators

            if hasattr(model, "max_depth"):
                metadata["max_depth"] = model.max_depth

            if hasattr(model, "learning_rate"):
                metadata["learning_rate"] = model.learning_rate

            if hasattr(model, "alpha"):
                metadata["alpha"] = model.alpha

            if hasattr(model, "l1_ratio"):
                metadata["l1_ratio"] = model.l1_ratio

            return metadata

        except Exception as e:
            logger.error(f"Error extracting model metadata: {e}")
            return {"model_type": "unknown"}

    def _create_default_explanation(self, forecast_id: str, symbol: str, forecast_value: float) -> ForecastExplanation:
        """Create default explanation when error occurs.

        Args:
            forecast_id: Forecast identifier
            symbol: Trading symbol
            forecast_value: Forecasted value

        Returns:
            Default explanation
        """
        return ForecastExplanation(
            forecast_id=forecast_id,
            symbol=symbol,
            forecast_date=datetime.now(),
            forecast_value=forecast_value,
            confidence_interval_lower=forecast_value * 0.95,
            confidence_interval_upper=forecast_value * 1.05,
            confidence_level=0.95,
            feature_importance={},
            model_metadata=self._extract_model_metadata(None),
            explanation_text=f"Forecast for {symbol}: {forecast_value:.2f}",
            timestamp=datetime.now(),
        )

    def track_forecast_accuracy(
        self,
        forecast_id: str,
        symbol: str,
        actual_value: float,
        forecast_value: float,
        confidence_lower: float,
        confidence_upper: float,
    ) -> ForecastAccuracy:
        """Track forecast accuracy against actual values.

        Args:
            forecast_id: Forecast identifier
            symbol: Trading symbol
            actual_value: Actual observed value
            forecast_value: Forecasted value
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound

        Returns:
            Forecast accuracy metrics
        """
        try:
            # Calculate error metrics
            error = actual_value - forecast_value
            error_pct = (error / actual_value) * 100 if actual_value != 0 else 0

            # Check if within confidence interval
            within_confidence = confidence_lower <= actual_value <= confidence_upper

            # Calculate accuracy score (1 - normalized error)
            accuracy_score = max(0, 1 - abs(error_pct) / 100)

            # Create accuracy object
            accuracy = ForecastAccuracy(
                forecast_id=forecast_id,
                symbol=symbol,
                actual_value=actual_value,
                forecast_value=forecast_value,
                error=error,
                error_pct=error_pct,
                within_confidence=within_confidence,
                accuracy_score=accuracy_score,
                timestamp=datetime.now(),
            )

            # Store in history
            self.accuracy_history.append(accuracy)

            logger.info(f"Tracked accuracy for forecast {forecast_id}: {accuracy_score:.3f}")

            return accuracy

        except Exception as e:
            logger.error(f"Error tracking forecast accuracy: {e}")
            return ForecastAccuracy(
                forecast_id=forecast_id,
                symbol=symbol,
                actual_value=actual_value,
                forecast_value=forecast_value,
                error=0,
                error_pct=0,
                within_confidence=False,
                accuracy_score=0,
                timestamp=datetime.now(),
            )

    def get_forecast_vs_actual_plot_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get data for forecast vs actual plots.

        Args:
            symbol: Trading symbol
            days: Number of days to look back

        Returns:
            Plot data dictionary
        """
        try:
            # Filter accuracy history by symbol and time
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_accuracy = [a for a in self.accuracy_history if a.symbol == symbol and a.timestamp > cutoff_date]

            if not recent_accuracy:
                return {}

            # Prepare plot data
            dates = [a.timestamp for a in recent_accuracy]
            actual_values = [a.actual_value for a in recent_accuracy]
            forecast_values = [a.forecast_value for a in recent_accuracy]
            errors = [a.error for a in recent_accuracy]
            accuracy_scores = [a.accuracy_score for a in recent_accuracy]

            return {
                "dates": dates,
                "actual_values": actual_values,
                "forecast_values": forecast_values,
                "errors": errors,
                "accuracy_scores": accuracy_scores,
                "symbol": symbol,
                "total_forecasts": len(recent_accuracy),
                "avg_accuracy": np.mean(accuracy_scores),
                "avg_error_pct": np.mean([abs(a.error_pct) for a in recent_accuracy]),
            }

        except Exception as e:
            logger.error(f"Error getting forecast vs actual plot data: {e}")
            return {}

    def get_feature_importance_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get feature importance summary.

        Args:
            symbol: Trading symbol
            days: Number of days to look back

        Returns:
            Feature importance summary
        """
        try:
            # Filter explanations by symbol and time
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_explanations = [
                e for e in self.explanation_history if e.symbol == symbol and e.timestamp > cutoff_date
            ]

            if not recent_explanations:
                return {}

            # Aggregate feature importance
            all_features = {}
            feature_counts = {}

            for explanation in recent_explanations:
                for feature, importance in explanation.feature_importance.items():
                    if feature not in all_features:
                        all_features[feature] = []
                        feature_counts[feature] = 0

                    all_features[feature].append(importance)
                    feature_counts[feature] += 1

            # Calculate average importance
            avg_importance = {}
            for feature, importances in all_features.items():
                avg_importance[feature] = np.mean(importances)

            # Sort by average importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            return {
                "symbol": symbol,
                "total_forecasts": len(recent_explanations),
                "feature_importance": dict(sorted_features[:10]),  # Top 10 features
                "feature_counts": feature_counts,
                "most_important_features": [f[0] for f in sorted_features[:5]],
            }

        except Exception as e:
            logger.error(f"Error getting feature importance summary: {e}")
            return {}

    def get_explanation_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get explanation summary statistics.

        Args:
            symbol: Trading symbol
            days: Number of days to look back

        Returns:
            Explanation summary
        """
        try:
            # Filter explanations by symbol and time
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_explanations = [
                e for e in self.explanation_history if e.symbol == symbol and e.timestamp > cutoff_date
            ]

            if not recent_explanations:
                return {}

            # Calculate summary statistics
            forecast_values = [e.forecast_value for e in recent_explanations]
            confidence_levels = [e.confidence_level for e in recent_explanations]

            # Count explanation types
            explanation_types = {}
            for explanation in recent_explanations:
                if "bullish" in explanation.explanation_text.lower():
                    explanation_types["bullish"] = explanation_types.get("bullish", 0) + 1
                elif "bearish" in explanation.explanation_text.lower():
                    explanation_types["bearish"] = explanation_types.get("bearish", 0) + 1
                else:
                    explanation_types["neutral"] = explanation_types.get("neutral", 0) + 1

            return {
                "symbol": symbol,
                "total_explanations": len(recent_explanations),
                "avg_forecast_value": np.mean(forecast_values),
                "forecast_volatility": np.std(forecast_values),
                "avg_confidence_level": np.mean(confidence_levels),
                "explanation_types": explanation_types,
                "recent_explanations": [
                    {
                        "date": e.timestamp,
                        "forecast_value": e.forecast_value,
                        "confidence_level": e.confidence_level,
                        "explanation": e.explanation_text[:100] + "..."
                        if len(e.explanation_text) > 100
                        else e.explanation_text,
                    }
                    for e in recent_explanations[-5:]  # Last 5 explanations
                ],
            }

        except Exception as e:
            logger.error(f"Error getting explanation summary: {e}")
            return {}

    def export_explanation_report(self, filepath: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Export explanation report to a file.

        Args:
            filepath: Path to save the report
            symbol: Optional symbol to filter explanations

        Returns:
            Dictionary with export status
        """
        try:
            explanations = self.explanation_history
            if symbol:
                explanations = [e for e in explanations if e.symbol == symbol]

            report_data = [e.__dict__ for e in explanations]
            import json

            with open(filepath, "w") as f:
                json.dump(report_data, f, default=str, indent=4)

            return {
                "success": True,
                "message": f"Explanation report exported to {filepath}",
                "timestamp": datetime.now().isoformat(),
                "filepath": filepath,
                "explanations_count": len(report_data),
            }
        except Exception as e:
            logger.error(f"Error exporting explanation report: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}


# Global forecast explainability instance
forecast_explainability = ForecastExplainability()


def get_forecast_explainability() -> ForecastExplainability:
    """Get the global forecast explainability instance."""
    return forecast_explainability
