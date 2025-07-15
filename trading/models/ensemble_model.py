"""Enhanced ensemble model with weighted voting and strategy-aware routing."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelRegistry
from utils.model_cache import cache_model_operation, get_model_cache

# @ModelRegistry.register('Ensemble')


class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple forecasting models with adaptive weights."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble model.

        Args:
            config: Configuration dictionary containing:
                - models: List of model configurations to include
                - voting_method: 'mse', 'sharpe', 'regime', or 'custom'
                - weight_window: Window size for rolling performance
                - fallback_threshold: Minimum confidence for predictions
                - strategy_aware: Whether to use strategy-aware routing
                - ensemble_method: 'weighted_average' or 'vote_based'
                - dynamic_weighting: Whether to use dynamic weighting
                - regime_detection: Whether to use regime-based weighting
        """
        super().__init__(config)
        self._validate_config()
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.strategy_patterns = {}
        self.ensemble_method = config.get("ensemble_method", "weighted_average")
        self.dynamic_weighting = config.get("dynamic_weighting", True)
        self.regime_detection = config.get("regime_detection", False)
        self._load_strategy_patterns()

    def _validate_config(self):
        """Validate ensemble configuration."""
        required = ["models", "voting_method", "weight_window"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config["voting_method"] not in ["mse", "sharpe", "regime", "custom"]:
            raise ValueError("voting_method must be 'mse', 'sharpe', 'regime', or 'custom'")

        if self.ensemble_method not in ["weighted_average", "vote_based"]:
            raise ValueError("ensemble_method must be 'weighted_average' or 'vote_based'")

        if not isinstance(self.config["models"], list):
            raise ValueError("models must be a list of model configurations")

    def _load_strategy_patterns(self):
        """Load strategy patterns from memory.json."""
        try:
            with open("memory.json", "r") as f:
                memory = json.load(f)
                self.strategy_patterns = memory.get("model_strategy_patterns", {})
        except FileNotFoundError:
            self.strategy_patterns = {}

    def _save_strategy_patterns(self):
        """Save strategy patterns to memory.json."""
        try:
            with open("memory.json", "r") as f:
                memory = json.load(f)
        except FileNotFoundError:
            memory = {}

        memory["model_strategy_patterns"] = self.strategy_patterns

        with open("memory.json", "w") as f:
            json.dump(memory, f, indent=4)

    def _initialize_models(self):
        """Initialize all models in the ensemble."""
        for model_config in self.config["models"]:
            model_name = model_config["name"]
            model_class = ModelRegistry.get_model_class(model_name)
            self.models[model_name] = model_class(model_config)
            self.weights[model_name] = 1.0 / len(self.config["models"])
            self.performance_history[model_name] = []

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _update_weights(self, data: pd.DataFrame):
        """Update model weights based on recent performance using vectorized operations.

        Args:
            data: Recent data for performance calculation
        """
        window = self.config["weight_window"]
        recent_data = data.iloc[-window:]
        actual = recent_data["close"].values

        # Vectorized performance calculation
        model_scores = {}
        model_confidences = {}

        for model_name, model in self.models.items():
            try:
                # Get model predictions
                preds = model.predict(recent_data)

                # Calculate performance metrics using vectorized operations
                if self.config["voting_method"] == "mse":
                    score = -np.mean((actual - preds) ** 2)  # Negative MSE
                elif self.config["voting_method"] == "sharpe":
                    returns = np.diff(preds) / preds[:-1]
                    score = (
                        np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    )
                else:  # custom
                    score = self._calculate_custom_score(actual, preds)

                # Get confidence
                confidence = (
                    model.calculate_confidence(preds)
                    if hasattr(model, "calculate_confidence")
                    else 1.0
                )

                model_scores[model_name] = score
                model_confidences[model_name] = confidence

                # Update performance history
                self.performance_history[model_name].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": score,
                        "confidence": confidence,
                    }
                )

            except Exception as e:
                logging.error(f"Error updating weights for {model_name}: {e}")
                # Use default score for failed models
                model_scores[model_name] = 0.0
                model_confidences[model_name] = 0.0

        # Enhanced dynamic weighting with multiple factors
        self.weights = self._calculate_dynamic_weights(
            model_scores, model_confidences, data
        )

    def _calculate_dynamic_weights(
        self,
        model_scores: Dict[str, float],
        model_confidences: Dict[str, float],
        data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate dynamic weights using multiple performance factors.

        Args:
            model_scores: Performance scores for each model
            model_confidences: Confidence scores for each model
            data: Historical data for trend analysis

        Returns:
            Dictionary of normalized weights
        """
        # Get historical performance trends
        historical_performance = self._get_historical_performance()

        # Calculate trend-based weights
        trend_weights = self._calculate_trend_weights(data)

        # Calculate volatility-adjusted weights
        volatility_weights = self._calculate_volatility_weights(data)

        # Combine all factors
        final_weights = {}
        for model_name in self.models.keys():
            # Base score (current performance)
            base_score = model_scores.get(model_name, 0.0)

            # Historical performance (weighted average of recent scores)
            hist_score = historical_performance.get(model_name, 0.0)

            # Confidence score
            conf_score = model_confidences.get(model_name, 0.5)

            # Trend alignment score
            trend_score = trend_weights.get(model_name, 0.5)

            # Volatility adjustment
            vol_score = volatility_weights.get(model_name, 0.5)

            # Combine scores with configurable weights
            weights_config = self.config.get(
                "weight_factors",
                {
                    "current_performance": 0.4,
                    "historical_performance": 0.2,
                    "confidence": 0.15,
                    "trend_alignment": 0.15,
                    "volatility_adjustment": 0.1,
                },
            )

            final_score = (
                weights_config["current_performance"] * base_score
                + weights_config["historical_performance"] * hist_score
                + weights_config["confidence"] * conf_score
                + weights_config["trend_alignment"] * trend_score
                + weights_config["volatility_adjustment"] * vol_score
            )

            final_weights[model_name] = max(0.01, final_score)  # Minimum weight

        # Normalize weights
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            # Fallback to equal weights
            n_models = len(self.models)
            final_weights = {k: 1.0 / n_models for k in self.models.keys()}

        return final_weights

    def _get_historical_performance(self) -> Dict[str, float]:
        """Get historical performance scores for each model."""
        historical_scores = {}

        for model_name, history in self.performance_history.items():
            if len(history) > 0:
                # Calculate weighted average of recent performance
                recent_scores = [
                    entry["score"] for entry in history[-10:]
                ]  # Last 10 entries
                weights = np.exp(
                    np.linspace(0, 1, len(recent_scores))
                )  # Exponential weighting
                weighted_avg = np.average(recent_scores, weights=weights)
                historical_scores[model_name] = weighted_avg
            else:
                historical_scores[model_name] = 0.0

        return historical_scores

    def _calculate_trend_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate weights based on trend alignment."""
        # Calculate market trend
        returns = data["close"].pct_change().dropna()
        market_trend = returns.mean()
        market_volatility = returns.std()

        trend_weights = {}
        for model_name, model in self.models.items():
            try:
                # Get model's trend prediction
                preds = model.predict(data.iloc[-20:])
                pred_returns = np.diff(preds) / preds[:-1]
                pred_trend = np.mean(pred_returns)

                # Calculate trend alignment
                trend_alignment = 1 - abs(market_trend - pred_trend) / (
                    market_volatility + 1e-8
                )
                trend_alignment = np.clip(trend_alignment, 0, 1)

                trend_weights[model_name] = trend_alignment

            except Exception as e:
                logging.warning(f"Error calculating trend weight for {model_name}: {e}")
                trend_weights[model_name] = 0.5

        return trend_weights

    def _calculate_volatility_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate weights based on volatility regime."""
        # Calculate current volatility
        returns = data["close"].pct_change().dropna()
        current_volatility = returns.std()

        # Calculate historical volatility for comparison
        historical_volatility = returns.rolling(20).std().mean()

        # Determine volatility regime
        if current_volatility > historical_volatility * 1.5:
            regime = "high_volatility"
        elif current_volatility < historical_volatility * 0.5:
            regime = "low_volatility"
        else:
            regime = "normal_volatility"

        # Assign weights based on regime
        volatility_weights = {}
        for model_name in self.models.keys():
            if regime == "high_volatility":
                # Prefer models that handle volatility well
                volatility_weights[model_name] = (
                    0.8 if "lstm" in model_name.lower() else 0.6
                )
            elif regime == "low_volatility":
                # Prefer simpler models
                volatility_weights[model_name] = (
                    0.8 if "arima" in model_name.lower() else 0.6
                )
            else:
                # Normal regime - balanced weights
                volatility_weights[model_name] = 0.7

        return volatility_weights

    def _calculate_custom_score(self, actual: np.ndarray, preds: np.ndarray) -> float:
        """Calculate custom performance score.

        Args:
            actual: Actual values
            preds: Predicted values

        Returns:
            Custom score combining multiple metrics
        """
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(actual))
        direction_pred = np.sign(np.diff(preds))
        directional_accuracy = np.mean(direction_true == direction_pred)

        # Calculate normalized MSE
        mse = np.mean((actual - preds) ** 2)
        normalized_mse = 1 / (1 + mse)

        # Combine metrics
        return 0.7 * directional_accuracy + 0.3 * normalized_mse

    def _get_strategy_recommendation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get strategy-aware model recommendations.

        Args:
            data: Input data

        Returns:
            Dictionary with strategy recommendations
        """
        # Detect market regime
        returns = data["close"].pct_change()
        volatility = returns.std()
        trend = returns.mean()

        if trend > 0.001 and volatility < 0.02:
            regime = "bull"
        elif trend < -0.001 and volatility > 0.02:
            regime = "bear"
        else:
            regime = "neutral"

        # Get model confidences
        confidences = {}
        for model_name, model in self.models.items():
            try:
                preds = model.predict(data.iloc[-20:])  # Use last 20 points
                confidences[model_name] = (
                    model.calculate_confidence(preds)
                    if hasattr(model, "calculate_confidence")
                    else 1.0
                )
            except Exception as e:
                logging.error(f"Error calculating confidence for {model_name}: {e}")
                raise RuntimeError(
                    f"Failed to calculate confidence for {model_name}: {e}"
                )

        # Update strategy patterns
        self.strategy_patterns[regime] = {
            "timestamp": datetime.now().isoformat(),
            "model_confidences": confidences,
            "best_model": max(confidences.items(), key=lambda x: x[1])[0],
        }
        self._save_strategy_patterns()

        return {
            "regime": regime,
            "model_confidences": confidences,
            "recommended_model": self.strategy_patterns[regime]["best_model"],
        }

    def fit(self, data: pd.DataFrame):
        """Train all models in the ensemble.

        Args:
            data: Training data
        """
        self._initialize_models()
        for model in self.models.values():
            model.fit(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions using the specified ensemble method.

        Args:
            data: Input data for prediction

        Returns:
            Ensemble predictions
        """
        try:
            if not self.models:
                self._initialize_models()

            # Update weights if dynamic weighting is enabled
            if self.dynamic_weighting:
                self._update_weights(data)

            # Get strategy recommendations
            self._get_strategy_recommendation(data)

            # Get predictions from all models
            model_predictions = {}
            model_confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(data)
                    model_predictions[model_name] = pred
                    
                    # Get confidence if available
                    confidence = (
                        model.calculate_confidence(pred)
                        if hasattr(model, "calculate_confidence")
                        else 1.0
                    )
                    
                    # Apply fallback logic
                    if confidence < self.config.get("fallback_threshold", 0.5):
                        logger.warning(f"Low confidence for {model_name}, using fallback")
                        continue
                    
                    model_confidences[model_name] = confidence
                        
                except Exception as e:
                    logging.error(f"Error getting prediction from {model_name}: {e}")
                    continue

            if not model_predictions:
                raise ValueError("No valid model predictions available")

            # Calculate ensemble prediction based on method
            if self.ensemble_method == "weighted_average":
                return self._weighted_average_predict(model_predictions, model_confidences)
            else:
                return self._vote_based_predict(model_predictions, model_confidences, data)

        except Exception as e:
            logging.error(f"Error in ensemble prediction: {e}")
            raise

    def _weighted_average_predict(self, model_predictions: Dict[str, np.ndarray], 
                                model_confidences: Dict[str, float]) -> np.ndarray:
        """Generate weighted average ensemble predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_confidences: Dictionary of model confidence scores
            
        Returns:
            Weighted average predictions
        """
        try:
            # Get current weights
            current_weights = self.weights.copy()
            
            # Adjust weights by confidence
            adjusted_weights = {}
            total_weight = 0
            
            for model_name in model_predictions.keys():
                base_weight = current_weights.get(model_name, 1.0 / len(model_predictions))
                confidence = model_confidences.get(model_name, 1.0)
                adjusted_weight = base_weight * confidence
                adjusted_weights[model_name] = adjusted_weight
                total_weight += adjusted_weight
            
            # Normalize weights
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            # Calculate weighted average
            weighted_pred = np.zeros_like(list(model_predictions.values())[0])
            
            for model_name, pred in model_predictions.items():
                weight = adjusted_weights.get(model_name, 0)
                weighted_pred += weight * pred
            
            logging.info(f"Weighted average ensemble prediction completed using {len(model_predictions)} models")
            return weighted_pred
            
        except Exception as e:
            logging.error(f"Error in weighted average prediction: {e}")
            # Fallback to simple average
            return np.mean(list(model_predictions.values()), axis=0)

    def _vote_based_predict(self, model_predictions: Dict[str, np.ndarray], 
                          model_confidences: Dict[str, float], 
                          data: pd.DataFrame) -> np.ndarray:
        """Generate vote-based ensemble predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_confidences: Dictionary of model confidence scores
            data: Input data for context
            
        Returns:
            Vote-based predictions
        """
        try:
            # Convert predictions to directional signals
            directional_signals = {}
            
            for model_name, pred in model_predictions.items():
                # Calculate direction (positive = bullish, negative = bearish)
                if len(pred) > 1:
                    # For multi-step predictions, use the first step
                    direction = 1 if pred[0] > 0 else -1
                else:
                    direction = 1 if pred[0] > 0 else -1
                
                # Weight by confidence
                confidence = model_confidences.get(model_name, 1.0)
                weighted_direction = direction * confidence
                
                directional_signals[model_name] = weighted_direction
            
            # Calculate weighted vote
            total_vote = sum(directional_signals.values())
            
            # Determine final direction
            if total_vote > 0:
                final_direction = 1  # Bullish
            elif total_vote < 0:
                final_direction = -1  # Bearish
            else:
                final_direction = 0  # Neutral
            
            # Generate final prediction based on direction and magnitude
            avg_magnitude = np.mean([np.abs(pred[0]) for pred in model_predictions.values()])
            final_prediction = final_direction * avg_magnitude
            
            # Repeat for multi-step predictions if needed
            if len(list(model_predictions.values())[0]) > 1:
                final_prediction = np.full(len(list(model_predictions.values())[0]), final_prediction)
            
            logging.info(f"Vote-based ensemble prediction completed. Direction: {final_direction}, "
                        f"Confidence: {abs(total_vote) / len(model_predictions):.2f}")
            return final_prediction
            
        except Exception as e:
            logging.error(f"Error in vote-based prediction: {e}")
            # Fallback to simple average
            return np.mean(list(model_predictions.values()), axis=0)

    def save_model(self, filepath: str):
        """Save ensemble model.

        Args:
            filepath: Path to save model
        """
        model_dir = os.path.dirname(filepath)
        try:
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create model_dir: {e}")

        # Save ensemble metadata
        metadata = {
            "config": self.config,
            "weights": self.weights,
            "performance_history": self.performance_history,
            "strategy_patterns": self.strategy_patterns,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            model.save_model(model_path)

    def load_model(self, filepath: str):
        """Load ensemble model.

        Args:
            filepath: Path to saved model
        """
        # Load ensemble metadata
        with open(filepath, "r") as f:
            metadata = json.load(f)

        self.config = metadata["config"]
        self.weights = metadata["weights"]
        self.performance_history = metadata["performance_history"]
        self.strategy_patterns = metadata["strategy_patterns"]

        # Initialize and load individual models
        self._initialize_models()
        model_dir = os.path.dirname(filepath)

        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            model.load_model(model_path)

    def summary(self) -> str:
        """Get model summary.

        Returns:
            Formatted summary string
        """
        summary = ["Ensemble Model Summary", "=" * 20]

        # Add model configurations
        summary.append("\nModels:")
        for model_name, model in self.models.items():
            summary.append(f"- {model_name}: {model.summary()}")

        # Add current weights
        summary.append("\nCurrent Weights:")
        for model_name, weight in self.weights.items():
            summary.append(f"- {model_name}: {weight:.3f}")

        # Add strategy patterns
        summary.append("\nStrategy Patterns:")
        for regime, pattern in self.strategy_patterns.items():
            summary.append(f"- {regime}: {pattern['best_model']}")

        return "\n".join(summary)

    def shap_interpret(self, data: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for ensemble predictions.

        Args:
            data: Input data

        Returns:
            SHAP values
        """
        # Get SHAP values from each model
        shap_values = {}
        for model_name, model in self.models.items():
            if hasattr(model, "shap_interpret"):
                try:
                    shap_values[model_name] = model.shap_interpret(data)
                except Exception as e:
                    logging.error(f"Error getting SHAP values for {model_name}: {e}")
                    raise RuntimeError(
                        f"Failed to get SHAP values for {model_name}: {e}"
                    )

        if not shap_values:
            raise ValueError("No models support SHAP interpretation")

        # Weight SHAP values by model weights
        weighted_shap = np.zeros_like(next(iter(shap_values.values())))
        for model_name, values in shap_values.items():
            weighted_shap += self.weights[model_name] * values

        return weighted_shap

    @cache_model_operation
    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.

        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast

        Returns:
            Dictionary containing forecast results
        """
        try:
            # Make initial prediction
            self.predict(data)

            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()

            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])

                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row["close"] = pred[-1]  # Update with prediction
                current_data = pd.concat(
                    [current_data, pd.DataFrame([new_row])], ignore_index=True
                )
                current_data = current_data.iloc[1:]  # Remove oldest row

            return {
                "forecast": np.array(forecast_values),
                "confidence": 0.9,  # High confidence for ensemble
                "model": "Ensemble",
                "horizon": horizon,
                "weights": self.weights,
                "strategy_patterns": self.strategy_patterns,
            }

        except Exception as e:
            logging.warning(f"Error in ensemble model forecast: {e}. Returning fallback DataFrame.")
            
            # Return fallback DataFrame with required columns
            fallback_df = pd.DataFrame({
                'Forecast': [],
                'Confidence': []
            })
            
            return {
                "forecast": np.array([]),
                "confidence": 0.0,
                "model": "Ensemble_Fallback",
                "horizon": horizon,
                "weights": {},
                "strategy_patterns": {},
                "fallback_dataframe": fallback_df,
                "error": str(e)
            }

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot ensemble model results and predictions.

        Args:
            data: Input data DataFrame
            predictions: Optional predictions to plot
        """
        try:
            import matplotlib.pyplot as plt

            if predictions is None:
                predictions = self.predict(data)

            plt.figure(figsize=(15, 10))

            # Plot 1: Historical vs Predicted
            plt.subplot(2, 2, 1)
            plt.plot(data.index, data["close"], label="Actual", color="blue")
            plt.plot(
                data.index[-len(predictions) :],
                predictions,
                label="Predicted",
                color="red",
            )
            plt.title("Ensemble Model Predictions")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            # Plot 2: Model Weights
            plt.subplot(2, 2, 2)
            model_names = list(self.weights.keys())
            weights = list(self.weights.values())
            plt.bar(model_names, weights)
            plt.title("Model Weights")
            plt.xlabel("Models")
            plt.ylabel("Weight")
            plt.xticks(rotation=45)
            plt.grid(True)

            # Plot 3: Performance History
            plt.subplot(2, 2, 3)
            for model_name, history in self.performance_history.items():
                if history:
                    scores = [h["score"] for h in history]
                    plt.plot(scores, label=model_name)
            plt.title("Model Performance History")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)

            # Plot 4: Strategy Patterns
            plt.subplot(2, 2, 4)
            regimes = list(self.strategy_patterns.keys())
            best_models = [self.strategy_patterns[r]["best_model"] for r in regimes]
            plt.bar(regimes, [1] * len(regimes))
            plt.title("Strategy Patterns by Regime")
            plt.xlabel("Market Regime")
            plt.ylabel("Count")
            for i, (regime, model) in enumerate(zip(regimes, best_models)):
                plt.text(i, 0.5, model, ha="center", va="center")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error plotting ensemble results: {e}")
            logger.error(f"Could not plot results: {e}")
