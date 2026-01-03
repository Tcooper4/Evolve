"""
Forecast Dispatcher

This module provides intelligent forecast dispatching with fallback mechanisms,
confidence intervals, and consensus checking across multiple models.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.agents.base_agent_interface import AgentConfig, AgentResult, BaseAgent
from trading.models.arima_model import ARIMAModel
from trading.models.ensemble_model import EnsembleModel as EnsembleForecaster
from trading.models.lstm_model import LSTMForecaster
from trading.models.prophet_model import ProphetForecaster
from trading.models.xgboost_model import XGBoostForecaster


@dataclass
class ForecastResult:
    """Forecast result with metadata"""

    model_name: str
    forecast_values: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    confidence_level: float = 0.95
    model_metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    is_fallback: bool = False


@dataclass
class ConsensusResult:
    """Consensus analysis result"""

    agreement_level: float
    conflicting_models: List[str]
    consensus_forecast: np.ndarray
    consensus_confidence: float
    model_weights: Dict[str, float]


class ForecastDispatcher(BaseAgent):
    """Agent for dispatching forecasts with intelligent fallback and consensus checking."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig())
        self.logger = logging.getLogger(__name__)

        # Model registry
        self.models = {
            "lstm": LSTMForecaster(),
            "xgboost": XGBoostForecaster(),
            "prophet": ProphetForecaster(),
            "arima": ARIMAModel(),
            "ensemble": EnsembleForecaster(),
        }

        # Fallback configuration
        self.fallback_config = {
            "enabled": True,
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "fallback_models": ["ensemble", "xgboost", "lstm"],
            "nan_threshold": 0.5,  # Maximum fraction of NaN values allowed
        }

        # Consensus configuration
        self.consensus_config = {
            "enabled": True,
            "agreement_threshold": 0.7,
            "confidence_weight": 0.3,
            "performance_weight": 0.7,
            "min_models_for_consensus": 2,
        }

        # Performance tracking
        self.model_performance = {}
        self.last_working_model = None
        self.performance_file = Path("logs/forecast_performance.json")
        self._load_performance_history()

        # Confidence interval settings
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1Ïƒ, 2Ïƒ, 3Ïƒ

    def _load_performance_history(self) -> None:
        """Load model performance history."""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, "r") as f:
                    self.model_performance = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load performance history: {e}")
            self.model_performance = {}

    def _save_performance_history(self) -> None:
        """Save model performance history."""
        try:
            self.performance_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.performance_file, "w") as f:
                json.dump(self.model_performance, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")

    async def execute(self, **kwargs) -> AgentResult:
        """Execute forecast dispatching with fallback and consensus checking."""
        try:
            # Extract parameters
            data = kwargs.get("data")
            target_column = kwargs.get("target_column", "close")
            horizon = kwargs.get("horizon", 30)
            models_to_use = kwargs.get("models", list(self.models.keys()))
            enable_consensus = kwargs.get("consensus", self.consensus_config["enabled"])

            if data is None or data.empty:
                return AgentResult(
                    success=False, error_message="No data provided for forecasting"
                )

            # Execute forecasts
            forecast_results = await self._execute_forecasts(
                data, target_column, horizon, models_to_use
            )

            # Check for consensus if enabled
            consensus_result = None
            if (
                enable_consensus
                and len(forecast_results)
                >= self.consensus_config["min_models_for_consensus"]
            ):
                consensus_result = self._check_consensus(forecast_results)

            # Select best forecast
            best_forecast = self._select_best_forecast(
                forecast_results, consensus_result
            )

            # Update performance tracking
            self._update_performance_tracking(forecast_results)

            return AgentResult(
                success=True,
                data={
                    "forecast": best_forecast.forecast_values.tolist(),
                    "confidence_intervals": best_forecast.confidence_intervals,
                    "model_name": best_forecast.model_name,
                    "is_fallback": best_forecast.is_fallback,
                    "consensus_result": consensus_result,
                    "all_forecasts": {
                        result.model_name: {
                            "forecast": result.forecast_values.tolist(),
                            "confidence_intervals": result.confidence_intervals,
                            "execution_time": result.execution_time,
                            "is_fallback": result.is_fallback,
                        }
                        for result in forecast_results
                    },
                },
                extra_metrics={
                    "models_used": len(forecast_results),
                    "consensus_agreement": (
                        consensus_result.agreement_level if consensus_result else None
                    ),
                    "fallback_used": any(r.is_fallback for r in forecast_results),
                },
            )

        except Exception as e:
            self.logger.error(f"Forecast dispatching failed: {e}")
            return AgentResult(success=False, error_message=str(e))

    async def _execute_forecasts(
        self,
        data: pd.DataFrame,
        target_column: str,
        horizon: int,
        models_to_use: List[str],
    ) -> List[ForecastResult]:
        """Execute forecasts using multiple models with fallback."""
        results = []

        for model_name in models_to_use:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found, skipping")
                continue

            try:
                model = self.models[model_name]
                start_time = datetime.now()

                # Execute forecast
                forecast_result = await model.forecast(
                    data=data, target_column=target_column, horizon=horizon
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                if forecast_result.success:
                    # Extract forecast values
                    forecast_values = np.array(forecast_result.data.get("forecast", []))

                    # Check for NaN values
                    nan_fraction = np.isnan(forecast_values).sum() / len(
                        forecast_values
                    )

                    if nan_fraction <= self.fallback_config["nan_threshold"]:
                        # Valid forecast
                        result = ForecastResult(
                            model_name=model_name,
                            forecast_values=forecast_values,
                            confidence_intervals=forecast_result.data.get(
                                "confidence_intervals"
                            ),
                            confidence_level=0.95,
                            model_metadata=forecast_result.data.get("metadata", {}),
                            execution_time=execution_time,
                            is_fallback=False,
                        )
                        results.append(result)
                        self.last_working_model = model_name

                        self.logger.info(
                            f"Model {model_name} forecast completed successfully "
                            f"in {execution_time:.3f}s"
                        )
                    else:
                        self.logger.warning(
                            f"Model {model_name} returned {nan_fraction:.1%} NaN values, "
                            f"using fallback"
                        )
                        # Try fallback
                        fallback_result = await self._try_fallback(
                            data, target_column, horizon, model_name
                        )
                        if fallback_result:
                            results.append(fallback_result)
                else:
                    self.logger.warning(
                        f"Model {model_name} failed: {forecast_result.error_message}"
                    )
                    # Try fallback
                    fallback_result = await self._try_fallback(
                        data, target_column, horizon, model_name
                    )
                    if fallback_result:
                        results.append(fallback_result)

            except Exception as e:
                self.logger.error(f"Error executing {model_name} forecast: {e}")
                # Try fallback
                fallback_result = await self._try_fallback(
                    data, target_column, horizon, model_name
                )
                if fallback_result:
                    results.append(fallback_result)

        return results

    async def _try_fallback(
        self, data: pd.DataFrame, target_column: str, horizon: int, failed_model: str
    ) -> Optional[ForecastResult]:
        """Try fallback models when primary model fails."""
        if not self.fallback_config["enabled"]:
            return None

        # Use last working model or fallback list
        fallback_models = [self.last_working_model] if self.last_working_model else []
        fallback_models.extend(self.fallback_config["fallback_models"])

        # Remove the failed model and duplicates
        fallback_models = list(
            dict.fromkeys(
                [
                    m
                    for m in fallback_models
                    if m and m != failed_model and m in self.models
                ]
            )
        )

        for fallback_model in fallback_models:
            try:
                model = self.models[fallback_model]
                start_time = datetime.now()

                forecast_result = await model.forecast(
                    data=data, target_column=target_column, horizon=horizon
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                if forecast_result.success:
                    forecast_values = np.array(forecast_result.data.get("forecast", []))

                    # Check for NaN values
                    nan_fraction = np.isnan(forecast_values).sum() / len(
                        forecast_values
                    )

                    if nan_fraction <= self.fallback_config["nan_threshold"]:
                        result = ForecastResult(
                            model_name=fallback_model,
                            forecast_values=forecast_values,
                            confidence_intervals=forecast_result.data.get(
                                "confidence_intervals"
                            ),
                            confidence_level=0.95,
                            model_metadata=forecast_result.data.get("metadata", {}),
                            execution_time=execution_time,
                            is_fallback=True,
                        )

                        self.logger.info(
                            f"Fallback to {fallback_model} successful "
                            f"after {failed_model} failed"
                        )

                        return result

            except Exception as e:
                self.logger.error(f"Fallback model {fallback_model} also failed: {e}")

        return None

    def _check_consensus(
        self, forecast_results: List[ForecastResult]
    ) -> ConsensusResult:
        """Check consensus among multiple forecasts."""
        if len(forecast_results) < 2:
            return None

        # Calculate agreement level
        forecasts = [result.forecast_values for result in forecast_results]
        [result.model_name for result in forecast_results]

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(forecasts)):
            for j in range(i + 1, len(forecasts)):
                if len(forecasts[i]) == len(forecasts[j]):
                    corr = np.corrcoef(forecasts[i], forecasts[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        agreement_level = np.mean(correlations) if correlations else 0.0

        # Identify conflicting models
        conflicting_models = []
        if agreement_level < self.consensus_config["agreement_threshold"]:
            # Find models that disagree most
            for i, result in enumerate(forecast_results):
                other_forecasts = [f for j, f in enumerate(forecasts) if j != i]
                avg_corr = np.mean(
                    [
                        np.corrcoef(result.forecast_values, other)[0, 1]
                        for other in other_forecasts
                        if len(result.forecast_values) == len(other)
                    ]
                )
                if avg_corr < self.consensus_config["agreement_threshold"]:
                    conflicting_models.append(result.model_name)

        # Calculate consensus forecast (weighted average)
        weights = self._calculate_model_weights(forecast_results)
        consensus_forecast = np.zeros_like(forecast_results[0].forecast_values)

        for result in forecast_results:
            weight = weights.get(result.model_name, 1.0 / len(forecast_results))
            consensus_forecast += weight * result.forecast_values

        # Calculate consensus confidence
        consensus_confidence = agreement_level * self.consensus_config[
            "confidence_weight"
        ] + (1 - len(conflicting_models) / len(forecast_results)) * (
            1 - self.consensus_config["confidence_weight"]
        )

        return ConsensusResult(
            agreement_level=agreement_level,
            conflicting_models=conflicting_models,
            consensus_forecast=consensus_forecast,
            consensus_confidence=consensus_confidence,
            model_weights=weights,
        )

    def _calculate_model_weights(
        self, forecast_results: List[ForecastResult]
    ) -> Dict[str, float]:
        """Calculate weights for models based on performance and confidence."""
        weights = {}
        total_weight = 0.0

        for result in forecast_results:
            # Performance weight
            performance_score = self.model_performance.get(result.model_name, {}).get(
                "score", 0.5
            )

            # Confidence weight (based on confidence intervals if available)
            confidence_score = 0.5
            if result.confidence_intervals:
                # Calculate average confidence interval width
                lower, upper = result.confidence_intervals
                avg_width = np.mean(upper - lower)
                confidence_score = 1.0 / (
                    1.0 + avg_width
                )  # Narrower intervals = higher confidence

            # Combined weight
            weight = (
                performance_score * self.consensus_config["performance_weight"]
                + confidence_score * self.consensus_config["confidence_weight"]
            )

            weights[result.model_name] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight

        return weights

    def _select_best_forecast(
        self,
        forecast_results: List[ForecastResult],
        consensus_result: Optional[ConsensusResult],
    ) -> ForecastResult:
        """Select the best forecast from available results."""
        if not forecast_results:
            raise ValueError("No forecast results available")

        # If consensus is available and has high agreement, use it
        if (
            consensus_result
            and consensus_result.agreement_level
            > self.consensus_config["agreement_threshold"]
        ):
            # Create consensus result
            best_result = ForecastResult(
                model_name="consensus",
                forecast_values=consensus_result.consensus_forecast,
                confidence_level=consensus_result.consensus_confidence,
                model_metadata={"agreement_level": consensus_result.agreement_level},
                is_fallback=False,
            )

            self.logger.info(
                f"Using consensus forecast with {consensus_result.agreement_level:.2f} agreement"
            )

            return best_result

        # Otherwise, select best individual model
        best_result = max(
            forecast_results,
            key=lambda r: self.model_performance.get(r.model_name, {}).get(
                "score", 0.5
            ),
        )

        self.logger.info(f"Selected best individual model: {best_result.model_name}")

        return best_result

    def _update_performance_tracking(
        self, forecast_results: List[ForecastResult]
    ) -> None:
        """Update model performance tracking."""
        for result in forecast_results:
            if result.model_name not in self.model_performance:
                self.model_performance[result.model_name] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "avg_execution_time": 0.0,
                    "score": 0.5,
                }

            perf = self.model_performance[result.model_name]
            perf["total_runs"] += 1

            if not result.is_fallback:
                perf["successful_runs"] += 1

            # Update average execution time
            perf["avg_execution_time"] = (
                perf["avg_execution_time"] * (perf["total_runs"] - 1)
                + result.execution_time
            ) / perf["total_runs"]

            # Update score (success rate + performance factor)
            success_rate = perf["successful_runs"] / perf["total_runs"]
            performance_factor = 1.0 / (1.0 + perf["avg_execution_time"])
            perf["score"] = 0.7 * success_rate + 0.3 * performance_factor

        self._save_performance_history()

    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance statistics."""
        return self.model_performance.copy()

    def get_last_working_model(self) -> Optional[str]:
        """Get the last working model name."""
        return self.last_working_model

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking."""
        self.model_performance = {}
        self.last_working_model = None
        self._save_performance_history()
