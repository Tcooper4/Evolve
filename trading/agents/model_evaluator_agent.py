"""
Model Evaluator Agent for Trading System

This agent evaluates model performance using various metrics and provides
comprehensive analysis of model effectiveness for trading decisions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from trading.agents.base_agent_interface import AgentConfig, AgentResult, BaseAgent
from trading.memory.agent_memory import AgentMemory
from trading.models.model_registry import ModelRegistry
from trading.utils.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics."""

    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


class ModelStatus(Enum):
    """Model evaluation status."""

    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class EvaluationResult:
    """Result of model evaluation."""

    evaluation_id: str
    model_id: str
    model_type: str
    evaluation_timestamp: datetime
    metrics: Dict[str, float]
    status: ModelStatus
    confidence_score: float
    recommendations: List[str]
    performance_summary: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance data."""

    model_id: str
    predictions: np.ndarray
    actual_values: np.ndarray
    timestamps: List[datetime]
    confidence_scores: Optional[np.ndarray] = None


@dataclass
class ModelEvaluationRequest:
    """Request for model evaluation."""

    model_id: str
    performance_data: Dict[str, Any]
    evaluation_metrics: Optional[List[str]] = None
    evaluation_window: Optional[int] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelEvaluationResult:
    """Result of model evaluation request."""

    request: ModelEvaluationRequest
    evaluation_result: EvaluationResult
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0


class ModelEvaluatorAgent(BaseAgent):
    """
    Model Evaluator Agent that:
    - Evaluates model performance using multiple metrics
    - Provides comprehensive analysis and recommendations
    - Tracks evaluation history and trends
    - Identifies model degradation and improvement opportunities
    - Generates performance reports and alerts
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ModelEvaluatorAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={},
            )
        super().__init__(config)

        self.config_dict = config.custom_config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.model_registry = ModelRegistry()

        # Configuration
        self.evaluation_window = self.config_dict.get("evaluation_window", 30)
        self.min_data_points = self.config_dict.get("min_data_points", 100)
        self.performance_thresholds = self.config_dict.get(
            "performance_thresholds",
            {"excellent": 0.8, "good": 0.6, "average": 0.4, "poor": 0.2},
        )

        # Storage
        self.evaluation_history: List[EvaluationResult] = []
        self.model_performance_cache: Dict[str, ModelPerformance] = {}
        self.current_evaluation_id = None

        # Load existing data
        self._load_evaluation_history()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model evaluation logic.
        Args:
            **kwargs: action, model_id, performance_data, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get("action", "evaluate_model")

            if action == "evaluate_model":
                model_id = kwargs.get("model_id")
                performance_data = kwargs.get("performance_data")

                if not model_id or not performance_data:
                    return AgentResult(
                        success=False,
                        error_message="Missing model_id or performance_data",
                    )

                result = await self.evaluate_model(model_id, performance_data)
                return AgentResult(
                    success=True,
                    data={
                        "evaluation_result": result.__dict__,
                        "status": result.status.value,
                        "confidence_score": result.confidence_score,
                    },
                )

            elif action == "get_evaluation_history":
                model_id = kwargs.get("model_id")
                history = self.get_evaluation_history(model_id)
                return AgentResult(
                    success=True,
                    data={"evaluation_history": [eval.__dict__ for eval in history]},
                )

            elif action == "get_model_status":
                model_id = kwargs.get("model_id")
                if not model_id:
                    return AgentResult(success=False, error_message="Missing model_id")

                status = self.get_model_status(model_id)
                return AgentResult(success=True, data={"model_status": status})

            elif action == "compare_models":
                model_ids = kwargs.get("model_ids", [])
                if not model_ids:
                    return AgentResult(success=False, error_message="Missing model_ids")

                comparison = self.compare_models(model_ids)
                return AgentResult(success=True, data={"model_comparison": comparison})

            elif action == "generate_report":
                model_id = kwargs.get("model_id")
                report = self.generate_evaluation_report(model_id)
                return AgentResult(success=True, data={"evaluation_report": report})

            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

        except Exception as e:
            return self.handle_error(e)

    async def evaluate_model(
        self, model_id: str, performance_data: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate a model's performance.

        Args:
            model_id: ID of the model to evaluate
            performance_data: Performance data including predictions and actual values

        Returns:
            Evaluation result
        """
        try:
            evaluation_id = (
                f"eval_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.current_evaluation_id = evaluation_id

            self.logger.info(f"Starting evaluation for model {model_id}")

            # Create performance object
            performance = ModelPerformance(
                model_id=model_id,
                predictions=np.array(performance_data.get("predictions", [])),
                actual_values=np.array(performance_data.get("actual_values", [])),
                timestamps=performance_data.get("timestamps", []),
                confidence_scores=np.array(
                    performance_data.get("confidence_scores", [])
                )
                if performance_data.get("confidence_scores")
                else None,
            )

            # Validate data
            if len(performance.predictions) < self.min_data_points:
                raise ValueError(
                    f"Insufficient data points: {len(performance.predictions)} < {self.min_data_points}"
                )

            # Calculate metrics
            metrics = self._calculate_metrics(performance)

            # Determine status
            status = self._determine_model_status(metrics)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(metrics, performance)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, status)

            # Create performance summary
            performance_summary = self._create_performance_summary(performance, metrics)

            # Create evaluation result
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                model_id=model_id,
                model_type=performance_data.get("model_type", "unknown"),
                evaluation_timestamp=datetime.now(),
                metrics=metrics,
                status=status,
                confidence_score=confidence_score,
                recommendations=recommendations,
                performance_summary=performance_summary,
            )

            # Store result
            self.evaluation_history.append(result)
            self.model_performance_cache[model_id] = performance

            # Store in memory
            self._store_evaluation_result(result)

            self.logger.info(
                f"Completed evaluation for model {model_id}: {status.value}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating model {model_id}: {str(e)}")

            return EvaluationResult(
                evaluation_id=evaluation_id
                if "evaluation_id" in locals()
                else f"eval_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                model_type=performance_data.get("model_type", "unknown"),
                evaluation_timestamp=datetime.now(),
                metrics={},
                status=ModelStatus.FAILED,
                confidence_score=0.0,
                recommendations=["Evaluation failed"],
                performance_summary={},
                error_message=str(e),
            )

    def _calculate_metrics(self, performance: ModelPerformance) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            predictions = performance.predictions
            actual_values = performance.actual_values

            # Basic regression metrics
            mse = np.mean((predictions - actual_values) ** 2)
            mae = np.mean(np.abs(predictions - actual_values))
            rmse = np.sqrt(mse)

            # Percentage error
            mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100

            # Trading-specific metrics
            returns = np.diff(actual_values) / actual_values[:-1]
            predicted_returns = np.diff(predictions) / predictions[:-1]

            # Sharpe ratio
            sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0.0

            # Maximum drawdown
            max_drawdown = calculate_max_drawdown(returns) if len(returns) > 1 else 0.0

            # Win rate
            correct_directions = np.sum(np.sign(returns) == np.sign(predicted_returns))
            win_rate = correct_directions / len(returns) if len(returns) > 0 else 0.0

            # Profit factor
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            profit_factor = (
                np.sum(positive_returns) / abs(np.sum(negative_returns))
                if len(negative_returns) > 0 and np.sum(negative_returns) != 0
                else 0.0
            )

            # Calmar ratio
            calmar_ratio = np.mean(returns) / max_drawdown if max_drawdown > 0 else 0.0

            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = (
                np.std(downside_returns) if len(downside_returns) > 0 else 1.0
            )
            sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0.0

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "calmar_ratio": calmar_ratio,
                "sortino_ratio": sortino_ratio,
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _determine_model_status(self, metrics: Dict[str, float]) -> ModelStatus:
        """Determine model status based on metrics."""
        try:
            if not metrics:
                return ModelStatus.FAILED

            # Calculate composite score
            score = 0.0
            count = 0

            # Normalize and weight metrics
            if "sharpe_ratio" in metrics:
                sharpe_norm = min(1.0, max(0.0, metrics["sharpe_ratio"] / 2.0))
                score += 0.3 * sharpe_norm
                count += 1

            if "win_rate" in metrics:
                score += 0.2 * metrics["win_rate"]
                count += 1

            if "profit_factor" in metrics:
                profit_norm = min(1.0, metrics["profit_factor"] / 2.0)
                score += 0.2 * profit_norm
                count += 1

            if "mape" in metrics:
                mape_norm = max(0.0, 1.0 - metrics["mape"] / 100.0)
                score += 0.15 * mape_norm
                count += 1

            if "max_drawdown" in metrics:
                drawdown_norm = max(0.0, 1.0 - abs(metrics["max_drawdown"]))
                score += 0.15 * drawdown_norm
                count += 1

            if count > 0:
                final_score = score / count
            else:
                final_score = 0.0

            # Determine status
            if final_score >= self.performance_thresholds["excellent"]:
                return ModelStatus.EXCELLENT
            elif final_score >= self.performance_thresholds["good"]:
                return ModelStatus.GOOD
            elif final_score >= self.performance_thresholds["average"]:
                return ModelStatus.AVERAGE
            elif final_score >= self.performance_thresholds["poor"]:
                return ModelStatus.POOR
            else:
                return ModelStatus.FAILED

        except Exception as e:
            self.logger.error(f"Error determining model status: {str(e)}")
            return ModelStatus.FAILED

    def _calculate_confidence_score(
        self, metrics: Dict[str, Any], performance: ModelPerformance
    ) -> float:
        """Calculate confidence score for the evaluation."""
        try:
            confidence_score = 0.5  # Base confidence

            # Data quality factors
            if len(performance.predictions) >= 1000:
                confidence_score += 0.2
            elif len(performance.predictions) >= 500:
                confidence_score += 0.1

            # Metric consistency
            if metrics.get("sharpe_ratio", 0) > 0.5:
                confidence_score += 0.1

            if metrics.get("win_rate", 0) > 0.6:
                confidence_score += 0.1

            if metrics.get("profit_factor", 0) > 1.5:
                confidence_score += 0.1

            return min(1.0, confidence_score)

        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5

    def _generate_recommendations(
        self, metrics: Dict[str, float], status: ModelStatus
    ) -> List[str]:
        """Generate recommendations based on evaluation."""
        try:
            recommendations = []

            if status == ModelStatus.EXCELLENT:
                recommendations.append(
                    "Model performing excellently - consider increasing position sizes"
                )
                recommendations.append("Monitor for potential overfitting")
            elif status == ModelStatus.GOOD:
                recommendations.append("Model performing well - continue monitoring")
                recommendations.append("Consider fine-tuning for further improvement")
            elif status == ModelStatus.AVERAGE:
                recommendations.append(
                    "Model performance is average - consider retraining"
                )
                recommendations.append("Review feature engineering and hyperparameters")
            elif status == ModelStatus.POOR:
                recommendations.append(
                    "Model performance is poor - immediate retraining recommended"
                )
                recommendations.append("Consider alternative model architectures")
            elif status == ModelStatus.FAILED:
                recommendations.append("Model has failed - replace immediately")
                recommendations.append("Conduct thorough analysis of failure causes")

            # Specific recommendations based on metrics
            if metrics.get("sharpe_ratio", 0) < 0.5:
                recommendations.append(
                    "Low Sharpe ratio - optimize for risk-adjusted returns"
                )

            if metrics.get("max_drawdown", 0) > 0.2:
                recommendations.append(
                    "High drawdown - implement better risk management"
                )

            if metrics.get("win_rate", 0) < 0.5:
                recommendations.append("Low win rate - review prediction accuracy")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations"]

    def _create_performance_summary(
        self, performance: ModelPerformance, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create performance summary."""
        try:
            return {
                "data_points": len(performance.predictions),
                "evaluation_period": {
                    "start": performance.timestamps[0]
                    if performance.timestamps
                    else None,
                    "end": performance.timestamps[-1]
                    if performance.timestamps
                    else None,
                },
                "prediction_range": {
                    "min": float(np.min(performance.predictions)),
                    "max": float(np.max(performance.predictions)),
                    "mean": float(np.mean(performance.predictions)),
                },
                "actual_range": {
                    "min": float(np.min(performance.actual_values)),
                    "max": float(np.max(performance.actual_values)),
                    "mean": float(np.mean(performance.actual_values)),
                },
                "metrics_summary": metrics,
            }

        except Exception as e:
            self.logger.error(f"Error creating performance summary: {str(e)}")
            return {}

    def get_evaluation_history(
        self, model_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Get evaluation history for a model or all models."""
        if model_id:
            return [
                eval for eval in self.evaluation_history if eval.model_id == model_id
            ]
        return self.evaluation_history.copy()

    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a model."""
        try:
            # Get latest evaluation
            model_evaluations = [
                eval for eval in self.evaluation_history if eval.model_id == model_id
            ]
            if not model_evaluations:
                return None

            latest_eval = max(model_evaluations, key=lambda x: x.evaluation_timestamp)

            return {
                "model_id": model_id,
                "status": latest_eval.status.value,
                "confidence_score": latest_eval.confidence_score,
                "last_evaluation": latest_eval.evaluation_timestamp.isoformat(),
                "recommendations": latest_eval.recommendations,
                "key_metrics": {
                    "sharpe_ratio": latest_eval.metrics.get("sharpe_ratio", 0),
                    "win_rate": latest_eval.metrics.get("win_rate", 0),
                    "max_drawdown": latest_eval.metrics.get("max_drawdown", 0),
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting model status: {str(e)}")
            return None

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models."""
        try:
            comparison = {}

            for model_id in model_ids:
                status = self.get_model_status(model_id)
                if status:
                    comparison[model_id] = status

            # Add ranking
            if comparison:
                ranked_models = sorted(
                    comparison.items(),
                    key=lambda x: x[1]["confidence_score"],
                    reverse=True,
                )
                comparison["ranking"] = [model_id for model_id, _ in ranked_models]

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {}

    def generate_evaluation_report(
        self, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        try:
            if model_id:
                evaluations = [
                    eval
                    for eval in self.evaluation_history
                    if eval.model_id == model_id
                ]
            else:
                evaluations = self.evaluation_history

            if not evaluations:
                return {"error": "No evaluations found"}

            # Group by model
            model_evaluations = {}
            for eval in evaluations:
                if eval.model_id not in model_evaluations:
                    model_evaluations[eval.model_id] = []
                model_evaluations[eval.model_id].append(eval)

            report = {
                "generated_at": datetime.now().isoformat(),
                "total_evaluations": len(evaluations),
                "models_evaluated": len(model_evaluations),
                "model_summaries": {},
            }

            for model_id, model_evals in model_evaluations.items():
                latest_eval = max(model_evals, key=lambda x: x.evaluation_timestamp)

                report["model_summaries"][model_id] = {
                    "latest_status": latest_eval.status.value,
                    "latest_confidence": latest_eval.confidence_score,
                    "total_evaluations": len(model_evals),
                    "performance_trend": self._calculate_performance_trend(model_evals),
                    "latest_metrics": latest_eval.metrics,
                    "recommendations": latest_eval.recommendations,
                }

            return report

        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return {"error": str(e)}

    def _calculate_performance_trend(self, evaluations: List[EvaluationResult]) -> str:
        """Calculate performance trend from evaluations."""
        try:
            if len(evaluations) < 2:
                return "insufficient_data"

            # Sort by timestamp
            sorted_evals = sorted(evaluations, key=lambda x: x.evaluation_timestamp)

            # Compare latest vs previous
            latest = sorted_evals[-1]
            previous = sorted_evals[-2]

            if latest.confidence_score > previous.confidence_score * 1.1:
                return "improving"
            elif latest.confidence_score < previous.confidence_score * 0.9:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            self.logger.error(f"Error calculating performance trend: {str(e)}")
            return "unknown"

    def _store_evaluation_result(self, result: EvaluationResult):
        """Store evaluation result in memory."""
        try:
            self.memory.store(
                f"evaluation_{result.evaluation_id}",
                {"result": result.__dict__, "timestamp": datetime.now()},
            )
        except Exception as e:
            self.logger.error(f"Error storing evaluation result: {str(e)}")

    def _load_evaluation_history(self):
        """Load evaluation history from memory."""
        try:
            # Load recent evaluations
            evaluation_data = self.memory.get("evaluation_history")
            if evaluation_data:
                self.evaluation_history = [
                    EvaluationResult(**eval) for eval in evaluation_data
                ]
        except Exception as e:
            self.logger.error(f"Error loading evaluation history: {str(e)}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "status": "active",
            "last_update": datetime.now().isoformat(),
            "evaluations_completed": len(self.evaluation_history),
            "current_evaluation": self.current_evaluation_id,
            "models_evaluated": len(
                set(eval.model_id for eval in self.evaluation_history)
            ),
        }
