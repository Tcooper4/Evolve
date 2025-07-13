# -*- coding: utf-8 -*-
"""
Model Selector Agent

This agent dynamically selects the best forecasting model based on:
- Forecasting horizon (short-term, medium-term, long-term)
- Market regime (trending, mean-reverting, volatile)
- Performance metrics (accuracy, Sharpe ratio, drawdown)
- Model type (LSTM, Transformer, Prophet, etc.)

The agent maintains a registry of available models and their performance
characteristics, and uses meta-learning to improve selection over time.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class ForecastingHorizon(Enum):
    """Forecasting horizon categories."""

    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months


class MarketRegime(Enum):
    """Market regime categories."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"


class ModelType(Enum):
    """Available model types."""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    PROPHET = "prophet"
    ARIMA = "arima"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ENSEMBLE = "ensemble"
    DEEP_ENSEMBLE = "deep_ensemble"


@dataclass
class ModelPerformance:
    """Model performance metrics."""

    model_id: str
    model_type: ModelType
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    information_ratio: float
    last_updated: datetime
    training_samples: int
    inference_latency: float  # milliseconds


@dataclass
class ModelCapability:
    """Model capability profile."""

    model_id: str
    model_type: ModelType
    supported_horizons: List[ForecastingHorizon]
    supported_regimes: List[MarketRegime]
    min_data_points: int
    max_forecast_horizon: int
    feature_requirements: List[str]
    computational_complexity: str  # "low", "medium", "high"
    memory_requirements: str  # "low", "medium", "high"


class ModelSelectorAgent(BaseAgent):
    """Agent for dynamic model selection based on market conditions and requirements."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[AgentConfig] = None):
        """Initialize the Model Selector Agent."""
        if config is None:
            config = AgentConfig(
                name="ModelSelectorAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={"config_path": config_path},
            )

        super().__init__(config)

        self.config_path = config_path or config.custom_config.get("config_path", "config/model_selector_config.json")
        self.model_registry: Dict[str, Dict] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.capability_profiles: Dict[str, ModelCapability] = {}
        self.selection_history: List[Dict] = []

        # Load configuration and initialize registry
        self._load_configuration()
        self._initialize_model_registry()

        # Meta-learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.performance_window = 30  # days

    def _setup(self):
        """Setup method called during initialization."""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model selector agent.

        Args:
            **kwargs: Parameters including action, horizon, market_regime, etc.

        Returns:
            AgentResult: Result of the execution
        """
        try:
            action = kwargs.get("action", "select_model")

            if action == "select_model":
                horizon = kwargs.get("horizon")
                market_regime = kwargs.get("market_regime")
                data_length = kwargs.get("data_length", 100)
                required_features = kwargs.get("required_features", [])
                performance_weight = kwargs.get("performance_weight", 0.6)
                capability_weight = kwargs.get("capability_weight", 0.4)

                if not horizon or not market_regime:
                    return AgentResult(
                        success=False, error_message="Missing required parameters: horizon and market_regime"
                    )

                selected_model, confidence = self.select_model(
                    horizon, market_regime, data_length, required_features, performance_weight, capability_weight
                )

                return AgentResult(
                    success=True,
                    data={
                        "selected_model": selected_model,
                        "confidence": confidence,
                        "recommendations": self.get_model_recommendations(horizon, market_regime),
                    },
                )

            elif action == "update_performance":
                model_id = kwargs.get("model_id")
                performance_data = kwargs.get("performance")

                if not model_id or not performance_data:
                    return AgentResult(
                        success=False, error_message="Missing required parameters: model_id and performance"
                    )

                performance = ModelPerformance(**performance_data)
                self.update_model_performance(model_id, performance)

                return AgentResult(success=True, data={"message": f"Updated performance for model {model_id}"})

            elif action == "detect_regime":
                price_data = kwargs.get("price_data")

                if price_data is None:
                    return AgentResult(success=False, error_message="Missing required parameter: price_data")

                regime = self.detect_market_regime(price_data)

                return AgentResult(success=True, data={"market_regime": regime.value})

            elif action == "get_recommendations":
                horizon = kwargs.get("horizon")
                market_regime = kwargs.get("market_regime")
                top_k = kwargs.get("top_k", 3)

                if not horizon or not market_regime:
                    return AgentResult(
                        success=False, error_message="Missing required parameters: horizon and market_regime"
                    )

                recommendations = self.get_model_recommendations(horizon, market_regime, top_k)

                return AgentResult(success=True, data={"recommendations": recommendations})

            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")

        except Exception as e:
            return self.handle_error(e)

    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.learning_rate = config.get("learning_rate", 0.01)
                    self.exploration_rate = config.get("exploration_rate", 0.1)
                    self.performance_window = config.get("performance_window", 30)
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _initialize_model_registry(self):
        """Initialize the model registry with default models."""
        default_models = {
            "lstm_short": {
                "type": ModelType.LSTM,
                "horizons": [ForecastingHorizon.SHORT_TERM],
                "regimes": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE],
                "min_data": 100,
                "max_horizon": 7,
                "complexity": "medium",
            },
            "transformer_medium": {
                "type": ModelType.TRANSFORMER,
                "horizons": [ForecastingHorizon.MEDIUM_TERM],
                "regimes": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.MEAN_REVERTING],
                "min_data": 200,
                "max_horizon": 28,
                "complexity": "high",
            },
            "prophet_long": {
                "type": ModelType.PROPHET,
                "horizons": [ForecastingHorizon.LONG_TERM],
                "regimes": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.SIDEWAYS],
                "min_data": 365,
                "max_horizon": 365,
                "complexity": "low",
            },
            "xgboost_ensemble": {
                "type": ModelType.ENSEMBLE,
                "horizons": [ForecastingHorizon.SHORT_TERM, ForecastingHorizon.MEDIUM_TERM],
                "regimes": [MarketRegime.MEAN_REVERTING, MarketRegime.VOLATILE],
                "min_data": 150,
                "max_horizon": 14,
                "complexity": "medium",
            },
        }

        for model_id, config in default_models.items():
            self.register_model(model_id, config)

    def register_model(self, model_id: str, config: Dict[str, Any]):
        """Register a new model in the registry."""
        try:
            capability = ModelCapability(
                model_id=model_id,
                model_type=config["type"],
                supported_horizons=config["horizons"],
                supported_regimes=config["regimes"],
                min_data_points=config["min_data"],
                max_forecast_horizon=config["max_horizon"],
                feature_requirements=config.get("features", []),
                computational_complexity=config["complexity"],
                memory_requirements=config.get("memory", "medium"),
            )

            self.capability_profiles[model_id] = capability
            self.performance_history[model_id] = []

            logger.info(f"Registered model: {model_id}")

        except Exception as e:
            logger.error(f"Error registering model {model_id}: {e}")

    def update_model_performance(self, model_id: str, performance: ModelPerformance):
        """Update performance metrics for a model."""
        try:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []

            self.performance_history[model_id].append(performance)

            # Keep only recent performance data
            cutoff_date = datetime.now() - timedelta(days=self.performance_window)
            self.performance_history[model_id] = [
                p for p in self.performance_history[model_id] if p.last_updated > cutoff_date
            ]

            logger.info(f"Updated performance for model: {model_id}")

        except Exception as e:
            logger.error(f"Error updating performance for model {model_id}: {e}")

    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from price data."""
        try:
            if len(price_data) < 50:
                return MarketRegime.VOLATILE

            returns = price_data["close"].pct_change().dropna()

            # Calculate regime indicators
            volatility = returns.std() * np.sqrt(252)
            trend_strength = self._calculate_trend_strength(price_data["close"])
            mean_reversion_strength = self._calculate_mean_reversion_strength(returns)

            # Determine regime based on indicators
            if volatility > 0.3:  # High volatility
                return MarketRegime.VOLATILE
            elif trend_strength > 0.7:  # Strong trend
                if trend_strength > 0:  # Upward trend
                    return MarketRegime.TRENDING_UP
                else:  # Downward trend
                    return MarketRegime.TRENDING_DOWN
            elif mean_reversion_strength > 0.6:  # Mean reverting
                return MarketRegime.MEAN_REVERTING
            else:  # Sideways
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.VOLATILE

    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression."""
        try:
            x = np.arange(len(prices))
            y = prices.values
            slope = np.polyfit(x, y, 1)[0]
            return slope / prices.mean()  # Normalized slope
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0

    def _calculate_mean_reversion_strength(self, returns: pd.Series) -> float:
        """Calculate mean reversion strength using autocorrelation."""
        try:
            autocorr = returns.autocorr(lag=1)
            return abs(autocorr) if not pd.isna(autocorr) else 0.0
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error calculating mean reversion strength: {e}")
            return 0.0

    def select_model(
        self,
        horizon: ForecastingHorizon,
        market_regime: MarketRegime,
        data_length: int,
        required_features: List[str] = None,
        performance_weight: float = 0.6,
        capability_weight: float = 0.4,
    ) -> Tuple[str, float]:
        """
        Select the best model for given criteria.

        Args:
            horizon: Forecasting horizon
            market_regime: Market regime
            data_length: Available data length
            required_features: Required features
            performance_weight: Weight for performance score
            capability_weight: Weight for capability score

        Returns:
            Tuple of (selected_model_id, confidence_score)
        """
        try:
            # Filter candidate models
            candidates = self._filter_candidates(horizon, market_regime, data_length, required_features)

            if not candidates:
                self.logger.warning(
                    f"No models meet strict criteria for horizon={horizon.value}, regime={market_regime.value}"
                )
                # Fallback: relax criteria and try again
                candidates = self._filter_candidates_fallback(horizon, market_regime, data_length, required_features)

                if not candidates:
                    self.logger.error("No models available even with relaxed criteria")
                    return self._get_fallback_model(), 0.0

            # Calculate scores for all candidates
            model_scores = []
            for model_id in candidates:
                capability_score = self._calculate_capability_score(model_id, horizon, market_regime, data_length)
                performance_score = self._calculate_performance_score(model_id)

                # Combined score
                combined_score = capability_score * capability_weight + performance_score * performance_weight

                model_scores.append(
                    {
                        "model_id": model_id,
                        "capability_score": capability_score,
                        "performance_score": performance_score,
                        "combined_score": combined_score,
                        "model_type": self.model_registry.get(model_id, {}).get("model_type", "unknown"),
                    }
                )

            # Sort by combined score
            model_scores.sort(key=lambda x: x["combined_score"], reverse=True)

            # Log top N models with scores
            self._log_top_models(model_scores, horizon, market_regime, top_n=5)

            # Select best model
            best_model = model_scores[0]
            selected_model_id = best_model["model_id"]
            confidence = best_model["combined_score"]

            # Log selection
            self._log_selection(selected_model_id, horizon, market_regime, confidence)

            return selected_model_id, confidence

        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            return self._get_fallback_model(), 0.0
        # TODO: Specify exception type instead of using bare except

    def _filter_candidates_fallback(
        self, horizon: ForecastingHorizon, market_regime: MarketRegime, data_length: int, required_features: List[str]
    ) -> List[str]:
        """
        Filter candidates with relaxed criteria when strict filtering fails.

        Args:
            horizon: Forecasting horizon
            market_regime: Market regime
            data_length: Available data length
            required_features: Required features

        Returns:
            List of candidate model IDs
        """
        candidates = []

        for model_id, model_info in self.model_registry.items():
            capability = self.capability_profiles.get(model_id)
            if not capability:
                continue

            # Relaxed criteria checks
            meets_criteria = True

            # Check horizon support (relaxed)
            if horizon not in capability.supported_horizons:
                # Try to find similar horizons
                if not self._has_similar_horizon_support(capability, horizon):
                    meets_criteria = False

            # Check regime support (relaxed)
            if market_regime not in capability.supported_regimes:
                # Allow models that support any regime
                if not capability.supported_regimes:
                    meets_criteria = False

            # Check data length (relaxed - allow models with higher requirements)
            if data_length < capability.min_data_points * 0.5:  # Allow 50% less data
                meets_criteria = False

            # Check feature requirements (relaxed)
            if required_features and capability.feature_requirements:
                missing_features = set(required_features) - set(capability.feature_requirements)
                if len(missing_features) > len(required_features) * 0.3:  # Allow 30% missing features
                    meets_criteria = False

            if meets_criteria:
                candidates.append(model_id)

        self.logger.info(f"Fallback filtering found {len(candidates)} candidates with relaxed criteria")
        return candidates

    def _has_similar_horizon_support(self, capability: ModelCapability, target_horizon: ForecastingHorizon) -> bool:
        """
        Check if model supports similar horizons to the target.

        Args:
            capability: Model capability profile
            target_horizon: Target forecasting horizon

        Returns:
            True if model supports similar horizons
        """
        # Define horizon similarity mapping
        horizon_similarity = {
            ForecastingHorizon.SHORT_TERM: [ForecastingHorizon.SHORT_TERM, ForecastingHorizon.MEDIUM_TERM],
            ForecastingHorizon.MEDIUM_TERM: [
                ForecastingHorizon.SHORT_TERM,
                ForecastingHorizon.MEDIUM_TERM,
                ForecastingHorizon.LONG_TERM,
            ],
            ForecastingHorizon.LONG_TERM: [ForecastingHorizon.MEDIUM_TERM, ForecastingHorizon.LONG_TERM],
        }

        similar_horizons = horizon_similarity.get(target_horizon, [])
        return any(horizon in capability.supported_horizons for horizon in similar_horizons)

    def _get_fallback_model(self) -> str:
        """
        Get a fallback model when no models meet criteria.

        Returns:
            Fallback model ID
        """
        # Try to find any available model
        if self.model_registry:
            # Return the first available model
            fallback_id = list(self.model_registry.keys())[0]
            self.logger.warning(f"Using fallback model: {fallback_id}")
            return fallback_id

        # If no models in registry, return a default
        self.logger.error("No models available in registry")
        return "default_model"

    def _log_top_models(
        self, model_scores: List[Dict], horizon: ForecastingHorizon, market_regime: MarketRegime, top_n: int = 5
    ):
        """
        Log top N models with their scores for debugging and analysis.

        Args:
            model_scores: List of model scores
            horizon: Forecasting horizon
            market_regime: Market regime
            top_n: Number of top models to log
        """
        top_models = model_scores[:top_n]

        self.logger.info(f"Top {len(top_models)} models for horizon={horizon.value}, regime={market_regime.value}:")

        for i, model in enumerate(top_models, 1):
            self.logger.info(
                f"  {i}. {model['model_id']} ({model['model_type']}) - "
                f"Combined: {model['combined_score']:.3f}, "
                f"Capability: {model['capability_score']:.3f}, "
                f"Performance: {model['performance_score']:.3f}"
            )

        # Log selection statistics
        if model_scores:
            scores = [m["combined_score"] for m in model_scores]
            self.logger.info(
                f"Score statistics - Mean: {np.mean(scores):.3f}, "
                f"Std: {np.std(scores):.3f}, "
                f"Min: {np.min(scores):.3f}, "
                f"Max: {np.max(scores):.3f}"
            )

        # Store in selection history for analysis
        selection_record = {
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon.value,
            "market_regime": market_regime.value,
            "top_models": top_models,
            "total_candidates": len(model_scores),
        }
        self.selection_history.append(selection_record)

        # Keep only last 1000 selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

    def _filter_candidates(
        self, horizon: ForecastingHorizon, market_regime: MarketRegime, data_length: int, required_features: List[str]
    ) -> List[str]:
        """Filter models based on basic requirements."""
        candidates = []

        for model_id, capability in self.capability_profiles.items():
            # Check horizon support
            if horizon not in capability.supported_horizons:
                continue

            # Check regime support
            if market_regime not in capability.supported_regimes:
                continue

            # Check data requirements
            if data_length < capability.min_data_points:
                continue

            # Check feature requirements
            if required_features:
                missing_features = set(required_features) - set(capability.feature_requirements)
                if missing_features:
                    continue

            candidates.append(model_id)

        return candidates

    def _calculate_capability_score(
        self, model_id: str, horizon: ForecastingHorizon, market_regime: MarketRegime, data_length: int
    ) -> float:
        """Calculate capability-based score for a model."""
        capability = self.capability_profiles[model_id]

        # Base score
        score = 1.0

        # Adjust for data utilization
        data_utilization = min(data_length / capability.min_data_points, 2.0)
        score *= data_utilization

        # Adjust for complexity (prefer simpler models)
        complexity_penalty = {"low": 1.0, "medium": 0.9, "high": 0.8}
        score *= complexity_penalty.get(capability.computational_complexity, 0.8)

        return min(score, 1.0)

    def _calculate_performance_score(self, model_id: str) -> float:
        """Calculate performance-based score for a model."""
        if model_id not in self.performance_history or not self.performance_history[model_id]:
            return 0.5  # Default score for new models

        recent_performance = self.performance_history[model_id]

        # Calculate average metrics
        avg_accuracy = np.mean([p.accuracy for p in recent_performance])
        avg_sharpe = np.mean([p.sharpe_ratio for p in recent_performance])
        avg_drawdown = np.mean([p.max_drawdown for p in recent_performance])

        # Normalize and combine metrics
        accuracy_score = avg_accuracy
        sharpe_score = min(avg_sharpe / 2.0, 1.0)  # Normalize to 0-1
        drawdown_score = 1.0 - min(avg_drawdown, 1.0)  # Invert drawdown

        # Weighted combination
        performance_score = 0.4 * accuracy_score + 0.4 * sharpe_score + 0.2 * drawdown_score

        return performance_score

    def _log_selection(
        self, model_id: str, horizon: ForecastingHorizon, market_regime: MarketRegime, confidence: float
    ):
        """Log model selection for analysis."""
        selection = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "horizon": horizon.value,
            "market_regime": market_regime.value,
            "confidence": confidence,
            "available_models": len(self.capability_profiles),
        }

        self.selection_history.append(selection)

        # Keep only recent selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

    def get_model_recommendations(
        self, horizon: ForecastingHorizon, market_regime: MarketRegime, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get top-k model recommendations with explanations."""
        try:
            candidates = self._filter_candidates(horizon, market_regime, 1000, [])  # Assume sufficient data

            recommendations = []
            for model_id in candidates:
                capability_score = self._calculate_capability_score(model_id, horizon, market_regime, 1000)
                performance_score = self._calculate_performance_score(model_id)

                capability = self.capability_profiles[model_id]

                recommendation = {
                    "model_id": model_id,
                    "model_type": capability.model_type.value,
                    "capability_score": capability_score,
                    "performance_score": performance_score,
                    "total_score": 0.4 * capability_score + 0.6 * performance_score,
                    "strengths": self._get_model_strengths(model_id, horizon, market_regime),
                    "weaknesses": self._get_model_weaknesses(model_id, horizon, market_regime),
                    "recent_performance": self._get_recent_performance_summary(model_id),
                }

                recommendations.append(recommendation)

            # Sort by total score and return top-k
            recommendations.sort(key=lambda x: x["total_score"], reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def _get_model_strengths(
        self, model_id: str, horizon: ForecastingHorizon, market_regime: MarketRegime
    ) -> List[str]:
        """Get model strengths for given scenario."""
        capability = self.capability_profiles[model_id]
        strengths = []

        if capability.model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
            strengths.append("Excellent at capturing complex patterns")
        if capability.model_type == ModelType.PROPHET:
            strengths.append("Handles seasonality and trends well")
        if capability.model_type == ModelType.ENSEMBLE:
            strengths.append("Robust and reduces overfitting")

        if market_regime in capability.supported_regimes:
            strengths.append(f"Optimized for {market_regime.value} markets")

        return strengths

    def _get_model_weaknesses(
        self, model_id: str, horizon: ForecastingHorizon, market_regime: MarketRegime
    ) -> List[str]:
        """Get model weaknesses for given scenario."""
        capability = self.capability_profiles[model_id]
        weaknesses = []

        if capability.computational_complexity == "high":
            weaknesses.append("High computational requirements")
        if capability.memory_requirements == "high":
            weaknesses.append("High memory usage")
        if market_regime not in capability.supported_regimes:
            weaknesses.append(f"Not optimized for {market_regime.value} markets")

        return weaknesses

    def _get_recent_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get recent performance summary for a model."""
        if model_id not in self.performance_history or not self.performance_history[model_id]:
            return {"status": "No recent performance data"}

        recent = self.performance_history[model_id][-5:]  # Last 5 evaluations

        return {
            "avg_accuracy": np.mean([p.accuracy for p in recent]),
            "avg_sharpe": np.mean([p.sharpe_ratio for p in recent]),
            "avg_drawdown": np.mean([p.max_drawdown for p in recent]),
            "evaluations_count": len(recent),
            "last_evaluated": recent[-1].last_updated.isoformat() if recent else None,
        }

    def save_state(self, filepath: str):
        """Save agent state to file."""
        try:
            state = {
                "model_registry": self.model_registry,
                "performance_history": {
                    model_id: [p.__dict__ for p in performances]
                    for model_id, performances in self.performance_history.items()
                },
                "capability_profiles": {
                    model_id: {
                        "model_id": cp.model_id,
                        "model_type": cp.model_type.value,
                        "supported_horizons": [h.value for h in cp.supported_horizons],
                        "supported_regimes": [r.value for r in cp.supported_regimes],
                        "min_data_points": cp.min_data_points,
                        "max_forecast_horizon": cp.max_forecast_horizon,
                        "feature_requirements": cp.feature_requirements,
                        "computational_complexity": cp.computational_complexity,
                        "memory_requirements": cp.memory_requirements,
                    }
                    for model_id, cp in self.capability_profiles.items()
                },
                "selection_history": self.selection_history,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "performance_window": self.performance_window,
            }

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"Saved Model Selector Agent state to {filepath}")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self, filepath: str):
        """Load agent state from file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore capability profiles
            self.capability_profiles = {}
            for model_id, cp_data in state["capability_profiles"].items():
                capability = ModelCapability(
                    model_id=cp_data["model_id"],
                    model_type=ModelType(cp_data["model_type"]),
                    supported_horizons=[ForecastingHorizon(h) for h in cp_data["supported_horizons"]],
                    supported_regimes=[MarketRegime(r) for r in cp_data["supported_regimes"]],
                    min_data_points=cp_data["min_data_points"],
                    max_forecast_horizon=cp_data["max_forecast_horizon"],
                    feature_requirements=cp_data["feature_requirements"],
                    computational_complexity=cp_data["computational_complexity"],
                    memory_requirements=cp_data["memory_requirements"],
                )
                self.capability_profiles[model_id] = capability

            # Restore performance history
            self.performance_history = {}
            for model_id, performances in state["performance_history"].items():
                self.performance_history[model_id] = [ModelPerformance(**p) for p in performances]

            # Restore other state
            self.selection_history = state["selection_history"]
            self.learning_rate = state.get("learning_rate", 0.01)
            self.exploration_rate = state.get("exploration_rate", 0.1)
            self.performance_window = state.get("performance_window", 30)

            logger.info(f"Loaded Model Selector Agent state from {filepath}")

        except Exception as e:
            logger.error(f"Error loading state: {e}")
