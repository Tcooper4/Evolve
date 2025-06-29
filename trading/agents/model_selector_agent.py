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

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path

from trading.models.model_registry import get_available_models, ModelRegistry
from trading.models.base_model import BaseModel
from trading.market.market_analyzer import MarketAnalyzer
from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.memory.agent_memory import AgentMemory


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
    supported_horizons: List[ForecastingHorizon]
    supported_regimes: List[MarketRegime]
    min_data_points: int
    max_forecast_horizon: int
    feature_requirements: List[str]
    computational_complexity: str  # "low", "medium", "high"
    memory_requirements: str  # "low", "medium", "high"


class ModelSelectorAgent:
    """Agent for dynamic model selection based on market conditions and requirements."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Model Selector Agent."""
        self.config_path = config_path or "config/model_selector_config.json"
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
        
    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.learning_rate = config.get('learning_rate', 0.01)
                    self.exploration_rate = config.get('exploration_rate', 0.1)
                    self.performance_window = config.get('performance_window', 30)
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
                "complexity": "medium"
            },
            "transformer_medium": {
                "type": ModelType.TRANSFORMER,
                "horizons": [ForecastingHorizon.MEDIUM_TERM],
                "regimes": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.MEAN_REVERTING],
                "min_data": 200,
                "max_horizon": 28,
                "complexity": "high"
            },
            "prophet_long": {
                "type": ModelType.PROPHET,
                "horizons": [ForecastingHorizon.LONG_TERM],
                "regimes": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.SIDEWAYS],
                "min_data": 365,
                "max_horizon": 365,
                "complexity": "low"
            },
            "xgboost_ensemble": {
                "type": ModelType.ENSEMBLE,
                "horizons": [ForecastingHorizon.SHORT_TERM, ForecastingHorizon.MEDIUM_TERM],
                "regimes": [MarketRegime.MEAN_REVERTING, MarketRegime.VOLATILE],
                "min_data": 150,
                "max_horizon": 14,
                "complexity": "medium"
            }
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
                memory_requirements=config.get("memory", "medium")
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
                p for p in self.performance_history[model_id]
                if p.last_updated > cutoff_date
            ]
            
            logger.info(f"Updated performance for model: {model_id}")
            
        except Exception as e:
            logger.error(f"Error updating performance for model {model_id}: {e}")
            
    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from price data."""
        try:
            if len(price_data) < 50:
                return MarketRegime.VOLATILE
                
            returns = price_data['close'].pct_change().dropna()
            
            # Calculate regime indicators
            volatility = returns.std() * np.sqrt(252)
            trend_strength = self._calculate_trend_strength(price_data['close'])
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
        except:
            return 0.0
            
    def _calculate_mean_reversion_strength(self, returns: pd.Series) -> float:
        """Calculate mean reversion strength using autocorrelation."""
        try:
            autocorr = returns.autocorr(lag=1)
            return abs(autocorr) if not pd.isna(autocorr) else 0.0
        except:
            return 0.0
            
    def select_model(self, 
                    horizon: ForecastingHorizon,
                    market_regime: MarketRegime,
                    data_length: int,
                    required_features: List[str] = None,
                    performance_weight: float = 0.6,
                    capability_weight: float = 0.4) -> Tuple[str, float]:
        """
        Select the best model for given requirements.
        
        Args:
            horizon: Forecasting horizon
            market_regime: Current market regime
            data_length: Available data points
            required_features: Required features for the model
            performance_weight: Weight for performance-based selection
            capability_weight: Weight for capability-based selection
            
        Returns:
            Tuple of (selected_model_id, confidence_score)
        """
        try:
            # Filter models by basic requirements
            candidate_models = self._filter_candidates(
                horizon, market_regime, data_length, required_features
            )
            
            if not candidate_models:
                logger.warning("No models match the requirements")
                return None, 0.0
                
            # Calculate scores for each candidate
            model_scores = {}
            for model_id in candidate_models:
                capability_score = self._calculate_capability_score(
                    model_id, horizon, market_regime, data_length
                )
                performance_score = self._calculate_performance_score(model_id)
                
                # Combined score
                total_score = (capability_weight * capability_score + 
                             performance_weight * performance_score)
                model_scores[model_id] = total_score
                
            # Select best model
            best_model = max(model_scores, key=model_scores.get)
            confidence = model_scores[best_model]
            
            # Log selection
            self._log_selection(best_model, horizon, market_regime, confidence)
            
            return best_model, confidence
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return None, 0.0
            
    def _filter_candidates(self, 
                          horizon: ForecastingHorizon,
                          market_regime: MarketRegime,
                          data_length: int,
                          required_features: List[str]) -> List[str]:
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
        
    def _calculate_capability_score(self, 
                                   model_id: str,
                                   horizon: ForecastingHorizon,
                                   market_regime: MarketRegime,
                                   data_length: int) -> float:
        """Calculate capability-based score for a model."""
        capability = self.capability_profiles[model_id]
        
        # Base score
        score = 1.0
        
        # Adjust for data utilization
        data_utilization = min(data_length / capability.min_data_points, 2.0)
        score *= data_utilization
        
        # Adjust for complexity (prefer simpler models)
        complexity_penalty = {
            "low": 1.0,
            "medium": 0.9,
            "high": 0.8
        }
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
        performance_score = (0.4 * accuracy_score + 
                           0.4 * sharpe_score + 
                           0.2 * drawdown_score)
        
        return performance_score
        
    def _log_selection(self, model_id: str, horizon: ForecastingHorizon, 
                      market_regime: MarketRegime, confidence: float):
        """Log model selection for analysis."""
        selection = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "horizon": horizon.value,
            "market_regime": market_regime.value,
            "confidence": confidence,
            "available_models": len(self.capability_profiles)
        }
        
        self.selection_history.append(selection)
        
        # Keep only recent selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
            
    def get_model_recommendations(self, 
                                 horizon: ForecastingHorizon,
                                 market_regime: MarketRegime,
                                 top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k model recommendations with explanations."""
        try:
            candidates = self._filter_candidates(
                horizon, market_regime, 1000, []  # Assume sufficient data
            )
            
            recommendations = []
            for model_id in candidates:
                capability_score = self._calculate_capability_score(
                    model_id, horizon, market_regime, 1000
                )
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
                    "recent_performance": self._get_recent_performance_summary(model_id)
                }
                
                recommendations.append(recommendation)
                
            # Sort by total score and return top-k
            recommendations.sort(key=lambda x: x["total_score"], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
            
    def _get_model_strengths(self, model_id: str, horizon: ForecastingHorizon, 
                            market_regime: MarketRegime) -> List[str]:
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
        
    def _get_model_weaknesses(self, model_id: str, horizon: ForecastingHorizon, 
                             market_regime: MarketRegime) -> List[str]:
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
            "last_evaluated": recent[-1].last_updated.isoformat() if recent else None
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
                        "memory_requirements": cp.memory_requirements
                    }
                    for model_id, cp in self.capability_profiles.items()
                },
                "selection_history": self.selection_history,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "performance_window": self.performance_window
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info(f"Saved Model Selector Agent state to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    def load_state(self, filepath: str):
        """Load agent state from file."""
        try:
            with open(filepath, 'r') as f:
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
                    memory_requirements=cp_data["memory_requirements"]
                )
                self.capability_profiles[model_id] = capability
                
            # Restore performance history
            self.performance_history = {}
            for model_id, performances in state["performance_history"].items():
                self.performance_history[model_id] = [
                    ModelPerformance(**p) for p in performances
                ]
                
            # Restore other state
            self.selection_history = state["selection_history"]
            self.learning_rate = state.get("learning_rate", 0.01)
            self.exploration_rate = state.get("exploration_rate", 0.1)
            self.performance_window = state.get("performance_window", 30)
            
            logger.info(f"Loaded Model Selector Agent state from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}") 