"""
Forecast Controller

This module provides intelligent routing for forecast requests based on context,
confidence, and historical accuracy. It offloads hybrid model logic to a separate
hybrid_engine.py file for better modularity.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.models.model_registry import ModelRegistry
from trading.models.forecast_router import ForecastRouter
from trading.utils.model_selector import ModelSelector

logger = logging.getLogger(__name__)


class ForecastController:
    """
    Intelligent forecast controller with routing based on context, confidence, and historical accuracy.
    
    Features:
    - Context-aware model selection
    - Confidence-based routing
    - Historical accuracy tracking
    - Hybrid model orchestration
    - Fallback mechanisms
    """

    def __init__(
        self,
        model_selector: Optional[ModelSelector] = None,
        forecast_router: Optional[ForecastRouter] = None,
        enable_hybrid: bool = True,
        confidence_threshold: float = 0.7,
        max_models_per_request: int = 3,
    ):
        """
        Initialize the forecast controller.

        Args:
            model_selector: Model selector instance
            forecast_router: Forecast router instance
            enable_hybrid: Whether to enable hybrid model logic
            confidence_threshold: Minimum confidence for model selection
            max_models_per_request: Maximum models to use per request
        """
        self.model_selector = model_selector or ModelSelector()
        self.forecast_router = forecast_router or ForecastRouter()
        self.enable_hybrid = enable_hybrid
        self.confidence_threshold = confidence_threshold
        self.max_models_per_request = max_models_per_request
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.model_accuracy: Dict[str, float] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Initialize hybrid engine if enabled
        self.hybrid_engine = None
        if self.enable_hybrid:
            self._setup_hybrid_engine()
        
        logger.info("ForecastController initialized successfully")

    def _setup_hybrid_engine(self) -> None:
        """Setup hybrid model engine."""
        try:
            from system.hybrid_engine import HybridEngine
            self.hybrid_engine = HybridEngine()
            logger.info("Hybrid engine initialized successfully")
        except ImportError:
            logger.warning("Hybrid engine not available, hybrid mode disabled")
            self.enable_hybrid = False
        except Exception as e:
            logger.error(f"Failed to initialize hybrid engine: {e}")
            self.enable_hybrid = False

    def route_forecast_request(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
        horizon: int = 30,
        confidence_required: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Route forecast request to appropriate models based on context and confidence.

        Args:
            data: Input data for forecasting
            context: Context information (market conditions, volatility, etc.)
            horizon: Forecast horizon
            confidence_required: Required confidence level

        Returns:
            Dictionary containing forecast results and routing information
        """
        request_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            logger.info(f"Processing forecast request {request_id}")
            
            # Analyze context and select best models
            selected_models = self._select_models_for_context(data, context, horizon)
            
            # Route to model selector for best model selection
            best_models = self.model_selector.select_best_models(
                data=data,
                context=context,
                horizon=horizon,
                confidence_threshold=confidence_required,
                max_models=self.max_models_per_request,
                available_models=selected_models
            )
            
            # Execute forecasts
            forecast_results = self._execute_forecasts(data, best_models, horizon, context)
            
            # Combine results if multiple models used
            if len(forecast_results) > 1 and self.enable_hybrid:
                final_result = self._combine_forecasts(forecast_results, context)
            else:
                final_result = forecast_results[0] if forecast_results else self._get_fallback_result()
            
            # Update performance tracking
            self._update_performance_tracking(request_id, best_models, final_result)
            
            # Add routing information
            final_result.update({
                "request_id": request_id,
                "models_used": [result["model"] for result in forecast_results],
                "routing_info": {
                    "context_analyzed": context,
                    "models_selected": len(best_models),
                    "hybrid_used": len(forecast_results) > 1 and self.enable_hybrid,
                    "confidence_achieved": final_result.get("confidence", 0.0),
                }
            })
            
            logger.info(f"Forecast request {request_id} completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Forecast request {request_id} failed: {e}")
            return self._get_error_result(request_id, str(e))

    def _select_models_for_context(
        self, data: pd.DataFrame, context: Dict[str, Any], horizon: int
    ) -> List[str]:
        """
        Select appropriate models based on context.

        Args:
            data: Input data
            context: Context information
            horizon: Forecast horizon

        Returns:
            List of model names to consider
        """
        available_models = []
        
        # Analyze data characteristics
        data_length = len(data)
        volatility = data.select_dtypes(include=[np.number]).std().mean()
        trend_strength = self._calculate_trend_strength(data)
        
        # Market conditions from context
        market_volatility = context.get("market_volatility", "medium")
        market_trend = context.get("market_trend", "neutral")
        seasonality = context.get("seasonality", False)
        
        # Select models based on characteristics
        if data_length < 50:
            # Short series - use simple models
            available_models.extend(["ARIMA", "Prophet"])
        elif data_length < 200:
            # Medium series - add ML models
            available_models.extend(["ARIMA", "Prophet", "XGBoost", "LSTM"])
        else:
            # Long series - use all models including transformer
            available_models.extend(["ARIMA", "Prophet", "XGBoost", "LSTM", "Transformer"])
        
        # Adjust based on volatility
        if volatility > 0.1:  # High volatility
            available_models = [m for m in available_models if m in ["ARIMA", "XGBoost", "Transformer"]]
        elif volatility < 0.01:  # Low volatility
            available_models = [m for m in available_models if m in ["Prophet", "LSTM"]]
        
        # Adjust based on trend strength
        if trend_strength > 0.7:  # Strong trend
            available_models = [m for m in available_models if m in ["ARIMA", "XGBoost"]]
        elif trend_strength < 0.3:  # Weak trend
            available_models = [m for m in available_models if m in ["Prophet", "LSTM", "Transformer"]]
        
        # Adjust based on seasonality
        if seasonality:
            available_models = [m for m in available_models if m in ["Prophet", "ARIMA"]]
        
        # Ensure we have at least one model
        if not available_models:
            available_models = ["ARIMA"]  # Default fallback
        
        logger.info(f"Selected models for context: {available_models}")
        return available_models

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength in the data.

        Args:
            data: Input data

        Returns:
            Trend strength (0-1)
        """
        try:
            # Use first numeric column for trend calculation
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return 0.5
            
            series = data[numeric_cols[0]]
            
            # Calculate linear trend
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            
            # Normalize slope by series range
            series_range = series.max() - series.min()
            if series_range == 0:
                return 0.5
            
            normalized_slope = abs(slope) / series_range
            
            # Convert to 0-1 scale
            trend_strength = min(1.0, normalized_slope * 100)
            
            return trend_strength
            
        except Exception as e:
            logger.warning(f"Failed to calculate trend strength: {e}")
            return 0.5

    def _execute_forecasts(
        self, data: pd.DataFrame, models: List[str], horizon: int, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute forecasts using selected models.

        Args:
            data: Input data
            models: List of model names to use
            horizon: Forecast horizon
            context: Context information

        Returns:
            List of forecast results
        """
        results = []
        
        for model_name in models:
            try:
                logger.info(f"Executing forecast with {model_name}")
                
                # Get model instance
                model = self._get_model_instance(model_name)
                if model is None:
                    continue
                
                # Execute forecast
                forecast_result = model.forecast(data, horizon)
                
                # Add model information
                forecast_result["model"] = model_name
                forecast_result["execution_time"] = datetime.now().isoformat()
                
                results.append(forecast_result)
                
            except Exception as e:
                logger.error(f"Failed to execute forecast with {model_name}: {e}")
                continue
        
        return results

    def _get_model_instance(self, model_name: str) -> Any:
        """
        Get model instance from registry.

        Args:
            model_name: Name of the model

        Returns:
            Model instance or None if not available
        """
        try:
            # Try to get from model registry
            model_class = ModelRegistry.get_model(model_name)
            if model_class:
                return model_class()
            
            # Try to import and instantiate
            if model_name == "ARIMA":
                from trading.models.arima_model import ARIMAModel
                return ARIMAModel()
            elif model_name == "Prophet":
                from trading.models.prophet_model import ProphetForecaster
                return ProphetForecaster()
            elif model_name == "XGBoost":
                from trading.models.xgboost_model import XGBoostForecaster
                return XGBoostForecaster()
            elif model_name == "LSTM":
                from trading.models.lstm_model import LSTMForecaster
                return LSTMForecaster()
            elif model_name == "Transformer":
                from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
                return TransformerForecaster()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model instance for {model_name}: {e}")
            return None

    def _combine_forecasts(
        self, forecast_results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine multiple forecast results using hybrid engine.

        Args:
            forecast_results: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        if self.hybrid_engine is None:
            # Simple averaging if hybrid engine not available
            return self._simple_forecast_combination(forecast_results)
        
        try:
            return self.hybrid_engine.combine_forecasts(forecast_results, context)
        except Exception as e:
            logger.error(f"Hybrid combination failed: {e}")
            return self._simple_forecast_combination(forecast_results)

    def _simple_forecast_combination(self, forecast_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple combination of forecasts using weighted averaging.

        Args:
            forecast_results: List of forecast results

        Returns:
            Combined forecast result
        """
        if not forecast_results:
            return self._get_fallback_result()
        
        # Extract forecasts and weights
        forecasts = []
        weights = []
        
        for result in forecast_results:
            if "forecast" in result:
                forecasts.append(np.array(result["forecast"]))
                # Use confidence as weight
                weight = result.get("confidence", 0.5)
                weights.append(weight)
        
        if not forecasts:
            return self._get_fallback_result()
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        combined_forecast = np.zeros_like(forecasts[0])
        for forecast, weight in zip(forecasts, weights):
            combined_forecast += weight * forecast
        
        # Calculate combined confidence
        combined_confidence = np.average(weights)
        
        return {
            "forecast": combined_forecast,
            "confidence": combined_confidence,
            "model": "Hybrid",
            "models_combined": len(forecasts),
            "combination_method": "weighted_average",
        }

    def _update_performance_tracking(
        self, request_id: str, models_used: List[str], result: Dict[str, Any]
    ) -> None:
        """
        Update performance tracking information.

        Args:
            request_id: Request identifier
            models_used: List of models used
            result: Forecast result
        """
        # Store request history
        self.request_history.append({
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "models_used": models_used,
            "confidence": result.get("confidence", 0.0),
            "success": "error" not in result,
        })
        
        # Keep only recent history
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

    def _get_fallback_result(self) -> Dict[str, Any]:
        """Get fallback forecast result."""
        return {
            "forecast": np.array([0.0] * 30),
            "confidence": 0.1,
            "model": "Fallback",
            "error": "No models available",
        }

    def _get_error_result(self, request_id: str, error_message: str) -> Dict[str, Any]:
        """Get error result."""
        return {
            "request_id": request_id,
            "error": error_message,
            "forecast": np.array([0.0] * 30),
            "confidence": 0.0,
            "model": "Error",
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.request_history:
            return {"total_requests": 0, "success_rate": 0.0}
        
        total_requests = len(self.request_history)
        successful_requests = len([r for r in self.request_history if r["success"]])
        success_rate = successful_requests / total_requests
        
        # Model usage statistics
        model_usage = {}
        for request in self.request_history:
            for model in request.get("models_used", []):
                model_usage[model] = model_usage.get(model, 0) + 1
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "model_usage": model_usage,
            "average_confidence": np.mean([r.get("confidence", 0.0) for r in self.request_history]),
        }

    def get_routing_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get routing recommendations for a given context.

        Args:
            context: Context information

        Returns:
            List of routing recommendations
        """
        recommendations = []
        
        # Analyze context and provide recommendations
        market_volatility = context.get("market_volatility", "medium")
        data_length = context.get("data_length", 100)
        
        if market_volatility == "high":
            recommendations.append({
                "model": "ARIMA",
                "reason": "High volatility - ARIMA handles volatility well",
                "confidence": 0.8,
            })
            recommendations.append({
                "model": "XGBoost",
                "reason": "High volatility - XGBoost can capture complex patterns",
                "confidence": 0.7,
            })
        elif market_volatility == "low":
            recommendations.append({
                "model": "Prophet",
                "reason": "Low volatility - Prophet good for stable trends",
                "confidence": 0.9,
            })
            recommendations.append({
                "model": "LSTM",
                "reason": "Low volatility - LSTM can learn subtle patterns",
                "confidence": 0.8,
            })
        
        if data_length < 50:
            recommendations.append({
                "model": "ARIMA",
                "reason": "Short series - ARIMA works well with limited data",
                "confidence": 0.9,
            })
        elif data_length > 200:
            recommendations.append({
                "model": "Transformer",
                "reason": "Long series - Transformer can capture long-range dependencies",
                "confidence": 0.8,
            })
        
        return recommendations


# Convenience function for backward compatibility
def route_forecast(
    data: pd.DataFrame,
    context: Dict[str, Any],
    horizon: int = 30,
    confidence_required: float = 0.8,
) -> Dict[str, Any]:
    """
    Route forecast request using ForecastController.

    Args:
        data: Input data for forecasting
        context: Context information
        horizon: Forecast horizon
        confidence_required: Required confidence level

    Returns:
        Dictionary containing forecast results
    """
    controller = ForecastController()
    return controller.route_forecast_request(data, context, horizon, confidence_required)
