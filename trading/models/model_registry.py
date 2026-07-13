"""
Smart Model Registry with filtering by use case.

Prevents inappropriate model selection (e.g., GNN in single-asset forecasting).
"""

from typing import Dict, Type, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for forecasting models with smart filtering."""
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._model_metadata: Dict[str, Dict] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all available models with metadata.

        BUG FIX: every `except ImportError as e:` block in this method
        was widened to `except Exception as e:`. Verified concretely that
        a narrower fix wasn't sufficient: different optional dependencies
        fail with different exception types when unavailable in this
        environment (TypeError from the LSTM/base_model import chain
        subclassing a None Dataset class, AttributeError from the GNN
        import chain accessing .Module on a None torch.nn). Catching only
        ImportError (or even ImportError+TypeError) meant one model's
        registration failure crashed the ENTIRE registry - every other
        model, XGBoost/ARIMA/Prophet/etc., never got registered either -
        instead of gracefully skipping just the one broken model, which
        is this method's entire design intent
        (see every other block's own try/except ImportError pattern).
        """
        
        # === SINGLE-ASSET MODELS (for quick forecast) ===
        
        try:
            # Try LSTMModel first (if it's a BaseModel), otherwise use LSTMForecaster
            try:
                from trading.models.lstm_model import LSTMModel
                # Check if LSTMModel is a BaseModel
                from trading.models.base_model import BaseModel
                if issubclass(LSTMModel, BaseModel):
                    LSTMClass = LSTMModel
                else:
                    from trading.models.lstm_model import LSTMForecaster
                    LSTMClass = LSTMForecaster
            except (ImportError, TypeError):
                from trading.models.lstm_model import LSTMForecaster
                LSTMClass = LSTMForecaster
            
            self.register('LSTM', LSTMClass, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Deep learning neural network for time series',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"LSTM model not available: {e}")
        
        try:
            from trading.models.xgboost_model import XGBoostModel
            self.register('XGBoost', XGBoostModel, {
                'type': 'single_asset',
                'complexity': 'medium',
                'description': 'Gradient boosting tree model',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 50
            })
        except Exception as e:
            logger.warning(f"XGBoost model not available: {e}")
        
        try:
            from trading.models.prophet_model import ProphetModel
            self.register('Prophet', ProphetModel, {
                'type': 'single_asset',
                'complexity': 'medium',
                'description': 'Facebook Prophet for seasonality',
                'use_case': 'seasonal',
                'requires_gpu': False,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"Prophet model not available: {e}")
        
        try:
            from trading.models.arima_model import ARIMAModel
            self.register('ARIMA', ARIMAModel, {
                'type': 'single_asset',
                'complexity': 'low',
                'description': 'Statistical time series model',
                'use_case': 'stationary',
                'requires_gpu': False,
                'min_data_points': 50
            })
        except Exception as e:
            logger.warning(f"ARIMA model not available: {e}")
        
        try:
            from trading.models.ensemble_model import EnsembleModel
            self.register('Ensemble', EnsembleModel, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Combines multiple models',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"Ensemble model not available: {e}")
        
        # === ADVANCED SINGLE-ASSET MODELS ===
        
        try:
            from trading.models.tcn_model import TCNModel
            self.register('TCN', TCNModel, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Temporal Convolutional Network',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"TCN model not available: {e}")
        
        try:
            from trading.models.garch_model import GARCHModel
            self.register('GARCH', GARCHModel, {
                'type': 'single_asset',
                'use_case': 'volatility',
                'complexity': 'medium',
                'description': 'Volatility forecasting model',
                'requires_gpu': False,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"GARCH model not available: {e}")
        
        # Old Autoformer removed - now using NeuralForecast version below
        
        try:
            # BUG FIX: this previously imported a non-existent name
            # 'TimeSeriesTransformer' - the real class is
            # TransformerForecaster (class TransformerForecaster(BaseModel),
            # confirmed and already fixed earlier this session). Gracefully
            # caught by the except ImportError below (no crash), but it
            # silently excluded a genuine, working model from the
            # registry - anyone calling get('Transformer') or checking
            # list_models() would never see it, even though the
            # underlying implementation is sound.
            from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
            self.register('Transformer', TransformerForecaster, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Advanced transformer with attention mechanism',
                'use_case': 'general',
                'requires_gpu': True,
                'min_data_points': 200
            })
        except Exception as e:
            logger.warning(f"Transformer model not available: {e}")
        
        try:
            from trading.models.catboost_model import CatBoostModel
            self.register('CatBoost', CatBoostModel, {
                'type': 'single_asset',
                'complexity': 'medium',
                'description': 'Gradient boosting with categorical features',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 50
            })
        except Exception as e:
            logger.warning(f"CatBoost model not available: {e}")
        
        try:
            from trading.models.ridge_model import RidgeModel
            self.register('Ridge', RidgeModel, {
                'type': 'single_asset',
                'complexity': 'low',
                'description': 'Linear regression baseline',
                'use_case': 'baseline',
                'requires_gpu': False,
                'min_data_points': 30
            })
        except Exception as e:
            logger.warning(f"Ridge model not available: {e}")
        
        try:
            from trading.forecasting.hybrid_model import HybridModel
            self.register('Hybrid', HybridModel, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Combines statistical and ML approaches',
                'use_case': 'general',
                'requires_gpu': False,
                'min_data_points': 150
            })
        except Exception as e:
            logger.warning(f"Hybrid model not available: {e}")
        
        # === MULTI-ASSET MODELS (NOT for quick forecast!) ===
        
        try:
            from trading.models.advanced.gnn.gnn_model import GNNForecaster
            self.register('GNN', GNNForecaster, {
                'type': 'multi_asset',  # ← CRITICAL: Prevents showing in quick forecast
                'complexity': 'high',
                'description': 'Graph Neural Network for multi-asset relationships',
                'use_case': 'portfolio',
                'requires_gpu': False,
                'min_assets': 3,
                'max_assets': 20,
                'min_data_points': 100
            })
        except Exception as e:
            logger.warning(f"GNN model not available: {e}")
        
        # ============================================================================
        # NEURALFORECAST MODELS (State-of-the-Art)
        # ============================================================================
        
        logger.info("Registering NeuralForecast models...")
        
        try:
            from trading.models.neuralforecast_models import (
                NBEATSModel,
                NHITSModel,
                PatchTSTModel,
                TFTModel,
                neuralforecast_installed,
            )

            if neuralforecast_installed():
                self.register("N-BEATS", NBEATSModel, {
                    "type": "single_asset",
                    "complexity": "high",
                    "description": "Neural basis expansion (NeuralForecast)",
                    "use_case": "general",
                    "requires_gpu": False,
                    "min_data_points": 100,
                    "best_for": "Interpretable trend/seasonality",
                })
                self.register("N-HiTS", NHITSModel, {
                    "type": "single_asset",
                    "complexity": "medium",
                    "description": "Neural hierarchical interpolation (NeuralForecast)",
                    "use_case": "general",
                    "requires_gpu": False,
                    "min_data_points": 100,
                    "best_for": "Fast training, good accuracy",
                })
                self.register("PatchTST", PatchTSTModel, {
                    "type": "single_asset",
                    "complexity": "high",
                    "description": "Patch-based transformer (NeuralForecast)",
                    "use_case": "general",
                    "requires_gpu": False,
                    "min_data_points": 100,
                    "best_for": "Long-horizon accuracy",
                })
                self.register("TFT", TFTModel, {
                    "type": "single_asset",
                    "complexity": "high",
                    "description": "Temporal Fusion Transformer (NeuralForecast)",
                    "use_case": "general",
                    "requires_gpu": False,
                    "min_data_points": 100,
                    "best_for": "Multi-horizon with covariates",
                })
                logger.info(
                    "NeuralForecast models registered: N-BEATS, N-HiTS, PatchTST, TFT"
                )
            else:
                logger.warning(
                    "NeuralForecast not installed — bonus models N-BEATS, N-HiTS, "
                    "PatchTST, TFT disabled"
                )

        except Exception as e:
            logger.warning(
                "NeuralForecast registration skipped (import error): %s", e
            )
        
        logger.info(f"[OK] Registered {len(self._models)} models")
    
    def register(self, name: str, model_class: Type, metadata: Dict = None):
        """Register a model class with metadata.
        
        Args:
            name: Model name
            model_class: Model class
            metadata: Model metadata for filtering
        """
        self._models[name] = model_class
        self._model_metadata[name] = metadata or {}
        logger.debug(f"Registered model: {name}")
    
    def get(self, name: str) -> Optional[Type]:
        """Get a model class by name."""
        return self._models.get(name)
    
    def list_models(self, filter_by: Optional[Dict] = None) -> List[str]:
        """Get list of model names, optionally filtered.
        
        Args:
            filter_by: Dictionary of metadata filters
                Examples:
                - {'type': 'single_asset'} - only single-asset models
                - {'type': 'multi_asset'} - only multi-asset models
                - {'use_case': 'volatility'} - only volatility models
                - {'complexity': 'low'} - only simple models
        
        Returns:
            List of matching model names
        """
        if filter_by is None:
            return sorted(self._models.keys())
        
        matching_models = []
        for name, metadata in self._model_metadata.items():
            # Check if all filter criteria match
            matches = all(
                metadata.get(key) == value 
                for key, value in filter_by.items()
            )
            
            if matches:
                matching_models.append(name)
        
        return sorted(matching_models)
    
    def get_quick_forecast_models(self) -> List[str]:
        """Get models suitable for quick single-asset forecasting.
        
        Returns only the core 4 models: LSTM, XGBoost, Prophet, ARIMA
        """
        core_models = ['LSTM', 'XGBoost', 'Prophet', 'ARIMA']
        return [m for m in core_models if m in self._models]
    
    def get_advanced_models(self) -> List[str]:
        """Get all single-asset models including advanced ones."""
        return self.list_models(filter_by={'type': 'single_asset'})
    
    def get_multi_asset_models(self) -> List[str]:
        """Get models that require multiple assets."""
        return self.list_models(filter_by={'type': 'multi_asset'})
    
    def get_model_info(self, name: str) -> Dict:
        """Get detailed information about a model."""
        model_class = self.get(name)
        if model_class is None:
            return {}
        
        info = {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
        }
        
        # Add metadata
        if name in self._model_metadata:
            info.update(self._model_metadata[name])
        
        return info
    
    def list_all_info(self) -> List[Dict]:
        """Get information about all registered models."""
        return [self.get_model_info(name) for name in sorted(self._models.keys())]
    
    @property
    def registry(self) -> Dict[str, Type]:
        """Get the model registry dictionary."""
        return self._models


# Global registry instance
_global_registry = None

_COMPLEXITY_RANK = {"low": 0, "medium": 1, "high": 2}


def filter_eligible_models(
    features: Optional[Dict] = None,
    available_data_points: int = 0,
    n_assets: int = 1,
    registry: Optional["ModelRegistry"] = None,
) -> List[str]:
    """Inclusion list from registry metadata — not weights, not ranking.

    Uses existing per-model metadata only:
      * min_data_points vs available_data_points
      * type (single_asset / multi_asset) vs n_assets
      * min_assets / max_assets for multi-asset models
      * optional features['max_complexity'] in {low, medium, high} as a
        hard cost ceiling (omitted → complexity does not filter)

    ``features`` may also carry Phase-1 descriptors; they are ignored here
    so eligibility stays separate from FFORMA-style weighting (Phase 3).
    """
    reg = registry if registry is not None else get_registry()
    features = features or {}
    try:
        n_points = int(available_data_points)
    except Exception:
        n_points = 0
    try:
        n_assets_i = int(n_assets)
    except Exception:
        n_assets_i = 0

    max_complexity = features.get("max_complexity")
    if max_complexity not in _COMPLEXITY_RANK:
        max_complexity = None

    eligible: List[str] = []
    for name in reg.list_models():
        info = reg.get_model_info(name) or {}
        min_dp = int(info.get("min_data_points") or 0)
        if n_points < min_dp:
            continue

        mtype = str(info.get("type") or "single_asset")
        if mtype == "multi_asset":
            min_a = int(info.get("min_assets") or 3)
            max_a = int(info.get("max_assets") or 10**9)
            if n_assets_i < min_a or n_assets_i > max_a:
                continue
        else:
            # single-asset (and unknown types): need at least one series
            if n_assets_i < 1:
                continue

        if max_complexity is not None:
            rank = _COMPLEXITY_RANK.get(str(info.get("complexity") or "medium"), 1)
            if rank > _COMPLEXITY_RANK[max_complexity]:
                continue

        eligible.append(name)
    return sorted(eligible)


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
