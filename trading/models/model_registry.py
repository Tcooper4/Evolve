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
        """Register all available models with metadata."""
        
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
            logger.warning(f"GARCH model not available: {e}")
        
        # Old Autoformer removed - now using NeuralForecast version below
        
        try:
            from trading.models.advanced.transformer.time_series_transformer import TimeSeriesTransformer
            self.register('Transformer', TimeSeriesTransformer, {
                'type': 'single_asset',
                'complexity': 'high',
                'description': 'Advanced transformer with attention mechanism',
                'use_case': 'general',
                'requires_gpu': True,
                'min_data_points': 200
            })
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
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
        except ImportError as e:
            logger.warning(f"GNN model not available: {e}")
        
        # ============================================================================
        # NEURALFORECAST MODELS (State-of-the-Art)
        # ============================================================================
        
        logger.info("Registering NeuralForecast models...")
        
        try:
            from trading.models.neuralforecast_models import (
                AutoformerModel,
                InformerModel,
                TFTModel,
                NBEATSModel,
                PatchTSTModel,
                NHITSModel,
                NEURALFORECAST_AVAILABLE
            )
            
            if NEURALFORECAST_AVAILABLE:
                # Autoformer - Decomposition transformer
                self.register('Autoformer', AutoformerModel, {
                    'type': 'single_asset',
                    'complexity': 'high',
                    'description': 'Decomposition-based transformer (SOTA)',
                    'use_case': 'general',
                    'requires_gpu': False,  # Can use CPU
                    'min_data_points': 100,
                    'best_for': 'Long-term forecasting with trend/seasonality'
                })
                logger.info("✅ Registered: Autoformer (NeuralForecast)")
                
                # Informer - Efficient transformer
                self.register('Informer', InformerModel, {
                    'type': 'single_asset',
                    'complexity': 'high',
                    'description': 'Efficient transformer for long sequences',
                    'use_case': 'general',
                    'requires_gpu': False,
                    'min_data_points': 100,
                    'best_for': 'Very long-term forecasting (100+ days)'
                })
                logger.info("✅ Registered: Informer")
                
                # TFT - Temporal Fusion Transformer
                self.register('TFT', TFTModel, {
                    'type': 'single_asset',
                    'complexity': 'high',
                    'description': 'Temporal Fusion Transformer',
                    'use_case': 'general',
                    'requires_gpu': False,
                    'min_data_points': 100,
                    'best_for': 'Multi-horizon with interpretability'
                })
                logger.info("✅ Registered: TFT")
                
                # N-BEATS - Neural basis expansion
                self.register('N-BEATS', NBEATSModel, {
                    'type': 'single_asset',
                    'complexity': 'high',
                    'description': 'Neural basis expansion (interpretable)',
                    'use_case': 'general',
                    'requires_gpu': False,
                    'min_data_points': 100,
                    'best_for': 'Interpretable trend/seasonality'
                })
                logger.info("✅ Registered: N-BEATS")
                
                # PatchTST - Latest SOTA
                self.register('PatchTST', PatchTSTModel, {
                    'type': 'single_asset',
                    'complexity': 'high',
                    'description': 'Patch-based transformer (Latest SOTA)',
                    'use_case': 'general',
                    'requires_gpu': False,
                    'min_data_points': 100,
                    'best_for': 'Best overall performance, long-term'
                })
                logger.info("✅ Registered: PatchTST (Latest SOTA)")
                
                # N-HiTS - Fast and accurate
                self.register('N-HiTS', NHITSModel, {
                    'type': 'single_asset',
                    'complexity': 'medium',
                    'description': 'Hierarchical interpolation (fast)',
                    'use_case': 'general',
                    'requires_gpu': False,
                    'min_data_points': 100,
                    'best_for': 'Fast training, good accuracy'
                })
                logger.info("✅ Registered: N-HiTS")
                
                logger.info("✅ All NeuralForecast models registered successfully")
            else:
                logger.warning("⚠️  NeuralForecast not available - models not registered")
                
        except ImportError as e:
            logger.warning(f"⚠️  Could not register NeuralForecast models: {e}")
            logger.info("Install with: pip install neuralforecast")
        
        logger.info(f"✅ Registered {len(self._models)} models")
    
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


# Global registry instance
_global_registry = None

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
