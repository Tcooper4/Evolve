"""
Forecasting Postprocessor - Batch 17
Dynamic EWMA alpha tuning for improved forecast processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AlphaMethod(Enum):
    """Methods for dynamic alpha calculation."""
    STD_BASED = "std_based"
    VOLATILITY_BASED = "volatility_based"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"

@dataclass
class PostprocessConfig:
    """Configuration for postprocessing."""
    alpha_method: AlphaMethod = AlphaMethod.STD_BASED
    min_alpha: float = 0.01
    max_alpha: float = 0.3
    window_size: int = 20
    volatility_threshold: float = 0.1
    enable_dynamic_alpha: bool = True
    smoothing_factor: float = 0.1

class ForecastPostprocessor:
    """
    Enhanced forecast postprocessor with dynamic EWMA alpha tuning.
    
    Features:
    - Dynamic alpha calculation based on forecast statistics
    - Multiple alpha calculation methods
    - Adaptive smoothing based on volatility
    - Outlier detection and handling
    """
    
    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize forecast postprocessor.
        
        Args:
            config: Postprocessing configuration
        """
        self.config = config or PostprocessConfig()
        self.alpha_history: List[float] = []
        self.processed_forecasts: List[Dict[str, Any]] = []
        
        logger.info(f"ForecastPostprocessor initialized with {self.config.alpha_method.value} alpha method")
    
    def calculate_dynamic_alpha(self, 
                              forecast: Union[np.ndarray, pd.Series],
                              method: Optional[AlphaMethod] = None) -> float:
        """
        Calculate dynamic alpha based on forecast characteristics.
        
        Args:
            forecast: Forecast values
            method: Alpha calculation method (overrides config)
            
        Returns:
            Dynamic alpha value
        """
        if not self.config.enable_dynamic_alpha:
            return 0.1  # Default static alpha
        
        method = method or self.config.alpha_method
        
        if method == AlphaMethod.STD_BASED:
            return self._calculate_std_based_alpha(forecast)
        elif method == AlphaMethod.VOLATILITY_BASED:
            return self._calculate_volatility_based_alpha(forecast)
        elif method == AlphaMethod.ADAPTIVE:
            return self._calculate_adaptive_alpha(forecast)
        elif method == AlphaMethod.HYBRID:
            return self._calculate_hybrid_alpha(forecast)
        else:
            return self.config.min_alpha
    
    def _calculate_std_based_alpha(self, forecast: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate alpha based on standard deviation as specified in Batch 17.
        
        Args:
            forecast: Forecast values
            
        Returns:
            Alpha value: min(0.3, 1 - std(forecast) / max(forecast))
        """
        if len(forecast) < 2:
            return self.config.min_alpha
        
        forecast_array = np.array(forecast)
        forecast_std = np.std(forecast_array)
        forecast_max = np.max(np.abs(forecast_array))
        
        if forecast_max == 0:
            return self.config.min_alpha
        
        # Calculate alpha as specified: alpha = min(0.3, 1 - std(forecast) / max(forecast))
        alpha = 1 - (forecast_std / forecast_max)
        alpha = min(self.config.max_alpha, max(self.config.min_alpha, alpha))
        
        logger.debug(f"STD-based alpha: {alpha:.4f} (std: {forecast_std:.4f}, max: {forecast_max:.4f})")
        return alpha
    
    def _calculate_volatility_based_alpha(self, forecast: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate alpha based on volatility.
        
        Args:
            forecast: Forecast values
            
        Returns:
            Alpha value based on volatility
        """
        if len(forecast) < 2:
            return self.config.min_alpha
        
        forecast_array = np.array(forecast)
        
        # Calculate rolling volatility
        if len(forecast_array) >= self.config.window_size:
            rolling_vol = pd.Series(forecast_array).rolling(
                window=self.config.window_size
            ).std().fillna(method='bfill')
            volatility = rolling_vol.iloc[-1]
        else:
            volatility = np.std(forecast_array)
        
        # Normalize volatility
        forecast_range = np.max(forecast_array) - np.min(forecast_array)
        if forecast_range == 0:
            return self.config.min_alpha
        
        normalized_vol = volatility / forecast_range
        
        # Higher volatility = lower alpha (more smoothing)
        alpha = self.config.max_alpha * (1 - normalized_vol)
        alpha = max(self.config.min_alpha, alpha)
        
        logger.debug(f"Volatility-based alpha: {alpha:.4f} (vol: {volatility:.4f})")
        return alpha
    
    def _calculate_adaptive_alpha(self, forecast: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate adaptive alpha based on recent alpha history.
        
        Args:
            forecast: Forecast values
            
        Returns:
            Adaptive alpha value
        """
        if not self.alpha_history:
            return self._calculate_std_based_alpha(forecast)
        
        # Get recent alpha trend
        recent_alphas = self.alpha_history[-10:]  # Last 10 alphas
        alpha_trend = np.mean(np.diff(recent_alphas))
        
        # Base alpha from std method
        base_alpha = self._calculate_std_based_alpha(forecast)
        
        # Adjust based on trend
        if alpha_trend > 0:
            # Alpha is increasing, be more conservative
            alpha = base_alpha * 0.9
        elif alpha_trend < 0:
            # Alpha is decreasing, be more aggressive
            alpha = base_alpha * 1.1
        else:
            alpha = base_alpha
        
        alpha = min(self.config.max_alpha, max(self.config.min_alpha, alpha))
        
        logger.debug(f"Adaptive alpha: {alpha:.4f} (trend: {alpha_trend:.4f})")
        return alpha
    
    def _calculate_hybrid_alpha(self, forecast: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate hybrid alpha combining multiple methods.
        
        Args:
            forecast: Forecast values
            
        Returns:
            Hybrid alpha value
        """
        std_alpha = self._calculate_std_based_alpha(forecast)
        vol_alpha = self._calculate_volatility_based_alpha(forecast)
        
        # Weighted combination
        alpha = 0.6 * std_alpha + 0.4 * vol_alpha
        alpha = min(self.config.max_alpha, max(self.config.min_alpha, alpha))
        
        logger.debug(f"Hybrid alpha: {alpha:.4f} (std: {std_alpha:.4f}, vol: {vol_alpha:.4f})")
        return alpha
    
    def apply_ewma_smoothing(self, 
                           forecast: Union[np.ndarray, pd.Series],
                           alpha: Optional[float] = None) -> np.ndarray:
        """
        Apply EWMA smoothing with dynamic alpha.
        
        Args:
            forecast: Forecast values
            alpha: Alpha value (if None, calculated dynamically)
            
        Returns:
            Smoothed forecast values
        """
        forecast_array = np.array(forecast)
        
        if alpha is None:
            alpha = self.calculate_dynamic_alpha(forecast_array)
        
        # Store alpha for history
        self.alpha_history.append(alpha)
        
        # Apply EWMA smoothing
        smoothed = pd.Series(forecast_array).ewm(alpha=alpha).mean().values
        
        logger.debug(f"Applied EWMA smoothing with alpha: {alpha:.4f}")
        return smoothed
    
    def postprocess_forecast(self, 
                           forecast: Union[np.ndarray, pd.Series],
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Postprocess forecast with dynamic alpha tuning.
        
        Args:
            forecast: Raw forecast values
            metadata: Additional metadata
            
        Returns:
            Dictionary with processed forecast and metadata
        """
        if metadata is None:
            metadata = {}
        
        forecast_array = np.array(forecast)
        
        # Calculate dynamic alpha
        alpha = self.calculate_dynamic_alpha(forecast_array)
        
        # Apply smoothing
        smoothed_forecast = self.apply_ewma_smoothing(forecast_array, alpha)
        
        # Calculate statistics
        original_stats = {
            'mean': float(np.mean(forecast_array)),
            'std': float(np.std(forecast_array)),
            'min': float(np.min(forecast_array)),
            'max': float(np.max(forecast_array)),
            'range': float(np.max(forecast_array) - np.min(forecast_array))
        }
        
        smoothed_stats = {
            'mean': float(np.mean(smoothed_forecast)),
            'std': float(np.std(smoothed_forecast)),
            'min': float(np.min(smoothed_forecast)),
            'max': float(np.max(smoothed_forecast)),
            'range': float(np.max(smoothed_forecast) - np.min(smoothed_forecast))
        }
        
        # Create result
        result = {
            'original_forecast': forecast_array,
            'smoothed_forecast': smoothed_forecast,
            'alpha': alpha,
            'original_stats': original_stats,
            'smoothed_stats': smoothed_stats,
            'metadata': metadata,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Store for history
        self.processed_forecasts.append(result)
        
        logger.info(f"Postprocessed forecast with alpha: {alpha:.4f}")
        logger.info(f"Original std: {original_stats['std']:.4f}, Smoothed std: {smoothed_stats['std']:.4f}")
        
        return result
    
    def batch_postprocess(self, 
                         forecasts: List[Union[np.ndarray, pd.Series]],
                         metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Postprocess multiple forecasts in batch.
        
        Args:
            forecasts: List of forecast arrays/series
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of postprocessed results
        """
        if metadata_list is None:
            metadata_list = [{}] * len(forecasts)
        
        results = []
        for i, (forecast, metadata) in enumerate(zip(forecasts, metadata_list)):
            metadata['batch_index'] = i
            result = self.postprocess_forecast(forecast, metadata)
            results.append(result)
        
        return results
    
    def get_alpha_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about alpha values used.
        
        Returns:
            Dictionary with alpha statistics
        """
        if not self.alpha_history:
            return {}
        
        alphas = np.array(self.alpha_history)
        
        return {
            'total_forecasts': len(self.alpha_history),
            'mean_alpha': float(np.mean(alphas)),
            'std_alpha': float(np.std(alphas)),
            'min_alpha': float(np.min(alphas)),
            'max_alpha': float(np.max(alphas)),
            'alpha_trend': float(np.polyfit(range(len(alphas)), alphas, 1)[0])
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all processed forecasts.
        
        Returns:
            Dictionary with processing summary
        """
        if not self.processed_forecasts:
            return {}
        
        total_forecasts = len(self.processed_forecasts)
        alphas = [result['alpha'] for result in self.processed_forecasts]
        
        # Calculate smoothing effectiveness
        smoothing_ratios = []
        for result in self.processed_forecasts:
            original_std = result['original_stats']['std']
            smoothed_std = result['smoothed_stats']['std']
            if original_std > 0:
                smoothing_ratio = smoothed_std / original_std
                smoothing_ratios.append(smoothing_ratio)
        
        return {
            'total_forecasts': total_forecasts,
            'mean_alpha': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'mean_smoothing_ratio': float(np.mean(smoothing_ratios)) if smoothing_ratios else 0.0,
            'smoothing_effectiveness': 'High' if np.mean(smoothing_ratios) < 0.8 else 'Medium' if np.mean(smoothing_ratios) < 0.95 else 'Low'
        }
    
    def reset_history(self):
        """Reset processing history."""
        self.alpha_history.clear()
        self.processed_forecasts.clear()
        logger.info("Processing history reset")
    
    def update_config(self, new_config: PostprocessConfig):
        """Update postprocessing configuration."""
        self.config = new_config
        logger.info(f"Updated config: {new_config.alpha_method.value}, dynamic_alpha: {new_config.enable_dynamic_alpha}")

def create_forecast_postprocessor(config: Optional[PostprocessConfig] = None) -> ForecastPostprocessor:
    """Factory function to create a forecast postprocessor."""
    return ForecastPostprocessor(config)
