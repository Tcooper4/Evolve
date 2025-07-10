"""
XGBoost Model for Time Series Forecasting

This module provides an XGBoost-based model for time series forecasting
in the Evolve trading system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import joblib
from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost-based time series forecasting model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Default configuration
        self.config = config or {}
        self.model_params = self.config.get('model_params', {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        })
        
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    def _setup_model(self):
        """Setup XGBoost model with config-driven parameters."""
        try:
            import xgboost as xgb
            
            # Get XGBoost parameters from config or use defaults
            xgboost_config = self.config.get('xgboost', {})
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }
            
            # Merge config with defaults
            model_params = {**default_params, **xgboost_config}
            
            self.model = xgb.XGBRegressor(**model_params)
            logger.info(f"XGBoost model initialized with parameters: {model_params}")
        except ImportError as e:
            logger.error(f"XGBoost not available: {e}")
            raise ImportError("XGBoost is required for this model. Install with: pip install xgboost")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for XGBoost model.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (features, target)
        """
        try:
            # Create lag features
            lags = [1, 2, 3, 5, 10]
            features = pd.DataFrame()
            
            for lag in lags:
                features[f'lag_{lag}'] = data['close'].shift(lag)
            
            # Add technical indicators
            features['sma_5'] = data['close'].rolling(5).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['rsi'] = self._calculate_rsi(data['close'])
            features['volatility'] = data['close'].rolling(20).std()
            
            # Add time features
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            
            # Target variable
            target = data['close'].shift(-1)  # Next day's price
            
            # Remove NaN values
            features = features.dropna()
            target = target[features.index]
            
            self.feature_names = features.columns.tolist()
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the XGBoost model.
        
        Args:
            data: Training data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting XGBoost model training...")
            
            # Setup model if not already done
            if self.model is None:
                self._setup_model()
            
            # Prepare features
            features, target = self.prepare_features(data)
            
            if len(features) < 50:
                raise ValueError("Insufficient data for training (need at least 50 samples)")
            
            # Train model
            self.model.fit(features, target)
            self.is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(features)
            mse = np.mean((target - train_predictions) ** 2)
            mae = np.mean(np.abs(target - train_predictions))
            
            logger.info(f"XGBoost training completed. MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            return {
                'mse': mse,
                'mae': mae,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare features
            features, _ = self.prepare_features(data)
            
            if features.empty:
                raise ValueError("No valid features for prediction")
            
            # Make predictions
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained or self.model is None:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, filepath: str) -> bool:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_trained': self.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            
            logger.info(f"XGBoost model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Model metadata dictionary
        """
        return {
            'model_type': 'xgboost',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }