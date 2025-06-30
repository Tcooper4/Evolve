"""XGBoostForecaster: XGBoost wrapper for time series forecasting."""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from .base_model import BaseModel, ValidationError, ModelRegistry

class XGBoostForecaster(BaseModel):
    """XGBoost model for time series forecasting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        if config is None:
            config = {}
        
        # Set default configuration
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'sequence_length': 10
        }
        default_config.update(config)
        
        super().__init__(default_config)
        self._validate_config()
        self._setup_model()
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        required_params = ['feature_columns', 'target_column', 'sequence_length']
        for param in required_params:
            if param not in self.config:
                raise ValidationError(f"Missing required parameter: {param}")
    
    def _setup_model(self) -> None:
        """Setup the XGBoost model."""
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=self.config.get('random_state', 42)
            )
        except ImportError:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data.
        
        Args:
            data: Training data
        """
        X, y = self._prepare_data(data, is_training=True)
        self.model.fit(X, y)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        X, _ = self._prepare_data(data, is_training=False)
        return self.model.predict(X)
    
    def _prepare_data(self, data: pd.DataFrame, is_training: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training or prediction.
        
        Args:
            data: Input data
            is_training: Whether data is for training
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Validate data
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")
        
        # Check if all required columns exist
        missing_cols = [col for col in self.config['feature_columns'] 
                       if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Create sequences
        X_sequences = []
        y_values = []
        
        for i in range(len(data) - self.config['sequence_length']):
            X_seq = data[self.config['feature_columns']].iloc[i:i + self.config['sequence_length']].values
            X_sequences.append(X_seq.flatten())  # Flatten for XGBoost
            y_values.append(data[self.config['target_column']].iloc[i + self.config['sequence_length']])
        
        X = np.array(X_sequences)
        y = np.array(y_values)
        
        return X, y
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_names = []
        for i in range(self.config['sequence_length']):
            for col in self.config['feature_columns']:
                feature_names.append(f"{col}_t{i}")
        
        return dict(zip(feature_names, self.model.feature_importances_))

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.
        
        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            if not self.is_fitted:
                raise RuntimeError('Model must be fitted before forecasting.')
            
            # Make initial prediction
            predictions = self.predict(data)
            
            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()
            
            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])
                
                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.config.get('target_column', 'close')] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.85,  # XGBoost confidence
                'model': 'XGBoost',
                'horizon': horizon,
                'feature_importance': self.get_feature_importance()
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in XGBoost model forecast: {e}")
            raise RuntimeError(f"XGBoost model forecasting failed: {e}")

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot XGBoost model results and predictions.
        
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
            target_col = self.config.get('target_column', 'close')
            plt.plot(data.index, data[target_col], label='Actual', color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted', color='red')
            plt.title('XGBoost Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Feature importance
            plt.subplot(2, 2, 2)
            importance = self.get_feature_importance()
            if importance:
                # Get top 10 features
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_names = [f[0] for f in top_features]
                importance_values = [f[1] for f in top_features]
                plt.barh(range(len(feature_names)), importance_values)
                plt.yticks(range(len(feature_names)), feature_names)
                plt.title('Top 10 Feature Importance')
                plt.xlabel('Importance')
            else:
                plt.text(0.5, 0.5, 'Feature importance not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Importance')
            plt.grid(True)
            
            # Plot 3: Prediction residuals
            plt.subplot(2, 2, 3)
            if len(predictions) == len(data):
                residuals = data[target_col].values - predictions
                plt.plot(residuals)
                plt.title('Prediction Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'Residuals not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Prediction Residuals')
            
            # Plot 4: Model information
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f'Model: XGBoost', fontsize=12)
            plt.text(0.1, 0.6, f'Features: {len(self.config.get("feature_columns", []))}', fontsize=12)
            plt.text(0.1, 0.4, f'Target: {target_col}', fontsize=12)
            plt.text(0.1, 0.2, f'Fitted: {self.is_fitted}', fontsize=12)
            plt.title('Model Information')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            import logging
            logging.error(f"Error plotting XGBoost results: {e}")
            print(f"Could not plot results: {e}")

class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting (alias for XGBoostForecaster)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.forecaster = XGBoostForecaster(config)
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'XGBoostModel':
        """Fit the model to the data.
        
        Args:
            data: Training data
            
        Returns:
            Self for chaining
        """
        self.forecaster.fit(data)
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.forecaster.predict(data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        return self.forecaster.get_feature_importance()

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.
        
        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            if not self.is_fitted:
                raise RuntimeError('Model must be fitted before forecasting.')
            
            # Make initial prediction
            predictions = self.predict(data)
            
            # Generate multi-step forecast
            forecast_values = []
            current_data = data.copy()
            
            for i in range(horizon):
                # Get prediction for next step
                pred = self.predict(current_data)
                forecast_values.append(pred[-1])
                
                # Update data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row[self.config.get('target_column', 'close')] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.85,  # XGBoost confidence
                'model': 'XGBoost',
                'horizon': horizon,
                'feature_importance': self.get_feature_importance()
            }
            
        except Exception as e:
            import logging
            logging.error(f"Error in XGBoost model forecast: {e}")
            raise RuntimeError(f"XGBoost model forecasting failed: {e}")

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot XGBoost model results and predictions.
        
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
            target_col = self.config.get('target_column', 'close')
            plt.plot(data.index, data[target_col], label='Actual', color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted', color='red')
            plt.title('XGBoost Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Feature importance
            plt.subplot(2, 2, 2)
            importance = self.get_feature_importance()
            if importance:
                # Get top 10 features
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_names = [f[0] for f in top_features]
                importance_values = [f[1] for f in top_features]
                plt.barh(range(len(feature_names)), importance_values)
                plt.yticks(range(len(feature_names)), feature_names)
                plt.title('Top 10 Feature Importance')
                plt.xlabel('Importance')
            else:
                plt.text(0.5, 0.5, 'Feature importance not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Importance')
            plt.grid(True)
            
            # Plot 3: Prediction residuals
            plt.subplot(2, 2, 3)
            if len(predictions) == len(data):
                residuals = data[target_col].values - predictions
                plt.plot(residuals)
                plt.title('Prediction Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'Residuals not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Prediction Residuals')
            
            # Plot 4: Model information
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f'Model: XGBoost', fontsize=12)
            plt.text(0.1, 0.6, f'Features: {len(self.config.get("feature_columns", []))}', fontsize=12)
            plt.text(0.1, 0.4, f'Target: {target_col}', fontsize=12)
            plt.text(0.1, 0.2, f'Fitted: {self.is_fitted}', fontsize=12)
            plt.title('Model Information')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            import logging
            logging.error(f"Error plotting XGBoost results: {e}")
            print(f"Could not plot results: {e}")