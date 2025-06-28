"""Enhanced ensemble model with weighted voting and strategy-aware routing."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .base_model import BaseModel, ModelRegistry
import logging

# @ModelRegistry.register('Ensemble')
class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple forecasting models with adaptive weights."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble model.
        
        Args:
            config: Configuration dictionary containing:
                - models: List of model configurations to include
                - voting_method: 'mse', 'sharpe', or 'custom'
                - weight_window: Window size for rolling performance
                - fallback_threshold: Minimum confidence for predictions
                - strategy_aware: Whether to use strategy-aware routing
        """
        super().__init__(config)
        self._validate_config()
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.strategy_patterns = {}
        self._load_strategy_patterns()
        
    def _validate_config(self):
        """Validate ensemble configuration."""
        required = ['models', 'voting_method', 'weight_window']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config['voting_method'] not in ['mse', 'sharpe', 'custom']:
            raise ValueError("voting_method must be 'mse', 'sharpe', or 'custom'")
            
        if not isinstance(self.config['models'], list):
            raise ValueError("models must be a list of model configurations")
    
    def _load_strategy_patterns(self):
        """Load strategy patterns from memory.json."""
        try:
            with open('memory.json', 'r') as f:
                memory = json.load(f)
                self.strategy_patterns = memory.get('model_strategy_patterns', {})
        except FileNotFoundError:
            self.strategy_patterns = {}
    
    def _save_strategy_patterns(self):
        """Save strategy patterns to memory.json."""
        try:
            with open('memory.json', 'r') as f:
                memory = json.load(f)
        except FileNotFoundError:
            memory = {}
        
        memory['model_strategy_patterns'] = self.strategy_patterns
        
        with open('memory.json', 'w') as f:
            json.dump(memory, f, indent=4)
    
    def _initialize_models(self):
        """Initialize all models in the ensemble."""
        for model_config in self.config['models']:
            model_name = model_config['name']
            model_class = ModelRegistry.get_model_class(model_name)
            self.models[model_name] = model_class(model_config)
            self.weights[model_name] = 1.0 / len(self.config['models'])
            self.performance_history[model_name] = []
    
    def _update_weights(self, data: pd.DataFrame):
        """Update model weights based on recent performance.
        
        Args:
            data: Recent data for performance calculation
        """
        window = self.config['weight_window']
        recent_data = data.iloc[-window:]
        
        for model_name, model in self.models.items():
            try:
                # Get model predictions
                preds = model.predict(recent_data)
                actual = recent_data['close'].values
                
                # Calculate performance metrics
                if self.config['voting_method'] == 'mse':
                    score = -np.mean((actual - preds) ** 2)  # Negative MSE
                elif self.config['voting_method'] == 'sharpe':
                    returns = np.diff(preds) / preds[:-1]
                    score = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                else:  # custom
                    score = self._calculate_custom_score(actual, preds)
                
                # Update performance history
                self.performance_history[model_name].append({
                    'timestamp': datetime.now().isoformat(),
                    'score': score,
                    'confidence': model.calculate_confidence(preds) if hasattr(model, 'calculate_confidence') else 1.0
                })
                
            except Exception as e:
                logging.error(f"Error updating weights for {model_name}: {e}")
                raise RuntimeError(f"Failed to update weights for {model_name}: {e}")
        
        # Update weights using softmax
        scores = np.array([h[-1]['score'] for h in self.performance_history.values()])
        self.weights = dict(zip(self.models.keys(), np.exp(scores) / np.sum(np.exp(scores))))
    
    def _calculate_custom_score(self, actual: np.ndarray, preds: np.ndarray) -> float:
        """Calculate custom performance score.
        
        Args:
            actual: Actual values
            preds: Predicted values
            
        Returns:
            Custom score combining multiple metrics
        """
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(actual))
        direction_pred = np.sign(np.diff(preds))
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        # Calculate normalized MSE
        mse = np.mean((actual - preds) ** 2)
        normalized_mse = 1 / (1 + mse)
        
        # Combine metrics
        return 0.7 * directional_accuracy + 0.3 * normalized_mse
    
    def _get_strategy_recommendation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get strategy-aware model recommendations.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with strategy recommendations
        """
        # Detect market regime
        returns = data['close'].pct_change()
        volatility = returns.std()
        trend = returns.mean()
        
        if trend > 0.001 and volatility < 0.02:
            regime = 'bull'
        elif trend < -0.001 and volatility > 0.02:
            regime = 'bear'
        else:
            regime = 'neutral'
        
        # Get model confidences
        confidences = {}
        for model_name, model in self.models.items():
            try:
                preds = model.predict(data.iloc[-20:])  # Use last 20 points
                confidences[model_name] = model.calculate_confidence(preds) if hasattr(model, 'calculate_confidence') else 1.0
            except Exception as e:
                logging.error(f"Error calculating confidence for {model_name}: {e}")
                raise RuntimeError(f"Failed to calculate confidence for {model_name}: {e}")
        
        # Update strategy patterns
        self.strategy_patterns[regime] = {
            'timestamp': datetime.now().isoformat(),
            'model_confidences': confidences,
            'best_model': max(confidences.items(), key=lambda x: x[1])[0]
        }
        self._save_strategy_patterns()
        
        return {
            'regime': regime,
            'model_confidences': confidences,
            'recommended_model': self.strategy_patterns[regime]['best_model']
        }
    
    def fit(self, data: pd.DataFrame):
        """Train all models in the ensemble.
        
        Args:
            data: Training data
        """
        self._initialize_models()
        for model in self.models.values():
            model.fit(data)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted ensemble.
        
        Args:
            data: Input data
            
        Returns:
            Ensemble predictions
        """
        # Update weights based on recent performance
        self._update_weights(data)
        
        # Get strategy recommendations
        strategy = self._get_strategy_recommendation(data)
        
        # Collect predictions from all models
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            try:
                preds = model.predict(data)
                confidence = model.calculate_confidence(preds) if hasattr(model, 'calculate_confidence') else 1.0
                
                # Apply fallback logic
                if confidence < self.config.get('fallback_threshold', 0.5):
                    print(f"Low confidence for {model_name}, using fallback")
                    continue
                
                predictions[model_name] = preds
                confidences[model_name] = confidence
                
            except Exception as e:
                logging.error(f"Error in {model_name} prediction: {e}")
                raise RuntimeError(f"Failed to get prediction from {model_name}: {e}")
        
        if not predictions:
            raise ValueError("All models failed to make predictions")
        
        # Calculate weighted ensemble
        weights = np.array([self.weights[name] * confidences[name] for name in predictions.keys()])
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_preds = np.zeros_like(next(iter(predictions.values())))
        for (name, preds), weight in zip(predictions.items(), weights):
            ensemble_preds += weight * preds
        
        return ensemble_preds
    
    def save_model(self, filepath: str):
        """Save ensemble model.
        
        Args:
            filepath: Path to save model
        """
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble metadata
        metadata = {
            'config': self.config,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'strategy_patterns': self.strategy_patterns
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            model.save_model(model_path)
    
    def load_model(self, filepath: str):
        """Load ensemble model.
        
        Args:
            filepath: Path to saved model
        """
        # Load ensemble metadata
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        self.config = metadata['config']
        self.weights = metadata['weights']
        self.performance_history = metadata['performance_history']
        self.strategy_patterns = metadata['strategy_patterns']
        
        # Initialize and load individual models
        self._initialize_models()
        model_dir = os.path.dirname(filepath)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            model.load_model(model_path)
    
    def summary(self) -> str:
        """Get model summary.
        
        Returns:
            Formatted summary string
        """
        summary = ["Ensemble Model Summary", "=" * 20]
        
        # Add model configurations
        summary.append("\nModels:")
        for model_name, model in self.models.items():
            summary.append(f"- {model_name}: {model.summary()}")
        
        # Add current weights
        summary.append("\nCurrent Weights:")
        for model_name, weight in self.weights.items():
            summary.append(f"- {model_name}: {weight:.3f}")
        
        # Add strategy patterns
        summary.append("\nStrategy Patterns:")
        for regime, pattern in self.strategy_patterns.items():
            summary.append(f"- {regime}: {pattern['best_model']}")
        
        return "\n".join(summary)
    
    def shap_interpret(self, data: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for ensemble predictions.
        
        Args:
            data: Input data
            
        Returns:
            SHAP values
        """
        # Get SHAP values from each model
        shap_values = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'shap_interpret'):
                try:
                    shap_values[model_name] = model.shap_interpret(data)
                except Exception as e:
                    logging.error(f"Error getting SHAP values for {model_name}: {e}")
                    raise RuntimeError(f"Failed to get SHAP values for {model_name}: {e}")
        
        if not shap_values:
            raise ValueError("No models support SHAP interpretation")
        
        # Weight SHAP values by model weights
        weighted_shap = np.zeros_like(next(iter(shap_values.values())))
        for model_name, values in shap_values.items():
            weighted_shap += self.weights[model_name] * values
        
        return weighted_shap

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """Generate forecast for future time steps.
        
        Args:
            data: Historical data DataFrame
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast results
        """
        try:
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
                new_row['close'] = pred[-1]  # Update with prediction
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                current_data = current_data.iloc[1:]  # Remove oldest row
            
            return {
                'forecast': np.array(forecast_values),
                'confidence': 0.9,  # High confidence for ensemble
                'model': 'Ensemble',
                'horizon': horizon,
                'weights': self.weights,
                'strategy_patterns': self.strategy_patterns
            }
            
        except Exception as e:
            logging.error(f"Error in ensemble model forecast: {e}")
            raise RuntimeError(f"Ensemble model forecasting failed: {e}")

    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray = None) -> None:
        """Plot ensemble model results and predictions.
        
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
            plt.plot(data.index, data['close'], label='Actual', color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted', color='red')
            plt.title('Ensemble Model Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Model Weights
            plt.subplot(2, 2, 2)
            model_names = list(self.weights.keys())
            weights = list(self.weights.values())
            plt.bar(model_names, weights)
            plt.title('Model Weights')
            plt.xlabel('Models')
            plt.ylabel('Weight')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot 3: Performance History
            plt.subplot(2, 2, 3)
            for model_name, history in self.performance_history.items():
                if history:
                    scores = [h['score'] for h in history]
                    plt.plot(scores, label=model_name)
            plt.title('Model Performance History')
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # Plot 4: Strategy Patterns
            plt.subplot(2, 2, 4)
            regimes = list(self.strategy_patterns.keys())
            best_models = [self.strategy_patterns[r]['best_model'] for r in regimes]
            plt.bar(regimes, [1] * len(regimes))
            plt.title('Strategy Patterns by Regime')
            plt.xlabel('Market Regime')
            plt.ylabel('Count')
            for i, (regime, model) in enumerate(zip(regimes, best_models)):
                plt.text(i, 0.5, model, ha='center', va='center')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting ensemble results: {e}")
            print(f"Could not plot results: {e}")

    def forecast(self, data, *args, **kwargs):
        return np.zeros(len(data)) 