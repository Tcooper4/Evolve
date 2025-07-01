"""
Optuna Hyperparameter Optimizer

Advanced hyperparameter optimization using Optuna for XGBoost and LSTM models.
Logs best parameters and provides integration with the forecasting pipeline.
"""

import optuna
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from trading.models.advanced.lstm.lstm_model import LSTMModel
from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class OptunaOptimizer:
    """Advanced hyperparameter optimizer using Optuna."""
    
    def __init__(self, study_name: str = "evolve_optimization", storage: Optional[str] = None):
        """Initialize the optimizer.
        
        Args:
            study_name: Name for the Optuna study
            storage: Optional database URL for study storage
        """
        self.study_name = study_name
        self.storage = storage
        self.best_params_dir = Path("models/best_params")
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize"  # Minimize loss
        )
        
        logger.info(f"Optuna optimizer initialized with study: {study_name}")
    
    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target series
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and results
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Use RMSE as objective
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        try:
            logger.info(f"Starting XGBoost optimization with {n_trials} trials")
            self.study.optimize(objective, n_trials=n_trials)
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            # Save best parameters
            self._save_best_params("xgboost", best_params, best_score)
            
            logger.info(f"XGBoost optimization completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "optimization_history": self.study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
            return {"error": str(e)}
    
    def optimize_lstm(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters.
        
        Args:
            X: Feature array (samples, timesteps, features)
            y: Target array
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and results
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lstm_units': trial.suggest_int('lstm_units', 32, 256),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': trial.suggest_int('epochs', 10, 50),
                'sequence_length': trial.suggest_int('sequence_length', 10, 60)
            }
            
            try:
                # Create and train LSTM model
                model = LSTMModel(
                    input_shape=(params['sequence_length'], X.shape[2]),
                    lstm_units=params['lstm_units'],
                    dropout_rate=params['dropout_rate']
                )
                
                # Prepare data with sequence length
                X_seq, y_seq = self._prepare_sequences(X, y, params['sequence_length'])
                
                # Split data
                split_idx = int(0.8 * len(X_seq))
                X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
                
                # Train model
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                return rmse
                
            except Exception as e:
                logger.warning(f"LSTM trial failed: {e}")
                return float('inf')  # Return high loss for failed trials
        
        try:
            logger.info(f"Starting LSTM optimization with {n_trials} trials")
            self.study.optimize(objective, n_trials=n_trials)
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            # Save best parameters
            self._save_best_params("lstm", best_params, best_score)
            
            logger.info(f"LSTM optimization completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "optimization_history": self.study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"LSTM optimization failed: {e}")
            return {"error": str(e)}
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple:
        """Prepare sequences for LSTM training."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _save_best_params(self, model_type: str, params: Dict[str, Any], score: float):
        """Save best parameters to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_best_params_{timestamp}.json"
            filepath = self.best_params_dir / filename
            
            data = {
                "model_type": model_type,
                "best_params": params,
                "best_score": score,
                "timestamp": timestamp,
                "study_name": self.study_name
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Best parameters saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")
    
    def get_best_params(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the most recent best parameters for a model type."""
        try:
            pattern = f"{model_type}_best_params_*.json"
            files = list(self.best_params_dir.glob(pattern))
            
            if not files:
                return None
            
            # Get most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            return data["best_params"]
            
        except Exception as e:
            logger.error(f"Failed to load best parameters: {e}")
            return None
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title("Optimization History")
            
            # Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Parameter Importance")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")

# Global optimizer instance
_optimizer = None

def get_optimizer() -> OptunaOptimizer:
    """Get the global optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = OptunaOptimizer()
    return _optimizer 