"""
Advanced Hyperparameter Optimizer

Supports multiple optimization backends: Optuna (Bayesian), Random Search, and Grid Search.
Logs best parameters and provides integration with the forecasting pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from trading.models.advanced.lstm.lstm_model import LSTMModel
from trading.utils.logging_utils import setup_logger

# Try to import optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import skopt
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    skopt = None

logger = setup_logger(__name__)

class HyperparameterOptimizer:
    """Advanced hyperparameter optimizer supporting multiple backends."""
    
    def __init__(self, backend: str = "optuna", study_name: str = "evolve_optimization", 
                 storage: Optional[str] = None):
        """Initialize the optimizer.
        
        Args:
            backend: Optimization backend ('optuna', 'skopt', 'random', 'grid')
            study_name: Name for the study
            storage: Optional database URL for study storage
        """
        self.backend = backend
        self.study_name = study_name
        self.storage = storage
        self.best_params_dir = Path("models/best_params")
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend-specific components
        if backend == "optuna" and OPTUNA_AVAILABLE:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction="minimize"
            )
        elif backend == "skopt" and SKOPT_AVAILABLE:
            self.study = None  # skopt doesn't use study objects
        else:
            self.study = None
        
        logger.info(f"Hyperparameter optimizer initialized with backend: {backend}")
    
    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using the selected backend.
        
        Args:
            X: Feature matrix
            y: Target series
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and results
        """
        if self.backend == "optuna" and OPTUNA_AVAILABLE:
            return self._optimize_xgboost_optuna(X, y, n_trials)
        elif self.backend == "skopt" and SKOPT_AVAILABLE:
            return self._optimize_xgboost_skopt(X, y, n_trials)
        elif self.backend == "random":
            return self._optimize_xgboost_random(X, y, n_trials)
        elif self.backend == "grid":
            return self._optimize_xgboost_grid(X, y)
        else:
            raise ValueError(f"Backend {self.backend} not available. Install required packages.")
    
    def _optimize_xgboost_optuna(self, X: pd.DataFrame, y: pd.Series, n_trials: int) -> Dict[str, Any]:
        """Optimize XGBoost using Optuna (Bayesian optimization)."""
        def objective(trial):
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
            
            return self._evaluate_xgboost_params(X, y, params)
        
        try:
            logger.info(f"Starting XGBoost optimization with Optuna ({n_trials} trials)")
            self.study.optimize(objective, n_trials=n_trials)
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            self._save_best_params("xgboost_optuna", best_params, best_score)
            
            logger.info(f"XGBoost Optuna optimization completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "optuna",
                "optimization_history": self.study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"XGBoost Optuna optimization failed: {e}")
            return {"error": str(e)}
    
    def _optimize_xgboost_skopt(self, X: pd.DataFrame, y: pd.Series, n_trials: int) -> Dict[str, Any]:
        """Optimize XGBoost using scikit-optimize (Bayesian optimization)."""
        # Define search space
        space = [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Real(1e-8, 10.0, prior='log-uniform', name='reg_alpha'),
            Real(1e-8, 10.0, prior='log-uniform', name='reg_lambda')
        ]
        
        def objective(params):
            param_dict = {
                'n_estimators': int(params[0]),
                'max_depth': int(params[1]),
                'learning_rate': params[2],
                'subsample': params[3],
                'colsample_bytree': params[4],
                'reg_alpha': params[5],
                'reg_lambda': params[6],
                'random_state': 42
            }
            return self._evaluate_xgboost_params(X, y, param_dict)
        
        try:
            logger.info(f"Starting XGBoost optimization with scikit-optimize ({n_trials} trials)")
            result = gp_minimize(objective, space, n_calls=n_trials, random_state=42)
            
            best_params = {
                'n_estimators': int(result.x[0]),
                'max_depth': int(result.x[1]),
                'learning_rate': result.x[2],
                'subsample': result.x[3],
                'colsample_bytree': result.x[4],
                'reg_alpha': result.x[5],
                'reg_lambda': result.x[6],
                'random_state': 42
            }
            
            self._save_best_params("xgboost_skopt", best_params, result.fun)
            
            logger.info(f"XGBoost scikit-optimize completed. Best RMSE: {result.fun:.4f}")
            return {
                "best_params": best_params,
                "best_score": result.fun,
                "n_trials": n_trials,
                "backend": "skopt",
                "optimization_history": result
            }
            
        except Exception as e:
            logger.error(f"XGBoost scikit-optimize failed: {e}")
            return {"error": str(e)}
    
    def _optimize_xgboost_random(self, X: pd.DataFrame, y: pd.Series, n_trials: int) -> Dict[str, Any]:
        """Optimize XGBoost using Random Search."""
        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0],
            'reg_lambda': [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
        }
        
        try:
            logger.info(f"Starting XGBoost Random Search ({n_trials} trials)")
            
            tscv = TimeSeriesSplit(n_splits=5)
            random_search = RandomizedSearchCV(
                xgb.XGBRegressor(random_state=42),
                param_distributions=param_distributions,
                n_iter=n_trials,
                cv=tscv,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X, y)
            
            best_params = random_search.best_params_
            best_score = np.sqrt(-random_search.best_score_)
            
            self._save_best_params("xgboost_random", best_params, best_score)
            
            logger.info(f"XGBoost Random Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "random",
                "optimization_history": random_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"XGBoost Random Search failed: {e}")
            return {"error": str(e)}
    
    def _optimize_xgboost_grid(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost using Grid Search."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [1e-4, 1e-2],
            'reg_lambda': [1e-4, 1e-2]
        }
        
        try:
            logger.info("Starting XGBoost Grid Search")
            
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                xgb.XGBRegressor(random_state=42),
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            best_score = np.sqrt(-grid_search.best_score_)
            
            self._save_best_params("xgboost_grid", best_params, best_score)
            
            logger.info(f"XGBoost Grid Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": len(grid_search.cv_results_['params']),
                "backend": "grid",
                "optimization_history": grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"XGBoost Grid Search failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_xgboost_params(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> float:
        """Evaluate XGBoost parameters using time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
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
    

    
    def optimize_lstm(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters using the selected backend.
        
        Args:
            X: Feature array (samples, timesteps, features)
            y: Target array
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and results
        """
        if self.backend == "optuna" and OPTUNA_AVAILABLE:
            return self._optimize_lstm_optuna(X, y, n_trials)
        elif self.backend == "skopt" and SKOPT_AVAILABLE:
            return self._optimize_lstm_skopt(X, y, n_trials)
        elif self.backend == "random":
            return self._optimize_lstm_random(X, y, n_trials)
        elif self.backend == "grid":
            return self._optimize_lstm_grid(X, y)
        else:
            raise ValueError(f"Backend {self.backend} not available. Install required packages.")
    
    def _optimize_lstm_optuna(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize LSTM using Optuna (Bayesian optimization)."""
        def objective(trial):
            params = {
                'lstm_units': trial.suggest_int('lstm_units', 32, 256),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': trial.suggest_int('epochs', 10, 50),
                'sequence_length': trial.suggest_int('sequence_length', 10, 60)
            }
            
            return self._evaluate_lstm_params(X, y, params)
        
        try:
            logger.info(f"Starting LSTM optimization with Optuna ({n_trials} trials)")
            self.study.optimize(objective, n_trials=n_trials)
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            self._save_best_params("lstm_optuna", best_params, best_score)
            
            logger.info(f"LSTM Optuna optimization completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "optuna",
                "optimization_history": self.study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"LSTM Optuna optimization failed: {e}")
            return {"error": str(e)}
    
    def _optimize_lstm_skopt(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize LSTM using scikit-optimize (Bayesian optimization)."""
        space = [
            Integer(32, 256, name='lstm_units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
            Categorical([16, 32, 64], name='batch_size'),
            Integer(10, 50, name='epochs'),
            Integer(10, 60, name='sequence_length')
        ]
        
        def objective(params):
            param_dict = {
                'lstm_units': int(params[0]),
                'dropout_rate': params[1],
                'learning_rate': params[2],
                'batch_size': params[3],
                'epochs': int(params[4]),
                'sequence_length': int(params[5])
            }
            return self._evaluate_lstm_params(X, y, param_dict)
        
        try:
            logger.info(f"Starting LSTM optimization with scikit-optimize ({n_trials} trials)")
            result = gp_minimize(objective, space, n_calls=n_trials, random_state=42)
            
            best_params = {
                'lstm_units': int(result.x[0]),
                'dropout_rate': result.x[1],
                'learning_rate': result.x[2],
                'batch_size': result.x[3],
                'epochs': int(result.x[4]),
                'sequence_length': int(result.x[5])
            }
            
            self._save_best_params("lstm_skopt", best_params, result.fun)
            
            logger.info(f"LSTM scikit-optimize completed. Best RMSE: {result.fun:.4f}")
            return {
                "best_params": best_params,
                "best_score": result.fun,
                "n_trials": n_trials,
                "backend": "skopt",
                "optimization_history": result
            }
            
        except Exception as e:
            logger.error(f"LSTM scikit-optimize failed: {e}")
            return {"error": str(e)}
    
    def _optimize_lstm_random(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize LSTM using Random Search."""
        param_distributions = {
            'lstm_units': [32, 64, 128, 256],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'batch_size': [16, 32, 64],
            'epochs': [10, 20, 30, 40, 50],
            'sequence_length': [10, 20, 30, 40, 50, 60]
        }
        
        try:
            logger.info(f"Starting LSTM Random Search ({n_trials} trials)")
            
            # For LSTM, we need to use a custom evaluation function
            best_score = float('inf')
            best_params = None
            optimization_history = []
            
            for i in range(n_trials):
                # Randomly sample parameters
                params = {}
                for key, values in param_distributions.items():
                    params[key] = np.random.choice(values)
                
                try:
                    score = self._evaluate_lstm_params(X, y, params)
                    optimization_history.append({
                        'trial': i,
                        'params': params,
                        'score': score
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.warning(f"LSTM trial {i} failed: {e}")
                    continue
            
            if best_params is not None:
                self._save_best_params("lstm_random", best_params, best_score)
                logger.info(f"LSTM Random Search completed. Best RMSE: {best_score:.4f}")
                return {
                    "best_params": best_params,
                    "best_score": best_score,
                    "n_trials": n_trials,
                    "backend": "random",
                    "optimization_history": optimization_history
                }
            else:
                return {"error": "All LSTM trials failed"}
                
        except Exception as e:
            logger.error(f"LSTM Random Search failed: {e}")
            return {"error": str(e)}
    
    def _optimize_lstm_grid(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize LSTM using Grid Search."""
        param_grid = {
            'lstm_units': [64, 128],
            'dropout_rate': [0.2, 0.3],
            'learning_rate': [1e-3, 5e-3],
            'batch_size': [32, 64],
            'epochs': [20, 30],
            'sequence_length': [20, 40]
        }
        
        try:
            logger.info("Starting LSTM Grid Search")
            
            best_score = float('inf')
            best_params = None
            optimization_history = []
            
            # Generate all parameter combinations
            import itertools
            keys = param_grid.keys()
            values = param_grid.values()
            
            for i, combination in enumerate(itertools.product(*values)):
                params = dict(zip(keys, combination))
                
                try:
                    score = self._evaluate_lstm_params(X, y, params)
                    optimization_history.append({
                        'trial': i,
                        'params': params,
                        'score': score
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.warning(f"LSTM grid trial {i} failed: {e}")
                    continue
            
            if best_params is not None:
                self._save_best_params("lstm_grid", best_params, best_score)
                logger.info(f"LSTM Grid Search completed. Best RMSE: {best_score:.4f}")
                return {
                    "best_params": best_params,
                    "best_score": best_score,
                    "n_trials": len(optimization_history),
                    "backend": "grid",
                    "optimization_history": optimization_history
                }
            else:
                return {"error": "All LSTM grid trials failed"}
                
        except Exception as e:
            logger.error(f"LSTM Grid Search failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_lstm_params(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Evaluate LSTM parameters using time series cross-validation."""
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
            logger.warning(f"LSTM evaluation failed: {e}")
            return float('inf')  # Return high loss for failed trials
    
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

def get_optimizer(backend: str = "optuna") -> HyperparameterOptimizer:
    """Get a configured optimizer instance.
    
    Args:
        backend: Optimization backend ('optuna', 'skopt', 'random', 'grid')
        
    Returns:
        Configured hyperparameter optimizer
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = HyperparameterOptimizer(backend=backend)
    return _optimizer 