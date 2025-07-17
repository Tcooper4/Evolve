"""
Enhanced Optuna Tuner for Model Optimization

This module provides comprehensive hyperparameter optimization using Optuna
with Sharpe ratio as the primary objective for trading model performance.

Features:
- LSTM optimization: num_layers, dropout, learning_rate, lookback
- XGBoost optimization: max_depth, learning_rate, n_estimators  
- Transformer optimization: d_model, num_heads, ff_dim, dropout
- Sharpe ratio-based objective function
- Integration with forecasting/model selection
- Comprehensive validation and backtesting
- Model persistence and result tracking
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from utils.math_utils import calculate_sharpe_ratio

logger = logging.getLogger(__name__)


class SharpeOptunaTuner:
    """
    Enhanced Optuna-based hyperparameter tuner for trading models.
    
    Uses Sharpe ratio as the primary objective and integrates with
    the forecasting/model selection pipeline.
    """
    
    def __init__(
        self,
        study_name: str = "trading_model_optimization",
        n_trials: int = 100,
        timeout: Optional[int] = 3600,
        validation_split: float = 0.2,
        random_state: int = 42,
        storage: Optional[str] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        """
        Initialize the Optuna tuner.
        
        Args:
            study_name: Name for the Optuna study
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
            validation_split: Fraction of data for validation
            random_state: Random seed for reproducibility
            storage: Optuna storage URL (e.g., "sqlite:///optuna.db")
            pruner: Optuna pruner for early stopping
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.validation_split = validation_split
        self.random_state = random_state
        self.storage = storage
        self.pruner = pruner or MedianPruner()
        
        # Results storage
        self.best_params = {}
        self.best_scores = {}
        self.optimization_history = {}
        
        # Create results directory
        self.results_dir = Path("trading/optimization/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SharpeOptunaTuner initialized: {study_name}")
    
    def _split_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and validation sets."""
        split_idx = int(len(data) * (1 - self.validation_split))
        
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        X_train = train_data.drop(columns=[target_column])
        X_val = val_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        y_val = val_data[target_column]
        
        return X_train, X_val, y_train, y_val
    
    def _calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive trading metrics including Sharpe ratio.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of trading metrics
        """
        try:
            # Calculate returns
            if len(y_true) > 1:
                actual_returns = np.diff(y_true) / y_true[:-1]
                pred_returns = np.diff(y_pred) / y_pred[:-1]
                
                # Trading signals (simple strategy)
                signals = np.where(pred_returns > 0, 1, -1)
                strategy_returns = actual_returns * signals
                
                # Sharpe ratio
                sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
                
                # Win rate
                win_rate = np.mean(strategy_returns > 0)
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Total return
                total_return = (cumulative_returns[-1] - 1) if len(cumulative_returns) > 0 else 0
                
                # Directional accuracy
                actual_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction)
                
            else:
                sharpe_ratio = 0.0
                win_rate = 0.0
                max_drawdown = 0.0
                total_return = 0.0
                directional_accuracy = 0.0
            
            # Basic error metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            
            return {
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "directional_accuracy": directional_accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {
                "sharpe_ratio": -1.0,
                "win_rate": 0.0,
                "max_drawdown": -1.0,
                "total_return": -1.0,
                "directional_accuracy": 0.0,
                "mse": float('inf'),
                "rmse": float('inf'),
                "mae": float('inf')
            }
    
    def optimize_lstm(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters using Sharpe ratio as objective.
        
        Args:
            data: Training data
            target_column: Target column name
            feature_columns: Feature column names (optional)
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info("Starting LSTM hyperparameter optimization...")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X_train, X_val, y_train, y_val = self._split_data(data, target_column)
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for LSTM optimization."""
            try:
                # Suggest hyperparameters
                params = {
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'lookback': trial.suggest_int('lookback', 10, 60),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200),
                    'sequence_length': trial.suggest_int('sequence_length', 10, 50)
                }
                
                # Import and create LSTM model
                from trading.models.lstm_model import LSTMModel
                
                # Create model with suggested parameters
                model = LSTMModel(
                    input_dim=len(feature_columns),
                    hidden_dim=params['hidden_size'],
                    output_dim=1,
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                )
                
                # Prepare sequences
                X_train_seq = self._prepare_sequences(X_train.values, params['sequence_length'])
                X_val_seq = self._prepare_sequences(X_val.values, params['sequence_length'])
                y_train_seq = y_train.values[params['sequence_length']:]
                y_val_seq = y_val.values[params['sequence_length']:]
                
                # Train model
                model.fit(
                    X_train_seq, 
                    y_train_seq, 
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    verbose=0
                )
                
                # Make predictions
                y_pred = model.predict(X_val_seq)
                
                # Calculate metrics
                metrics = self._calculate_trading_metrics(y_val_seq, y_pred)
                
                # Report intermediate value for pruning
                trial.report(metrics['sharpe_ratio'], step=params['epochs'])
                
                # Return negative Sharpe ratio (Optuna minimizes)
                return -metrics['sharpe_ratio']
                
            except Exception as e:
                logger.warning(f"LSTM trial failed: {e}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            pruner=self.pruner,
            storage=self.storage,
            study_name=f"{self.study_name}_lstm"
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.best_params['lstm'] = study.best_params
        self.best_scores['lstm'] = -study.best_value  # Convert back to positive
        self.optimization_history['lstm'] = study.trials_dataframe()
        
        logger.info(f"LSTM optimization completed. Best Sharpe ratio: {self.best_scores['lstm']:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': self.best_scores['lstm'],
            'study': study,
            'model_type': 'lstm'
        }
    
    def optimize_xgboost(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Sharpe ratio as objective.
        
        Args:
            data: Training data
            target_column: Target column name
            feature_columns: Feature column names (optional)
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info("Starting XGBoost hyperparameter optimization...")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X_train, X_val, y_train, y_val = self._split_data(data, target_column)
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for XGBoost optimization."""
            try:
                # Suggest hyperparameters
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1.0)
                }
                
                # Import and create XGBoost model
                from trading.models.xgboost_model import XGBoostModel
                
                # Create model with suggested parameters
                model = XGBoostModel(params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = self._calculate_trading_metrics(y_val.values, y_pred)
                
                # Return negative Sharpe ratio (Optuna minimizes)
                return -metrics['sharpe_ratio']
                
            except Exception as e:
                logger.warning(f"XGBoost trial failed: {e}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            pruner=self.pruner,
            storage=self.storage,
            study_name=f"{self.study_name}_xgboost"
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.best_params['xgboost'] = study.best_params
        self.best_scores['xgboost'] = -study.best_value  # Convert back to positive
        self.optimization_history['xgboost'] = study.trials_dataframe()
        
        logger.info(f"XGBoost optimization completed. Best Sharpe ratio: {self.best_scores['xgboost']:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': self.best_scores['xgboost'],
            'study': study,
            'model_type': 'xgboost'
        }
    
    def optimize_transformer(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize Transformer hyperparameters using Sharpe ratio as objective.
        
        Args:
            data: Training data
            target_column: Target column name
            feature_columns: Feature column names (optional)
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info("Starting Transformer hyperparameter optimization...")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X_train, X_val, y_train, y_val = self._split_data(data, target_column)
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Transformer optimization."""
            try:
                # Suggest hyperparameters
                params = {
                    'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
                    'num_heads': trial.suggest_int('num_heads', 2, 16),
                    'ff_dim': trial.suggest_categorical('ff_dim', [128, 256, 512, 1024]),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'num_layers': trial.suggest_int('num_layers', 1, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200),
                    'sequence_length': trial.suggest_int('sequence_length', 10, 50)
                }
                
                # Import and create Transformer model
                from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
                
                # Create model with suggested parameters
                model = TransformerForecaster(params)
                
                # Prepare sequences
                X_train_seq = self._prepare_sequences(X_train.values, params['sequence_length'])
                X_val_seq = self._prepare_sequences(X_val.values, params['sequence_length'])
                y_train_seq = y_train.values[params['sequence_length']:]
                y_val_seq = y_val.values[params['sequence_length']:]
                
                # Train model
                model.fit(
                    X_train_seq, 
                    y_train_seq, 
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    verbose=0
                )
                
                # Make predictions
                y_pred = model.predict(X_val_seq)
                
                # Calculate metrics
                metrics = self._calculate_trading_metrics(y_val_seq, y_pred)
                
                # Report intermediate value for pruning
                trial.report(metrics['sharpe_ratio'], step=params['epochs'])
                
                # Return negative Sharpe ratio (Optuna minimizes)
                return -metrics['sharpe_ratio']
                
            except Exception as e:
                logger.warning(f"Transformer trial failed: {e}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            pruner=self.pruner,
            storage=self.storage,
            study_name=f"{self.study_name}_transformer"
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.best_params['transformer'] = study.best_params
        self.best_scores['transformer'] = -study.best_value  # Convert back to positive
        self.optimization_history['transformer'] = study.trials_dataframe()
        
        logger.info(f"Transformer optimization completed. Best Sharpe ratio: {self.best_scores['transformer']:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': self.best_scores['transformer'],
            'study': study,
            'model_type': 'transformer'
        }
    
    def _prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Prepare sequences for time series models."""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def optimize_all_models(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize all model types and return the best one.
        
        Args:
            data: Training data
            target_column: Target column name
            model_types: List of model types to optimize (default: all)
            
        Returns:
            Dictionary with best model and results
        """
        if model_types is None:
            model_types = ['lstm', 'xgboost', 'transformer']
        
        results = {}
        
        for model_type in model_types:
            try:
                if model_type == 'lstm':
                    results[model_type] = self.optimize_lstm(data, target_column)
                elif model_type == 'xgboost':
                    results[model_type] = self.optimize_xgboost(data, target_column)
                elif model_type == 'transformer':
                    results[model_type] = self.optimize_transformer(data, target_column)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
                results[model_type] = {'best_score': -1.0, 'error': str(e)}
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k].get('best_score', -1.0))
        best_score = results[best_model].get('best_score', -1.0)
        
        logger.info(f"Best model: {best_model} with Sharpe ratio: {best_score:.4f}")
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'all_results': results,
            'recommendation': {
                'model_type': best_model,
                'params': results[best_model].get('best_params', {}),
                'expected_sharpe': best_score
            }
        }
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save optimization results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sharpe_optimization_results_{timestamp}.pkl"
        
        filepath = self.results_dir / filename
        
        results = {
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now().isoformat(),
            'study_name': self.study_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from file."""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.best_params = results.get('best_params', {})
        self.best_scores = results.get('best_scores', {})
        self.optimization_history = results.get('optimization_history', {})
        
        logger.info(f"Results loaded from: {filepath}")
        return results
    
    def get_best_params(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get best parameters for a specific model type."""
        return self.best_params.get(model_type)
    
    def get_best_score(self, model_type: str) -> Optional[float]:
        """Get best score for a specific model type."""
        return self.best_scores.get(model_type)
    
    def get_model_recommendation(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Get model recommendation for given data.
        
        Args:
            data: Training data
            target_column: Target column name
            
        Returns:
            Dictionary with model recommendation
        """
        # Quick evaluation of different models
        results = self.optimize_all_models(data, target_column)
        
        best_model = results['best_model']
        best_score = results['best_score']
        best_params = results['all_results'][best_model]['best_params']
        
        return {
            'recommended_model': best_model,
            'expected_sharpe': best_score,
            'parameters': best_params,
            'confidence': 'high' if best_score > 0.5 else 'medium' if best_score > 0.2 else 'low',
            'reasoning': f"Model {best_model} achieved the highest Sharpe ratio of {best_score:.4f}"
        }


def get_sharpe_optuna_tuner(
    study_name: str = "trading_optimization",
    n_trials: int = 100,
    timeout: int = 3600
) -> SharpeOptunaTuner:
    """
    Get a configured Sharpe Optuna tuner instance.
    
    Args:
        study_name: Name for the study
        n_trials: Number of trials
        timeout: Timeout in seconds
        
    Returns:
        Configured SharpeOptunaTuner instance
    """
    return SharpeOptunaTuner(
        study_name=study_name,
        n_trials=n_trials,
        timeout=timeout
    ) 