import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for LSTM, XGBoost, and Transformer models.
    Uses a historical validation split to evaluate each trial.
    """
    def __init__(
        self,
        study_name: str = "model_optimization",
        n_trials: int = 50,
        timeout: Optional[int] = 1800,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.validation_split = validation_split
        self.random_state = random_state
        self.study = None
        self.best_params = {}
        self.best_scores = {}
        logger.info(f"OptunaTuner initialized: {study_name}")

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(X) * (1 - self.validation_split))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def optimize_lstm(self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters."""
        logger.info("Starting LSTM hyperparameter optimization...")
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values

        def objective(trial: optuna.Trial) -> float:
            params = {
                'layers': trial.suggest_int('layers', 1, 3),
                'units': trial.suggest_categorical('units', [32, 64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'lookback': trial.suggest_int('lookback', 10, 60),
                'epochs': trial.suggest_int('epochs', 20, 100),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
            }
            try:
                from trading.models.lstm_model import LSTMModel
                X_train, X_val, y_train, y_val = self._split_data(X, y)
                model = LSTMModel(params)
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])
                preds = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                return rmse
            except Exception as e:
                logger.warning(f"LSTM trial failed: {e}")
                return float('inf')

        self.study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_state))
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params['lstm'] = self.study.best_params
        self.best_scores['lstm'] = self.study.best_value
        logger.info(f"LSTM optimization completed. Best score: {self.study.best_value:.6f}")
        return {'best_params': self.study.best_params, 'best_score': self.study.best_value}

    def optimize_xgboost(self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        logger.info("Starting XGBoost hyperparameter optimization...")
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values

        def objective(trial: optuna.Trial) -> float:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            try:
                from trading.models.xgboost_model import XGBoostModel
                X_train, X_val, y_train, y_val = self._split_data(X, y)
                model = XGBoostModel(params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                return rmse
            except Exception as e:
                logger.warning(f"XGBoost trial failed: {e}")
                return float('inf')

        self.study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_state))
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params['xgboost'] = self.study.best_params
        self.best_scores['xgboost'] = self.study.best_value
        logger.info(f"XGBoost optimization completed. Best score: {self.study.best_value:.6f}")
        return {'best_params': self.study.best_params, 'best_score': self.study.best_value}

    def optimize_transformer(self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize Transformer hyperparameters."""
        logger.info("Starting Transformer hyperparameter optimization...")
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values

        def objective(trial: optuna.Trial) -> float:
            params = {
                'num_heads': trial.suggest_int('num_heads', 2, 8),
                'depth': trial.suggest_int('depth', 1, 6),
                'feedforward_dim': trial.suggest_categorical('feedforward_dim', [64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': trial.suggest_int('epochs', 20, 100),
                'sequence_length': trial.suggest_int('sequence_length', 10, 50)
            }
            try:
                from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
                X_train, X_val, y_train, y_val = self._split_data(X, y)
                model = TransformerForecaster(params)
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])
                preds = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                return rmse
            except Exception as e:
                logger.warning(f"Transformer trial failed: {e}")
                return float('inf')

        self.study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_state))
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params['transformer'] = self.study.best_params
        self.best_scores['transformer'] = self.study.best_value
        logger.info(f"Transformer optimization completed. Best score: {self.study.best_value:.6f}")
        return {'best_params': self.study.best_params, 'best_score': self.study.best_value}

    def optimize_prophet(self, data: pd.DataFrame, target_column: str, date_column: str = 'ds') -> Dict[str, Any]:
        """Optimize Prophet hyperparameters."""
        logger.info("Starting Prophet hyperparameter optimization...")
        
        # Prepare data for Prophet (requires 'ds' and 'y columns)
        prophet_data = data.copy()
        if date_column != 'ds':
            prophet_data['ds'] = prophet_data[date_column]
        prophet_data['y'] = prophet_data[target_column]
        
        # Split data for validation
        split_idx = int(len(prophet_data) * (1 - self.validation_split))
        train_data = prophet_data.iloc[:split_idx]
        val_data = prophet_data.iloc[split_idx:]

        def objective(trial: optuna.Trial) -> float:
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.1, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.1),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
                'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False, 3, 5]),
                'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False, 12])
            }
            try:
                from prophet import Prophet
                model = Prophet(**params)
                model.fit(train_data)
                
                # Make predictions on validation set
                future = model.make_future_dataframe(periods=len(val_data))
                forecast = model.predict(future)
                
                # Calculate RMSE on validation set
                val_predictions = forecast.iloc[-len(val_data):]['yhat'].values
                val_actual = val_data['y'].values
                rmse = np.sqrt(np.mean((val_actual - val_predictions) ** 2))
                return rmse
            except Exception as e:
                logger.warning(f"Prophet trial failed: {e}")
                return float('inf')

        self.study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_state))
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params['prophet'] = self.study.best_params
        self.best_scores['prophet'] = self.study.best_value
        logger.info(f"Prophet optimization completed. Best score: {self.study.best_value:.6f}")
        return {'best_params': self.study.best_params, 'best_score': self.study.best_value}

    def get_best_params(self, model_type: str) -> Optional[Dict[str, Any]]:
        return self.best_params.get(model_type)

    def get_best_score(self, model_type: str) -> Optional[float]:
        return self.best_scores.get(model_type) 