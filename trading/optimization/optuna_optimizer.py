"""
Advanced Hyperparameter Optimizer

Supports multiple optimization backends: Optuna (Bayesian), Random Search, and Grid Search.
Logs best parameters and provides integration with the forecasting pipeline.
Enhanced with parameter importance analysis and comprehensive trial logging.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

from trading.models.lstm_model import LSTMModel
from trading.utils.logging_utils import setup_logger

# Try to import optional dependencies
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import skopt
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    skopt = None

logger = setup_logger(__name__)


class HyperparameterOptimizer:
    """Advanced hyperparameter optimizer supporting multiple backends."""

    def __init__(
        self,
        backend: str = "optuna",
        study_name: str = "evolve_optimization",
        storage: Optional[str] = None,
        save_top_trials: int = 10,
    ):
        """Initialize the optimizer.

        Args:
            backend: Optimization backend ('optuna', 'skopt', 'random', 'grid')
            study_name: Name for the study
            storage: Optional database URL for study storage
            save_top_trials: Number of top trials to save
        """
        self.backend = backend
        self.study_name = study_name
        self.storage = storage
        self.save_top_trials = save_top_trials

        # Create directories
        self.best_params_dir = Path("models/best_params")
        self.best_params_dir.mkdir(parents=True, exist_ok=True)

        self.trials_dir = Path("models/optimization_trials")
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path("models/optimization_plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend-specific components
        if backend == "optuna" and OPTUNA_AVAILABLE:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction="minimize",
            )
        elif backend == "skopt" and SKOPT_AVAILABLE:
            self.study = None  # skopt doesn't use study objects
        else:
            self.study = None

        logger.info(f"Hyperparameter optimizer initialized with backend: {backend}")

    def optimize_xgboost(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100
    ) -> Dict[str, Any]:
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
            raise ValueError(
                f"Backend {self.backend} not available. Install required packages."
            )

    def _optimize_xgboost_optuna(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize XGBoost using Optuna (Bayesian optimization)."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
            }

            return self._evaluate_xgboost_params(X, y, params)

        try:
            logger.info(
                f"Starting XGBoost optimization with Optuna ({n_trials} trials)"
            )
            self.study.optimize(objective, n_trials=n_trials)

            best_params = self.study.best_params
            best_score = self.study.best_value

            # Save best parameters
            self._save_best_params("xgboost_optuna", best_params, best_score)

            # Save top trials
            self._save_top_trials("xgboost_optuna")

            # Generate and save parameter importance plots
            self._generate_parameter_importance_plots("xgboost_optuna")

            # Log parameter importance
            self._log_parameter_importance("xgboost_optuna")

            logger.info(
                f"XGBoost Optuna optimization completed. Best RMSE: {best_score:.4f}"
            )
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "optuna",
                "optimization_history": self.study.trials_dataframe(),
                "parameter_importance": self._get_parameter_importance(),
                "top_trials": self._get_top_trials(),
            }

        except Exception as e:
            logger.error(f"XGBoost Optuna optimization failed: {e}")
            return {"error": str(e)}

    def _optimize_xgboost_skopt(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize XGBoost using scikit-optimize (Bayesian optimization)."""
        # Define search space
        space = [
            Integer(50, 500, name="n_estimators"),
            Integer(3, 10, name="max_depth"),
            Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
            Real(0.6, 1.0, name="subsample"),
            Real(0.6, 1.0, name="colsample_bytree"),
            Real(1e-8, 10.0, prior="log-uniform", name="reg_alpha"),
            Real(1e-8, 10.0, prior="log-uniform", name="reg_lambda"),
        ]

        def objective(params):
            param_dict = {
                "n_estimators": int(params[0]),
                "max_depth": int(params[1]),
                "learning_rate": params[2],
                "subsample": params[3],
                "colsample_bytree": params[4],
                "reg_alpha": params[5],
                "reg_lambda": params[6],
                "random_state": 42,
            }
            return self._evaluate_xgboost_params(X, y, param_dict)

        try:
            logger.info(
                f"Starting XGBoost optimization with scikit-optimize ({n_trials} trials)"
            )
            result = gp_minimize(objective, space, n_calls=n_trials, random_state=42)

            best_params = {
                "n_estimators": int(result.x[0]),
                "max_depth": int(result.x[1]),
                "learning_rate": result.x[2],
                "subsample": result.x[3],
                "colsample_bytree": result.x[4],
                "reg_alpha": result.x[5],
                "reg_lambda": result.x[6],
                "random_state": 42,
            }

            self._save_best_params("xgboost_skopt", best_params, result.fun)

            # Save optimization results
            self._save_skopt_results("xgboost_skopt", result, best_params)

            logger.info(
                f"XGBoost scikit-optimize completed. Best RMSE: {result.fun:.4f}"
            )
            return {
                "best_params": best_params,
                "best_score": result.fun,
                "n_trials": n_trials,
                "backend": "skopt",
                "optimization_history": result,
            }

        except Exception as e:
            logger.error(f"XGBoost scikit-optimize failed: {e}")
            return {"error": str(e)}

    def _optimize_xgboost_random(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize XGBoost using Random Search."""
        param_distributions = {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0],
            "reg_lambda": [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0],
        }

        try:
            logger.info(f"Starting XGBoost Random Search ({n_trials} trials)")

            tscv = TimeSeriesSplit(n_splits=5)
            random_search = RandomizedSearchCV(
                xgb.XGBRegressor(random_state=42),
                param_distributions=param_distributions,
                n_iter=n_trials,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
            )

            random_search.fit(X, y)

            best_params = random_search.best_params_
            best_score = np.sqrt(-random_search.best_score_)

            self._save_best_params("xgboost_random", best_params, best_score)

            # Save random search results
            self._save_random_search_results("xgboost_random", random_search)

            logger.info(f"XGBoost Random Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "random",
                "optimization_history": random_search.cv_results_,
            }

        except Exception as e:
            logger.error(f"XGBoost Random Search failed: {e}")
            return {"error": str(e)}

    def _optimize_xgboost_grid(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost using Grid Search."""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [1e-4, 1e-2],
            "reg_lambda": [1e-4, 1e-2],
        }

        try:
            logger.info("Starting XGBoost Grid Search")

            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                xgb.XGBRegressor(random_state=42),
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )

            grid_search.fit(X, y)

            best_params = grid_search.best_params_
            best_score = np.sqrt(-grid_search.best_score_)

            self._save_best_params("xgboost_grid", best_params, best_score)

            # Save grid search results
            self._save_grid_search_results("xgboost_grid", grid_search)

            logger.info(f"XGBoost Grid Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "backend": "grid",
                "optimization_history": grid_search.cv_results_,
            }

        except Exception as e:
            logger.error(f"XGBoost Grid Search failed: {e}")
            return {"error": str(e)}

    def _evaluate_xgboost_params(
        self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]
    ) -> float:
        """Evaluate XGBoost parameters using time series cross-validation."""
        try:
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

        except Exception as e:
            logger.error(f"XGBoost evaluation error: {e}")
            return float("inf")

    def optimize_lstm(
        self, X: np.ndarray, y: np.ndarray, n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters using the selected backend."""
        if self.backend == "optuna" and OPTUNA_AVAILABLE:
            return self._optimize_lstm_optuna(X, y, n_trials)
        elif self.backend == "skopt" and SKOPT_AVAILABLE:
            return self._optimize_lstm_skopt(X, y, n_trials)
        elif self.backend == "random":
            return self._optimize_lstm_random(X, y, n_trials)
        elif self.backend == "grid":
            return self._optimize_lstm_grid(X, y)
        else:
            raise ValueError(
                f"Backend {self.backend} not available. Install required packages."
            )

    def _optimize_lstm_optuna(
        self, X: np.ndarray, y: np.ndarray, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize LSTM using Optuna."""

        def objective(trial):
            params = {
                "lstm_units": trial.suggest_int("lstm_units", 32, 256),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.01, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16, 32, 64, 128]
                ),
                "epochs": trial.suggest_int("epochs", 50, 200),
                "sequence_length": trial.suggest_int("sequence_length", 10, 50),
            }

            return self._evaluate_lstm_params(X, y, params)

        try:
            logger.info(f"Starting LSTM optimization with Optuna ({n_trials} trials)")
            self.study.optimize(objective, n_trials=n_trials)

            best_params = self.study.best_params
            best_score = self.study.best_value

            # Save best parameters
            self._save_best_params("lstm_optuna", best_params, best_score)

            # Save top trials
            self._save_top_trials("lstm_optuna")

            # Generate and save parameter importance plots
            self._generate_parameter_importance_plots("lstm_optuna")

            # Log parameter importance
            self._log_parameter_importance("lstm_optuna")

            logger.info(
                f"LSTM Optuna optimization completed. Best RMSE: {best_score:.4f}"
            )
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "optuna",
                "optimization_history": self.study.trials_dataframe(),
                "parameter_importance": self._get_parameter_importance(),
                "top_trials": self._get_top_trials(),
            }

        except Exception as e:
            logger.error(f"LSTM Optuna optimization failed: {e}")
            return {"error": str(e)}

    def _optimize_lstm_skopt(
        self, X: np.ndarray, y: np.ndarray, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize LSTM using scikit-optimize."""
        space = [
            Integer(32, 256, name="lstm_units"),
            Real(0.1, 0.5, name="dropout_rate"),
            Real(0.001, 0.01, prior="log-uniform", name="learning_rate"),
            Categorical([16, 32, 64, 128], name="batch_size"),
            Integer(50, 200, name="epochs"),
            Integer(10, 50, name="sequence_length"),
        ]

        def objective(params):
            param_dict = {
                "lstm_units": int(params[0]),
                "dropout_rate": params[1],
                "learning_rate": params[2],
                "batch_size": params[3],
                "epochs": int(params[4]),
                "sequence_length": int(params[5]),
            }
            return self._evaluate_lstm_params(X, y, param_dict)

        try:
            logger.info(
                f"Starting LSTM optimization with scikit-optimize ({n_trials} trials)"
            )
            result = gp_minimize(objective, space, n_calls=n_trials, random_state=42)

            best_params = {
                "lstm_units": int(result.x[0]),
                "dropout_rate": result.x[1],
                "learning_rate": result.x[2],
                "batch_size": result.x[3],
                "epochs": int(result.x[4]),
                "sequence_length": int(result.x[5]),
            }

            self._save_best_params("lstm_skopt", best_params, result.fun)

            # Save optimization results
            self._save_skopt_results("lstm_skopt", result, best_params)

            logger.info(f"LSTM scikit-optimize completed. Best RMSE: {result.fun:.4f}")
            return {
                "best_params": best_params,
                "best_score": result.fun,
                "n_trials": n_trials,
                "backend": "skopt",
                "optimization_history": result,
            }

        except Exception as e:
            logger.error(f"LSTM scikit-optimize failed: {e}")
            return {"error": str(e)}

    def _optimize_lstm_random(
        self, X: np.ndarray, y: np.ndarray, n_trials: int
    ) -> Dict[str, Any]:
        """Optimize LSTM using Random Search."""
        param_distributions = {
            "lstm_units": [32, 64, 128, 256],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "learning_rate": [0.001, 0.005, 0.01],
            "batch_size": [16, 32, 64, 128],
            "epochs": [50, 100, 150, 200],
            "sequence_length": [10, 20, 30, 50],
        }

        try:
            logger.info(f"Starting LSTM Random Search ({n_trials} trials)")

            # For LSTM, we'll use a simpler evaluation approach
            results = []
            for i in range(n_trials):
                params = {}
                for param, values in param_distributions.items():
                    params[param] = np.random.choice(values)

                score = self._evaluate_lstm_params(X, y, params)
                results.append({"params": params, "score": score, "trial": i})

            # Find best result
            best_result = min(results, key=lambda x: x["score"])
            best_params = best_result["params"]
            best_score = best_result["score"]

            self._save_best_params("lstm_random", best_params, best_score)

            # Save random search results
            self._save_random_search_results("lstm_random", results)

            logger.info(f"LSTM Random Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "n_trials": n_trials,
                "backend": "random",
                "optimization_history": results,
            }

        except Exception as e:
            logger.error(f"LSTM Random Search failed: {e}")
            return {"error": str(e)}

    def _optimize_lstm_grid(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize LSTM using Grid Search."""
        param_grid = {
            "lstm_units": [64, 128],
            "dropout_rate": [0.2, 0.3],
            "learning_rate": [0.001, 0.01],
            "batch_size": [32, 64],
            "epochs": [100, 150],
            "sequence_length": [20, 30],
        }

        try:
            logger.info("Starting LSTM Grid Search")

            # For LSTM, we'll use a simpler evaluation approach
            results = []
            from itertools import product

            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            for combination in product(*param_values):
                params = dict(zip(param_names, combination))
                score = self._evaluate_lstm_params(X, y, params)
                results.append({"params": params, "score": score})

            # Find best result
            best_result = min(results, key=lambda x: x["score"])
            best_params = best_result["params"]
            best_score = best_result["score"]

            self._save_best_params("lstm_grid", best_params, best_score)

            # Save grid search results
            self._save_grid_search_results("lstm_grid", results)

            logger.info(f"LSTM Grid Search completed. Best RMSE: {best_score:.4f}")
            return {
                "best_params": best_params,
                "best_score": best_score,
                "backend": "grid",
                "optimization_history": results,
            }

        except Exception as e:
            logger.error(f"LSTM Grid Search failed: {e}")
            return {"error": str(e)}

    def _evaluate_lstm_params(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> float:
        """Evaluate LSTM parameters using time series cross-validation."""
        try:
            sequence_length = params.get("sequence_length", 20)
            X_seq, y_seq = self._prepare_sequences(X, y, sequence_length)

            if len(X_seq) < 10:
                return float("inf")

            # Simple evaluation with train/test split
            split_idx = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

            # Create and train model
            model = LSTMModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=params.get("lstm_units", 128),
                dropout_rate=params.get("dropout_rate", 0.2),
                learning_rate=params.get("learning_rate", 0.001),
            )

            model.fit(
                X_train,
                y_train,
                epochs=params.get("epochs", 100),
                batch_size=params.get("batch_size", 32),
                verbose=0,
            )

            # Predict and evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            return rmse

        except Exception as e:
            logger.error(f"LSTM evaluation error: {e}")
            return float("inf")

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> tuple:
        """Prepare sequences for LSTM training."""
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def _save_best_params(self, model_type: str, params: Dict[str, Any], score: float):
        """Save best parameters to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_best_params_{timestamp}.json"
            filepath = self.best_params_dir / filename

            data = {
                "model_type": model_type,
                "best_params": params,
                "best_score": score,
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Best parameters saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")

    def _save_top_trials(self, model_type: str):
        """Save top trials to JSON file."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_top_trials_{timestamp}.json"
            filepath = self.trials_dir / filename

            # Get top trials
            top_trials = []
            for i, trial in enumerate(self.study.trials[: self.save_top_trials]):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    top_trials.append(
                        {
                            "rank": i + 1,
                            "trial_number": trial.number,
                            "params": trial.params,
                            "value": trial.value,
                            "datetime_start": trial.datetime_start.isoformat()
                            if trial.datetime_start
                            else None,
                            "datetime_complete": trial.datetime_complete.isoformat()
                            if trial.datetime_complete
                            else None,
                        }
                    )

            data = {
                "model_type": model_type,
                "top_trials": top_trials,
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend,
                "study_name": self.study_name,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Top {len(top_trials)} trials saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save top trials: {e}")

    def _generate_parameter_importance_plots(self, model_type: str):
        """Generate and save parameter importance plots."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plot optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(
                str(
                    self.plots_dir
                    / f"{model_type}_optimization_history_{timestamp}.html"
                )
            )

            # Plot parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(
                str(self.plots_dir / f"{model_type}_param_importances_{timestamp}.html")
            )

            # Plot parallel coordinate
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(
                str(
                    self.plots_dir
                    / f"{model_type}_parallel_coordinate_{timestamp}.html"
                )
            )

            logger.info(f"Parameter importance plots saved for {model_type}")

        except Exception as e:
            logger.error(f"Failed to generate parameter importance plots: {e}")

    def _log_parameter_importance(self, model_type: str):
        """Log parameter importance information."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return

        try:
            # Get parameter importance
            importance = optuna.importance.get_param_importances(self.study)

            logger.info(f"Parameter importance for {model_type}:")
            for param, imp in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {param}: {imp:.4f}")

        except Exception as e:
            logger.error(f"Failed to log parameter importance: {e}")

    def _get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance dictionary."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return {}

        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.error(f"Failed to get parameter importance: {e}")
            return {}

    def _get_top_trials(self) -> List[Dict[str, Any]]:
        """Get top trials information."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return []

        try:
            top_trials = []
            for i, trial in enumerate(self.study.trials[: self.save_top_trials]):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    top_trials.append(
                        {
                            "rank": i + 1,
                            "trial_number": trial.number,
                            "params": trial.params,
                            "value": trial.value,
                        }
                    )
            return top_trials
        except Exception as e:
            logger.error(f"Failed to get top trials: {e}")
            return []

    def _save_skopt_results(self, model_type: str, result, best_params: Dict[str, Any]):
        """Save scikit-optimize results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_skopt_results_{timestamp}.json"
            filepath = self.trials_dir / filename

            data = {
                "model_type": model_type,
                "best_params": best_params,
                "best_score": float(result.fun),
                "n_iterations": len(result.x_iters),
                "timestamp": datetime.now().isoformat(),
                "backend": "skopt",
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"scikit-optimize results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save scikit-optimize results: {e}")

    def _save_random_search_results(self, model_type: str, results):
        """Save random search results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_random_search_results_{timestamp}.json"
            filepath = self.trials_dir / filename

            data = {
                "model_type": model_type,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "backend": "random",
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Random search results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save random search results: {e}")

    def _save_grid_search_results(self, model_type: str, results):
        """Save grid search results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_grid_search_results_{timestamp}.json"
            filepath = self.trials_dir / filename

            data = {
                "model_type": model_type,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "backend": "grid",
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Grid search results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save grid search results: {e}")

    def get_best_params(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get best parameters for a model type."""
        try:
            # Look for the most recent best params file
            pattern = f"{model_type}_best_params_*.json"
            files = list(self.best_params_dir.glob(pattern))

            if not files:
                return None

            # Get the most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, "r") as f:
                data = json.load(f)

            return data.get("best_params")

        except Exception as e:
            logger.error(f"Failed to get best parameters: {e}")
            return None

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("Optuna not available or no study found")
            return

        try:
            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")


def get_optimizer(backend: str = "optuna") -> HyperparameterOptimizer:
    """Get optimizer instance."""
    return HyperparameterOptimizer(backend=backend)
