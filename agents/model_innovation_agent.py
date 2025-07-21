"""
Model Innovation Agent

This agent automatically discovers and evaluates new forecasting models using AutoML.
It integrates with FLAML for efficient hyperparameter optimization and automatically
updates the model registry when better performing models are found.

Features:
- Automated model architecture search
- Performance comparison against existing ensemble
- Automatic model registry updates
- Hybrid weight optimization
- Comprehensive evaluation metrics
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# AutoML imports
try:
    from flaml import AutoML

    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    logging.warning("FLAML not available, falling back to basic model search")

try:
    from optuna import create_study
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, using FLAML only")

# ML imports
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

# Trading imports
from utils.cache_utils import cache_model_operation
from utils.weight_registry import get_weight_registry, optimize_ensemble_weights

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Represents a candidate model for evaluation."""

    name: str
    model_type: str  # 'linear', 'tree', 'neural', 'ensemble'
    model: Any
    hyperparameters: Dict[str, Any]
    training_time: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluation:
    """Results of model evaluation."""

    model_name: str
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float
    training_time: float
    inference_time: float
    model_size_mb: float
    evaluation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InnovationConfig:
    """Configuration for model innovation agent."""

    # AutoML settings
    automl_time_budget: int = 300  # seconds
    automl_metric: str = "mse"
    automl_task: str = "regression"

    # Model search settings
    max_models_per_search: int = 10
    min_improvement_threshold: float = 0.05  # 5% improvement required
    evaluation_window_days: int = 30

    # Model types to search
    enable_linear_models: bool = True
    enable_tree_models: bool = True
    enable_neural_models: bool = True
    enable_ensemble_models: bool = True

    # Evaluation settings
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Storage settings
    models_dir: str = "models/innovated"
    cache_dir: str = "cache/model_innovation"

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.15
    min_r2_score: float = 0.3


class ModelInnovationAgent:
    """
    Agent for automatically discovering and evaluating new forecasting models.

    This agent uses AutoML techniques to search for better model architectures
    and automatically integrates them into the existing ensemble when they
    outperform current models.
    """

    def __init__(self, config: Optional[InnovationConfig] = None):
        """
        Initialize the model innovation agent.

        Args:
            config: Configuration for the agent
        """
        self.config = config or InnovationConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize storage directories
        self._init_directories()

        # Initialize model registry
        self.weight_registry = get_weight_registry()

        # Track discovered models
        self.discovered_models: List[ModelCandidate] = []
        self.evaluations: List[ModelEvaluation] = []

        # Performance tracking
        self.innovation_history: List[Dict[str, Any]] = []

        # Check dependencies
        self._check_dependencies()

        self.logger.info("ModelInnovationAgent initialized successfully")

    def _init_directories(self):
        """Initialize storage directories."""
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _check_dependencies(self):
        """Check and log dependency availability."""
        dependencies = {
            "FLAML": FLAML_AVAILABLE,
            "Optuna": OPTUNA_AVAILABLE,
            "Scikit-learn": SKLEARN_AVAILABLE,
            "PyTorch": TORCH_AVAILABLE,
        }

        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            self.logger.info(
                f"{status} {dep}: {'Available' if available else 'Not available'}"
            )

        if not any([FLAML_AVAILABLE, OPTUNA_AVAILABLE]):
            raise RuntimeError("No AutoML library available (FLAML or Optuna required)")

    @cache_model_operation(model_type="innovation")
    def discover_models(
        self, data: pd.DataFrame, target_col: str = "target"
    ) -> List[ModelCandidate]:
        """
        Discover new model candidates using AutoML.

        Args:
            data: Training data
            target_col: Target column name

        Returns:
            List of discovered model candidates
        """
        self.logger.info(f"Starting model discovery with {len(data)} samples")

        candidates = []

        # Prepare data
        X, y = self._prepare_data(data, target_col)

        # Use FLAML for AutoML
        if FLAML_AVAILABLE:
            flaml_candidates = self._discover_with_flaml(X, y)
            candidates.extend(flaml_candidates)

        # Use Optuna for additional search
        if OPTUNA_AVAILABLE:
            optuna_candidates = self._discover_with_optuna(X, y)
            candidates.extend(optuna_candidates)

        # Manual model search as fallback
        if not candidates:
            manual_candidates = self._discover_manual_models(X, y)
            candidates.extend(manual_candidates)

        # Limit number of candidates
        candidates = candidates[: self.config.max_models_per_search]

        self.logger.info(f"Discovered {len(candidates)} model candidates")
        return candidates

    def _discover_with_flaml(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[ModelCandidate]:
        """Discover models using FLAML AutoML."""
        candidates = []

        try:
            # Configure FLAML
            automl = AutoML()

            # Define search space
            search_space = {}

            if self.config.enable_linear_models:
                search_space.update(
                    {
                        "linear": {
                            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                            "fit_intercept": [True, False],
                        }
                    }
                )

            if self.config.enable_tree_models:
                search_space.update(
                    {
                        "rf": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [3, 5, 7, 10, None],
                            "min_samples_split": [2, 5, 10],
                        },
                        "xgboost": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [3, 5, 7],
                            "learning_rate": [0.01, 0.1, 0.2],
                        },
                    }
                )

            if self.config.enable_neural_models and TORCH_AVAILABLE:
                search_space.update(
                    {
                        "neural": {
                            "hidden_size": [32, 64, 128],
                            "num_layers": [1, 2, 3],
                            "dropout": [0.1, 0.2, 0.3],
                        }
                    }
                )

            # Run AutoML
            start_time = datetime.now()

            automl.fit(
                X,
                y,
                task=self.config.automl_task,
                metric=self.config.automl_metric,
                time_budget=self.config.automl_time_budget,
                search_space=search_space,
                n_splits=self.config.cv_folds,
                random_state=self.config.random_state,
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Create candidate from best model
            best_model = automl.model
            best_config = automl.best_config

            candidate = ModelCandidate(
                name=(
                    f"flaml_{best_config.get('estimator', 'unknown')}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ),
                model_type=self._classify_model_type(best_config.get("estimator", "")),
                model=best_model,
                hyperparameters=best_config,
                training_time=training_time,
                metadata={
                    "automl_library": "flaml",
                    "best_score": automl.best_loss,
                    "search_time": training_time,
                },
            )

            candidates.append(candidate)
            self.logger.info(
                f"FLAML discovered: {candidate.name} (score: {automl.best_loss:.4f})"
            )

        except Exception as e:
            self.logger.error(f"FLAML discovery failed: {e}")

        return candidates

    def _discover_with_optuna(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[ModelCandidate]:
        """Discover models using Optuna hyperparameter optimization."""
        candidates = []

        try:
            # Create study
            study = create_study(
                direction="minimize", sampler=TPESampler(seed=self.config.random_state)
            )

            # Define objective function
            def objective(trial):
                # Sample model type
                model_type = trial.suggest_categorical(
                    "model_type", ["linear", "tree", "neural"]
                )

                if model_type == "linear":
                    model = Ridge(
                        alpha=trial.suggest_float("alpha", 0.001, 10.0, log=True),
                        fit_intercept=trial.suggest_categorical(
                            "fit_intercept", [True, False]
                        ),
                    )
                elif model_type == "tree":
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int("n_estimators", 50, 200),
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                        random_state=self.config.random_state,
                    )
                elif model_type == "neural" and TORCH_AVAILABLE:
                    # Create neural network
                    input_size = X.shape[1]
                    hidden_size = trial.suggest_int("hidden_size", 32, 128)
                    num_layers = trial.suggest_int("num_layers", 1, 3)
                    dropout = trial.suggest_float("dropout", 0.1, 0.3)

                    model = self._create_neural_network(
                        input_size, hidden_size, num_layers, dropout
                    )
                elif model_type == "neural" and not TORCH_AVAILABLE:
                    # Fallback to linear model if PyTorch is not available
                    model = Ridge(alpha=1.0)
                else:
                    # This should not happen given the categorical choices, but provide fallback
                    model = Ridge(alpha=1.0)

                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
                scores = []

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if model_type == "neural":
                        score = self._train_neural_network(
                            model, X_train, y_train, X_val, y_val
                        )
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred)

                    scores.append(score)

                return np.mean(scores)

            # Optimize
            start_time = datetime.now()
            study.optimize(
                objective, n_trials=50, timeout=self.config.automl_time_budget
            )
            training_time = (datetime.now() - start_time).total_seconds()

            # Create candidate from best trial
            best_trial = study.best_trial
            best_params = best_trial.params

            # Recreate best model
            model_type = best_params["model_type"]
            if model_type == "linear":
                best_model = Ridge(
                    **{k: v for k, v in best_params.items() if k != "model_type"}
                )
            elif model_type == "tree":
                best_model = RandomForestRegressor(
                    **{k: v for k, v in best_params.items() if k != "model_type"}
                )
            elif model_type == "neural":
                best_model = self._create_neural_network(
                    X.shape[1],
                    best_params["hidden_size"],
                    best_params["num_layers"],
                    best_params["dropout"],
                )

            candidate = ModelCandidate(
                name=f"optuna_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type=model_type,
                model=best_model,
                hyperparameters=best_params,
                training_time=training_time,
                metadata={
                    "automl_library": "optuna",
                    "best_score": best_trial.value,
                    "n_trials": len(study.trials),
                },
            )

            candidates.append(candidate)
            self.logger.info(
                f"Optuna discovered: {candidate.name} (score: {best_trial.value:.4f})"
            )

        except Exception as e:
            self.logger.error(f"Optuna discovery failed: {e}")

        return candidates

    def _discover_manual_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[ModelCandidate]:
        """Discover models using manual search as fallback."""
        candidates = []

        if not SKLEARN_AVAILABLE:
            return candidates

        try:
            # Linear models
            if self.config.enable_linear_models:
                linear_models = [
                    ("ridge", Ridge(alpha=1.0)),
                    ("lasso", Lasso(alpha=0.1)),
                    ("linear", LinearRegression()),
                ]

                for name, model in linear_models:
                    start_time = datetime.now()
                    model.fit(X, y)
                    training_time = (datetime.now() - start_time).total_seconds()

                    candidate = ModelCandidate(
                        name=(
                            f"manual_{name}_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        ),
                        model_type="linear",
                        model=model,
                        hyperparameters={"model": name},
                        training_time=training_time,
                        metadata={"search_method": "manual"},
                    )
                    candidates.append(candidate)

            # Tree models
            if self.config.enable_tree_models:
                tree_models = [
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=100, random_state=self.config.random_state
                        ),
                    ),
                    (
                        "gbm",
                        GradientBoostingRegressor(
                            n_estimators=100, random_state=self.config.random_state
                        ),
                    ),
                ]

                for name, model in tree_models:
                    start_time = datetime.now()
                    model.fit(X, y)
                    training_time = (datetime.now() - start_time).total_seconds()

                    candidate = ModelCandidate(
                        name=(
                            f"manual_{name}_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        ),
                        model_type="tree",
                        model=model,
                        hyperparameters={"model": name},
                        training_time=training_time,
                        metadata={"search_method": "manual"},
                    )
                    candidates.append(candidate)

            self.logger.info(f"Manual discovery found {len(candidates)} models")

        except Exception as e:
            self.logger.error(f"Manual discovery failed: {e}")

        return candidates

    def _create_neural_network(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Create a neural network model."""

        class ForecastingNN(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                layers = []

                # Input layer
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

                # Hidden layers
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))

                # Output layer
                layers.append(nn.Linear(hidden_size, 1))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        return ForecastingNN(input_size, hidden_size, num_layers, dropout)

    def _train_neural_network(
        self,
        model: nn.Module,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Train neural network and return validation score."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(50):  # Simple training
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        return val_loss.item()

    def _classify_model_type(self, estimator_name: str) -> str:
        """Classify model type based on estimator name."""
        linear_models = ["linear", "ridge", "lasso", "elasticnet"]
        tree_models = ["rf", "xgboost", "lgbm", "catboost"]
        neural_models = ["neural", "mlp", "lstm", "transformer"]

        if any(linear in estimator_name.lower() for linear in linear_models):
            return "linear"
        elif any(tree in estimator_name.lower() for tree in tree_models):
            return "tree"
        elif any(neural in estimator_name.lower() for neural in neural_models):
            return "neural"
        else:
            return "unknown"

    def _prepare_data(
        self, data: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training."""
        # Remove target column
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Handle missing values
        X = X.fillna(method="ffill").fillna(method="bfill")
        y = y.fillna(method="ffill").fillna(method="bfill")

        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        return X, y

    def evaluate_candidate(
        self, candidate: ModelCandidate, data: pd.DataFrame, target_col: str = "target"
    ) -> ModelEvaluation:
        """
        Evaluate a model candidate against existing ensemble.

        Args:
            candidate: Model candidate to evaluate
            data: Evaluation data
            target_col: Target column name

        Returns:
            Model evaluation results
        """
        self.logger.info(f"Evaluating candidate: {candidate.name}")

        # Prepare data
        X, y = self._prepare_data(data, target_col)

        # Split data
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        start_time = datetime.now()

        if candidate.model_type == "neural" and TORCH_AVAILABLE:
            # Train neural network
            self._train_neural_network(
                candidate.model, X_train, y_train, X_test, y_test
            )
            training_time = (datetime.now() - start_time).total_seconds()

            # Predict
            candidate.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test.values)
                y_pred = candidate.model(X_test_tensor).squeeze().numpy()
            
            # For neural networks, inference time is negligible (already computed)
            inference_time = 0.0
        else:
            # Train sklearn model
            candidate.model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Predict
            pred_start = datetime.now()
            y_pred = candidate.model.predict(X_test)
            inference_time = (datetime.now() - pred_start).total_seconds()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate trading metrics
        returns = pd.Series(y_test.values) - pd.Series(y_pred)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate other metrics
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = returns.std()

        # Calculate model size
        model_size_mb = self._calculate_model_size(candidate.model)

        evaluation = ModelEvaluation(
            model_name=candidate.name,
            mse=mse,
            mae=mae,
            r2_score=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            volatility=volatility,
            training_time=training_time,
            inference_time=inference_time,
            model_size_mb=model_size_mb,
        )

        self.logger.info(f"Evaluation complete: {candidate.name}")
        self.logger.info(f"  MSE: {mse:.4f}, R²: {r2:.4f}, Sharpe: {sharpe_ratio:.4f}")

        return evaluation

    def _calculate_model_size(self, model: Any) -> float:
        """Calculate model size in MB."""
        try:
            # Save model to temporary file
            temp_path = f"{self.config.cache_dir}/temp_model.pkl"
            with open(temp_path, "wb") as f:
                pickle.dump(model, f)

            # Get file size
            size_bytes = os.path.getsize(temp_path)
            size_mb = size_bytes / (1024 * 1024)

            # Clean up
            os.remove(temp_path)

            return size_mb
        except Exception as e:
            # Log the error for debugging but return 0.0 to avoid breaking the pipeline
            self.logger.debug(f"Error calculating model size: {e}")
            return 0.0

    def compare_with_ensemble(self, evaluation: ModelEvaluation) -> Dict[str, Any]:
        """
        Compare candidate model with existing ensemble.

        Args:
            evaluation: Model evaluation results

        Returns:
            Comparison results
        """
        # Get current ensemble performance
        current_models = self.weight_registry.registry["models"]

        if not current_models:
            return {
                "improvement": True,
                "improvement_percentage": float("inf"),
                "reason": "No existing models to compare against",
            }

        # Calculate ensemble metrics
        ensemble_metrics = {
            "mse": np.mean(
                [
                    m["performance"]["mse"]
                    for m in current_models.values()
                    if "mse" in m["performance"]
                ]
            ),
            "sharpe": np.mean(
                [
                    m["performance"]["sharpe_ratio"]
                    for m in current_models.values()
                    if "sharpe_ratio" in m["performance"]
                ]
            ),
            "r2": np.mean(
                [
                    m["performance"]["r2_score"]
                    for m in current_models.values()
                    if "r2_score" in m["performance"]
                ]
            ),
        }

        # Check for improvement
        improvements = []

        # MSE improvement (lower is better)
        if evaluation.mse < ensemble_metrics["mse"]:
            mse_improvement = (
                ensemble_metrics["mse"] - evaluation.mse
            ) / ensemble_metrics["mse"]
            improvements.append(f"MSE improved by {mse_improvement:.2%}")

        # Sharpe ratio improvement (higher is better)
        if evaluation.sharpe_ratio > ensemble_metrics["sharpe"]:
            sharpe_improvement = (
                evaluation.sharpe_ratio - ensemble_metrics["sharpe"]
            ) / abs(ensemble_metrics["sharpe"])
            improvements.append(f"Sharpe ratio improved by {sharpe_improvement:.2%}")

        # R² improvement (higher is better)
        if evaluation.r2_score > ensemble_metrics["r2"]:
            r2_improvement = (evaluation.r2_score - ensemble_metrics["r2"]) / abs(
                ensemble_metrics["r2"]
            )
            improvements.append(f"R² improved by {r2_improvement:.2%}")

        # Overall improvement
        has_improvement = len(improvements) > 0 and any(
            [
                evaluation.mse
                < ensemble_metrics["mse"] * (1 - self.config.min_improvement_threshold),
                evaluation.sharpe_ratio
                > ensemble_metrics["sharpe"]
                * (1 + self.config.min_improvement_threshold),
                evaluation.r2_score
                > ensemble_metrics["r2"] * (1 + self.config.min_improvement_threshold),
            ]
        )

        return {
            "improvement": has_improvement,
            "improvements": improvements,
            "current_ensemble": ensemble_metrics,
            "candidate_metrics": {
                "mse": evaluation.mse,
                "sharpe_ratio": evaluation.sharpe_ratio,
                "r2_score": evaluation.r2_score,
            },
        }

    def integrate_model(
        self, candidate: ModelCandidate, evaluation: ModelEvaluation
    ) -> bool:
        """
        Integrate a successful model into the ensemble.

        Args:
            candidate: Model candidate
            evaluation: Model evaluation results

        Returns:
            True if integration was successful
        """
        try:
            self.logger.info(f"Integrating model: {candidate.name}")

            # Save model
            model_path = f"{self.config.models_dir}/{candidate.name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(candidate.model, f)

            # Register model in weight registry
            initial_weights = {"base_weight": 0.1}  # Start with low weight

            success = self.weight_registry.register_model(
                model_name=candidate.name,
                model_type=candidate.model_type,
                initial_weights=initial_weights,
                metadata={
                    "discovery_method": candidate.metadata.get(
                        "automl_library", "manual"
                    ),
                    "hyperparameters": candidate.hyperparameters,
                    "training_time": candidate.training_time,
                    "model_path": model_path,
                },
            )

            if not success:
                self.logger.error(f"Failed to register model: {candidate.name}")
                return False

            # Update performance metrics
            performance_metrics = {
                "mse": evaluation.mse,
                "mae": evaluation.mae,
                "r2_score": evaluation.r2_score,
                "sharpe_ratio": evaluation.sharpe_ratio,
                "max_drawdown": evaluation.max_drawdown,
                "total_return": evaluation.total_return,
                "volatility": evaluation.volatility,
            }

            self.weight_registry.update_performance(candidate.name, performance_metrics)

            # Optimize ensemble weights
            current_models = list(self.weight_registry.registry["models"].keys())
            optimized_weights = optimize_ensemble_weights(
                model_names=current_models, method="performance_weighted"
            )

            # Update weights for all models
            for model_name, weight in optimized_weights.items():
                self.weight_registry.update_weights(
                    model_name=model_name,
                    new_weights={"base_weight": weight},
                    reason="ensemble_optimization",
                )

            # Record innovation
            innovation_record = {
                "timestamp": datetime.now().isoformat(),
                "model_name": candidate.name,
                "model_type": candidate.model_type,
                "improvement_metrics": {
                    "mse": evaluation.mse,
                    "sharpe_ratio": evaluation.sharpe_ratio,
                    "r2_score": evaluation.r2_score,
                },
                "integration_success": True,
            }

            self.innovation_history.append(innovation_record)

            self.logger.info(f"Successfully integrated model: {candidate.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to integrate model {candidate.name}: {e}")
            return False

    def run_innovation_cycle(
        self, data: pd.DataFrame, target_col: str = "target"
    ) -> Dict[str, Any]:
        """
        Run a complete model innovation cycle.

        Args:
            data: Training data
            target_col: Target column name

        Returns:
            Innovation cycle results
        """
        self.logger.info("Starting model innovation cycle")

        start_time = datetime.now()
        results = {
            "cycle_start": start_time.isoformat(),
            "candidates_discovered": 0,
            "candidates_evaluated": 0,
            "models_integrated": 0,
            "improvements_found": 0,
            "errors": [],
        }

        try:
            # Step 1: Discover new models
            candidates = self.discover_models(data, target_col)
            results["candidates_discovered"] = len(candidates)

            if not candidates:
                self.logger.warning("No candidates discovered")
                return results

            # Step 2: Evaluate candidates
            for candidate in candidates:
                try:
                    evaluation = self.evaluate_candidate(candidate, data, target_col)
                    results["candidates_evaluated"] += 1

                    # Step 3: Compare with ensemble
                    comparison = self.compare_with_ensemble(evaluation)

                    if comparison["improvement"]:
                        results["improvements_found"] += 1

                        # Step 4: Integrate if better
                        if self.integrate_model(candidate, evaluation):
                            results["models_integrated"] += 1
                            self.logger.info(
                                f"Integrated improved model: {candidate.name}"
                            )
                        else:
                            results["errors"].append(
                                f"Failed to integrate {candidate.name}"
                            )
                    else:
                        self.logger.info(
                            f"Model {candidate.name} did not improve ensemble"
                        )

                    # Store evaluation
                    self.evaluations.append(evaluation)

                except Exception as e:
                    error_msg = f"Error evaluating {candidate.name}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

            # Step 5: Final ensemble optimization
            current_models = list(self.weight_registry.registry["models"].keys())
            if len(current_models) > 1:
                optimized_weights = optimize_ensemble_weights(
                    model_names=current_models, method="performance_weighted"
                )

                for model_name, weight in optimized_weights.items():
                    self.weight_registry.update_weights(
                        model_name=model_name,
                        new_weights={"base_weight": weight},
                        reason="post_innovation_optimization",
                    )

            cycle_time = (datetime.now() - start_time).total_seconds()
            results["cycle_time_seconds"] = cycle_time
            results["cycle_end"] = datetime.now().isoformat()

            self.logger.info(f"Innovation cycle completed in {cycle_time:.2f} seconds")
            self.logger.info(
                f"Results: {results['models_integrated']} models integrated, "
                f"{results['improvements_found']} improvements found"
            )

        except Exception as e:
            error_msg = f"Innovation cycle failed: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        return results

    def get_innovation_statistics(self) -> Dict[str, Any]:
        """Get statistics about model innovation activities."""
        return {
            "total_cycles": len(self.innovation_history),
            "total_models_integrated": sum(
                1
                for record in self.innovation_history
                if record.get("integration_success", False)
            ),
            "total_evaluations": len(self.evaluations),
            "recent_innovations": (
                self.innovation_history[-10:] if self.innovation_history else []
            ),
            "model_type_distribution": self._get_model_type_distribution(),
            "performance_improvements": self._get_performance_improvements(),
        }

    def _get_model_type_distribution(self) -> Dict[str, int]:
        """Get distribution of model types in ensemble."""
        distribution = {}
        for model_name, model_info in self.weight_registry.registry["models"].items():
            model_type = model_info.get("type", "unknown")
            distribution[model_type] = distribution.get(model_type, 0) + 1
        return distribution

    def _get_performance_improvements(self) -> List[Dict[str, Any]]:
        """Get list of performance improvements over time."""
        improvements = []
        for record in self.innovation_history:
            if record.get("integration_success", False):
                improvements.append(
                    {
                        "timestamp": record["timestamp"],
                        "model_name": record["model_name"],
                        "model_type": record["model_type"],
                        "metrics": record["improvement_metrics"],
                    }
                )
        return improvements


# Convenience function to create innovation agent
def create_model_innovation_agent(
    config: Optional[InnovationConfig] = None,
) -> ModelInnovationAgent:
    """
    Create a configured model innovation agent.

    Args:
        config: Configuration for the agent

    Returns:
        Configured ModelInnovationAgent instance
    """
    return ModelInnovationAgent(config)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_model_innovation_agent()

    # Example data (replace with actual data)
    sample_data = pd.DataFrame(
        {
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
            "target": np.random.randn(1000),
        }
    )

    # Run innovation cycle
    results = agent.run_innovation_cycle(sample_data, target_col="target")
    print(f"Innovation results: {results}")

    # Get statistics
    stats = agent.get_innovation_statistics()
    print(f"Innovation statistics: {stats}")
