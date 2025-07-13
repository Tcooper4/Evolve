#!/usr/bin/env python3
"""
ML pipeline management script.
Provides commands for managing machine learning pipelines and model versioning.

This script supports:
- Model training and optimization
- Hyperparameter tuning
- Model evaluation and deployment
- Model monitoring and versioning
- Integration with MLflow, Ray, and other ML platforms

Usage:
    python manage_ml.py <command> [options]

Commands:
    train       Train a new model
    optimize    Optimize model hyperparameters
    evaluate    Evaluate model performance
    deploy      Deploy model to production
    monitor     Monitor model performance
    version     Manage model versions

Examples:
    # Train a new model
    python manage_ml.py train --model-type pytorch --data-path data/train.csv --params '{"learning_rate": 0.001}'

    # Optimize model hyperparameters
    python manage_ml.py optimize --model-type xgboost --data-path data/train.csv --param-grid '{"max_depth": [3,5,7]}'

    # Evaluate model performance
    python manage_ml.py evaluate --model-path models/model.pkl --data-path data/test.csv

    # Deploy model to production
    python manage_ml.py deploy --model-path models/model.pkl --deployment-type mlflow

    # Monitor model performance
    python manage_ml.py monitor --model-path models/model.pkl --data-path data/live.csv --duration 3600

    # Manage model versions
    python manage_ml.py version --action list
"""

import argparse
import asyncio
import json
import logging
import logging.config
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bentoml
import joblib
import kserve
import mlflow
import mlflow.catboost
import mlflow.fastai
import mlflow.gluon
import mlflow.h2o
import mlflow.lightgbm
import mlflow.mleap
import mlflow.onnx
import mlflow.paddle
import mlflow.pmdarima
import mlflow.prophet
import mlflow.pyfunc
import mlflow.pyspark.ml
import mlflow.pytorch
import mlflow.sklearn
import mlflow.spacy
import mlflow.spark
import mlflow.statsmodels
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import ray
import ray.serve
import seldon_core
import torch
import torch.nn as nn
import torch.optim as optim
import torchserve
import triton
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split


class MLManager:
    """Manager for machine learning pipeline operations.

    This class provides methods for training, optimizing, evaluating, deploying,
    and monitoring machine learning models. It supports various model types and
    integrates with MLflow for experiment tracking and model versioning.

    Attributes:
        config (dict): Application configuration
        logger (logging.Logger): Logger instance
        models_dir (Path): Directory for storing models
        experiments_dir (Path): Directory for storing experiments
        mlflow_dir (Path): Directory for MLflow tracking

    Example:
        manager = MLManager()
        model = await manager.train_model("pytorch", "data/train.csv")
        metrics = await manager.evaluate_model(model, "data/test.csv")
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the ML manager.

        Args:
            config_path: Path to the application configuration file

        Raises:
            SystemExit: If the configuration file is not found
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_dir = Path("mlruns")
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(self.mlflow_dir))

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary

        Raises:
            SystemExit: If the configuration file is not found
        """
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration.

        Raises:
            SystemExit: If the logging configuration file is not found
        """
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)

        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)

        logging.config.dictConfig(log_config)

    async def train_model(self, model_type: str, data_path: str, params: Optional[Dict[str, Any]] = None):
        """Train a machine learning model.

        Args:
            model_type: Type of model to train (e.g., "pytorch", "xgboost")
            data_path: Path to the training data
            params: Optional model parameters

        Returns:
            Trained model

        Raises:
            Exception: If model training fails
        """
        self.logger.info(f"Training {model_type} model with data from {data_path}")

        try:
            # Load data
            data = self._load_data(data_path)

            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(data)

            # Initialize model
            model = self._initialize_model(model_type, params)

            # Train model
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params or {})

                # Train model
                if model_type in ["pytorch"]:
                    model = await self._train_deep_learning_model(model, X_train, y_train)
                else:
                    model = await self._train_traditional_model(model, X_train, y_train)

                # Evaluate model
                metrics = await self._evaluate_model(model, X_test, y_test)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Save model
                model_path = self._save_model(model, model_type)

                # Log model
                mlflow.log_artifacts(model_path)

            self.logger.info(f"Model trained and saved to {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            raise

    async def optimize_model(self, model_type: str, data_path: str, param_grid: Dict[str, List[Any]]):
        """Optimize model hyperparameters.

        Args:
            model_type: Type of model to optimize
            data_path: Path to the training data
            param_grid: Dictionary of parameter grids for optimization

        Returns:
            Optimized model

        Raises:
            Exception: If model optimization fails
        """
        self.logger.info(f"Optimizing {model_type} model with data from {data_path}")

        try:
            # Load data
            data = self._load_data(data_path)

            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(data)

            # Initialize model
            model = self._initialize_model(model_type)

            # Optimize model
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(param_grid)

                # Optimize model
                if model_type in ["pytorch"]:
                    model = await self._optimize_deep_learning_model(model, X_train, y_train, param_grid)
                else:
                    model = await self._optimize_traditional_model(model, X_train, y_train, param_grid)

                # Evaluate model
                metrics = await self._evaluate_model(model, X_test, y_test)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Save model
                model_path = self._save_model(model, model_type)

                # Log model
                mlflow.log_artifacts(model_path)

            self.logger.info(f"Model optimized and saved to {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to optimize model: {e}")
            raise

    async def evaluate_model(self, model_path: str, data_path: str):
        """Evaluate a trained model."""
        self.logger.info(f"Evaluating model from {model_path} with data from {data_path}")

        try:
            # Load model
            model = self._load_model(model_path)

            # Load data
            data = self._load_data(data_path)

            # Prepare data
            X_test, y_test = self._prepare_evaluation_data(data)

            # Evaluate model
            metrics = await self._evaluate_model(model, X_test, y_test)

            # Print metrics
            self._print_metrics(metrics)

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {e}")
            raise

    async def deploy_model(self, model_path: str, deployment_type: str = "local"):
        """Deploy a trained model."""
        self.logger.info(f"Deploying model from {model_path} to {deployment_type}")

        try:
            # Load model
            model = self._load_model(model_path)

            # Deploy model
            if deployment_type == "local":
                deployment = await self._deploy_local(model)
            elif deployment_type == "mlflow":
                deployment = await self._deploy_mlflow(model)
            elif deployment_type == "ray":
                deployment = await self._deploy_ray(model)
            elif deployment_type == "bentoml":
                deployment = await self._deploy_bentoml(model)
            elif deployment_type == "seldon":
                deployment = await self._deploy_seldon(model)
            elif deployment_type == "kserve":
                deployment = await self._deploy_kserve(model)
            elif deployment_type == "triton":
                deployment = await self._deploy_triton(model)
            elif deployment_type == "torchserve":
                deployment = await self._deploy_torchserve(model)
            else:
                raise ValueError(f"Unsupported deployment type: {deployment_type}")

            self.logger.info(f"Model deployed to {deployment_type}")
            return deployment
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            raise

    async def monitor_model(self, model_path: str, data_path: str, duration: int = 300):
        """Monitor model performance."""
        self.logger.info(f"Monitoring model from {model_path} with data from {data_path}")

        try:
            # Load model
            model = self._load_model(model_path)

            # Load data
            data = self._load_data(data_path)

            # Prepare data
            X_test, y_test = self._prepare_evaluation_data(data)

            # Monitor model
            metrics = []
            start_time = time.time()

            while time.time() - start_time < duration:
                # Evaluate model
                current_metrics = await self._evaluate_model(model, X_test, y_test)
                metrics.append(current_metrics)

                # Log metrics
                self._log_metrics(current_metrics)

                await asyncio.sleep(1)

            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.experiments_dir / f"metrics_{timestamp}.json"

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Metrics saved to {metrics_file}")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to monitor model: {e}")
            raise

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file."""
        try:
            if data_path.endswith(".csv"):
                return pd.read_csv(data_path)
            elif data_path.endswith(".json"):
                return pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        try:
            # Split features and target
            X = data.drop("target", axis=1)
            y = data["target"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise

    def _prepare_evaluation_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation."""
        try:
            # Split features and target
            X = data.drop("target", axis=1)
            y = data["target"]

            return X, y
        except Exception as e:
            self.logger.error(f"Failed to prepare evaluation data: {e}")
            raise

    def _initialize_model(self, model_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a model."""
        try:
            if model_type == "pytorch":
                return self._initialize_pytorch_model(params)
            elif model_type == "sklearn":
                return self._initialize_sklearn_model(params)
            elif model_type == "xgboost":
                return self._initialize_xgboost_model(params)
            elif model_type == "lightgbm":
                return self._initialize_lightgbm_model(params)
            elif model_type == "catboost":
                return self._initialize_catboost_model(params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise

    def _initialize_pytorch_model(self, params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Initialize a PyTorch model."""
        try:
            model = nn.Sequential(
                nn.Linear(params.get("input_size", 10), params.get("hidden_size", 64)),
                nn.ReLU(),
                nn.Dropout(params.get("dropout", 0.2)),
                nn.Linear(params.get("hidden_size", 64), params.get("output_size", 1)),
                nn.Sigmoid(),
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize PyTorch model: {e}")
            raise

    def _initialize_sklearn_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a scikit-learn model."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), random_state=42
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize scikit-learn model: {e}")
            raise

    def _initialize_xgboost_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize an XGBoost model."""
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize XGBoost model: {e}")
            raise

    def _initialize_lightgbm_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a LightGBM model."""
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", -1),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize LightGBM model: {e}")
            raise

    def _initialize_catboost_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a CatBoost model."""
        try:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=params.get("iterations", 100),
                depth=params.get("depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize CatBoost model: {e}")
            raise

    async def _train_deep_learning_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a deep learning model."""
        try:
            if isinstance(model, nn.Module):
                # PyTorch model
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters())

                for epoch in range(100):
                    optimizer.zero_grad()
                    outputs = model(torch.FloatTensor(X_train))
                    loss = criterion(outputs, torch.FloatTensor(y_train))
                    loss.backward()
                    optimizer.step()

            return model
        except Exception as e:
            self.logger.error(f"Failed to train deep learning model: {e}")
            raise

    async def _train_traditional_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a traditional model."""
        try:
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"Failed to train traditional model: {e}")
            raise

    async def _optimize_deep_learning_model(
        self, model: Any, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, List[Any]]
    ) -> Any:
        """Optimize a deep learning model."""
        try:
            if isinstance(model, nn.Module):
                # PyTorch model
                def objective(trial):
                    model = self._initialize_pytorch_model(
                        {
                            "input_size": trial.suggest_int("input_size", 10, 100),
                            "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                            "output_size": 1,
                        }
                    )
                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(model.parameters())

                    for epoch in range(100):
                        optimizer.zero_grad()
                        outputs = model(torch.FloatTensor(X_train))
                        loss = criterion(outputs, torch.FloatTensor(y_train))
                        loss.backward()
                        optimizer.step()

                    return loss.item()

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=100)

                return self._initialize_pytorch_model(study.best_params)
        except Exception as e:
            self.logger.error(f"Failed to optimize deep learning model: {e}")
            raise

    async def _optimize_traditional_model(
        self, model: Any, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, List[Any]]
    ) -> Any:
        """Optimize a traditional model."""
        try:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, y_train)

            return grid_search.best_estimator_
        except Exception as e:
            self.logger.error(f"Failed to optimize traditional model: {e}")
            raise

    async def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a model."""
        try:
            if isinstance(model, nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test))
                    y_pred = (y_pred > 0.5).float()
            else:
                # Other models
                y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {e}")
            raise

    def _save_model(self, model: Any, model_type: str) -> str:
        """Save a model."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"model_{model_type}_{timestamp}"

            if model_type == "pytorch":
                torch.save(model.state_dict(), model_path)
            else:
                joblib.dump(model, model_path)

            return str(model_path)
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def _load_model(self, model_path: str) -> Any:
        """Load a model."""
        try:
            if model_path.endswith(".pt"):
                # PyTorch model
                model = self._initialize_pytorch_model()
                model.load_state_dict(torch.load(model_path))
                return model
            else:
                # Other models
                return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _print_metrics(self, metrics: Dict[str, float]):
        """Print model metrics."""
        print("\nModel Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log model metrics."""
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

    async def _deploy_local(self, model: Any) -> Any:
        """Deploy model locally."""
        try:
            return model
        except Exception as e:
            self.logger.error(f"Failed to deploy model locally: {e}")
            raise

    async def _deploy_mlflow(self, model: Any) -> Any:
        """Deploy model using MLflow."""
        try:
            mlflow.pyfunc.log_model(python_model=model, artifact_path="model")
            return mlflow.pyfunc.load_model("model")
        except Exception as e:
            self.logger.error(f"Failed to deploy model using MLflow: {e}")
            raise

    async def _deploy_ray(self, model: Any) -> Any:
        """Deploy model using Ray Serve."""
        try:
            ray.init()
            ray.serve.start()
            ray.serve.create_backend("model", model)
            ray.serve.create_endpoint("model", backend="model")
            return ray.serve.get_handle("model")
        except Exception as e:
            self.logger.error(f"Failed to deploy model using Ray Serve: {e}")
            raise

    async def _deploy_bentoml(self, model: Any) -> Any:
        """Deploy model using BentoML."""
        try:
            bento_model = bentoml.pytorch.save_model("model", model, signatures={"predict": {"batchable": True}})
            return bento_model
        except Exception as e:
            self.logger.error(f"Failed to deploy model using BentoML: {e}")
            raise

    async def _deploy_seldon(self, model: Any) -> Any:
        """Deploy model using Seldon Core."""
        try:
            seldon_model = seldon_core.SeldonModel(model)
            return seldon_model
        except Exception as e:
            self.logger.error(f"Failed to deploy model using Seldon Core: {e}")
            raise

    async def _deploy_kserve(self, model: Any) -> Any:
        """Deploy model using KServe."""
        try:
            kserve_model = kserve.Model(model)
            return kserve_model
        except Exception as e:
            self.logger.error(f"Failed to deploy model using KServe: {e}")
            raise

    async def _deploy_triton(self, model: Any) -> Any:
        """Deploy model using Triton Inference Server."""
        try:
            triton_model = triton.Model(model)
            return triton_model
        except Exception as e:
            self.logger.error(f"Failed to deploy model using Triton: {e}")
            raise

    async def _deploy_torchserve(self, model: Any) -> Any:
        """Deploy model using TorchServe."""
        try:
            torchserve_model = torchserve.Model(model)
            return torchserve_model
        except Exception as e:
            self.logger.error(f"Failed to deploy model using TorchServe: {e}")
            raise


def main():
    """Main entry point for the ML management script."""
    parser = argparse.ArgumentParser(description="ML Pipeline Manager")
    parser.add_argument(
        "command", choices=["train", "optimize", "evaluate", "deploy", "monitor", "version"], help="Command to run"
    )
    parser.add_argument("--model-type", choices=["pytorch", "xgboost", "lightgbm", "catboost"], help="Type of model")
    parser.add_argument("--data-path", help="Path to data file")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--params", type=json.loads, help="Model parameters as JSON string")
    parser.add_argument("--param-grid", type=json.loads, help="Parameter grid for optimization as JSON string")
    parser.add_argument(
        "--deployment-type",
        choices=["local", "mlflow", "ray", "bentoml", "seldon", "kserve", "triton", "torchserve"],
        default="local",
        help="Type of deployment",
    )
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds")
    parser.add_argument("--help", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.help:
        print(__doc__)
        return

    manager = MLManager()
    if args.command == "train":
        asyncio.run(manager.train_model(args.model_type, args.data_path, args.params))
    elif args.command == "optimize":
        asyncio.run(manager.optimize_model(args.model_type, args.data_path, args.param_grid))
    elif args.command == "evaluate":
        asyncio.run(manager.evaluate_model(args.model_path, args.data_path))
    elif args.command == "deploy":
        asyncio.run(manager.deploy_model(args.model_path, args.deployment_type))
    elif args.command == "monitor":
        asyncio.run(manager.monitor_model(args.model_path, args.data_path, args.duration))
    elif args.command == "version":
        # Implement version management
        pass


if __name__ == "__main__":
    main()
