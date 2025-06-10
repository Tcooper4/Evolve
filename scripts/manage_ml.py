#!/usr/bin/env python3
"""
ML pipeline management script.
Provides commands for managing machine learning pipelines and model versioning.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import joblib
import pickle
import dill
import cloudpickle
import onnx
import onnxruntime
import tensorrt
import openvino
import tvm
import ray.serve
import bentoml
import seldon_core
import kserve
import triton
import torchserve
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.spark
import mlflow.tensorflow
import mlflow.keras
import mlflow.h2o
import mlflow.statsmodels
import mlflow.prophet
import mlflow.pmdarima
import mlflow.spacy
import mlflow.fastai
import mlflow.gluon
import mlflow.mleap
import mlflow.onnx
import mlflow.paddle
import mlflow.pyspark.ml
import mlflow.sklearn
import mlflow.spark
import mlflow.tensorflow
import mlflow.keras
import mlflow.pytorch
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.h2o
import mlflow.statsmodels
import mlflow.prophet
import mlflow.pmdarima
import mlflow.spacy
import mlflow.fastai
import mlflow.gluon
import mlflow.mleap
import mlflow.onnx
import mlflow.paddle
import mlflow.pyspark.ml

class MLManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the ML manager."""
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
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    async def train_model(self, model_type: str, data_path: str, params: Optional[Dict[str, Any]] = None):
        """Train a machine learning model."""
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
                if model_type in ["pytorch", "tensorflow", "keras"]:
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
        """Optimize model hyperparameters."""
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
                if model_type in ["pytorch", "tensorflow", "keras"]:
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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            
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
            elif model_type == "tensorflow":
                return self._initialize_tensorflow_model(params)
            elif model_type == "keras":
                return self._initialize_keras_model(params)
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
                nn.Sigmoid()
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize PyTorch model: {e}")
            raise

    def _initialize_tensorflow_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a TensorFlow model."""
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(params.get("hidden_size", 64), activation="relu"),
                tf.keras.layers.Dropout(params.get("dropout", 0.2)),
                tf.keras.layers.Dense(params.get("output_size", 1), activation="sigmoid")
            ])
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorFlow model: {e}")
            raise

    def _initialize_keras_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a Keras model."""
        try:
            from keras.models import Sequential
            from keras.layers import Dense, Dropout
            model = Sequential([
                Dense(params.get("hidden_size", 64), activation="relu"),
                Dropout(params.get("dropout", 0.2)),
                Dense(params.get("output_size", 1), activation="sigmoid")
            ])
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize Keras model: {e}")
            raise

    def _initialize_sklearn_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize a scikit-learn model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42
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
                random_state=42
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
                random_state=42
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
                random_state=42
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
            else:
                # TensorFlow/Keras model
                model.compile(
                    optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2
                )
            
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

    async def _optimize_deep_learning_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, List[Any]]) -> Any:
        """Optimize a deep learning model."""
        try:
            if isinstance(model, nn.Module):
                # PyTorch model
                def objective(trial):
                    model = self._initialize_pytorch_model({
                        "input_size": trial.suggest_int("input_size", 10, 100),
                        "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                        "output_size": 1
                    })
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
            else:
                # TensorFlow/Keras model
                def objective(trial):
                    model = self._initialize_tensorflow_model({
                        "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                        "output_size": 1
                    })
                    model.compile(
                        optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"]
                    )
                    history = model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2
                    )
                    return history.history["val_loss"][-1]
                
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=100)
                
                return self._initialize_tensorflow_model(study.best_params)
        except Exception as e:
            self.logger.error(f"Failed to optimize deep learning model: {e}")
            raise

    async def _optimize_traditional_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, List[Any]]) -> Any:
        """Optimize a traditional model."""
        try:
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
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
                "f1": f1_score(y_test, y_pred)
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
            elif model_type in ["tensorflow", "keras"]:
                model.save(model_path)
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
            elif model_path.endswith(".h5"):
                # TensorFlow/Keras model
                from tensorflow.keras.models import load_model
                return load_model(model_path)
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
            mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path="model"
            )
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
            bento_model = bentoml.pytorch.save_model(
                "model",
                model,
                signatures={"predict": {"batchable": True}}
            )
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
    """Main function."""
    parser = argparse.ArgumentParser(description="ML Manager")
    parser.add_argument(
        "command",
        choices=["train", "optimize", "evaluate", "deploy", "monitor"],
        help="Command to execute"
    )
    parser.add_argument(
        "--model-type",
        choices=["pytorch", "tensorflow", "keras", "sklearn", "xgboost", "lightgbm", "catboost"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--data-path",
        help="Path to data file"
    )
    parser.add_argument(
        "--model-path",
        help="Path to model file"
    )
    parser.add_argument(
        "--deployment-type",
        choices=["local", "mlflow", "ray", "bentoml", "seldon", "kserve", "triton", "torchserve"],
        default="local",
        help="Type of deployment to use"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for monitoring in seconds"
    )
    
    args = parser.parse_args()
    manager = MLManager()
    
    commands = {
        "train": lambda: asyncio.run(
            manager.train_model(args.model_type, args.data_path)
        ),
        "optimize": lambda: asyncio.run(
            manager.optimize_model(args.model_type, args.data_path, {})
        ),
        "evaluate": lambda: asyncio.run(
            manager.evaluate_model(args.model_path, args.data_path)
        ),
        "deploy": lambda: asyncio.run(
            manager.deploy_model(args.model_path, args.deployment_type)
        ),
        "monitor": lambda: asyncio.run(
            manager.monitor_model(args.model_path, args.data_path, args.duration)
        )
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 