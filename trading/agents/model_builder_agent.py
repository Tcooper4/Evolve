"""
Model Builder Agent

This agent is responsible for building and initializing various ML models
including LSTM, XGBoost, and ensemble models from scratch.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import pickle

# Local imports
from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.models.lstm_model import LSTMForecaster
from trading.models.xgboost_model import XGBoostForecaster
from trading.models.ensemble_model import EnsembleForecaster
from trading.data.preprocessing import DataPreprocessor
from trading.feature_engineering.feature_engineer import FeatureEngineering
from trading.utils.common import timer, handle_exceptions
from trading.memory.performance_memory import PerformanceMemory
from trading.memory.agent_memory import AgentMemory
from trading.utils.reward_function import RewardFunction


@dataclass
class ModelBuildRequest:
    """Request for model building."""
    model_type: str  # 'lstm', 'xgboost', 'ensemble'
    data_path: str
    target_column: str
    features: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    random_state: int = 42
    request_id: Optional[str] = None


@dataclass
class ModelBuildResult:
    """Result of model building."""
    request_id: str
    model_type: str
    model_path: str
    model_id: str
    build_timestamp: str
    training_metrics: Dict[str, float]
    model_config: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    build_status: str = "success"
    error_message: Optional[str] = None


class ModelBuilderAgent(BaseAgent):
    """Agent responsible for building ML models from scratch."""
    
    # Agent metadata
    version = "1.0.0"
    description = "Builds and initializes various ML models including LSTM, XGBoost, and ensemble models"
    author = "Evolve Trading System"
    tags = ["model-building", "ml", "training"]
    capabilities = ["lstm_building", "xgboost_building", "ensemble_building", "hyperparameter_tuning"]
    dependencies = ["trading.models", "trading.data", "trading.feature_engineering"]
    
    def _setup(self) -> None:
        """Setup method called during initialization."""
        self.memory = PerformanceMemory()
        self.models_dir = Path("trading/models/built")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineering()
        
        # Model registry
        self.model_registry: Dict[str, ModelBuildResult] = {}
        
        self.agent_memory = AgentMemory("trading/agents/agent_memory.json")
        
        self.reward_function = RewardFunction()
        
        self.logger.info("ModelBuilderAgent initialized")
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model building logic.
        
        Args:
            **kwargs: Must contain 'request' with ModelBuildRequest
            
        Returns:
            AgentResult: Result of the model building execution
        """
        request = kwargs.get('request')
        if not request:
            return AgentResult(
                success=False,
                error_message="ModelBuildRequest is required"
            )
        
        if not isinstance(request, ModelBuildRequest):
            return AgentResult(
                success=False,
                error_message="Request must be a ModelBuildRequest instance"
            )
        
        try:
            result = self.build_model(request)
            
            if result.build_status == "success":
                return AgentResult(
                    success=True,
                    data={
                        "model_id": result.model_id,
                        "model_type": result.model_type,
                        "model_path": result.model_path,
                        "training_metrics": result.training_metrics,
                        "build_timestamp": result.build_timestamp
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=result.error_message or "Model building failed"
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if input is valid
        """
        request = kwargs.get('request')
        if not request:
            return False
        
        if not isinstance(request, ModelBuildRequest):
            return False
        
        # Validate required fields
        if not request.model_type or not request.data_path or not request.target_column:
            return False
        
        # Validate model type
        valid_types = ['lstm', 'xgboost', 'ensemble']
        if request.model_type.lower() not in valid_types:
            return False
        
        # Validate data path exists
        if not Path(request.data_path).exists():
            return False
        
        return True
    
    @handle_exceptions
    def build_model(self, request: ModelBuildRequest) -> ModelBuildResult:
        """Build a model based on the request.
        
        Args:
            request: Model build request
            
        Returns:
            Model build result
        """
        request_id = request.request_id or str(uuid.uuid4())
        self.logger.info(f"Building {request.model_type} model with ID: {request_id}")
        
        try:
            # Load and preprocess data
            data = self._load_and_preprocess_data(request)
            
            # Build model based on type
            if request.model_type.lower() == 'lstm':
                result = self._build_lstm_model(request, data, request_id)
            elif request.model_type.lower() == 'xgboost':
                result = self._build_xgboost_model(request, data, request_id)
            elif request.model_type.lower() == 'ensemble':
                result = self._build_ensemble_model(request, data, request_id)
            else:
                raise ValueError(f"Unsupported model type: {request.model_type}")
            
            # Save model and metadata
            self._save_model_metadata(result)
            # Compute reward score for the model
            reward = self.reward_function.compute(result.training_metrics)
            # Log outcome to agent memory (add reward)
            self.agent_memory.log_outcome(
                agent="ModelBuilderAgent",
                run_type="build",
                outcome={
                    "model_id": result.model_id,
                    "model_type": result.model_type,
                    "status": result.build_status,
                    "training_metrics": result.training_metrics,
                    "reward": reward,
                    "timestamp": result.build_timestamp
                }
            )
            
            # Update registry
            self.model_registry[request_id] = result
            
            self.logger.info(f"Successfully built {request.model_type} model: {request_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to build model {request_id}: {str(e)}")
            self.agent_memory.log_outcome(
                agent="ModelBuilderAgent",
                run_type="build",
                outcome={
                    "model_id": request_id,
                    "model_type": request.model_type,
                    "status": "failed",
                    "error_message": str(e),
                    "reward": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return ModelBuildResult(
                request_id=request_id,
                model_type=request.model_type,
                model_path="",
                model_id="",
                build_timestamp=datetime.now().isoformat(),
                training_metrics={},
                model_config={},
                build_status="failed",
                error_message=str(e)
            )
    
    def _load_and_preprocess_data(self, request: ModelBuildRequest) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data for model training.
        
        Args:
            request: Model build request
            
        Returns:
            Tuple of features and target
        """
        # Load data
        if request.data_path.endswith('.csv'):
            data = pd.read_csv(request.data_path)
        elif request.data_path.endswith('.parquet'):
            data = pd.read_parquet(request.data_path)
        else:
            raise ValueError(f"Unsupported data format: {request.data_path}")
        
        # Preprocess data
        data = self.preprocessor.fit_transform(data)
        
        # Engineer features
        data = self.feature_engineer.engineer_features(data)
        
        # Split features and target
        if request.features:
            X = data[request.features]
        else:
            # Use all numeric columns except target
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_cols].drop(columns=[request.target_column])
        
        y = data[request.target_column]
        
        return X, y
    
    @timer
    def _build_lstm_model(self, request: ModelBuildRequest, data: Tuple[pd.DataFrame, pd.Series], request_id: str) -> ModelBuildResult:
        """Build LSTM model.
        
        Args:
            request: Model build request
            data: Tuple of features and target
            request_id: Request ID
            
        Returns:
            Model build result
        """
        X, y = data
        
        # Default LSTM hyperparameters
        hyperparams = request.hyperparameters or {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'sequence_length': 20
        }
        
        # Initialize and train model
        model = LSTMForecaster(hyperparams)
        
        # Split data
        split_idx = int(len(X) * (1 - request.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Evaluate model
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        training_metrics = {
            'train_mse': np.mean((y_train - train_pred) ** 2),
            'val_mse': np.mean((y_val - val_pred) ** 2),
            'train_mae': np.mean(np.abs(y_train - train_pred)),
            'val_mae': np.mean(np.abs(y_val - val_pred))
        }
        
        # Save model
        model_id = f"lstm_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_id}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return ModelBuildResult(
            request_id=request_id,
            model_type='lstm',
            model_path=str(model_path),
            model_id=model_id,
            build_timestamp=datetime.now().isoformat(),
            training_metrics=training_metrics,
            model_config=hyperparams
        )
    
    @timer
    def _build_xgboost_model(self, request: ModelBuildRequest, data: Tuple[pd.DataFrame, pd.Series], request_id: str) -> ModelBuildResult:
        """Build XGBoost model.
        
        Args:
            request: Model build request
            data: Tuple of features and target
            request_id: Request ID
            
        Returns:
            Model build result
        """
        X, y = data
        
        # Default XGBoost hyperparameters
        hyperparams = request.hyperparameters or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': request.random_state
        }
        
        # Initialize and train model
        model = XGBoostForecaster(hyperparams)
        
        # Split data
        split_idx = int(len(X) * (1 - request.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        training_metrics = {
            'train_mse': np.mean((y_train - train_pred) ** 2),
            'val_mse': np.mean((y_val - val_pred) ** 2),
            'train_mae': np.mean(np.abs(y_train - train_pred)),
            'val_mae': np.mean(np.abs(y_val - val_pred))
        }
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.model.feature_importances_))
        
        # Save model
        model_id = f"xgboost_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_id}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return ModelBuildResult(
            request_id=request_id,
            model_type='xgboost',
            model_path=str(model_path),
            model_id=model_id,
            build_timestamp=datetime.now().isoformat(),
            training_metrics=training_metrics,
            model_config=hyperparams,
            feature_importance=feature_importance
        )
    
    @timer
    def _build_ensemble_model(self, request: ModelBuildRequest, data: Tuple[pd.DataFrame, pd.Series], request_id: str) -> ModelBuildResult:
        """Build ensemble model.
        
        Args:
            request: Model build request
            data: Tuple of features and target
            request_id: Request ID
            
        Returns:
            Model build result
        """
        X, y = data
        
        # Default ensemble hyperparameters
        hyperparams = request.hyperparameters or {
            'models': ['lstm', 'xgboost'],
            'weights': [0.5, 0.5],
            'voting_method': 'weighted_average'
        }
        
        # Build individual models first
        lstm_request = ModelBuildRequest(
            model_type='lstm',
            data_path=request.data_path,
            target_column=request.target_column,
            features=request.features,
            validation_split=request.validation_split,
            random_state=request.random_state
        )
        
        xgb_request = ModelBuildRequest(
            model_type='xgboost',
            data_path=request.data_path,
            target_column=request.target_column,
            features=request.features,
            validation_split=request.validation_split,
            random_state=request.random_state
        )
        
        lstm_result = self._build_lstm_model(lstm_request, (X, y), f"{request_id}_lstm")
        xgb_result = self._build_xgboost_model(xgb_request, (X, y), f"{request_id}_xgb")
        
        # Create ensemble
        ensemble_config = {
            'models': [
                {'type': 'lstm', 'path': lstm_result.model_path, 'weight': 0.5},
                {'type': 'xgboost', 'path': xgb_result.model_path, 'weight': 0.5}
            ],
            'voting_method': hyperparams['voting_method']
        }
        
        # Save ensemble configuration
        model_id = f"ensemble_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_id}.json"
        
        with open(model_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        # Combine metrics
        training_metrics = {
            'lstm_val_mse': lstm_result.training_metrics['val_mse'],
            'xgb_val_mse': xgb_result.training_metrics['val_mse'],
            'ensemble_val_mse': (lstm_result.training_metrics['val_mse'] + xgb_result.training_metrics['val_mse']) / 2
        }
        
        return ModelBuildResult(
            request_id=request_id,
            model_type='ensemble',
            model_path=str(model_path),
            model_id=model_id,
            build_timestamp=datetime.now().isoformat(),
            training_metrics=training_metrics,
            model_config=ensemble_config
        )
    
    def _save_model_metadata(self, result: ModelBuildResult) -> None:
        """Save model metadata to memory.
        
        Args:
            result: Model build result
        """
        metadata = {
            'model_id': result.model_id,
            'model_type': result.model_type,
            'build_timestamp': result.build_timestamp,
            'training_metrics': result.training_metrics,
            'model_config': result.model_config,
            'status': result.build_status
        }
        
        self.memory.store_model_metadata(result.model_id, metadata)
    
    def get_model_status(self, model_id: str) -> Optional[ModelBuildResult]:
        """Get status of a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model build result if found
        """
        return self.model_registry.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all built models.
        
        Returns:
            List of model information
        """
        return [
            {
                'model_id': result.model_id,
                'model_type': result.model_type,
                'build_timestamp': result.build_timestamp,
                'status': result.build_status,
                'training_metrics': result.training_metrics
            }
            for result in self.model_registry.values()
        ]
    
    def cleanup_old_models(self, max_age_days: int = 30) -> int:
        """Clean up old models.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of models cleaned up
        """
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for model_id, result in list(self.model_registry.items()):
            build_timestamp = datetime.fromisoformat(result.build_timestamp).timestamp()
            if build_timestamp < cutoff_date:
                # Remove model file
                if Path(result.model_path).exists():
                    Path(result.model_path).unlink()
                
                # Remove from registry
                del self.model_registry[model_id]
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old models")
        return cleaned_count 