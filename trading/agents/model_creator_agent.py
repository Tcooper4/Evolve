"""
Enhanced Model Creator Agent

A sophisticated agent that can dynamically create, validate, test, and evaluate ML models
with comprehensive backtesting, performance metrics, and automatic model management.
"""

import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Try to import scikit-learn
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling sklearn-based models.")
    print(f"   Missing: {e}")
    GradientBoostingRegressor = None
    RandomForestRegressor = None
    ElasticNet = None
    Lasso = None
    Ridge = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    train_test_split = None
    MLPRegressor = None
    SVR = None
    SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print("⚠️ XGBoost not available. Disabling XGBoost models.")
    print(f"   Missing: {e}")
    xgb = None
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    print("⚠️ LightGBM not available. Disabling LightGBM models.")
    print(f"   Missing: {e}")
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Try to import PyTorch
try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print("⚠️ PyTorch not available. Disabling PyTorch models.")
    print(f"   Missing: {e}")
    nn = None
    TORCH_AVAILABLE = False
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelSpecification:
    """Enhanced specification for model creation."""

    name: str
    framework: str
    model_type: str
    parameters: Dict[str, Any]
    requirements: str
    created_at: datetime
    version: str = "1.0"
    description: str = ""
    architecture_blueprint: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    compilation_status: str = "pending"


@dataclass
class ModelEvaluation:
    """Enhanced model evaluation results."""

    model_name: str
    metrics: Dict[str, float]
    evaluation_date: datetime
    dataset_size: int
    training_time: float
    inference_time: float
    memory_usage: float
    validation_score: float = 0.0
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    performance_grade: str = "F"
    is_approved: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ModelLeaderboardEntry:
    """Leaderboard entry for model comparison."""

    model_name: str
    framework: str
    model_type: str
    created_at: datetime
    evaluation_date: datetime
    rmse: float
    mape: float
    mae: float
    sharpe_ratio: float
    win_rate: float
    overall_score: float
    performance_grade: str
    is_approved: bool
    rank: int = 0


class ModelValidator:
    """Model validation and compilation utilities."""

    @staticmethod
    def validate_model_specification(
        spec: ModelSpecification,
    ) -> Tuple[bool, List[str]]:
        """Validate model specification before creation."""
        errors = []

        # Check required fields
        if not spec.name or not spec.name.strip():
            errors.append("Model name is required")

        if not spec.framework or spec.framework not in [
            "sklearn",
            "xgboost",
            "lightgbm",
            "pytorch",
        ]:
            errors.append("Invalid framework specified")

        if not spec.model_type or spec.model_type not in [
            "regression",
            "classification",
            "forecasting",
        ]:
            errors.append("Invalid model type specified")

        # Validate parameters
        if spec.parameters:
            for param, value in spec.parameters.items():
                if isinstance(value, (int, float)) and value < 0:
                    errors.append(f"Parameter {param} cannot be negative")

        return len(errors) == 0, errors

    @staticmethod
    def compile_model(model, framework: str) -> Tuple[bool, str]:
        """Compile and validate model instance."""
        try:
            # Basic model validation
            if not hasattr(model, "fit"):
                return False, "Model does not have fit method"

            if not hasattr(model, "predict"):
                return False, "Model does not have predict method"

            # Framework-specific validation
            if framework == "sklearn":
                if not hasattr(model, "score"):
                    return False, "Sklearn model missing score method"

            elif framework == "xgboost":
                if not hasattr(model, "feature_importances_"):
                    return False, "XGBoost model missing feature_importances_"

            elif framework == "lightgbm":
                if not hasattr(model, "feature_importances_"):
                    return False, "LightGBM model missing feature_importances_"

            elif framework == "pytorch":
                if not isinstance(model, nn.Module):
                    return False, "PyTorch model must inherit from nn.Module"

            return True, "Model compiled successfully"

        except Exception as e:
            return False, f"Compilation error: {str(e)}"


class BacktestEngine:
    """Backtesting engine for model evaluation."""

    def __init__(self):
        self.backtest_results = {}

    def run_backtest(
        self, model, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Run comprehensive backtest on model."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Make predictions
            start_time = datetime.now()
            y_pred = model.predict(X_test)
            inference_time = (datetime.now() - start_time).total_seconds()

            # Calculate returns for trading metrics
            returns = np.diff(y_test) / y_test[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]

            # Trading signals (simple strategy)
            signals = np.where(pred_returns > 0, 1, -1)
            actual_direction = np.where(returns > 0, 1, -1)

            # Calculate trading metrics
            win_rate = np.mean(signals == actual_direction)

            # Sharpe ratio (simplified)
            if len(returns) > 0:
                sharpe_ratio = (
                    np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0.0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdown)

            return {
                "training_time": training_time,
                "inference_time": inference_time,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": cumulative_returns[-1] - 1
                if len(cumulative_returns) > 0
                else 0,
                "volatility": np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                "calmar_ratio": (np.mean(returns) * 252) / abs(max_drawdown)
                if max_drawdown != 0
                else 0,
            }

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {}


class ModelEvaluator:
    """Enhanced model evaluation with comprehensive metrics."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # MAPE
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

            # Directional accuracy
            if len(y_true) > 1:
                actual_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction)
            else:
                directional_accuracy = 0.0

            # Trading metrics (simplified)
            returns = np.diff(y_true) / (y_true[:-1] + 1e-8)
            pred_returns = np.diff(y_pred) / (y_pred[:-1] + 1e-8)

            signals = np.where(pred_returns > 0, 1, -1)
            actual_direction = np.where(returns > 0, 1, -1)
            win_rate = np.mean(signals == actual_direction)

            # Sharpe ratio
            if len(returns) > 0:
                sharpe_ratio = (
                    np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0.0

            return {
                "RMSE": rmse,
                "MAPE": mape,
                "MAE": mae,
                "R2": r2,
                "Directional_Accuracy": directional_accuracy,
                "Win_Rate": win_rate,
                "Sharpe_Ratio": sharpe_ratio,
                "MSE": mse,
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}

    @staticmethod
    def grade_performance(metrics: Dict[str, float]) -> Tuple[str, float]:
        """Grade model performance and calculate overall score."""
        try:
            # Normalize metrics to 0-1 scale
            normalized_metrics = {}

            # RMSE (lower is better)
            if "RMSE" in metrics:
                normalized_metrics["RMSE"] = max(0, 1 - metrics["RMSE"] / 100)

            # MAPE (lower is better)
            if "MAPE" in metrics:
                normalized_metrics["MAPE"] = max(0, 1 - metrics["MAPE"] / 100)

            # MAE (lower is better)
            if "MAE" in metrics:
                normalized_metrics["MAE"] = max(0, 1 - metrics["MAE"] / 100)

            # R2 (higher is better)
            if "R2" in metrics:
                normalized_metrics["R2"] = max(0, min(1, metrics["R2"]))

            # Directional accuracy (higher is better)
            if "Directional_Accuracy" in metrics:
                normalized_metrics["Directional_Accuracy"] = metrics[
                    "Directional_Accuracy"
                ]

            # Win rate (higher is better)
            if "Win_Rate" in metrics:
                normalized_metrics["Win_Rate"] = metrics["Win_Rate"]

            # Sharpe ratio (higher is better, normalize)
            if "Sharpe_Ratio" in metrics:
                normalized_metrics["Sharpe_Ratio"] = max(
                    0, min(1, metrics["Sharpe_Ratio"] / 2)
                )

            # Calculate overall score
            if normalized_metrics:
                overall_score = np.mean(list(normalized_metrics.values()))
            else:
                overall_score = 0.0

            # Assign grade
            if overall_score >= 0.9:
                grade = "A+"
            elif overall_score >= 0.8:
                grade = "A"
            elif overall_score >= 0.7:
                grade = "B+"
            elif overall_score >= 0.6:
                grade = "B"
            elif overall_score >= 0.5:
                grade = "C+"
            elif overall_score >= 0.4:
                grade = "C"
            elif overall_score >= 0.3:
                grade = "D"
            else:
                grade = "F"

            return grade, overall_score

        except Exception as e:
            self.logger.error(f"Error grading performance: {e}")
            return "F", 0.0


class ModelLeaderboard:
    """Centralized leaderboard for model comparison and management."""

    def __init__(self, leaderboard_file: str = "data/model_leaderboard.json"):
        self.leaderboard_file = Path(leaderboard_file)
        self.entries: List[ModelLeaderboardEntry] = []
        self.load_leaderboard()

    def add_entry(self, entry: ModelLeaderboardEntry):
        """Add a new entry to the leaderboard."""
        self.entries.append(entry)
        self._update_ranks()
        self.save_leaderboard()

    def _update_ranks(self):
        """Update ranks based on overall score."""
        # Sort by overall score (descending)
        self.entries.sort(key=lambda x: x.overall_score, reverse=True)

        # Update ranks
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1

    def get_top_models(self, n: int = 10) -> List[ModelLeaderboardEntry]:
        """Get top N models."""
        return self.entries[:n]

    def get_approved_models(self) -> List[ModelLeaderboardEntry]:
        """Get all approved models."""
        return [entry for entry in self.entries if entry.is_approved]

    def get_models_by_framework(self, framework: str) -> List[ModelLeaderboardEntry]:
        """Get models by framework."""
        return [entry for entry in self.entries if entry.framework == framework]

    def get_models_by_grade(self, grade: str) -> List[ModelLeaderboardEntry]:
        """Get models by performance grade."""
        return [entry for entry in self.entries if entry.performance_grade == grade]

    def remove_poor_models(self, threshold_score: float = 0.3) -> List[str]:
        """Remove models below threshold score."""
        poor_models = [
            entry for entry in self.entries if entry.overall_score < threshold_score
        ]
        removed_names = [entry.model_name for entry in poor_models]

        # Remove from entries
        self.entries = [
            entry for entry in self.entries if entry.overall_score >= threshold_score
        ]
        self._update_ranks()
        self.save_leaderboard()

        return removed_names

    def load_leaderboard(self):
        """Load leaderboard from file."""
        try:
            if self.leaderboard_file.exists():
                with open(self.leaderboard_file, "r") as f:
                    data = json.load(f)
                    self.entries = [ModelLeaderboardEntry(**entry) for entry in data]
                self.logger.info(f"Loaded {len(self.entries)} leaderboard entries")
        except Exception as e:
            self.logger.warning(f"Could not load leaderboard: {e}")

    def save_leaderboard(self):
        """Save leaderboard to file."""
        try:
            self.leaderboard_file.parent.mkdir(exist_ok=True)
            with open(self.leaderboard_file, "w") as f:
                json.dump(
                    [asdict(entry) for entry in self.entries], f, indent=2, default=str
                )
        except Exception as e:
            self.logger.error(f"Could not save leaderboard: {e}")


class EnhancedModelCreatorAgent:
    """Enhanced agent for dynamic model creation, validation, testing, and management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced model creator agent."""
        self.config = config or {}
        self.creation_history = []
        self.model_registry = {}
        self.framework_registry = self._initialize_framework_registry()

        # Initialize components
        self.validator = ModelValidator()
        self.backtest_engine = BacktestEngine()
        self.evaluator = ModelEvaluator()
        self.leaderboard = ModelLeaderboard()

        # Performance thresholds
        self.min_score_threshold = self.config.get("min_score_threshold", 0.3)
        self.auto_approval_threshold = self.config.get("auto_approval_threshold", 0.7)

        # Load existing models
        self._load_existing_models()

        self.logger.info("Enhanced Model Creator Agent initialized")

    def _initialize_framework_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available ML frameworks."""
        registry = {
            "sklearn": {
                "available": SKLEARN_AVAILABLE,
                "models": {
                    "RandomForest": RandomForestRegressor
                    if SKLEARN_AVAILABLE
                    else None,
                    "GradientBoosting": GradientBoostingRegressor
                    if SKLEARN_AVAILABLE
                    else None,
                    "Ridge": Ridge if SKLEARN_AVAILABLE else None,
                    "Lasso": Lasso if SKLEARN_AVAILABLE else None,
                    "ElasticNet": ElasticNet if SKLEARN_AVAILABLE else None,
                    "SVR": SVR if SKLEARN_AVAILABLE else None,
                    "MLP": MLPRegressor if SKLEARN_AVAILABLE else None,
                },
                "description": "Scikit-learn models for traditional ML",
            },
            "xgboost": {
                "available": XGBOOST_AVAILABLE,
                "models": {
                    "XGBRegressor": xgb.XGBRegressor if XGBOOST_AVAILABLE else None
                },
                "description": "XGBoost for gradient boosting",
            },
            "lightgbm": {
                "available": LIGHTGBM_AVAILABLE,
                "models": {
                    "LGBMRegressor": lgb.LGBMRegressor if LIGHTGBM_AVAILABLE else None
                },
                "description": "LightGBM for fast gradient boosting",
            },
        }

        # Add PyTorch if available
        if PYTORCH_AVAILABLE:
            registry["pytorch"] = {
                "available": True,
                "models": {"LSTM": "LSTM", "GRU": "GRU", "Transformer": "Transformer"},
                "description": "PyTorch for deep learning",
            }

        return registry

    def _load_existing_models(self):
        """Load existing models from storage."""
        try:
            model_file = Path("data/model_registry.json")
            if model_file.exists():
                with open(model_file, "r") as f:
                    self.model_registry = json.load(f)
                self.logger.info(f"Loaded {len(self.model_registry)} existing models")
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")

    def _save_model_registry(self):
        """Save model registry to storage."""
        try:
            model_file = Path("data/model_registry.json")
            model_file.parent.mkdir(exist_ok=True)
            with open(model_file, "w") as f:
                json.dump(self.model_registry, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save model registry: {e}")

    def parse_requirements(self, requirements: str) -> Dict[str, Any]:
        """Parse natural language requirements into structured format."""
        requirements_lower = requirements.lower()

        # Extract model type
        model_type = "regression"  # default
        if any(
            word in requirements_lower
            for word in ["classification", "classify", "class"]
        ):
            model_type = "classification"
        elif any(
            word in requirements_lower
            for word in ["forecast", "time series", "sequence"]
        ):
            model_type = "forecasting"

        # Extract framework preference
        framework = "auto"
        if "sklearn" in requirements_lower or "scikit" in requirements_lower:
            framework = "sklearn"
        elif "xgboost" in requirements_lower or "xgb" in requirements_lower:
            framework = "xgboost"
        elif "lightgbm" in requirements_lower or "lgb" in requirements_lower:
            framework = "lightgbm"
        elif "pytorch" in requirements_lower or "torch" in requirements_lower:
            framework = "pytorch"

        # Extract complexity level
        complexity = "medium"
        if any(word in requirements_lower for word in ["simple", "basic", "fast"]):
            complexity = "simple"
        elif any(
            word in requirements_lower for word in ["complex", "advanced", "deep"]
        ):
            complexity = "complex"

        # Extract specific parameters
        parameters = {}

        # Extract numbers for common parameters
        n_estimators_match = re.search(
            r"(\d+)\s*(estimators?|trees?)", requirements_lower
        )
        if n_estimators_match:
            parameters["n_estimators"] = int(n_estimators_match.group(1))

        max_depth_match = re.search(r"max\s*depth\s*(\d+)", requirements_lower)
        if max_depth_match:
            parameters["max_depth"] = int(max_depth_match.group(1))

        learning_rate_match = re.search(
            r"learning\s*rate\s*([\d.]+)", requirements_lower
        )
        if learning_rate_match:
            parameters["learning_rate"] = float(learning_rate_match.group(1))

        return {
            "model_type": model_type,
            "framework": framework,
            "complexity": complexity,
            "parameters": parameters,
            "original_requirements": requirements,
        }

    def select_framework(self, parsed_requirements: Dict[str, Any]) -> str:
        """Select the best framework based on requirements."""
        framework_pref = parsed_requirements["framework"]
        complexity = parsed_requirements["complexity"]
        model_type = parsed_requirements["model_type"]

        if framework_pref != "auto":
            if self.framework_registry[framework_pref]["available"]:
                return framework_pref
            else:
                self.logger.warning(f"Preferred framework {framework_pref} not available")

        # Auto-select based on requirements
        if model_type == "forecasting" and PYTORCH_AVAILABLE:
            return "pytorch"
        elif complexity == "simple" and SKLEARN_AVAILABLE:
            return "sklearn"
        elif complexity == "complex" and XGBOOST_AVAILABLE:
            return "xgboost"
        elif SKLEARN_AVAILABLE:
            return "sklearn"
        elif XGBOOST_AVAILABLE:
            return "xgboost"
        elif LIGHTGBM_AVAILABLE:
            return "lightgbm"
        else:
            raise ValueError("No suitable ML framework available")

    def create_and_validate_model(
        self, requirements: str, model_name: Optional[str] = None
    ) -> Tuple[ModelSpecification, bool, List[str]]:
        """Create a model with automatic validation and compilation."""
        try:
            logger.info(
                f"Creating and validating model with requirements: {requirements}"
            )

            # Parse requirements
            parsed_req = self.parse_requirements(requirements)

            # Select framework
            framework = self.select_framework(parsed_req)

            # Generate model name if not provided
            if not model_name:
                model_name = f"{framework}_{parsed_req['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get default parameters for the framework
            default_params = self._get_default_parameters(framework, parsed_req)

            # Merge with user-specified parameters
            final_params = {**default_params, **parsed_req["parameters"]}

            # Create model specification
            spec = ModelSpecification(
                name=model_name,
                framework=framework,
                model_type=parsed_req["model_type"],
                parameters=final_params,
                requirements=requirements,
                created_at=datetime.now(),
                description=f"Model created from requirements: {requirements[:100]}...",
                architecture_blueprint=self._create_architecture_blueprint(
                    framework, final_params
                ),
            )

            # Validate specification
            is_valid, validation_errors = self.validator.validate_model_specification(
                spec
            )

            if not is_valid:
                spec.validation_status = "failed"
                return spec, False, validation_errors

            # Create model instance and compile
            try:
                model = self._create_model_instance(spec)
                is_compiled, compilation_message = self.validator.compile_model(
                    model, framework
                )

                if is_compiled:
                    spec.validation_status = "passed"
                    spec.compilation_status = "success"
                else:
                    spec.validation_status = "failed"
                    spec.compilation_status = "failed"
                    validation_errors.append(compilation_message)

            except Exception as e:
                spec.validation_status = "failed"
                spec.compilation_status = "failed"
                validation_errors.append(f"Model creation error: {str(e)}")

            # Add to registry if valid
            if spec.validation_status == "passed":
                self.model_registry[model_name] = asdict(spec)
                self.creation_history.append(asdict(spec))
                self._save_model_registry()
                self.logger.info(f"Successfully created and validated model: {model_name}")

            return spec, spec.validation_status == "passed", validation_errors

        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise

    def _create_architecture_blueprint(
        self, framework: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create architecture blueprint for model."""
        blueprint = {
            "framework": framework,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        if framework == "pytorch":
            blueprint["architecture_type"] = "neural_network"
            blueprint["layers"] = self._get_pytorch_architecture(parameters)
        else:
            blueprint["architecture_type"] = "traditional_ml"

        return blueprint

    def _get_pytorch_architecture(
        self, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get PyTorch architecture configuration."""
        # Default LSTM architecture
        return [
            {"type": "LSTM", "input_size": 10, "hidden_size": 64, "num_layers": 2},
            {"type": "Linear", "in_features": 64, "out_features": 32},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 32, "out_features": 1},
        ]

    def run_full_evaluation(
        self,
        model_name: str,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> ModelEvaluation:
        """Run comprehensive evaluation including forecasting and backtesting."""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")

            spec_dict = self.model_registry[model_name]
            spec = ModelSpecification(**spec_dict)

            # Create model instance
            model = self._create_model_instance(spec)

            # Generate data if not provided
            if X is None or y is None:
                X, y = self._generate_mock_data()

            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(model, X, y)

            # Calculate comprehensive metrics
            y_pred = model.predict(X)
            metrics = self.evaluator.calculate_all_metrics(y, y_pred)

            # Grade performance
            grade, overall_score = self.evaluator.grade_performance(metrics)

            # Determine if model should be approved
            is_approved = overall_score >= self.auto_approval_threshold

            # Generate recommendations
            recommendations = self._generate_recommendations(
                metrics, overall_score, spec
            )

            # Create evaluation
            evaluation = ModelEvaluation(
                model_name=model_name,
                metrics=metrics,
                evaluation_date=datetime.now(),
                dataset_size=len(X),
                training_time=backtest_results.get("training_time", 0),
                inference_time=backtest_results.get("inference_time", 0),
                memory_usage=self._estimate_memory_usage(model),
                validation_score=overall_score,
                backtest_results=backtest_results,
                performance_grade=grade,
                is_approved=is_approved,
                recommendations=recommendations,
            )

            # Update model registry
            self.model_registry[model_name]["evaluation"] = asdict(evaluation)
            self._save_model_registry()

            # Add to leaderboard
            leaderboard_entry = ModelLeaderboardEntry(
                model_name=model_name,
                framework=spec.framework,
                model_type=spec.model_type,
                created_at=spec.created_at,
                evaluation_date=evaluation.evaluation_date,
                rmse=metrics.get("RMSE", 0),
                mape=metrics.get("MAPE", 0),
                mae=metrics.get("MAE", 0),
                sharpe_ratio=metrics.get("Sharpe_Ratio", 0),
                win_rate=metrics.get("Win_Rate", 0),
                overall_score=overall_score,
                performance_grade=grade,
                is_approved=is_approved,
            )
            self.leaderboard.add_entry(leaderboard_entry)

            self.logger.info(
                f"Successfully evaluated model: {model_name} (Grade: {grade}, Score: {overall_score:.3f})"
            )
            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            raise

    def _generate_recommendations(
        self, metrics: Dict[str, float], overall_score: float, spec: ModelSpecification
    ) -> List[str]:
        """Generate improvement recommendations for the model."""
        recommendations = []

        if overall_score < 0.5:
            recommendations.append("Consider using a more complex model architecture")
            recommendations.append("Increase training data or use data augmentation")
            recommendations.append("Try ensemble methods or different frameworks")

        if metrics.get("RMSE", float("inf")) > 10:
            recommendations.append("High RMSE: Consider feature engineering or scaling")

        if metrics.get("MAPE", float("inf")) > 20:
            recommendations.append("High MAPE: Model may need better feature selection")

        if metrics.get("Win_Rate", 0) < 0.5:
            recommendations.append(
                "Low win rate: Consider directional prediction models"
            )

        if metrics.get("Sharpe_Ratio", 0) < 0.5:
            recommendations.append(
                "Low Sharpe ratio: Consider risk-adjusted optimization"
            )

        if spec.framework == "sklearn" and overall_score < 0.6:
            recommendations.append("Consider upgrading to XGBoost or LightGBM")

        return recommendations

    def auto_cleanup_poor_models(self) -> List[str]:
        """Automatically remove poor performing models."""
        removed_models = self.leaderboard.remove_poor_models(self.min_score_threshold)

        # Also remove from registry
        for model_name in removed_models:
            if model_name in self.model_registry:
                del self.model_registry[model_name]

        self._save_model_registry()

        if removed_models:
            logger.info(
                f"Automatically removed {len(removed_models)} poor performing models"
            )

        return removed_models

    def get_model_suggestions(self, requirements: str) -> List[Dict[str, Any]]:
        """Get model improvement suggestions based on requirements."""
        suggestions = []

        # Parse requirements to understand context
        parsed_req = self.parse_requirements(requirements)

        # Get top performing models for comparison
        top_models = self.leaderboard.get_top_models(5)

        for model in top_models:
            if model.framework != parsed_req["framework"]:
                suggestions.append(
                    {
                        "type": "framework_upgrade",
                        "current": parsed_req["framework"],
                        "suggested": model.framework,
                        "reason": f"Top performing model uses {model.framework}",
                        "expected_improvement": f"Score: {model.overall_score:.3f}",
                    }
                )

        # Add general suggestions
        if parsed_req["complexity"] == "simple":
            suggestions.append(
                {
                    "type": "complexity_upgrade",
                    "current": "simple",
                    "suggested": "medium",
                    "reason": "Simple models may underperform on complex data",
                    "expected_improvement": "Better accuracy and robustness",
                }
            )

        return suggestions

    def save_model_blueprint(self, model_name: str, filepath: str) -> bool:
        """Save model architecture blueprint for future reuse."""
        try:
            if model_name not in self.model_registry:
                return False

            spec_dict = self.model_registry[model_name]
            blueprint = {
                "specification": spec_dict,
                "architecture": spec_dict.get("architecture_blueprint", {}),
                "performance": spec_dict.get("evaluation", {}),
                "created_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(blueprint, f, indent=2, default=str)

            self.logger.info(f"Saved model blueprint to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model blueprint: {e}")
            return False

    def load_model_blueprint(self, filepath: str) -> Optional[ModelSpecification]:
        """Load model architecture blueprint for reuse."""
        try:
            with open(filepath, "r") as f:
                blueprint = json.load(f)

            spec_data = blueprint["specification"]
            spec = ModelSpecification(**spec_data)

            self.logger.info(f"Loaded model blueprint from {filepath}")
            return spec

        except Exception as e:
            self.logger.error(f"Error loading model blueprint: {e}")
            return None

    # Existing helper methods (updated for enhanced functionality)
    def _get_default_parameters(
        self, framework: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get default parameters for a framework and complexity level."""
        complexity = requirements["complexity"]

        if framework == "sklearn":
            if complexity == "simple":
                return {"n_estimators": 50, "max_depth": 5, "random_state": 42}
            elif complexity == "complex":
                return {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                }
            else:  # medium
                return {
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "random_state": 42,
                }

        elif framework == "xgboost":
            if complexity == "simple":
                return {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "random_state": 42,
                }
            elif complexity == "complex":
                return {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                }
            else:  # medium
                return {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": 42,
                }

        elif framework == "lightgbm":
            if complexity == "simple":
                return {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "random_state": 42,
                }
            elif complexity == "complex":
                return {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                }
            else:  # medium
                return {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": 42,
                }

        elif framework == "pytorch":
            return {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
            }

        return {}

    def _create_model_instance(self, spec: ModelSpecification):
        """Create a model instance from specification."""
        framework = spec.framework
        model_type = spec.model_type
        params = spec.parameters

        if framework == "sklearn":
            if model_type == "regression":
                return RandomForestRegressor(**params)
            elif model_type == "classification":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**params)

        elif framework == "xgboost":
            if model_type == "regression":
                return xgb.XGBRegressor(**params)
            elif model_type == "classification":
                return xgb.XGBClassifier(**params)

        elif framework == "lightgbm":
            if model_type == "regression":
                return lgb.LGBMRegressor(**params)
            elif model_type == "classification":
                return lgb.LGBMClassifier(**params)

        elif framework == "pytorch":
            return self._create_pytorch_model(params, model_type)

        raise ValueError(f"Unsupported framework: {framework}")

    def _create_pytorch_model(self, params: Dict[str, Any], model_type: str):
        """Create PyTorch model instance."""
        if not PYTORCH_AVAILABLE:
            raise ValueError("PyTorch not available")

        # Simple LSTM model for demonstration
        class SimpleLSTM(nn.Module):
            def __init__(
                self, input_size=10, hidden_size=64, num_layers=2, output_size=1
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        return SimpleLSTM(
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
        )

    def _generate_mock_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock data for evaluation."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        return X, y

    def _estimate_memory_usage(self, model) -> float:
        """Estimate model memory usage in MB."""
        try:
            import sys

            return sys.getsizeof(model) / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

    # Public interface methods
    def get_creation_history(self) -> List[Dict[str, Any]]:
        """Get model creation history."""
        return self.creation_history

    def get_model_registry(self) -> Dict[str, Any]:
        """Get current model registry."""
        return self.model_registry

    def get_framework_status(self) -> Dict[str, bool]:
        """Get status of available frameworks."""
        return {
            name: info["available"] for name, info in self.framework_registry.items()
        }

    def get_leaderboard(self) -> List[ModelLeaderboardEntry]:
        """Get current leaderboard."""
        return self.leaderboard.entries

    def get_top_performing_models(self, n: int = 10) -> List[ModelLeaderboardEntry]:
        """Get top N performing models."""
        return self.leaderboard.get_top_models(n)

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from the registry."""
        try:
            if model_name in self.model_registry:
                del self.model_registry[model_name]
                self._save_model_registry()
                self.logger.info(f"Deleted model: {model_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False


# Factory function


def get_model_creator_agent(
    config: Optional[Dict[str, Any]] = None
) -> EnhancedModelCreatorAgent:
    """Get a configured enhanced model creator agent."""
    return EnhancedModelCreatorAgent(config)
