"""
Rolling Retraining + Walk-Forward Agent

Implements walk-forward validation and rolling retraining for continuous model improvement.
Provides performance tracking and automatic model updates.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from trading.utils.safe_math import safe_rsi
from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_score: float
    test_score: float
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    predictions: List[float]
    actuals: List[float]


@dataclass
class RetrainingConfig:
    """Configuration for rolling retraining."""

    retrain_frequency: int = 30  # days
    lookback_window: int = 252  # days
    min_train_size: int = 60  # days
    test_size: int = 20  # days
    performance_threshold: float = 0.1  # 10% degradation threshold
    max_models: int = 10  # maximum models to keep


@dataclass
class RetrainingRequest:
    """Request for model retraining."""

    symbol: str
    data: pd.DataFrame
    model_type: str = "auto"
    target_col: str = "returns"
    force_retrain: bool = False
    performance_threshold: Optional[float] = None
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class RetrainingResult:
    """Result from model retraining."""

    success: bool
    model_path: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    retrain_date: Optional[datetime] = None


class RollingRetrainingAgent(BaseAgent):
    """Advanced rolling retraining and walk-forward validation agent."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        model_dir: str = "models/rolling_retraining",
        results_dir: str = "logs/retraining_results",
    ):
        if config is None:
            config = AgentConfig(
                name="RollingRetrainingAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={},
            )
        super().__init__(config)

        # Extract retraining config from custom_config or use defaults
        custom_config = config.custom_config or {}
        self.retraining_config = RetrainingConfig(
            retrain_frequency=custom_config.get("retrain_frequency", 30),
            lookback_window=custom_config.get("lookback_window", 252),
            min_train_size=custom_config.get("min_train_size", 60),
            test_size=custom_config.get("test_size", 20),
            performance_threshold=custom_config.get("performance_threshold", 0.1),
            max_models=custom_config.get("max_models", 10),
        )

        self.model_dir = model_dir
        self.results_dir = results_dir

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize tracking
        self.model_history = []
        self.performance_history = []
        self.last_retrain_date = None
        self.current_model = None
        self.current_model_path = None

        # Load existing history
        self._load_history()

        logger.info("Rolling Retraining Agent initialized successfully")

    def _load_history(self):
        """Load existing retraining history."""
        try:
            # Load model history
            history_path = os.path.join(self.results_dir, "model_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    self.model_history = json.load(f)

            # Load performance history
            perf_path = os.path.join(self.results_dir, "performance_history.json")
            if os.path.exists(perf_path):
                with open(perf_path, "r") as f:
                    self.performance_history = json.load(f)

            # Set last retrain date
            if self.model_history:
                self.last_retrain_date = datetime.fromisoformat(
                    self.model_history[-1]["retrain_date"]
                )

            logger.info(f"Loaded {len(self.model_history)} model versions")

        except Exception as e:
            logger.warning(f"Error loading history: {e}")

    def _save_history(self):
        """Save retraining history."""
        try:
            # Save model history
            history_path = os.path.join(self.results_dir, "model_history.json")
            with open(history_path, "w") as f:
                json.dump(self.model_history, f, indent=2)

            # Save performance history
            perf_path = os.path.join(self.results_dir, "performance_history.json")
            with open(perf_path, "w") as f:
                json.dump(self.performance_history, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def prepare_data(
        self, data: pd.DataFrame, target_col: str = "returns"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        try:
            # Create features
            features = self._create_features(data)

            # Create target
            if target_col not in data.columns:
                # Create returns if not present
                data[target_col] = data["Close"].pct_change()

            target = data[target_col].dropna()

            # Align features and target
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]

            return features, target

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame(), pd.Series()

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for the model."""
        try:
            features = pd.DataFrame(index=data.index)

            # Price-based features with safe division
            features["returns"] = data["Close"].pct_change()

            close_shifted_1 = data["Close"].shift(1)
            features["log_returns"] = np.where(
                close_shifted_1 > 1e-10,
                np.log(data["Close"] / close_shifted_1),
                0.0
            )

            close_shifted_5 = data["Close"].shift(5)
            features["price_momentum"] = np.where(
                close_shifted_5 > 1e-10,
                data["Close"] / close_shifted_5 - 1,
                0.0
            )
            features["price_acceleration"] = features["price_momentum"].diff()

            # Volatility features
            features["volatility_5d"] = features["returns"].rolling(5).std()
            features["volatility_20d"] = features["returns"].rolling(20).std()
            
            # Volatility ratio with safe division
            features["volatility_ratio"] = np.where(
                features["volatility_20d"] > 1e-10,
                features["volatility_5d"] / features["volatility_20d"],
                1.0
            )

            # Moving averages
            features["sma_5"] = data["Close"].rolling(5).mean()
            features["sma_20"] = data["Close"].rolling(20).mean()
            features["sma_50"] = data["Close"].rolling(50).mean()
            
            # Moving average ratios with safe division
            features["ma_ratio_5_20"] = np.where(
                features["sma_20"] > 1e-10,
                features["sma_5"] / features["sma_20"],
                1.0
            )
            features["ma_ratio_20_50"] = np.where(
                features["sma_50"] > 1e-10,
                features["sma_20"] / features["sma_50"],
                1.0
            )

            # Volume features
            if "Volume" in data.columns:
                features["volume_ma"] = data["Volume"].rolling(20).mean()
                features["volume_ratio"] = data["Volume"] / features["volume_ma"]
                features["volume_trend"] = features["volume_ratio"].rolling(5).mean()
            else:
                features["volume_ma"] = 1.0
                features["volume_ratio"] = 1.0
                features["volume_trend"] = 1.0

            # Technical indicators
            features["rsi"] = self._calculate_rsi(data["Close"])
            features["macd"] = self._calculate_macd(data["Close"])
            features["bollinger_position"] = self._calculate_bollinger_position(
                data["Close"]
            )

            # Time features
            features["day_of_week"] = data.index.dayofweek
            features["month"] = data.index.month
            features["quarter"] = data.index.quarter

            # Drop NaN values
            features = features.dropna()

            return features

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator using safe division."""
        try:
            return safe_rsi(prices, period=period)
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index, data=0)

    def _calculate_bollinger_position(
        self, prices: pd.Series, period: int = 20
    ) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        try:
            from trading.utils.safe_math import safe_divide
            
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Use safe_divide for Bollinger position: (price - lower) / (upper - lower)
            band_range = upper_band - lower_band
            position = safe_divide(prices - lower_band, band_range, default=0.5)
            
            # Clamp to [0, 1] range
            position = position.clip(0.0, 1.0)
            
            return position
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating Bollinger position: {e}")
            return pd.Series(index=prices.index, data=0.5)

    def walk_forward_validation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_factory: Callable,
        n_splits: int = 5,
    ) -> List[WalkForwardResult]:
        """Perform walk-forward validation."""
        try:
            results = []

            # Create time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for train_idx, test_idx in tscv.split(features):
                # Split data
                train_features = features.iloc[train_idx]
                train_target = target.iloc[train_idx]
                test_features = features.iloc[test_idx]
                test_target = target.iloc[test_idx]

                # Train model
                model = model_factory()
                model.fit(train_features, train_target)

                # Make predictions
                train_pred = model.predict(train_features)
                test_pred = model.predict(test_features)

                # Calculate metrics
                train_score = r2_score(train_target, train_pred)
                test_score = r2_score(test_target, test_pred)

                # Calculate additional metrics
                mse = mean_squared_error(test_target, test_pred)
                mae = mean_absolute_error(test_target, test_pred)

                # Get feature importance if available
                feature_importance = {}
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(
                        zip(features.columns, model.feature_importances_)
                    )

                # Create result
                result = WalkForwardResult(
                    train_start=train_features.index[0],
                    train_end=train_features.index[-1],
                    test_start=test_features.index[0],
                    test_end=test_features.index[-1],
                    train_score=train_score,
                    test_score=test_score,
                    model_performance={"mse": mse, "mae": mae, "r2": test_score},
                    feature_importance=feature_importance,
                    predictions=test_pred.tolist(),
                    actuals=test_target.tolist(),
                )

                results.append(result)

            logger.info(f"Walk-forward validation completed with {len(results)} folds")
            return results

        except Exception as e:
            logger.error(f"Error in walk-forward validation: {e}")
            return []

    def should_retrain(self, current_performance: float) -> bool:
        """Determine if retraining is needed."""
        try:
            # Check if enough time has passed
            if self.last_retrain_date:
                days_since_retrain = (datetime.now() - self.last_retrain_date).days
                if days_since_retrain < self.retraining_config.retrain_frequency:
                    return False

            # Check performance degradation
            if self.performance_history:
                recent_performance = self.performance_history[-1]["test_score"]
                degradation = (
                    recent_performance - current_performance
                ) / recent_performance

                if degradation > self.retraining_config.performance_threshold:
                    logger.info(f"Performance degradation detected: {degradation:.2%}")
                    return True

            # Default retraining schedule
            return True

        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return True

    def retrain_model(
        self, data: pd.DataFrame, model_factory: Callable, target_col: str = "returns"
    ) -> pd.Series:
        """Retrain the model with new data."""
        try:
            logger.info("Starting model retraining...")

            # Prepare data
            features, target = self.prepare_data(data, target_col)

            if features.empty or target.empty:
                raise ValueError("No valid data for training")

            # Ensure retraining only happens if enough new data
            new_data = features.tail(self.retraining_config.min_train_size)
            if len(new_data) < self.retraining_config.min_train_size:
                logger.info(
                    f"Insufficient new data for retraining. Need {self.retraining_config.min_train_size}, have {len(new_data)}"
                )
                return False

            # Perform walk-forward validation
            wf_results = self.walk_forward_validation(features, target, model_factory)

            if not wf_results:
                raise ValueError("Walk-forward validation failed")

            # Calculate average performance
            avg_test_score = np.mean([r.test_score for r in wf_results])
            avg_train_score = np.mean([r.train_score for r in wf_results])

            # Train final model on all data
            final_model = model_factory()
            final_model.fit(features, target)

            # Save model
            model_version = len(self.model_history) + 1
            model_path = os.path.join(self.model_dir, f"model_v{model_version}.pkl")
            joblib.dump(final_model, model_path)

            # Update tracking
            self.current_model = final_model
            self.current_model_path = model_path
            self.last_retrain_date = datetime.now()

            # Record model history
            model_record = {
                "version": model_version,
                "retrain_date": self.last_retrain_date.isoformat(),
                "model_path": model_path,
                "avg_train_score": avg_train_score,
                "avg_test_score": avg_test_score,
                "n_features": len(features.columns),
                "n_samples": len(features),
            }

            self.model_history.append(model_record)

            # Record performance history
            for i, result in enumerate(wf_results):
                perf_record = {
                    "model_version": model_version,
                    "fold": i + 1,
                    "train_score": result.train_score,
                    "test_score": result.test_score,
                    "mse": result.model_performance["mse"],
                    "mae": result.model_performance["mae"],
                    "train_start": result.train_start.isoformat(),
                    "train_end": result.train_end.isoformat(),
                    "test_start": result.test_start.isoformat(),
                    "test_end": result.test_end.isoformat(),
                }
                self.performance_history.append(perf_record)

            # Clean up old models
            self._cleanup_old_models()

            # Save history
            self._save_history()

            logger.info(
                f"Model retraining completed. Version {model_version}, Test Score: {avg_test_score:.4f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return False

    def _cleanup_old_models(self):
        """Remove old model files."""
        try:
            if len(self.model_history) > self.retraining_config.max_models:
                # Remove oldest models
                models_to_remove = self.model_history[
                    : -self.retraining_config.max_models
                ]

                for model_record in models_to_remove:
                    model_path = model_record["model_path"]
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Removed old model: {model_path}")

                # Update history
                self.model_history = self.model_history[
                    -self.retraining_config.max_models :
                ]

        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")

    def load_latest_model(self) -> Optional[Any]:
        """Load the latest trained model."""
        try:
            if not self.model_history:
                logger.warning("No trained models found")
                return None

            latest_model_path = self.model_history[-1]["model_path"]

            if os.path.exists(latest_model_path):
                self.current_model = joblib.load(latest_model_path)
                self.current_model_path = latest_model_path
                logger.info(f"Loaded latest model: {latest_model_path}")
                return self.current_model
            else:
                logger.error(f"Model file not found: {latest_model_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the current model."""
        try:
            if self.current_model is None:
                self.load_latest_model()

            if self.current_model is None:
                raise ValueError("No model available for prediction")

            return self.current_model.predict(features)

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance over time."""
        try:
            if not self.performance_history:
                return {"message": "No performance history available"}

            # Calculate performance trends
            recent_performance = [
                p["test_score"] for p in self.performance_history[-10:]
            ]
            avg_recent = np.mean(recent_performance)
            std_recent = np.std(recent_performance)

            # Performance trend
            if len(recent_performance) >= 2:
                trend = (recent_performance[-1] - recent_performance[0]) / len(
                    recent_performance
                )
            else:
                trend = 0

            # Best and worst performance
            all_scores = [p["test_score"] for p in self.performance_history]
            best_score = max(all_scores)
            worst_score = min(all_scores)

            return {
                "total_models": len(self.model_history),
                "total_folds": len(self.performance_history),
                "avg_recent_performance": avg_recent,
                "recent_volatility": std_recent,
                "performance_trend": trend,
                "best_score": best_score,
                "worst_score": worst_score,
                "latest_model_version": (
                    self.model_history[-1]["version"] if self.model_history else None
                ),
                "last_retrain_date": (
                    self.last_retrain_date.isoformat()
                    if self.last_retrain_date
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the current model."""
        try:
            if self.current_model is None:
                self.load_latest_model()

            if self.current_model is None or not hasattr(
                self.current_model, "feature_importances_"
            ):
                return {}

            # Get feature names from the latest walk-forward result
            if self.performance_history:
                self.performance_history[-1]
                # This would need to be stored separately in practice
                return {}

            return dict(
                zip(
                    self.current_model.feature_names_in_,
                    self.current_model.feature_importances_,
                )
            )

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def export_results(self, filepath: str = "logs/retraining_export.json"):
        """Export retraining results to file."""
        try:
            export_data = {
                "model_history": self.model_history,
                "performance_history": self.performance_history,
                "config": {
                    "retrain_frequency": self.retraining_config.retrain_frequency,
                    "lookback_window": self.retraining_config.lookback_window,
                    "min_train_size": self.retraining_config.min_train_size,
                    "test_size": self.retraining_config.test_size,
                    "performance_threshold": self.retraining_config.performance_threshold,
                    "max_models": self.retraining_config.max_models,
                },
                "summary": self.get_performance_summary(),
                "export_date": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Retraining results exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting results: {e}")

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the rolling retraining logic.
        Args:
            **kwargs: data, model_factory, target_col, action, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get("action", "should_retrain")

            if action == "should_retrain":
                current_performance = kwargs.get("current_performance")

                if current_performance is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: current_performance",
                    )

                should_retrain = self.should_retrain(current_performance)
                return AgentResult(
                    success=True,
                    data={
                        "should_retrain": should_retrain,
                        "last_retrain_date": (
                            self.last_retrain_date.isoformat()
                            if self.last_retrain_date
                            else None
                        ),
                    },
                )

            elif action == "retrain_model":
                data = kwargs.get("data")
                model_factory = kwargs.get("model_factory")
                target_col = kwargs.get("target_col", "returns")

                if data is None or model_factory is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: data, model_factory",
                    )

                performance = self.retrain_model(data, model_factory, target_col)
                return AgentResult(
                    success=True,
                    data={
                        "performance_metrics": (
                            performance.to_dict()
                            if hasattr(performance, "to_dict")
                            else dict(performance)
                        ),
                        "model_history_count": len(self.model_history),
                    },
                )

            elif action == "walk_forward_validation":
                features = kwargs.get("features")
                target = kwargs.get("target")
                model_factory = kwargs.get("model_factory")
                n_splits = kwargs.get("n_splits", 5)

                if features is None or target is None or model_factory is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: features, target, model_factory",
                    )

                results = self.walk_forward_validation(
                    features, target, model_factory, n_splits
                )
                return AgentResult(
                    success=True,
                    data={
                        "walk_forward_results": [result.__dict__ for result in results],
                        "results_count": len(results),
                    },
                )

            elif action == "get_performance_summary":
                summary = self.get_performance_summary()
                return AgentResult(success=True, data={"performance_summary": summary})

            elif action == "get_feature_importance":
                importance = self.get_feature_importance()
                return AgentResult(
                    success=True, data={"feature_importance": importance}
                )

            elif action == "predict":
                features = kwargs.get("features")

                if features is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: features",
                    )

                predictions = self.predict(features)
                return AgentResult(
                    success=True,
                    data={
                        "predictions": (
                            predictions.tolist()
                            if hasattr(predictions, "tolist")
                            else list(predictions)
                        ),
                        "prediction_count": len(predictions),
                    },
                )

            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

        except Exception as e:
            return self.handle_error(e)
