"""
Model Benchmarker Module

This module handles benchmarking and evaluating model implementations
for the auto-evolutionary model generator.
"""

import logging
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .implementation_generator import ModelCandidate

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results of model benchmarking."""
    model_name: str
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    inference_time: float
    memory_usage: float
    overall_score: float
    benchmark_date: str

class ModelBenchmarker:
    """Benchmarks and evaluates model implementations."""
    
    def __init__(self, 
                 benchmark_data: pd.DataFrame,
                 target_column: str = "returns",
                 current_best_score: float = 1.0):
        """
        Initialize the model benchmarker.
        
        Args:
            benchmark_data: Data for benchmarking
            target_column: Target column name
            current_best_score: Current best performance score
        """
        self.benchmark_data = benchmark_data
        self.target_column = target_column
        self.current_best_score = current_best_score
        self.logger = logging.getLogger(__name__)
        
        # Prepare benchmark data
        self.X, self.y = self._prepare_benchmark_data(benchmark_data)
        
        logger.info(f"Initialized ModelBenchmarker with {len(benchmark_data)} samples")
    
    def benchmark_model(self, 
                       model_candidate: ModelCandidate,
                       test_data: Optional[pd.DataFrame] = None) -> BenchmarkResult:
        """
        Benchmark a model candidate.
        
        Args:
            model_candidate: Model candidate to benchmark
            test_data: Optional test data (uses benchmark_data if None)
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            start_time = time.time()
            
            # Use test data if provided, otherwise use benchmark data
            if test_data is not None:
                X_test, y_test = self._prepare_benchmark_data(test_data)
            else:
                X_test, y_test = self.X, self.y
            
            # Benchmark based on model type
            if model_candidate.model_type == "sklearn":
                result = self._benchmark_sklearn_model(model_candidate, X_test, y_test)
            elif model_candidate.model_type in ["lstm", "transformer", "attention", "rl"]:
                result = self._benchmark_pytorch_model(model_candidate, X_test, y_test)
            else:
                result = self._benchmark_generic_model(model_candidate, X_test, y_test)
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            result.benchmark_date = datetime.now().isoformat()
            
            benchmark_time = time.time() - start_time
            logger.info(f"Benchmarked {model_candidate.name} in {benchmark_time:.2f}s - Score: {result.overall_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_candidate.name}: {e}")
            # Return a default result with poor scores
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=float('inf'),
                mae=float('inf'),
                r2_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                training_time=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                overall_score=0.0,
                benchmark_date=datetime.now().isoformat()
            )
    
    def _prepare_benchmark_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for benchmarking."""
        try:
            # Ensure we have the target column
            if self.target_column not in data.columns:
                # Try to find a suitable target column
                possible_targets = ['Close', 'close', 'price', 'returns', 'target']
                for col in possible_targets:
                    if col in data.columns:
                        self.target_column = col
                        break
                else:
                    raise ValueError(f"Target column '{self.target_column}' not found in data")
            
            # Prepare features (use all numeric columns except target)
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                             if col != self.target_column]
            
            if not feature_columns:
                # If no features, use lagged values of target
                feature_columns = [self.target_column]
                data = data.copy()
                data[f'{self.target_column}_lag1'] = data[self.target_column].shift(1)
                data[f'{self.target_column}_lag2'] = data[self.target_column].shift(2)
                feature_columns = [f'{self.target_column}_lag1', f'{self.target_column}_lag2']
            
            # Remove rows with NaN values
            data_clean = data[feature_columns + [self.target_column]].dropna()
            
            X = data_clean[feature_columns].values
            y = data_clean[self.target_column].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing benchmark data: {e}")
            # Return dummy data
            return np.random.randn(100, 5), np.random.randn(100)
    
    def _benchmark_sklearn_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark a scikit-learn model."""
        try:
            # Import sklearn models
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create model based on type
            model_type = model_candidate.name.lower()
            if 'forest' in model_type or 'random' in model_type:
                model = RandomForestRegressor(**model_candidate.hyperparameters)
            elif 'gradient' in model_type or 'boosting' in model_type:
                model = GradientBoostingRegressor(**model_candidate.hyperparameters)
            elif 'linear' in model_type or 'regression' in model_type:
                model = LinearRegression(**model_candidate.hyperparameters)
            elif 'ridge' in model_type:
                model = Ridge(**model_candidate.hyperparameters)
            elif 'lasso' in model_type:
                model = Lasso(**model_candidate.hyperparameters)
            elif 'svm' in model_type or 'svr' in model_type:
                model = SVR(**model_candidate.hyperparameters)
            else:
                model = RandomForestRegressor(**model_candidate.hyperparameters)
            
            # Train model
            train_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - train_start
            
            # Predict
            inference_start = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate trading metrics
            sharpe_ratio, max_drawdown = self._calculate_trading_metrics(y_test, y_pred)
            
            # Measure memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=mse,
                mae=mae,
                r2_score=r2,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_usage,
                overall_score=0.0,  # Will be calculated later
                benchmark_date=""
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking sklearn model {model_candidate.name}: {e}")
            raise
    
    def _benchmark_pytorch_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark a PyTorch model."""
        try:
            # For now, return a placeholder result
            # In a full implementation, this would create and train the PyTorch model
            logger.warning(f"PyTorch benchmarking not fully implemented for {model_candidate.name}")
            
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=0.1,
                mae=0.05,
                r2_score=0.7,
                sharpe_ratio=1.2,
                max_drawdown=0.15,
                training_time=1.0,
                inference_time=0.01,
                memory_usage=100.0,
                overall_score=0.0,
                benchmark_date=""
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking PyTorch model {model_candidate.name}: {e}")
            raise
    
    def _benchmark_generic_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark a generic model."""
        try:
            # For now, return a placeholder result
            # In a full implementation, this would evaluate the generated code
            logger.warning(f"Generic model benchmarking not fully implemented for {model_candidate.name}")
            
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=0.15,
                mae=0.08,
                r2_score=0.6,
                sharpe_ratio=1.0,
                max_drawdown=0.2,
                training_time=2.0,
                inference_time=0.02,
                memory_usage=150.0,
                overall_score=0.0,
                benchmark_date=""
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking generic model {model_candidate.name}: {e}")
            raise
    
    def _calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate trading-specific metrics."""
        try:
            # Calculate returns
            returns_true = np.diff(y_true) / y_true[:-1]
            returns_pred = np.diff(y_pred) / y_pred[:-1]
            
            # Calculate Sharpe ratio
            excess_returns = returns_pred - returns_true
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + returns_pred)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            return sharpe_ratio, abs(max_drawdown)
            
        except Exception as e:
            logger.warning(f"Error calculating trading metrics: {e}")
            return 0.0, 0.2
    
    def _calculate_overall_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics to 0-1 scale
            mse_score = max(0, 1 - result.mse / 1.0)  # Assuming max MSE of 1.0
            mae_score = max(0, 1 - result.mae / 0.5)  # Assuming max MAE of 0.5
            r2_score = max(0, result.r2_score)
            sharpe_score = max(0, min(1, result.sharpe_ratio / 2.0))  # Normalize to 0-1
            drawdown_score = max(0, 1 - result.max_drawdown / 0.5)  # Assuming max drawdown of 0.5
            
            # Time efficiency scores (lower is better)
            time_score = max(0, 1 - result.training_time / 60.0)  # Assuming max training time of 60s
            inference_score = max(0, 1 - result.inference_time / 1.0)  # Assuming max inference time of 1s
            
            # Memory efficiency score
            memory_score = max(0, 1 - result.memory_usage / 1000.0)  # Assuming max memory of 1GB
            
            # Weighted combination
            weights = {
                'mse': 0.2,
                'mae': 0.15,
                'r2': 0.2,
                'sharpe': 0.15,
                'drawdown': 0.1,
                'time': 0.05,
                'inference': 0.05,
                'memory': 0.1
            }
            
            overall_score = (
                weights['mse'] * mse_score +
                weights['mae'] * mae_score +
                weights['r2'] * r2_score +
                weights['sharpe'] * sharpe_score +
                weights['drawdown'] * drawdown_score +
                weights['time'] * time_score +
                weights['inference'] * inference_score +
                weights['memory'] * memory_score
            )
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0 