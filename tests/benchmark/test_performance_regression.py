import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path
from trading.data.preprocessing import DataPreprocessor, FeatureEngineering
from trading.models.tcn_model import TCNModel
from trading.models.lstm_model import LSTMModel

class TestPerformanceRegression:
    @pytest.fixture
    def benchmark_data_dir(self):
        """Create directory for storing benchmark data."""
        benchmark_dir = Path("benchmarks")
        benchmark_dir.mkdir(exist_ok=True)
        return benchmark_dir
    
    @pytest.fixture
    def performance_history_file(self, benchmark_data_dir):
        """Get path to performance history file."""
        return benchmark_data_dir / "performance_history.json"
    
    @pytest.fixture
    def large_sample_data(self):
        """Create large sample time series data for performance testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, len(dates)),
            'High': np.random.normal(102, 2, len(dates)),
            'Low': np.random.normal(98, 2, len(dates)),
            'Close': np.random.normal(100, 2, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1) + 1
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1) - 1
        
        return data
    
    @pytest.fixture
    def preprocessor(self):
        """Create data preprocessor with default settings."""
        return DataPreprocessor()
    
    @pytest.fixture
    def feature_engineering(self):
        """Create feature engineering with default settings."""
        return FeatureEngineering()
    
    @pytest.fixture
    def tcn_model(self):
        """Create TCN model with default settings."""
        return TCNModel(config={
            'input_size': 20,
            'output_size': 1,
            'num_channels': [64, 32, 16],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': [
                'Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                'Volume_MA_5', 'Volume_MA_10', 'Fourier_Sin_7', 'Fourier_Cos_7',
                'Close_Lag_1', 'Close_Return_Lag_1', 'Volume_Lag_1',
                'Volume_Return_Lag_1', 'HL_Range_Lag_1', 'ROC_5', 'Momentum_5',
                'Stoch_K', 'Stoch_D', 'Volume_Trend'
            ],
            'target_column': 'Close'
        })
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model with default settings."""
        return LSTMModel(config={
            'input_size': 20,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': [
                'Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                'Volume_MA_5', 'Volume_MA_10', 'Fourier_Sin_7', 'Fourier_Cos_7',
                'Close_Lag_1', 'Close_Return_Lag_1', 'Volume_Lag_1',
                'Volume_Return_Lag_1', 'HL_Range_Lag_1', 'ROC_5', 'Momentum_5',
                'Stoch_K', 'Stoch_D', 'Volume_Trend'
            ],
            'target_column': 'Close'
        })
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def load_performance_history(self, performance_history_file):
        """Load performance history from file."""
        if performance_history_file.exists():
            with open(performance_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_performance_history(self, performance_history_file, history):
        """Save performance history to file."""
        with open(performance_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def run_benchmark(self, func, *args, **kwargs):
        """Run a benchmark and return performance metrics."""
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        memory_used = self.get_memory_usage() - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'result': result
        }
    
    def test_preprocessing_regression(self, preprocessor, large_sample_data, performance_history_file):
        """Test for performance regression in preprocessing."""
        # Run benchmark
        benchmark_result = self.run_benchmark(
            preprocessor.preprocess_data,
            large_sample_data
        )
        
        # Load history
        history = self.load_performance_history(performance_history_file)
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': benchmark_result['execution_time'],
            'memory_used': benchmark_result['memory_used'],
            'rows_processed': len(large_sample_data)
        }
        
        # Update history
        if 'preprocessing' not in history:
            history['preprocessing'] = []
        history['preprocessing'].append(current_metrics)
        
        # Save updated history
        self.save_performance_history(performance_history_file, history)
        
        # Check for regression
        if len(history['preprocessing']) > 1:
            previous_metrics = history['preprocessing'][-2]
            
            # Calculate performance changes
            time_change = (current_metrics['execution_time'] - previous_metrics['execution_time']) / previous_metrics['execution_time']
            memory_change = (current_metrics['memory_used'] - previous_metrics['memory_used']) / previous_metrics['memory_used']
            
            print(f"\nPreprocessing Performance Changes:")
            print(f"Execution time change: {time_change:.2%}")
            print(f"Memory usage change: {memory_change:.2%}")
            
            # Assert no significant regression
            assert time_change < 0.1, f"Performance regression detected: {time_change:.2%} slower"
            assert memory_change < 0.2, f"Memory usage regression detected: {memory_change:.2%} more memory"
    
    def test_feature_engineering_regression(self, feature_engineering, preprocessor, large_sample_data, performance_history_file):
        """Test for performance regression in feature engineering."""
        # Preprocess data first
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        
        # Run benchmark
        benchmark_result = self.run_benchmark(
            feature_engineering.engineer_features,
            preprocessed_data
        )
        
        # Load history
        history = self.load_performance_history(performance_history_file)
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': benchmark_result['execution_time'],
            'memory_used': benchmark_result['memory_used'],
            'features_generated': len(benchmark_result['result'].columns)
        }
        
        # Update history
        if 'feature_engineering' not in history:
            history['feature_engineering'] = []
        history['feature_engineering'].append(current_metrics)
        
        # Save updated history
        self.save_performance_history(performance_history_file, history)
        
        # Check for regression
        if len(history['feature_engineering']) > 1:
            previous_metrics = history['feature_engineering'][-2]
            
            # Calculate performance changes
            time_change = (current_metrics['execution_time'] - previous_metrics['execution_time']) / previous_metrics['execution_time']
            memory_change = (current_metrics['memory_used'] - previous_metrics['memory_used']) / previous_metrics['memory_used']
            
            print(f"\nFeature Engineering Performance Changes:")
            print(f"Execution time change: {time_change:.2%}")
            print(f"Memory usage change: {memory_change:.2%}")
            
            # Assert no significant regression
            assert time_change < 0.1, f"Performance regression detected: {time_change:.2%} slower"
            assert memory_change < 0.2, f"Memory usage regression detected: {memory_change:.2%} more memory"
    
    def test_model_training_regression(self, tcn_model, lstm_model, feature_engineering, preprocessor, large_sample_data, performance_history_file):
        """Test for performance regression in model training."""
        # Prepare data
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Run benchmarks for both models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        
        tcn_benchmark = self.run_benchmark(
            tcn_model.fit,
            tcn_data['X'],
            tcn_data['y']
        )
        
        lstm_benchmark = self.run_benchmark(
            lstm_model.fit,
            lstm_data['X'],
            lstm_data['y']
        )
        
        # Load history
        history = self.load_performance_history(performance_history_file)
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'tcn': {
                'execution_time': tcn_benchmark['execution_time'],
                'memory_used': tcn_benchmark['memory_used'],
                'samples_trained': len(tcn_data['X'])
            },
            'lstm': {
                'execution_time': lstm_benchmark['execution_time'],
                'memory_used': lstm_benchmark['memory_used'],
                'samples_trained': len(lstm_data['X'])
            }
        }
        
        # Update history
        if 'model_training' not in history:
            history['model_training'] = []
        history['model_training'].append(current_metrics)
        
        # Save updated history
        self.save_performance_history(performance_history_file, history)
        
        # Check for regression
        if len(history['model_training']) > 1:
            previous_metrics = history['model_training'][-2]
            
            # Calculate performance changes for both models
            for model_name in ['tcn', 'lstm']:
                time_change = (current_metrics[model_name]['execution_time'] - 
                             previous_metrics[model_name]['execution_time']) / previous_metrics[model_name]['execution_time']
                memory_change = (current_metrics[model_name]['memory_used'] - 
                               previous_metrics[model_name]['memory_used']) / previous_metrics[model_name]['memory_used']
                
                print(f"\n{model_name.upper()} Model Training Performance Changes:")
                print(f"Execution time change: {time_change:.2%}")
                print(f"Memory usage change: {memory_change:.2%}")
                
                # Assert no significant regression
                assert time_change < 0.1, f"{model_name.upper()} performance regression detected: {time_change:.2%} slower"
                assert memory_change < 0.2, f"{model_name.upper()} memory usage regression detected: {memory_change:.2%} more memory"
    
    def test_prediction_regression(self, tcn_model, lstm_model, feature_engineering, preprocessor, large_sample_data, performance_history_file):
        """Test for performance regression in model prediction."""
        # Prepare data
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Prepare and train models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        tcn_model.fit(tcn_data['X'], tcn_data['y'])
        lstm_model.fit(lstm_data['X'], lstm_data['y'])
        
        # Run benchmarks for both models
        tcn_benchmark = self.run_benchmark(
            tcn_model.predict,
            tcn_data['X']
        )
        
        lstm_benchmark = self.run_benchmark(
            lstm_model.predict,
            lstm_data['X']
        )
        
        # Load history
        history = self.load_performance_history(performance_history_file)
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'tcn': {
                'execution_time': tcn_benchmark['execution_time'],
                'memory_used': tcn_benchmark['memory_used'],
                'samples_predicted': len(tcn_data['X'])
            },
            'lstm': {
                'execution_time': lstm_benchmark['execution_time'],
                'memory_used': lstm_benchmark['memory_used'],
                'samples_predicted': len(lstm_data['X'])
            }
        }
        
        # Update history
        if 'model_prediction' not in history:
            history['model_prediction'] = []
        history['model_prediction'].append(current_metrics)
        
        # Save updated history
        self.save_performance_history(performance_history_file, history)
        
        # Check for regression
        if len(history['model_prediction']) > 1:
            previous_metrics = history['model_prediction'][-2]
            
            # Calculate performance changes for both models
            for model_name in ['tcn', 'lstm']:
                time_change = (current_metrics[model_name]['execution_time'] - 
                             previous_metrics[model_name]['execution_time']) / previous_metrics[model_name]['execution_time']
                memory_change = (current_metrics[model_name]['memory_used'] - 
                               previous_metrics[model_name]['memory_used']) / previous_metrics[model_name]['memory_used']
                
                print(f"\n{model_name.upper()} Model Prediction Performance Changes:")
                print(f"Execution time change: {time_change:.2%}")
                print(f"Memory usage change: {memory_change:.2%}")
                
                # Assert no significant regression
                assert time_change < 0.1, f"{model_name.upper()} performance regression detected: {time_change:.2%} slower"
                assert memory_change < 0.2, f"{model_name.upper()} memory usage regression detected: {memory_change:.2%} more memory"
    
    def test_full_pipeline_regression(self, preprocessor, feature_engineering, tcn_model, lstm_model, large_sample_data, performance_history_file):
        """Test for performance regression in the full pipeline."""
        def run_pipeline():
            # Preprocess data
            preprocessed_data = preprocessor.preprocess_data(large_sample_data)
            # Engineer features
            features = feature_engineering.engineer_features(preprocessed_data)
            # Prepare data for models
            tcn_data = tcn_model._prepare_data(features)
            lstm_data = lstm_model._prepare_data(features)
            # Train models
            tcn_model.fit(tcn_data['X'], tcn_data['y'])
            lstm_model.fit(lstm_data['X'], lstm_data['y'])
            # Make predictions
            tcn_pred = tcn_model.predict(tcn_data['X'])
            lstm_pred = lstm_model.predict(lstm_data['X'])
            return {
                'tcn_predictions': tcn_pred,
                'lstm_predictions': lstm_pred
            }
        
        # Run benchmark
        benchmark_result = self.run_benchmark(run_pipeline)
        
        # Load history
        history = self.load_performance_history(performance_history_file)
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': benchmark_result['execution_time'],
            'memory_used': benchmark_result['memory_used'],
            'rows_processed': len(large_sample_data)
        }
        
        # Update history
        if 'full_pipeline' not in history:
            history['full_pipeline'] = []
        history['full_pipeline'].append(current_metrics)
        
        # Save updated history
        self.save_performance_history(performance_history_file, history)
        
        # Check for regression
        if len(history['full_pipeline']) > 1:
            previous_metrics = history['full_pipeline'][-2]
            
            # Calculate performance changes
            time_change = (current_metrics['execution_time'] - previous_metrics['execution_time']) / previous_metrics['execution_time']
            memory_change = (current_metrics['memory_used'] - previous_metrics['memory_used']) / previous_metrics['memory_used']
            
            print(f"\nFull Pipeline Performance Changes:")
            print(f"Execution time change: {time_change:.2%}")
            print(f"Memory usage change: {memory_change:.2%}")
            
            # Assert no significant regression
            assert time_change < 0.1, f"Performance regression detected: {time_change:.2%} slower"
            assert memory_change < 0.2, f"Memory usage regression detected: {memory_change:.2%} more memory"
    
    def test_performance_trend_analysis(self, performance_history_file):
        """Analyze performance trends over time."""
        history = self.load_performance_history(performance_history_file)
        
        if not history:
            pytest.skip("No performance history available")
        
        print("\nPerformance Trend Analysis:")
        
        for component, metrics in history.items():
            if len(metrics) < 2:
                continue
            
            # Calculate trends
            execution_times = [m['execution_time'] for m in metrics]
            memory_usage = [m['memory_used'] for m in metrics]
            
            # Calculate average changes
            avg_time_change = sum(
                (execution_times[i] - execution_times[i-1]) / execution_times[i-1]
                for i in range(1, len(execution_times))
            ) / (len(execution_times) - 1)
            
            avg_memory_change = sum(
                (memory_usage[i] - memory_usage[i-1]) / memory_usage[i-1]
                for i in range(1, len(memory_usage))
            ) / (len(memory_usage) - 1)
            
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"Average execution time change: {avg_time_change:.2%}")
            print(f"Average memory usage change: {avg_memory_change:.2%}")
            
            # Check for consistent degradation
            if avg_time_change > 0.05:  # 5% threshold
                print(f"Warning: Consistent performance degradation detected in {component}")
            if avg_memory_change > 0.1:  # 10% threshold
                print(f"Warning: Consistent memory usage increase detected in {component}") 