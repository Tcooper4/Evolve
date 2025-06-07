import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from datetime import datetime, timedelta
from trading.data.preprocessing import DataPreprocessor, FeatureEngineering
from trading.models.tcn_model import TCNModel
from trading.models.lstm_model import LSTMModel

class TestPerformance:
    @pytest.fixture
    def large_sample_data(self):
        """Create large sample time series data for performance testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')  # 4 years of hourly data
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, len(dates)),
            'High': np.random.normal(102, 2, len(dates)),
            'Low': np.random.normal(98, 2, len(dates)),
            'Close': np.random.normal(100, 2, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
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
    
    def benchmark_preprocessing(self, preprocessor, large_sample_data):
        """Benchmark data preprocessing performance."""
        # Measure memory before
        memory_before = self.get_memory_usage()
        
        # Measure execution time
        start_time = time.time()
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        execution_time = time.time() - start_time
        
        # Measure memory after
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'rows_processed': len(large_sample_data)
        }
    
    def benchmark_feature_engineering(self, feature_engineering, preprocessed_data):
        """Benchmark feature engineering performance."""
        memory_before = self.get_memory_usage()
        
        start_time = time.time()
        features = feature_engineering.engineer_features(preprocessed_data)
        execution_time = time.time() - start_time
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'features_generated': len(features.columns)
        }
    
    def benchmark_model_training(self, model, model_data):
        """Benchmark model training performance."""
        memory_before = self.get_memory_usage()
        
        start_time = time.time()
        model.fit(model_data['X'], model_data['y'])
        execution_time = time.time() - start_time
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'samples_trained': len(model_data['X'])
        }
    
    def benchmark_model_prediction(self, model, model_data):
        """Benchmark model prediction performance."""
        memory_before = self.get_memory_usage()
        
        start_time = time.time()
        predictions = model.predict(model_data['X'])
        execution_time = time.time() - start_time
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'samples_predicted': len(predictions)
        }
    
    def test_preprocessing_performance(self, preprocessor, large_sample_data):
        """Test preprocessing performance with large dataset."""
        results = self.benchmark_preprocessing(preprocessor, large_sample_data)
        
        # Print performance metrics
        print(f"\nPreprocessing Performance:")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Memory used: {results['memory_used']:.2f} MB")
        print(f"Rows processed: {results['rows_processed']}")
        print(f"Processing speed: {results['rows_processed']/results['execution_time']:.2f} rows/second")
        
        # Assert performance requirements
        assert results['execution_time'] < 60  # Should process within 60 seconds
        assert results['memory_used'] < 1000  # Should use less than 1GB memory
    
    def test_feature_engineering_performance(self, feature_engineering, preprocessor, large_sample_data):
        """Test feature engineering performance with large dataset."""
        # First preprocess the data
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        
        # Then benchmark feature engineering
        results = self.benchmark_feature_engineering(feature_engineering, preprocessed_data)
        
        print(f"\nFeature Engineering Performance:")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Memory used: {results['memory_used']:.2f} MB")
        print(f"Features generated: {results['features_generated']}")
        print(f"Processing speed: {results['features_generated']/results['execution_time']:.2f} features/second")
        
        assert results['execution_time'] < 120  # Should process within 120 seconds
        assert results['memory_used'] < 2000  # Should use less than 2GB memory
    
    def test_model_training_performance(self, tcn_model, lstm_model, feature_engineering, preprocessor, large_sample_data):
        """Test model training performance."""
        # Prepare data
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Benchmark TCN model
        tcn_data = tcn_model._prepare_data(features)
        tcn_results = self.benchmark_model_training(tcn_model, tcn_data)
        
        print(f"\nTCN Model Training Performance:")
        print(f"Execution time: {tcn_results['execution_time']:.2f} seconds")
        print(f"Memory used: {tcn_results['memory_used']:.2f} MB")
        print(f"Samples trained: {tcn_results['samples_trained']}")
        print(f"Training speed: {tcn_results['samples_trained']/tcn_results['execution_time']:.2f} samples/second")
        
        # Benchmark LSTM model
        lstm_data = lstm_model._prepare_data(features)
        lstm_results = self.benchmark_model_training(lstm_model, lstm_data)
        
        print(f"\nLSTM Model Training Performance:")
        print(f"Execution time: {lstm_results['execution_time']:.2f} seconds")
        print(f"Memory used: {lstm_results['memory_used']:.2f} MB")
        print(f"Samples trained: {lstm_results['samples_trained']}")
        print(f"Training speed: {lstm_results['samples_trained']/lstm_results['execution_time']:.2f} samples/second")
        
        # Assert performance requirements
        assert tcn_results['execution_time'] < 300  # Should train within 5 minutes
        assert lstm_results['execution_time'] < 300
        assert tcn_results['memory_used'] < 4000  # Should use less than 4GB memory
        assert lstm_results['memory_used'] < 4000
    
    def test_model_prediction_performance(self, tcn_model, lstm_model, feature_engineering, preprocessor, large_sample_data):
        """Test model prediction performance."""
        # Prepare data
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        # Prepare and train models
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        tcn_model.fit(tcn_data['X'], tcn_data['y'])
        lstm_model.fit(lstm_data['X'], lstm_data['y'])
        
        # Benchmark TCN model prediction
        tcn_results = self.benchmark_model_prediction(tcn_model, tcn_data)
        
        print(f"\nTCN Model Prediction Performance:")
        print(f"Execution time: {tcn_results['execution_time']:.2f} seconds")
        print(f"Memory used: {tcn_results['memory_used']:.2f} MB")
        print(f"Samples predicted: {tcn_results['samples_predicted']}")
        print(f"Prediction speed: {tcn_results['samples_predicted']/tcn_results['execution_time']:.2f} samples/second")
        
        # Benchmark LSTM model prediction
        lstm_results = self.benchmark_model_prediction(lstm_model, lstm_data)
        
        print(f"\nLSTM Model Prediction Performance:")
        print(f"Execution time: {lstm_results['execution_time']:.2f} seconds")
        print(f"Memory used: {lstm_results['memory_used']:.2f} MB")
        print(f"Samples predicted: {lstm_results['samples_predicted']}")
        print(f"Prediction speed: {lstm_results['samples_predicted']/lstm_results['execution_time']:.2f} samples/second")
        
        # Assert performance requirements
        assert tcn_results['execution_time'] < 10  # Should predict within 10 seconds
        assert lstm_results['execution_time'] < 10
        assert tcn_results['memory_used'] < 1000  # Should use less than 1GB memory
        assert lstm_results['memory_used'] < 1000
    
    def test_full_pipeline_performance(self, preprocessor, feature_engineering, tcn_model, lstm_model, large_sample_data):
        """Test full pipeline performance from preprocessing to prediction."""
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        # Run full pipeline
        preprocessed_data = preprocessor.preprocess_data(large_sample_data)
        features = feature_engineering.engineer_features(preprocessed_data)
        
        tcn_data = tcn_model._prepare_data(features)
        lstm_data = lstm_model._prepare_data(features)
        
        tcn_model.fit(tcn_data['X'], tcn_data['y'])
        lstm_model.fit(lstm_data['X'], lstm_data['y'])
        
        tcn_pred = tcn_model.predict(tcn_data['X'])
        lstm_pred = lstm_model.predict(lstm_data['X'])
        
        execution_time = time.time() - start_time
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        print(f"\nFull Pipeline Performance:")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Total memory used: {memory_used:.2f} MB")
        print(f"Total samples processed: {len(large_sample_data)}")
        print(f"Overall processing speed: {len(large_sample_data)/execution_time:.2f} samples/second")
        
        # Assert performance requirements
        assert execution_time < 600  # Should complete within 10 minutes
        assert memory_used < 8000  # Should use less than 8GB memory
    
    def test_scalability(self, preprocessor, feature_engineering, tcn_model, lstm_model):
        """Test performance scalability with different data sizes."""
        sizes = [1000, 5000, 10000, 50000]  # Different data sizes to test
        results = []
        
        for size in sizes:
            # Generate data of specific size
            dates = pd.date_range(start='2023-01-01', periods=size, freq='H')
            data = pd.DataFrame({
                'Open': np.random.normal(100, 2, size),
                'High': np.random.normal(102, 2, size),
                'Low': np.random.normal(98, 2, size),
                'Close': np.random.normal(100, 2, size),
                'Volume': np.random.normal(1000000, 200000, size)
            }, index=dates)
            
            # Run pipeline and measure performance
            start_time = time.time()
            memory_before = self.get_memory_usage()
            
            preprocessed_data = preprocessor.preprocess_data(data)
            features = feature_engineering.engineer_features(preprocessed_data)
            
            tcn_data = tcn_model._prepare_data(features)
            lstm_data = lstm_model._prepare_data(features)
            
            tcn_model.fit(tcn_data['X'], tcn_data['y'])
            lstm_model.fit(lstm_data['X'], lstm_data['y'])
            
            execution_time = time.time() - start_time
            memory_used = self.get_memory_usage() - memory_before
            
            results.append({
                'size': size,
                'execution_time': execution_time,
                'memory_used': memory_used
            })
        
        # Print scalability results
        print("\nScalability Test Results:")
        for result in results:
            print(f"\nData size: {result['size']} rows")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            print(f"Memory used: {result['memory_used']:.2f} MB")
            print(f"Processing speed: {result['size']/result['execution_time']:.2f} rows/second")
        
        # Assert scalability requirements
        for i in range(1, len(results)):
            # Check if time complexity is better than O(n²)
            time_ratio = results[i]['execution_time'] / results[i-1]['execution_time']
            size_ratio = results[i]['size'] / results[i-1]['size']
            assert time_ratio < size_ratio * 1.5  # Allow some overhead
            
            # Check if memory usage is linear
            memory_ratio = results[i]['memory_used'] / results[i-1]['memory_used']
            assert memory_ratio < size_ratio * 1.2  # Allow some overhead
    
    def test_batch_processing_performance(self, preprocessor, feature_engineering, tcn_model, lstm_model, large_sample_data):
        """Test performance with different batch sizes."""
        batch_sizes = [1000, 5000, 10000, 50000]
        results = []
        
        for batch_size in batch_sizes:
            # Process data in batches
            start_time = time.time()
            memory_before = self.get_memory_usage()
            
            # Split data into batches
            batches = [large_sample_data[i:i + batch_size] for i in range(0, len(large_sample_data), batch_size)]
            
            processed_batches = []
            for batch in batches:
                # Preprocess batch
                preprocessed_batch = preprocessor.preprocess_data(batch)
                # Engineer features
                features_batch = feature_engineering.engineer_features(preprocessed_batch)
                processed_batches.append(features_batch)
            
            # Combine processed batches
            processed_data = pd.concat(processed_batches)
            
            execution_time = time.time() - start_time
            memory_used = self.get_memory_usage() - memory_before
            
            results.append({
                'batch_size': batch_size,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'total_rows': len(processed_data)
            })
        
        # Print batch processing results
        print("\nBatch Processing Performance:")
        for result in results:
            print(f"\nBatch size: {result['batch_size']}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            print(f"Memory used: {result['memory_used']:.2f} MB")
            print(f"Processing speed: {result['total_rows']/result['execution_time']:.2f} rows/second")
        
        # Assert batch processing requirements
        for i in range(1, len(results)):
            # Check if larger batches are more efficient
            time_per_row = results[i]['execution_time'] / results[i]['total_rows']
            prev_time_per_row = results[i-1]['execution_time'] / results[i-1]['total_rows']
            assert time_per_row <= prev_time_per_row * 1.2  # Allow 20% overhead
    
    def test_memory_efficiency(self, preprocessor, feature_engineering, tcn_model, lstm_model, large_sample_data):
        """Test memory efficiency with different data sizes and cleanup."""
        sizes = [1000, 5000, 10000, 50000]
        memory_usage = []
        
        for size in sizes:
            # Generate data of specific size
            data = large_sample_data.iloc[:size].copy()
            
            # Track memory usage at each step
            steps = []
            
            # Preprocessing
            memory_before = self.get_memory_usage()
            preprocessed_data = preprocessor.preprocess_data(data)
            steps.append({
                'step': 'preprocessing',
                'memory': self.get_memory_usage() - memory_before
            })
            
            # Feature engineering
            memory_before = self.get_memory_usage()
            features = feature_engineering.engineer_features(preprocessed_data)
            steps.append({
                'step': 'feature_engineering',
                'memory': self.get_memory_usage() - memory_before
            })
            
            # Model preparation
            memory_before = self.get_memory_usage()
            tcn_data = tcn_model._prepare_data(features)
            lstm_data = lstm_model._prepare_data(features)
            steps.append({
                'step': 'model_preparation',
                'memory': self.get_memory_usage() - memory_before
            })
            
            # Model training
            memory_before = self.get_memory_usage()
            tcn_model.fit(tcn_data['X'], tcn_data['y'])
            lstm_model.fit(lstm_data['X'], lstm_data['y'])
            steps.append({
                'step': 'model_training',
                'memory': self.get_memory_usage() - memory_before
            })
            
            # Cleanup
            del preprocessed_data
            del features
            del tcn_data
            del lstm_data
            import gc
            gc.collect()
            
            steps.append({
                'step': 'after_cleanup',
                'memory': self.get_memory_usage()
            })
            
            memory_usage.append({
                'size': size,
                'steps': steps
            })
        
        # Print memory efficiency results
        print("\nMemory Efficiency Analysis:")
        for result in memory_usage:
            print(f"\nData size: {result['size']} rows")
            for step in result['steps']:
                print(f"{step['step']}: {step['memory']:.2f} MB")
        
        # Assert memory efficiency requirements
        for i in range(1, len(memory_usage)):
            # Check if memory usage scales linearly
            memory_ratio = memory_usage[i]['steps'][-1]['memory'] / memory_usage[i-1]['steps'][-1]['memory']
            size_ratio = memory_usage[i]['size'] / memory_usage[i-1]['size']
            assert memory_ratio <= size_ratio * 1.5  # Allow 50% overhead
    
    def test_concurrent_processing(self, preprocessor, feature_engineering, tcn_model, lstm_model, large_sample_data):
        """Test performance with concurrent processing of multiple models."""
        import concurrent.futures
        
        # Split data into chunks
        chunk_size = len(large_sample_data) // 4
        chunks = [large_sample_data[i:i + chunk_size] for i in range(0, len(large_sample_data), chunk_size)]
        
        def process_chunk(chunk):
            # Preprocess
            preprocessed = preprocessor.preprocess_data(chunk)
            # Engineer features
            features = feature_engineering.engineer_features(preprocessed)
            # Prepare data
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
                'lstm_predictions': lstm_pred,
                'rows_processed': len(chunk)
            }
        
        # Process chunks concurrently
        start_time = time.time()
        memory_before = self.get_memory_usage()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        execution_time = time.time() - start_time
        memory_used = self.get_memory_usage() - memory_before
        
        # Calculate total rows processed
        total_rows = sum(result['rows_processed'] for result in results)
        
        print("\nConcurrent Processing Performance:")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Total rows processed: {total_rows}")
        print(f"Processing speed: {total_rows/execution_time:.2f} rows/second")
        
        # Assert concurrent processing requirements
        assert execution_time < 300  # Should complete within 5 minutes
        assert memory_used < 8000  # Should use less than 8GB memory
        assert total_rows == len(large_sample_data)  # All data should be processed 