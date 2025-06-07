import pytest
import torch
import numpy as np
import pandas as pd
import time
import psutil
import os
import gc
from trading.models.lstm_model import LSTMForecaster

# Register timeout mark
def pytest_configure(config):
    config.addinivalue_line("markers", "timeout: mark test to timeout after specified seconds")

class TestLSTMPerformance:
    """Test suite for LSTM model performance with autonomous system considerations."""
    
    # Global limits
    MAX_MEMORY_MB = 1024  # 1GB limit for entire test suite
    MAX_BENCHMARK_TIME = 60  # 60 seconds per benchmark operation
    MAX_CLEANUP_INTERVAL = 10  # Cleanup every 10 operations
    
    # Performance thresholds
    MAX_FORWARD_PASS_TIME = 0.5  # Maximum time for forward pass in seconds
    MAX_TRAINING_TIME = 5.0  # Maximum time for training in seconds
    MAX_MEMORY_USAGE = 500  # Maximum memory usage in MB
    MAX_LOSS = 1.0  # Maximum acceptable loss value
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear memory before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        yield
        
        # Clear memory after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for LSTM model with autonomous system settings."""
        return {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'sequence_length': 10,
            'feature_columns': [
                'close', 'volume', 'returns', 'ma_5', 'ma_10'
            ],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True,
            # Autonomous system specific settings
            'auto_retrain': True,
            'retrain_threshold': 0.1,
            'max_retrain_attempts': 3,
            'performance_monitoring': True,
            'adaptive_batch_size': True,
            'min_batch_size': 32,
            'max_batch_size': 256,
            'batch_size_step': 32,
            # Resource limits
            'max_sequence_length': 100,
            'max_batch_size': 512,
            'max_epochs': 100
        }
    
    @pytest.fixture(scope="class")
    def large_sample_data(self):
        """Generate a large sample time series dataset for performance testing."""
        np.random.seed(42)
        n = 1000
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
        close = np.cumsum(np.random.randn(n)) + 100
        volume = np.abs(np.random.randn(n) * 10000 + 5000)
        returns = np.insert(np.diff(close), 0, 0)
        ma_5 = pd.Series(close).rolling(window=5).mean().values
        ma_10 = pd.Series(close).rolling(window=10).mean().values
        data = pd.DataFrame({
            'close': close,
            'volume': volume,
            'returns': returns,
            'ma_5': ma_5,
            'ma_10': ma_10,
        }, index=dates)
        data = data.dropna()
        return data
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self):
        """Check if memory usage exceeds the limit."""
        current_memory = self.get_memory_usage()
        if current_memory > self.MAX_MEMORY_MB:
            raise MemoryError(f"Memory usage exceeded {self.MAX_MEMORY_MB}MB")
    
    def benchmark_forward_pass(self, model, X, num_runs=50):
        """Benchmark forward pass performance with autonomous monitoring."""
        model.eval()
        times = []
        memory_usage = []
        predictions = []
        start_time = time.time()
        
        try:
            with torch.no_grad():
                for i in range(num_runs):
                    # Check time limit
                    if time.time() - start_time > self.MAX_BENCHMARK_TIME:
                        raise TimeoutError("Benchmark operation exceeded time limit")
                    
                    # Check memory limit
                    self.check_memory_limit()
                    
                    start_mem = self.get_memory_usage()
                    iter_start_time = time.time()
                    
                    try:
                        # Process in batches
                        batch_size = min(256, X.size(0))
                        batch_predictions = []
                        for j in range(0, X.size(0), batch_size):
                            batch_X = X[j:j+batch_size]
                            pred = model(batch_X)
                            batch_predictions.append(pred.detach().cpu())
                        pred = torch.cat(batch_predictions, dim=0)
                        predictions.append(pred)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise MemoryError("GPU out of memory")
                        raise e
                    
                    end_time = time.time()
                    end_mem = self.get_memory_usage()
                    
                    times.append(end_time - iter_start_time)
                    memory_usage.append(max(0, end_mem - start_mem))  # Ensure non-negative memory usage
                    
                    # Periodic cleanup
                    if (i + 1) % self.MAX_CLEANUP_INTERVAL == 0:
                        if len(predictions) > 10:
                            predictions = predictions[-10:]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                # Calculate statistics with error handling
                try:
                    pred_std = torch.std(torch.cat(predictions, dim=0))
                    pred_range = torch.max(torch.cat(predictions, dim=0)) - torch.min(torch.cat(predictions, dim=0))
                except RuntimeError:
                    pred_std = torch.tensor(0.0)
                    pred_range = torch.tensor(0.0)
                
                return {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'mean_memory': np.mean(memory_usage),
                    'max_memory': np.max(memory_usage),
                    'prediction_std': pred_std.item(),
                    'prediction_range': pred_range.item()
                }
        finally:
            # Final cleanup
            del predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def benchmark_training(self, model, X, y, num_epochs=2, batch_size=32):
        """Benchmark training performance with autonomous monitoring."""
        model.train()
        start_mem = self.get_memory_usage()
        start_time = time.time()
        
        try:
            # Validate input shapes
            if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise TypeError("Inputs must be torch.Tensor")
            if X.shape[0] != y.shape[0]:
                raise ValueError("Input and target batch sizes must match")
            
            # Use gradient accumulation for large batches
            accumulation_steps = max(1, batch_size // 32)
            effective_batch_size = batch_size // accumulation_steps
            
            # Track training metrics
            train_metrics = model.fit(
                X,
                y,
                epochs=num_epochs,
                batch_size=effective_batch_size,
                gradient_accumulation_steps=accumulation_steps
            )
            
            end_time = time.time()
            end_mem = self.get_memory_usage()
            
            # Check time limit
            if end_time - start_time > self.MAX_BENCHMARK_TIME:
                raise TimeoutError("Training benchmark exceeded time limit")
            
            # Check memory limit
            if end_mem - start_mem > self.MAX_MEMORY_MB:
                raise MemoryError(f"Training memory usage exceeded {self.MAX_MEMORY_MB}MB")
            
            return {
                'total_time': end_time - start_time,
                'memory_used': max(0, end_mem - start_mem),  # Ensure non-negative memory usage
                'samples_trained': len(X) * num_epochs,
                'final_loss': train_metrics.get('loss', float('inf')),
                'learning_rate': train_metrics.get('learning_rate', 0.0)
            }
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("GPU out of memory during training")
            raise e
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @pytest.mark.timeout(300)
    def test_batch_norm_performance(self, base_config, large_sample_data):
        """Test performance impact of batch normalization with autonomous monitoring."""
        # Test without batch norm
        config_no_bn = base_config.copy()
        model_no_bn = LSTMForecaster(config=config_no_bn)
        X_no_bn, y_no_bn = model_no_bn._prepare_data(large_sample_data, is_training=True)
        
        # Test with batch norm
        config_with_bn = base_config.copy()
        config_with_bn['use_batch_norm'] = True
        model_with_bn = LSTMForecaster(config=config_with_bn)
        X_with_bn, y_with_bn = model_with_bn._prepare_data(large_sample_data, is_training=True)
        
        try:
            # Benchmark forward pass
            no_bn_results = self.benchmark_forward_pass(model_no_bn, X_no_bn)
            with_bn_results = self.benchmark_forward_pass(model_with_bn, X_with_bn)
            
            # Benchmark training
            no_bn_train = self.benchmark_training(model_no_bn, X_no_bn, y_no_bn)
            with_bn_train = self.benchmark_training(model_with_bn, X_with_bn, y_with_bn)
            
            # Assert absolute performance requirements
            assert no_bn_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Base model forward pass too slow"
            assert with_bn_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Batch norm model forward pass too slow"
            assert no_bn_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Base model memory usage too high"
            assert with_bn_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Batch norm model memory usage too high"
            assert no_bn_train['total_time'] < self.MAX_TRAINING_TIME, "Base model training too slow"
            assert with_bn_train['total_time'] < self.MAX_TRAINING_TIME, "Batch norm model training too slow"
            assert no_bn_train['final_loss'] < self.MAX_LOSS, "Base model loss too high"
            assert with_bn_train['final_loss'] < self.MAX_LOSS, "Batch norm model loss too high"
        finally:
            # Cleanup
            del model_no_bn, model_with_bn
            del X_no_bn, y_no_bn, X_with_bn, y_with_bn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    @pytest.mark.timeout(300)
    def test_attention_performance(self, base_config, large_sample_data):
        """Test performance impact of attention mechanism with autonomous monitoring."""
        # Test without attention
        config_no_attn = base_config.copy()
        model_no_attn = LSTMForecaster(config=config_no_attn)
        X_no_attn, y_no_attn = model_no_attn._prepare_data(large_sample_data, is_training=True)
        
        # Test with attention
        config_with_attn = base_config.copy()
        config_with_attn.update({
            'use_attention': True,
            'num_attention_heads': 4,
            'attention_dropout': 0.1
        })
        model_with_attn = LSTMForecaster(config=config_with_attn)
        X_with_attn, y_with_attn = model_with_attn._prepare_data(large_sample_data, is_training=True)
        
        try:
            # Benchmark forward pass
            no_attn_results = self.benchmark_forward_pass(model_no_attn, X_no_attn)
            with_attn_results = self.benchmark_forward_pass(model_with_attn, X_with_attn)
            
            # Benchmark training
            no_attn_train = self.benchmark_training(model_no_attn, X_no_attn, y_no_attn)
            with_attn_train = self.benchmark_training(model_with_attn, X_with_attn, y_with_attn)
            
            # Assert absolute performance requirements
            assert no_attn_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Base model forward pass too slow"
            assert with_attn_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Attention model forward pass too slow"
            assert no_attn_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Base model memory usage too high"
            assert with_attn_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Attention model memory usage too high"
            assert no_attn_train['total_time'] < self.MAX_TRAINING_TIME, "Base model training too slow"
            assert with_attn_train['total_time'] < self.MAX_TRAINING_TIME, "Attention model training too slow"
            assert no_attn_train['final_loss'] < self.MAX_LOSS, "Base model loss too high"
            assert with_attn_train['final_loss'] < self.MAX_LOSS, "Attention model loss too high"
        finally:
            # Cleanup
            del model_no_attn, model_with_attn
            del X_no_attn, y_no_attn, X_with_attn, y_with_attn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    @pytest.mark.timeout(300)
    def test_combined_features_performance(self, base_config, large_sample_data):
        """Test performance impact of combining all features with autonomous monitoring."""
        # Test with minimal features
        config_minimal = base_config.copy()
        model_minimal = LSTMForecaster(config=config_minimal)
        X_minimal, y_minimal = model_minimal._prepare_data(large_sample_data, is_training=True)
        
        # Test with all features
        config_all = base_config.copy()
        config_all.update({
            'use_batch_norm': True,
            'use_layer_norm': True,
            'use_attention': True,
            'use_residual': True,
            'additional_dropout': 0.2,
            'num_attention_heads': 4,
            'attention_dropout': 0.1
        })
        model_all = LSTMForecaster(config=config_all)
        X_all, y_all = model_all._prepare_data(large_sample_data, is_training=True)
        
        try:
            # Benchmark forward pass
            minimal_results = self.benchmark_forward_pass(model_minimal, X_minimal)
            all_results = self.benchmark_forward_pass(model_all, X_all)
            
            # Benchmark training
            minimal_train = self.benchmark_training(model_minimal, X_minimal, y_minimal)
            all_train = self.benchmark_training(model_all, X_all, y_all)
            
            # Assert absolute performance requirements
            assert minimal_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Minimal model forward pass too slow"
            assert all_results['mean_time'] < self.MAX_FORWARD_PASS_TIME, "Full model forward pass too slow"
            assert minimal_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Minimal model memory usage too high"
            assert all_results['mean_memory'] < self.MAX_MEMORY_USAGE, "Full model memory usage too high"
            assert minimal_train['total_time'] < self.MAX_TRAINING_TIME, "Minimal model training too slow"
            assert all_train['total_time'] < self.MAX_TRAINING_TIME, "Full model training too slow"
            assert minimal_train['final_loss'] < self.MAX_LOSS, "Minimal model loss too high"
            assert all_train['final_loss'] < self.MAX_LOSS, "Full model loss too high"
        finally:
            # Cleanup
            del model_minimal, model_all
            del X_minimal, y_minimal, X_all, y_all
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    @pytest.mark.timeout(300)
    def test_memory_efficiency(self, base_config, large_sample_data):
        """Test memory efficiency with different batch sizes and autonomous monitoring."""
        batch_sizes = [32, 64, 128, 256]
        memory_usage = []
        training_metrics = []
        
        for batch_size in batch_sizes:
            model = LSTMForecaster(config=base_config)
            X, y = model._prepare_data(large_sample_data, is_training=True)
            
            try:
                # Measure memory usage during training
                start_mem = self.get_memory_usage()
                train_results = self.benchmark_training(model, X, y, batch_size=batch_size)
                end_mem = self.get_memory_usage()
                
                memory_usage.append(max(0, end_mem - start_mem))  # Ensure non-negative memory usage
                training_metrics.append(train_results)
                
                # Assert absolute performance requirements
                assert train_results['total_time'] < self.MAX_TRAINING_TIME, f"Training too slow for batch size {batch_size}"
                assert train_results['memory_used'] < self.MAX_MEMORY_USAGE, f"Memory usage too high for batch size {batch_size}"
                assert train_results['final_loss'] < self.MAX_LOSS, f"Loss too high for batch size {batch_size}"
            finally:
                # Cleanup after each batch size test
                del model
                del X, y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Assert memory usage scales sub-linearly with batch size
        for i in range(1, len(batch_sizes)):
            ratio = batch_sizes[i] / batch_sizes[i-1]
            memory_ratio = memory_usage[i] / (memory_usage[i-1] + 1e-6)  # Add small epsilon to avoid division by zero
            assert memory_ratio < ratio * 1.5, f"Memory usage should scale sub-linearly with batch size. Got ratio {memory_ratio:.2f} for batch size ratio {ratio:.2f}" 