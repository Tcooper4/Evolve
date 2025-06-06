import pytest
import os
import logging
from pathlib import Path
from trading.utils.logging import (
    LogManager,
    ModelLogger,
    DataLogger,
    PerformanceLogger,
    setup_logging
)

class TestLogging:
    """Test suite for logging utilities."""
    
    @pytest.fixture
    def log_manager(self, tmp_path):
        return LogManager(log_dir=str(tmp_path))
    
    @pytest.fixture
    def model_logger(self, tmp_path):
        return ModelLogger(log_dir=str(tmp_path))
    
    @pytest.fixture
    def data_logger(self, tmp_path):
        return DataLogger(log_dir=str(tmp_path))
    
    @pytest.fixture
    def performance_logger(self, tmp_path):
        return PerformanceLogger(log_dir=str(tmp_path))
    
    def test_log_manager_initialization(self, log_manager):
        """Test log manager initialization."""
        assert log_manager is not None
        assert log_manager.log_dir is not None
        assert os.path.exists(log_manager.log_dir)
    
    def test_log_manager_setup(self, log_manager):
        """Test log manager setup."""
        # Test logger setup
        logger = log_manager.setup_logger('test_logger')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
        
        # Test log level
        assert logger.level == logging.INFO
        
        # Test handler setup
        assert len(logger.handlers) > 0
    
    def test_model_logger(self, model_logger):
        """Test model logger."""
        # Test logging model info
        model_logger.log_model_info({
            'model_type': 'transformer',
            'd_model': 64,
            'nhead': 8
        })
        
        # Test logging training progress
        model_logger.log_training_progress(epoch=1, loss=0.5, val_loss=0.4)
        
        # Test logging model metrics
        model_logger.log_model_metrics({
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.93
        })
        
        # Check log file
        log_file = Path(model_logger.log_dir) / 'model.log'
        assert log_file.exists()
        
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'model_type' in log_content
            assert 'Epoch 1' in log_content
            assert 'accuracy' in log_content
    
    def test_data_logger(self, data_logger):
        """Test data logger."""
        # Test logging data info
        data_logger.log_data_info({
            'data_source': 'yfinance',
            'symbols': ['AAPL', 'MSFT'],
            'start_date': '2024-01-01'
        })
        
        # Test logging data statistics
        data_logger.log_data_statistics({
            'mean': 100.5,
            'std': 2.5,
            'min': 95.0,
            'max': 105.0
        })
        
        # Test logging data validation
        data_logger.log_data_validation({
            'missing_values': 0,
            'outliers': 2,
            'duplicates': 0
        })
        
        # Check log file
        log_file = Path(data_logger.log_dir) / 'data.log'
        assert log_file.exists()
        
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'data_source' in log_content
            assert 'mean' in log_content
            assert 'missing_values' in log_content
    
    def test_performance_logger(self, performance_logger):
        """Test performance logger."""
        # Test logging performance metrics
        performance_logger.log_performance_metrics({
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'max_drawdown': -0.1
        })
        
        # Test logging trade information
        performance_logger.log_trade_info({
            'entry_price': 100.0,
            'exit_price': 105.0,
            'profit': 5.0,
            'duration': '1d'
        })
        
        # Test logging portfolio value
        performance_logger.log_portfolio_value(100000.0)
        
        # Check log file
        log_file = Path(performance_logger.log_dir) / 'performance.log'
        assert log_file.exists()
        
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'sharpe_ratio' in log_content
            assert 'entry_price' in log_content
            assert 'Portfolio Value: 100000.00' in log_content
    
    def test_log_rotation(self, log_manager):
        """Test log rotation."""
        # Create multiple log files
        for i in range(5):
            logger = log_manager.setup_logger(f'test_logger_{i}')
            logger.info(f'Test message {i}')
        
        # Check log files
        log_files = list(Path(log_manager.log_dir).glob('*.log'))
        assert len(log_files) > 0
    
    def test_log_levels(self, log_manager):
        """Test different log levels."""
        logger = log_manager.setup_logger('test_logger')
        
        # Test different log levels
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # Check log file
        log_file = Path(log_manager.log_dir) / 'test_logger.log'
        assert log_file.exists()
        
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'Debug message' not in log_content
            assert 'Info message' in log_content
            assert 'Warning message' in log_content
            assert 'Error message' in log_content
            assert 'Critical message' in log_content
    
    def test_log_formatting(self, log_manager):
        """Test log formatting."""
        logger = log_manager.setup_logger('test_logger')
        
        # Test log formatting
        logger.info('Test message with %s', 'formatting')
        
        # Check log file
        log_file = Path(log_manager.log_dir) / 'test_logger.log'
        assert log_file.exists()
        
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'Test message with formatting' in log_content
    
    def test_log_cleanup(self, log_manager):
        """Test log cleanup."""
        # Create some log files
        loggers = []
        for i in range(3):
            logger = log_manager.setup_logger(f'test_logger_{i}')
            logger.info(f'Test message {i}')
            loggers.append(logger)
        
        # Remove handlers to release file handles
        for logger in loggers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        # Clean up logs
        log_manager.cleanup_logs(days=0)
        
        # Check if logs are cleaned up
        log_files = list(Path(log_manager.log_dir).glob('*.log'))
        assert len(log_files) == 0

def test_setup_logging():
    """Test that logging setup creates a logger with the correct handlers."""
    log_dir = 'test_logs'
    logger = setup_logging(log_dir=log_dir, log_level=logging.DEBUG)
    
    # Check that the logger has the correct level
    assert logger.level == logging.DEBUG
    
    # Check that the logger has two handlers (file and console)
    assert len(logger.handlers) == 2
    
    # Check that the log directory and file are created
    log_file = Path(log_dir) / 'trading.log'
    assert log_file.exists()
    
    # Clean up
    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    log_file.unlink()
    Path(log_dir).rmdir() 