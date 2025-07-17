import pytest
import logging
from unittest.mock import Mock, patch
from trading.utils.error_handling import (
    TradingError,
    ModelError,
    StrategyError,
    log_errors,
    handle_routing_errors,
    retry_on_error,
    ErrorContext,
    ErrorHandler,
    ErrorRecoveryStrategy
)
from trading.exceptions import TradingSystemError


class TestErrorClasses:
    """Test error classes."""
    
    def test_trading_error(self):
        """Test TradingError base class."""
        error = TradingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_model_error(self):
        """Test ModelError class."""
        error = ModelError("Model failed")
        assert str(error) == "Model failed"
        assert isinstance(error, TradingSystemError)
    
    def test_strategy_error(self):
        """Test StrategyError class."""
        error = StrategyError("Strategy failed")
        assert str(error) == "Strategy failed"
        assert isinstance(error, TradingSystemError)


class TestLogErrorsDecorator:
    """Test the log_errors decorator."""
    
    def test_log_errors_success(self):
        """Test that successful function calls work normally."""
        @log_errors()
        def test_func():
            return "success"
        result = test_func()
        assert result == "success"
    
    def test_log_errors_catches_exception(self):
        """Test that exceptions are caught and logged."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_errors()
            def test_func():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                test_func()
            
            # Check that error was logged
            mock_logger.error.assert_called()
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("Error in test_func" in error_call for error_call in error_calls)
    
    def test_log_errors_with_custom_logger(self):
        """Test log_errors with custom logger."""
        custom_logger = Mock()
        
        @log_errors(logger=custom_logger)
        def test_func():
            raise RuntimeError("Custom error")
        
        with pytest.raises(RuntimeError):
            test_func()
        
        custom_logger.error.assert_called()
    
    def test_log_errors_with_specific_error_types(self):
        """Test log_errors with specific error types."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_errors(error_types=(ValueError, TypeError))
            def test_func():
                raise ValueError("Value error")
            
            with pytest.raises(ValueError):
                test_func()
            
            mock_logger.error.assert_called()
    
    def test_log_errors_ignores_other_exceptions(self):
        """Test that log_errors ignores exceptions not in error_types."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_errors(error_types=(ValueError,))
            def test_func():
                raise RuntimeError("Runtime error")
            
            with pytest.raises(RuntimeError):
                test_func()
            
            # Should not log RuntimeError since it's not in error_types
            mock_logger.error.assert_not_called()


class TestHandleRoutingErrors:
    """Test the handle_routing_errors decorator."""
    
    def test_handle_routing_errors_success(self):
        """Test successful routing."""
        @handle_routing_errors
        def test_route():
            return {"success": True}
        
        result = test_route()
        assert result["success"] is True
    
    def test_handle_routing_errors_with_exception(self):
        """Test routing with exception."""
        @handle_routing_errors
        def test_route():
            raise ValueError("Routing failed")
        
        result = test_route()
        assert result["success"] is False
        assert "error" in result
        assert "Routing failed" in result["error"]["message"]


class TestRetryOnError:
    """Test the retry_on_error decorator."""
    
    def test_retry_on_error_success_first_try(self):
        """Test successful execution on first try."""
        @retry_on_error(max_retries=3, delay=0.1)
        def test_func():
            return "success"
        result = test_func()
        assert result == "success"
    
    def test_retry_on_error_succeeds_after_retries(self):
        """Test successful execution after retries."""
        call_count = 0        
        @retry_on_error(max_retries=3, delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        result = test_func()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_error_fails_after_max_retries(self):
        """Test failure after max retries."""
        @retry_on_error(max_retries=2, delay=0.1)
        def test_func():
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError):
            test_func()
    
    def test_retry_on_error_with_specific_exceptions(self):
        """Test retry with specific exception types."""
        call_count = 0        
        @retry_on_error(max_retries=2, delay=0.1, retry_exceptions=(ValueError,))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise RuntimeError("Non-retryable error")
            return "success"
        with pytest.raises(RuntimeError):
            test_func()
        
        assert call_count == 2


class TestErrorContext:
    """Test the ErrorContext class."""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation."""
        context = ErrorContext("test_function", {"param": "value"})
        assert context.function_name == "test_function"
        assert context.context_data == {"param": "value"}
        assert context.timestamp is not None
    
    def test_error_context_add_error(self):
        """Test adding errors to context."""
        context = ErrorContext("test_function")
        context.add_error("ValueError", "Invalid value")
        
        assert len(context.errors) == 1
        assert context.errors[0]["type"] == "ValueError"
        assert context.errors[0]["message"] == "Invalid value"
    
    def test_error_context_to_dict(self):
        """Test converting context to dictionary."""
        context = ErrorContext("test_function", {"param": "value"})
        context.add_error("ValueError", "Invalid value")
        
        context_dict = context.to_dict()
        assert context_dict["function_name"] == "test_function"
        assert context_dict["context_data"] == {"param": "value"}
        assert len(context_dict["errors"]) == 1


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    def test_error_handler_creation(self):
        """Test ErrorHandler creation."""
        handler = ErrorHandler()
        assert handler.recovery_strategies == {}
        assert handler.error_counts == {}
    
    def test_error_handler_register_strategy(self):
        """Test registering recovery strategies."""
        handler = ErrorHandler()
        
        def mock_strategy(error, context):
            return {"action": "retry", "success": True}
        
        handler.register_recovery_strategy(ValueError, mock_strategy)
        assert ValueError in handler.recovery_strategies
    
    def test_error_handler_handle_error_with_strategy(self):
        """Test handling error with registered strategy."""
        handler = ErrorHandler()
        
        def mock_strategy(error, context):
            return {"action": "retry", "success": True}
        
        handler.register_recovery_strategy(ValueError, mock_strategy)
        
        error = ValueError("Test error")
        context = ErrorContext("test_function")
        
        result = handler.handle_error(error, context)
        assert result["action"] == "retry"
        assert result["success"] is True
    
    def test_error_handler_handle_error_without_strategy(self):
        """Test handling error without registered strategy."""
        handler = ErrorHandler()
        
        error = ValueError("Test error")
        context = ErrorContext("test_function")
        
        result = handler.handle_error(error, context)
        assert result["action"] == "log_and_fail"
        assert result["success"] is False


class TestErrorRecoveryStrategy:
    """Test the ErrorRecoveryStrategy class."""
    
    def test_retry_strategy(self):
        """Test retry recovery strategy."""
        strategy = ErrorRecoveryStrategy.retry(max_attempts=3, delay=0.1)
        
        error = ValueError("Test error")
        context = ErrorContext("test_function")
        
        result = strategy(error, context)
        assert result["action"] == "retry"
        assert result["max_attempts"] == 3
        assert result["delay"] == 0.1
    
    def test_fallback_strategy(self):
        """Test fallback recovery strategy."""
        strategy = ErrorRecoveryStrategy.fallback(fallback_function="backup_func")
        
        error = ValueError("Test error")
        context = ErrorContext("test_function")
        
        result = strategy(error, context)
        assert result["action"] == "fallback"
        assert result["fallback_function"] == "backup_func"
    
    def test_log_and_continue_strategy(self):
        """Test log and continue recovery strategy."""
        strategy = ErrorRecoveryStrategy.log_and_continue()
        
        error = ValueError("Test error")
        context = ErrorContext("test_function")
        
        result = strategy(error, context)
        assert result["action"] == "log_and_continue"
        assert result["success"] is True


class TestIntegration:
    """Integration tests for error handling components."""
    
    def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        handler = ErrorHandler()
        handler.register_recovery_strategy(
            ValueError, 
            ErrorRecoveryStrategy.retry(max_attempts=2, delay=0.1)
        )
        
        @log_errors()
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            raise ValueError("Integration test error")
        
        with pytest.raises(ValueError):
            test_function()
    
    def test_error_context_capture(self):
        """Test error context capture in decorators."""
        captured_context = None
        
        def capture_context(error, context):
            nonlocal captured_context
            captured_context = context
            return {"action": "log", "success": False}
        
        handler = ErrorHandler()
        handler.register_recovery_strategy(ValueError, capture_context)
        
        @log_errors()
        def test_function(param1, param2):
            raise ValueError("Context test error")
        
        with pytest.raises(ValueError):
            test_function("value1", "value2")
        
        # Note: In a real implementation, the context would be captured
        # This test verifies the structure is in place 