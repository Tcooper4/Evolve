import pytest
import logging
from unittest.mock import patch, MagicMock
from trading.execution.execution_engine import ExecutionEngine, ExecutionError
import pandas as pd

def test_execution_engine():
    """Test the execution engine functionality."""
    ee = ExecutionEngine()
    
    # Test market order execution
    market_trade = ee.execute_market_order('AAPL', 10, 150.0)
    assert market_trade['asset'] == 'AAPL'
    assert market_trade['quantity'] == 10
    assert market_trade['price'] == 150.0
    assert market_trade['type'] == 'market'
    
    # Test limit order execution
    limit_trade = ee.execute_limit_order('GOOGL', 5, 2800.0)
    assert limit_trade is not None
    assert limit_trade['asset'] == 'GOOGL'
    assert limit_trade['quantity'] == 5
    assert limit_trade['type'] == 'limit'
    
    # Test trade history
    trade_history = ee.get_trade_history()
    assert len(trade_history) == 2
    assert trade_history[0] == market_trade
    assert trade_history[1] == limit_trade

    # Test fill or kill order - should return None if price higher than limit
    fok_trade = ee.execute_fill_or_kill('AAPL', 1, 100.0)
    assert fok_trade is None or fok_trade['type'] == 'fill_or_kill'

def test_missing_signal_input():
    """Test execution engine behavior with missing signal input."""
    ee = ExecutionEngine()
    
    # Test with None signal
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order(None, 10, 150.0)
    assert "Invalid signal" in str(exc_info.value)
    
    # Test with empty signal
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('', 10, 150.0)
    assert "Invalid signal" in str(exc_info.value)
    
    # Test with missing required fields
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', None, 150.0)
    assert "Invalid quantity" in str(exc_info.value)
    
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 10, None)
    assert "Invalid price" in str(exc_info.value)

def test_invalid_ticker_timeframe():
    """Test execution engine behavior with invalid ticker or timeframe."""
    ee = ExecutionEngine()
    
    # Test with invalid ticker
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('INVALID_TICKER', 10, 150.0)
    assert "Invalid ticker" in str(exc_info.value)
    
    # Test with invalid timeframe
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 10, 150.0, timeframe='invalid')
    assert "Invalid timeframe" in str(exc_info.value)
    
    # Test with future date
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 10, 150.0, execution_time='2099-12-31')
    assert "Invalid execution time" in str(exc_info.value)
    
    # Test with invalid quantity
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', -10, 150.0)
    assert "Invalid quantity" in str(exc_info.value)
    
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 0, 150.0)
    assert "Invalid quantity" in str(exc_info.value)

@patch('trading.execution.execution_engine.yf.Ticker')
def test_trade_result_mismatch(mock_ticker):
    """Test execution engine behavior with trade result mismatches."""
    ee = ExecutionEngine()
    
    # Mock yfinance data
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame({
        'Open': [150.0],
        'High': [151.0],
        'Low': [149.0],
        'Close': [150.5],
        'Volume': [1000000]
    })
    mock_ticker.return_value = mock_ticker_instance
    
    # Test price mismatch
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 10, 100.0)  # Price much lower than market
    assert "Price mismatch" in str(exc_info.value)
    
    # Test quantity mismatch
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 1000000, 150.0)  # Quantity too large
    assert "Quantity mismatch" in str(exc_info.value)
    
    # Test execution time mismatch
    with pytest.raises(ExecutionError) as exc_info:
        ee.execute_market_order('AAPL', 10, 150.0, execution_time='2023-01-01')
    assert "Execution time mismatch" in str(exc_info.value)

def test_error_handling():
    """Test execution engine error handling."""
    ee = ExecutionEngine()
    
    # Test network error
    with patch('trading.execution.execution_engine.requests.get', side_effect=Exception("Network error")):
        with pytest.raises(ExecutionError) as exc_info:
            ee.execute_market_order('AAPL', 10, 150.0)
        assert "Network error" in str(exc_info.value)
    
    # Test API rate limit
    with patch('trading.execution.execution_engine.requests.get', side_effect=Exception("Rate limit exceeded")):
        with pytest.raises(ExecutionError) as exc_info:
            ee.execute_market_order('AAPL', 10, 150.0)
        assert "Rate limit" in str(exc_info.value)
    
    # Test invalid response
    with patch('trading.execution.execution_engine.requests.get', return_value=MagicMock(status_code=400)):
        with pytest.raises(ExecutionError) as exc_info:
            ee.execute_market_order('AAPL', 10, 150.0)
        assert "Invalid response" in str(exc_info.value)
    
    # Test timeout
    with patch('trading.execution.execution_engine.requests.get', side_effect=Exception("Timeout")):
        with pytest.raises(ExecutionError) as exc_info:
            ee.execute_market_order('AAPL', 10, 150.0)
        assert "Timeout" in str(exc_info.value)

if __name__ == '__main__':
    pytest.main([__file__])
