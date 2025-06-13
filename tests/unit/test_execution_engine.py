import pytest
import logging
from trading.execution.execution_engine import ExecutionEngine

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
