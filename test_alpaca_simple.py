"""
Simple Alpaca Migration Test

This test only imports and tests Alpaca-related functionality
without importing the full trading package that has dependency issues.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def test_alpaca_imports():
    """Test that alpaca-py can be imported successfully"""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        logger.info("✓ All alpaca-py imports successful")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import alpaca-py: {e}")

def test_alpaca_enum_values():
    """Test that alpaca-py enum values are correct"""
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert TimeInForce.DAY.value == "day"
    assert TimeInForce.GTC.value == "gtc"
    logger.info("✓ All alpaca-py enum values correct")

def test_alpaca_client_creation():
    """Test that TradingClient can be created"""
    from alpaca.trading.client import TradingClient
    
    # This should not raise an exception
    client = TradingClient(api_key="test", secret_key="test", paper=True)
    assert client is not None
    logger.info("✓ TradingClient creation successful")

def test_alpaca_order_request():
    """Test that order requests can be created"""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    
    order_request = MarketOrderRequest(
        symbol="AAPL",
        qty=10,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    
    assert order_request.symbol == "AAPL"
    assert order_request.qty == 10
    assert order_request.side == OrderSide.BUY
    assert order_request.time_in_force == TimeInForce.DAY
    logger.info("✓ MarketOrderRequest creation successful")

def test_old_api_not_available():
    """Test that the old alpaca-trade-api is not available"""
    try:
        import alpaca_trade_api
        pytest.fail("Old alpaca-trade-api should not be available")
    except ImportError:
        logger.info("✓ Old alpaca-trade-api correctly not available")
        assert True

if __name__ == "__main__":
    logger.info("Running Alpaca Migration Tests...")
    test_alpaca_imports()
    test_alpaca_enum_values()
    test_alpaca_client_creation()
    test_alpaca_order_request()
    test_old_api_not_available()
    logger.info("All Alpaca migration tests passed!") 