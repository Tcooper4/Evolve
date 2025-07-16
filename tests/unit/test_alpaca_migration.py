"""
Test Alpaca Migration from alpaca-trade-api to alpaca-py

This test suite verifies that the migration from the old alpaca-trade-api
to the new alpaca-py SDK works correctly across all components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

# Test the execution providers
from trading.agents.execution.execution_providers import (
    AlpacaProvider,
    ExecutionMode,
    create_execution_provider
)

# Test the live trading interface
from execution.live_trading_interface import (
    AlpacaTradingInterface,
    OrderRequest,
    OrderStatus,
    Position,
    AccountInfo
)

# Test the broker adapter
from execution.broker_adapter import (
    AlpacaBrokerAdapter,
    OrderRequest as BrokerOrderRequest,
    OrderExecution,
    Position as BrokerPosition,
    AccountInfo as BrokerAccountInfo
)


class TestAlpacaProviderMigration:
    """Test AlpacaProvider migration to alpaca-py"""
    
    @pytest.fixture
    def alpaca_config(self):
        """Test configuration for Alpaca"""
        return {
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
            "base_url": "https://paper-api.alpaca.markets"
        }
    
    @pytest.fixture
    def alpaca_provider(self, alpaca_config):
        """Create AlpacaProvider instance"""
        return AlpacaProvider(alpaca_config)
    
    @pytest.mark.asyncio
    async def test_alpaca_provider_connect(self, alpaca_provider):
        """Test AlpacaProvider connection with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.data.historical.StockHistoricalDataClient') as mock_data_client:
            
            # Mock successful connection
            mock_account = Mock()
            mock_account.cash = "10000.00"
            mock_account.buying_power = "15000.00"
            mock_account.equity = "12000.00"
            mock_account.portfolio_value = "12000.00"
            mock_account.daytrade_count = 0
            mock_account.pattern_day_trader = False
            
            mock_trading_client.return_value.get_account.return_value = mock_account
            
            # Test connection
            result = await alpaca_provider.connect()
            
            assert result is True
            assert alpaca_provider.is_connected is True
            assert alpaca_provider.trading_client is not None
            assert alpaca_provider.data_client is not None
    
    @pytest.mark.asyncio
    async def test_alpaca_provider_execute_trade(self, alpaca_provider):
        """Test AlpacaProvider trade execution with alpaca-py"""
        # Mock the trade signal
        from trading.agents.execution.trade_signals import TradeSignal, TradeDirection
        
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_price=150.0,
            size=10,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.trading.requests.MarketOrderRequest') as mock_order_request, \
             patch('alpaca.trading.enums.OrderSide') as mock_order_side, \
             patch('alpaca.trading.enums.TimeInForce') as mock_time_in_force:
            
            # Mock successful order submission
            mock_order = Mock()
            mock_order.id = "test_order_123"
            mock_trading_client.return_value.submit_order.return_value = mock_order
            
            # Connect first
            alpaca_provider.is_connected = True
            alpaca_provider.trading_client = mock_trading_client.return_value
            
            # Test trade execution
            result = await alpaca_provider.execute_trade(signal)
            
            assert result["success"] is True
            assert result["order_id"] == "test_order_123"
            assert result["execution_price"] == 150.0
    
    @pytest.mark.asyncio
    async def test_alpaca_provider_get_account_info(self, alpaca_provider):
        """Test AlpacaProvider account info retrieval"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock account info
            mock_account = Mock()
            mock_account.cash = "10000.00"
            mock_account.buying_power = "15000.00"
            mock_account.equity = "12000.00"
            mock_account.portfolio_value = "12000.00"
            
            mock_trading_client.return_value.get_account.return_value = mock_account
            
            # Connect first
            alpaca_provider.is_connected = True
            alpaca_provider.trading_client = mock_trading_client.return_value
            
            # Test account info retrieval
            result = await alpaca_provider.get_account_info()
            
            assert result["balance"] == 10000.0
            assert result["buying_power"] == 15000.0
            assert result["equity"] == 12000.0
            assert result["cash"] == 10000.0
    
    @pytest.mark.asyncio
    async def test_alpaca_provider_get_positions(self, alpaca_provider):
        """Test AlpacaProvider positions retrieval"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock positions
            mock_position = Mock()
            mock_position.symbol = "AAPL"
            mock_position.qty = "100"
            mock_position.avg_entry_price = "150.00"
            mock_position.market_value = "15000.00"
            
            mock_trading_client.return_value.get_all_positions.return_value = [mock_position]
            
            # Connect first
            alpaca_provider.is_connected = True
            alpaca_provider.trading_client = mock_trading_client.return_value
            
            # Test positions retrieval
            result = await alpaca_provider.get_positions()
            
            assert "AAPL" in result
            assert result["AAPL"]["quantity"] == 100.0
            assert result["AAPL"]["avg_entry_price"] == 150.0
            assert result["AAPL"]["market_value"] == 15000.0


class TestAlpacaTradingInterfaceMigration:
    """Test AlpacaTradingInterface migration to alpaca-py"""
    
    @pytest.fixture
    def alpaca_config(self):
        """Test configuration for Alpaca"""
        return {
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
            "paper_trading": True
        }
    
    @pytest.fixture
    def alpaca_interface(self, alpaca_config):
        """Create AlpacaTradingInterface instance"""
        return AlpacaTradingInterface(
            api_key=alpaca_config["api_key"],
            secret_key=alpaca_config["secret_key"],
            paper_trading=alpaca_config["paper_trading"]
        )
    
    def test_alpaca_interface_initialization(self, alpaca_interface):
        """Test AlpacaTradingInterface initialization"""
        assert alpaca_interface.api_key == "test_api_key"
        assert alpaca_interface.secret_key == "test_secret_key"
        assert alpaca_interface.paper_trading is True
    
    def test_alpaca_interface_place_market_order(self, alpaca_interface):
        """Test placing market order with alpaca-py"""
        order_request = OrderRequest(
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            time_in_force="day"
        )
        
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.trading.requests.MarketOrderRequest') as mock_order_request, \
             patch('alpaca.trading.enums.OrderSide') as mock_order_side, \
             patch('alpaca.trading.enums.TimeInForce') as mock_time_in_force:
            
            # Mock successful order submission
            mock_order = Mock()
            mock_order.id = "test_order_123"
            mock_order.symbol = "AAPL"
            mock_order.side.value = "buy"
            mock_order.order_type.value = "market"
            mock_order.qty = 10
            mock_order.filled_qty = 0
            mock_order.status.value = "submitted"
            mock_order.limit_price = None
            mock_order.stop_price = None
            mock_order.filled_avg_price = None
            mock_order.created_at = datetime.now()
            mock_order.filled_at = None
            mock_order.client_order_id = None
            
            mock_trading_client.return_value.submit_order.return_value = mock_order
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test order placement
            result = alpaca_interface.place_order(order_request)
            
            assert result.order_id == "test_order_123"
            assert result.symbol == "AAPL"
            assert result.side == "buy"
            assert result.order_type == "market"
            assert result.quantity == 10.0
    
    def test_alpaca_interface_place_limit_order(self, alpaca_interface):
        """Test placing limit order with alpaca-py"""
        order_request = OrderRequest(
            symbol="AAPL",
            side="sell",
            quantity=5,
            order_type="limit",
            limit_price=155.0,
            time_in_force="day"
        )
        
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.trading.requests.LimitOrderRequest') as mock_order_request, \
             patch('alpaca.trading.enums.OrderSide') as mock_order_side, \
             patch('alpaca.trading.enums.TimeInForce') as mock_time_in_force:
            
            # Mock successful order submission
            mock_order = Mock()
            mock_order.id = "test_limit_order_456"
            mock_order.symbol = "AAPL"
            mock_order.side.value = "sell"
            mock_order.order_type.value = "limit"
            mock_order.qty = 5
            mock_order.filled_qty = 0
            mock_order.status.value = "submitted"
            mock_order.limit_price = 155.0
            mock_order.stop_price = None
            mock_order.filled_avg_price = None
            mock_order.created_at = datetime.now()
            mock_order.filled_at = None
            mock_order.client_order_id = None
            
            mock_trading_client.return_value.submit_order.return_value = mock_order
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test order placement
            result = alpaca_interface.place_order(order_request)
            
            assert result.order_id == "test_limit_order_456"
            assert result.symbol == "AAPL"
            assert result.side == "sell"
            assert result.order_type == "limit"
            assert result.quantity == 5.0
            assert result.limit_price == 155.0
    
    def test_alpaca_interface_cancel_order(self, alpaca_interface):
        """Test canceling order with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock successful cancellation
            mock_trading_client.return_value.cancel_order_by_id.return_value = None
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test order cancellation
            result = alpaca_interface.cancel_order("test_order_123")
            
            assert result is True
            mock_trading_client.return_value.cancel_order_by_id.assert_called_once_with("test_order_123")
    
    def test_alpaca_interface_get_order_status(self, alpaca_interface):
        """Test getting order status with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock order details
            mock_order = Mock()
            mock_order.id = "test_order_123"
            mock_order.symbol = "AAPL"
            mock_order.side.value = "buy"
            mock_order.order_type.value = "market"
            mock_order.qty = 10
            mock_order.filled_qty = 10
            mock_order.status.value = "filled"
            mock_order.limit_price = None
            mock_order.stop_price = None
            mock_order.filled_avg_price = 150.5
            mock_order.created_at = datetime.now()
            mock_order.filled_at = datetime.now()
            mock_order.client_order_id = None
            
            mock_trading_client.return_value.get_order_by_id.return_value = mock_order
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test order status retrieval
            result = alpaca_interface.get_order_status("test_order_123")
            
            assert result is not None
            assert result.order_id == "test_order_123"
            assert result.symbol == "AAPL"
            assert result.side == "buy"
            assert result.status == "filled"
            assert result.filled_quantity == 10.0
            assert result.filled_price == 150.5
    
    def test_alpaca_interface_get_positions(self, alpaca_interface):
        """Test getting positions with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock positions
            mock_position = Mock()
            mock_position.symbol = "AAPL"
            mock_position.qty = "100"
            mock_position.avg_entry_price = "150.00"
            mock_position.market_value = "15000.00"
            mock_position.unrealized_pl = "500.00"
            
            mock_trading_client.return_value.get_all_positions.return_value = [mock_position]
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test positions retrieval
            result = alpaca_interface.get_positions()
            
            assert "AAPL" in result
            position = result["AAPL"]
            assert position.symbol == "AAPL"
            assert position.quantity == 100.0
            assert position.average_price == 150.0
            assert position.market_value == 15000.0
            assert position.unrealized_pl == 500.0
    
    def test_alpaca_interface_get_account_info(self, alpaca_interface):
        """Test getting account info with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock account info
            mock_account = Mock()
            mock_account.id = "test_account_123"
            mock_account.cash = "10000.00"
            mock_account.buying_power = "15000.00"
            mock_account.equity = "12000.00"
            mock_account.portfolio_value = "12000.00"
            mock_account.daytrade_count = 0
            mock_account.pattern_day_trader = False
            
            mock_trading_client.return_value.get_account.return_value = mock_account
            
            # Set up the interface
            alpaca_interface.trading_client = mock_trading_client.return_value
            
            # Test account info retrieval
            result = alpaca_interface.get_account_info()
            
            assert result.account_id == "test_account_123"
            assert result.cash == 10000.0
            assert result.buying_power == 15000.0
            assert result.equity == 12000.0
            assert result.portfolio_value == 12000.0
            assert result.day_trade_count == 0
            assert result.pattern_day_trader is False


class TestAlpacaBrokerAdapterMigration:
    """Test AlpacaBrokerAdapter migration to alpaca-py"""
    
    @pytest.fixture
    def alpaca_config(self):
        """Test configuration for Alpaca"""
        return {
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
            "base_url": "https://paper-api.alpaca.markets"
        }
    
    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create AlpacaBrokerAdapter instance"""
        return AlpacaBrokerAdapter(alpaca_config)
    
    @pytest.mark.asyncio
    async def test_alpaca_adapter_connect(self, alpaca_adapter):
        """Test AlpacaBrokerAdapter connection with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.data.historical.StockHistoricalDataClient') as mock_data_client:
            
            # Mock successful connection
            mock_account = Mock()
            mock_account.id = "test_account_123"
            mock_trading_client.return_value.get_account.return_value = mock_account
            
            # Test connection
            result = await alpaca_adapter.connect()
            
            assert result is True
            assert alpaca_adapter.is_connected is True
            assert alpaca_adapter.client is not None
    
    @pytest.mark.asyncio
    async def test_alpaca_adapter_submit_market_order(self, alpaca_adapter):
        """Test submitting market order with alpaca-py"""
        order_request = BrokerOrderRequest(
            order_id="test_order_123",
            ticker="AAPL",
            side="buy",
            order_type="market",
            quantity=10
        )
        
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client, \
             patch('alpaca.trading.requests.MarketOrderRequest') as mock_order_request, \
             patch('alpaca.trading.enums.OrderSide') as mock_order_side, \
             patch('alpaca.trading.enums.TimeInForce') as mock_time_in_force:
            
            # Mock successful order submission
            mock_order = Mock()
            mock_order.id = "alpaca_order_456"
            mock_trading_client.return_value.submit_order.return_value = mock_order
            
            # Mock order details
            mock_order_details = Mock()
            mock_order_details.id = "alpaca_order_456"
            mock_order_details.symbol = "AAPL"
            mock_order_details.side.value = "buy"
            mock_order_details.order_type.value = "market"
            mock_order_details.qty = 10
            mock_order_details.filled_qty = 10
            mock_order_details.status.value = "filled"
            mock_order_details.limit_price = None
            mock_order_details.filled_avg_price = 150.5
            mock_order_details.submitted_at = datetime.now()
            
            mock_trading_client.return_value.get_order_by_id.return_value = mock_order_details
            
            # Connect first
            alpaca_adapter.is_connected = True
            alpaca_adapter.client = mock_trading_client.return_value
            
            # Test order submission
            result = await alpaca_adapter.submit_order(order_request)
            
            assert result.order_id == "test_order_123"
            assert result.ticker == "AAPL"
            assert result.side == "buy"
            assert result.order_type == "market"
            assert result.executed_quantity == 10.0
            assert result.average_price == 150.5
            assert result.status == "filled"
    
    @pytest.mark.asyncio
    async def test_alpaca_adapter_get_position(self, alpaca_adapter):
        """Test getting position with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock position
            mock_position = Mock()
            mock_position.symbol = "AAPL"
            mock_position.qty = "100"
            mock_position.avg_entry_price = "150.00"
            mock_position.market_value = "15000.00"
            mock_position.unrealized_pl = "500.00"
            
            mock_trading_client.return_value.get_position.return_value = mock_position
            
            # Connect first
            alpaca_adapter.is_connected = True
            alpaca_adapter.client = mock_trading_client.return_value
            
            # Test position retrieval
            result = await alpaca_adapter.get_position("AAPL")
            
            assert result is not None
            assert result.ticker == "AAPL"
            assert result.quantity == 100.0
            assert result.average_price == 150.0
            assert result.market_value == 15000.0
            assert result.unrealized_pnl == 500.0
    
    @pytest.mark.asyncio
    async def test_alpaca_adapter_get_account_info(self, alpaca_adapter):
        """Test getting account info with alpaca-py"""
        with patch('alpaca.trading.client.TradingClient') as mock_trading_client:
            # Mock account info
            mock_account = Mock()
            mock_account.id = "test_account_123"
            mock_account.cash = "10000.00"
            mock_account.buying_power = "15000.00"
            mock_account.equity = "12000.00"
            mock_account.margin_used = "0.00"
            
            mock_trading_client.return_value.get_account.return_value = mock_account
            
            # Connect first
            alpaca_adapter.is_connected = True
            alpaca_adapter.client = mock_trading_client.return_value
            
            # Test account info retrieval
            result = await alpaca_adapter.get_account_info()
            
            assert result.account_id == "test_account_123"
            assert result.cash == 10000.0
            assert result.buying_power == 15000.0
            assert result.equity == 12000.0
            assert result.margin_used == 0.0


class TestAlpacaMigrationIntegration:
    """Integration tests for Alpaca migration"""
    
    def test_import_compatibility(self):
        """Test that all alpaca-py imports work correctly"""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest
            )
            from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
            
            # Test that we can create instances
            client = TradingClient(api_key="test", secret_key="test", paper=True)
            assert client is not None
            
            # Test enum values
            assert OrderSide.BUY.value == "buy"
            assert OrderSide.SELL.value == "sell"
            assert OrderType.MARKET.value == "market"
            assert OrderType.LIMIT.value == "limit"
            assert TimeInForce.DAY.value == "day"
            assert TimeInForce.GTC.value == "gtc"
            
        except ImportError as e:
            pytest.fail(f"Failed to import alpaca-py components: {e}")
    
    def test_old_api_removed(self):
        """Test that the old alpaca-trade-api is no longer used"""
        import sys
        
        # Check that alpaca_trade_api is not imported anywhere
        for module_name in sys.modules:
            if 'alpaca_trade_api' in module_name:
                pytest.fail(f"Old alpaca-trade-api still imported: {module_name}")
    
    def test_new_api_available(self):
        """Test that the new alpaca-py is available"""
        try:
            import alpaca
            assert alpaca is not None
        except ImportError:
            pytest.fail("alpaca-py is not available")


if __name__ == "__main__":
    pytest.main([__file__]) 