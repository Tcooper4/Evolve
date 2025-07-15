"""
Tests for Execution Simulator

This module provides comprehensive tests for the execution simulator,
including edge cases and real-world scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from trading.execution.execution_simulator import (
    ExecutionSimulator,
    ExecutionOrder,
    ExecutionResult,
    OrderType,
    OrderStatus,
    FillType,
    OrderBook,
    OrderBookLevel
)


class TestExecutionSimulator:
    """Test cases for ExecutionSimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create a test execution simulator."""
        config = {
            "base_commission": 0.005,
            "commission_rate": 0.001,
            "base_slippage": 0.0001,
            "volume_impact_factor": 0.1,
            "volatility_impact_factor": 0.5,
            "base_fill_rate": 0.95,
            "market_hours_only": False,  # Disable for testing
            "max_order_size": 1000000,
            "min_order_size": 100
        }
        return ExecutionSimulator(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_orderbook(self):
        """Create sample orderbook."""
        mid_price = 150.0
        spread = 0.002
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = mid_price * (1 - spread/2 - i * 0.001)
            ask_price = mid_price * (1 + spread/2 + i * 0.001)
            
            bids.append(OrderBookLevel(
                price=bid_price,
                quantity=1000 * (0.8 ** i),
                side='bid',
                timestamp=datetime.now()
            ))
            
            asks.append(OrderBookLevel(
                price=ask_price,
                quantity=1000 * (0.8 ** i),
                side='ask',
                timestamp=datetime.now()
            ))
        
        return OrderBook(
            symbol='AAPL',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid_price
        )
    
    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator is not None
        assert len(simulator.orders) == 0
        assert len(simulator.execution_history) == 0
        assert simulator.base_commission == 0.005
        assert simulator.base_slippage == 0.0001
    
    def test_place_valid_order(self, simulator):
        """Test placing a valid order."""
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100,
            price=None
        )
        
        assert order_id in simulator.orders
        order = simulator.orders[order_id]
        assert order.symbol == 'AAPL'
        assert order.order_type == OrderType.MARKET
        assert order.side == 'buy'
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
    
    def test_place_invalid_order(self, simulator):
        """Test placing an invalid order."""
        with pytest.raises(ValueError):
            simulator.place_order(
                symbol='',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=0,
                price=None
            )
    
    def test_place_limit_order_without_price(self, simulator):
        """Test placing limit order without price."""
        with pytest.raises(ValueError):
            simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.LIMIT,
                side='buy',
                quantity=100,
                price=None
            )
    
    def test_order_size_constraints(self, simulator):
        """Test order size constraints."""
        # Test minimum order size
        with pytest.raises(ValueError):
            simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=1,  # Very small order
                price=100
            )
        
        # Test maximum order size
        with pytest.raises(ValueError):
            simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100000,  # Very large order
                price=100
            )
    
    def test_execute_market_order_success(self, simulator, sample_market_data, sample_orderbook):
        """Test successful market order execution."""
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Execute order
        result = simulator.execute_order(order_id, sample_market_data, sample_orderbook)
        
        assert result.success is True
        assert result.order_id == order_id
        assert result.quantity == 100
        assert result.price > 0
        assert result.slippage >= 0
        assert result.commission >= 0
        assert result.fill_type == FillType.IMMEDIATE
    
    def test_execute_market_order_failure(self, simulator):
        """Test market order execution failure."""
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Execute with empty market data (should fail)
        empty_data = pd.DataFrame()
        result = simulator.execute_order(order_id, empty_data)
        
        assert result.success is False
        assert result.failure_reason is not None
    
    def test_execute_limit_order(self, simulator, sample_market_data):
        """Test limit order execution."""
        # Place limit order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.LIMIT,
            side='buy',
            quantity=100,
            price=150.0
        )
        
        # Execute order
        result = simulator.execute_order(order_id, sample_market_data)
        
        # Limit orders may or may not execute depending on market conditions
        assert result.order_id == order_id
        if result.success:
            assert result.price <= 150.0  # Should not exceed limit price
    
    def test_nan_prices_handling(self, simulator):
        """Test handling of NaN prices in market data."""
        # Create market data with NaN values
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        market_data_with_nan = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'high': [110, 111, 112, np.nan, 114, 115, 116, 117, 118, 119],
            'low': [90, 91, 92, 93, np.nan, 95, 96, 97, 98, 99],
            'close': [105, 106, 107, 108, 109, np.nan, 111, 112, 113, 114],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Execute order with NaN data
        result = simulator.execute_order(order_id, market_data_with_nan)
        
        # Should handle NaN gracefully
        assert result.order_id == order_id
        # May fail due to NaN data, but shouldn't crash
    
    def test_no_trades_scenario(self, simulator):
        """Test scenario with no trades."""
        # Try to execute non-existent order
        result = simulator.execute_order("non_existent_order", pd.DataFrame())
        
        assert result.success is False
        assert "not found" in result.failure_reason
    
    def test_fill_failures(self, simulator, sample_market_data):
        """Test fill failure scenarios."""
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Mock low fill probability
        with patch.object(simulator, '_calculate_fill_probability', return_value=0.1):
            result = simulator.execute_order(order_id, sample_market_data)
            
            # With 10% fill probability, most executions should fail
            # This is probabilistic, so we test the structure
            assert result.order_id == order_id
            if not result.success:
                assert "Fill failed" in result.failure_reason
    
    def test_orderbook_generation(self, simulator, sample_market_data):
        """Test orderbook generation."""
        orderbook = simulator._generate_orderbook('AAPL', sample_market_data)
        
        assert orderbook.symbol == 'AAPL'
        assert len(orderbook.bids) == simulator.orderbook_levels
        assert len(orderbook.asks) == simulator.orderbook_levels
        assert orderbook.spread > 0
        assert orderbook.mid_price > 0
        
        # Check bid prices are below ask prices
        for bid, ask in zip(orderbook.bids, orderbook.asks):
            assert bid.price < ask.price
    
    def test_slippage_calculation(self, simulator, sample_market_data, sample_orderbook):
        """Test slippage calculation."""
        order = ExecutionOrder(
            order_id="test_order",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=1000,
            timestamp=datetime.now()
        )
        
        slippage = simulator._calculate_slippage(order, sample_market_data, sample_orderbook)
        
        assert slippage >= 0
        assert slippage <= 0.01  # Should be capped at 1%
    
    def test_commission_calculation(self, simulator):
        """Test commission calculation."""
        order = ExecutionOrder(
            order_id="test_order",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=100,
            timestamp=datetime.now()
        )
        
        commission = simulator._calculate_commission(order, 150.0)
        
        assert commission >= simulator.min_commission
        assert commission <= simulator.max_commission
    
    def test_market_impact_calculation(self, simulator, sample_market_data, sample_orderbook):
        """Test market impact calculation."""
        order = ExecutionOrder(
            order_id="test_order",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=1000,
            timestamp=datetime.now()
        )
        
        impact = simulator._calculate_market_impact(order, sample_market_data, sample_orderbook)
        
        assert impact >= 0
        assert impact <= 0.05  # Should be capped at 5%
    
    def test_execution_delay_calculation(self, simulator):
        """Test execution delay calculation."""
        order = ExecutionOrder(
            order_id="test_order",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=100,
            timestamp=datetime.now()
        )
        
        delay = simulator._calculate_execution_delay(order)
        
        assert delay >= simulator.min_execution_delay
        assert delay <= simulator.max_execution_delay
    
    def test_bulk_execution(self, simulator, sample_market_data):
        """Test bulk order execution."""
        orders_data = [
            {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'side': 'buy',
                'quantity': 100
            },
            {
                'symbol': 'GOOGL',
                'order_type': OrderType.LIMIT,
                'side': 'sell',
                'quantity': 50,
                'price': 200.0
            }
        ]
        
        # Place orders
        order_ids = []
        for order_data in orders_data:
            order_id = simulator.place_order(**order_data)
            order_ids.append(order_id)
        
        # Execute bulk
        market_data_dict = {
            'AAPL': sample_market_data,
            'GOOGL': sample_market_data
        }
        
        results = simulator.simulate_bulk_execution(orders_data, market_data_dict)
        
        assert len(results) == len(orders_data)
        for result in results:
            assert isinstance(result, ExecutionResult)
    
    def test_execution_summary(self, simulator, sample_market_data):
        """Test execution summary generation."""
        # Place and execute some orders
        for i in range(3):
            order_id = simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100
            )
            simulator.execute_order(order_id, sample_market_data)
        
        # Get summary
        summary = simulator.get_execution_summary()
        
        assert summary['total_executions'] >= 0
        assert 'successful_executions' in summary
        assert 'failed_executions' in summary
        assert 'success_rate' in summary
    
    def test_symbol_specific_summary(self, simulator, sample_market_data):
        """Test symbol-specific execution summary."""
        # Place and execute orders for specific symbol
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        simulator.execute_order(order_id, sample_market_data)
        
        # Get symbol-specific summary
        summary = simulator.get_execution_summary('AAPL')
        
        assert summary['symbol'] == 'AAPL'
        assert summary['total_executions'] >= 0
    
    def test_order_cancellation(self, simulator):
        """Test order cancellation."""
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.LIMIT,
            side='buy',
            quantity=100,
            price=150.0
        )
        
        # Cancel order
        success = simulator.cancel_order(order_id)
        assert success is True
        
        # Check order status
        order = simulator.get_order_status(order_id)
        assert order.status == OrderStatus.CANCELLED
    
    def test_get_pending_orders(self, simulator):
        """Test getting pending orders."""
        # Place some orders
        simulator.place_order('AAPL', OrderType.MARKET, 'buy', 100)
        simulator.place_order('GOOGL', OrderType.LIMIT, 'sell', 50, price=200.0)
        
        pending_orders = simulator.get_pending_orders()
        
        assert len(pending_orders) >= 0
        for order in pending_orders:
            assert order.status == OrderStatus.PENDING
    
    def test_clear_execution_history(self, simulator):
        """Test clearing execution history."""
        # Add some history
        simulator.execution_history.append(Mock())
        simulator.orderbook_history['AAPL'] = [Mock()]
        simulator.daily_volume['AAPL_2024-01-01'] = 1000.0
        
        # Clear history
        simulator.clear_execution_history()
        
        assert len(simulator.execution_history) == 0
        assert len(simulator.orderbook_history) == 0
        assert len(simulator.daily_volume) == 0
    
    def test_real_world_orderbook_emulation(self, simulator):
        """Test real-world orderbook emulation with historical bid/ask spread."""
        # Create realistic market data with volume and price information
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        realistic_market_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(150, 250, 50),
            'low': np.random.uniform(50, 150, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000000, 10000000, 50)  # High volume
        }, index=dates)
        
        # Generate orderbook
        orderbook = simulator._generate_orderbook('AAPL', realistic_market_data)
        
        # Test realistic properties
        assert orderbook.spread > 0
        assert orderbook.spread < 0.01  # Should be reasonable spread
        
        # Test volume decay
        for i in range(1, len(orderbook.bids)):
            assert orderbook.bids[i].quantity <= orderbook.bids[i-1].quantity
            assert orderbook.asks[i].quantity <= orderbook.asks[i-1].quantity
        
        # Test price progression
        for i in range(1, len(orderbook.bids)):
            assert orderbook.bids[i].price < orderbook.bids[i-1].price
            assert orderbook.asks[i].price > orderbook.asks[i-1].price


if __name__ == "__main__":
    pytest.main([__file__]) 