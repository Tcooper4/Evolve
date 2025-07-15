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

from trading.execution.trade_execution_simulator import (
    TradeExecutionSimulator,
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
        return TradeExecutionSimulator(config)
    
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
        if result.success:
            assert result.price <= 150.0  # Should not exceed limit price
        else:
            assert result.failure_reason is not None
    
    def test_execute_stop_order(self, simulator, sample_market_data):
        """Test stop order execution."""
        # Place stop order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.STOP,
            side='sell',
            quantity=100,
            price=140.0  # Stop price
        )
        
        # Execute order
        result = simulator.execute_order(order_id, sample_market_data)
        
        # Stop orders may or may not execute depending on market conditions
        if result.success:
            assert result.price >= 140.0  # Should not be below stop price
        else:
            assert result.failure_reason is not None
    
    def test_execute_stop_limit_order(self, simulator, sample_market_data):
        """Test stop-limit order execution."""
        # Place stop-limit order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.STOP_LIMIT,
            side='sell',
            quantity=100,
            price=140.0,  # Stop price
            limit_price=139.0  # Limit price
        )
        
        # Execute order
        result = simulator.execute_order(order_id, sample_market_data)
        
        # Stop-limit orders have complex execution logic
        if result.success:
            assert result.price >= 139.0  # Should not be below limit price
        else:
            assert result.failure_reason is not None
    
    def test_nan_prices_handling(self, simulator):
        """Test handling of NaN prices in market data."""
        # Create market data with NaN values
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        market_data = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'high': [110, 111, 112, np.nan, 114, 115, 116, 117, 118, 119],
            'low': [90, 91, 92, 93, np.nan, 95, 96, 97, 98, 99],
            'close': [105, 106, 107, 108, 109, np.nan, 111, 112, 113, 114],
            'volume': [1000] * 10
        }, index=dates)
        
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Execute order - should handle NaN values gracefully
        result = simulator.execute_order(order_id, market_data)
        
        # Should either succeed with valid price or fail with clear reason
        if result.success:
            assert not np.isnan(result.price)
        else:
            assert result.failure_reason is not None
    
    def test_no_trades_scenario(self, simulator):
        """Test scenario with no trades in market data."""
        # Create market data with no volume
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        market_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [0] * 10  # No volume
        }, index=dates)
        
        # Place order
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        # Execute order
        result = simulator.execute_order(order_id, market_data)
        
        # Should handle no volume scenario
        assert result.success is False
        assert "volume" in result.failure_reason.lower() or "trade" in result.failure_reason.lower()
    
    def test_fill_failures(self, simulator, sample_market_data):
        """Test various fill failure scenarios."""
        # Test with very low fill rate
        simulator.base_fill_rate = 0.01  # 1% fill rate
        
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=100
        )
        
        result = simulator.execute_order(order_id, sample_market_data)
        
        # With very low fill rate, should likely fail
        if not result.success:
            assert "fill" in result.failure_reason.lower() or "execution" in result.failure_reason.lower()
    
    def test_orderbook_generation(self, simulator, sample_market_data):
        """Test orderbook generation from market data."""
        orderbook = simulator._generate_orderbook(sample_market_data, 'AAPL')
        
        assert orderbook is not None
        assert orderbook.symbol == 'AAPL'
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        assert orderbook.spread > 0
        assert orderbook.mid_price > 0
    
    def test_slippage_calculation(self, simulator, sample_market_data, sample_orderbook):
        """Test slippage calculation."""
        # Place a large order to trigger slippage
        order_id = simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='buy',
            quantity=5000  # Large order
        )
        
        result = simulator.execute_order(order_id, sample_market_data, sample_orderbook)
        
        if result.success:
            assert result.slippage >= 0
            # Large orders should have some slippage
            assert result.slippage > 0
    
    def test_commission_calculation(self, simulator):
        """Test commission calculation."""
        # Test commission calculation directly
        commission = simulator._calculate_commission(1000, 150.0)
        
        assert commission >= 0
        assert commission >= simulator.base_commission  # Should be at least base commission
    
    def test_market_impact_calculation(self, simulator, sample_market_data, sample_orderbook):
        """Test market impact calculation."""
        # Test with different order sizes
        small_impact = simulator._calculate_market_impact(100, sample_orderbook)
        large_impact = simulator._calculate_market_impact(5000, sample_orderbook)
        
        assert small_impact >= 0
        assert large_impact >= 0
        assert large_impact >= small_impact  # Larger orders should have more impact
    
    def test_execution_delay_calculation(self, simulator):
        """Test execution delay calculation."""
        delay = simulator._calculate_execution_delay()
        
        assert delay >= 0
        assert delay <= simulator.max_execution_delay
    
    def test_bulk_execution(self, simulator, sample_market_data):
        """Test bulk order execution."""
        # Place multiple orders
        order_ids = []
        for i in range(5):
            order_id = simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100
            )
            order_ids.append(order_id)
        
        # Execute all orders
        results = []
        for order_id in order_ids:
            result = simulator.execute_order(order_id, sample_market_data)
            results.append(result)
        
        # Check results
        assert len(results) == 5
        for result in results:
            assert result.success is True or result.success is False
            if result.success:
                assert result.quantity == 100
                assert result.price > 0
    
    def test_execution_summary(self, simulator, sample_market_data):
        """Test execution summary generation."""
        # Execute some orders first
        for i in range(3):
            order_id = simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100
            )
            simulator.execute_order(order_id, sample_market_data)
        
        # Generate summary
        summary = simulator.get_execution_summary()
        
        assert summary is not None
        assert 'total_orders' in summary
        assert 'successful_orders' in summary
        assert 'failed_orders' in summary
        assert 'total_volume' in summary
        assert 'total_commission' in summary
        assert 'total_slippage' in summary
    
    def test_symbol_specific_summary(self, simulator, sample_market_data):
        """Test symbol-specific execution summary."""
        # Execute orders for different symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        for symbol in symbols:
            order_id = simulator.place_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100
            )
            simulator.execute_order(order_id, sample_market_data)
        
        # Get symbol-specific summary
        summary = simulator.get_symbol_summary('AAPL')
        
        assert summary is not None
        assert summary['symbol'] == 'AAPL'
        assert 'total_orders' in summary
        assert 'total_volume' in summary
    
    def test_order_cancellation(self, simulator):
        """Test order cancellation."""
        # Place order
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
        assert simulator.orders[order_id].status == OrderStatus.CANCELLED
    
    def test_get_pending_orders(self, simulator):
        """Test getting pending orders."""
        # Place some orders
        simulator.place_order(
            symbol='AAPL',
            order_type=OrderType.LIMIT,
            side='buy',
            quantity=100,
            price=150.0
        )
        
        simulator.place_order(
            symbol='GOOGL',
            order_type=OrderType.LIMIT,
            side='sell',
            quantity=50,
            price=200.0
        )
        
        # Get pending orders
        pending = simulator.get_pending_orders()
        
        assert len(pending) == 2
        for order in pending:
            assert order.status == OrderStatus.PENDING
    
    def test_clear_execution_history(self, simulator, sample_market_data):
        """Test clearing execution history."""
        # Execute some orders first
        for i in range(3):
            order_id = simulator.place_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                side='buy',
                quantity=100
            )
            simulator.execute_order(order_id, sample_market_data)
        
        # Clear history
        simulator.clear_execution_history()
        
        assert len(simulator.execution_history) == 0
    
    def test_real_world_orderbook_emulation(self, simulator):
        """Test realistic orderbook emulation."""
        # Create realistic market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        market_data = pd.DataFrame({
            'open': np.random.normal(150, 10, 100),
            'high': np.random.normal(155, 10, 100),
            'low': np.random.normal(145, 10, 100),
            'close': np.random.normal(150, 10, 100),
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Generate orderbook
        orderbook = simulator._generate_orderbook(market_data, 'AAPL')
        
        # Validate orderbook structure
        assert orderbook.symbol == 'AAPL'
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        assert orderbook.spread > 0
        assert orderbook.mid_price > 0
        
        # Validate bid/ask structure
        for i in range(len(orderbook.bids) - 1):
            assert orderbook.bids[i].price > orderbook.bids[i + 1].price  # Bids in descending order
        
        for i in range(len(orderbook.asks) - 1):
            assert orderbook.asks[i].price < orderbook.asks[i + 1].price  # Asks in ascending order
        
        # Validate spread
        assert orderbook.asks[0].price > orderbook.bids[0].price  # Spread should be positive 