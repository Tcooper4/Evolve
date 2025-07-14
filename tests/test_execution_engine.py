"""
Tests for Execution Engine

Tests both execution/execution_agent.py and execution/broker_adapter.py modules.
"""

import pytest
import json
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from execution.execution_agent import (
    ExecutionAgent,
    ExecutionMode,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderRequest,
    OrderExecution,
    MarketData,
    Position,
    create_execution_agent,
    submit_order
)
from execution.broker_adapter import (
    BrokerAdapter,
    BrokerType,
    BaseBrokerAdapter,
    AlpacaBrokerAdapter,
    IBKRBrokerAdapter,
    PolygonBrokerAdapter,
    SimulationBrokerAdapter,
    create_broker_adapter,
    test_broker_connection
)


class TestExecutionMode:
    """Test ExecutionMode enum"""
    
    def test_execution_mode_values(self):
        """Test ExecutionMode enum values"""
        assert ExecutionMode.SIMULATION.value == "simulation"
        assert ExecutionMode.LIVE.value == "live"
        assert ExecutionMode.PAPER.value == "paper"


class TestOrderType:
    """Test OrderType enum"""
    
    def test_order_type_values(self):
        """Test OrderType enum values"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"


class TestOrderSide:
    """Test OrderSide enum"""
    
    def test_order_side_values(self):
        """Test OrderSide enum values"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderStatus:
    """Test OrderStatus enum"""
    
    def test_order_status_values(self):
        """Test OrderStatus enum values"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestOrderRequest:
    """Test OrderRequest dataclass"""
    
    def test_order_request_creation(self):
        """Test creating an OrderRequest instance"""
        request = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=150.0,
            stop_price=None,
            time_in_force="day",
            client_order_id="client_123"
        )
        
        assert request.order_id == "test_order"
        assert request.ticker == "AAPL"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.MARKET
        assert request.quantity == 100
        assert request.price == 150.0


class TestOrderExecution:
    """Test OrderExecution dataclass"""
    
    def test_order_execution_creation(self):
        """Test creating an OrderExecution instance"""
        execution = OrderExecution(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=150.0,
            executed_quantity=100,
            average_price=150.5,
            commission=1.0,
            timestamp=datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fills=[{
                'fill_id': 'fill_123',
                'quantity': 100,
                'price': 150.5,
                'timestamp': datetime.now().isoformat(),
                'commission': 1.0
            }],
            metadata={'slippage_bps': 5}
        )
        
        assert execution.order_id == "test_order"
        assert execution.ticker == "AAPL"
        assert execution.executed_quantity == 100
        assert execution.average_price == 150.5
        assert execution.status == OrderStatus.FILLED


class TestMarketData:
    """Test MarketData dataclass"""
    
    def test_market_data_creation(self):
        """Test creating a MarketData instance"""
        market_data = MarketData(
            ticker="AAPL",
            bid=150.0,
            ask=150.1,
            last=150.05,
            volume=1000000,
            timestamp=datetime.now().isoformat(),
            spread=0.1,
            volatility=0.02
        )
        
        assert market_data.ticker == "AAPL"
        assert market_data.bid == 150.0
        assert market_data.ask == 150.1
        assert market_data.spread == 0.1


class TestPosition:
    """Test Position dataclass"""
    
    def test_position_creation(self):
        """Test creating a Position instance"""
        position = Position(
            ticker="AAPL",
            quantity=100,
            average_price=150.0,
            market_value=15000.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
            timestamp=datetime.now().isoformat()
        )
        
        assert position.ticker == "AAPL"
        assert position.quantity == 100
        assert position.average_price == 150.0
        assert position.market_value == 15000.0


class TestExecutionAgent:
    """Test ExecutionAgent functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def execution_agent(self, temp_dir):
        """Create ExecutionAgent instance for testing"""
        config = {
            'execution': {
                'mode': 'simulation',
                'spread_multiplier': 1.0,
                'slippage_bps': 5,
                'execution_delay_ms': 100,
                'commission_rate': 0.001,
                'min_commission': 1.0,
                'max_position_size': 0.1,
                'max_order_size': 10000,
                'max_daily_trades': 100
            }
        }
        
        with patch('execution.execution_agent.load_config', return_value=config):
            agent = ExecutionAgent()
            return agent
    
    def test_execution_agent_initialization(self, execution_agent):
        """Test execution agent initialization"""
        assert execution_agent.execution_mode == ExecutionMode.SIMULATION
        assert execution_agent.spread_multiplier == 1.0
        assert execution_agent.slippage_basis_points == 5
        assert execution_agent.execution_delay_ms == 100
        assert execution_agent.commission_rate == 0.001
        assert execution_agent.max_position_size == 0.1
    
    def test_get_current_market_data(self, execution_agent):
        """Test market data retrieval"""
        market_data = execution_agent._get_current_market_data("AAPL")
        
        assert isinstance(market_data, MarketData)
        assert market_data.ticker == "AAPL"
        assert market_data.bid > 0
        assert market_data.ask > 0
        assert market_data.spread > 0
    
    def test_simulate_order_execution(self, execution_agent):
        """Test order execution simulation"""
        order = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        execution = execution_agent._simulate_order_execution(order)
        
        assert isinstance(execution, OrderExecution)
        assert execution.order_id == "test_order"
        assert execution.ticker == "AAPL"
        assert execution.executed_quantity == 100
        assert execution.status == OrderStatus.FILLED
        assert execution.average_price > 0
        assert execution.commission >= 0
    
    def test_check_risk_limits(self, execution_agent):
        """Test risk limit checking"""
        order = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Test valid order
        risk_ok, message = execution_agent._check_risk_limits(order)
        assert risk_ok is True
        assert "passed" in message
        
        # Test order size limit
        large_order = OrderRequest(
            order_id="large_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000000  # Very large order
        )
        
        risk_ok, message = execution_agent._check_risk_limits(large_order)
        assert risk_ok is False
        assert "size limit" in message
    
    def test_update_position(self, execution_agent):
        """Test position updating"""
        # Create execution for buying
        execution = OrderExecution(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=150.0,
            executed_quantity=100,
            average_price=150.5,
            commission=1.0,
            timestamp=datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fills=[],
            metadata={}
        )
        
        # Update position
        execution_agent._update_position(execution)
        
        # Check position
        position = execution_agent.positions.get("AAPL")
        assert position is not None
        assert position.quantity == 100
        assert position.average_price == 150.5
        
        # Test selling
        sell_execution = OrderExecution(
            order_id="sell_order",
            ticker="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50,
            price=160.0,
            executed_quantity=50,
            average_price=160.0,
            commission=1.0,
            timestamp=datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fills=[],
            metadata={}
        )
        
        execution_agent._update_position(sell_execution)
        
        # Check updated position
        position = execution_agent.positions.get("AAPL")
        assert position.quantity == 50
        assert position.realized_pnl > 0  # Should have realized profit
    
    @pytest.mark.asyncio
    async def test_submit_order(self, execution_agent):
        """Test order submission"""
        # Start agent
        await execution_agent.start()
        
        # Submit order
        order_id = await execution_agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order_id is not None
        
        # Wait for execution
        await asyncio.sleep(0.5)
        
        # Check order status
        execution = execution_agent.get_order_status(order_id)
        assert execution is not None
        assert execution.status == OrderStatus.FILLED
        
        # Stop agent
        await execution_agent.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, execution_agent):
        """Test order cancellation"""
        # Submit order
        order_id = await execution_agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        # Cancel order
        cancelled = execution_agent.cancel_order(order_id)
        assert cancelled is True
    
    def test_get_performance_metrics(self, execution_agent):
        """Test performance metrics"""
        metrics = execution_agent.get_performance_metrics()
        
        assert 'daily_trades' in metrics
        assert 'daily_volume' in metrics
        assert 'daily_commission' in metrics
        assert 'total_positions' in metrics
        assert 'total_pnl' in metrics
        assert 'execution_mode' in metrics
    
    def test_set_execution_parameters(self, execution_agent):
        """Test execution parameter updates"""
        execution_agent.set_execution_parameters(
            spread_multiplier=1.5,
            slippage_bps=10,
            execution_delay_ms=200
        )
        
        assert execution_agent.spread_multiplier == 1.5
        assert execution_agent.slippage_basis_points == 10
        assert execution_agent.execution_delay_ms == 200


class TestBrokerType:
    """Test BrokerType enum"""
    
    def test_broker_type_values(self):
        """Test BrokerType enum values"""
        assert BrokerType.ALPACA.value == "alpaca"
        assert BrokerType.IBKR.value == "ibkr"
        assert BrokerType.POLYGON.value == "polygon"
        assert BrokerType.SIMULATION.value == "simulation"


class TestBaseBrokerAdapter:
    """Test BaseBrokerAdapter abstract class"""
    
    def test_rate_limit_checking(self):
        """Test rate limit checking"""
        config = {'test': 'config'}
        
        # Create mock adapter
        adapter = Mock(spec=BaseBrokerAdapter)
        adapter.config = config
        adapter.rate_limits = {
            'test_endpoint': {'max_requests': 2, 'window': 60}
        }
        adapter.last_request_time = {}
        adapter.logger = Mock()
        
        # Test rate limit checking
        adapter._check_rate_limit = BaseBrokerAdapter._check_rate_limit.__get__(adapter)
        
        # First request should pass
        assert adapter._check_rate_limit('test_endpoint') is True
        
        # Second request should pass
        assert adapter._check_rate_limit('test_endpoint') is True
        
        # Third request should fail
        assert adapter._check_rate_limit('test_endpoint') is False


class TestSimulationBrokerAdapter:
    """Test SimulationBrokerAdapter"""
    
    @pytest.fixture
    def simulation_adapter(self):
        """Create SimulationBrokerAdapter instance"""
        config = {'simulation': True}
        return SimulationBrokerAdapter(config)
    
    @pytest.mark.asyncio
    async def test_simulation_adapter_connection(self, simulation_adapter):
        """Test simulation adapter connection"""
        connected = await simulation_adapter.connect()
        assert connected is True
        assert simulation_adapter.is_connected is True
        
        await simulation_adapter.disconnect()
        assert simulation_adapter.is_connected is False
    
    @pytest.mark.asyncio
    async def test_simulation_order_submission(self, simulation_adapter):
        """Test simulation order submission"""
        await simulation_adapter.connect()
        
        order = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        execution = await simulation_adapter.submit_order(order)
        
        assert isinstance(execution, OrderExecution)
        assert execution.order_id == "test_order"
        assert execution.status == OrderStatus.FILLED
        assert execution.executed_quantity == 100
        
        await simulation_adapter.disconnect()
    
    @pytest.mark.asyncio
    async def test_simulation_market_data(self, simulation_adapter):
        """Test simulation market data"""
        await simulation_adapter.connect()
        
        market_data = await simulation_adapter.get_market_data("AAPL")
        
        assert isinstance(market_data, MarketData)
        assert market_data.ticker == "AAPL"
        assert market_data.bid > 0
        assert market_data.ask > 0
        assert market_data.spread > 0
        
        await simulation_adapter.disconnect()
    
    @pytest.mark.asyncio
    async def test_simulation_account_info(self, simulation_adapter):
        """Test simulation account info"""
        await simulation_adapter.connect()
        
        account_info = await simulation_adapter.get_account_info()
        
        assert account_info.account_id == "SIM_ACCOUNT"
        assert account_info.cash == 100000.0
        assert account_info.buying_power == 100000.0
        
        await simulation_adapter.disconnect()


class TestBrokerAdapter:
    """Test unified BrokerAdapter"""
    
    @pytest.mark.asyncio
    async def test_simulation_broker_adapter(self):
        """Test simulation broker adapter"""
        adapter = create_broker_adapter("simulation")
        
        # Test connection
        connected = await adapter.connect()
        assert connected is True
        
        # Test market data
        market_data = await adapter.get_market_data("AAPL")
        assert isinstance(market_data, MarketData)
        
        # Test order submission
        order = OrderRequest(
            order_id="test_order",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        execution = await adapter.submit_order(order)
        assert isinstance(execution, OrderExecution)
        
        await adapter.disconnect()
    
    @pytest.mark.asyncio
    async def test_invalid_broker_type(self):
        """Test invalid broker type"""
        with pytest.raises(ValueError):
            create_broker_adapter("invalid_broker")
    
    @pytest.mark.asyncio
    async def test_broker_connection_test(self):
        """Test broker connection testing"""
        # Test simulation broker
        connected = await test_broker_connection("simulation")
        assert connected is True
        
        # Test invalid broker
        connected = await test_broker_connection("invalid_broker")
        assert connected is False


class TestExecutionIntegration:
    """Integration tests for execution engine"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_execution_workflow(self, temp_dir):
        """Test complete execution workflow"""
        # Create execution agent
        config = {
            'execution': {
                'mode': 'simulation',
                'spread_multiplier': 1.0,
                'slippage_bps': 5,
                'execution_delay_ms': 50,
                'commission_rate': 0.001,
                'min_commission': 1.0
            }
        }
        
        with patch('execution.execution_agent.load_config', return_value=config):
            agent = ExecutionAgent()
        
        # Start agent
        await agent.start()
        
        # Submit multiple orders
        order_ids = []
        
        # Buy order
        buy_order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        order_ids.append(buy_order_id)
        
        # Limit sell order
        sell_order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=160.0
        )
        order_ids.append(sell_order_id)
        
        # Wait for execution
        await asyncio.sleep(1)
        
        # Check order statuses
        for order_id in order_ids:
            execution = agent.get_order_status(order_id)
            assert execution is not None
            assert execution.status == OrderStatus.FILLED
        
        # Check positions
        positions = agent.get_all_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 50  # 100 bought - 50 sold
        
        # Check performance metrics
        metrics = agent.get_performance_metrics()
        assert metrics['daily_trades'] >= 2
        assert metrics['total_positions'] >= 1
        
        # Stop agent
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, temp_dir):
        """Test risk management integration"""
        # Create execution agent with strict limits
        config = {
            'execution': {
                'mode': 'simulation',
                'max_order_size': 1000,  # Small limit
                'max_daily_trades': 5,
                'max_position_size': 0.05
            }
        }
        
        with patch('execution.execution_agent.load_config', return_value=config):
            agent = ExecutionAgent()
        
        await agent.start()
        
        # Test order size limit
        with pytest.raises(ValueError):
            await agent.submit_order(
                ticker="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10000  # Exceeds limit
            )
        
        # Test valid order
        order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        await asyncio.sleep(0.5)
        
        execution = agent.get_order_status(order_id)
        assert execution.status == OrderStatus.FILLED
        
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_broker_adapter_integration(self, temp_dir):
        """Test broker adapter integration"""
        # Create execution agent with broker adapter
        config = {
            'execution': {
                'mode': 'simulation',
                'broker': {
                    'type': 'simulation'
                }
            }
        }
        
        with patch('execution.execution_agent.load_config', return_value=config):
            agent = ExecutionAgent()
        
        # Verify broker adapter is initialized
        assert agent.broker_adapter is not None
        
        await agent.start()
        
        # Submit order through broker
        order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        await asyncio.sleep(0.5)
        
        # Check execution
        execution = agent.get_order_status(order_id)
        assert execution.status == OrderStatus.FILLED
        
        await agent.stop()


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_execution_agent(self):
        """Test create_execution_agent function"""
        with patch('execution.execution_agent.ExecutionAgent') as mock_agent:
            mock_instance = Mock()
            mock_agent.return_value = mock_instance
            
            agent = create_execution_agent()
            assert agent == mock_instance
    
    @pytest.mark.asyncio
    async def test_submit_order_function(self):
        """Test submit_order function"""
        mock_agent = Mock()
        mock_agent.submit_order = AsyncMock(return_value="order_123")
        
        order_id = await submit_order(
            agent=mock_agent,
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert order_id == "order_123"
        mock_agent.submit_order.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 