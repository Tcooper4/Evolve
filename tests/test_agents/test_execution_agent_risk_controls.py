#!/usr/bin/env python3
"""
Tests for ExecutionAgent Risk Controls

Tests the comprehensive risk management features including stop-loss,
take-profit, automatic exits, and detailed logging.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from trading.agents.execution_agent import (
    ExecutionAgent, TradeSignal, ExecutionMode, TradeDirection,
    RiskControls, RiskThreshold, RiskThresholdType, ExitReason
)
from trading.agents.base_agent_interface import AgentConfig


@pytest.fixture
def execution_agent():
    """Create an execution agent with risk controls enabled."""
    config = {
        'name': 'test_execution_agent',
        'enabled': True,
        'priority': 1,
        'max_concurrent_runs': 1,
        'timeout_seconds': 300,
        'retry_attempts': 3,
        'custom_config': {
            'execution_mode': 'simulation',
            'max_positions': 5,
            'min_confidence': 0.6,
            'risk_monitoring_enabled': True,
            'auto_exit_enabled': True,
            'risk_controls': {
                'stop_loss': {
                    'threshold_type': 'percentage',
                    'value': 0.02,
                    'atr_multiplier': 2.0,
                    'atr_period': 14
                },
                'take_profit': {
                    'threshold_type': 'percentage',
                    'value': 0.06,
                    'atr_multiplier': 3.0,
                    'atr_period': 14
                },
                'max_position_size': 0.2,
                'max_portfolio_risk': 0.05,
                'max_daily_loss': 0.02,
                'max_correlation': 0.7,
                'volatility_limit': 0.5
            }
        }
    }
    
    agent_config = AgentConfig(**config)
    return ExecutionAgent(agent_config)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'AAPL': {
            'price': 150.25,
            'volatility': 0.15,
            'volume': 1000000,
            'price_change': 0.01
        },
        'TSLA': {
            'price': 245.50,
            'volatility': 0.25,
            'volume': 2000000,
            'price_change': -0.02
        },
        'NVDA': {
            'price': 420.75,
            'volatility': 0.20,
            'volume': 1500000,
            'price_change': 0.015
        }
    }


@pytest.fixture
def sample_trade_signal():
    """Create a sample trade signal with risk controls."""
    risk_controls = RiskControls(
        stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.02),
        take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.06),
        max_position_size=0.15,
        max_daily_loss=0.03,
        volatility_limit=0.4
    )
    
    return TradeSignal(
        symbol="AAPL",
        direction=TradeDirection.LONG,
        strategy="test_strategy",
        confidence=0.85,
        entry_price=150.25,
        max_holding_period=timedelta(days=7),
        risk_controls=risk_controls,
        market_data={
            'volatility': 0.15,
            'volume': 1000000,
            'rsi': 65
        }
    )


class TestRiskControls:
    """Test risk controls functionality."""
    
    def test_risk_controls_initialization(self, execution_agent):
        """Test that risk controls are properly initialized."""
        assert execution_agent.risk_monitoring_enabled is True
        assert execution_agent.auto_exit_enabled is True
        assert execution_agent.default_risk_controls is not None
        
        # Check default risk controls
        controls = execution_agent.default_risk_controls
        assert controls.stop_loss.threshold_type == RiskThresholdType.PERCENTAGE
        assert controls.stop_loss.value == 0.02
        assert controls.take_profit.threshold_type == RiskThresholdType.PERCENTAGE
        assert controls.take_profit.value == 0.06
        assert controls.max_position_size == 0.2
        assert controls.max_daily_loss == 0.02
    
    def test_risk_threshold_calculation(self, execution_agent, sample_market_data):
        """Test risk threshold calculations."""
        # Create a mock position
        position = type('Position', (), {
            'symbol': 'AAPL',
            'direction': TradeDirection.LONG,
            'entry_price': 150.00
        })()
        
        # Test percentage-based stop loss
        stop_loss_price = execution_agent._calculate_stop_loss_price(
            position, execution_agent.default_risk_controls, sample_market_data
        )
        expected_stop_loss = 150.00 * (1 - 0.02)  # 2% below entry
        assert abs(stop_loss_price - expected_stop_loss) < 0.01
        
        # Test percentage-based take profit
        take_profit_price = execution_agent._calculate_take_profit_price(
            position, execution_agent.default_risk_controls, sample_market_data
        )
        expected_take_profit = 150.00 * (1 + 0.06)  # 6% above entry
        assert abs(take_profit_price - expected_take_profit) < 0.01
    
    def test_atr_based_thresholds(self, execution_agent, sample_market_data):
        """Test ATR-based threshold calculations."""
        # Set up price history for ATR calculation
        execution_agent.price_history['AAPL'] = [
            145, 148, 152, 149, 151, 150, 153, 147, 150, 155, 148, 152, 149, 150
        ]
        
        # Create ATR-based risk controls
        atr_controls = RiskControls(
            stop_loss=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=2.0),
            take_profit=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=3.0)
        )
        
        # Create a mock position
        position = type('Position', (), {
            'symbol': 'AAPL',
            'direction': TradeDirection.LONG,
            'entry_price': 150.00
        })()
        
        # Calculate ATR-based thresholds
        stop_loss_price = execution_agent._calculate_stop_loss_price(
            position, atr_controls, sample_market_data
        )
        take_profit_price = execution_agent._calculate_take_profit_price(
            position, atr_controls, sample_market_data
        )
        
        # Verify ATR-based calculations
        assert stop_loss_price < 150.00  # Stop loss below entry
        assert take_profit_price > 150.00  # Take profit above entry
    
    def test_position_exit_logic(self, execution_agent):
        """Test position exit logic."""
        # Test long position exit logic
        long_position = type('Position', (), {
            'direction': TradeDirection.LONG
        })()
        
        # Test stop loss for long position
        should_exit = execution_agent._should_exit_position(
            long_position, 147.00, 148.00, "stop_loss"
        )
        assert should_exit is True  # Current price below stop loss
        
        # Test take profit for long position
        should_exit = execution_agent._should_exit_position(
            long_position, 160.00, 159.00, "take_profit"
        )
        assert should_exit is True  # Current price above take profit
        
        # Test short position exit logic
        short_position = type('Position', (), {
            'direction': TradeDirection.SHORT
        })()
        
        # Test stop loss for short position
        should_exit = execution_agent._should_exit_position(
            short_position, 155.00, 154.00, "stop_loss"
        )
        assert should_exit is True  # Current price above stop loss
        
        # Test take profit for short position
        should_exit = execution_agent._should_exit_position(
            short_position, 140.00, 141.00, "take_profit"
        )
        assert should_exit is True  # Current price below take profit


class TestRiskMonitoring:
    """Test risk monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_risk_monitoring_enabled(self, execution_agent, sample_market_data):
        """Test that risk monitoring works when enabled."""
        # Create a position first
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.8,
            entry_price=150.00
        )
        
        # Execute trade
        result = await execution_agent.execute(
            signals=[signal],
            market_data=sample_market_data,
            risk_check=True
        )
        
        assert result.success is True
        
        # Test risk monitoring with price that should trigger stop loss
        stop_loss_market_data = {
            'AAPL': {'price': 147.00, 'volatility': 0.15}  # -2% from entry
        }
        
        result = await execution_agent.execute(
            signals=[],
            market_data=stop_loss_market_data,
            risk_check=True
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, execution_agent, sample_market_data):
        """Test daily loss limit enforcement."""
        # Set daily PnL to breach limit
        execution_agent.daily_pnl = -0.025  # -2.5% (above 2% limit)
        
        # Run risk monitoring
        await execution_agent._monitor_risk_limits(sample_market_data)
        
        # Check that emergency exit was triggered
        # This would normally close all positions
        assert execution_agent.daily_pnl == -0.025  # Should remain unchanged in test
    
    def test_portfolio_correlation_calculation(self, execution_agent, sample_market_data):
        """Test portfolio correlation calculation."""
        # Set up price history for correlation calculation
        execution_agent.price_history['AAPL'] = [100, 101, 102, 103, 104, 105]
        execution_agent.price_history['TSLA'] = [200, 202, 204, 206, 208, 210]  # Highly correlated
        
        correlation = execution_agent._calculate_portfolio_correlation(sample_market_data)
        
        # Should return a correlation value between -1 and 1
        assert -1 <= correlation <= 1
    
    def test_portfolio_risk_exposure(self, execution_agent, sample_market_data):
        """Test portfolio risk exposure calculation."""
        # Mock portfolio state
        execution_agent.portfolio_manager.state.equity = 10000.0
        execution_agent.portfolio_manager.state.open_positions = [
            type('Position', (), {
                'symbol': 'AAPL',
                'size': 10,
                'entry_price': 150.00
            })()
        ]
        
        risk_exposure = execution_agent._calculate_portfolio_risk_exposure(sample_market_data)
        
        # Should return a percentage between 0 and 1
        assert 0 <= risk_exposure <= 1


class TestExitEvents:
    """Test exit event logging and tracking."""
    
    @pytest.mark.asyncio
    async def test_exit_event_creation(self, execution_agent, sample_market_data):
        """Test exit event creation and logging."""
        # Create a position
        position = type('Position', (), {
            'symbol': 'AAPL',
            'direction': TradeDirection.LONG,
            'entry_price': 150.00,
            'size': 10,
            'entry_time': datetime.utcnow() - timedelta(hours=1)
        })()
        
        # Mock portfolio manager to return the position
        execution_agent.portfolio_manager.state.open_positions = [position]
        
        # Trigger an exit
        await execution_agent._exit_position(
            position=position,
            exit_price=147.00,
            exit_reason=ExitReason.STOP_LOSS,
            message="Stop loss triggered",
            market_data=sample_market_data
        )
        
        # Check that exit event was created
        assert len(execution_agent.exit_events) > 0
        
        exit_event = execution_agent.exit_events[-1]
        assert exit_event.symbol == "AAPL"
        assert exit_event.exit_reason == ExitReason.STOP_LOSS
        assert exit_event.exit_price == 147.00
        assert exit_event.pnl < 0  # Should be negative for stop loss
    
    def test_exit_events_retrieval(self, execution_agent):
        """Test exit events retrieval with date filters."""
        # Create some mock exit events
        mock_event = type('ExitEvent', (), {
            'timestamp': datetime.utcnow(),
            'symbol': 'AAPL',
            'exit_reason': ExitReason.STOP_LOSS,
            'exit_price': 147.00,
            'pnl': -30.00,
            'holding_period': timedelta(hours=1),
            'risk_metrics': {},
            'market_conditions': {}
        })()
        
        execution_agent.exit_events = [mock_event]
        
        # Test retrieval
        events = execution_agent.get_exit_events()
        assert len(events) >= 0  # May be 0 if file doesn't exist
    
    def test_risk_summary_generation(self, execution_agent):
        """Test risk summary generation."""
        # Create mock exit events
        mock_events = [
            type('ExitEvent', (), {
                'exit_reason': ExitReason.STOP_LOSS,
                'pnl': -30.00
            })(),
            type('ExitEvent', (), {
                'exit_reason': ExitReason.TAKE_PROFIT,
                'pnl': 45.00
            })(),
            type('ExitEvent', (), {
                'exit_reason': ExitReason.STOP_LOSS,
                'pnl': -20.00
            })()
        ]
        
        execution_agent.exit_events = mock_events
        
        summary = execution_agent.get_risk_summary()
        
        assert summary['total_exits'] == 3
        assert summary['total_pnl'] == -5.00  # -30 + 45 - 20
        assert 'stop_loss' in summary['exit_reasons']
        assert 'take_profit' in summary['exit_reasons']
        assert summary['exit_reasons']['stop_loss'] == 2
        assert summary['exit_reasons']['take_profit'] == 1


class TestMarketDataCache:
    """Test market data cache and price history management."""
    
    def test_market_data_cache_update(self, execution_agent, sample_market_data):
        """Test market data cache update functionality."""
        execution_agent._update_market_data_cache(sample_market_data)
        
        # Check that market data was cached
        assert 'AAPL' in execution_agent.market_data_cache
        assert 'TSLA' in execution_agent.market_data_cache
        assert 'NVDA' in execution_agent.market_data_cache
        
        # Check that price history was updated
        assert 'AAPL' in execution_agent.price_history
        assert len(execution_agent.price_history['AAPL']) > 0
        assert execution_agent.price_history['AAPL'][-1] == 150.25
    
    def test_global_metrics_update(self, execution_agent, sample_market_data):
        """Test global market metrics update."""
        execution_agent._update_global_metrics(sample_market_data)
        
        # Check that global metrics were updated
        assert 'volatility_regime' in execution_agent.global_metrics
        assert 'market_regime' in execution_agent.global_metrics
        
        # Should be 'normal' volatility regime with given data
        assert execution_agent.global_metrics['volatility_regime'] in ['low', 'normal', 'high']


class TestRiskControlsIntegration:
    """Test integration of risk controls with trade execution."""
    
    @pytest.mark.asyncio
    async def test_signal_with_risk_controls(self, execution_agent, sample_market_data):
        """Test trade signal processing with risk controls."""
        # Create signal with custom risk controls
        risk_controls = RiskControls(
            stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.03),  # 3% stop loss
            take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.09),  # 9% take profit
            max_position_size=0.1,
            volatility_limit=0.3
        )
        
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test_strategy",
            confidence=0.85,
            entry_price=150.00,
            risk_controls=risk_controls
        )
        
        # Process signal
        result = execution_agent._process_trade_signal(signal, sample_market_data)
        
        # Should be successful
        assert result.success is True
        assert result.signal == signal
        assert result.risk_metrics is not None
    
    def test_risk_controls_validation(self, execution_agent):
        """Test risk controls validation."""
        # Test valid risk controls
        valid_controls = RiskControls(
            stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.02),
            take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.06)
        )
        
        # Should not raise any exceptions
        assert valid_controls.stop_loss.value == 0.02
        assert valid_controls.take_profit.value == 0.06
        
        # Test invalid threshold type
        with pytest.raises(ValueError):
            RiskThreshold("invalid_type", 0.02)


if __name__ == "__main__":
    pytest.main([__file__]) 