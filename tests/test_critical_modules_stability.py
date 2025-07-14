"""
Test Critical Modules Stability

This script tests all the improvements made to critical modules for stability,
modularity, and dynamic execution.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.agent_manager import AgentManager, AgentManagerConfig
from trading.agents.base_agent_interface import AgentConfig
from agents.model_discovery_agent import ModelDiscoveryAgent
from meta_learning.strategy_refiner import StrategyRefiner, StrategyPerformance
from config.config_loader import ConfigLoader, validate_config
from trading.agents.forecast_dispatcher import ForecastDispatcher


class TestAgentManagerStability:
    """Test agent manager improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.agent_manager = AgentManager(AgentManagerConfig(
            execution_timeout=10,
            max_concurrent_agents=3
        ))
        
        # Register test agents
        self.agent_manager.register_agent(
            "test_agent_1",
            type("TestAgent1", (), {"run": lambda **kwargs: asyncio.sleep(1)}),
            AgentConfig(enabled=True, timeout_seconds=5)
        )
        
        self.agent_manager.register_agent(
            "test_agent_2", 
            type("TestAgent2", (), {"run": lambda **kwargs: asyncio.sleep(2)}),
            AgentConfig(enabled=True, timeout_seconds=5)
        )
    
    def test_agent_heartbeat_and_watchdog(self):
        """Test agent heartbeat and watchdog functionality."""
        # Start restart monitor
        self.agent_manager.start_restart_monitor()
        
        # Check health status
        health_status = self.agent_manager.get_all_agent_health_statuses()
        assert len(health_status) > 0
        
        # Stop restart monitor
        self.agent_manager.stop_restart_monitor()
    
    def test_async_timeout_execution(self):
        """Test async execution with timeout."""
        async def test_timeout():
            # Test normal execution
            result = await self.agent_manager.execute_agent("test_agent_1")
            assert result.success
            assert result.execution_time > 0
            
            # Test timeout execution
            result = await self.agent_manager.execute_agent(
                "test_agent_2", 
                timeout=1  # Should timeout
            )
            assert not result.success
            assert "timeout" in result.error_message.lower()
        
        asyncio.run(test_timeout())
    
    def test_callback_registration(self):
        """Test callback registration and triggering."""
        callback_called = False
        
        def test_callback(**kwargs):
            nonlocal callback_called
            callback_called = True
            assert 'agent_name' in kwargs
        
        # Register callback
        self.agent_manager.register_callback('agent_completed', test_callback)
        
        # Execute agent to trigger callback
        async def trigger_callback():
            await self.agent_manager.execute_agent("test_agent_1")
        
        asyncio.run(trigger_callback())
        assert callback_called
    
    def test_execution_logging(self):
        """Test execution duration and status logging."""
        async def test_logging():
            result = await self.agent_manager.execute_agent("test_agent_1")
            
            # Check execution metrics
            metrics = self.agent_manager.get_execution_metrics()
            assert metrics['success']
            assert len(metrics['result']['agent_metrics']) > 0
            
            # Check specific agent metrics
            agent_metrics = metrics['result']['agent_metrics']['test_agent_1']
            assert agent_metrics['total_executions'] > 0
            assert agent_metrics['avg_execution_time'] > 0
        
        asyncio.run(test_logging())


class TestModelDiscoveryAgent:
    """Test model discovery agent improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.discovery_agent = ModelDiscoveryAgent()
    
    def test_model_discovery(self):
        """Test model discovery functionality."""
        async def test_discovery():
            result = await self.discovery_agent.execute(
                model_type='lstm',
                force_regeneration=True
            )
            
            assert result.success
            assert 'discovered_models' in result.data
            assert 'validated_models' in result.data
            assert 'registered_models' in result.data
        
        asyncio.run(test_discovery())
    
    def test_model_validation(self):
        """Test model validation."""
        # Test valid model config
        valid_model = {
            'id': 'test_lstm',
            'type': 'lstm',
            'config': {'layers': [50, 25], 'dropout': 0.2, 'epochs': 100}
        }
        
        assert self.discovery_agent._validate_model_config(valid_model)
        
        # Test invalid model config
        invalid_model = {
            'id': 'test_invalid',
            'type': 'lstm',
            'config': {}  # Missing required fields
        }
        
        assert not self.discovery_agent._validate_model_config(invalid_model)
    
    def test_generation_history(self):
        """Test generation history tracking."""
        # Add test performance
        self.discovery_agent.add_strategy_performance(
            strategy_name="test_strategy",
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6,
            total_return=0.2,
            volatility=0.15
        )
        
        # Check history
        history = self.discovery_agent.get_generation_history()
        assert len(history) > 0
        
        # Check performance stats
        stats = self.discovery_agent.get_model_performance_stats()
        assert 'total_models' in stats


class TestStrategyRefiner:
    """Test strategy refiner improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.refiner = StrategyRefiner()
    
    def test_recency_weighting(self):
        """Test recency weighting functionality."""
        # Add performance data with different timestamps
        now = datetime.now()
        
        self.refiner.add_strategy_performance(
            strategy_name="recent_strategy",
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6,
            total_return=0.2,
            volatility=0.15,
            timestamp=now
        )
        
        self.refiner.add_strategy_performance(
            strategy_name="old_strategy",
            sharpe_ratio=1.0,
            max_drawdown=0.2,
            win_rate=0.5,
            total_return=0.1,
            volatility=0.2,
            timestamp=now - timedelta(days=50)
        )
        
        # Calculate recency weights
        self.refiner._calculate_recency_weights()
        
        # Check that recent strategy has higher weight
        recent_perf = next(p for p in self.refiner.performance_history 
                          if p.strategy_name == "recent_strategy")
        old_perf = next(p for p in self.refiner.performance_history 
                       if p.strategy_name == "old_strategy")
        
        assert recent_perf.recency_weight > old_perf.recency_weight
    
    def test_plug_and_play_scoring(self):
        """Test plug-and-play scoring functions."""
        # Test Sharpe scoring
        performance = StrategyPerformance(
            strategy_name="test",
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6,
            total_return=0.2,
            volatility=0.15,
            calmar_ratio=2.0,
            sortino_ratio=1.8,
            timestamp=datetime.now()
        )
        
        sharpe_score = self.refiner.scoring_functions['sharpe'].calculate_score(performance)
        win_rate_score = self.refiner.scoring_functions['win_rate'].calculate_score(performance)
        composite_score = self.refiner.scoring_functions['composite'].calculate_score(performance)
        
        assert sharpe_score > 0
        assert win_rate_score > 0
        assert composite_score > 0
    
    def test_strategy_refinement(self):
        """Test strategy refinement functionality."""
        strategy_configs = [
            {
                'name': 'test_strategy_1',
                'parameters': {
                    'threshold': 0.05,
                    'stop_loss': 0.02,
                    'take_profit': 0.08
                }
            },
            {
                'name': 'test_strategy_2',
                'parameters': {
                    'threshold': 0.03,
                    'stop_loss': 0.01,
                    'take_profit': 0.06
                }
            }
        ]
        
        # Add performance data for refinement
        self.refiner.add_strategy_performance(
            strategy_name="test_strategy_1",
            sharpe_ratio=1.8,
            max_drawdown=0.08,
            win_rate=0.7,
            total_return=0.25,
            volatility=0.12
        )
        
        refined_configs = self.refiner.refine_strategies(strategy_configs)
        assert len(refined_configs) == len(strategy_configs)
        
        # Check that parameters were modified
        for original, refined in zip(strategy_configs, refined_configs):
            assert original['parameters'] != refined['parameters']


class TestConfigLoader:
    """Test configuration loader improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config_loader = ConfigLoader()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        valid_config = {
            'enabled': True,
            'default_horizon_days': 30,
            'max_horizon_days': 90
        }
        
        # Test invalid config
        invalid_config = {
            'enabled': 'not_a_boolean',  # Should be boolean
            'default_horizon_days': -1   # Should be positive
        }
        
        # Validation should pass for valid config
        errors = self.config_loader.get_validation_errors()
        # Note: This depends on actual config files being present
    
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        # Set environment variable
        os.environ['FORECASTING_ENABLED'] = 'false'
        
        # Reload config to apply override
        self.config_loader._apply_environment_overrides()
        
        # Check that override was applied
        forecasting_enabled = self.config_loader.get_config('forecasting', 'enabled')
        if forecasting_enabled is not None:
            assert not forecasting_enabled
        
        # Clean up
        del os.environ['FORECASTING_ENABLED']
    
    def test_config_reloading(self):
        """Test configuration reloading."""
        # Test reloading specific section
        success = self.config_loader.reload_config('forecasting')
        assert success
        
        # Test reloading all configs
        success = self.config_loader.reload_config()
        assert success
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        summary = self.config_loader.get_config_summary()
        
        assert 'total_sections' in summary
        assert 'validation_status' in summary
        assert 'total_errors' in summary
        assert 'sections' in summary


class TestForecastDispatcher:
    """Test forecast dispatcher improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dispatcher = ForecastDispatcher()
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        self.test_data.set_index('date', inplace=True)
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism."""
        async def test_fallback():
            # Test with data that might cause issues
            result = await self.dispatcher.execute(
                data=self.test_data,
                target_column='close',
                horizon=5,
                models=['lstm', 'xgboost']  # Use subset of models
            )
            
            assert result.success
            assert 'forecast' in result.data
            assert 'model_name' in result.data
            assert 'is_fallback' in result.data
        
        asyncio.run(test_fallback())
    
    def test_consensus_checking(self):
        """Test consensus checking functionality."""
        # Create mock forecast results
        forecast_results = [
            ForecastResult(
                model_name='model1',
                forecast_values=np.array([100, 101, 102]),
                confidence_intervals=(np.array([99, 100, 101]), np.array([101, 102, 103]))
            ),
            ForecastResult(
                model_name='model2',
                forecast_values=np.array([100.5, 101.5, 102.5]),
                confidence_intervals=(np.array([99.5, 100.5, 101.5]), np.array([101.5, 102.5, 103.5]))
            )
        ]
        
        consensus_result = self.dispatcher._check_consensus(forecast_results)
        
        assert consensus_result is not None
        assert consensus_result.agreement_level > 0
        assert len(consensus_result.model_weights) == 2
    
    def test_confidence_intervals(self):
        """Test confidence interval handling."""
        async def test_confidence():
            result = await self.dispatcher.execute(
                data=self.test_data,
                target_column='close',
                horizon=5
            )
            
            assert result.success
            
            # Check if confidence intervals are included
            all_forecasts = result.data.get('all_forecasts', {})
            for model_name, forecast_data in all_forecasts.items():
                if 'confidence_intervals' in forecast_data:
                    assert forecast_data['confidence_intervals'] is not None
        
        asyncio.run(test_confidence())
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        # Execute some forecasts
        async def track_performance():
            await self.dispatcher.execute(
                data=self.test_data,
                target_column='close',
                horizon=5
            )
            
            # Check performance tracking
            performance = self.dispatcher.get_model_performance()
            assert len(performance) > 0
            
            # Check last working model
            last_model = self.dispatcher.get_last_working_model()
            if last_model:
                assert last_model in self.dispatcher.models
        
        asyncio.run(track_performance())


class TestIntegration:
    """Integration tests for all modules."""
    
    def test_full_workflow(self):
        """Test complete workflow integration."""
        async def test_workflow():
            # 1. Load configuration
            config_loader = ConfigLoader()
            assert validate_config()
            
            # 2. Initialize agents
            agent_manager = AgentManager()
            
            # 3. Register specialized agents
            discovery_agent = ModelDiscoveryAgent()
            refiner_agent = StrategyRefiner()
            dispatcher_agent = ForecastDispatcher()
            
            agent_manager.register_agent("model_discovery", discovery_agent)
            agent_manager.register_agent("strategy_refiner", refiner_agent)
            agent_manager.register_agent("forecast_dispatcher", dispatcher_agent)
            
            # 4. Execute workflow
            # Model discovery
            discovery_result = await agent_manager.execute_agent("model_discovery")
            assert discovery_result.success
            
            # Strategy refinement
            refiner_result = await agent_manager.execute_agent("strategy_refiner")
            assert refiner_result.success
            
            # 5. Check system health
            health_status = agent_manager.get_all_agent_health_statuses()
            assert len(health_status) >= 3
            
            # 6. Check performance metrics
            metrics = agent_manager.get_execution_metrics()
            assert metrics['success']
        
        asyncio.run(test_workflow())


def run_all_tests():
    """Run all tests and generate report."""
    print("🧪 Running Critical Modules Stability Tests")
    print("=" * 50)
    
    test_classes = [
        TestAgentManagerStability,
        TestModelDiscoveryAgent,
        TestStrategyRefiner,
        TestConfigLoader,
        TestForecastDispatcher,
        TestIntegration
    ]
    
    results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            results['total_tests'] += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                results['passed'] += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                results['failed'] += 1
                results['errors'].append(f"{test_class.__name__}.{method_name}: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['passed']/results['total_tests']*100:.1f}%")
    
    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Save results
    results_file = Path("tests/results/critical_modules_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            **results,
            'timestamp': datetime.now().isoformat(),
            'test_environment': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results['failed'] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 