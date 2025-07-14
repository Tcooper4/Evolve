"""
Portfolio Management Integration Tests

This module tests the complete integration of the portfolio management system
with the rest of the Evolve trading platform.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import portfolio modules
from portfolio import (
    PortfolioAllocator,
    PortfolioRiskManager,
    AllocationStrategy,
    AssetMetrics,
    create_allocator,
    create_risk_manager
)

# Import platform components
from meta.meta_controller import MetaControllerAgent, ActionType, TriggerCondition
from agents.model_innovation_agent import ModelInnovationAgent
from agents.strategy_research_agent import StrategyResearchAgent
from utils.weight_registry import WeightRegistry
from utils.cache_utils import cache_result
from utils.common_helpers import load_config


class TestPortfolioPlatformIntegration:
    """Test portfolio management integration with the platform"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_assets(self):
        """Create sample assets for testing"""
        return [
            AssetMetrics(
                ticker="AAPL",
                expected_return=0.12,
                volatility=0.20,
                sharpe_ratio=0.5,
                beta=1.1,
                correlation={"TSLA": 0.3, "NVDA": 0.4},
                market_cap=2000000000000,
                sector="Technology",
                sentiment_score=0.6
            ),
            AssetMetrics(
                ticker="TSLA",
                expected_return=0.18,
                volatility=0.35,
                sharpe_ratio=0.46,
                beta=1.8,
                correlation={"AAPL": 0.3, "NVDA": 0.5},
                market_cap=800000000000,
                sector="Automotive",
                sentiment_score=0.7
            ),
            AssetMetrics(
                ticker="NVDA",
                expected_return=0.15,
                volatility=0.30,
                sharpe_ratio=0.43,
                beta=1.5,
                correlation={"AAPL": 0.4, "TSLA": 0.5},
                market_cap=1200000000000,
                sector="Technology",
                sentiment_score=0.8
            )
        ]
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        market_data = {}
        
        for ticker in ['AAPL', 'TSLA', 'NVDA', 'SPY']:
            returns = np.random.normal(0.001, 0.02, len(dates))
            market_data[ticker] = pd.DataFrame({'returns': returns}, index=dates)
        
        return market_data
    
    def test_portfolio_allocator_creation(self):
        """Test portfolio allocator creation with platform config"""
        with patch('portfolio.allocator.load_config') as mock_load_config:
            mock_config = {
                'portfolio': {
                    'max_weight': 0.3,
                    'min_weight': 0.01,
                    'target_volatility': 0.15,
                    'risk_free_rate': 0.02
                }
            }
            mock_load_config.return_value = mock_config
            
            allocator = create_allocator()
            
            assert allocator is not None
            assert isinstance(allocator, PortfolioAllocator)
            assert allocator.max_weight == 0.3
            assert allocator.min_weight == 0.01
            assert allocator.target_volatility == 0.15
    
    def test_portfolio_risk_manager_creation(self):
        """Test portfolio risk manager creation with platform config"""
        with patch('portfolio.risk_manager.load_config') as mock_load_config:
            mock_config = {
                'risk_management': {
                    'max_drawdown': 0.15,
                    'max_exposure': 0.3,
                    'max_leverage': 2.0,
                    'target_volatility': 0.15,
                    'var_limit': 0.02
                }
            }
            mock_load_config.return_value = mock_config
            
            risk_manager = create_risk_manager()
            
            assert risk_manager is not None
            assert isinstance(risk_manager, PortfolioRiskManager)
            assert risk_manager.risk_limits.max_drawdown == 0.15
            assert risk_manager.risk_limits.max_exposure == 0.3
    
    def test_meta_controller_portfolio_integration(self, sample_assets, sample_market_data):
        """Test MetaControllerAgent integration with portfolio management"""
        with patch('meta.meta_controller.load_config') as mock_load_config:
            mock_config = {
                'meta_controller': {},
                'portfolio': {
                    'max_weight': 0.3,
                    'min_weight': 0.01,
                    'target_volatility': 0.15
                },
                'risk_management': {
                    'max_drawdown': 0.15,
                    'max_exposure': 0.3,
                    'var_limit': 0.02
                }
            }
            mock_load_config.return_value = mock_config
            
            # Create meta controller
            meta_controller = MetaControllerAgent()
            
            # Verify portfolio components are initialized
            assert meta_controller.portfolio_allocator is not None
            assert meta_controller.portfolio_risk_manager is not None
            assert isinstance(meta_controller.portfolio_allocator, PortfolioAllocator)
            assert isinstance(meta_controller.portfolio_risk_manager, PortfolioRiskManager)
            
            # Test portfolio state initialization
            assert 'current_allocation' in meta_controller.portfolio_state
            assert 'target_allocation' in meta_controller.portfolio_state
            assert 'risk_violations' in meta_controller.portfolio_state
            assert 'allocation_strategy' in meta_controller.portfolio_state
    
    def test_portfolio_allocation_workflow(self, sample_assets):
        """Test complete portfolio allocation workflow"""
        # Create allocator
        allocator = PortfolioAllocator()
        
        # Test all allocation strategies
        strategies = [
            AllocationStrategy.EQUAL_WEIGHT,
            AllocationStrategy.MAXIMUM_SHARPE,
            AllocationStrategy.RISK_PARITY,
            AllocationStrategy.KELLY_CRITERION
        ]
        
        for strategy in strategies:
            result = allocator.allocate_portfolio(sample_assets, strategy)
            
            assert result is not None
            assert result.strategy == strategy
            assert len(result.weights) == len(sample_assets)
            assert abs(sum(result.weights.values()) - 1.0) < 0.01
            assert result.constraints_satisfied is True
    
    def test_portfolio_risk_management_workflow(self, sample_assets, sample_market_data):
        """Test complete portfolio risk management workflow"""
        # Create risk manager
        risk_manager = PortfolioRiskManager()
        
        # Create sample positions
        positions = {"AAPL": 0.4, "TSLA": 0.3, "NVDA": 0.3}
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(positions, sample_market_data)
        
        assert 'volatility' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'current_drawdown' in risk_metrics
        
        # Check risk limits
        violations = risk_manager.check_risk_limits(positions, risk_metrics)
        
        assert isinstance(violations, list)
        
        # Generate rebalancing actions
        target_positions = {"AAPL": 0.33, "TSLA": 0.33, "NVDA": 0.34}
        actions = risk_manager.generate_rebalancing_actions(positions, target_positions, violations)
        
        assert isinstance(actions, list)
    
    def test_portfolio_simulation_workflow(self, sample_assets, sample_market_data):
        """Test portfolio simulation workflow"""
        # Create risk manager
        risk_manager = PortfolioRiskManager()
        
        # Create sample positions
        positions = {"AAPL": 0.4, "TSLA": 0.3, "NVDA": 0.3}
        
        # Run simulation
        simulation = risk_manager.simulate_portfolio_returns(positions, sample_market_data, 'monthly')
        
        assert isinstance(simulation, pd.DataFrame)
        assert len(simulation) > 0
        assert 'portfolio_value' in simulation.columns
        assert 'daily_return' in simulation.columns
        assert 'cumulative_return' in simulation.columns
        assert 'drawdown' in simulation.columns
    
    def test_stress_testing_workflow(self, sample_assets, sample_market_data):
        """Test stress testing workflow"""
        # Create risk manager
        risk_manager = PortfolioRiskManager()
        
        # Create sample positions
        positions = {"AAPL": 0.4, "TSLA": 0.3, "NVDA": 0.3}
        
        # Define stress scenarios
        scenarios = {
            'Market Crash': {
                'AAPL': -0.3, 'TSLA': -0.4, 'NVDA': -0.35
            },
            'Tech Rally': {
                'AAPL': 0.3, 'TSLA': 0.4, 'NVDA': 0.35
            }
        }
        
        # Run stress tests
        stress_results = risk_manager.stress_test_portfolio(positions, sample_market_data, scenarios)
        
        assert isinstance(stress_results, dict)
        assert 'Market Crash' in stress_results
        assert 'Tech Rally' in stress_results
        
        for scenario, metrics in stress_results.items():
            assert 'total_return' in metrics
            assert 'volatility' in metrics
            assert 'max_drawdown' in metrics
            assert 'var_95' in metrics
    
    def test_meta_controller_portfolio_actions(self, sample_assets, sample_market_data):
        """Test MetaControllerAgent portfolio actions"""
        with patch('meta.meta_controller.load_config') as mock_load_config:
            mock_config = {
                'meta_controller': {},
                'portfolio': {'max_weight': 0.3, 'min_weight': 0.01},
                'risk_management': {'max_drawdown': 0.15, 'max_exposure': 0.3}
            }
            mock_load_config.return_value = mock_config
            
            meta_controller = MetaControllerAgent()
            
            # Test portfolio rebalancing trigger
            rebalance_task = meta_controller.trigger_portfolio_rebalancing()
            assert rebalance_task is not None
            
            # Test risk management trigger
            risk_task = meta_controller.trigger_risk_management()
            assert risk_task is not None
    
    def test_portfolio_configuration_integration(self):
        """Test portfolio configuration integration"""
        # Test that portfolio config is properly loaded
        config = load_config("config/app_config.yaml")
        
        assert 'portfolio' in config
        assert 'risk_management' in config
        
        portfolio_config = config['portfolio']
        risk_config = config['risk_management']
        
        # Check portfolio config
        assert 'max_weight' in portfolio_config
        assert 'min_weight' in portfolio_config
        assert 'target_volatility' in portfolio_config
        assert 'risk_free_rate' in portfolio_config
        
        # Check risk management config
        assert 'max_drawdown' in risk_config
        assert 'max_exposure' in risk_config
        assert 'target_volatility' in risk_config
        assert 'var_limit' in risk_config
    
    def test_trigger_thresholds_integration(self):
        """Test trigger thresholds integration"""
        # Load trigger thresholds
        thresholds_path = Path("config/trigger_thresholds.json")
        
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            
            # Check portfolio-specific thresholds
            assert 'portfolio_rebalance' in thresholds
            assert 'allocation_optimization' in thresholds
            
            portfolio_thresholds = thresholds['portfolio_rebalance']
            allocation_thresholds = thresholds['allocation_optimization']
            
            # Check portfolio rebalance thresholds
            assert 'time_threshold_hours' in portfolio_thresholds
            assert 'drift_threshold' in portfolio_thresholds
            assert 'risk_violation_threshold' in portfolio_thresholds
            
            # Check allocation optimization thresholds
            assert 'sharpe_degradation_threshold' in allocation_thresholds
            assert 'time_threshold_hours' in allocation_thresholds
            assert 'market_regime_change_threshold' in allocation_thresholds
    
    def test_caching_integration(self, sample_assets):
        """Test caching integration with portfolio management"""
        # Test that portfolio allocation results can be cached
        allocator = PortfolioAllocator()
        
        # First allocation
        result1 = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        # Second allocation (should be cached)
        result2 = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        # Results should be identical
        assert result1.weights == result2.weights
        assert result1.expected_return == result2.expected_return
        assert result1.expected_volatility == result2.expected_volatility
    
    def test_weight_registry_integration(self, sample_assets):
        """Test weight registry integration with portfolio management"""
        # Create weight registry
        weight_registry = WeightRegistry()
        
        # Create allocation result
        allocator = PortfolioAllocator()
        result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        # Save weights to registry
        weight_registry.save_weights(
            strategy_name="maximum_sharpe",
            weights=result.weights,
            performance_metrics={
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'sharpe_ratio': result.sharpe_ratio
            }
        )
        
        # Load weights from registry
        loaded_weights = weight_registry.get_weights("maximum_sharpe")
        
        assert loaded_weights is not None
        assert loaded_weights['weights'] == result.weights
        assert loaded_weights['performance_metrics']['expected_return'] == result.expected_return
    
    def test_agent_integration(self, sample_assets):
        """Test integration with other agents"""
        # Test ModelInnovationAgent integration
        model_agent = ModelInnovationAgent()
        
        # Test StrategyResearchAgent integration
        strategy_agent = StrategyResearchAgent()
        
        # Both agents should be able to work with portfolio components
        assert model_agent is not None
        assert strategy_agent is not None
    
    def test_error_handling_integration(self):
        """Test error handling integration"""
        # Test with invalid assets
        allocator = PortfolioAllocator()
        
        with pytest.raises(ValueError):
            allocator.allocate_portfolio([], AllocationStrategy.MAXIMUM_SHARPE)
        
        # Test with invalid strategy
        assets = [AssetMetrics(
            ticker="AAPL",
            expected_return=0.12,
            volatility=0.20,
            sharpe_ratio=0.5,
            beta=1.1,
            correlation={},
            market_cap=1000000000,
            sector="Technology",
            sentiment_score=0.6
        )]
        
        with pytest.raises(ValueError):
            allocator.allocate_portfolio(assets, "invalid_strategy")
    
    def test_performance_integration(self, sample_assets, sample_market_data):
        """Test performance integration"""
        import time
        
        # Test allocation performance
        allocator = PortfolioAllocator()
        start_time = time.time()
        
        for _ in range(10):
            result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        allocation_time = time.time() - start_time
        assert allocation_time < 5.0  # Should complete within 5 seconds
        
        # Test risk calculation performance
        risk_manager = PortfolioRiskManager()
        positions = {"AAPL": 0.4, "TSLA": 0.3, "NVDA": 0.3}
        
        start_time = time.time()
        
        for _ in range(10):
            risk_metrics = risk_manager.calculate_portfolio_risk(positions, sample_market_data)
        
        risk_time = time.time() - start_time
        assert risk_time < 5.0  # Should complete within 5 seconds
    
    def test_data_persistence_integration(self, sample_assets):
        """Test data persistence integration"""
        # Test that portfolio results can be saved and loaded
        allocator = PortfolioAllocator()
        result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        # Save result
        result_data = {
            'strategy': result.strategy.value,
            'weights': result.weights,
            'expected_return': result.expected_return,
            'expected_volatility': result.expected_volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result_data, f)
            temp_file = f.name
        
        # Load from file
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify data integrity
        assert loaded_data['strategy'] == result.strategy.value
        assert loaded_data['weights'] == result.weights
        assert loaded_data['expected_return'] == result.expected_return
        assert loaded_data['expected_volatility'] == result.expected_volatility
        assert loaded_data['sharpe_ratio'] == result.sharpe_ratio
        
        # Clean up
        import os
        os.unlink(temp_file)
    
    def test_concurrent_access_integration(self, sample_assets):
        """Test concurrent access integration"""
        import threading
        import time
        
        allocator = PortfolioAllocator()
        results = []
        errors = []
        
        def allocate_portfolio():
            try:
                result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_portfolio)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
        
        # All results should be identical (due to caching)
        first_result = results[0]
        for result in results[1:]:
            assert result.weights == first_result.weights
            assert result.expected_return == first_result.expected_return
    
    def test_complete_workflow_integration(self, sample_assets, sample_market_data):
        """Test complete workflow integration"""
        # 1. Create portfolio components
        allocator = PortfolioAllocator()
        risk_manager = PortfolioRiskManager()
        
        # 2. Allocate portfolio
        allocation_result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.MAXIMUM_SHARPE)
        positions = allocation_result.weights
        
        # 3. Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(positions, sample_market_data)
        
        # 4. Check for violations
        violations = risk_manager.check_risk_limits(positions, risk_metrics)
        
        # 5. Generate rebalancing actions if needed
        if violations:
            actions = risk_manager.generate_rebalancing_actions(positions, positions, violations)
        else:
            actions = []
        
        # 6. Simulate portfolio performance
        simulation = risk_manager.simulate_portfolio_returns(positions, sample_market_data, 'monthly')
        
        # 7. Run stress tests
        scenarios = {
            'Market Crash': {'AAPL': -0.3, 'TSLA': -0.4, 'NVDA': -0.35},
            'Tech Rally': {'AAPL': 0.3, 'TSLA': 0.4, 'NVDA': 0.35}
        }
        stress_results = risk_manager.stress_test_portfolio(positions, sample_market_data, scenarios)
        
        # Verify all components work together
        assert allocation_result is not None
        assert risk_metrics is not None
        assert isinstance(violations, list)
        assert isinstance(actions, list)
        assert simulation is not None
        assert stress_results is not None
        
        # Verify data consistency
        assert len(positions) == len(sample_assets)
        assert abs(sum(positions.values()) - 1.0) < 0.01
        assert 'volatility' in risk_metrics
        assert 'var_95' in risk_metrics
        assert len(simulation) > 0
        assert 'Market Crash' in stress_results
        assert 'Tech Rally' in stress_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 