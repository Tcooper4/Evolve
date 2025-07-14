"""
Tests for Portfolio Modules

Tests both portfolio/allocator.py and portfolio/risk_manager.py modules.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from portfolio.allocator import (
    PortfolioAllocator,
    AllocationStrategy,
    AssetMetrics,
    AllocationResult,
    create_allocator,
    allocate_portfolio
)
from portfolio.risk_manager import (
    PortfolioRiskManager,
    RiskMetric,
    RiskLimits,
    PortfolioState,
    RiskViolation,
    RebalancingAction,
    create_risk_manager,
    calculate_portfolio_risk,
    check_risk_limits
)


class TestAllocationStrategy:
    """Test AllocationStrategy enum"""
    
    def test_allocation_strategy_values(self):
        """Test AllocationStrategy enum values"""
        assert AllocationStrategy.EQUAL_WEIGHT.value == "equal_weight"
        assert AllocationStrategy.MINIMUM_VARIANCE.value == "minimum_variance"
        assert AllocationStrategy.MAXIMUM_SHARPE.value == "maximum_sharpe"
        assert AllocationStrategy.RISK_PARITY.value == "risk_parity"
        assert AllocationStrategy.KELLY_CRITERION.value == "kelly_criterion"


class TestAssetMetrics:
    """Test AssetMetrics dataclass"""
    
    def test_asset_metrics_creation(self):
        """Test creating an AssetMetrics instance"""
        metrics = AssetMetrics(
            ticker="AAPL",
            expected_return=0.12,
            volatility=0.20,
            sharpe_ratio=0.5,
            beta=1.1,
            correlation={"TSLA": 0.3, "NVDA": 0.4},
            market_cap=2000000000000,
            sector="Technology",
            sentiment_score=0.6
        )
        
        assert metrics.ticker == "AAPL"
        assert metrics.expected_return == 0.12
        assert metrics.volatility == 0.20
        assert metrics.sharpe_ratio == 0.5
        assert len(metrics.correlation) == 2


class TestAllocationResult:
    """Test AllocationResult dataclass"""
    
    def test_allocation_result_creation(self):
        """Test creating an AllocationResult instance"""
        result = AllocationResult(
            strategy=AllocationStrategy.EQUAL_WEIGHT,
            weights={"AAPL": 0.5, "TSLA": 0.5},
            expected_return=0.15,
            expected_volatility=0.25,
            sharpe_ratio=0.52,
            risk_contribution={"AAPL": 0.5, "TSLA": 0.5},
            diversification_ratio=1.2,
            constraints_satisfied=True,
            optimization_status="success",
            metadata={"variance": 0.0625}
        )
        
        assert result.strategy == AllocationStrategy.EQUAL_WEIGHT
        assert result.expected_return == 0.15
        assert result.sharpe_ratio == 0.52
        assert result.constraints_satisfied is True


class TestPortfolioAllocator:
    """Test PortfolioAllocator functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def allocator(self, temp_dir):
        """Create PortfolioAllocator instance for testing"""
        config = {
            'portfolio': {
                'max_weight': 0.3,
                'min_weight': 0.01,
                'target_volatility': 0.15,
                'risk_free_rate': 0.02
            }
        }
        
        with patch('portfolio.allocator.load_config', return_value=config):
            allocator = PortfolioAllocator()
            return allocator
    
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
    
    def test_allocator_initialization(self, allocator):
        """Test allocator initialization"""
        assert allocator.max_weight == 0.3
        assert allocator.min_weight == 0.01
        assert allocator.target_volatility == 0.15
        assert allocator.risk_free_rate == 0.02
    
    def test_equal_weight_allocation(self, allocator, sample_assets):
        """Test equal weight allocation"""
        weights = allocator._equal_weight_allocation(sample_assets)
        
        assert len(weights) == 3
        assert np.allclose(weights, [1/3, 1/3, 1/3])
        assert np.isclose(np.sum(weights), 1.0)
    
    def test_minimum_variance_allocation(self, allocator, sample_assets):
        """Test minimum variance allocation"""
        # Mock covariance matrix
        covariance_matrix = np.array([
            [0.04, 0.021, 0.024],
            [0.021, 0.1225, 0.0525],
            [0.024, 0.0525, 0.09]
        ])
        
        weights = allocator._minimum_variance_allocation(covariance_matrix)
        
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert all(0 <= w <= 1 for w in weights)
    
    def test_maximum_sharpe_allocation(self, allocator, sample_assets):
        """Test maximum Sharpe ratio allocation"""
        expected_returns = np.array([0.12, 0.18, 0.15])
        covariance_matrix = np.array([
            [0.04, 0.021, 0.024],
            [0.021, 0.1225, 0.0525],
            [0.024, 0.0525, 0.09]
        ])
        
        weights = allocator._maximum_sharpe_allocation(expected_returns, covariance_matrix)
        
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert all(0 <= w <= 1 for w in weights)
    
    def test_kelly_criterion_allocation(self, allocator, sample_assets):
        """Test Kelly criterion allocation"""
        weights = allocator._kelly_criterion_allocation(sample_assets)
        
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert all(0 <= w <= 1 for w in weights)
    
    def test_build_correlation_matrix(self, allocator, sample_assets):
        """Test correlation matrix building"""
        correlation_matrix = allocator._build_correlation_matrix(sample_assets)
        
        assert correlation_matrix.shape == (3, 3)
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric
    
    def test_build_covariance_matrix(self, allocator, sample_assets):
        """Test covariance matrix building"""
        volatilities = np.array([0.20, 0.35, 0.30])
        correlation_matrix = allocator._build_correlation_matrix(sample_assets)
        covariance_matrix = allocator._build_covariance_matrix(volatilities, correlation_matrix)
        
        assert covariance_matrix.shape == (3, 3)
        assert np.allclose(covariance_matrix, covariance_matrix.T)  # Should be symmetric
        assert all(covariance_matrix[i, i] == volatilities[i]**2 for i in range(3))
    
    def test_calculate_portfolio_metrics(self, allocator):
        """Test portfolio metrics calculation"""
        weights = np.array([0.4, 0.3, 0.3])
        expected_returns = np.array([0.12, 0.18, 0.15])
        covariance_matrix = np.array([
            [0.04, 0.021, 0.024],
            [0.021, 0.1225, 0.0525],
            [0.024, 0.0525, 0.09]
        ])
        
        metrics = allocator._calculate_portfolio_metrics(weights, expected_returns, covariance_matrix)
        
        assert 'expected_return' in metrics
        assert 'expected_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'variance' in metrics
        assert metrics['expected_return'] > 0
        assert metrics['expected_volatility'] > 0
    
    def test_calculate_risk_contributions(self, allocator):
        """Test risk contribution calculation"""
        weights = np.array([0.4, 0.3, 0.3])
        covariance_matrix = np.array([
            [0.04, 0.021, 0.024],
            [0.021, 0.1225, 0.0525],
            [0.024, 0.0525, 0.09]
        ])
        
        risk_contributions = allocator._calculate_risk_contributions(weights, covariance_matrix)
        
        assert len(risk_contributions) == 3
        assert np.isclose(np.sum(risk_contributions), np.sqrt(weights.T @ covariance_matrix @ weights))
    
    def test_normalize_weights(self, allocator):
        """Test weight normalization"""
        weights = np.array([0.2, 0.3, 0.5])
        normalized = allocator._normalize_weights(weights)
        
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.allclose(normalized, weights / np.sum(weights))
    
    def test_check_constraints(self, allocator):
        """Test constraint checking"""
        weights = np.array([0.4, 0.3, 0.3])
        
        # Test valid weights
        assert allocator._check_constraints(weights) is True
        
        # Test invalid weights (sum != 1)
        invalid_weights = np.array([0.4, 0.3, 0.2])
        assert allocator._check_constraints(invalid_weights) is False
    
    def test_allocate_portfolio_equal_weight(self, allocator, sample_assets):
        """Test complete equal weight allocation"""
        result = allocator.allocate_portfolio(sample_assets, AllocationStrategy.EQUAL_WEIGHT)
        
        assert result.strategy == AllocationStrategy.EQUAL_WEIGHT
        assert len(result.weights) == 3
        assert all(abs(w - 1/3) < 0.01 for w in result.weights.values())
        assert result.constraints_satisfied is True
        assert result.optimization_status == "success"
    
    def test_compare_strategies(self, allocator, sample_assets):
        """Test strategy comparison"""
        results = allocator.compare_strategies(sample_assets)
        
        assert len(results) > 0
        assert 'equal_weight' in results
        assert all(isinstance(result, AllocationResult) for result in results.values())
    
    def test_get_optimal_strategy(self, allocator, sample_assets):
        """Test optimal strategy selection"""
        strategy, result = allocator.get_optimal_strategy(sample_assets, 'sharpe')
        
        assert isinstance(strategy, AllocationStrategy)
        assert isinstance(result, AllocationResult)
        assert result.strategy == strategy


class TestRiskMetric:
    """Test RiskMetric enum"""
    
    def test_risk_metric_values(self):
        """Test RiskMetric enum values"""
        assert RiskMetric.VAR.value == "value_at_risk"
        assert RiskMetric.CVAR.value == "conditional_var"
        assert RiskMetric.DRAWDOWN.value == "drawdown"
        assert RiskMetric.VOLATILITY.value == "volatility"


class TestRiskLimits:
    """Test RiskLimits dataclass"""
    
    def test_risk_limits_creation(self):
        """Test creating a RiskLimits instance"""
        limits = RiskLimits(
            max_drawdown=0.15,
            max_exposure=0.3,
            max_leverage=2.0,
            target_volatility=0.15,
            var_limit=0.02,
            max_correlation=0.7,
            sector_limit=0.4,
            liquidity_limit=0.1
        )
        
        assert limits.max_drawdown == 0.15
        assert limits.max_exposure == 0.3
        assert limits.max_leverage == 2.0
        assert limits.target_volatility == 0.15


class TestPortfolioState:
    """Test PortfolioState dataclass"""
    
    def test_portfolio_state_creation(self):
        """Test creating a PortfolioState instance"""
        state = PortfolioState(
            timestamp="2024-01-01T12:00:00",
            positions={"AAPL": 0.5, "TSLA": 0.5},
            portfolio_value=100000.0,
            cash=10000.0,
            leverage=1.1,
            volatility=0.15,
            drawdown=-0.05,
            var_95=-0.02,
            exposure_concentration=0.5,
            sector_exposure={"Technology": 0.8, "Automotive": 0.2},
            risk_metrics={"volatility": 0.15, "var_95": -0.02}
        )
        
        assert state.timestamp == "2024-01-01T12:00:00"
        assert state.portfolio_value == 100000.0
        assert state.leverage == 1.1
        assert len(state.positions) == 2


class TestRiskViolation:
    """Test RiskViolation dataclass"""
    
    def test_risk_violation_creation(self):
        """Test creating a RiskViolation instance"""
        violation = RiskViolation(
            timestamp="2024-01-01T12:00:00",
            risk_metric=RiskMetric.DRAWDOWN,
            current_value=0.20,
            limit_value=0.15,
            severity="critical",
            action_required="Reduce positions",
            affected_positions=["AAPL", "TSLA"]
        )
        
        assert violation.risk_metric == RiskMetric.DRAWDOWN
        assert violation.current_value == 0.20
        assert violation.limit_value == 0.15
        assert violation.severity == "critical"


class TestRebalancingAction:
    """Test RebalancingAction dataclass"""
    
    def test_rebalancing_action_creation(self):
        """Test creating a RebalancingAction instance"""
        action = RebalancingAction(
            action_type="sell",
            ticker="AAPL",
            current_weight=0.4,
            target_weight=0.3,
            trade_amount=0.1,
            priority=4,
            reason="Risk violation"
        )
        
        assert action.action_type == "sell"
        assert action.ticker == "AAPL"
        assert action.current_weight == 0.4
        assert action.target_weight == 0.3
        assert action.priority == 4


class TestPortfolioRiskManager:
    """Test PortfolioRiskManager functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def risk_manager(self, temp_dir):
        """Create PortfolioRiskManager instance for testing"""
        config = {
            'risk_management': {
                'max_drawdown': 0.15,
                'max_exposure': 0.3,
                'max_leverage': 2.0,
                'target_volatility': 0.15,
                'var_limit': 0.02,
                'max_correlation': 0.7,
                'sector_limit': 0.4,
                'liquidity_limit': 0.1
            }
        }
        
        with patch('portfolio.risk_manager.load_config', return_value=config):
            risk_manager = PortfolioRiskManager()
            return risk_manager
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing"""
        return {
            'AAPL': 0.4,
            'TSLA': 0.3,
            'NVDA': 0.3
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        market_data = {}
        
        for ticker in ['AAPL', 'TSLA', 'NVDA', 'SPY']:
            # Generate random returns
            returns = np.random.normal(0.001, 0.02, len(dates))
            market_data[ticker] = pd.DataFrame({
                'returns': returns
            }, index=dates)
        
        return market_data
    
    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initialization"""
        assert risk_manager.risk_limits.max_drawdown == 0.15
        assert risk_manager.risk_limits.max_exposure == 0.3
        assert risk_manager.risk_limits.max_leverage == 2.0
        assert risk_manager.var_confidence == 0.95
    
    def test_calculate_portfolio_risk(self, risk_manager, sample_positions, sample_market_data):
        """Test portfolio risk calculation"""
        risk_metrics = risk_manager.calculate_portfolio_risk(sample_positions, sample_market_data)
        
        assert 'volatility' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'var_99' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'current_drawdown' in risk_metrics
        assert 'exposure_concentration' in risk_metrics
        assert 'herfindahl_index' in risk_metrics
        assert 'max_sector_exposure' in risk_metrics
        
        assert risk_metrics['volatility'] > 0
        assert risk_metrics['exposure_concentration'] == 0.4  # Largest position
    
    def test_calculate_portfolio_returns(self, risk_manager, sample_positions, sample_market_data):
        """Test portfolio returns calculation"""
        returns_data = {ticker: data['returns'] for ticker, data in sample_market_data.items()}
        portfolio_returns = risk_manager._calculate_portfolio_returns(sample_positions, returns_data)
        
        assert len(portfolio_returns) > 0
        assert isinstance(portfolio_returns, pd.Series)
    
    def test_calculate_sector_exposure(self, risk_manager, sample_positions):
        """Test sector exposure calculation"""
        sector_exposure = risk_manager._calculate_sector_exposure(sample_positions)
        
        assert isinstance(sector_exposure, dict)
        assert 'Technology' in sector_exposure
        assert 'Automotive' in sector_exposure
    
    def test_check_risk_limits(self, risk_manager, sample_positions):
        """Test risk limit checking"""
        risk_metrics = {
            'volatility': 0.20,
            'current_drawdown': -0.10,
            'var_95': -0.015
        }
        
        violations = risk_manager.check_risk_limits(sample_positions, risk_metrics)
        
        assert isinstance(violations, list)
        # Should have exposure violation due to AAPL at 40%
        assert len(violations) > 0
        assert any(v.risk_metric == RiskMetric.EXPOSURE for v in violations)
    
    def test_generate_rebalancing_actions(self, risk_manager, sample_positions):
        """Test rebalancing action generation"""
        target_positions = {'AAPL': 0.25, 'TSLA': 0.25, 'NVDA': 0.25, 'MSFT': 0.25}
        violations = [
            RiskViolation(
                timestamp="2024-01-01T12:00:00",
                risk_metric=RiskMetric.EXPOSURE,
                current_value=0.4,
                limit_value=0.3,
                severity="warning",
                action_required="Reduce AAPL position",
                affected_positions=["AAPL"]
            )
        ]
        
        actions = risk_manager.generate_rebalancing_actions(sample_positions, target_positions, violations)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(action, RebalancingAction) for action in actions)
        
        # Should have action to reduce AAPL
        aapl_actions = [a for a in actions if a.ticker == 'AAPL']
        assert len(aapl_actions) > 0
    
    def test_simulate_portfolio_returns(self, risk_manager, sample_positions, sample_market_data):
        """Test portfolio simulation"""
        simulation = risk_manager.simulate_portfolio_returns(sample_positions, sample_market_data, 'monthly')
        
        assert isinstance(simulation, pd.DataFrame)
        assert len(simulation) > 0
        assert 'portfolio_value' in simulation.columns
        assert 'daily_return' in simulation.columns
        assert 'cumulative_return' in simulation.columns
        assert 'drawdown' in simulation.columns
    
    def test_stress_test_portfolio(self, risk_manager, sample_positions, sample_market_data):
        """Test portfolio stress testing"""
        scenarios = {
            'market_crash': {'AAPL': -0.3, 'TSLA': -0.4, 'NVDA': -0.25},
            'tech_rally': {'AAPL': 0.2, 'TSLA': 0.3, 'NVDA': 0.25},
            'volatility_spike': {'AAPL': 0.1, 'TSLA': -0.1, 'NVDA': 0.05}
        }
        
        results = risk_manager.stress_test_portfolio(sample_positions, sample_market_data, scenarios)
        
        assert isinstance(results, dict)
        assert 'market_crash' in results
        assert 'tech_rally' in results
        assert 'volatility_spike' in results
        
        for scenario_name, metrics in results.items():
            assert 'total_return' in metrics
            assert 'volatility' in metrics
            assert 'max_drawdown' in metrics
            assert 'var_95' in metrics
            assert 'worst_day' in metrics
    
    def test_optimize_position_sizing(self, risk_manager, sample_positions):
        """Test position sizing optimization"""
        risk_metrics = {'volatility': 0.20}
        target_volatility = 0.15
        
        optimized_positions = risk_manager.optimize_position_sizing(sample_positions, risk_metrics, target_volatility)
        
        assert isinstance(optimized_positions, dict)
        assert len(optimized_positions) == len(sample_positions)
        assert np.isclose(sum(optimized_positions.values()), 1.0)
        
        # Weights should be reduced to achieve lower volatility
        for ticker in sample_positions:
            assert optimized_positions[ticker] <= sample_positions[ticker]
    
    def test_calculate_risk_attribution(self, risk_manager, sample_positions):
        """Test risk attribution calculation"""
        risk_metrics = {
            'volatility': 0.15,
            'var_95': -0.02,
            'beta': 1.2
        }
        
        attribution = risk_manager.calculate_risk_attribution(sample_positions, risk_metrics)
        
        assert isinstance(attribution, dict)
        assert len(attribution) == len(sample_positions)
        
        for ticker, metrics in attribution.items():
            assert 'weight' in metrics
            assert 'volatility_contribution' in metrics
            assert 'var_contribution' in metrics
            assert 'beta_contribution' in metrics
    
    def test_get_risk_report(self, risk_manager, sample_positions):
        """Test risk report generation"""
        risk_metrics = {
            'volatility': 0.15,
            'var_95': -0.02,
            'max_drawdown': -0.05
        }
        violations = [
            RiskViolation(
                timestamp="2024-01-01T12:00:00",
                risk_metric=RiskMetric.EXPOSURE,
                current_value=0.4,
                limit_value=0.3,
                severity="warning",
                action_required="Reduce AAPL position",
                affected_positions=["AAPL"]
            )
        ]
        
        report = risk_manager.get_risk_report(sample_positions, risk_metrics, violations)
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'portfolio_summary' in report
        assert 'risk_metrics' in report
        assert 'risk_limits' in report
        assert 'violations' in report
        assert 'sector_exposure' in report
        assert 'risk_attribution' in report


class TestPortfolioIntegration:
    """Integration tests for portfolio modules"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_portfolio_workflow(self, temp_dir):
        """Test complete portfolio workflow"""
        # Create allocator
        allocator_config = {'portfolio': {'max_weight': 0.3, 'min_weight': 0.01}}
        with patch('portfolio.allocator.load_config', return_value=allocator_config):
            allocator = PortfolioAllocator()
        
        # Create risk manager
        risk_config = {'risk_management': {'max_drawdown': 0.15, 'max_exposure': 0.3}}
        with patch('portfolio.risk_manager.load_config', return_value=risk_config):
            risk_manager = PortfolioRiskManager()
        
        # Create sample assets
        assets = [
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
        
        # Allocate portfolio
        allocation_result = allocator.allocate_portfolio(assets, AllocationStrategy.MAXIMUM_SHARPE)
        
        assert allocation_result.strategy == AllocationStrategy.MAXIMUM_SHARPE
        assert len(allocation_result.weights) == 3
        assert allocation_result.constraints_satisfied is True
        
        # Create market data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        market_data = {}
        
        for asset in assets:
            returns = np.random.normal(0.001, asset.volatility/np.sqrt(252), len(dates))
            market_data[asset.ticker] = pd.DataFrame({'returns': returns}, index=dates)
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(allocation_result.weights, market_data)
        
        assert 'volatility' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        
        # Check risk limits
        violations = risk_manager.check_risk_limits(allocation_result.weights, risk_metrics)
        
        assert isinstance(violations, list)
        
        # Generate rebalancing actions if needed
        if violations:
            actions = risk_manager.generate_rebalancing_actions(
                allocation_result.weights, 
                allocation_result.weights, 
                violations
            )
            assert isinstance(actions, list)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test create_allocator
        with patch('portfolio.allocator.PortfolioAllocator') as mock_allocator:
            mock_instance = Mock()
            mock_allocator.return_value = mock_instance
            
            allocator = create_allocator()
            assert allocator == mock_instance
        
        # Test create_risk_manager
        with patch('portfolio.risk_manager.PortfolioRiskManager') as mock_risk_manager:
            mock_instance = Mock()
            mock_risk_manager.return_value = mock_instance
            
            risk_manager = create_risk_manager()
            assert risk_manager == mock_instance
        
        # Test allocate_portfolio
        with patch('portfolio.allocator.PortfolioAllocator') as mock_allocator:
            mock_instance = Mock()
            mock_instance.allocate_portfolio.return_value = Mock()
            mock_allocator.return_value = mock_instance
            
            assets = [Mock()]
            result = allocate_portfolio(assets, AllocationStrategy.EQUAL_WEIGHT)
            assert result is not None
        
        # Test calculate_portfolio_risk
        with patch('portfolio.risk_manager.PortfolioRiskManager') as mock_risk_manager:
            mock_instance = Mock()
            mock_instance.calculate_portfolio_risk.return_value = {'volatility': 0.15}
            mock_risk_manager.return_value = mock_instance
            
            positions = {'AAPL': 0.5, 'TSLA': 0.5}
            market_data = {'AAPL': pd.DataFrame(), 'TSLA': pd.DataFrame()}
            risk_metrics = calculate_portfolio_risk(positions, market_data)
            assert 'volatility' in risk_metrics
        
        # Test check_risk_limits
        with patch('portfolio.risk_manager.PortfolioRiskManager') as mock_risk_manager:
            mock_instance = Mock()
            mock_instance.check_risk_limits.return_value = []
            mock_risk_manager.return_value = mock_instance
            
            positions = {'AAPL': 0.5, 'TSLA': 0.5}
            risk_metrics = {'volatility': 0.15}
            violations = check_risk_limits(positions, risk_metrics)
            assert isinstance(violations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 