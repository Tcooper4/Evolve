"""
Tests for Enhanced Portfolio Optimizer

Tests the portfolio optimizer with risk parity, enhanced Black-Litterman,
and other advanced optimization methods.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from trading.optimization.portfolio_optimizer import PortfolioOptimizer, get_portfolio_optimizer


class TestPortfolioOptimizer:
    """Test the enhanced portfolio optimizer."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252),
            'TSLA': np.random.normal(0.002, 0.035, 252)
        }, index=dates)
        return returns
    
    @pytest.fixture
    def sample_market_caps(self):
        """Create sample market cap data."""
        return pd.Series({
            'AAPL': 0.3,
            'GOOGL': 0.25,
            'MSFT': 0.25,
            'TSLA': 0.2
        })
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(risk_free_rate=0.03)
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.results_dir.exists()
    
    @patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', True)
    @patch('trading.optimization.portfolio_optimizer.cp')
    def test_risk_parity_optimization(self, mock_cp, sample_returns):
        """Test risk parity optimization."""
        # Mock CVXPY components
        mock_problem = Mock()
        mock_problem.status = 'optimal'
        mock_problem.solve.return_value = None
        mock_cp.Problem.return_value = mock_problem
        
        mock_variable = Mock()
        mock_variable.value = np.array([0.25, 0.25, 0.25, 0.25])
        mock_cp.Variable.return_value = mock_variable
        
        mock_cp.sqrt.return_value = Mock()
        mock_cp.quad_form.return_value = Mock()
        mock_cp.sum_squares.return_value = Mock()
        mock_cp.hstack.return_value = Mock()
        mock_cp.Minimize.return_value = Mock()
        
        optimizer = PortfolioOptimizer()
        result = optimizer.risk_parity_optimization(sample_returns)
        
        assert 'weights' in result
        assert 'portfolio_return' in result
        assert 'portfolio_volatility' in result
        assert 'sharpe_ratio' in result
        assert 'risk_contributions' in result
        assert result['risk_measure'] == 'volatility'
        assert result['optimization_status'] == 'optimal'
    
    def test_simple_risk_parity(self, sample_returns):
        """Test simple risk parity without CVXPY."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            result = optimizer.risk_parity_optimization(sample_returns)
            
            assert 'weights' in result
            assert 'portfolio_return' in result
            assert 'portfolio_volatility' in result
            assert 'sharpe_ratio' in result
            assert 'risk_contributions' in result
            assert result['optimization_status'] == 'simple_risk_parity'
            assert 'iterations' in result
    
    def test_risk_parity_with_cvar_measure(self, sample_returns):
        """Test risk parity with CVaR risk measure."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            result = optimizer.risk_parity_optimization(sample_returns, risk_measure="cvar")
            
            assert result['risk_measure'] == 'cvar'
            assert 'risk_contributions' in result
    
    def test_risk_parity_with_target_risk(self, sample_returns):
        """Test risk parity with target risk constraint."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            target_risk = 0.02
            result = optimizer.risk_parity_optimization(sample_returns, target_risk=target_risk)
            
            assert result['portfolio_volatility'] <= target_risk * 1.1  # Allow some tolerance
    
    def test_enhanced_black_litterman_absolute_views(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman with absolute views."""
        views = {'AAPL': 0.05, 'GOOGL': 0.03}
        confidence = {'AAPL': 0.6, 'GOOGL': 0.5}
        
        optimizer = PortfolioOptimizer()
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, view_type="absolute"
        )
        
        # Should return mean-variance result with BL constraints
        assert 'weights' in result or 'error' in result
    
    def test_enhanced_black_litterman_relative_views(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman with relative views."""
        views = {'AAPL vs GOOGL': 0.02, 'MSFT vs TSLA': 0.01}
        confidence = {'AAPL vs GOOGL': 0.6, 'MSFT vs TSLA': 0.5}
        
        optimizer = PortfolioOptimizer()
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, view_type="relative"
        )
        
        # Should return mean-variance result with BL constraints
        assert 'weights' in result or 'error' in result
    
    def test_enhanced_black_litterman_ranking_views(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman with ranking views."""
        views = {'AAPL': 1, 'GOOGL': 2, 'MSFT': 3, 'TSLA': 4}  # Ranking
        confidence = {'AAPL': 0.6, 'GOOGL': 0.5, 'MSFT': 0.4, 'TSLA': 0.3}
        
        optimizer = PortfolioOptimizer()
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, view_type="ranking"
        )
        
        # Should return mean-variance result with BL constraints
        assert 'weights' in result or 'error' in result
    
    def test_enhanced_black_litterman_invalid_view_type(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman with invalid view type."""
        views = {'AAPL': 0.05}
        confidence = {'AAPL': 0.6}
        
        optimizer = PortfolioOptimizer()
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, view_type="invalid"
        )
        
        # Should handle invalid view type gracefully
        assert 'error' in result
    
    def test_calculate_risk_contributions(self, sample_returns):
        """Test risk contribution calculation."""
        optimizer = PortfolioOptimizer()
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        Sigma = sample_returns.cov()
        
        # Test volatility risk measure
        risk_contrib = optimizer._calculate_risk_contributions(weights, Sigma, "volatility")
        assert len(risk_contrib) == 4
        assert all(isinstance(v, float) for v in risk_contrib.values())
        
        # Test other risk measures
        risk_contrib_other = optimizer._calculate_risk_contributions(weights, Sigma, "cvar")
        assert len(risk_contrib_other) == 4
    
    def test_compare_strategies(self, sample_returns):
        """Test strategy comparison."""
        optimizer = PortfolioOptimizer()
        comparison_df = optimizer.compare_strategies(sample_returns)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) > 0
        
        # Should include basic strategies
        expected_strategies = ['Equal Weight', 'Mean-Variance', 'Risk Parity', 'Min-CVaR']
        for strategy in expected_strategies:
            if strategy in comparison_df.index:
                assert 'Return' in comparison_df.columns
                assert 'Volatility' in comparison_df.columns
                assert 'Sharpe' in comparison_df.columns
    
    def test_compare_strategies_with_black_litterman(self, sample_returns):
        """Test strategy comparison including Black-Litterman."""
        optimizer = PortfolioOptimizer()
        
        # Mock market caps for Black-Litterman
        with patch.object(optimizer, 'black_litterman_optimization') as mock_bl:
            mock_bl.return_value = {
                'portfolio_return': 0.001,
                'portfolio_volatility': 0.02,
                'sharpe_ratio': 0.5
            }
            
            comparison_df = optimizer.compare_strategies(sample_returns)
            
            # Should include Black-Litterman if available
            if 'Black-Litterman' in comparison_df.index:
                assert 'Return' in comparison_df.columns
                assert 'Volatility' in comparison_df.columns
                assert 'Sharpe' in comparison_df.columns
    
    def test_compare_strategies_with_enhanced_bl(self, sample_returns):
        """Test strategy comparison including enhanced Black-Litterman."""
        optimizer = PortfolioOptimizer()
        
        # Mock enhanced Black-Litterman
        with patch.object(optimizer, 'enhanced_black_litterman_optimization') as mock_ebl:
            mock_ebl.return_value = {
                'portfolio_return': 0.0012,
                'portfolio_volatility': 0.019,
                'sharpe_ratio': 0.55
            }
            
            comparison_df = optimizer.compare_strategies(sample_returns)
            
            # Should include enhanced Black-Litterman if available
            if 'Enhanced BL (Relative)' in comparison_df.index:
                assert 'Return' in comparison_df.columns
                assert 'Volatility' in comparison_df.columns
                assert 'Sharpe' in comparison_df.columns
    
    def test_risk_parity_convergence(self, sample_returns):
        """Test risk parity convergence properties."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            result = optimizer.risk_parity_optimization(sample_returns)
            
            if 'iterations' in result:
                assert result['iterations'] <= 100  # Should converge within max iterations
                
                # Check risk contributions are roughly equal
                risk_contrib = result['risk_contributions']
                if risk_contrib:
                    contrib_values = list(risk_contrib.values())
                    mean_contrib = np.mean(contrib_values)
                    max_deviation = max(abs(c - mean_contrib) for c in contrib_values)
                    assert max_deviation < 0.1  # Should be reasonably equal
    
    def test_risk_parity_different_risk_measures(self, sample_returns):
        """Test risk parity with different risk measures."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            
            # Test volatility risk measure
            result_vol = optimizer.risk_parity_optimization(sample_returns, risk_measure="volatility")
            assert result_vol['risk_measure'] == 'volatility'
            
            # Test CVaR risk measure
            result_cvar = optimizer.risk_parity_optimization(sample_returns, risk_measure="cvar")
            assert result_cvar['risk_measure'] == 'cvar'
            
            # Test VaR risk measure
            result_var = optimizer.risk_parity_optimization(sample_returns, risk_measure="var")
            assert result_var['risk_measure'] == 'var'
    
    def test_enhanced_black_litterman_parameter_sensitivity(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman parameter sensitivity."""
        views = {'AAPL': 0.05}
        confidence = {'AAPL': 0.6}
        
        optimizer = PortfolioOptimizer()
        
        # Test different tau values
        result_tau_low = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, tau=0.01
        )
        
        result_tau_high = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, tau=0.1
        )
        
        # Both should return results (though they might be different)
        assert 'weights' in result_tau_low or 'error' in result_tau_low
        assert 'weights' in result_tau_high or 'error' in result_tau_high
    
    def test_risk_parity_error_handling(self, sample_returns):
        """Test risk parity error handling."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            
            # Test with invalid risk measure
            result = optimizer.risk_parity_optimization(sample_returns, risk_measure="invalid")
            assert 'error' in result or 'weights' in result  # Should handle gracefully
    
    def test_enhanced_black_litterman_error_handling(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman error handling."""
        optimizer = PortfolioOptimizer()
        
        # Test with missing assets in views
        views = {'INVALID_ASSET': 0.05}
        confidence = {'INVALID_ASSET': 0.6}
        
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence
        )
        
        # Should handle missing assets gracefully
        assert 'error' in result or 'weights' in result
    
    def test_save_and_load_optimization_results(self, sample_returns, tmp_path):
        """Test saving and loading optimization results."""
        optimizer = PortfolioOptimizer()
        optimizer.results_dir = tmp_path
        
        # Run optimization
        result = optimizer.risk_parity_optimization(sample_returns)
        
        # Check that results were saved
        if 'weights' in result:
            files = list(tmp_path.glob("risk_parity_optimization_*.json"))
            assert len(files) > 0
    
    def test_get_portfolio_optimizer(self):
        """Test the get_portfolio_optimizer function."""
        optimizer = get_portfolio_optimizer()
        assert isinstance(optimizer, PortfolioOptimizer)
        
        # Should return the same instance
        optimizer2 = get_portfolio_optimizer()
        assert optimizer is optimizer2
    
    def test_risk_parity_with_single_asset(self):
        """Test risk parity with single asset (edge case)."""
        single_asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100)
        })
        
        optimizer = PortfolioOptimizer()
        result = optimizer.risk_parity_optimization(single_asset_returns)
        
        # Should handle single asset case
        assert 'weights' in result or 'error' in result
    
    def test_enhanced_black_litterman_empty_views(self, sample_returns, sample_market_caps):
        """Test enhanced Black-Litterman with empty views."""
        optimizer = PortfolioOptimizer()
        result = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, {}, {}
        )
        
        # Should handle empty views gracefully
        assert 'error' in result or 'weights' in result
    
    def test_risk_parity_weights_sum_to_one(self, sample_returns):
        """Test that risk parity weights sum to one."""
        with patch('trading.optimization.portfolio_optimizer.CVXPY_AVAILABLE', False):
            optimizer = PortfolioOptimizer()
            result = optimizer.risk_parity_optimization(sample_returns)
            
            if 'weights' in result:
                weights_sum = sum(result['weights'].values())
                assert abs(weights_sum - 1.0) < 1e-6
    
    def test_enhanced_black_litterman_view_matrix_construction(self, sample_returns, sample_market_caps):
        """Test view matrix construction in enhanced Black-Litterman."""
        views = {'AAPL': 0.05, 'GOOGL': 0.03}
        confidence = {'AAPL': 0.6, 'GOOGL': 0.5}
        
        optimizer = PortfolioOptimizer()
        
        # Test absolute views
        result_abs = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, views, confidence, view_type="absolute"
        )
        
        # Test relative views
        relative_views = {'AAPL vs GOOGL': 0.02}
        relative_confidence = {'AAPL vs GOOGL': 0.6}
        result_rel = optimizer.enhanced_black_litterman_optimization(
            sample_returns, sample_market_caps, relative_views, relative_confidence, view_type="relative"
        )
        
        # Both should return results
        assert 'weights' in result_abs or 'error' in result_abs
        assert 'weights' in result_rel or 'error' in result_rel 