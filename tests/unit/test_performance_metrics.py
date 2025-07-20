"""
Unit tests for custom performance metrics module.

This module tests all the financial performance metrics functions
to ensure they work correctly and match expected behavior.
"""

import numpy as np
import pandas as pd
import pytest
from utils.performance_metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cumulative_return,
    calmar_ratio,
    avg_drawdown,
    drawdown_details,
    omega_ratio,
    information_ratio,
    treynor_ratio,
    jensen_alpha,
    value_at_risk,
    conditional_value_at_risk,
    downside_deviation,
    gain_loss_ratio,
    profit_factor,
    recovery_factor,
    risk_reward_ratio
)


class TestPerformanceMetrics:
    """Test suite for performance metrics functions."""

    def setup_method(self):
        """Set up test data."""
        # Create sample return series
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
        self.negative_returns = pd.Series(np.random.normal(-0.001, 0.02, 252))
        self.zero_returns = pd.Series([0.0] * 252)
        self.empty_returns = pd.Series([])
        
        # Create benchmark returns
        self.benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test basic calculation
        sharpe = sharpe_ratio(self.returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test with risk-free rate
        sharpe_rf = sharpe_ratio(self.returns, risk_free=0.02, period=252)
        assert isinstance(sharpe_rf, float)
        assert not np.isnan(sharpe_rf)
        
        # Test with zero returns
        sharpe_zero = sharpe_ratio(self.zero_returns)
        assert sharpe_zero == 0.0
        
        # Test with empty returns
        sharpe_empty = sharpe_ratio(self.empty_returns)
        assert sharpe_empty == 0.0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Test basic calculation
        sortino = sortino_ratio(self.returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        
        # Test with risk-free rate
        sortino_rf = sortino_ratio(self.returns, risk_free=0.02, period=252)
        assert isinstance(sortino_rf, float)
        assert not np.isnan(sortino_rf)
        
        # Test with zero returns
        sortino_zero = sortino_ratio(self.zero_returns)
        assert sortino_zero == 0.0
        
        # Test with empty returns
        sortino_empty = sortino_ratio(self.empty_returns)
        assert sortino_empty == 0.0

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Test basic calculation
        max_dd = max_drawdown(self.returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0.0  # Drawdown should be negative or zero
        
        # Test with zero returns
        max_dd_zero = max_drawdown(self.zero_returns)
        assert max_dd_zero == 0.0
        
        # Test with empty returns
        max_dd_empty = max_drawdown(self.empty_returns)
        assert max_dd_empty == 0.0
        
        # Test with known drawdown
        test_returns = pd.Series([0.1, -0.05, 0.02, -0.1, 0.03])
        max_dd_known = max_drawdown(test_returns)
        assert max_dd_known < 0.0

    def test_cumulative_return(self):
        """Test cumulative return calculation."""
        # Test basic calculation
        cum_ret = cumulative_return(self.returns)
        assert isinstance(cum_ret, float)
        assert not np.isnan(cum_ret)
        
        # Test with zero returns
        cum_ret_zero = cumulative_return(self.zero_returns)
        assert cum_ret_zero == 0.0
        
        # Test with empty returns
        cum_ret_empty = cumulative_return(self.empty_returns)
        assert cum_ret_empty == 0.0
        
        # Test with known returns
        test_returns = pd.Series([0.1, 0.05, -0.02, 0.03])
        cum_ret_known = cumulative_return(test_returns)
        expected = (1.1 * 1.05 * 0.98 * 1.03) - 1
        assert abs(cum_ret_known - expected) < 1e-10

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        # Test basic calculation
        calmar = calmar_ratio(self.returns)
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)
        
        # Test with risk-free rate
        calmar_rf = calmar_ratio(self.returns, risk_free=0.02, period=252)
        assert isinstance(calmar_rf, float)
        assert not np.isnan(calmar_rf)
        
        # Test with zero returns
        calmar_zero = calmar_ratio(self.zero_returns)
        assert calmar_zero == 0.0
        
        # Test with empty returns
        calmar_empty = calmar_ratio(self.empty_returns)
        assert calmar_empty == 0.0

    def test_avg_drawdown(self):
        """Test average drawdown calculation."""
        # Test basic calculation
        avg_dd = avg_drawdown(self.returns)
        assert isinstance(avg_dd, float)
        assert avg_dd <= 0.0  # Average drawdown should be negative or zero
        
        # Test with zero returns
        avg_dd_zero = avg_drawdown(self.zero_returns)
        assert avg_dd_zero == 0.0
        
        # Test with empty returns
        avg_dd_empty = avg_drawdown(self.empty_returns)
        assert avg_dd_empty == 0.0

    def test_drawdown_details(self):
        """Test drawdown details calculation."""
        # Test basic calculation
        dd_details = drawdown_details(self.returns)
        assert isinstance(dd_details, pd.DataFrame)
        
        # Test with zero returns
        dd_details_zero = drawdown_details(self.zero_returns)
        assert isinstance(dd_details_zero, pd.DataFrame)
        
        # Test with empty returns
        dd_details_empty = drawdown_details(self.empty_returns)
        assert isinstance(dd_details_empty, pd.DataFrame)
        assert dd_details_empty.empty

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        # Test basic calculation
        var_95 = value_at_risk(self.returns, confidence_level=0.95)
        assert isinstance(var_95, float)
        assert not np.isnan(var_95)
        
        # Test different confidence levels
        var_99 = value_at_risk(self.returns, confidence_level=0.99)
        assert var_99 <= var_95  # 99% VaR should be more extreme than 95% VaR
        
        # Test with zero returns
        var_zero = value_at_risk(self.zero_returns, confidence_level=0.95)
        assert var_zero == 0.0
        
        # Test with empty returns
        var_empty = value_at_risk(self.empty_returns, confidence_level=0.95)
        assert var_empty == 0.0

    def test_conditional_value_at_risk(self):
        """Test Conditional Value at Risk calculation."""
        # Test basic calculation
        cvar_95 = conditional_value_at_risk(self.returns, confidence_level=0.95)
        assert isinstance(cvar_95, float)
        assert not np.isnan(cvar_95)
        
        # Test different confidence levels
        cvar_99 = conditional_value_at_risk(self.returns, confidence_level=0.99)
        assert cvar_99 <= cvar_95  # 99% CVaR should be more extreme than 95% CVaR
        
        # Test with zero returns
        cvar_zero = conditional_value_at_risk(self.zero_returns, confidence_level=0.95)
        assert cvar_zero == 0.0
        
        # Test with empty returns
        cvar_empty = conditional_value_at_risk(self.empty_returns, confidence_level=0.95)
        assert cvar_empty == 0.0

    def test_omega_ratio(self):
        """Test Omega ratio calculation."""
        # Test basic calculation
        omega = omega_ratio(self.returns)
        assert isinstance(omega, float)
        assert not np.isnan(omega)
        
        # Test with threshold
        omega_threshold = omega_ratio(self.returns, threshold=0.001)
        assert isinstance(omega_threshold, float)
        assert not np.isnan(omega_threshold)
        
        # Test with zero returns
        omega_zero = omega_ratio(self.zero_returns)
        assert omega_zero == 0.0
        
        # Test with empty returns
        omega_empty = omega_ratio(self.empty_returns)
        assert omega_empty == 0.0

    def test_information_ratio(self):
        """Test Information ratio calculation."""
        # Test basic calculation
        info_ratio = information_ratio(self.returns, self.benchmark_returns)
        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)
        
        # Test with mismatched lengths
        short_returns = self.returns[:100]
        info_ratio_mismatch = information_ratio(short_returns, self.benchmark_returns)
        assert info_ratio_mismatch == 0.0
        
        # Test with empty returns
        info_ratio_empty = information_ratio(self.empty_returns, self.benchmark_returns)
        assert info_ratio_empty == 0.0

    def test_treynor_ratio(self):
        """Test Treynor ratio calculation."""
        # Test basic calculation
        treynor = treynor_ratio(self.returns, self.benchmark_returns)
        assert isinstance(treynor, float)
        assert not np.isnan(treynor)
        
        # Test with risk-free rate
        treynor_rf = treynor_ratio(self.returns, self.benchmark_returns, risk_free=0.02)
        assert isinstance(treynor_rf, float)
        assert not np.isnan(treynor_rf)
        
        # Test with mismatched lengths
        short_returns = self.returns[:100]
        treynor_mismatch = treynor_ratio(short_returns, self.benchmark_returns)
        assert treynor_mismatch == 0.0

    def test_jensen_alpha(self):
        """Test Jensen's Alpha calculation."""
        # Test basic calculation
        alpha = jensen_alpha(self.returns, self.benchmark_returns)
        assert isinstance(alpha, float)
        assert not np.isnan(alpha)
        
        # Test with risk-free rate
        alpha_rf = jensen_alpha(self.returns, self.benchmark_returns, risk_free=0.02)
        assert isinstance(alpha_rf, float)
        assert not np.isnan(alpha_rf)
        
        # Test with mismatched lengths
        short_returns = self.returns[:100]
        alpha_mismatch = jensen_alpha(short_returns, self.benchmark_returns)
        assert alpha_mismatch == 0.0

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        # Test basic calculation
        downside_dev = downside_deviation(self.returns)
        assert isinstance(downside_dev, float)
        assert not np.isnan(downside_dev)
        
        # Test with threshold
        downside_dev_threshold = downside_deviation(self.returns, threshold=0.001)
        assert isinstance(downside_dev_threshold, float)
        assert not np.isnan(downside_dev_threshold)
        
        # Test with zero returns
        downside_dev_zero = downside_deviation(self.zero_returns)
        assert downside_dev_zero == 0.0
        
        # Test with empty returns
        downside_dev_empty = downside_deviation(self.empty_returns)
        assert downside_dev_empty == 0.0

    def test_gain_loss_ratio(self):
        """Test gain/loss ratio calculation."""
        # Test basic calculation
        gl_ratio = gain_loss_ratio(self.returns)
        assert isinstance(gl_ratio, float)
        assert not np.isnan(gl_ratio)
        
        # Test with zero returns
        gl_ratio_zero = gain_loss_ratio(self.zero_returns)
        assert gl_ratio_zero == 0.0
        
        # Test with empty returns
        gl_ratio_empty = gain_loss_ratio(self.empty_returns)
        assert gl_ratio_empty == 0.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        # Test basic calculation
        pf = profit_factor(self.returns)
        assert isinstance(pf, float)
        assert not np.isnan(pf)
        
        # Test with zero returns
        pf_zero = profit_factor(self.zero_returns)
        assert pf_zero == 0.0
        
        # Test with empty returns
        pf_empty = profit_factor(self.empty_returns)
        assert pf_empty == 0.0

    def test_recovery_factor(self):
        """Test recovery factor calculation."""
        # Test basic calculation
        rf = recovery_factor(self.returns)
        assert isinstance(rf, float)
        assert not np.isnan(rf)
        
        # Test with zero returns
        rf_zero = recovery_factor(self.zero_returns)
        assert rf_zero == 0.0
        
        # Test with empty returns
        rf_empty = recovery_factor(self.empty_returns)
        assert rf_empty == 0.0

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        # Test basic calculation
        rr_ratio = risk_reward_ratio(self.returns)
        assert isinstance(rr_ratio, float)
        assert not np.isnan(rr_ratio)
        
        # Test with zero returns
        rr_ratio_zero = risk_reward_ratio(self.zero_returns)
        assert rr_ratio_zero == 0.0
        
        # Test with empty returns
        rr_ratio_empty = risk_reward_ratio(self.empty_returns)
        assert rr_ratio_empty == 0.0

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with numpy arrays
        returns_array = np.array(self.returns)
        sharpe_array = sharpe_ratio(returns_array)
        assert isinstance(sharpe_array, float)
        
        # Test with single value
        single_return = pd.Series([0.01])
        sharpe_single = sharpe_ratio(single_return)
        assert isinstance(sharpe_single, float)
        
        # Test with all positive returns
        positive_returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        sharpe_positive = sharpe_ratio(positive_returns)
        assert isinstance(sharpe_positive, float)
        
        # Test with all negative returns
        negative_returns = pd.Series([-0.01, -0.02, -0.01, -0.03])
        sharpe_negative = sharpe_ratio(negative_returns)
        assert isinstance(sharpe_negative, float)

    def test_consistency(self):
        """Test consistency between related metrics."""
        # Sharpe and Sortino should be similar for symmetric distributions
        sharpe = sharpe_ratio(self.returns)
        sortino = sortino_ratio(self.returns)
        
        # They should be reasonably close (not necessarily equal due to different denominators)
        assert abs(sharpe - sortino) < 1.0  # Allow some difference
        
        # Max drawdown should be negative or zero
        max_dd = max_drawdown(self.returns)
        assert max_dd <= 0.0
        
        # Average drawdown should be less than or equal to max drawdown
        avg_dd = avg_drawdown(self.returns)
        assert avg_dd >= max_dd
        
        # VaR should be more extreme than CVaR
        var_95 = value_at_risk(self.returns, confidence_level=0.95)
        cvar_95 = conditional_value_at_risk(self.returns, confidence_level=0.95)
        assert cvar_95 <= var_95
