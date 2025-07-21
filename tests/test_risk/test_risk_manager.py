"""
Tests for Enhanced Risk Manager

Tests the risk manager with VaR, CVaR, stress testing, and other advanced risk measures.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading.risk.risk_manager import RiskManager, StressTestResult


class TestRiskManager:
    """Test the enhanced risk manager."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns

    @pytest.fixture
    def sample_benchmark_returns(self):
        """Create sample benchmark returns data."""
        np.random.seed(43)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
        return returns

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        config = {"risk_free_rate": 0.03, "max_position_size": 0.2}
        risk_manager = RiskManager(config)

        assert risk_manager.config == config
        assert risk_manager.returns is None
        assert risk_manager.benchmark_returns is None
        assert risk_manager.current_metrics is None
        assert len(risk_manager.metrics_history) == 0

    def test_update_returns(self, sample_returns, sample_benchmark_returns):
        """Test updating returns data."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns, sample_benchmark_returns)

        assert risk_manager.returns is not None
        assert risk_manager.benchmark_returns is not None
        assert risk_manager.current_metrics is not None
        assert len(risk_manager.metrics_history) == 1

    def test_calculate_metrics(self, sample_returns):
        """Test metric calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        metrics = risk_manager.current_metrics
        assert metrics is not None
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "var_95")
        assert hasattr(metrics, "cvar_95")
        assert hasattr(metrics, "var_99")
        assert hasattr(metrics, "cvar_99")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "volatility")
        assert hasattr(metrics, "beta")
        assert hasattr(metrics, "correlation")
        assert hasattr(metrics, "kelly_fraction")
        assert hasattr(metrics, "expected_shortfall")
        assert hasattr(metrics, "tail_risk")
        assert hasattr(metrics, "skewness")
        assert hasattr(metrics, "kurtosis")

    def test_calculate_historical_var(self, sample_returns):
        """Test historical VaR calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        var_95 = risk_manager.calculate_historical_var(0.95)
        var_99 = risk_manager.calculate_historical_var(0.99)

        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 <= var_95  # 99% VaR should be more extreme than 95% VaR

    def test_calculate_parametric_var(self, sample_returns):
        """Test parametric VaR calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        var_95 = risk_manager.calculate_parametric_var(0.95)
        var_99 = risk_manager.calculate_parametric_var(0.99)

        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 <= var_95  # 99% VaR should be more extreme than 95% VaR

    def test_calculate_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        var_95 = risk_manager.calculate_monte_carlo_var(0.95, n_simulations=1000)
        var_99 = risk_manager.calculate_monte_carlo_var(0.99, n_simulations=1000)

        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 <= var_95  # 99% VaR should be more extreme than 95% VaR

    def test_calculate_conditional_var(self, sample_returns):
        """Test conditional VaR calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        cvar_95 = risk_manager.calculate_conditional_var(0.95)
        cvar_99 = risk_manager.calculate_conditional_var(0.99)

        assert isinstance(cvar_95, float)
        assert isinstance(cvar_99, float)
        assert cvar_99 <= cvar_95  # 99% CVaR should be more extreme than 95% CVaR

    def test_calculate_expected_shortfall(self, sample_returns):
        """Test expected shortfall calculation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        es_95 = risk_manager.calculate_expected_shortfall(0.95)
        es_99 = risk_manager.calculate_expected_shortfall(0.99)

        assert isinstance(es_95, float)
        assert isinstance(es_99, float)
        assert es_99 <= es_95  # 99% ES should be more extreme than 95% ES

    def test_run_stress_tests(self, sample_returns):
        """Test stress testing functionality."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        stress_results = risk_manager.run_stress_tests(portfolio_value=1000000.0)

        assert isinstance(stress_results, list)
        assert len(stress_results) > 0

        # Check that all scenarios were tested
        scenario_names = [result.scenario_name for result in stress_results]
        expected_scenarios = [
            "Market Crash",
            "Interest Rate Hike",
            "Liquidity Crisis",
            "Currency Crisis",
            "Geopolitical Risk",
            "Economic Recession",
            "Technology Bubble Burst",
            "Oil Price Shock",
        ]

        for scenario in expected_scenarios:
            assert scenario in scenario_names

        # Check stress test result structure
        for result in stress_results:
            assert isinstance(result, StressTestResult)
            assert hasattr(result, "scenario_name")
            assert hasattr(result, "portfolio_value_change")
            assert hasattr(result, "var_change")
            assert hasattr(result, "cvar_change")
            assert hasattr(result, "max_drawdown_change")
            assert hasattr(result, "sharpe_ratio_change")
            assert hasattr(result, "timestamp")

    def test_stress_test_scenarios(self, sample_returns):
        """Test different stress test scenarios."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        stress_results = risk_manager.run_stress_tests()

        # Check that market crash scenario has significant impact
        market_crash = next(
            (r for r in stress_results if r.scenario_name == "Market Crash"), None
        )
        assert market_crash is not None
        assert market_crash.portfolio_value_change < 0  # Should be negative
        assert market_crash.var_change < 0  # VaR should worsen
        assert market_crash.cvar_change < 0  # CVaR should worsen

    def test_get_position_limits(self, sample_returns):
        """Test position limit calculation."""
        risk_manager = RiskManager({"max_position_size": 0.2, "max_leverage": 1.0})
        risk_manager.update_returns(sample_returns)

        limits = risk_manager.get_position_limits()

        assert isinstance(limits, dict)
        assert "position_limit" in limits
        assert "leverage_limit" in limits
        assert "kelly_fraction" in limits

        assert limits["position_limit"] > 0
        assert limits["leverage_limit"] > 0
        assert limits["kelly_fraction"] >= 0

    def test_optimize_position_sizes(self, sample_returns):
        """Test position size optimization."""
        risk_manager = RiskManager()

        # Create sample expected returns and covariance
        assets = ["AAPL", "GOOGL", "MSFT"]
        expected_returns = pd.Series([0.1, 0.12, 0.08], index=assets)
        covariance = pd.DataFrame(
            {
                "AAPL": [0.04, 0.02, 0.01],
                "GOOGL": [0.02, 0.09, 0.015],
                "MSFT": [0.01, 0.015, 0.06],
            },
            index=assets,
        )

        weights = risk_manager.optimize_position_sizes(expected_returns, covariance)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(assets)
        assert abs(weights.sum() - 1.0) < 1e-6  # Should sum to 1
        assert all(weights >= 0)  # Should be non-negative

    def test_generate_risk_summary(self, sample_returns):
        """Test risk summary generation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        summary = risk_manager.generate_risk_summary()

        assert isinstance(summary, dict)
        assert "overall_risk_score" in summary
        assert "risk_level" in summary
        assert "key_metrics" in summary
        assert "risk_breakdown" in summary
        assert "recommendations" in summary

        assert 0 <= summary["overall_risk_score"] <= 100
        assert summary["risk_level"] in ["Low", "Medium", "High"]
        assert isinstance(summary["recommendations"], list)
        assert len(summary["recommendations"]) > 0

    def test_risk_recommendations(self, sample_returns):
        """Test risk recommendation generation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        # Test high risk scenario
        high_risk_recommendations = risk_manager._generate_risk_recommendations(80.0)
        assert len(high_risk_recommendations) > 0
        assert any(
            "reducing position sizes" in rec.lower()
            for rec in high_risk_recommendations
        )

        # Test medium risk scenario
        medium_risk_recommendations = risk_manager._generate_risk_recommendations(50.0)
        assert len(medium_risk_recommendations) > 0
        assert any(
            "moderate position sizes" in rec.lower()
            for rec in medium_risk_recommendations
        )

        # Test low risk scenario
        low_risk_recommendations = risk_manager._generate_risk_recommendations(10.0)
        assert len(low_risk_recommendations) > 0
        assert any("acceptable" in rec.lower() for rec in low_risk_recommendations)

    def test_export_risk_report(self, sample_returns, tmp_path):
        """Test risk report export."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        report_file = tmp_path / "risk_report.json"
        result = risk_manager.export_risk_report(str(report_file))

        assert result["success"] is True
        assert report_file.exists()

        # Check report content
        import json

        with open(report_file, "r") as f:
            report_data = json.load(f)

        assert "current_metrics" in report_data
        assert "additional_risk_measures" in report_data
        assert "stress_test_results" in report_data
        assert "risk_summary" in report_data
        assert "returns_summary" in report_data
        assert "data_summary" in report_data

    def test_plot_risk_metrics(self, sample_returns):
        """Test risk metrics plotting."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        # Add some history
        for _ in range(5):
            risk_manager.update_returns(sample_returns)

        fig = risk_manager.plot_risk_metrics()

        assert fig is not None
        # Additional assertions could be added for plotly figure structure

    def test_cleanup_old_results(self, tmp_path):
        """Test cleanup of old result files."""
        risk_manager = RiskManager()

        # Create some dummy files
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        for i in range(10):
            (results_dir / f"result_{i}.json").write_text("dummy content")

        with patch.object(risk_manager, "logger"):
            result = risk_manager.cleanup_old_results(max_files=5)

        # Should have removed 5 files
        remaining_files = list(results_dir.glob("*.json"))
        assert len(remaining_files) == 5

    def test_risk_metrics_edge_cases(self):
        """Test risk metrics with edge cases."""
        risk_manager = RiskManager()

        # Test with empty returns
        empty_returns = pd.Series(dtype=float)
        risk_manager.update_returns(empty_returns)

        # Should handle gracefully
        assert risk_manager.current_metrics is None

        # Test with constant returns
        constant_returns = pd.Series([0.001] * 100)
        risk_manager.update_returns(constant_returns)

        # Should calculate metrics even with constant returns
        assert risk_manager.current_metrics is not None
        assert risk_manager.current_metrics.volatility == 0.0

    def test_var_methods_comparison(self, sample_returns):
        """Test comparison of different VaR calculation methods."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        historical_var = risk_manager.calculate_historical_var(0.95)
        parametric_var = risk_manager.calculate_parametric_var(0.95)
        monte_carlo_var = risk_manager.calculate_monte_carlo_var(0.95)

        # All methods should return reasonable values
        assert isinstance(historical_var, float)
        assert isinstance(parametric_var, float)
        assert isinstance(monte_carlo_var, float)

        # Values should be in reasonable range for normal returns
        assert -0.1 < historical_var < 0.1
        assert -0.1 < parametric_var < 0.1
        assert -0.1 < monte_carlo_var < 0.1

    def test_stress_test_portfolio_value_impact(self, sample_returns):
        """Test portfolio value impact in stress tests."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        portfolio_value = 1000000.0
        stress_results = risk_manager.run_stress_tests(portfolio_value)

        for result in stress_results:
            # Portfolio value change should be reasonable
            assert -portfolio_value < result.portfolio_value_change < portfolio_value

            # Most scenarios should have negative impact
            if result.scenario_name in [
                "Market Crash",
                "Liquidity Crisis",
                "Currency Crisis",
            ]:
                assert result.portfolio_value_change < 0

    def test_risk_metrics_with_benchmark(
        self, sample_returns, sample_benchmark_returns
    ):
        """Test risk metrics calculation with benchmark."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns, sample_benchmark_returns)

        metrics = risk_manager.current_metrics
        assert metrics is not None
        assert metrics.beta != 1.0  # Should be different from default
        assert -1.0 <= metrics.correlation <= 1.0

    def test_risk_metrics_history(self, sample_returns):
        """Test risk metrics history tracking."""
        risk_manager = RiskManager()

        # Update returns multiple times
        for i in range(5):
            risk_manager.update_returns(sample_returns)

        assert len(risk_manager.metrics_history) == 5

        # Check that metrics are being tracked over time
        timestamps = [m.timestamp for m in risk_manager.metrics_history]
        assert len(set(timestamps)) == 5  # All timestamps should be unique

    def test_risk_summary_risk_levels(self, sample_returns):
        """Test risk level classification in summary."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        # Test different risk scores
        high_risk_summary = risk_manager._generate_risk_summary_with_score(85.0)
        assert high_risk_summary["risk_level"] == "High"

        medium_risk_summary = risk_manager._generate_risk_summary_with_score(50.0)
        assert medium_risk_summary["risk_level"] == "Medium"

        low_risk_summary = risk_manager._generate_risk_summary_with_score(15.0)
        assert low_risk_summary["risk_level"] == "Low"

    def _generate_risk_summary_with_score(self, risk_score):
        """Helper method to test risk summary with specific score."""
        risk_manager = RiskManager()
        risk_manager.update_returns(pd.Series([0.001] * 100))

        # Mock the risk score calculation
        def mock_generate_risk_summary():
            return {
                "overall_risk_score": risk_score,
                "risk_level": (
                    "High"
                    if risk_score > 70
                    else "Medium" if risk_score > 30 else "Low"
                ),
                "key_metrics": {},
                "risk_breakdown": {},
                "recommendations": [],
            }

        with patch.object(
            risk_manager, "generate_risk_summary", mock_generate_risk_summary
        ):
            return risk_manager.generate_risk_summary()

    def test_stress_test_error_handling(self):
        """Test stress test error handling."""
        risk_manager = RiskManager()

        # Test without returns data
        stress_results = risk_manager.run_stress_tests()
        assert stress_results == []

    def test_var_calculation_error_handling(self):
        """Test VaR calculation error handling."""
        risk_manager = RiskManager()

        # Test without returns data
        var_95 = risk_manager.calculate_historical_var(0.95)
        assert var_95 == 0.0

        parametric_var = risk_manager.calculate_parametric_var(0.95)
        assert parametric_var == 0.0

        monte_carlo_var = risk_manager.calculate_monte_carlo_var(0.95)
        assert monte_carlo_var == 0.0

    def test_risk_metrics_validation(self, sample_returns):
        """Test risk metrics validation."""
        risk_manager = RiskManager()
        risk_manager.update_returns(sample_returns)

        metrics = risk_manager.current_metrics

        # Check that metrics are within reasonable bounds
        assert -10 <= metrics.sharpe_ratio <= 10
        assert -10 <= metrics.sortino_ratio <= 10
        assert -1 <= metrics.max_drawdown <= 0
        assert 0 <= metrics.volatility <= 1
        assert -1 <= metrics.correlation <= 1
        assert 0 <= metrics.tail_risk <= 1
        assert -10 <= metrics.skewness <= 10
        assert 0 <= metrics.kurtosis <= 100
