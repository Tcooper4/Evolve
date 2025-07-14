"""
Test Optimization Modular Components

This module tests the modular optimization components.
Updated for the new modular structure.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from trading.agents.optimization.parameter_validator import ParameterValidator, OptimizationParameter
from trading.agents.optimization.strategy_optimizer import (
    StrategyOptimizer, StrategyConfig, OptimizationType, OptimizationMetric,
    OptimizationConfig
)
from trading.agents.optimization.performance_analyzer import PerformanceAnalyzer, OptimizationResult


class TestParameterValidator:
    """Test ParameterValidator functionality."""

    def test_parameter_creation(self):
        """Test optimization parameter creation."""
        param = OptimizationParameter(
            name="rsi_period",
            min_value=10,
            max_value=30,
            step=2,
            parameter_type="int"
        )
        
        assert param.name == "rsi_period"
        assert param.min_value == 10
        assert param.max_value == 30
        assert param.step == 2
        assert param.parameter_type == "int"

    def test_parameter_validation(self):
        """Test parameter validation."""
        validator = ParameterValidator()
        
        # Valid parameter
        valid_param = OptimizationParameter(
            name="rsi_period",
            min_value=10,
            max_value=30,
            step=2,
            parameter_type="int"
        )
        
        validated_params = validator.validate_optimization_parameters([valid_param])
        assert len(validated_params) == 1
        assert validated_params[0].name == "rsi_period"
        
        # Invalid parameter (min >= max)
        invalid_param = OptimizationParameter(
            name="invalid_param",
            min_value=30,
            max_value=10,  # Invalid: min > max
            step=2,
            parameter_type="int"
        )
        
        validated_params = validator.validate_optimization_parameters([invalid_param])
        assert len(validated_params) == 0

    def test_parameter_range_generation(self):
        """Test parameter range generation."""
        validator = ParameterValidator()
        
        param = OptimizationParameter(
            name="rsi_period",
            min_value=10,
            max_value=20,
            step=2,
            parameter_type="int"
        )
        
        param_range = validator.generate_parameter_range(param)
        expected_range = [10, 12, 14, 16, 18, 20]
        
        assert param_range == expected_range

    def test_categorical_parameter(self):
        """Test categorical parameter validation."""
        validator = ParameterValidator()
        
        categorical_param = OptimizationParameter(
            name="strategy_type",
            min_value=0,
            max_value=0,
            step=0,
            parameter_type="categorical",
            categories=["rsi", "macd", "bollinger"]
        )
        
        validated_params = validator.validate_optimization_parameters([categorical_param])
        assert len(validated_params) == 1
        
        param_range = validator.generate_parameter_range(categorical_param)
        assert param_range == ["rsi", "macd", "bollinger"]

    def test_parameter_combination_validation(self):
        """Test parameter combination validation."""
        validator = ParameterValidator()
        
        params = [
            OptimizationParameter(
                name="macd_fast",
                min_value=5,
                max_value=15,
                step=1,
                parameter_type="int"
            ),
            OptimizationParameter(
                name="macd_slow",
                min_value=20,
                max_value=30,
                step=1,
                parameter_type="int"
            )
        ]
        
        # Valid combination
        valid_combination = {"macd_fast": 10, "macd_slow": 25}
        assert validator.validate_parameter_combination(valid_combination, params)
        
        # Invalid combination (fast >= slow)
        invalid_combination = {"macd_fast": 25, "macd_slow": 10}
        assert not validator.validate_parameter_combination(invalid_combination, params)


class TestStrategyOptimizer:
    """Test StrategyOptimizer functionality."""

    def test_strategy_config_creation(self):
        """Test strategy configuration creation."""
        config = StrategyConfig(
            strategy_name="rsi_strategy",
            enabled=True,
            weight=1.0,
            parameters={"rsi_period": 14, "overbought": 70, "oversold": 30}
        )
        
        assert config.strategy_name == "rsi_strategy"
        assert config.enabled is True
        assert config.weight == 1.0
        assert config.parameters["rsi_period"] == 14

    def test_optimization_config_creation(self):
        """Test optimization configuration creation."""
        strategy_configs = [
            StrategyConfig(strategy_name="rsi_strategy", enabled=True),
            StrategyConfig(strategy_name="macd_strategy", enabled=True)
        ]
        
        parameters = [
            OptimizationParameter(
                name="rsi_period",
                min_value=10,
                max_value=30,
                step=2,
                parameter_type="int"
            )
        ]
        
        config = OptimizationConfig(
            optimization_type=OptimizationType.THRESHOLD_OPTIMIZATION,
            target_metric=OptimizationMetric.SHARPE_RATIO,
            symbols=["AAPL", "GOOGL"],
            time_periods=[{"start_date": "2023-01-01", "end_date": "2023-12-31"}],
            strategy_configs=strategy_configs,
            parameters_to_optimize=parameters
        )
        
        assert config.optimization_type == OptimizationType.THRESHOLD_OPTIMIZATION
        assert config.target_metric == OptimizationMetric.SHARPE_RATIO
        assert len(config.symbols) == 2
        assert len(config.strategy_configs) == 2

    def test_strategy_combinations_generation(self):
        """Test strategy combinations generation."""
        optimizer = StrategyOptimizer({})
        
        strategy_configs = [
            StrategyConfig(strategy_name="rsi_strategy", enabled=True),
            StrategyConfig(strategy_name="macd_strategy", enabled=True),
            StrategyConfig(strategy_name="bollinger_strategy", enabled=True)
        ]
        
        combinations = optimizer._generate_strategy_combinations(strategy_configs)
        
        # Should generate combinations of 1, 2, and 3 strategies
        assert len(combinations) == 7  # 3C1 + 3C2 + 3C3 = 3 + 3 + 1 = 7

    def test_parameter_combinations_generation(self):
        """Test parameter combinations generation."""
        optimizer = StrategyOptimizer({})
        
        parameters = [
            OptimizationParameter(
                name="rsi_period",
                min_value=10,
                max_value=12,
                step=1,
                parameter_type="int"
            ),
            OptimizationParameter(
                name="overbought",
                min_value=70,
                max_value=75,
                step=5,
                parameter_type="int"
            )
        ]
        
        combinations = optimizer._generate_parameter_combinations(parameters)
        
        # Should generate 3 * 2 = 6 combinations
        assert len(combinations) == 6

    def test_optimization_score_calculation(self):
        """Test optimization score calculation."""
        optimizer = StrategyOptimizer({})
        
        metrics = {
            "sharpe_ratio": 1.5,
            "total_return": 0.15,
            "max_drawdown": 0.10,
            "win_rate": 0.65
        }
        
        # Test Sharpe ratio optimization
        score = optimizer._calculate_optimization_score(
            metrics, OptimizationMetric.SHARPE_RATIO
        )
        assert score == 1.5
        
        # Test composite score
        score = optimizer._calculate_optimization_score(
            metrics, OptimizationMetric.COMPOSITE_SCORE
        )
        assert score > 0  # Should be positive


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer functionality."""

    def test_optimization_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            parameter_combination={"rsi_period": 14, "overbought": 70},
            performance_metrics={"sharpe_ratio": 1.5, "total_return": 0.15},
            backtest_results={"trades": 100, "win_rate": 0.65},
            optimization_score=1.5
        )
        
        assert result.parameter_combination["rsi_period"] == 14
        assert result.performance_metrics["sharpe_ratio"] == 1.5
        assert result.optimization_score == 1.5

    def test_performance_analyzer_initialization(self):
        """Test performance analyzer initialization."""
        analyzer = PerformanceAnalyzer()
        
        assert len(analyzer.optimization_history) == 0
        assert len(analyzer.best_results) == 0

    def test_add_optimization_result(self):
        """Test adding optimization result."""
        analyzer = PerformanceAnalyzer()
        
        result = OptimizationResult(
            parameter_combination={"rsi_period": 14},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results={},
            optimization_score=1.5
        )
        
        analyzer.add_optimization_result(result)
        
        assert len(analyzer.optimization_history) == 1
        assert analyzer.best_results["best_score"] == 1.5

    def test_get_optimization_stats(self):
        """Test getting optimization statistics."""
        analyzer = PerformanceAnalyzer()
        
        # Add some results
        for i in range(5):
            result = OptimizationResult(
                parameter_combination={"param": i},
                performance_metrics={"sharpe_ratio": 1.0 + i * 0.1},
                backtest_results={},
                optimization_score=1.0 + i * 0.1
            )
            analyzer.add_optimization_result(result)
        
        stats = analyzer.get_optimization_stats()
        
        assert stats["total_optimizations"] == 5
        assert stats["best_score"] == 1.4
        assert stats["average_score"] > 1.0

    def test_find_optimal_parameters(self):
        """Test finding optimal parameters."""
        analyzer = PerformanceAnalyzer()
        
        # Add results with different metrics
        for i in range(3):
            result = OptimizationResult(
                parameter_combination={"rsi_period": 10 + i * 5},
                performance_metrics={"sharpe_ratio": 1.0 + i * 0.2},
                backtest_results={},
                optimization_score=1.0 + i * 0.2
            )
            analyzer.add_optimization_result(result)
        
        optimal_params = analyzer.find_optimal_parameters("sharpe_ratio", top_n=2)
        
        assert len(optimal_params) == 2
        assert optimal_params[0]["performance_metrics"]["sharpe_ratio"] > optimal_params[1]["performance_metrics"]["sharpe_ratio"]


if __name__ == "__main__":
    pytest.main([__file__]) 