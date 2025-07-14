"""
Strategy Optimizer Module

This module contains strategy optimization functionality for the optimizer agent.
Extracted from the original optimizer_agent.py for modularity.
"""

import itertools
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .parameter_validator import OptimizationParameter


class OptimizationType(Enum):
    """Optimization type enum."""

    STRATEGY_COMBINATION = "strategy_combination"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    INDICATOR_OPTIMIZATION = "indicator_optimization"
    HYBRID_OPTIMIZATION = "hybrid_optimization"


class OptimizationMetric(Enum):
    """Optimization metric enum."""

    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    COMPOSITE_SCORE = "composite_score"


@dataclass
class StrategyConfig:
    """Strategy configuration for optimization."""

    strategy_name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "enabled": self.enabled,
            "weight": self.weight,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationRequest:
    """Optimization request data class."""

    optimization_type: OptimizationType
    target_metric: OptimizationMetric
    symbols: List[str]
    time_periods: List[Dict[str, Any]]
    strategy_configs: List[StrategyConfig]
    parameters_to_optimize: List[OptimizationParameter]
    max_iterations: int = 1000
    parallel_workers: int = 4
    min_trades: int = 10
    confidence_threshold: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimization_type": self.optimization_type.value,
            "target_metric": self.target_metric.value,
            "symbols": self.symbols,
            "time_periods": self.time_periods,
            "strategy_configs": [config.to_dict() for config in self.strategy_configs],
            "parameters_to_optimize": [
                param.to_dict() for param in self.parameters_to_optimize
            ],
            "max_iterations": self.max_iterations,
            "parallel_workers": self.parallel_workers,
            "min_trades": self.min_trades,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationRequest":
        """Create from dictionary."""
        data["optimization_type"] = OptimizationType(data["optimization_type"])
        data["target_metric"] = OptimizationMetric(data["target_metric"])
        data["strategy_configs"] = [
            StrategyConfig.from_dict(config) for config in data["strategy_configs"]
        ]
        data["parameters_to_optimize"] = [
            OptimizationParameter.from_dict(param)
            for param in data["parameters_to_optimize"]
        ]
        return cls(**data)


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    optimization_type: OptimizationType
    target_metric: OptimizationMetric
    symbols: List[str]
    time_periods: List[Dict[str, Any]]  # List of time period configs
    strategy_configs: List[StrategyConfig]
    parameters_to_optimize: List[OptimizationParameter]
    max_iterations: int = 1000
    parallel_workers: int = 4
    min_trades: int = 10
    confidence_threshold: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimization_type": self.optimization_type.value,
            "target_metric": self.target_metric.value,
            "symbols": self.symbols,
            "time_periods": self.time_periods,
            "strategy_configs": [config.to_dict() for config in self.strategy_configs],
            "parameters_to_optimize": [
                param.to_dict() for param in self.parameters_to_optimize
            ],
            "max_iterations": self.max_iterations,
            "parallel_workers": self.parallel_workers,
            "min_trades": self.min_trades,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary."""
        data["optimization_type"] = OptimizationType(data["optimization_type"])
        data["target_metric"] = OptimizationMetric(data["target_metric"])
        data["strategy_configs"] = [
            StrategyConfig.from_dict(config) for config in data["strategy_configs"]
        ]
        data["parameters_to_optimize"] = [
            OptimizationParameter.from_dict(param)
            for param in data["parameters_to_optimize"]
        ]
        return cls(**data)


class StrategyOptimizer:
    """Handles strategy optimization operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_strategies = {
            "bollinger": "BollingerStrategy",
            "macd": "MACDStrategy", 
            "rsi": "RSIStrategy"
        }

    async def optimize_strategy_combinations(
        self, config: OptimizationConfig
    ) -> List[Dict[str, Any]]:
        """Optimize strategy combinations."""
        results = []
        
        # Generate strategy combinations
        combinations = self._generate_strategy_combinations(config.strategy_configs)
        
        # Test each combination
        for combination in combinations:
            result = await self._test_strategy_combination(combination, config)
            if result:
                results.append(result)
        
        return results

    async def optimize_thresholds(
        self, config: OptimizationConfig
    ) -> List[Dict[str, Any]]:
        """Optimize strategy thresholds."""
        results = []
        
        # Generate parameter combinations
        combinations = self._generate_parameter_combinations(config.parameters_to_optimize)
        
        # Test each combination
        for combination in combinations:
            result = await self._test_parameter_combination(combination, config)
            if result:
                results.append(result)
        
        return results

    async def optimize_indicators(
        self, config: OptimizationConfig
    ) -> List[Dict[str, Any]]:
        """Optimize indicator parameters."""
        results = []
        
        # Generate indicator combinations
        combinations = self._generate_parameter_combinations(config.parameters_to_optimize)
        
        # Test each combination
        for combination in combinations:
            result = await self._test_indicator_combination(combination, config)
            if result:
                results.append(result)
        
        return results

    async def optimize_hybrid(
        self, config: OptimizationConfig
    ) -> List[Dict[str, Any]]:
        """Optimize hybrid strategy combinations."""
        results = []
        
        # Generate hybrid combinations
        strategy_combinations = self._generate_strategy_combinations(config.strategy_configs)
        parameter_combinations = self._generate_parameter_combinations(config.parameters_to_optimize)
        
        # Combine strategies and parameters
        for strategy_combo in strategy_combinations:
            for param_combo in parameter_combinations:
                hybrid_combo = {**strategy_combo, **param_combo}
                result = await self._test_hybrid_combination(hybrid_combo, config)
                if result:
                    results.append(result)
        
        return results

    def _generate_strategy_combinations(
        self, strategy_configs: List[StrategyConfig]
    ) -> List[List[StrategyConfig]]:
        """Generate strategy combinations."""
        enabled_strategies = [config for config in strategy_configs if config.enabled]
        
        combinations = []
        for r in range(1, len(enabled_strategies) + 1):
            combinations.extend(list(itertools.combinations(enabled_strategies, r)))
        
        return [list(combo) for combo in combinations]

    def _generate_parameter_combinations(
        self, parameters: List[OptimizationParameter]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations."""
        from .parameter_validator import ParameterValidator
        
        validator = ParameterValidator()
        validated_params = validator.validate_optimization_parameters(parameters)
        
        # Generate parameter ranges
        param_ranges = {}
        for param in validated_params:
            param_ranges[param.name] = validator.generate_parameter_range(param)
        
        # Generate combinations
        combinations = []
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combo in itertools.product(*param_values):
            combination = dict(zip(param_names, combo))
            
            # Validate combination
            if validator.validate_parameter_combination(combination, validated_params):
                combinations.append(combination)
        
        return combinations

    async def _test_strategy_combination(
        self, combination: List[StrategyConfig], config: OptimizationConfig
    ) -> Optional[Dict[str, Any]]:
        """Test a strategy combination."""
        try:
            # Create backtest integration
            from .backtest_integration import BacktestIntegration
            backtest = BacktestIntegration()
            
            # Run backtest for each symbol and time period
            results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await backtest.run_backtest(
                        strategies=combination,
                        symbol=symbol,
                        time_period=time_period,
                        config=config
                    )
                    if result:
                        results.append(result)
            
            if not results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(results)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return {
                "parameter_combination": {config.strategy_name: config.parameters for config in combination},
                "performance_metrics": aggregated_metrics,
                "backtest_results": results,
                "optimization_score": optimization_score,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error testing strategy combination: {e}")
            return None

    async def _test_parameter_combination(
        self, combination: Dict[str, Any], config: OptimizationConfig
    ) -> Optional[Dict[str, Any]]:
        """Test a parameter combination."""
        try:
            # Create backtest integration
            from .backtest_integration import BacktestIntegration
            backtest = BacktestIntegration()
            
            # Apply parameters to strategies
            strategies_with_params = []
            for strategy_config in config.strategy_configs:
                strategy_with_params = StrategyConfig(
                    strategy_name=strategy_config.strategy_name,
                    enabled=strategy_config.enabled,
                    weight=strategy_config.weight,
                    parameters={**strategy_config.parameters, **combination}
                )
                strategies_with_params.append(strategy_with_params)
            
            # Run backtest
            results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await backtest.run_backtest(
                        strategies=strategies_with_params,
                        symbol=symbol,
                        time_period=time_period,
                        config=config
                    )
                    if result:
                        results.append(result)
            
            if not results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(results)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return {
                "parameter_combination": combination,
                "performance_metrics": aggregated_metrics,
                "backtest_results": results,
                "optimization_score": optimization_score,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error testing parameter combination: {e}")
            return None

    async def _test_indicator_combination(
        self, combination: Dict[str, Any], config: OptimizationConfig
    ) -> Optional[Dict[str, Any]]:
        """Test an indicator combination."""
        # Similar to parameter combination testing
        return await self._test_parameter_combination(combination, config)

    async def _test_hybrid_combination(
        self, combination: Dict[str, Any], config: OptimizationConfig
    ) -> Optional[Dict[str, Any]]:
        """Test a hybrid combination."""
        try:
            # Create backtest integration
            from .backtest_integration import BacktestIntegration
            backtest = BacktestIntegration()
            
            # Extract strategy and parameter parts
            strategy_params = {k: v for k, v in combination.items() if k.startswith("strategy_")}
            indicator_params = {k: v for k, v in combination.items() if not k.startswith("strategy_")}
            
            # Apply parameters to strategies
            strategies_with_params = []
            for strategy_config in config.strategy_configs:
                strategy_with_params = StrategyConfig(
                    strategy_name=strategy_config.strategy_name,
                    enabled=strategy_config.enabled,
                    weight=strategy_config.weight,
                    parameters={**strategy_config.parameters, **indicator_params}
                )
                strategies_with_params.append(strategy_with_params)
            
            # Run backtest
            results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await backtest.run_backtest(
                        strategies=strategies_with_params,
                        symbol=symbol,
                        time_period=time_period,
                        config=config
                    )
                    if result:
                        results.append(result)
            
            if not results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(results)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return {
                "parameter_combination": combination,
                "performance_metrics": aggregated_metrics,
                "backtest_results": results,
                "optimization_score": optimization_score,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error testing hybrid combination: {e}")
            return None

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from multiple backtest results."""
        if not results:
            return {}
        
        # Calculate averages
        metrics = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                values = [result[key] for result in results if key in result]
                if values:
                    metrics[key] = sum(values) / len(values)
        
        return metrics

    def _calculate_optimization_score(
        self, metrics: Dict[str, float], target_metric: OptimizationMetric
    ) -> float:
        """Calculate optimization score based on target metric."""
        if target_metric == OptimizationMetric.SHARPE_RATIO:
            return metrics.get("sharpe_ratio", 0.0)
        elif target_metric == OptimizationMetric.TOTAL_RETURN:
            return metrics.get("total_return", 0.0)
        elif target_metric == OptimizationMetric.MAX_DRAWDOWN:
            return -metrics.get("max_drawdown", 0.0)  # Negative because lower is better
        elif target_metric == OptimizationMetric.WIN_RATE:
            return metrics.get("win_rate", 0.0)
        elif target_metric == OptimizationMetric.PROFIT_FACTOR:
            return metrics.get("profit_factor", 0.0)
        elif target_metric == OptimizationMetric.CALMAR_RATIO:
            return metrics.get("calmar_ratio", 0.0)
        elif target_metric == OptimizationMetric.COMPOSITE_SCORE:
            # Calculate composite score
            sharpe = metrics.get("sharpe_ratio", 0.0)
            return_metric = metrics.get("total_return", 0.0)
            drawdown = metrics.get("max_drawdown", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            
            # Normalize and weight
            composite = (
                sharpe * 0.3 +
                return_metric * 0.3 +
                (1 - drawdown) * 0.2 +
                win_rate * 0.2
            )
            return composite
        else:
            return 0.0 