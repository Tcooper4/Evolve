"""
Optimizer Agent

This agent systematically optimizes strategy combinations, thresholds, and indicators
for different tickers and time periods. It evaluates performance and updates
configurations based on top results.
"""

import asyncio
import logging
import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentStatus, AgentResult
from trading.memory.agent_memory import AgentMemory
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.rsi_strategy import RSIStrategy
from trading.backtesting.backtester import Backtester
from trading.evaluation.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate

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
class OptimizationParameter:
    """Optimization parameter configuration."""
    name: str
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Union[float, int]
    parameter_type: str  # 'float', 'int', 'categorical'
    categories: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step': self.step,
            'parameter_type': self.parameter_type,
            'categories': self.categories
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationParameter':
        """Create from dictionary."""
        return cls(**data)

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
            'strategy_name': self.strategy_name,
            'enabled': self.enabled,
            'weight': self.weight,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class OptimizationResult:
    """Optimization result data class."""
    parameter_combination: Dict[str, Any]
    performance_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    optimization_score: float
    timestamp: datetime = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create from dictionary."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
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
            'optimization_type': self.optimization_type.value,
            'target_metric': self.target_metric.value,
            'symbols': self.symbols,
            'time_periods': self.time_periods,
            'strategy_configs': [config.to_dict() for config in self.strategy_configs],
            'parameters_to_optimize': [param.to_dict() for param in self.parameters_to_optimize],
            'max_iterations': self.max_iterations,
            'parallel_workers': self.parallel_workers,
            'min_trades': self.min_trades,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        data['optimization_type'] = OptimizationType(data['optimization_type'])
        data['target_metric'] = OptimizationMetric(data['target_metric'])
        data['strategy_configs'] = [StrategyConfig.from_dict(config) for config in data['strategy_configs']]
        data['parameters_to_optimize'] = [OptimizationParameter.from_dict(param) for param in data['parameters_to_optimize']]
        return cls(**data)

class OptimizerAgent(BaseAgent):
    """Agent for optimizing strategy combinations, thresholds, and indicators."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the optimizer agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        # Initialize components
        self.memory = AgentMemory()
        self.portfolio_manager = PortfolioManager(initial_capital=100000.0)
        self.backtester = Backtester()
        
        # Optimization state
        self.optimization_results: List[OptimizationResult] = []
        self.best_results: Dict[str, OptimizationResult] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Strategy registry
        self.strategy_registry = {
            'bollinger': BollingerStrategy,
            'macd': MACDStrategy,
            'rsi': RSIStrategy
        }
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'average_improvement': 0.0
        }
        
        self.logger.info(f"OptimizerAgent initialized with {len(self.strategy_registry)} strategies")
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the optimization process.
        
        Args:
            **kwargs: May include:
                - optimization_config: OptimizationConfig object
                - symbols: List of symbols to optimize
                - time_periods: List of time periods
                - optimization_type: Type of optimization
                
        Returns:
            AgentResult: Result of the optimization
        """
        try:
            optimization_config = kwargs.get('optimization_config')
            if not optimization_config:
                return AgentResult(
                    success=False,
                    message="No optimization configuration provided",
                    error=ValueError("optimization_config is required")
                )
            
            # Start optimization
            self.logger.info(f"Starting {optimization_config.optimization_type.value} optimization")
            
            # Perform optimization based on type
            if optimization_config.optimization_type == OptimizationType.STRATEGY_COMBINATION:
                results = await self._optimize_strategy_combinations(optimization_config)
            elif optimization_config.optimization_type == OptimizationType.THRESHOLD_OPTIMIZATION:
                results = await self._optimize_thresholds(optimization_config)
            elif optimization_config.optimization_type == OptimizationType.INDICATOR_OPTIMIZATION:
                results = await self._optimize_indicators(optimization_config)
            elif optimization_config.optimization_type == OptimizationType.HYBRID_OPTIMIZATION:
                results = await self._optimize_hybrid(optimization_config)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_config.optimization_type}")
            
            # Update configurations based on results
            await self._update_configurations(results, optimization_config)
            
            # Generate summary
            summary = self._generate_optimization_summary(results, optimization_config)
            
            return AgentResult(
                success=True,
                message=f"Optimization completed with {len(results)} results",
                data={
                    'results': [r.to_dict() for r in results],
                    'summary': summary,
                    'best_results': {k: v.to_dict() for k, v in self.best_results.items()}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
            return AgentResult(
                success=False,
                message=f"Optimization error: {str(e)}",
                error=e
            )
    
    async def _optimize_strategy_combinations(self, config: OptimizationConfig) -> List[OptimizationResult]:
        """Optimize strategy combinations.
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of optimization results
        """
        results = []
        
        # Generate strategy combinations
        strategy_combinations = self._generate_strategy_combinations(config.strategy_configs)
        
        self.logger.info(f"Testing {len(strategy_combinations)} strategy combinations")
        
        # Test each combination
        for i, combination in enumerate(strategy_combinations):
            try:
                result = await self._test_strategy_combination(combination, config)
                if result:
                    results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(strategy_combinations)} combinations tested")
                    
            except Exception as e:
                self.logger.error(f"Error testing combination {i}: {e}")
                continue
        
        # Sort results by optimization score
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return results
    
    async def _optimize_thresholds(self, config: OptimizationConfig) -> List[OptimizationResult]:
        """Optimize strategy thresholds.
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of optimization results
        """
        results = []
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(config.parameters_to_optimize)
        
        self.logger.info(f"Testing {len(parameter_combinations)} threshold combinations")
        
        # Test each combination
        for i, combination in enumerate(parameter_combinations):
            try:
                result = await self._test_parameter_combination(combination, config)
                if result:
                    results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(parameter_combinations)} combinations tested")
                    
            except Exception as e:
                self.logger.error(f"Error testing combination {i}: {e}")
                continue
        
        # Sort results by optimization score
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return results
    
    async def _optimize_indicators(self, config: OptimizationConfig) -> List[OptimizationResult]:
        """Optimize indicator parameters.
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of optimization results
        """
        results = []
        
        # Generate indicator combinations
        indicator_combinations = self._generate_indicator_combinations(config.parameters_to_optimize)
        
        self.logger.info(f"Testing {len(indicator_combinations)} indicator combinations")
        
        # Test each combination
        for i, combination in enumerate(indicator_combinations):
            try:
                result = await self._test_indicator_combination(combination, config)
                if result:
                    results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(indicator_combinations)} combinations tested")
                    
            except Exception as e:
                self.logger.error(f"Error testing combination {i}: {e}")
                continue
        
        # Sort results by optimization score
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return results
    
    async def _optimize_hybrid(self, config: OptimizationConfig) -> List[OptimizationResult]:
        """Perform hybrid optimization (strategies + thresholds + indicators).
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of optimization results
        """
        results = []
        
        # Generate hybrid combinations
        hybrid_combinations = self._generate_hybrid_combinations(config)
        
        self.logger.info(f"Testing {len(hybrid_combinations)} hybrid combinations")
        
        # Test each combination
        for i, combination in enumerate(hybrid_combinations):
            try:
                result = await self._test_hybrid_combination(combination, config)
                if result:
                    results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(hybrid_combinations)} combinations tested")
                    
            except Exception as e:
                self.logger.error(f"Error testing combination {i}: {e}")
                continue
        
        # Sort results by optimization score
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return results
    
    def _generate_strategy_combinations(self, strategy_configs: List[StrategyConfig]) -> List[List[StrategyConfig]]:
        """Generate strategy combinations to test.
        
        Args:
            strategy_configs: List of strategy configurations
            
        Returns:
            List of strategy combinations
        """
        combinations = []
        
        # Generate all possible combinations of enabled strategies
        enabled_strategies = [config for config in strategy_configs if config.enabled]
        
        for r in range(1, len(enabled_strategies) + 1):
            for combo in itertools.combinations(enabled_strategies, r):
                combinations.append(list(combo))
        
        return combinations
    
    def _generate_parameter_combinations(self, parameters: List[OptimizationParameter]) -> List[Dict[str, Any]]:
        """Generate parameter combinations to test.
        
        Args:
            parameters: List of optimization parameters
            
        Returns:
            List of parameter combinations
        """
        combinations = []
        
        # Generate parameter values for each parameter
        param_values = {}
        for param in parameters:
            if param.parameter_type == 'categorical':
                param_values[param.name] = param.categories or []
            else:
                values = []
                current = param.min_value
                while current <= param.max_value:
                    values.append(current)
                    current += param.step
                param_values[param.name] = values
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        
        for combo in itertools.product(*param_value_lists):
            combination = dict(zip(param_names, combo))
            combinations.append(combination)
        
        return combinations
    
    def _generate_indicator_combinations(self, parameters: List[OptimizationParameter]) -> List[Dict[str, Any]]:
        """Generate indicator combinations to test.
        
        Args:
            parameters: List of optimization parameters
            
        Returns:
            List of indicator combinations
        """
        # Similar to parameter combinations but focused on indicators
        return self._generate_parameter_combinations(parameters)
    
    def _generate_hybrid_combinations(self, config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Generate hybrid combinations (strategies + parameters + indicators).
        
        Args:
            config: Optimization configuration
            
        Returns:
            List of hybrid combinations
        """
        combinations = []
        
        # Generate strategy combinations
        strategy_combinations = self._generate_strategy_combinations(config.strategy_configs)
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(config.parameters_to_optimize)
        
        # Combine strategies and parameters (limit combinations to prevent explosion)
        max_combinations = min(config.max_iterations, 1000)
        strategy_count = min(len(strategy_combinations), max_combinations // 2)
        parameter_count = min(len(parameter_combinations), max_combinations // strategy_count)
        
        for i, strategy_combo in enumerate(strategy_combinations[:strategy_count]):
            for j, param_combo in enumerate(parameter_combinations[:parameter_count]):
                combination = {
                    'strategies': strategy_combo,
                    'parameters': param_combo
                }
                combinations.append(combination)
                
                if len(combinations) >= max_combinations:
                    break
            if len(combinations) >= max_combinations:
                break
        
        return combinations
    
    async def _test_strategy_combination(self, combination: List[StrategyConfig], 
                                       config: OptimizationConfig) -> Optional[OptimizationResult]:
        """Test a strategy combination.
        
        Args:
            combination: Strategy combination to test
            config: Optimization configuration
            
        Returns:
            Optimization result or None if failed
        """
        try:
            # Create strategies with combination
            strategies = []
            for strategy_config in combination:
                strategy_class = self.strategy_registry.get(strategy_config.strategy_name)
                if strategy_class:
                    strategy = strategy_class(**strategy_config.parameters)
                    strategies.append(strategy)
            
            if not strategies:
                return None
            
            # Run backtest for each symbol and time period
            all_results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await self._run_backtest(
                        strategies, symbol, time_period, config
                    )
                    if result:
                        all_results.append(result)
            
            if not all_results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(all_results)
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return OptimizationResult(
                parameter_combination={'strategies': [s.to_dict() for s in combination]},
                performance_metrics=aggregated_metrics,
                backtest_results={'individual_results': all_results},
                optimization_score=optimization_score
            )
            
        except Exception as e:
            self.logger.error(f"Error testing strategy combination: {e}")
            return None
    
    async def _test_parameter_combination(self, combination: Dict[str, Any], 
                                        config: OptimizationConfig) -> Optional[OptimizationResult]:
        """Test a parameter combination.
        
        Args:
            combination: Parameter combination to test
            config: Optimization configuration
            
        Returns:
            Optimization result or None if failed
        """
        try:
            # Create strategies with parameter combination
            strategies = []
            for strategy_config in config.strategy_configs:
                if strategy_config.enabled:
                    # Merge default parameters with optimization parameters
                    parameters = strategy_config.parameters.copy()
                    parameters.update(combination)
                    
                    strategy_class = self.strategy_registry.get(strategy_config.strategy_name)
                    if strategy_class:
                        strategy = strategy_class(**parameters)
                        strategies.append(strategy)
            
            if not strategies:
                return None
            
            # Run backtest for each symbol and time period
            all_results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await self._run_backtest(
                        strategies, symbol, time_period, config
                    )
                    if result:
                        all_results.append(result)
            
            if not all_results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(all_results)
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return OptimizationResult(
                parameter_combination=combination,
                performance_metrics=aggregated_metrics,
                backtest_results={'individual_results': all_results},
                optimization_score=optimization_score
            )
            
        except Exception as e:
            self.logger.error(f"Error testing parameter combination: {e}")
            return None
    
    async def _test_indicator_combination(self, combination: Dict[str, Any], 
                                        config: OptimizationConfig) -> Optional[OptimizationResult]:
        """Test an indicator combination.
        
        Args:
            combination: Indicator combination to test
            config: Optimization configuration
            
        Returns:
            Optimization result or None if failed
        """
        # Similar to parameter combination but focused on indicators
        return await self._test_parameter_combination(combination, config)
    
    async def _test_hybrid_combination(self, combination: Dict[str, Any], 
                                     config: OptimizationConfig) -> Optional[OptimizationResult]:
        """Test a hybrid combination.
        
        Args:
            combination: Hybrid combination to test
            config: Optimization configuration
            
        Returns:
            Optimization result or None if failed
        """
        try:
            # Extract strategies and parameters
            strategy_configs = combination.get('strategies', [])
            parameters = combination.get('parameters', {})
            
            # Create strategies with both strategy and parameter combinations
            strategies = []
            for strategy_config in strategy_configs:
                # Merge parameters
                merged_params = strategy_config.parameters.copy()
                merged_params.update(parameters)
                
                strategy_class = self.strategy_registry.get(strategy_config.strategy_name)
                if strategy_class:
                    strategy = strategy_class(**merged_params)
                    strategies.append(strategy)
            
            if not strategies:
                return None
            
            # Run backtest for each symbol and time period
            all_results = []
            for symbol in config.symbols:
                for time_period in config.time_periods:
                    result = await self._run_backtest(
                        strategies, symbol, time_period, config
                    )
                    if result:
                        all_results.append(result)
            
            if not all_results:
                return None
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(all_results)
            optimization_score = self._calculate_optimization_score(
                aggregated_metrics, config.target_metric
            )
            
            return OptimizationResult(
                parameter_combination=combination,
                performance_metrics=aggregated_metrics,
                backtest_results={'individual_results': all_results},
                optimization_score=optimization_score
            )
            
        except Exception as e:
            self.logger.error(f"Error testing hybrid combination: {e}")
            return None
    
    async def _run_backtest(self, strategies: List, symbol: str, time_period: Dict[str, Any], 
                           config: OptimizationConfig) -> Optional[Dict[str, Any]]:
        """Run backtest for a specific configuration.
        
        Args:
            strategies: List of strategies to test
            symbol: Symbol to test
            time_period: Time period configuration
            config: Optimization configuration
            
        Returns:
            Backtest results or None if failed
        """
        try:
            # Run backtest
            backtest_result = await self.backtester.run_backtest(
                strategies=strategies,
                symbol=symbol,
                start_date=time_period['start_date'],
                end_date=time_period['end_date'],
                initial_capital=100000.0
            )
            
            if not backtest_result or backtest_result['total_trades'] < config.min_trades:
                return None
            
            # Calculate metrics
            metrics = {
                'total_return': backtest_result['total_return'],
                'sharpe_ratio': calculate_sharpe_ratio(backtest_result['returns']),
                'max_drawdown': calculate_max_drawdown(backtest_result['equity_curve']),
                'win_rate': calculate_win_rate(backtest_result['trades']),
                'profit_factor': backtest_result.get('profit_factor', 0.0),
                'total_trades': backtest_result['total_trades']
            }
            
            return {
                'symbol': symbol,
                'time_period': time_period,
                'metrics': metrics,
                'backtest_result': backtest_result
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest for {symbol}: {e}")
            return None
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple backtest results.
        
        Args:
            results: List of backtest results
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return {}
        
        # Extract metrics
        metrics_list = [result['metrics'] for result in results]
        
        # Aggregate using weighted average (by number of trades)
        total_trades = sum(metrics['total_trades'] for metrics in metrics_list)
        
        aggregated = {}
        for metric_name in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
            if metric_name in metrics_list[0]:
                weighted_sum = sum(
                    metrics[metric_name] * metrics['total_trades'] 
                    for metrics in metrics_list
                )
                aggregated[metric_name] = weighted_sum / total_trades if total_trades > 0 else 0.0
        
        return aggregated
    
    def _calculate_optimization_score(self, metrics: Dict[str, float], 
                                    target_metric: OptimizationMetric) -> float:
        """Calculate optimization score based on target metric.
        
        Args:
            metrics: Performance metrics
            target_metric: Target optimization metric
            
        Returns:
            Optimization score
        """
        if target_metric == OptimizationMetric.SHARPE_RATIO:
            return metrics.get('sharpe_ratio', 0.0)
        elif target_metric == OptimizationMetric.TOTAL_RETURN:
            return metrics.get('total_return', 0.0)
        elif target_metric == OptimizationMetric.MAX_DRAWDOWN:
            return -abs(metrics.get('max_drawdown', 0.0))  # Negative because lower is better
        elif target_metric == OptimizationMetric.WIN_RATE:
            return metrics.get('win_rate', 0.0)
        elif target_metric == OptimizationMetric.PROFIT_FACTOR:
            return metrics.get('profit_factor', 0.0)
        elif target_metric == OptimizationMetric.CALMAR_RATIO:
            total_return = metrics.get('total_return', 0.0)
            max_dd = abs(metrics.get('max_drawdown', 0.001))
            return total_return / max_dd if max_dd > 0 else 0.0
        elif target_metric == OptimizationMetric.COMPOSITE_SCORE:
            # Composite score combining multiple metrics
            sharpe = metrics.get('sharpe_ratio', 0.0)
            return_pct = metrics.get('total_return', 0.0) * 100
            win_rate = metrics.get('win_rate', 0.0)
            profit_factor = metrics.get('profit_factor', 0.0)
            
            # Normalize and weight
            score = (
                sharpe * 0.3 +
                return_pct * 0.3 +
                win_rate * 0.2 +
                min(profit_factor, 5.0) / 5.0 * 0.2
            )
            return score
        else:
            return 0.0
    
    async def _update_configurations(self, results: List[OptimizationResult], 
                                   config: OptimizationConfig) -> None:
        """Update configurations based on optimization results.
        
        Args:
            results: Optimization results
            config: Optimization configuration
        """
        if not results:
            return
        
        # Get best result
        best_result = results[0]
        
        # Update best results by symbol
        for symbol in config.symbols:
            symbol_key = f"{symbol}_{config.optimization_type.value}"
            self.best_results[symbol_key] = best_result
        
        # Update agent configurations
        await self._update_agent_configs(best_result, config)
        
        # Update strategy configurations
        await self._update_strategy_configs(best_result, config)
        
        # Log optimization results
        self._log_optimization_results(results, config)
    
    async def _update_agent_configs(self, best_result: OptimizationResult, 
                                  config: OptimizationConfig) -> None:
        """Update agent configurations based on best result.
        
        Args:
            best_result: Best optimization result
            config: Optimization configuration
        """
        try:
            # Update execution agent config if needed
            if 'execution_agent' in self.config.custom_config:
                execution_config = self.config.custom_config['execution_agent']
                
                # Update risk parameters if they were optimized
                if 'risk_per_trade' in best_result.parameter_combination:
                    execution_config['risk_per_trade'] = best_result.parameter_combination['risk_per_trade']
                
                if 'max_position_size' in best_result.parameter_combination:
                    execution_config['max_position_size'] = best_result.parameter_combination['max_position_size']
            
            # Update other agent configs as needed
            # This would depend on the specific agents in your system
            
        except Exception as e:
            self.logger.error(f"Error updating agent configs: {e}")
    
    async def _update_strategy_configs(self, best_result: OptimizationResult, 
                                     config: OptimizationConfig) -> None:
        """Update strategy configurations based on best result.
        
        Args:
            best_result: Best optimization result
            config: Optimization configuration
        """
        try:
            # Update strategy parameters
            if 'strategies' in best_result.parameter_combination:
                strategy_configs = best_result.parameter_combination['strategies']
                
                for strategy_config in strategy_configs:
                    strategy_name = strategy_config['strategy_name']
                    
                    # Update strategy configuration file or database
                    await self._save_strategy_config(strategy_name, strategy_config)
            
            # Update general parameters
            if 'parameters' in best_result.parameter_combination:
                parameters = best_result.parameter_combination['parameters']
                
                for param_name, param_value in parameters.items():
                    await self._save_parameter_config(param_name, param_value)
            
        except Exception as e:
            self.logger.error(f"Error updating strategy configs: {e}")
    
    async def _save_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> None:
        """Save strategy configuration.
        
        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration
        """
        try:
            # Save to configuration file
            config_path = f"config/strategies/{strategy_name}_config.json"
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Updated strategy config: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving strategy config: {e}")
    
    async def _save_parameter_config(self, param_name: str, param_value: Any) -> None:
        """Save parameter configuration.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
        """
        try:
            # Save to configuration file
            config_path = "config/optimization_params.json"
            
            # Load existing config
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
            except FileNotFoundError:
                existing_config = {}
            
            # Update parameter
            existing_config[param_name] = param_value
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)
            
            self.logger.info(f"Updated parameter config: {param_name} = {param_value}")
            
        except Exception as e:
            self.logger.error(f"Error saving parameter config: {e}")
    
    def _log_optimization_results(self, results: List[OptimizationResult], 
                                config: OptimizationConfig) -> None:
        """Log optimization results.
        
        Args:
            results: Optimization results
            config: Optimization configuration
        """
        try:
            # Add to optimization history
            history_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'optimization_type': config.optimization_type.value,
                'target_metric': config.target_metric.value,
                'symbols': config.symbols,
                'total_results': len(results),
                'best_score': results[0].optimization_score if results else 0.0,
                'best_parameters': results[0].parameter_combination if results else {}
            }
            
            self.optimization_history.append(history_entry)
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            # Log to memory
            self.memory.log_decision(
                agent_name=self.config.name,
                decision_type='optimization_completed',
                details=history_entry
            )
            
        except Exception as e:
            self.logger.error(f"Error logging optimization results: {e}")
    
    def _generate_optimization_summary(self, results: List[OptimizationResult], 
                                     config: OptimizationConfig) -> Dict[str, Any]:
        """Generate optimization summary.
        
        Args:
            results: Optimization results
            config: Optimization configuration
            
        Returns:
            Optimization summary
        """
        if not results:
            return {
                'total_results': 0,
                'best_score': 0.0,
                'average_score': 0.0,
                'improvement': 0.0
            }
        
        scores = [r.optimization_score for r in results]
        
        summary = {
            'total_results': len(results),
            'best_score': max(scores),
            'average_score': np.mean(scores),
            'std_score': np.std(scores),
            'improvement': self._calculate_improvement(results, config),
            'top_5_scores': sorted(scores, reverse=True)[:5],
            'optimization_type': config.optimization_type.value,
            'target_metric': config.target_metric.value,
            'symbols_tested': config.symbols,
            'best_parameters': results[0].parameter_combination
        }
        
        return summary
    
    def _calculate_improvement(self, results: List[OptimizationResult], 
                             config: OptimizationConfig) -> float:
        """Calculate improvement over baseline.
        
        Args:
            results: Optimization results
            config: Optimization configuration
            
        Returns:
            Improvement percentage
        """
        if not results:
            return 0.0
        
        best_score = results[0].optimization_score
        
        # Get baseline score from previous optimization or default
        baseline_score = self._get_baseline_score(config)
        
        if baseline_score == 0:
            return 0.0
        
        improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100
        return improvement
    
    def _get_baseline_score(self, config: OptimizationConfig) -> float:
        """Get baseline score for comparison.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Baseline score
        """
        # Look for previous optimization results
        for entry in reversed(self.optimization_history):
            if (entry['optimization_type'] == config.optimization_type.value and
                entry['target_metric'] == config.target_metric.value):
                return entry['best_score']
        
        # Return default baseline
        return 0.0
    
    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get optimization history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of optimization history entries
        """
        return self.optimization_history[-limit:]
    
    def get_best_results(self) -> Dict[str, OptimizationResult]:
        """Get best optimization results by symbol.
        
        Returns:
            Dictionary of best results by symbol
        """
        return self.best_results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Returns:
            Optimization statistics
        """
        return self.optimization_stats

def create_optimizer_agent(config: Optional[Dict[str, Any]] = None) -> OptimizerAgent:
    """Create an optimizer agent with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        OptimizerAgent instance
    """
    default_config = {
        'name': 'optimizer_agent',
        'agent_type': 'optimizer',
        'enabled': True,
        'custom_config': {
            'max_optimizations_per_day': 10,
            'parallel_workers': 4,
            'optimization_timeout': 3600,  # 1 hour
            'save_results': True
        }
    }
    
    if config:
        default_config.update(config)
    
    agent_config = AgentConfig(**default_config)
    return OptimizerAgent(agent_config)