"""
Optimizer Agent

This agent systematically optimizes strategy combinations, thresholds, and indicators
for different tickers and time periods. It uses modular components for parameter validation,
strategy optimization, backtesting integration, and performance analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.agents.base_agent_interface import AgentConfig, AgentResult, BaseAgent
from trading.memory.agent_memory import AgentMemory

# Modular imports
from .parameter_validator import ParameterValidator, OptimizationParameter
from .strategy_optimizer import (
    StrategyOptimizer, StrategyConfig, OptimizationType, OptimizationMetric,
    OptimizationRequest, OptimizationConfig
)
from .backtest_integration import BacktestIntegration
from .performance_analyzer import PerformanceAnalyzer, OptimizationResult


class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent for strategy and parameter optimization.
    
    This agent handles:
    - Strategy combination optimization
    - Parameter threshold optimization
    - Indicator parameter optimization
    - Hybrid optimization
    - Performance analysis and result management
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.parameter_validator = ParameterValidator()
        self.strategy_optimizer = StrategyOptimizer(config.get("optimizer_config", {}))
        self.backtest_integration = BacktestIntegration(config.get("backtest_config", {}))
        self.performance_analyzer = PerformanceAnalyzer(config.get("performance_config", {}))
        self.agent_memory = AgentMemory(config.get("memory_config", {}))
        
        # Optimization state
        self.current_optimization = None
        self.optimization_results = []
        
        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage for optimization data."""
        storage_dir = Path("data/optimization")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs_file = storage_dir / "optimization_configs.json"
        self.results_file = storage_dir / "optimization_results.json"
        
        # Load existing data
        self._load_optimization_configs()
        self._load_optimization_results()

    def _load_optimization_configs(self) -> None:
        """Load optimization configurations from storage."""
        try:
            if self.configs_file.exists():
                with open(self.configs_file, 'r') as f:
                    self.optimization_configs = json.load(f)
            else:
                self.optimization_configs = {}
        except Exception as e:
            self.logger.error(f"Failed to load optimization configs: {e}")
            self.optimization_configs = {}

    def _load_optimization_results(self) -> None:
        """Load optimization results from storage."""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    results_data = json.load(f)
                    self.optimization_results = [
                        OptimizationResult.from_dict(result) for result in results_data
                    ]
            else:
                self.optimization_results = []
        except Exception as e:
            self.logger.error(f"Failed to load optimization results: {e}")
            self.optimization_results = []

    def _save_optimization_configs(self) -> None:
        """Save optimization configurations to storage."""
        try:
            with open(self.configs_file, 'w') as f:
                json.dump(self.optimization_configs, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save optimization configs: {e}")

    def _save_optimization_results(self) -> None:
        """Save optimization results to storage."""
        try:
            results_data = [result.to_dict() for result in self.optimization_results]
            with open(self.results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    async def execute(self, **kwargs) -> AgentResult:
        """
        Main execution method for the optimizer agent.
        
        Args:
            **kwargs: Optimization parameters including:
                - optimization_type: Type of optimization to perform
                - target_metric: Metric to optimize for
                - symbols: List of symbols to optimize for
                - time_periods: List of time periods to test
                - strategy_configs: Strategy configurations
                - parameters_to_optimize: Parameters to optimize
        
        Returns:
            AgentResult with optimization status and results
        """
        try:
            # Extract parameters
            optimization_type = kwargs.get("optimization_type", "threshold_optimization")
            target_metric = kwargs.get("target_metric", "sharpe_ratio")
            symbols = kwargs.get("symbols", ["AAPL", "GOOGL", "MSFT"])
            time_periods = kwargs.get("time_periods", [{"start_date": "2023-01-01", "end_date": "2023-12-31"}])
            strategy_configs = kwargs.get("strategy_configs", [])
            parameters_to_optimize = kwargs.get("parameters_to_optimize", [])
            
            # Create optimization configuration
            config = self._create_optimization_config(
                optimization_type, target_metric, symbols, time_periods,
                strategy_configs, parameters_to_optimize
            )
            
            # Run optimization
            results = await self._run_optimization(config)
            
            # Analyze and store results
            self._process_optimization_results(results, config)
            
            # Update configurations if optimization was successful
            if results:
                await self._update_configurations(results, config)
            
            return AgentResult(
                success=True,
                message=f"Optimization completed with {len(results)} results",
                data={
                    "optimization_type": optimization_type,
                    "target_metric": target_metric,
                    "results_count": len(results),
                    "best_score": max([r.optimization_score for r in results]) if results else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return AgentResult(
                success=False,
                message=f"Optimization failed: {str(e)}",
                data={"error": str(e)}
            )

    def _create_optimization_config(
        self,
        optimization_type: str,
        target_metric: str,
        symbols: List[str],
        time_periods: List[Dict[str, Any]],
        strategy_configs: List[Dict[str, Any]],
        parameters_to_optimize: List[Dict[str, Any]]
    ) -> OptimizationConfig:
        """Create optimization configuration."""
        # Convert strategy configs
        strategy_configs_objects = [
            StrategyConfig.from_dict(config) for config in strategy_configs
        ]
        
        # Convert parameters
        parameters_objects = [
            OptimizationParameter.from_dict(param) for param in parameters_to_optimize
        ]
        
        return OptimizationConfig(
            optimization_type=OptimizationType(optimization_type),
            target_metric=OptimizationMetric(target_metric),
            symbols=symbols,
            time_periods=time_periods,
            strategy_configs=strategy_configs_objects,
            parameters_to_optimize=parameters_objects
        )

    async def _run_optimization(self, config: OptimizationConfig) -> List[OptimizationResult]:
        """Run optimization based on configuration."""
        try:
            if config.optimization_type == OptimizationType.STRATEGY_COMBINATION:
                results = await self.strategy_optimizer.optimize_strategy_combinations(config)
            elif config.optimization_type == OptimizationType.THRESHOLD_OPTIMIZATION:
                results = await self.strategy_optimizer.optimize_thresholds(config)
            elif config.optimization_type == OptimizationType.INDICATOR_OPTIMIZATION:
                results = await self.strategy_optimizer.optimize_indicators(config)
            elif config.optimization_type == OptimizationType.HYBRID_OPTIMIZATION:
                results = await self.strategy_optimizer.optimize_hybrid(config)
            else:
                raise ValueError(f"Unsupported optimization type: {config.optimization_type}")
            
            # Convert to OptimizationResult objects
            optimization_results = []
            for result in results:
                optimization_result = OptimizationResult(
                    parameter_combination=result["parameter_combination"],
                    performance_metrics=result["performance_metrics"],
                    backtest_results=result["backtest_results"],
                    optimization_score=result["optimization_score"],
                    timestamp=datetime.fromisoformat(result["timestamp"]) if isinstance(result["timestamp"], str) else result["timestamp"]
                )
                optimization_results.append(optimization_result)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization execution failed: {e}")
            return []

    def _process_optimization_results(
        self, results: List[OptimizationResult], config: OptimizationConfig
    ) -> None:
        """Process and store optimization results."""
        for result in results:
            # Add to performance analyzer
            self.performance_analyzer.add_optimization_result(result)
            
            # Add to local results
            self.optimization_results.append(result)
        
        # Save results
        self._save_optimization_results()
        
        # Log results
        self._log_optimization_results(results, config)

    def _log_optimization_results(
        self, results: List[OptimizationResult], config: OptimizationConfig
    ) -> None:
        """Log optimization results."""
        if not results:
            self.logger.info("No optimization results generated")
            return
        
        # Find best result
        best_result = max(results, key=lambda x: x.optimization_score)
        
        self.logger.info(f"Optimization completed:")
        self.logger.info(f"  Type: {config.optimization_type.value}")
        self.logger.info(f"  Target Metric: {config.target_metric.value}")
        self.logger.info(f"  Symbols: {config.symbols}")
        self.logger.info(f"  Results: {len(results)}")
        self.logger.info(f"  Best Score: {best_result.optimization_score:.4f}")
        self.logger.info(f"  Best Parameters: {best_result.parameter_combination}")

    async def _update_configurations(
        self, results: List[OptimizationResult], config: OptimizationConfig
    ) -> None:
        """Update configurations based on optimization results."""
        if not results:
            return
        
        # Find best result
        best_result = max(results, key=lambda x: x.optimization_score)
        
        # Update agent configurations
        await self._update_agent_configs(best_result, config)
        
        # Update strategy configurations
        await self._update_strategy_configs(best_result, config)

    async def _update_agent_configs(
        self, best_result: OptimizationResult, config: OptimizationConfig
    ) -> None:
        """Update agent configurations with best parameters."""
        try:
            # This would typically update agent configuration files
            # For now, just log the update
            self.logger.info(f"Updating agent configs with best parameters: {best_result.parameter_combination}")
            
            # Store in memory for future reference
            self.agent_memory.store(
                "optimization_results",
                "best_parameters",
                best_result.parameter_combination
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update agent configs: {e}")

    async def _update_strategy_configs(
        self, best_result: OptimizationResult, config: OptimizationConfig
    ) -> None:
        """Update strategy configurations with best parameters."""
        try:
            # Update strategy configurations
            for strategy_config in config.strategy_configs:
                strategy_name = strategy_config.strategy_name
                
                # Get parameters for this strategy
                strategy_params = {
                    k: v for k, v in best_result.parameter_combination.items()
                    if k.startswith(f"{strategy_name}_")
                }
                
                if strategy_params:
                    await self._save_strategy_config(strategy_name, strategy_params)
            
        except Exception as e:
            self.logger.error(f"Failed to update strategy configs: {e}")

    async def _save_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> None:
        """Save strategy configuration."""
        try:
            config_file = Path(f"config/strategies/{strategy_name}_config.json")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Saved strategy config for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy config for {strategy_name}: {e}")

    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.performance_analyzer.get_optimization_history(limit)

    def get_best_results(self) -> Dict[str, Any]:
        """Get best optimization results."""
        return self.performance_analyzer.get_best_results()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.performance_analyzer.get_optimization_stats()

    def find_optimal_parameters(self, metric: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Find optimal parameter combinations for a metric."""
        return self.performance_analyzer.find_optimal_parameters(
            OptimizationMetric(metric), top_n
        )

    def export_optimization_report(self, format: str = "json") -> str:
        """Export optimization report."""
        return self.performance_analyzer.export_optimization_report(format)


def create_optimizer_agent(config: Optional[Dict[str, Any]] = None) -> OptimizerAgent:
    """Factory function to create an optimizer agent."""
    if config is None:
        config = {}
    
    agent_config = AgentConfig(
        name="OptimizerAgent",
        agent_type="optimization",
        config=config
    )
    
    return OptimizerAgent(agent_config)
