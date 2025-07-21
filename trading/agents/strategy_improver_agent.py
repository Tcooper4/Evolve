"""
Strategy Improver Agent for dynamic strategy parameter optimization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from trading.memory.agent_memory import AgentMemory
from trading.memory.strategy_logger import StrategyLogger
from trading.optimization.core_optimizer import BayesianOptimizer, GeneticOptimizer
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.rsi_strategy import RSIStrategy
from trading.utils.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

from .base_agent_interface import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class StrategyImprovementRequest:
    """Request for strategy improvement."""

    strategy_name: str
    improvement_type: (
        str  # 'parameter_optimization', 'logic_update', 'threshold_adjustment'
    )
    performance_thresholds: Optional[Dict[str, float]] = None
    optimization_method: str = "bayesian"
    max_iterations: int = 30
    timeout: int = 1800
    priority: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    market_conditions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyImprovementResult:
    """Result of strategy improvement process."""

    success: bool
    strategy_name: str
    improvement_type: str
    old_performance: Optional[Dict[str, float]] = None
    new_performance: Optional[Dict[str, float]] = None
    improvement_metrics: Optional[Dict[str, float]] = None
    changes_made: Optional[Dict[str, Any]] = None
    optimization_history: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StrategyImproverAgent(BaseAgent):
    """
    Agent responsible for monitoring strategy performance and adjusting logic.

    This agent performs dynamic parameter tuning for strategies like RSI thresholds,
    Bollinger band widths, and MACD parameters based on recent market conditions
    and performance metrics.
    """

    def __init__(
        self, name: str = "strategy_improver", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Strategy Improver Agent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)

        # Initialize components
        self.memory = AgentMemory()
        self.strategy_logger = StrategyLogger()

        # Strategy registry
        self.strategies = {
            "bollinger": BollingerStrategy,
            "rsi": RSIStrategy,
            "macd": MACDStrategy,
        }

        # Strategy performance tracking for auto-pruning
        self.strategy_metrics = {}

        # Performance tracking
        self.improvement_history: List[Dict[str, Any]] = []
        self.last_improvement: Dict[str, datetime] = {}

        # Configuration
        self.improvement_interval = config.get(
            "improvement_interval", 86400
        )  # 24 hours
        self.performance_thresholds = config.get(
            "performance_thresholds",
            {
                "min_sharpe": 0.8,
                "max_drawdown": 0.25,
                "min_win_rate": 0.45,
                "min_profit_factor": 1.1,
            },
        )

        # Optimization settings
        self.optimization_method = config.get("optimization_method", "bayesian")
        self.max_optimization_iterations = config.get("max_optimization_iterations", 30)
        self.optimization_timeout = config.get(
            "optimization_timeout", 1800
        )  # 30 minutes

        # Initialize optimizers
        self.bayesian_optimizer = BayesianOptimizer()
        self.genetic_optimizer = GeneticOptimizer()

        logger.info(
            f"Initialized StrategyImproverAgent with {self.optimization_method} optimization"
        )

    async def execute(self, **kwargs) -> AgentResult:
        """Main entry point for the strategy improver agent."""
        try:
            action = kwargs.get("action", "improve_all")
            if action == "improve_all":
                return await self._improve_all_strategies()
            elif action == "improve_specific":
                strategy_name = kwargs.get("strategy_name")
                if not strategy_name:
                    return AgentResult(
                        success=False, error_message="Missing strategy_name"
                    )
                performance = self._get_strategy_performance(strategy_name)
                return AgentResult(success=True, data={"performance": performance})
            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

            # Auto-prune underperforming strategies
            self._prune_underperforming_strategies()
        except Exception as e:
            return self.handle_error(e)

    async def _improve_all_strategies(self) -> AgentResult:
        """Improve all registered strategies based on performance."""
        try:
            improvements = []

            for strategy_name in self.strategies.keys():
                try:
                    # Check if improvement is needed
                    if self._should_improve_strategy(strategy_name):
                        improvement = await self._improve_specific_strategy(
                            strategy_name
                        )
                        if improvement.success:
                            improvements.append(improvement.data)
                        else:
                            logger.warning(
                                f"Failed to improve strategy {strategy_name}: {improvement.error_message}"
                            )
                except Exception as e:
                    logger.error(f"Error improving strategy {strategy_name}: {str(e)}")

            return AgentResult(
                success=True,
                data={
                    "improvements_made": len(improvements),
                    "improvements": improvements,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error in strategy improvement cycle: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    async def _improve_specific_strategy(self, strategy_name: str) -> AgentResult:
        """Improve a specific strategy through parameter optimization."""
        try:
            # Get strategy class
            strategy_class = self.strategies.get(strategy_name)
            if not strategy_class:
                return AgentResult(
                    success=False, error_message=f"Strategy {strategy_name} not found"
                )

            # Get recent performance
            performance = self._get_strategy_performance(strategy_name)
            if not performance:
                return AgentResult(
                    success=False,
                    error_message=f"No performance data for {strategy_name}",
                )

            # Check if improvement is needed
            if not self._needs_improvement(performance):
                return AgentResult(
                    success=True,
                    data={
                        "message": f"Strategy {strategy_name} performing well, no improvement needed"
                    },
                )

            # Get current parameters
            current_params = self._get_current_strategy_params(strategy_name)

            # Define optimization objective
            def objective(params: Dict[str, Any]) -> float:
                """Optimization objective function."""
                try:
                    # Estimate performance with new parameters
                    estimated_improvement = self._estimate_strategy_improvement(
                        strategy_name, params, performance
                    )

                    # Return negative score (minimize)
                    return -estimated_improvement

                except Exception as e:
                    logger.error(f"Error in optimization objective: {str(e)}")
                    return 0.0

            # Define parameter space
            param_space = self._get_strategy_parameter_space(strategy_name)

            # Run optimization
            if self.optimization_method == "bayesian":
                best_params = await self._run_bayesian_optimization(
                    objective, param_space, current_params
                )
            else:
                best_params = await self._run_genetic_optimization(
                    objective, param_space, current_params
                )

            # Apply improvements
            if best_params:
                # Update strategy parameters
                self._update_strategy_params(strategy_name, best_params)

                # Log improvement
                improvement_record = {
                    "strategy_name": strategy_name,
                    "timestamp": datetime.now().isoformat(),
                    "old_params": current_params,
                    "new_params": best_params,
                    "performance_before": performance,
                    "estimated_improvement": self._estimate_strategy_improvement(
                        strategy_name, best_params, performance
                    ),
                }

                self.improvement_history.append(improvement_record)
                self.last_improvement[strategy_name] = datetime.now()

                # Store in memory
                self.memory.log_outcome(
                    agent=self.name,
                    run_type="strategy_improvement",
                    outcome=improvement_record,
                )

                logger.info(f"Improved strategy {strategy_name} with new parameters")

                return AgentResult(
                    success=True,
                    data={
                        "strategy_name": strategy_name,
                        "improvement": improvement_record,
                        "message": f"Successfully improved {strategy_name}",
                    },
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=f"Failed to find better parameters for {strategy_name}",
                )

        except Exception as e:
            logger.error(f"Error improving strategy {strategy_name}: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    def _should_improve_strategy(self, strategy_name: str) -> bool:
        """Check if a strategy should be improved."""
        try:
            # Check if enough time has passed since last improvement
            if strategy_name in self.last_improvement:
                time_since_improvement = (
                    datetime.now() - self.last_improvement[strategy_name]
                )
                if time_since_improvement.total_seconds() < self.improvement_interval:
                    return False

            # Check performance
            performance = self._get_strategy_performance(strategy_name)
            if not performance:
                return True  # No performance data, needs improvement

            return self._needs_improvement(performance)

        except Exception as e:
            logger.error(f"Error checking if strategy should be improved: {str(e)}")
            return False

    def _get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get recent performance metrics for a strategy."""
        try:
            # Get recent strategy decisions and outcomes
            recent_data = self.memory.get_recent_outcomes(
                agent=strategy_name, run_type="strategy_decision", limit=50
            )

            if not recent_data:
                return None

            # Calculate performance metrics
            [entry.get("decision", "hold") for entry in recent_data]
            returns = [entry.get("return", 0.0) for entry in recent_data]
            [entry.get("timestamp", "") for entry in recent_data]

            if len(returns) < 10:
                return None

            # Calculate metrics
            total_return = np.sum(returns)
            win_rate = np.mean([1 if r > 0 else 0 for r in returns])

            # Calculate Sharpe ratio and drawdown
            if len(returns) > 1:
                sharpe = calculate_sharpe_ratio(returns)
                max_dd = calculate_max_drawdown(returns)
            else:
                sharpe = 0.0
                max_dd = 0.0

            # Calculate profit factor
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]

            if losing_trades:
                profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades))
            else:
                profit_factor = float("inf") if winning_trades else 1.0

            return {
                "total_return": total_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "profit_factor": profit_factor,
                "num_trades": len(returns),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting strategy performance: {str(e)}")
            return None

    def _needs_improvement(self, performance: Dict[str, Any]) -> bool:
        """Check if performance indicates need for improvement."""
        try:
            if not performance:
                return True

            # Check against thresholds
            if (
                performance.get("sharpe_ratio", 0)
                < self.performance_thresholds["min_sharpe"]
            ):
                return True

            if (
                performance.get("max_drawdown", 1)
                > self.performance_thresholds["max_drawdown"]
            ):
                return True

            if (
                performance.get("win_rate", 0)
                < self.performance_thresholds["min_win_rate"]
            ):
                return True

            if (
                performance.get("profit_factor", 0)
                < self.performance_thresholds["min_profit_factor"]
            ):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking improvement need: {str(e)}")
            return True

    def _get_current_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get current parameters for a strategy."""
        try:
            # Default parameters for each strategy type
            if strategy_name == "bollinger":
                return {"window": 20, "num_std": 2.0, "min_trades": 5}
            elif strategy_name == "rsi":
                return {"period": 14, "oversold": 30, "overbought": 70, "min_trades": 5}
            elif strategy_name == "macd":
                return {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "min_trades": 5,
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting current strategy params: {str(e)}")
            return {}

    def _get_strategy_parameter_space(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameter search space for a strategy."""
        try:
            if strategy_name == "bollinger":
                return {
                    "window": (10, 50),
                    "num_std": (1.5, 3.0),
                    "min_trades": (3, 10),
                }
            elif strategy_name == "rsi":
                return {
                    "period": (7, 21),
                    "oversold": (20, 40),
                    "overbought": (60, 80),
                    "min_trades": (3, 10),
                }
            elif strategy_name == "macd":
                return {
                    "fast_period": (8, 16),
                    "slow_period": (20, 32),
                    "signal_period": (7, 12),
                    "min_trades": (3, 10),
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting strategy parameter space: {str(e)}")
            return {}

    def _estimate_strategy_improvement(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        current_performance: Dict[str, Any],
    ) -> float:
        """Estimate performance improvement from parameter changes."""
        try:
            # This is a simplified estimation - in practice, you'd use more sophisticated methods

            improvement_score = 0.0

            # Strategy-specific parameter validation
            if strategy_name == "rsi":
                oversold = params.get("oversold", 30)
                overbought = params.get("overbought", 70)

                # Check for reasonable RSI thresholds
                if 20 <= oversold <= 40 and 60 <= overbought <= 80:
                    improvement_score += 0.3
                if oversold < overbought:
                    improvement_score += 0.2

            elif strategy_name == "bollinger":
                window = params.get("window", 20)
                num_std = params.get("num_std", 2.0)

                # Check for reasonable Bollinger parameters
                if 10 <= window <= 50:
                    improvement_score += 0.2
                if 1.5 <= num_std <= 3.0:
                    improvement_score += 0.2

            elif strategy_name == "macd":
                fast = params.get("fast_period", 12)
                slow = params.get("slow_period", 26)

                # Check for reasonable MACD parameters
                if fast < slow:
                    improvement_score += 0.3
                if 8 <= fast <= 16 and 20 <= slow <= 32:
                    improvement_score += 0.2

            # Consider current performance
            if current_performance.get("sharpe_ratio", 0) < 0.5:
                improvement_score += 0.3  # High potential for improvement

            if current_performance.get("win_rate", 0) < 0.4:
                improvement_score += 0.2

            return min(1.0, improvement_score)

        except Exception as e:
            logger.error(f"Error estimating strategy improvement: {str(e)}")
            return 0.0

    def _update_strategy_params(
        self, strategy_name: str, new_params: Dict[str, Any]
    ) -> None:
        """Update strategy parameters."""
        try:
            # In a real implementation, you would update the strategy configuration
            # For now, we'll log the update
            logger.info(f"Updated {strategy_name} parameters: {new_params}")

            # Store the update in memory
            self.memory.log_outcome(
                agent=self.name,
                run_type="parameter_update",
                outcome={
                    "strategy_name": strategy_name,
                    "new_params": new_params,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error updating strategy params: {str(e)}")

    async def _run_bayesian_optimization(
        self,
        objective: callable,
        param_space: Dict[str, Any],
        current_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run Bayesian optimization for parameter tuning."""
        try:
            # Initialize with current parameters
            initial_params = [current_params]

            # Run optimization
            best_params = self.bayesian_optimizer.optimize(
                objective=objective,
                param_space=param_space,
                initial_points=initial_params,
                n_iterations=self.max_optimization_iterations,
                timeout=self.optimization_timeout,
            )

            return best_params

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return None

    async def _run_genetic_optimization(
        self,
        objective: callable,
        param_space: Dict[str, Any],
        current_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run genetic optimization for parameter tuning."""
        try:
            # Initialize with current parameters
            initial_population = [current_params]

            # Run optimization
            best_params = self.genetic_optimizer.optimize(
                objective=objective,
                param_space=param_space,
                initial_population=initial_population,
                generations=8,
                population_size=15,
            )

            return best_params

        except Exception as e:
            logger.error(f"Error in genetic optimization: {str(e)}")
            return None

    async def _force_improvement(self) -> AgentResult:
        """Force improvement cycle for all strategies."""
        try:
            # Reset last improvement times
            self.last_improvement = {}

            # Run improvement
            return await self._improve_all_strategies()

        except Exception as e:
            logger.error(f"Error in forced improvement: {str(e)}")
            return AgentResult(success=False, error_message=str(e))

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of recent improvements."""
        try:
            recent_improvements = self.improvement_history[-10:]

            return {
                "total_improvements": len(self.improvement_history),
                "recent_improvements": len(recent_improvements),
                "strategies_improved": list(
                    set(imp["strategy_name"] for imp in recent_improvements)
                ),
                "last_improvement": (
                    recent_improvements[-1]["timestamp"]
                    if recent_improvements
                    else None
                ),
                "average_improvement_score": (
                    np.mean(
                        [
                            imp.get("estimated_improvement", 0)
                            for imp in recent_improvements
                        ]
                    )
                    if recent_improvements
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting improvement summary: {str(e)}")
            return {}

    def update_performance_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update performance thresholds."""
        self.performance_thresholds.update(new_thresholds)
        logger.info(f"Updated performance thresholds: {new_thresholds}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        base_status = super().get_status()
        base_status.update(
            {
                "improvement_interval": self.improvement_interval,
                "optimization_method": self.optimization_method,
                "performance_thresholds": self.performance_thresholds,
                "improvement_summary": self.get_improvement_summary(),
                "strategies_tracked": len(self.strategies),
            }
        )
        return base_status

    def _prune_underperforming_strategies(self):
        """Auto-prune underperforming strategies."""
        try:
            # Update strategy metrics
            for strategy_name in list(self.strategies.keys()):
                performance = self._get_strategy_performance(strategy_name)
                if performance and "sharpe" in performance:
                    self.strategy_metrics[strategy_name] = performance["sharpe"]

            # Prune strategies with Sharpe ratio below threshold
            self.strategies = {
                name: strategy
                for name, strategy in self.strategies.items()
                if self.strategy_metrics.get(name, 0) > 0.5
            }

            logger.info(
                f"Auto-pruned underperforming strategies. Remaining: {len(self.strategies)}"
            )

        except Exception as e:
            logger.error(f"Error pruning strategies: {e}")
