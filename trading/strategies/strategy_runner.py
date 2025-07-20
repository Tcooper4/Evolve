"""
Strategy Runner

This module provides an async strategy runner for parallel execution of
multiple trading strategies with ensemble combination capabilities.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base_strategy import BaseStrategy
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)


class AsyncStrategyRunner:
    """
    Async strategy runner for parallel execution of multiple strategies.

    Features:
    - Parallel strategy execution using asyncio
    - Result gathering and ensemble combination
    - Error handling and timeout management
    - Performance monitoring and logging
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async strategy runner.

        Args:
            config: Configuration dictionary containing:
                - max_concurrent_strategies: Maximum concurrent strategies (default: 5)
                - strategy_timeout: Timeout per strategy in seconds (default: 30)
                - enable_ensemble: Enable ensemble combination (default: True)
                - ensemble_method: Ensemble method ('weighted', 'voting', 'average')
                - log_performance: Enable performance logging (default: True)
        """
        self.config = config or {}
        self.max_concurrent = self.config.get("max_concurrent_strategies", 5)
        self.strategy_timeout = self.config.get("strategy_timeout", 30)
        self.enable_ensemble = self.config.get("enable_ensemble", True)
        self.ensemble_method = self.config.get("ensemble_method", "weighted")
        self.log_performance = self.config.get("log_performance", True)

        # Strategy registry
        self.strategy_registry = StrategyRegistry()

        # Performance tracking
        self.execution_history = []
        self.strategy_performance = {}

        # Semaphore for limiting concurrent executions
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        logger.info(f"AsyncStrategyRunner initialized with max_concurrent={self.max_concurrent}")

    async def run_strategies_parallel(
        self,
        strategies: List[Union[str, BaseStrategy]],
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        ensemble_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple strategies in parallel and optionally combine results.

        Args:
            strategies: List of strategy names or strategy instances
            data: Market data for strategy execution
            parameters: Dictionary mapping strategy names to parameters
            ensemble_config: Configuration for ensemble combination

        Returns:
            Dictionary containing individual results and ensemble result
        """
        start_time = time.time()

        try:
            logger.info(f"Starting parallel execution of {len(strategies)} strategies")

            # Create async tasks for each strategy
            tasks = []
            for strategy in strategies:
                if isinstance(strategy, str):
                    # Strategy name - get from registry
                    strategy_name = strategy
                    strategy_instance = self.strategy_registry.get_strategy(strategy_name)
                    if not strategy_instance:
                        logger.warning(f"Strategy '{strategy_name}' not found in registry")
                        continue
                else:
                    # Strategy instance
                    strategy_instance = strategy
                    strategy_name = strategy_instance.__class__.__name__

                # Get parameters for this strategy
                strategy_params = parameters.get(strategy_name, {}) if parameters else {}

                # Create async task
                task = self._run_single_strategy_async(
                    strategy_instance, data, strategy_params, strategy_name
                )
                tasks.append(task)

            if not tasks:
                raise ValueError("No valid strategies to execute")

            # Execute all strategies in parallel
            logger.info(f"Executing {len(tasks)} strategies with asyncio.gather")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            strategy_results = {}
            successful_results = []
            failed_strategies = []

            for i, result in enumerate(results):
                strategy_name = strategies[i] if isinstance(strategies[i], str) else strategies[i].__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Strategy '{strategy_name}' failed: {result}")
                    failed_strategies.append(strategy_name)
                    strategy_results[strategy_name] = {
                        "success": False,
                        "error": str(result),
                        "signals": pd.DataFrame(),
                        "performance": {}
                    }
                else:
                    logger.info(f"Strategy '{strategy_name}' completed successfully")
                    successful_results.append(result)
                    strategy_results[strategy_name] = result

            # Create ensemble result if enabled
            ensemble_result = None
            if self.enable_ensemble and successful_results:
                ensemble_result = await self._create_ensemble_result(successful_results, ensemble_config)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log performance
            if self.log_performance:
                self._log_execution_performance(
                    execution_time, len(successful_results), len(strategies), failed_strategies
                )

            # Store execution history
            execution_record = {
                "timestamp": time.time(),
                "strategies": [s if isinstance(s, str) else s.__class__.__name__ for s in strategies],
                "execution_time": execution_time,
                "success_count": len(successful_results),
                "total_count": len(strategies),
                "failed_strategies": failed_strategies,
                "ensemble_enabled": self.enable_ensemble
            }
            self.execution_history.append(execution_record)

            return {
                "success": True,
                "execution_time": execution_time,
                "strategy_results": strategy_results,
                "ensemble_result": ensemble_result,
                "successful_count": len(successful_results),
                "failed_count": len(failed_strategies),
                "failed_strategies": failed_strategies
            }

        except Exception as e:
            logger.error(f"Parallel strategy execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "strategy_results": {},
                "ensemble_result": None
            }

    async def _run_single_strategy_async(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        strategy_name: str
    ) -> Dict[str, Any]:
        """Run a single strategy asynchronously with timeout and error handling."""
        async with self.semaphore:
            try:
                logger.debug(f"Starting strategy '{strategy_name}' with timeout {self.strategy_timeout}s")

                # Execute strategy with timeout
                result = await asyncio.wait_for(
                    self._execute_strategy_core(strategy, data),
                    timeout=self.strategy_timeout
                )

                # Add strategy metadata
                result["strategy_name"] = strategy_name
                result["parameters"] = parameters
                result["success"] = True

                # Update performance tracking
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        "execution_count": 0,
                        "success_count": 0,
                        "total_execution_time": 0.0,
                        "average_execution_time": 0.0
                    }

                perf = self.strategy_performance[strategy_name]
                perf["execution_count"] += 1
                perf["success_count"] += 1
                perf["total_execution_time"] += result.get("execution_time", 0.0)
                perf["average_execution_time"] = perf["total_execution_time"] / perf["success_count"]

                return result

            except asyncio.TimeoutError:
                logger.error(f"Strategy '{strategy_name}' timed out after {self.strategy_timeout}s")
                return {
                    "strategy_name": strategy_name,
                    "success": False,
                    "error": f"Strategy timed out after {self.strategy_timeout}s",
                    "signals": pd.DataFrame(),
                    "performance": {}
                }
            except Exception as e:
                logger.error(f"Strategy '{strategy_name}' failed: {e}")
                return {
                    "strategy_name": strategy_name,
                    "success": False,
                    "error": str(e),
                    "signals": pd.DataFrame(),
                    "performance": {}
                }

    async def _execute_strategy_core(
        self, strategy: BaseStrategy, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute the core strategy logic."""
        start_time = time.time()

        try:
            # Execute strategy
            signals = strategy.generate_signals(data)
            performance = self._calculate_performance_metrics(data, signals)

            execution_time = time.time() - start_time

            return {
                "signals": signals,
                "performance": performance,
                "execution_time": execution_time
            }

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise

    async def _create_ensemble_result(
        self,
        strategy_results: List[Dict[str, Any]],
        ensemble_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create ensemble result by combining multiple strategy results."""
        if not strategy_results:
            return None

        try:
            # Extract signals from all successful strategies
            all_signals = []
            weights = []

            for result in strategy_results:
                if result.get("success", True) and not result["signals"].empty:
                    all_signals.append(result["signals"])
                    # Use strategy confidence as weight, or default to 1.0
                    weight = result.get("performance", {}).get("confidence", 1.0)
                    weights.append(weight)

            if not all_signals:
                logger.warning("No valid signals for ensemble combination")
                return None

            # Combine signals based on ensemble method
            if self.ensemble_method == "weighted":
                combined_signals = self._combine_signals_weighted(all_signals, weights)
            elif self.ensemble_method == "voting":
                combined_signals = self._combine_signals_voting(all_signals)
            elif self.ensemble_method == "average":
                combined_signals = self._combine_signals_average(all_signals)
            else:
                logger.warning(f"Unknown ensemble method: {self.ensemble_method}")
                combined_signals = all_signals[0]  # Use first strategy as fallback

            # Calculate ensemble performance
            ensemble_performance = self._calculate_performance_metrics(
                strategy_results[0].get("data", pd.DataFrame()), combined_signals
            )

            return {
                "signals": combined_signals,
                "performance": ensemble_performance,
                "strategy_count": len(strategy_results),
                "ensemble_method": self.ensemble_method
            }

        except Exception as e:
            logger.error(f"Ensemble combination failed: {e}")
            return None

    def _combine_signals_weighted(self, signals_list: List[pd.DataFrame], weights: List[float]) -> pd.DataFrame:
        """Combine signals using weighted average."""
        if len(signals_list) == 1:
            return signals_list[0]

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Combine signals
        combined = signals_list[0].copy()
        for i, signals in enumerate(signals_list[1:], 1):
            combined = combined.add(signals * normalized_weights[i], fill_value=0)

        return combined

    def _combine_signals_voting(self, signals_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine signals using voting mechanism."""
        if len(signals_list) == 1:
            return signals_list[0]

        # Simple voting: sum all signals and use majority
        combined = signals_list[0].copy()
        for signals in signals_list[1:]:
            combined = combined.add(signals, fill_value=0)

        # Apply voting threshold
        threshold = len(signals_list) / 2
        combined = (combined >= threshold).astype(float)

        return combined

    def _combine_signals_average(self, signals_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine signals using simple average."""
        if len(signals_list) == 1:
            return signals_list[0]

        # Simple average
        combined = signals_list[0].copy()
        for signals in signals_list[1:]:
            combined = combined.add(signals, fill_value=0)

        combined = combined / len(signals_list)
        return combined

    def _calculate_performance_metrics(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics for strategy results."""
        try:
            if signals.empty or data.empty:
                return {"confidence": 0.0, "signal_count": 0}

            # Basic metrics
            signal_count = len(signals)
            buy_signals = len(signals[signals > 0])
            sell_signals = len(signals[signals < 0])

            # Calculate confidence based on signal strength
            signal_strength = signals.abs().mean().mean() if not signals.empty else 0.0
            confidence = min(signal_strength, 1.0)

            return {
                "confidence": confidence,
                "signal_count": signal_count,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "signal_strength": signal_strength
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"confidence": 0.0, "signal_count": 0}

    def _log_execution_performance(
        self,
        execution_time: float,
        success_count: int,
        total_count: int,
        failed_strategies: List[str]
    ) -> None:
        """Log execution performance metrics."""
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0

        logger.info(f"Strategy execution completed:")
        logger.info(f"  - Total time: {execution_time:.2f}s")
        logger.info(f"  - Success rate: {success_rate:.1f}% ({success_count}/{total_count})")
        logger.info(f"  - Failed strategies: {failed_strategies}")

        if failed_strategies:
            logger.warning(f"Failed strategies: {', '.join(failed_strategies)}")

    async def run_rsi_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run RSI strategy with given parameters."""
        strategy = self.strategy_registry.get_strategy("RSI_Strategy")
        if not strategy:
            raise ValueError("RSI_Strategy not found in registry")
        return await self._run_single_strategy_async(strategy, data, parameters, "RSI_Strategy")

    async def run_macd_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run MACD strategy with given parameters."""
        strategy = self.strategy_registry.get_strategy("MACD_Strategy")
        if not strategy:
            raise ValueError("MACD_Strategy not found in registry")
        return await self._run_single_strategy_async(strategy, data, parameters, "MACD_Strategy")

    async def run_bollinger_bands_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bollinger Bands strategy with given parameters."""
        strategy = self.strategy_registry.get_strategy("Bollinger_Bands")
        if not strategy:
            raise ValueError("Bollinger_Bands strategy not found in registry")
        return await self._run_single_strategy_async(strategy, data, parameters, "Bollinger_Bands")

    async def run_momentum_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run momentum strategy with given parameters."""
        strategy = self.strategy_registry.get_strategy("Momentum_Strategy")
        if not strategy:
            raise ValueError("Momentum_Strategy not found in registry")
        return await self._run_single_strategy_async(strategy, data, parameters, "Momentum_Strategy")

    async def run_mean_reversion_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run mean reversion strategy with given parameters."""
        strategy = self.strategy_registry.get_strategy("Mean_Reversion_Strategy")
        if not strategy:
            raise ValueError("Mean_Reversion_Strategy not found in registry")
        return await self._run_single_strategy_async(strategy, data, parameters, "Mean_Reversion_Strategy")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        return self.strategy_performance

    def clear_history(self) -> None:
        """Clear execution history and performance data."""
        self.execution_history.clear()
        self.strategy_performance.clear()
        logger.info("Execution history and performance data cleared")


async def run_rsi() -> Dict[str, Any]:
    """Example: Run RSI strategy."""
    runner = AsyncStrategyRunner()
    data = pd.DataFrame()  # Add your data here
    parameters = {"period": 14, "oversold": 30, "overbought": 70}
    return await runner.run_rsi_strategy(data, parameters)


async def run_macd() -> Dict[str, Any]:
    """Example: Run MACD strategy."""
    runner = AsyncStrategyRunner()
    data = pd.DataFrame()  # Add your data here
    parameters = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    return await runner.run_macd_strategy(data, parameters)


async def run_bollinger_bands() -> Dict[str, Any]:
    """Example: Run Bollinger Bands strategy."""
    runner = AsyncStrategyRunner()
    data = pd.DataFrame()  # Add your data here
    parameters = {"period": 20, "std_dev": 2}
    return await runner.run_bollinger_bands_strategy(data, parameters)


async def run_strategies_parallel_example(
    data: pd.DataFrame,
    strategy_names: List[str] = None,
    parameters: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Example: Run multiple strategies in parallel."""
    runner = AsyncStrategyRunner()
    strategy_names = strategy_names or ["RSI_Strategy", "MACD_Strategy", "Bollinger_Bands"]
    parameters = parameters or {}
    return await runner.run_strategies_parallel(strategy_names, data, parameters) 