"""
Async Strategy Runner

This module provides async execution of multiple trading strategies in parallel,
with result gathering and ensemble combination capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .registry import StrategyRegistry
from .base_strategy import BaseStrategy
from .ensemble import WeightedEnsembleStrategy

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
                        "execution_time": 0
                    }
                else:
                    strategy_results[strategy_name] = result
                    if result.get("success", False):
                        successful_results.append(result)
            
            # Create ensemble result if enabled and we have successful results
            ensemble_result = None
            if self.enable_ensemble and successful_results:
                ensemble_result = await self._create_ensemble_result(
                    successful_results, ensemble_config
                )
            
            # Calculate execution statistics
            execution_time = time.time() - start_time
            success_count = len(successful_results)
            total_count = len(strategies)
            
            # Log performance
            if self.log_performance:
                self._log_execution_performance(
                    execution_time, success_count, total_count, failed_strategies
                )
            
            return {
                "success": True,
                "individual_results": strategy_results,
                "ensemble_result": ensemble_result,
                "execution_stats": {
                    "total_strategies": total_count,
                    "successful_strategies": success_count,
                    "failed_strategies": len(failed_strategies),
                    "execution_time": execution_time,
                    "parallel_efficiency": execution_time / (total_count * 0.1) if total_count > 0 else 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in parallel strategy execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

    async def _run_single_strategy_async(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Run a single strategy asynchronously with timeout and error handling.
        
        Args:
            strategy: Strategy instance to execute
            data: Market data
            parameters: Strategy parameters
            strategy_name: Name of the strategy
            
        Returns:
            Strategy execution result
        """
        async with self.semaphore:
            start_time = time.time()
            
            try:
                logger.debug(f"Starting execution of strategy: {strategy_name}")
                
                # Set parameters if provided
                if parameters:
                    strategy.set_parameters(parameters)
                
                # Execute strategy with timeout
                result = await asyncio.wait_for(
                    self._execute_strategy_core(strategy, data),
                    timeout=self.strategy_timeout
                )
                
                execution_time = time.time() - start_time
                
                # Add metadata
                result.update({
                    "strategy_name": strategy_name,
                    "execution_time": execution_time,
                    "parameters_used": parameters,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.debug(f"Strategy '{strategy_name}' completed in {execution_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                logger.warning(f"Strategy '{strategy_name}' timed out after {execution_time:.2f}s")
                return {
                    "success": False,
                    "error": f"Strategy execution timed out after {self.strategy_timeout}s",
                    "strategy_name": strategy_name,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Strategy '{strategy_name}' failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "strategy_name": strategy_name,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }

    async def _execute_strategy_core(
        self, strategy: BaseStrategy, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Core strategy execution logic.
        
        Args:
            strategy: Strategy instance
            data: Market data
            
        Returns:
            Strategy result
        """
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data, signals)
        
        # Get strategy metadata
        metadata = {
            "strategy_type": strategy.__class__.__name__,
            "description": getattr(strategy, 'description', ''),
            "parameters": strategy.get_parameters()
        }
        
        return {
            "success": True,
            "signals": signals,
            "performance_metrics": performance_metrics,
            "metadata": metadata
        }

    async def _create_ensemble_result(
        self, 
        strategy_results: List[Dict[str, Any]], 
        ensemble_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create ensemble result from individual strategy results.
        
        Args:
            strategy_results: List of successful strategy results
            ensemble_config: Ensemble configuration
            
        Returns:
            Ensemble result
        """
        try:
            if not strategy_results:
                return None
            
            # Extract signals and weights
            signals_list = []
            weights = []
            
            for result in strategy_results:
                signals = result.get("signals")
                if signals is not None:
                    signals_list.append(signals)
                    # Use performance-based weights or equal weights
                    performance = result.get("performance_metrics", {})
                    weight = performance.get("sharpe_ratio", 1.0) if performance else 1.0
                    weights.append(max(0, weight))  # Ensure non-negative weights
            
            if not signals_list:
                return None
            
            # Normalize weights
            if weights:
                weights = np.array(weights)
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(signals_list)) / len(signals_list)
            
            # Create ensemble strategy
            ensemble = WeightedEnsembleStrategy(
                strategies=signals_list,
                weights=weights,
                method=self.ensemble_method
            )
            
            # Combine signals
            combined_signals = ensemble.combine_signals()
            
            # Calculate ensemble performance
            ensemble_performance = self._calculate_performance_metrics(
                strategy_results[0].get("data", pd.DataFrame()), 
                combined_signals
            )
            
            return {
                "success": True,
                "combined_signals": combined_signals,
                "performance_metrics": ensemble_performance,
                "ensemble_config": {
                    "method": self.ensemble_method,
                    "weights": weights.tolist(),
                    "num_strategies": len(signals_list)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble result: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_performance_metrics(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for strategy signals.
        
        Args:
            data: Market data
            signals: Strategy signals
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if data.empty or signals.empty:
                return {}
            
            # Calculate returns
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
            else:
                returns = data.iloc[:, 0].pct_change().dropna()
            
            # Calculate basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": (returns > 0).mean() if len(returns) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _log_execution_performance(
        self, 
        execution_time: float, 
        success_count: int, 
        total_count: int, 
        failed_strategies: List[str]
    ) -> None:
        """
        Log execution performance statistics.
        
        Args:
            execution_time: Total execution time
            success_count: Number of successful strategies
            total_count: Total number of strategies
            failed_strategies: List of failed strategy names
        """
        success_rate = success_count / total_count if total_count > 0 else 0
        avg_time_per_strategy = execution_time / total_count if total_count > 0 else 0
        
        logger.info(
            f"Strategy execution completed: "
            f"{success_count}/{total_count} successful "
            f"({success_rate:.1%}), "
            f"total time: {execution_time:.2f}s "
            f"(avg: {avg_time_per_strategy:.2f}s per strategy)"
        )
        
        if failed_strategies:
            logger.warning(f"Failed strategies: {', '.join(failed_strategies)}")
        
        # Store in execution history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "total_strategies": total_count,
            "successful_strategies": success_count,
            "execution_time": execution_time,
            "success_rate": success_rate,
            "failed_strategies": failed_strategies
        })

    async def run_rsi_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run RSI strategy asynchronously."""
        try:
            from .rsi_strategy import RSIStrategy
            strategy = RSIStrategy()
            return await self._run_single_strategy_async(strategy, data, parameters, "RSI")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_macd_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run MACD strategy asynchronously."""
        try:
            from .macd_strategy import MACDStrategy
            strategy = MACDStrategy()
            return await self._run_single_strategy_async(strategy, data, parameters, "MACD")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_bollinger_bands_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bollinger Bands strategy asynchronously."""
        try:
            from .bollinger_bands_strategy import BollingerBandsStrategy
            strategy = BollingerBandsStrategy()
            return await self._run_single_strategy_async(strategy, data, parameters, "BollingerBands")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_momentum_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Momentum strategy asynchronously."""
        try:
            from .momentum_strategy import MomentumStrategy
            strategy = MomentumStrategy()
            return await self._run_single_strategy_async(strategy, data, parameters, "Momentum")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_mean_reversion_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Mean Reversion strategy asynchronously."""
        try:
            from .mean_reversion_strategy import MeanReversionStrategy
            strategy = MeanReversionStrategy()
            return await self._run_single_strategy_async(strategy, data, parameters, "MeanReversion")
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history.copy()

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        return self.strategy_performance.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.strategy_performance.clear()


# Convenience functions for easy usage
async def run_rsi() -> Dict[str, Any]:
    """Run RSI strategy (placeholder - requires data and parameters)."""
    runner = AsyncStrategyRunner()
    # This would need actual data and parameters
    return {"success": False, "error": "Data and parameters required"}

async def run_macd() -> Dict[str, Any]:
    """Run MACD strategy (placeholder - requires data and parameters)."""
    runner = AsyncStrategyRunner()
    # This would need actual data and parameters
    return {"success": False, "error": "Data and parameters required"}

async def run_bollinger_bands() -> Dict[str, Any]:
    """Run Bollinger Bands strategy (placeholder - requires data and parameters)."""
    runner = AsyncStrategyRunner()
    # This would need actual data and parameters
    return {"success": False, "error": "Data and parameters required"}

async def run_strategies_parallel_example(
    data: pd.DataFrame,
    strategy_names: List[str] = None,
    parameters: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Example function showing how to run strategies in parallel.
    
    Args:
        data: Market data
        strategy_names: List of strategy names to run
        parameters: Strategy parameters
        
    Returns:
        Combined results from all strategies
    """
    if strategy_names is None:
        strategy_names = ["RSI", "MACD", "BollingerBands"]
    
    if parameters is None:
        parameters = {}
    
    runner = AsyncStrategyRunner()
    results = await runner.run_strategies_parallel(
        strategies=strategy_names,
        data=data,
        parameters=parameters,
        ensemble_config={"method": "weighted"}
    )
    
    return results
