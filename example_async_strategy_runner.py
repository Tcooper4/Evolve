"""
Example: Async Strategy Runner Usage

This module demonstrates how to use the AsyncStrategyRunner for parallel
strategy execution, ensemble combination, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from trading.strategies.strategy_runner import AsyncStrategyRunner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })

    data.set_index('date', inplace=True)
    return data


class AsyncStrategyExecutor:
    """Example class showing how to use the async strategy runner."""

    def __init__(self):
        self.runner = AsyncStrategyRunner({
            "max_concurrent_strategies": 3,
            "strategy_timeout": 10,
            "enable_ensemble": True,
            "ensemble_method": "weighted",
            "log_performance": True
        })
        self.data = None

    async def initialize_data(self):
        """Initialize sample market data."""
        self.data = generate_sample_data(200)
        logger.info(f"Generated sample data with {len(self.data)} rows")

    async def run_rsi(self) -> dict:
        """Run RSI strategy asynchronously."""
        logger.info("Starting RSI strategy execution")

        parameters = {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }

        result = await self.runner.run_rsi_strategy(self.data, parameters)
        logger.info(f"RSI strategy completed: {result.get('success', False)}")
        return result

    async def run_macd(self) -> dict:
        """Run MACD strategy asynchronously."""
        logger.info("Starting MACD strategy execution")

        parameters = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }

        result = await self.runner.run_macd_strategy(self.data, parameters)
        logger.info(f"MACD strategy completed: {result.get('success', False)}")
        return result

    async def run_bollinger_bands(self) -> dict:
        """Run Bollinger Bands strategy asynchronously."""
        logger.info("Starting Bollinger Bands strategy execution")

        parameters = {
            "period": 20,
            "std_dev": 2
        }

        result = await self.runner.run_bollinger_bands_strategy(self.data, parameters)
        logger.info(f"Bollinger Bands strategy completed: {result.get('success', False)}")
        return result

    async def run_momentum(self) -> dict:
        """Run Momentum strategy asynchronously."""
        logger.info("Starting Momentum strategy execution")

        parameters = {
            "period": 10,
            "threshold": 0.02
        }

        result = await self.runner.run_momentum_strategy(self.data, parameters)
        logger.info(f"Momentum strategy completed: {result.get('success', False)}")
        return result

    async def run_mean_reversion(self) -> dict:
        """Run Mean Reversion strategy asynchronously."""
        logger.info("Starting Mean Reversion strategy execution")

        parameters = {
            "period": 20,
            "std_dev": 2
        }

        result = await self.runner.run_mean_reversion_strategy(self.data, parameters)
        logger.info(f"Mean Reversion strategy completed: {result.get('success', False)}")
        return result


async def example_parallel_execution():
    """Example of parallel strategy execution."""
    logger.info("=== Parallel Strategy Execution Example ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Run multiple strategies in parallel
    start_time = time.time()
    results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bollinger_bands(),
        executor.run_momentum(),
        executor.run_mean_reversion()
    )
    end_time = time.time()

    # Analyze results
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    logger.info(f"Execution time: {end_time - start_time:.2f}s")
    logger.info(f"Successful strategies: {len(successful)}")
    logger.info(f"Failed strategies: {len(failed)}")

    return results


async def example_ensemble_execution():
    """Example of ensemble strategy execution."""
    logger.info("=== Ensemble Strategy Execution Example ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Define strategies and parameters
    strategies = ["RSI_Strategy", "MACD_Strategy", "Bollinger_Bands"]
    parameters = {
        "RSI_Strategy": {"period": 14, "overbought": 70, "oversold": 30},
        "MACD_Strategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "Bollinger_Bands": {"period": 20, "std_dev": 2}
    }

    # Run ensemble
    start_time = time.time()
    result = await executor.runner.run_strategies_parallel(
        strategies=strategies,
        data=executor.data,
        parameters=parameters,
        ensemble_config={
            "method": "weighted",
            "weights": {"RSI_Strategy": 0.4, "MACD_Strategy": 0.3, "Bollinger_Bands": 0.3}
        }
    )
    end_time = time.time()

    if result.get('success'):
        logger.info(f"Ensemble execution time: {end_time - start_time:.2f}s")
        ensemble_result = result.get('ensemble_result', {})
        logger.info(f"Ensemble performance: {ensemble_result}")
    else:
        logger.error(f"Ensemble failed: {result.get('error')}")

    return result


async def example_sequential_vs_parallel():
    """Example comparing sequential vs parallel execution."""
    logger.info("=== Sequential vs Parallel Execution Example ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Sequential execution
    logger.info("Running strategies sequentially...")
    start_time = time.time()
    sequential_results = []
    for strategy_func in [executor.run_rsi, executor.run_macd, executor.run_bollinger_bands]:
        result = await strategy_func()
        sequential_results.append(result)
    sequential_time = time.time() - start_time

    # Parallel execution
    logger.info("Running strategies in parallel...")
    start_time = time.time()
    parallel_results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bollinger_bands()
    )
    parallel_time = time.time() - start_time

    # Comparison
    logger.info(f"Sequential time: {sequential_time:.2f}s")
    logger.info(f"Parallel time: {parallel_time:.2f}s")
    logger.info(f"Speedup: {sequential_time / parallel_time:.2f}x")

    return {
        "sequential": {"time": sequential_time, "results": sequential_results},
        "parallel": {"time": parallel_time, "results": parallel_results}
    }


async def example_error_handling():
    """Example of error handling in strategy execution."""
    logger.info("=== Error Handling Example ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Run strategies with potential errors
    try:
        results = await asyncio.gather(
            executor.run_rsi(),
            executor.run_macd(),
            executor.run_bollinger_bands(),
            return_exceptions=True
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {i} failed with exception: {result}")
            elif not result.get('success'):
                logger.warning(f"Strategy {i} completed but failed: {result.get('error')}")
            else:
                logger.info(f"Strategy {i} completed successfully")

    except Exception as e:
        logger.error(f"Error in strategy execution: {e}")

    return results


async def main():
    """Main example function."""
    logger.info("Starting Async Strategy Runner Examples")
    logger.info("=" * 50)

    try:
        # Example 1: Parallel execution
        await example_parallel_execution()
        logger.info("")

        # Example 2: Ensemble execution
        await example_ensemble_execution()
        logger.info("")

        # Example 3: Sequential vs Parallel comparison
        await example_sequential_vs_parallel()
        logger.info("")

        # Example 4: Error handling
        await example_error_handling()

        logger.info("All examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 