"""
Example: Async Strategy Runner Usage

This script demonstrates how to use the AsyncStrategyRunner to execute
multiple strategies in parallel using asyncio, exactly as requested:

async def run_rsi(): ...
async def run_macd(): ...
results = await asyncio.gather(run_rsi(), run_macd(), run_bb())
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

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
            "std_dev": 1.5
        }

        result = await self.runner.run_mean_reversion_strategy(self.data, parameters)
        logger.info(f"Mean Reversion strategy completed: {result.get('success', False)}")
        return result


async def example_parallel_execution():
    """Example 1: Run strategies in parallel using asyncio.gather()"""
    logger.info("=== Example 1: Parallel Strategy Execution ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Run strategies in parallel - exactly as requested!
    start_time = datetime.now()

    results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bollinger_bands()
    )

    execution_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"Parallel execution completed in {execution_time:.2f} seconds")

    # Process results
    for i, result in enumerate(['RSI', 'MACD', 'Bollinger Bands']):
        if results[i].get('success'):
            perf = results[i].get('performance_metrics', {})
            logger.info(f"{result}: Sharpe={perf.get('sharpe_ratio', 0):.3f}, "
                       f"Return={perf.get('total_return', 0):.3f}")
        else:
            logger.error(f"{result}: Failed - {results[i].get('error', 'Unknown error')}")

    return results


async def example_ensemble_execution():
    """Example 2: Run strategies and create ensemble result"""
    logger.info("\n=== Example 2: Ensemble Strategy Execution ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Run multiple strategies with ensemble combination
    strategy_names = ["RSI", "MACD", "Bollinger Bands", "Momentum", "Mean Reversion"]

    start_time = datetime.now()

    # Use the parallel execution method
    ensemble_result = await executor.runner.run_strategies_parallel(
        strategies=strategy_names,
        data=executor.data,
        parameters={
            "RSI": {"period": 14, "overbought": 70, "oversold": 30},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "Bollinger Bands": {"period": 20, "std_dev": 2},
            "Momentum": {"period": 10, "threshold": 0.02},
            "Mean Reversion": {"period": 20, "std_dev": 1.5}
        },
        ensemble_config={"method": "weighted"}
    )

    execution_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"Ensemble execution completed in {execution_time:.2f} seconds")

    # Display results
    if ensemble_result.get('success'):
        stats = ensemble_result.get('execution_stats', {})
        logger.info(f"Total strategies: {stats.get('total_strategies', 0)}")
        logger.info(f"Successful: {stats.get('successful_strategies', 0)}")
        logger.info(f"Failed: {stats.get('failed_strategies', 0)}")

        # Show ensemble result
        ensemble = ensemble_result.get('ensemble_result')
        if ensemble and ensemble.get('success'):
            perf = ensemble.get('performance_metrics', {})
            logger.info(f"Ensemble Performance: Sharpe={perf.get('sharpe_ratio', 0):.3f}, "
                       f"Return={perf.get('total_return', 0):.3f}")

    return ensemble_result


async def example_sequential_vs_parallel():
    """Example 3: Compare sequential vs parallel execution"""
    logger.info("\n=== Example 3: Sequential vs Parallel Comparison ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Sequential execution
    logger.info("Running strategies sequentially...")
    start_time = datetime.now()

    rsi_result = await executor.run_rsi()
    macd_result = await executor.run_macd()
    bb_result = await executor.run_bollinger_bands()

    sequential_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Sequential execution time: {sequential_time:.2f} seconds")

    # Parallel execution
    logger.info("Running strategies in parallel...")
    start_time = datetime.now()

    parallel_results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bollinger_bands()
    )

    parallel_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Parallel execution time: {parallel_time:.2f} seconds")

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"Speedup: {speedup:.2f}x")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup
    }


async def example_error_handling():
    """Example 4: Error handling and timeout management"""
    logger.info("\n=== Example 4: Error Handling ===")

    executor = AsyncStrategyExecutor()
    await executor.initialize_data()

    # Create a runner with short timeout to test error handling
    test_runner = AsyncStrategyRunner({
        "max_concurrent_strategies": 2,
        "strategy_timeout": 1,  # Very short timeout
        "enable_ensemble": False,
        "log_performance": True
    })

    # Run strategies that might timeout
    results = await test_runner.run_strategies_parallel(
        strategies=["RSI", "MACD", "Bollinger Bands"],
        data=executor.data,
        parameters={
            "RSI": {"period": 14},
            "MACD": {"fast_period": 12, "slow_period": 26},
            "Bollinger Bands": {"period": 20}
        }
    )

    logger.info(f"Execution completed: {results.get('success', False)}")

    # Show which strategies succeeded/failed
    individual_results = results.get('individual_results', {})
    for strategy_name, result in individual_results.items():
        if result.get('success'):
            logger.info(f"✓ {strategy_name}: Success")
        else:
            logger.error(f"✗ {strategy_name}: {result.get('error', 'Unknown error')}")

    return results


async def main():
    """Run all examples."""
    logger.info("🚀 Starting Async Strategy Runner Examples")
    logger.info("=" * 50)

    try:
        # Example 1: Basic parallel execution
        await example_parallel_execution()

        # Example 2: Ensemble execution
        await example_ensemble_execution()

        # Example 3: Performance comparison
        await example_sequential_vs_parallel()

        # Example 4: Error handling
        await example_error_handling()

        logger.info("\n✅ All examples completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

