"""
Demo: Async Strategy Execution

This module demonstrates async strategy execution with parallel processing,
ensemble combination, and performance comparison.
"""

import asyncio
import logging
import time

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample market data."""
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    prices = 100 + np.cumsum(np.random.randn(days) * 0.5)

    data = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, days),
        },
        index=dates,
    )

    return data


class StrategyExecutor:
    """Strategy executor with async methods."""

    def __init__(self):
        self.data = None
        self.runner = None

    async def initialize(self):
        """Initialize data and runner."""
        from trading.strategies.strategy_runner import AsyncStrategyRunner

        self.data = generate_sample_data(200)
        self.runner = AsyncStrategyRunner(
            {
                "max_concurrent_strategies": 3,
                "strategy_timeout": 10,
                "enable_ensemble": True,
                "ensemble_method": "weighted",
            }
        )

        logger.info(f"Initialized with {len(self.data)} data points")

    async def run_rsi(self) -> dict:
        """Run RSI strategy asynchronously."""
        logger.info("🚀 Starting RSI strategy...")

        parameters = {"period": 14, "overbought": 70, "oversold": 30}

        result = await self.runner.run_rsi_strategy(self.data, parameters)

        if result.get("success"):
            perf = result.get("performance_metrics", {})
            logger.info(f"✅ RSI completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ RSI failed: {result.get('error', 'Unknown error')}")

        return result

    async def run_macd(self) -> dict:
        """Run MACD strategy asynchronously."""
        logger.info("🚀 Starting MACD strategy...")

        parameters = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

        result = await self.runner.run_macd_strategy(self.data, parameters)

        if result.get("success"):
            perf = result.get("performance_metrics", {})
            logger.info(f"✅ MACD completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ MACD failed: {result.get('error', 'Unknown error')}")

        return result

    async def run_bb(self) -> dict:
        """Run Bollinger Bands strategy asynchronously."""
        logger.info("🚀 Starting Bollinger Bands strategy...")

        parameters = {"period": 20, "std_dev": 2}

        result = await self.runner.run_bollinger_bands_strategy(self.data, parameters)

        if result.get("success"):
            perf = result.get("performance_metrics", {})
            logger.info(
                f"✅ Bollinger Bands completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}"
            )
        else:
            logger.error(
                f"❌ Bollinger Bands failed: {result.get('error', 'Unknown error')}"
            )

        return result

    async def run_momentum(self) -> dict:
        """Run Momentum strategy asynchronously."""
        logger.info("🚀 Starting Momentum strategy...")

        parameters = {"period": 10, "threshold": 0.02}

        result = await self.runner.run_momentum_strategy(self.data, parameters)

        if result.get("success"):
            perf = result.get("performance_metrics", {})
            logger.info(
                f"✅ Momentum completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}"
            )
        else:
            logger.error(f"❌ Momentum failed: {result.get('error', 'Unknown error')}")

        return result

    async def run_mean_reversion(self) -> dict:
        """Run Mean Reversion strategy asynchronously."""
        logger.info("🚀 Starting Mean Reversion strategy...")

        parameters = {"period": 20, "std_dev": 2}

        result = await self.runner.run_mean_reversion_strategy(self.data, parameters)

        if result.get("success"):
            perf = result.get("performance_metrics", {})
            logger.info(
                f"✅ Mean Reversion completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}"
            )
        else:
            logger.error(
                f"❌ Mean Reversion failed: {result.get('error', 'Unknown error')}"
            )

        return result


async def demo_parallel_execution():
    """Demo parallel strategy execution."""
    logger.info("🎯 Demo: Parallel Strategy Execution")
    logger.info("=" * 50)

    executor = StrategyExecutor()
    await executor.initialize()

    # Run strategies in parallel
    start_time = time.time()
    results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bb(),
        executor.run_momentum(),
        executor.run_mean_reversion(),
    )
    end_time = time.time()

    # Analyze results
    successful_strategies = [r for r in results if r.get("success")]
    failed_strategies = [r for r in results if not r.get("success")]

    logger.info(f"⏱️  Total execution time: {end_time - start_time:.2f}s")
    logger.info(f"✅ Successful strategies: {len(successful_strategies)}")
    logger.info(f"❌ Failed strategies: {len(failed_strategies)}")

    # Calculate average performance
    if successful_strategies:
        avg_sharpe = sum(
            s.get("performance_metrics", {}).get("sharpe_ratio", 0)
            for s in successful_strategies
        ) / len(successful_strategies)
        logger.info(f"📊 Average Sharpe ratio: {avg_sharpe:.3f}")

    return results


async def demo_ensemble_execution():
    """Demo ensemble strategy execution."""
    logger.info("🎯 Demo: Ensemble Strategy Execution")
    logger.info("=" * 50)

    executor = StrategyExecutor()
    await executor.initialize()

    # Define strategy list
    strategies = ["RSI_Strategy", "MACD_Strategy", "Bollinger_Bands"]
    parameters = {
        "RSI_Strategy": {"period": 14, "overbought": 70, "oversold": 30},
        "MACD_Strategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "Bollinger_Bands": {"period": 20, "std_dev": 2},
    }

    # Run ensemble
    start_time = time.time()
    result = await executor.runner.run_strategies_parallel(
        strategies=strategies,
        data=executor.data,
        parameters=parameters,
        ensemble_config={
            "method": "weighted",
            "weights": {
                "RSI_Strategy": 0.4,
                "MACD_Strategy": 0.3,
                "Bollinger_Bands": 0.3,
            },
        },
    )
    end_time = time.time()

    if result.get("success"):
        logger.info(f"⏱️  Ensemble execution time: {end_time - start_time:.2f}s")
        logger.info(f"📊 Ensemble result: {result.get('ensemble_result', {})}")
    else:
        logger.error(f"❌ Ensemble failed: {result.get('error')}")

    return result


async def demo_sequential_vs_parallel():
    """Demo sequential vs parallel execution comparison."""
    logger.info("🎯 Demo: Sequential vs Parallel Execution")
    logger.info("=" * 50)

    executor = StrategyExecutor()
    await executor.initialize()

    # Sequential execution
    logger.info("🔄 Running strategies sequentially...")
    start_time = time.time()
    sequential_results = []
    for strategy_func in [executor.run_rsi, executor.run_macd, executor.run_bb]:
        result = await strategy_func()
        sequential_results.append(result)
    sequential_time = time.time() - start_time

    # Parallel execution
    logger.info("⚡ Running strategies in parallel...")
    start_time = time.time()
    parallel_results = await asyncio.gather(
        executor.run_rsi(), executor.run_macd(), executor.run_bb()
    )
    parallel_time = time.time() - start_time

    # Comparison
    logger.info(f"⏱️  Sequential time: {sequential_time:.2f}s")
    logger.info(f"⏱️  Parallel time: {parallel_time:.2f}s")
    logger.info(f"🚀 Speedup: {sequential_time / parallel_time:.2f}x")

    return {
        "sequential": {"time": sequential_time, "results": sequential_results},
        "parallel": {"time": parallel_time, "results": parallel_results},
    }


async def main():
    """Main demo function."""
    logger.info("🚀 Starting Async Strategy Demo")
    logger.info("=" * 60)

    try:
        # Demo 1: Parallel execution
        await demo_parallel_execution()
        logger.info("")

        # Demo 2: Ensemble execution
        await demo_ensemble_execution()
        logger.info("")

        # Demo 3: Sequential vs Parallel comparison
        await demo_sequential_vs_parallel()

        logger.info("✅ All demos completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
