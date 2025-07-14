#!/usr/bin/env python3
"""
Demo: Async Strategy Execution

This demo shows the exact pattern requested:
async def run_rsi(): ...
async def run_macd(): ...
results = await asyncio.gather(run_rsi(), run_macd(), run_bb())
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample market data."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    prices = 100 + np.cumsum(np.random.randn(days) * 0.5)
    
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
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
        self.runner = AsyncStrategyRunner({
            "max_concurrent_strategies": 3,
            "strategy_timeout": 10,
            "enable_ensemble": True,
            "ensemble_method": "weighted"
        })
        
        logger.info(f"Initialized with {len(self.data)} data points")
    
    async def run_rsi(self) -> dict:
        """Run RSI strategy asynchronously."""
        logger.info("🚀 Starting RSI strategy...")
        
        parameters = {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
        
        result = await self.runner.run_rsi_strategy(self.data, parameters)
        
        if result.get('success'):
            perf = result.get('performance_metrics', {})
            logger.info(f"✅ RSI completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ RSI failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def run_macd(self) -> dict:
        """Run MACD strategy asynchronously."""
        logger.info("🚀 Starting MACD strategy...")
        
        parameters = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
        
        result = await self.runner.run_macd_strategy(self.data, parameters)
        
        if result.get('success'):
            perf = result.get('performance_metrics', {})
            logger.info(f"✅ MACD completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ MACD failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def run_bb(self) -> dict:
        """Run Bollinger Bands strategy asynchronously."""
        logger.info("🚀 Starting Bollinger Bands strategy...")
        
        parameters = {
            "period": 20,
            "std_dev": 2
        }
        
        result = await self.runner.run_bollinger_bands_strategy(self.data, parameters)
        
        if result.get('success'):
            perf = result.get('performance_metrics', {})
            logger.info(f"✅ Bollinger Bands completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ Bollinger Bands failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def run_momentum(self) -> dict:
        """Run Momentum strategy asynchronously."""
        logger.info("🚀 Starting Momentum strategy...")
        
        parameters = {
            "period": 10,
            "threshold": 0.02
        }
        
        result = await self.runner.run_momentum_strategy(self.data, parameters)
        
        if result.get('success'):
            perf = result.get('performance_metrics', {})
            logger.info(f"✅ Momentum completed - Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            logger.error(f"❌ Momentum failed: {result.get('error', 'Unknown error')}")
        
        return result


async def demo_parallel_execution():
    """Demo the exact pattern requested by the user."""
    logger.info("=" * 60)
    logger.info("DEMO: Async Strategy Execution")
    logger.info("=" * 60)
    
    # Initialize executor
    executor = StrategyExecutor()
    await executor.initialize()
    
    # The exact pattern requested:
    # async def run_rsi(): ...
    # async def run_macd(): ...
    # results = await asyncio.gather(run_rsi(), run_macd(), run_bb())
    
    logger.info("📊 Running strategies in parallel using asyncio.gather()...")
    start_time = datetime.now()
    
    # Execute strategies in parallel - EXACTLY as requested!
    results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bb()
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"⏱️  Parallel execution completed in {execution_time:.2f} seconds")
    logger.info("=" * 60)
    
    # Display results
    strategy_names = ["RSI", "MACD", "Bollinger Bands"]
    for i, (name, result) in enumerate(zip(strategy_names, results)):
        logger.info(f"\n📈 {name} Results:")
        if result.get('success'):
            perf = result.get('performance_metrics', {})
            logger.info(f"   ✅ Success")
            logger.info(f"   📊 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            logger.info(f"   📈 Total Return: {perf.get('total_return', 0):.3f}")
            logger.info(f"   📉 Max Drawdown: {perf.get('max_drawdown', 0):.3f}")
            logger.info(f"   ⏱️  Execution Time: {result.get('execution_time', 0):.2f}s")
        else:
            logger.error(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return results


async def demo_ensemble_execution():
    """Demo ensemble execution with more strategies."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Ensemble Strategy Execution")
    logger.info("=" * 60)
    
    executor = StrategyExecutor()
    await executor.initialize()
    
    logger.info("🎯 Running multiple strategies with ensemble combination...")
    start_time = datetime.now()
    
    # Run multiple strategies including the new momentum strategy
    results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bb(),
        executor.run_momentum()
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"⏱️  Ensemble execution completed in {execution_time:.2f} seconds")
    
    # Show ensemble result
    successful_results = [r for r in results if r.get('success')]
    logger.info(f"📊 {len(successful_results)}/{len(results)} strategies successful")
    
    if successful_results:
        # Calculate average performance
        avg_sharpe = np.mean([r.get('performance_metrics', {}).get('sharpe_ratio', 0) for r in successful_results])
        avg_return = np.mean([r.get('performance_metrics', {}).get('total_return', 0) for r in successful_results])
        
        logger.info(f"📈 Average Sharpe Ratio: {avg_sharpe:.3f}")
        logger.info(f"📊 Average Total Return: {avg_return:.3f}")
    
    return results


async def demo_sequential_vs_parallel():
    """Demo the performance difference between sequential and parallel execution."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Sequential vs Parallel Performance")
    logger.info("=" * 60)
    
    executor = StrategyExecutor()
    await executor.initialize()
    
    # Sequential execution
    logger.info("🔄 Running strategies sequentially...")
    start_time = datetime.now()
    
    rsi_result = await executor.run_rsi()
    macd_result = await executor.run_macd()
    bb_result = await executor.run_bb()
    
    sequential_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️  Sequential execution time: {sequential_time:.2f} seconds")
    
    # Parallel execution
    logger.info("⚡ Running strategies in parallel...")
    start_time = datetime.now()
    
    parallel_results = await asyncio.gather(
        executor.run_rsi(),
        executor.run_macd(),
        executor.run_bb()
    )
    
    parallel_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️  Parallel execution time: {parallel_time:.2f} seconds")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"🚀 Speedup: {speedup:.2f}x faster with parallel execution!")
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup
    }


async def main():
    """Run all demos."""
    try:
        # Demo 1: Basic parallel execution (exact pattern requested)
        await demo_parallel_execution()
        
        # Demo 2: Ensemble execution
        await demo_ensemble_execution()
        
        # Demo 3: Performance comparison
        await demo_sequential_vs_parallel()
        
        logger.info("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error running demos: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 