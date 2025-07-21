"""
TaskAgent Example

This example demonstrates how to use the enhanced TaskAgent with the new
forecast, strategy, and backtest task types. The agent will automatically
call builder, evaluator, updater in sequence and loop if needed until
performance meets the threshold.
"""

import asyncio
import logging

from agents.task_agent import (
    TaskAgent,
    TaskType,
    execute_backtest_task,
    execute_forecast_task,
    execute_strategy_task,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_forecast_task():
    """Example of executing a forecast task."""
    logger.info("=== Example: Forecast Task ===")

    prompt = "Build a forecasting model for AAPL stock price prediction"
    parameters = {
        "symbol": "AAPL",
        "forecast_horizon": 30,
        "data_path": "data/aapl_data.csv",
        "hyperparameters": {"epochs": 150, "batch_size": 64, "lookback_window": 90},
    }

    result = await execute_forecast_task(
        prompt=prompt,
        parameters=parameters,
        max_depth=3,
        performance_threshold=0.7,  # Sharpe ratio threshold
    )

    logger.info(f"Forecast task completed: {result.success}")
    logger.info(f"Final performance score: {result.performance_score:.3f}")
    logger.info(f"Message: {result.message}")

    if result.data:
        logger.info(f"Model ID: {result.data.get('model_id', 'N/A')}")
        logger.info(f"Sharpe Ratio: {result.data.get('sharpe_ratio', 'N/A')}")

    return result


async def example_strategy_task():
    """Example of executing a strategy task."""
    logger.info("\n=== Example: Strategy Task ===")

    prompt = "Create a momentum-based trading strategy for SPY ETF"
    parameters = {
        "symbol": "SPY",
        "strategy_type": "momentum",
        "data_path": "data/spy_data.csv",
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 12,
            "learning_rate": 0.05,
        },
    }

    result = await execute_strategy_task(
        prompt=prompt,
        parameters=parameters,
        max_depth=4,
        performance_threshold=0.8,  # Higher threshold for strategies
    )

    logger.info(f"Strategy task completed: {result.success}")
    logger.info(f"Final performance score: {result.performance_score:.3f}")
    logger.info(f"Message: {result.message}")

    if result.data:
        logger.info(f"Model ID: {result.data.get('model_id', 'N/A')}")
        logger.info(f"Total Return: {result.data.get('total_return', 'N/A')}")
        logger.info(f"Max Drawdown: {result.data.get('max_drawdown', 'N/A')}")

    return result


async def example_backtest_task():
    """Example of executing a backtest task."""
    logger.info("\n=== Example: Backtest Task ===")

    prompt = "Run a comprehensive backtest of a mean reversion strategy"
    parameters = {
        "symbol": "QQQ",
        "strategy_type": "mean_reversion",
        "data_path": "data/qqq_data.csv",
        "backtest_period": "2020-01-01:2023-12-31",
        "hyperparameters": {"n_estimators": 200, "max_depth": 10, "subsample": 0.9},
    }

    result = await execute_backtest_task(
        prompt=prompt,
        parameters=parameters,
        max_depth=5,
        performance_threshold=0.6,  # Lower threshold for backtests
    )

    logger.info(f"Backtest task completed: {result.success}")
    logger.info(f"Final performance score: {result.performance_score:.3f}")
    logger.info(f"Message: {result.message}")

    if result.data:
        logger.info(f"Model ID: {result.data.get('model_id', 'N/A')}")
        logger.info(f"Win Rate: {result.data.get('win_rate', 'N/A')}")
        logger.info(f"Profit Factor: {result.data.get('profit_factor', 'N/A')}")

    return result


async def example_custom_task():
    """Example of executing a custom task with the general TaskAgent."""
    logger.info("\n=== Example: Custom Task ===")

    agent = TaskAgent()

    prompt = "Build and evaluate a custom ensemble model for cryptocurrency trading"
    parameters = {
        "model_type": "ensemble",
        "data_path": "data/crypto_data.csv",
        "target_column": "btc_returns",
        "hyperparameters": {
            "models": ["lstm", "xgboost", "random_forest"],
            "ensemble_method": "weighted_average",
        },
    }

    result = await agent.execute_task(
        prompt=prompt,
        task_type=TaskType.MODEL_BUILD,
        parameters=parameters,
        max_depth=3,
        performance_threshold=0.75,
    )

    logger.info(f"Custom task completed: {result.success}")
    logger.info(f"Final performance score: {result.performance_score:.3f}")
    logger.info(f"Message: {result.message}")

    return result


async def main():
    """Run all examples."""
    logger.info("Starting TaskAgent Examples")
    logger.info("=" * 50)

    try:
        # Run forecast example
        forecast_result = await example_forecast_task()

        # Run strategy example
        strategy_result = await example_strategy_task()

        # Run backtest example
        backtest_result = await example_backtest_task()

        # Run custom example
        custom_result = await example_custom_task()

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Forecast Task: {'âœ“' if forecast_result.success else 'âœ—'}")
        logger.info(f"Strategy Task: {'âœ“' if strategy_result.success else 'âœ—'}")
        logger.info(f"Backtest Task: {'âœ“' if backtest_result.success else 'âœ—'}")
        logger.info(f"Custom Task: {'âœ“' if custom_result.success else 'âœ—'}")

        # Show task history
        agent = TaskAgent()
        all_tasks = agent.get_all_tasks()
        logger.info(f"\nTotal tasks executed: {len(all_tasks)}")

        for i, task in enumerate(all_tasks[:3]):  # Show first 3 tasks
            logger.info(f"Task {i + 1}: {task['task_type']} - Depth: {task['depth']}")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
